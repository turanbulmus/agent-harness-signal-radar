"""Structured JSON logging via structlog with OTel trace IDs and active PII scrubbing."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

import structlog

from .pii_redactor import SensitiveDataRedactor
from .tracing import current_trace_context


def _inject_trace_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
  trace_ctx = current_trace_context()
  event_dict.setdefault("trace_id", trace_ctx["trace_id"])
  event_dict.setdefault("span_id", trace_ctx["span_id"])
  return event_dict


def _normalize_severity(
    _logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
  level = event_dict.pop("level", method_name).upper()
  event_dict["severity"] = level
  return event_dict


def _build_pii_processor(redactor: SensitiveDataRedactor):
  def _scrub_event_dict(
      _logger: Any, _method_name: str, event_dict: dict[str, Any]
  ) -> dict[str, Any]:
    return redactor.scrub_payload(event_dict)

  return _scrub_event_dict


def configure_structured_logging(
    *,
    level: str = "INFO",
    stream: TextIO | None = None,
    redactor: SensitiveDataRedactor | None = None,
) -> None:
  """Configures structlog and stdlib logging to emit scrubbed JSON log lines."""
  active_redactor = redactor or SensitiveDataRedactor()
  numeric_level = getattr(logging, level.upper(), logging.INFO)
  target_stream = stream or sys.stderr

  logging.basicConfig(
      format="%(message)s",
      stream=target_stream,
      level=numeric_level,
      force=True,
  )

  structlog.configure(
      processors=[
          structlog.contextvars.merge_contextvars,
          structlog.stdlib.add_logger_name,
          structlog.stdlib.add_log_level,
          _normalize_severity,
          structlog.processors.TimeStamper(fmt="iso", utc=True),
          _inject_trace_context,
          _build_pii_processor(active_redactor),
          structlog.processors.StackInfoRenderer(),
          structlog.processors.format_exc_info,
          structlog.processors.JSONRenderer(),
      ],
      wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
      logger_factory=structlog.stdlib.LoggerFactory(),
      cache_logger_on_first_use=False,
  )


def get_logger(name: str = "harness_signal_radar") -> structlog.stdlib.BoundLogger:
  """Returns a structured JSON logger bound with component metadata."""
  if not structlog.is_configured():
    configure_structured_logging()
  return structlog.get_logger(name)
