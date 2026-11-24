"""Observability, OpenTelemetry tracing, PII redaction, and Intent/Outcome audit hooks."""

from .intent_outcome import IntentOutcomeAuditPlugin, IntentOutcomeRecord
from .logging_config import configure_structured_logging, get_logger
from .pii_redactor import SensitiveDataRedactor
from .tracing import (
    clear_registered_exporters,
    current_trace_context,
    get_tracer,
    register_span_exporter,
    start_span,
    traced_operation,
)

__all__ = [
    "IntentOutcomeAuditPlugin",
    "IntentOutcomeRecord",
    "SensitiveDataRedactor",
    "clear_registered_exporters",
    "configure_structured_logging",
    "current_trace_context",
    "get_logger",
    "get_tracer",
    "register_span_exporter",
    "start_span",
    "traced_operation",
]
