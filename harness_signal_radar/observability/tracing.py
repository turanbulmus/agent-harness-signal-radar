"""OpenTelemetry distributed tracing setup and span helpers."""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

F = TypeVar("F", bound=Callable[..., Any])


class _DynamicExporterFanoutProcessor(SpanProcessor):
  """Forwards spans to all dynamically registered exporters (tests, console, GCP)."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._exporters: list[SpanExporter] = []

  def add_exporter(self, exporter: SpanExporter) -> None:
    with self._lock:
      if exporter not in self._exporters:
        self._exporters.append(exporter)

  def clear_exporters(self) -> None:
    with self._lock:
      for exporter in self._exporters:
        try:
          exporter.shutdown()
        except Exception:
          pass
      self._exporters.clear()

  def on_end(self, span: ReadableSpan) -> None:
    with self._lock:
      exporters = list(self._exporters)
    for exporter in exporters:
      try:
        exporter.export([span])
      except Exception:
        pass

  def shutdown(self) -> None:
    self.clear_exporters()

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True


_PROVIDER_LOCK = threading.Lock()
_PROVIDER_INITIALIZED = False
_FANOUT_PROCESSOR = _DynamicExporterFanoutProcessor()


def ensure_tracer_provider(
    *,
    service_name: str = "harness-signal-radar",
    enable_console: bool = False,
    enable_cloud_trace: bool = False,
    project_id: str | None = None,
) -> TracerProvider:
  """Initializes the global OpenTelemetry TracerProvider once cleanly."""
  global _PROVIDER_INITIALIZED
  with _PROVIDER_LOCK:
    if not _PROVIDER_INITIALIZED:
      resource = Resource.create({"service.name": service_name})
      provider = TracerProvider(resource=resource)
      provider.add_span_processor(_FANOUT_PROCESSOR)
      if enable_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
      if enable_cloud_trace:
        try:
          from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

          provider.add_span_processor(
              SimpleSpanProcessor(CloudTraceSpanExporter(project_id=project_id))
          )
        except Exception:
          pass
      trace.set_tracer_provider(provider)
      _PROVIDER_INITIALIZED = True
    return trace.get_tracer_provider()  # type: ignore[return-value]


def register_span_exporter(exporter: SpanExporter) -> None:
  """Registers an additional OpenTelemetry SpanExporter (e.g., InMemorySpanExporter)."""
  ensure_tracer_provider()
  _FANOUT_PROCESSOR.add_exporter(exporter)


def clear_registered_exporters() -> None:
  """Clears dynamically registered span exporters (primarily for test isolation)."""
  _FANOUT_PROCESSOR.clear_exporters()


def get_tracer(name: str = "harness_signal_radar") -> trace.Tracer:
  """Returns an OpenTelemetry tracer backed by the initialized provider."""
  ensure_tracer_provider()
  return trace.get_tracer(name)


def current_trace_context() -> dict[str, str]:
  """Extracts the active OpenTelemetry trace_id and span_id as hex strings."""
  span = trace.get_current_span()
  ctx = span.get_span_context() if span else None
  if ctx is not None and ctx.is_valid:
    return {
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
    }
  return {
      "trace_id": "0" * 32,
      "span_id": "0" * 16,
  }


@contextmanager
def start_span(
    span_name: str,
    *,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "harness_signal_radar",
) -> Iterator[trace.Span]:
  """Context manager opening a linked child span and recording errors cleanly."""
  tracer = get_tracer(tracer_name)
  with tracer.start_as_current_span(span_name) as span:
    if attributes:
      for key, value in attributes.items():
        if value is not None:
          span.set_attribute(key, value)
    yield span


def traced_operation(
    span_name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
  """Decorator wrapping sync or async functions in an OpenTelemetry span."""

  def decorator(func: F) -> F:
    if inspect.iscoroutinefunction(func):

      @functools.wraps(func)
      async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        with start_span(span_name, attributes=attributes):
          return await func(*args, **kwargs)

      return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
      with start_span(span_name, attributes=attributes):
        return func(*args, **kwargs)

    return sync_wrapper  # type: ignore[return-value]

  return decorator
