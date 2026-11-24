"""ADK plugin capturing linked Intent vs. Outcome audit events and OpenTelemetry spans."""

from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from .logging_config import get_logger
from .pii_redactor import SensitiveDataRedactor
from .tracing import current_trace_context, get_tracer


class IntentOutcomeRecord(BaseModel):
  """Structured audit entry recording either an intended action or its actual outcome."""

  model_config = ConfigDict(extra="forbid")

  correlation_id: str
  phase: str
  timestamp_utc: str
  invocation_id: str
  session_id: str
  actor: str
  action_type: str
  intended_target: str
  outcome_status: str | None = None
  duration_ms: float | None = None
  intent_fulfilled: bool | None = None
  trace_id: str
  span_id: str
  details: dict[str, Any] = Field(default_factory=dict)


class IntentOutcomeAuditPlugin(BasePlugin):
  """Captures explicit pre-execution INTENT and post-execution OUTCOME audit logs.

  Links every ADK run, agent invocation, LLM call, and tool execution to paired
  OpenTelemetry child spans and PII-scrubbed structured JSON records.
  """

  def __init__(
      self,
      *,
      name: str = "intent_outcome_audit",
      redactor: SensitiveDataRedactor | None = None,
      max_in_memory_records: int = 500,
  ) -> None:
    super().__init__(name=name)
    self._redactor = redactor or SensitiveDataRedactor()
    self._logger = get_logger("harness_signal_radar.audit")
    self._tracer = get_tracer("harness_signal_radar.audit")
    self._records: deque[IntentOutcomeRecord] = deque(maxlen=max_in_memory_records)
    self._active_spans: dict[str, tuple[trace.Span, object, float, str]] = {}
    self._active_corr_stacks: dict[str, list[str]] = {}

  @property
  def records(self) -> list[IntentOutcomeRecord]:
    """Returns a snapshot of captured Intent/Outcome audit records."""
    return list(self._records)

  def clear_records(self) -> None:
    """Clears in-memory audit records (for test isolation)."""
    self._records.clear()

  def _now_iso(self) -> str:
    return datetime.now(tz=UTC).isoformat()

  def _begin_stage(
      self,
      *,
      stack_key: str,
      span_name: str,
      invocation_id: str,
      session_id: str,
      actor: str,
      action_type: str,
      intended_target: str,
      details: dict[str, Any],
  ) -> str:
    correlation_id = uuid.uuid4().hex[:16]
    span = self._tracer.start_span(span_name)
    span.set_attribute("audit.correlation_id", correlation_id)
    span.set_attribute("audit.action_type", action_type)
    span.set_attribute("audit.target", intended_target)
    span.set_attribute("audit.invocation_id", invocation_id)
    span.set_attribute("audit.session_id", session_id)
    token = otel_context.attach(trace.set_span_in_context(span))
    self._active_spans[correlation_id] = (
        span,
        token,
        time.perf_counter(),
        intended_target,
    )
    self._active_corr_stacks.setdefault(stack_key, []).append(correlation_id)

    trace_ctx = current_trace_context()
    scrubbed_details = self._redactor.scrub_payload(details)
    record = IntentOutcomeRecord(
        correlation_id=correlation_id,
        phase="INTENT",
        timestamp_utc=self._now_iso(),
        invocation_id=invocation_id,
        session_id=session_id,
        actor=actor,
        action_type=action_type,
        intended_target=intended_target,
        trace_id=trace_ctx["trace_id"],
        span_id=trace_ctx["span_id"],
        details=scrubbed_details,
    )
    self._records.append(record)
    self._logger.info("intent_captured", **record.model_dump())
    return correlation_id

  def _end_stage(
      self,
      *,
      stack_key: str,
      invocation_id: str,
      session_id: str,
      actor: str,
      action_type: str,
      outcome_status: str,
      intent_fulfilled: bool,
      details: dict[str, Any],
      error: Exception | None = None,
  ) -> None:
    stack = self._active_corr_stacks.get(stack_key)
    correlation_id = stack.pop() if stack else uuid.uuid4().hex[:16]
    if stack is not None and not stack:
      self._active_corr_stacks.pop(stack_key, None)

    span_bundle = self._active_spans.pop(correlation_id, None)
    duration_ms = 0.0
    intended_target = actor
    if span_bundle is not None:
      span, token, start_time, intended_target = span_bundle
      duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
      trace_ctx = current_trace_context()
      span.set_attribute("audit.outcome_status", outcome_status)
      span.set_attribute("audit.duration_ms", duration_ms)
      span.set_attribute("audit.intent_fulfilled", intent_fulfilled)
      if error is not None:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, self._redactor.scrub_text(str(error))))
      elif not intent_fulfilled:
        span.set_status(Status(StatusCode.ERROR, outcome_status))
      else:
        span.set_status(Status(StatusCode.OK))
      try:
        otel_context.detach(token)
      except Exception:
        pass
      span.end()
    else:
      trace_ctx = current_trace_context()

    scrubbed_details = self._redactor.scrub_payload(details)
    record = IntentOutcomeRecord(
        correlation_id=correlation_id,
        phase="OUTCOME",
        timestamp_utc=self._now_iso(),
        invocation_id=invocation_id,
        session_id=session_id,
        actor=actor,
        action_type=action_type,
        intended_target=intended_target,
        outcome_status=outcome_status,
        duration_ms=duration_ms,
        intent_fulfilled=intent_fulfilled,
        trace_id=trace_ctx["trace_id"],
        span_id=trace_ctx["span_id"],
        details=scrubbed_details,
    )
    self._records.append(record)
    log_fn = self._logger.info if intent_fulfilled else self._logger.warning
    log_fn("outcome_captured", **record.model_dump())

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> types.Content | None:
    session_id = invocation_context.session.id if invocation_context.session else "unknown"
    self._begin_stage(
        stack_key=f"run:{invocation_context.invocation_id}",
        span_name="radar.run",
        invocation_id=invocation_context.invocation_id,
        session_id=session_id,
        actor=invocation_context.app_name,
        action_type="run_invocation",
        intended_target=invocation_context.agent.name if invocation_context.agent else "root",
        details={"app_name": invocation_context.app_name},
    )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    session_id = invocation_context.session.id if invocation_context.session else "unknown"
    self._end_stage(
        stack_key=f"run:{invocation_context.invocation_id}",
        invocation_id=invocation_context.invocation_id,
        session_id=session_id,
        actor=invocation_context.app_name,
        action_type="run_invocation",
        outcome_status="completed",
        intent_fulfilled=True,
        details={"app_name": invocation_context.app_name},
    )

  @override
  async def on_run_error_callback(
      self, *, invocation_context: InvocationContext, error: Exception
  ) -> None:
    session_id = invocation_context.session.id if invocation_context.session else "unknown"
    self._end_stage(
        stack_key=f"run:{invocation_context.invocation_id}",
        invocation_id=invocation_context.invocation_id,
        session_id=session_id,
        actor=invocation_context.app_name,
        action_type="run_invocation",
        outcome_status="error",
        intent_fulfilled=False,
        details={"error": str(error)},
        error=error,
    )

  @override
  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> types.Content | None:
    self._begin_stage(
        stack_key=f"agent:{callback_context.invocation_id}:{agent.name}",
        span_name=f"radar.agent.{agent.name}",
        invocation_id=callback_context.invocation_id,
        session_id=callback_context.session.id,
        actor=agent.name,
        action_type="agent_execution",
        intended_target=agent.name,
        details={"agent_description": agent.description},
    )
    return None

  @override
  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> types.Content | None:
    self._end_stage(
        stack_key=f"agent:{callback_context.invocation_id}:{agent.name}",
        invocation_id=callback_context.invocation_id,
        session_id=callback_context.session.id,
        actor=agent.name,
        action_type="agent_execution",
        outcome_status="completed",
        intent_fulfilled=True,
        details={"state_keys_count": len(callback_context.state.to_dict())},
    )
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    prompt_preview = ""
    if llm_request.contents and llm_request.contents[-1].parts:
      prompt_preview = (llm_request.contents[-1].parts[0].text or "")[:180]
    self._begin_stage(
        stack_key=f"model:{callback_context.invocation_id}:{callback_context.agent_name}",
        span_name="radar.llm_call",
        invocation_id=callback_context.invocation_id,
        session_id=callback_context.session.id,
        actor=callback_context.agent_name,
        action_type="llm_generation",
        intended_target=str(llm_request.model or "default"),
        details={
            "message_count": len(llm_request.contents),
            "prompt_preview": prompt_preview,
        },
    )
    return None

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    part_count = (
        len(llm_response.content.parts)
        if llm_response.content and llm_response.content.parts
        else 0
    )
    tool_calls = [
        p.function_call.name
        for p in (llm_response.content.parts if llm_response.content and llm_response.content.parts else [])
        if p.function_call and p.function_call.name
    ]
    self._end_stage(
        stack_key=f"model:{callback_context.invocation_id}:{callback_context.agent_name}",
        invocation_id=callback_context.invocation_id,
        session_id=callback_context.session.id,
        actor=callback_context.agent_name,
        action_type="llm_generation",
        outcome_status="completed",
        intent_fulfilled=True,
        details={"part_count": part_count, "requested_tools": tool_calls},
    )
    return None

  @override
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> LlmResponse | None:
    self._end_stage(
        stack_key=f"model:{callback_context.invocation_id}:{callback_context.agent_name}",
        invocation_id=callback_context.invocation_id,
        session_id=callback_context.session.id,
        actor=callback_context.agent_name,
        action_type="llm_generation",
        outcome_status="error",
        intent_fulfilled=False,
        details={"model": str(llm_request.model or "default"), "error": str(error)},
        error=error,
    )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    self._begin_stage(
        stack_key=f"tool:{tool_context.invocation_id}:{tool.name}",
        span_name=f"radar.tool.{tool.name}",
        invocation_id=tool_context.invocation_id,
        session_id=tool_context.session.id,
        actor=tool_context.agent_name,
        action_type="tool_invocation",
        intended_target=tool.name,
        details={"tool_args": tool_args},
    )
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> dict[str, Any] | None:
    del tool_args
    status = str(result.get("status", "success"))
    fulfilled = status not in {"error", "blocked", "failed"}
    self._end_stage(
        stack_key=f"tool:{tool_context.invocation_id}:{tool.name}",
        invocation_id=tool_context.invocation_id,
        session_id=tool_context.session.id,
        actor=tool_context.agent_name,
        action_type="tool_invocation",
        outcome_status=status,
        intent_fulfilled=fulfilled,
        details={"result_status": status, "result_keys": sorted(result.keys())},
    )
    return None

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, Any] | None:
    del tool_args
    self._end_stage(
        stack_key=f"tool:{tool_context.invocation_id}:{tool.name}",
        invocation_id=tool_context.invocation_id,
        session_id=tool_context.session.id,
        actor=tool_context.agent_name,
        action_type="tool_invocation",
        outcome_status="error",
        intent_fulfilled=False,
        details={"tool": tool.name, "error": str(error)},
        error=error,
    )
    return None
