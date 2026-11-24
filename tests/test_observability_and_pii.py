"""Tests for structured JSON logging, Intent vs Outcome capture, OTel tracing, and PII redaction."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from harness_signal_radar.observability.intent_outcome import IntentOutcomeAuditPlugin
from harness_signal_radar.observability.logging_config import (
  configure_structured_logging,
  get_logger,
)
from harness_signal_radar.observability.pii_redactor import SensitiveDataRedactor
from harness_signal_radar.tools import build_fetch_github_releases_tool


class _ToolCallingFakeLlm(BaseLlm):
  def __init__(self) -> None:
    super().__init__(model="gemini-2.5-flash")

  async def generate_content_async(self, llm_request, stream=False):
    del stream
    has_tool_resp = any(
        p.function_response is not None
        for c in (llm_request.contents or [])
        for p in (c.parts or [])
    )
    if not has_tool_resp:
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_function_call(
                      name="fetch_github_agent_framework_releases",
                      args={"repositories": ["google/adk-python"], "lookback_days": 30},
                  )
              ],
          )
      )
    else:
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text="Verified ADK v2.15.0 release.")],
          )
      )


def test_zero_print_statements_in_package_code():
  pkg_root = Path(__file__).resolve().parent.parent / "harness_signal_radar"
  for py_file in pkg_root.rglob("*.py"):
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        assert node.func.id != "print", f"Disallowed print() found in {py_file}"


def test_structured_json_logging_scrubs_pii_and_tokens():
  buf = io.StringIO()
  redactor = SensitiveDataRedactor()
  configure_structured_logging(level="INFO", stream=buf, redactor=redactor)
  logger = get_logger("tests.logging")

  fake_google_key = "AIza" + ("X" * 32)
  fake_gh_token = "ghp_" + ("a1b2c3" * 4)
  logger.info(
      "test_pii_event",
      user_email="alice@example.com",
      raw_message=(
          f"Call 415-555-0142 or SSN 123-45-6789 with key "
          f"{fake_google_key} and {fake_gh_token}"
      ),
  )

  line = buf.getvalue().strip().splitlines()[-1]
  parsed = json.loads(line)
  assert parsed["severity"] == "INFO"
  assert parsed["event"] == "test_pii_event"
  assert "timestamp" in parsed
  assert "trace_id" in parsed
  assert parsed["user_email"] == "[REDACTED_EMAIL]"
  assert "alice@example.com" not in line
  assert "415-555-0142" not in line
  assert "123-45-6789" not in line
  assert fake_google_key not in line
  assert fake_gh_token not in line


def test_optional_cloud_dlp_integration():
  fake_dlp = MagicMock()
  fake_dlp.deidentify_content.return_value = MagicMock(
      item=MagicMock(value="scrubbed via dlp")
  )
  redactor = SensitiveDataRedactor(
      enable_cloud_dlp=True,
      project_id="demo-project",
      dlp_client=fake_dlp,
  )
  out = redactor.scrub_text("hello world")
  assert out == "scrubbed via dlp"
  fake_dlp.deidentify_content.assert_called_once()


async def test_intent_vs_outcome_capture_and_linked_otel_spans(span_exporter):
  audit_plugin = IntentOutcomeAuditPlugin()
  agent = Agent(
      name="audit_probe_agent",
      model=_ToolCallingFakeLlm(),
      instruction="Fetch ADK releases.",
      tools=[build_fetch_github_releases_tool()],
  )
  app = App(name="audit_test_app", root_agent=agent, plugins=[audit_plugin])
  runner = InMemoryRunner(app=app)
  session = await runner.session_service.create_session(
      app_name="audit_test_app", user_id="u_obs"
  )

  msg = types.Content(
      role="user",
      parts=[types.Part.from_text(text="Check ADK release for user@example.com")],
  )
  _ = [
      e
      async for e in runner.run_async(
          user_id="u_obs", session_id=session.id, new_message=msg
      )
  ]

  records = audit_plugin.records
  intents = [r for r in records if r.phase == "INTENT"]
  outcomes = [r for r in records if r.phase == "OUTCOME"]

  assert len(intents) >= 3  # run, agent, llm, tool
  assert len(outcomes) == len(intents)
  assert {r.correlation_id for r in intents} == {r.correlation_id for r in outcomes}

  tool_intents = [r for r in intents if r.action_type == "tool_invocation"]
  tool_outcomes = [r for r in outcomes if r.action_type == "tool_invocation"]
  assert len(tool_intents) == 1
  assert tool_outcomes[0].intent_fulfilled is True
  assert tool_outcomes[0].outcome_status == "success"
  assert tool_outcomes[0].duration_ms is not None and tool_outcomes[0].duration_ms >= 0.0

  # Verify linked OpenTelemetry spans from run -> agent -> llm / tool
  spans = span_exporter.get_finished_spans()
  span_names = {s.name for s in spans}
  assert "radar.run" in span_names
  assert "radar.agent.audit_probe_agent" in span_names
  assert "radar.llm_call" in span_names
  assert "radar.tool.fetch_github_agent_framework_releases" in span_names

  trace_ids = {s.context.trace_id for s in spans if s.name.startswith("radar.")}
  assert len(trace_ids) == 1, "All stages of a run must share one linked trace_id"
  child_spans = [s for s in spans if s.name != "radar.run" and s.name.startswith("radar.")]
  assert all(s.parent is not None for s in child_spans)
