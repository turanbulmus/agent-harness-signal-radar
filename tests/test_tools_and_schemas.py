"""Tests for strict JSON schemas, descriptive naming, docstrings, and guided error recovery."""

from __future__ import annotations

import httpx
import pytest
from google.adk import Agent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types
from harness_signal_radar.schemas import NormalizedSignalItem, SignalCorrelationInput
from harness_signal_radar.tools import (
  build_correlate_signals_tool,
  build_editorial_approval_tool,
  build_fetch_github_releases_tool,
  build_publish_digest_tool,
  build_retrieve_archived_briefings_tool,
  build_scan_hackernews_signals_tool,
  build_search_arxiv_papers_tool,
  correlate_and_score_research_signals,
)


class _NoopLlm(BaseLlm):
  def __init__(self) -> None:
    super().__init__(model="noop")

  async def generate_content_async(self, llm_request, stream=False):
    del llm_request, stream
    yield LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text="ok")])
    )


async def _make_tool_context(app_name: str) -> ToolContext:
  agent = Agent(name="probe", model=_NoopLlm(), instruction="Probe.")
  runner = InMemoryRunner(app=App(name=app_name, root_agent=agent))
  session = await runner.session_service.create_session(app_name=app_name, user_id="u1")
  inv_ctx = runner._new_invocation_context(session=session)
  return ToolContext(inv_ctx)


def test_all_tools_expose_comprehensive_docstrings_and_explicit_schemas():
  strict_tools = [
      build_fetch_github_releases_tool(),
      build_search_arxiv_papers_tool(),
      build_scan_hackernews_signals_tool(),
      build_correlate_signals_tool(),
      build_retrieve_archived_briefings_tool(),
      build_publish_digest_tool(),
  ]
  expected_names = {
      "fetch_github_agent_framework_releases",
      "search_arxiv_agent_orchestration_papers",
      "scan_hackernews_workflow_automation_signals",
      "correlate_and_score_research_signals",
      "retrieve_archived_signal_briefings",
      "publish_approved_executive_signal_digest",
  }
  assert {t.name for t in strict_tools} == expected_names

  for tool in strict_tools:
    assert len(tool.description) >= 80
    decl = tool._get_declaration()
    assert decl is not None
    assert decl.parameters_json_schema is not None
    assert decl.response_json_schema is not None
    assert decl.parameters_json_schema.get("additionalProperties") is False

  hitl_tool = build_editorial_approval_tool()
  assert hitl_tool.name == "request_human_editorial_publication_approval"
  assert hitl_tool.is_long_running is True
  assert len(hitl_tool.description) >= 80


async def test_strict_tool_returns_guided_recovery_on_invalid_arguments():
  tool = build_fetch_github_releases_tool()
  tool_ctx = await _make_tool_context("test_invalid_args")

  result = await tool.run_async(
      args={
          "repositories": ["invalid_slug_without_slash"],
          "lookback_days": 30,
          "unexpected_extra": "should_forbid",
      },
      tool_context=tool_ctx,
  )

  assert result["status"] == "error"
  advice = result["recovery_advice"]
  assert advice["error_code"] == "INVALID_TOOL_ARGUMENTS"
  assert advice["retryable"] is True
  assert "GitHubReleaseQueryInput" in advice["recovery_instructions"]
  assert "repositories" in advice["suggested_parameters"]


async def test_live_http_failure_returns_guided_recovery_when_offline_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setenv("OFFLINE_FIXTURE_FALLBACK", "false")
  monkeypatch.setenv("RADAR_FORCE_LIVE_HTTP", "true")

  async def _failing_get(*args, **kwargs):
    del args, kwargs
    raise httpx.ConnectTimeout("Simulated upstream timeout")

  monkeypatch.setattr(httpx.AsyncClient, "get", _failing_get)

  tool = build_search_arxiv_papers_tool()
  tool_ctx = await _make_tool_context("test_http_recovery")
  result = await tool.run_async(
      args={"search_query": "agent harness workflow", "max_results": 3},
      tool_context=tool_ctx,
  )

  assert result["status"] == "error"
  assert result["recovery_advice"]["error_code"] == "ARXIV_QUERY_FAILED"
  assert result["recovery_advice"]["retryable"] is True


def test_signal_correlator_deduplicates_and_detects_convergence():
  items = [
      NormalizedSignalItem(
          signal_id="gh:1",
          title="ADK Graph Workflows",
          canonical_url="https://github.com/google/adk-python/releases/tag/v2.15.0",
          source_category="github_release",
          published_at="2026-11-18T00:00:00Z",
          summary="Typed workflow orchestration and compaction.",
          authority_score=0.95,
          engagement_velocity=0.90,
          harness_or_workflow_tags=["graph_orchestration", "history_compaction"],
      ),
      NormalizedSignalItem(
          signal_id="gh:dup",
          title="ADK Graph Workflows Duplicate",
          canonical_url="https://github.com/google/adk-python/releases/tag/v2.15.0/",
          source_category="github_release",
          published_at="2026-11-18T00:00:00Z",
          summary="Duplicate URL with lower authority.",
          authority_score=0.50,
          engagement_velocity=0.40,
          harness_or_workflow_tags=["graph_orchestration"],
      ),
      NormalizedSignalItem(
          signal_id="arxiv:1",
          title="Typed DAG Harnesses",
          canonical_url="https://arxiv.org/abs/2510.14201",
          source_category="arxiv_paper",
          published_at="2026-11-15T00:00:00Z",
          summary="Bounded verification in graph orchestration.",
          authority_score=0.91,
          engagement_velocity=0.80,
          harness_or_workflow_tags=["graph_orchestration"],
      ),
  ]

  output = correlate_and_score_research_signals(
      SignalCorrelationInput(signals=items, focus_area="graph orchestration", top_k=5)
  )
  assert output.status == "success"
  assert output.deduplicated_count == 1
  assert len(output.ranked_signals) == 2
  assert output.ranked_signals[0].signal_id == "gh:1"
  assert any("graph_orchestration" in hl for hl in output.convergence_highlights)
