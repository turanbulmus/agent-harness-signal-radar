"""Deterministic and live evaluation runner for the golden dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types
from harness_signal_radar.config.settings import RadarSettings
from harness_signal_radar.guardrails.policy_plugin import SecurityAndPolicyGuardrailPlugin
from harness_signal_radar.observability.intent_outcome import IntentOutcomeAuditPlugin
from harness_signal_radar.observability.pii_redactor import SensitiveDataRedactor
from harness_signal_radar.routing.model_router import TaskTier
from harness_signal_radar.schemas import (
  ConsolidatedResearchBrief,
  DeepResearchWorkflowInput,
  ResearchPlanSchema,
  SelfEvalCritiqueVerdict,
)
from harness_signal_radar.tools import build_fetch_github_releases_tool, build_publish_digest_tool
from harness_signal_radar.workflows import build_deep_research_workflow
from pydantic import BaseModel, ConfigDict


class EvalCaseResult(BaseModel):
  """Result for a single golden evaluation case."""

  model_config = ConfigDict(extra="forbid")

  eval_id: str
  category: str
  passed: bool
  score: float
  details: str


class EvaluationSummary(BaseModel):
  """Aggregated evaluation report across the golden dataset."""

  model_config = ConfigDict(extra="forbid")

  total_cases: int
  passed_cases: int
  overall_score: float
  metrics: dict[str, float]
  case_results: list[EvalCaseResult]


class _DeterministicEvalLlm(BaseLlm):
  """Scripted BaseLlm executing realistic tool calls and structured workflow schemas."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  def __init__(self, *, model_name: str, tier: TaskTier) -> None:
    super().__init__(model=model_name)
    self._tier = tier

  async def generate_content_async(self, llm_request: Any, stream: bool = False):
    del stream
    system_text = ""
    if llm_request.config and llm_request.config.system_instruction:
      system_text = str(llm_request.config.system_instruction)

    has_function_response = any(
        part.function_response is not None
        for content in (llm_request.contents or [])
        for part in (content.parts or [])
    )

    if "ResearchPlannerAgent" in system_text:
      plan = ResearchPlanSchema(
          core_thesis_question="How are agent harnesses and workflow automation evolving?",
          github_repositories=[
              "google/adk-python",
              "langchain-ai/langgraph",
              "modelcontextprotocol/specification",
              "a2aproject/A2A",
          ],
          arxiv_search_queries=[
              "deterministic agent harness workflow compaction guardrails"
          ],
          community_keywords=["agent harness", "workflow automation", "MCP"],
          priority_themes=[
              "graph_orchestration",
              "history_compaction",
              "mcp_sandbox",
              "hitl_checkpointing",
          ],
      )
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=plan.model_dump_json())],
          )
      )
      return

    if "GitHubReleaseHarvesterAgent" in system_text:
      if not has_function_response:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="fetch_github_agent_framework_releases",
                        args={
                            "repositories": [
                                "google/adk-python",
                                "langchain-ai/langgraph",
                                "modelcontextprotocol/specification",
                            ],
                            "lookback_days": 30,
                            "include_prereleases": False,
                        },
                    )
                ],
            )
        )
        return
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text="Collected GitHub releases for google/adk-python, langgraph, MCP, and A2A."
                  )
              ],
          )
      )
      return

    if "ArxivResearchHarvesterAgent" in system_text:
      if not has_function_response:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="search_arxiv_agent_orchestration_papers",
                        args={
                            "search_query": "agent harness workflow compaction guardrails",
                            "max_results": 3,
                        },
                    )
                ],
            )
        )
        return
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text="Collected scholarly findings on bounded DAG harnesses and compaction."
                  )
              ],
          )
      )
      return

    if "CommunitySignalHarvesterAgent" in system_text:
      if not has_function_response:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="scan_hackernews_workflow_automation_signals",
                        args={
                            "keywords": ["agent harness", "workflow automation", "MCP"],
                            "min_points": 25,
                            "lookback_days": 30,
                        },
                    )
                ],
            )
        )
        return
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text="Collected community engineering threads on MCP sandboxes and typed DAGs."
                  )
              ],
          )
      )
      return

    if "BriefingSynthesizerAgent" in system_text:
      brief = ConsolidatedResearchBrief(
          headline="Agent Harness & Workflow Automation Convergence Report",
          executive_summary=(
              "Across Google ADK 2.15, LangGraph 1.0.10, MCP, and A2A, production teams "
              "are converging on typed Workflow graphs, token-threshold history compaction, "
              "memory consolidation, and policy-gated HITL checkpoints."
          ),
          agent_harness_trends=[
              "Strict JSON input/output schemas and MCP tool sandboxes eliminate malformed tool loops.",
              "EventsCompactionConfig and sliding-window summarization trim long-horizon context bloat.",
          ],
          workflow_automation_trends=[
              "Typed DAG Workflow graphs with JoinNode fan-in replace fragile monolithic ReAct loops.",
              "Durable checkpointing and RequestInput HITL gates govern high-stakes external writes.",
          ],
          architectural_implications=[
              "Route extraction and self-eval tasks to Flash while reserving Pro for planning and synthesis.",
              "Enforce PII redaction and Intent-vs-Outcome OpenTelemetry spans across every tool boundary.",
          ],
          citations=[
              "https://github.com/google/adk-python/releases/tag/v2.15.0",
              "https://github.com/langchain-ai/langgraph/releases/tag/v1.0.10",
              "https://github.com/modelcontextprotocol/modelcontextprotocol/releases",
              "https://arxiv.org/abs/2510.14201",
              "https://news.ycombinator.com/item?id=42901821",
          ],
          confidence_score=0.93,
      )
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=brief.model_dump_json())],
          )
      )
      return

    if "SelfEvalCriticAgent" in system_text:
      verdict = SelfEvalCritiqueVerdict(
          passed=True,
          quality_score=0.95,
          citation_coverage_ok=True,
          hallucination_risk="low",
          improvement_instructions="All claims map to verified citations with low risk.",
      )
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=verdict.model_dump_json())],
          )
      )
      return

    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Acknowledged.")],
        )
    )


async def _run_workflow_eval_case(
    case: dict[str, Any], tmp_dir: str
) -> EvalCaseResult:
  settings = RadarSettings(
      vector_db_path=str(Path(tmp_dir) / "vectors.db"),
      offline_fixture_fallback=True,
  )
  model_map = {
      TaskTier.DEEP_REASONING: _DeterministicEvalLlm(
          model_name="gemini-2.5-pro", tier=TaskTier.DEEP_REASONING
      ),
      TaskTier.FAST_EXTRACTION: _DeterministicEvalLlm(
          model_name="gemini-2.5-flash", tier=TaskTier.FAST_EXTRACTION
      ),
  }
  workflow = build_deep_research_workflow(
      settings=settings,
      model_override=model_map,
  )
  audit_plugin = IntentOutcomeAuditPlugin()
  guardrail_plugin = SecurityAndPolicyGuardrailPlugin()
  app = App(
      name=f"eval_{case['eval_id']}",
      root_agent=workflow,
      plugins=[guardrail_plugin, audit_plugin],
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = InMemoryRunner(app=app)
  session = await runner.session_service.create_session(
      app_name=app.name, user_id="eval_user"
  )

  wf_input = DeepResearchWorkflowInput(
      research_question=case["user_query"],
      lookback_days=30,
      require_external_publication=bool(case.get("require_external_publication", False)),
      target_channel="engineering_digest",
  )
  msg = types.Content(
      role="user",
      parts=[types.Part.from_text(text=wf_input.model_dump_json())],
  )
  events = [
      event
      async for event in runner.run_async(
          user_id="eval_user",
          session_id=session.id,
          new_message=msg,
      )
  ]

  # Handle HITL pause & resume when external publication is required
  if case.get("require_external_publication"):
    fc_events = [e for e in events if e.get_function_calls()]
    if not fc_events:
      return EvalCaseResult(
          eval_id=case["eval_id"],
          category=case["category"],
          passed=False,
          score=0.0,
          details="Expected RequestInput interrupt for HITL gate, but none was emitted.",
      )
    fc = fc_events[-1].get_function_calls()[0]
    decision_payload = case["human_decision"]
    resume_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response=decision_payload,
                )
            )
        ],
    )
    resumed_events = [
        event
        async for event in runner.run_async(
            user_id="eval_user",
            session_id=session.id,
            new_message=resume_msg,
        )
    ]
    events.extend(resumed_events)

  updated_session = await runner.session_service.get_session(
      app_name=app.name, user_id="eval_user", session_id=session.id
  )
  final_brief = (updated_session.state if updated_session else {}).get(
      "final_research_brief", {}
  )

  # Check trajectory tools if expected
  if "expected_tools" in case:
    invoked_tools = {
        rec.intended_target
        for rec in audit_plugin.records
        if rec.action_type == "tool_invocation" and rec.phase == "INTENT"
    }
    missing_tools = set(case["expected_tools"]) - invoked_tools
    if missing_tools:
      return EvalCaseResult(
          eval_id=case["eval_id"],
          category=case["category"],
          passed=False,
          score=0.5,
          details=f"Missing expected tool calls: {sorted(missing_tools)}",
      )

  # Check citation domains if expected
  if "expected_citation_domains" in case:
    citations = " ".join(final_brief.get("citations", []))
    missing_domains = [
        d for d in case["expected_citation_domains"] if d not in citations
    ]
    if missing_domains:
      return EvalCaseResult(
          eval_id=case["eval_id"],
          category=case["category"],
          passed=False,
          score=0.5,
          details=f"Missing expected citation domains: {missing_domains}",
      )

  if case.get("expected_delivery_mode"):
    receipt = final_brief.get("publication_receipt", {})
    approval_state = (updated_session.state if updated_session else {}).get(
        "editorial_approval_status"
    )
    if (
        receipt.get("delivery_mode") != case["expected_delivery_mode"]
        or approval_state != case["expected_approval_status"]
    ):
      return EvalCaseResult(
          eval_id=case["eval_id"],
          category=case["category"],
          passed=False,
          score=0.0,
          details=(
              f"Unexpected HITL publish state: approval={approval_state}, "
              f"receipt={receipt}"
          ),
      )

  return EvalCaseResult(
      eval_id=case["eval_id"],
      category=case["category"],
      passed=True,
      score=1.0,
      details="Passed trajectory, schema, citation grounding, and HITL assertions.",
  )


async def evaluate_golden_dataset(
    dataset_path: Path | None = None,
) -> EvaluationSummary:
  """Executes all golden dataset evaluation cases and computes static regression metrics."""
  resolved_path = dataset_path or (Path(__file__).parent / "golden_dataset.json")
  dataset = json.loads(resolved_path.read_text(encoding="utf-8"))
  cases: list[dict[str, Any]] = dataset["cases"]

  results: list[EvalCaseResult] = []
  category_scores: dict[str, list[float]] = defaultdict(list)

  with tempfile.TemporaryDirectory() as tmp_dir:
    for case in cases:
      category = case["category"]
      if category in {
          "multi_source_deep_research",
          "protocol_convergence",
          "scholarly_synthesis",
          "hitl_pause_resume",
      }:
        res = await _run_workflow_eval_case(case, tmp_dir)
      elif category == "guided_error_handling":
        tool = build_fetch_github_releases_tool()
        runner = InMemoryRunner(app=App(name="tool_eval", root_agent=build_deep_research_workflow(
            settings=RadarSettings(vector_db_path=str(Path(tmp_dir) / "v2.db")),
            model_override=_DeterministicEvalLlm(
                model_name="gemini-2.5-flash", tier=TaskTier.FAST_EXTRACTION
            ),
        )))
        session = await runner.session_service.create_session(
            app_name="tool_eval", user_id="u1"
        )
        inv_ctx = runner._new_invocation_context(session=session)
        tool_ctx = ToolContext(inv_ctx)
        out = await tool.run_async(args=case["tool_args"], tool_context=tool_ctx)
        advice = out.get("recovery_advice") or {}
        ok = (
            out.get("status") == case["expected_status"]
            and advice.get("error_code") == case["expected_error_code"]
            and advice.get("retryable") == case["expected_retryable"]
        )
        res = EvalCaseResult(
            eval_id=case["eval_id"],
            category=category,
            passed=ok,
            score=1.0 if ok else 0.0,
            details=f"Guided error response: {advice.get('error_code')}",
        )
      elif category == "security_guardrail":
        plugin = SecurityAndPolicyGuardrailPlugin()
        reason = plugin.inspect_untrusted_text(case["user_query"])
        ok = bool(reason) is bool(case["expected_blocked"])
        res = EvalCaseResult(
            eval_id=case["eval_id"],
            category=category,
            passed=ok,
            score=1.0 if ok else 0.0,
            details=reason or "Not blocked",
        )
      elif category == "pii_redaction":
        redactor = SensitiveDataRedactor()
        scrubbed = redactor.scrub_text(case["raw_text"])
        no_leaks = all(s not in scrubbed for s in case["forbidden_substrings"])
        has_markers = all(m in scrubbed for m in case["expected_redaction_markers"])
        ok = no_leaks and has_markers
        res = EvalCaseResult(
            eval_id=case["eval_id"],
            category=category,
            passed=ok,
            score=1.0 if ok else 0.0,
            details=f"Scrubbed output: {scrubbed}",
        )
      elif category == "hitl_policy_gate":
        plugin = SecurityAndPolicyGuardrailPlugin()
        publish_tool = build_publish_digest_tool()
        runner = InMemoryRunner(app=App(name="hitl_eval", root_agent=build_deep_research_workflow(
            settings=RadarSettings(vector_db_path=str(Path(tmp_dir) / "v3.db")),
            model_override=_DeterministicEvalLlm(
                model_name="gemini-2.5-flash", tier=TaskTier.FAST_EXTRACTION
            ),
        )))
        session = await runner.session_service.create_session(
            app_name="hitl_eval", user_id="u1"
        )
        inv_ctx = runner._new_invocation_context(session=session)
        tool_ctx = ToolContext(inv_ctx)
        blocked_out = await plugin.before_tool_callback(
            tool=publish_tool,
            tool_args=case["tool_args"],
            tool_context=tool_ctx,
        )
        advice = (blocked_out or {}).get("recovery_advice") or {}
        ok = (
            blocked_out is not None
            and blocked_out.get("status") == case["expected_status"]
            and advice.get("error_code") == case["expected_error_code"]
        )
        res = EvalCaseResult(
            eval_id=case["eval_id"],
            category=category,
            passed=ok,
            score=1.0 if ok else 0.0,
            details=f"HITL policy enforcement: {advice.get('error_code')}",
        )
      else:
        res = EvalCaseResult(
            eval_id=case["eval_id"],
            category=category,
            passed=False,
            score=0.0,
            details=f"Unknown eval category: {category}",
        )

      results.append(res)
      category_scores[category].append(res.score)

  overall_score = round(sum(r.score for r in results) / max(1, len(results)), 4)
  passed_cases = sum(1 for r in results if r.passed)
  metrics = {
      cat: round(sum(scores) / len(scores), 4)
      for cat, scores in sorted(category_scores.items())
  }
  return EvaluationSummary(
      total_cases=len(results),
      passed_cases=passed_cases,
      overall_score=overall_score,
      metrics=metrics,
      case_results=results,
  )


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Run golden dataset regression evaluation for Agent Harness Signal Radar."
  )
  parser.add_argument(
      "--dataset",
      type=Path,
      default=Path(__file__).parent / "golden_dataset.json",
      help="Path to golden dataset JSON file.",
  )
  parser.add_argument(
      "--min-score",
      type=float,
      default=0.95,
      help="Minimum overall score required for passing CI gate.",
  )
  args = parser.parse_args()

  summary = asyncio.run(evaluate_golden_dataset(args.dataset))
  sys.stdout.write(summary.model_dump_json(indent=2) + "\n")
  return 0 if summary.overall_score >= args.min_score else 1


if __name__ == "__main__":
  raise SystemExit(main())
