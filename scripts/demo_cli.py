"""CLI runner for demonstrating simulated and live deep research signal consolidation."""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import Runner
from google.genai import types
from harness_signal_radar.agent import build_radar_app
from harness_signal_radar.config.settings import RadarSettings
from harness_signal_radar.guardrails.policy_plugin import SecurityAndPolicyGuardrailPlugin
from harness_signal_radar.memory.async_consolidator import (
  AsyncMemoryConsolidator,
  AsyncMemoryLifecyclePlugin,
)
from harness_signal_radar.memory.compaction import (
  TokenBudgetCompactionPlugin,
  build_context_cache_config,
  build_events_compaction_config,
)
from harness_signal_radar.memory.persistence import (
  PersistentSignalVectorStore,
  create_persistent_memory_service,
  create_persistent_session_service,
)
from harness_signal_radar.observability.intent_outcome import IntentOutcomeAuditPlugin
from harness_signal_radar.observability.logging_config import configure_structured_logging
from harness_signal_radar.observability.pii_redactor import SensitiveDataRedactor
from harness_signal_radar.observability.tracing import (
  clear_registered_exporters,
  register_span_exporter,
)
from harness_signal_radar.routing.model_router import (
  DynamicModelRoutingPlugin,
  StrategicModelRouter,
  TaskTier,
)
from harness_signal_radar.schemas import (
  ConsolidatedResearchBrief,
  DeepResearchWorkflowInput,
  ResearchPlanSchema,
  SelfEvalCritiqueVerdict,
)
from harness_signal_radar.workflows import (
  build_deep_research_workflow,
  format_research_brief_markdown,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ConfigDict


class _DemoScriptedLlm(BaseLlm):
  """Deterministic multi-stage FakeLlm demonstrating parallel fan-out, revision, and HITL."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  def __init__(self, *, model_name: str) -> None:
    super().__init__(model=model_name)
    self._critic_calls = 0

  async def generate_content_async(self, llm_request: Any, stream: bool = False):
    del stream
    system_text = ""
    if llm_request.config and llm_request.config.system_instruction:
      system_text = str(llm_request.config.system_instruction)

    has_tool_response = any(
        part.function_response is not None
        for content in (llm_request.contents or [])
        for part in (content.parts or [])
    )

    if "ResearchPlannerAgent" in system_text:
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text=ResearchPlanSchema(
                          core_thesis_question=(
                              "How are AI agent harnesses and workflow automation evolving?"
                          ),
                          github_repositories=[
                              "google/adk-python",
                              "langchain-ai/langgraph",
                              "modelcontextprotocol/specification",
                              "a2aproject/A2A",
                              "temporalio/sdk-python",
                          ],
                          arxiv_search_queries=[
                              "deterministic agent harnesses typed workflow graphs compaction"
                          ],
                          community_keywords=[
                              "agent harness",
                              "workflow automation",
                              "MCP",
                          ],
                          priority_themes=[
                              "graph_orchestration",
                              "history_compaction",
                              "mcp_sandbox",
                              "hitl_checkpointing",
                          ],
                      ).model_dump_json()
                  )
              ],
          )
      )
      return

    if "GitHubReleaseHarvesterAgent" in system_text:
      if not has_tool_response:
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
                                "a2aproject/A2A",
                                "temporalio/sdk-python",
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
                      text="Harvested GitHub framework releases for ADK, LangGraph, MCP, A2A, and Temporal."
                  )
              ],
          )
      )
      return

    if "ArxivResearchHarvesterAgent" in system_text:
      if not has_tool_response:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="search_arxiv_agent_orchestration_papers",
                        args={
                            "search_query": "deterministic agent harness workflow compaction",
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
                      text="Harvested 3 arXiv preprints on DAG harnesses, compaction, and policy guardrails."
                  )
              ],
          )
      )
      return

    if "CommunitySignalHarvesterAgent" in system_text:
      if not has_tool_response:
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
                      text="Harvested practitioner signals on MCP tool sandboxes and CI trajectory evals."
                  )
              ],
          )
      )
      return

    if "BriefingSynthesizerAgent" in system_text:
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text=ConsolidatedResearchBrief(
                          headline="Agent Harness & Workflow Automation Signal Consolidation (Initial Draft)",
                          executive_summary=(
                              "Across Google ADK 2.15, LangGraph 1.0.10, MCP, A2A, and Temporal, "
                              "teams are replacing unbounded loops with typed DAG workflows."
                          ),
                          agent_harness_trends=[
                              "Scoped MCP elicitation and strict JSON tool schemas reduce tool call failures.",
                          ],
                          workflow_automation_trends=[
                              "Typed DAG orchestration with JoinNode fan-in and durable checkpoints dominates.",
                          ],
                          architectural_implications=[
                              "Adopt strict Pydantic schemas and CI golden-dataset evals.",
                          ],
                          citations=[
                              "https://github.com/google/adk-python/releases/tag/v2.15.0",
                          ],
                          confidence_score=0.78,
                      ).model_dump_json()
                  )
              ],
          )
      )
      return

    if "BriefingRevisionAgent" in system_text:
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text=ConsolidatedResearchBrief(
                          headline="Agent Harness & Workflow Automation Signal Consolidation Brief",
                          executive_summary=(
                              "Cross-source synthesis across GitHub releases, arXiv preprints, and "
                              "practitioner reports shows decisive convergence toward deterministic "
                              "agent harnesses: schema-enforced tools, token-threshold compaction, "
                              "typed Workflow graphs, and HITL-gated external side effects."
                          ),
                          agent_harness_trends=[
                              "Strict JSON input/output tool declarations with guided recovery payloads (ADK, MCP).",
                              "Sliding-window and token-threshold history compaction paired with async memory banks.",
                              "Deterministic FakeLlm trajectory evals gating CI/CD pipelines alongside OTel tracing.",
                          ],
                          workflow_automation_trends=[
                              "Explicit Workflow graphs (fan-out, JoinNode fan-in, conditional routing) replace ReAct loops.",
                              "Durable saga checkpointing (LangGraph, Temporal) and resumable RequestInput HITL gates.",
                              "Strategic Pro/Flash model routing balancing planning rigor with extraction speed.",
                          ],
                          architectural_implications=[
                              "Separate reasoning (Pro) from parallel extraction and critic passes (Flash).",
                              "Enforce PII redaction and Secret Manager injection across all observability and storage layers.",
                              "Gate external publishing behind policy plugins and human editorial approval.",
                          ],
                          citations=[
                              "https://github.com/google/adk-python/releases/tag/v2.15.0",
                              "https://github.com/langchain-ai/langgraph/releases/tag/v1.0.10",
                              "https://github.com/modelcontextprotocol/modelcontextprotocol/releases",
                              "https://github.com/a2aproject/A2A/releases",
                              "https://arxiv.org/abs/2510.14201",
                              "https://arxiv.org/abs/2511.03184",
                              "https://news.ycombinator.com/item?id=42901821",
                          ],
                          confidence_score=0.94,
                      ).model_dump_json()
                  )
              ],
          )
      )
      return

    if "SelfEvalCriticAgent" in system_text:
      self._critic_calls += 1
      if self._critic_calls == 1:
        verdict = SelfEvalCritiqueVerdict(
            passed=False,
            quality_score=0.72,
            citation_coverage_ok=False,
            hallucination_risk="medium",
            improvement_instructions=(
                "Expand citations to include arXiv and Hacker News URLs and detail compaction."
            ),
        )
      else:
        verdict = SelfEvalCritiqueVerdict(
            passed=True,
            quality_score=0.95,
            citation_coverage_ok=True,
            hallucination_risk="low",
            improvement_instructions="All sections grounded in verified multi-source citations.",
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
            parts=[types.Part.from_text(text="Completed.")],
        )
    )


async def run_simulated_demo(question: str) -> None:
  """Executes the full deep research workflow offline with scripted LLM responses."""
  span_exporter = InMemorySpanExporter()
  register_span_exporter(span_exporter)

  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = Path(tmp_dir)
      settings = RadarSettings(
          session_db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}",
          memory_db_path=str(tmp_path / "memory.db"),
          vector_db_path=str(tmp_path / "vectors.db"),
          offline_fixture_fallback=True,
          log_level="WARNING",
      )
      redactor = SensitiveDataRedactor()
      configure_structured_logging(level=settings.log_level, redactor=redactor)

      vector_store = PersistentSignalVectorStore(
          db_path=settings.vector_db_path, redactor=redactor
      )
      memory_service = create_persistent_memory_service(settings, redactor=redactor)
      session_service = create_persistent_session_service(settings)
      consolidator = AsyncMemoryConsolidator(
          memory_service=memory_service,
          vector_store=vector_store,
          redactor=redactor,
      )

      model_map = {
          TaskTier.DEEP_REASONING: _DemoScriptedLlm(model_name="gemini-2.5-pro"),
          TaskTier.FAST_EXTRACTION: _DemoScriptedLlm(model_name="gemini-2.5-flash"),
      }
      workflow = build_deep_research_workflow(
          settings=settings,
          router=StrategicModelRouter.from_settings(settings),
          model_override=model_map,
          vector_store=vector_store,
      )
      audit_plugin = IntentOutcomeAuditPlugin(redactor=redactor)
      app = App(
          name="demo_signal_radar",
          root_agent=workflow,
          plugins=[
              SecurityAndPolicyGuardrailPlugin(redactor=redactor),
              DynamicModelRoutingPlugin(router=StrategicModelRouter.from_settings(settings)),
              TokenBudgetCompactionPlugin(redactor=redactor),
              ContextFilterPlugin(num_invocations_to_keep=8),
              audit_plugin,
              AsyncMemoryLifecyclePlugin(consolidator=consolidator),
          ],
          events_compaction_config=build_events_compaction_config(),
          context_cache_config=build_context_cache_config(),
          resumability_config=ResumabilityConfig(is_resumable=True),
      )

      runner = Runner(
          app=app,
          session_service=session_service,
          memory_service=memory_service,
      )
      session = await session_service.create_session(
          app_name=app.name, user_id="demo_engineer"
      )

      payload = DeepResearchWorkflowInput(
          research_question=question,
          lookback_days=30,
          require_external_publication=True,
          target_channel="engineering_digest",
      )
      user_msg = types.Content(
          role="user",
          parts=[types.Part.from_text(text=payload.model_dump_json())],
      )

      events = [
          e
          async for e in runner.run_async(
              user_id="demo_engineer",
              session_id=session.id,
              new_message=user_msg,
          )
      ]

      fc_events = [e for e in events if e.get_function_calls()]
      if fc_events:
        fc = fc_events[-1].get_function_calls()[0]
        approval_msg = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={
                            "approved": True,
                            "reviewer_note": "Approved in simulated CLI demo.",
                            "approved_channel": "engineering_digest",
                        },
                    )
                )
            ],
        )
        resumed = [
            e
            async for e in runner.run_async(
                user_id="demo_engineer",
                session_id=session.id,
                new_message=approval_msg,
            )
        ]
        events.extend(resumed)

      await consolidator.wait_until_idle(timeout=5.0)

      persisted = await session_service.get_session(
          app_name=app.name, user_id="demo_engineer", session_id=session.id
      )
      final_brief = (persisted.state if persisted else {}).get(
          "final_research_brief", {}
      )
      markdown_report = format_research_brief_markdown(final_brief)
      spans = span_exporter.get_finished_spans()
      intent_count = sum(1 for r in audit_plugin.records if r.phase == "INTENT")
      outcome_count = sum(1 for r in audit_plugin.records if r.phase == "OUTCOME")

      sys.stdout.write(markdown_report + "\n\n")
      sys.stdout.write("---\n")
      sys.stdout.write(
          f"Telemetry Summary: spans={len(spans)}, "
          f"intent_records={intent_count}, outcome_records={outcome_count}, "
          f"async_memory_jobs={consolidator.completed_count}\n"
      )
      await memory_service.close()
  finally:
    clear_registered_exporters()


async def run_live_query(question: str) -> None:
  """Executes the full coordinator app against live Gemini models."""
  settings = RadarSettings.from_env(load_env_file=True)
  settings.ensure_local_data_dirs()
  session_service = create_persistent_session_service(settings)
  memory_service = create_persistent_memory_service(settings)
  consolidator = AsyncMemoryConsolidator(
      memory_service=memory_service,
      vector_store=PersistentSignalVectorStore(db_path=settings.vector_db_path),
  )
  app = build_radar_app(settings=settings, consolidator=consolidator)
  runner = Runner(
      app=app,
      session_service=session_service,
      memory_service=memory_service,
  )
  session = await session_service.create_session(
      app_name=app.name, user_id="cli_user"
  )
  msg = types.Content(role="user", parts=[types.Part.from_text(text=question)])
  async for event in runner.run_async(
      user_id="cli_user", session_id=session.id, new_message=msg
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          sys.stdout.write(f"[{event.author}] {part.text}\n")
  await consolidator.wait_until_idle(timeout=5.0)
  await memory_service.close()


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Agent Harness & Workflow Automation Deep Research Signal Radar CLI"
  )
  parser.add_argument(
      "--query",
      default=(
          "Consolidate the latest signals on AI agent harnesses and workflow automation "
          "across ADK, LangGraph, MCP, A2A, and arXiv research."
      ),
      help="Research question to investigate.",
  )
  mode_group = parser.add_mutually_exclusive_group()
  mode_group.add_argument(
      "--simulate",
      action="store_true",
      help="Run offline deterministic simulation exercising all workflow stages.",
  )
  mode_group.add_argument(
      "--live",
      action="store_true",
      help="Run live query using configured GOOGLE_API_KEY or Vertex AI credentials.",
  )
  args = parser.parse_args()

  if args.live:
    asyncio.run(run_live_query(args.query))
  else:
    asyncio.run(run_simulated_demo(args.query))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
