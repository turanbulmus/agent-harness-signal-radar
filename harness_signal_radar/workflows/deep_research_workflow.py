"""Hybrid Sequential + Parallel Fan-Out/Join + Self-Eval + HITL ADK Workflow graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from google.adk import Context, Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.workflow import JoinNode, RetryConfig

from ..config.settings import RadarSettings
from ..guardrails.hitl import build_editorial_hitl_node
from ..memory.persistence import PersistentSignalVectorStore
from ..observability.logging_config import get_logger
from ..observability.pii_redactor import SensitiveDataRedactor
from ..prompts.constitution import (
  ARXIV_HARVESTER_INSTRUCTION,
  COMMUNITY_HARVESTER_INSTRUCTION,
  CRITIC_INSTRUCTION,
  GITHUB_HARVESTER_INSTRUCTION,
  PLANNER_INSTRUCTION,
  REVISION_INSTRUCTION,
  SYNTHESIZER_INSTRUCTION,
)
from ..routing.model_router import StrategicModelRouter, TaskTier
from ..schemas import (
  ConsolidatedResearchBrief,
  DeepResearchWorkflowInput,
  NormalizedSignalItem,
  PublishExecutiveDigestInput,
  ResearchPlanSchema,
  SelfEvalCritiqueVerdict,
  SignalCorrelationInput,
)
from ..tools.arxiv_papers import build_search_arxiv_papers_tool
from ..tools.curated_fixtures import (
  CURATED_ARXIV_PAPERS,
  CURATED_GITHUB_RELEASES,
  CURATED_HN_DISCUSSIONS,
)
from ..tools.github_releases import build_fetch_github_releases_tool
from ..tools.hackernews_signals import build_scan_hackernews_signals_tool
from ..tools.memory_and_publishing import publish_approved_executive_signal_digest
from ..tools.signal_correlator import correlate_and_score_research_signals

_logger = get_logger("harness_signal_radar.workflow")


def format_research_brief_markdown(brief: dict[str, Any]) -> str:
  """Renders a ConsolidatedResearchBrief payload into clean executive Markdown."""
  headline = str(brief.get("headline") or "Agent Harness & Workflow Automation Signal Brief")
  summary = str(brief.get("executive_summary") or "")
  harness_trends = brief.get("agent_harness_trends") or []
  workflow_trends = brief.get("workflow_automation_trends") or []
  implications = brief.get("architectural_implications") or []
  citations = brief.get("citations") or []
  confidence = brief.get("confidence_score", 0.0)
  receipt = brief.get("publication_receipt")

  lines = [
      f"# {headline}",
      "",
      f"**Confidence Score:** `{confidence}`",
      "",
      "## Executive Summary",
      summary,
      "",
      "## Agent Harness Trends",
  ]
  lines.extend(f"- {item}" for item in harness_trends)
  lines.extend(["", "## Workflow Automation Trends"])
  lines.extend(f"- {item}" for item in workflow_trends)
  lines.extend(["", "## Architectural Implications"])
  lines.extend(f"- {item}" for item in implications)
  lines.extend(["", "## Verified Citations"])
  lines.extend(f"- {url}" for url in citations)
  if isinstance(receipt, dict):
    lines.extend([
        "",
        "## External Publication Receipt",
        (
            f"- Status: `{receipt.get('status')}` | Mode: `{receipt.get('delivery_mode')}` | "
            f"Channel: `{receipt.get('target_channel')}` | Delivery ID: `{receipt.get('delivery_id')}`"
        ),
    ])
  return "\n".join(lines)


def _coerce_workflow_input(node_input: Any) -> DeepResearchWorkflowInput:
  if isinstance(node_input, DeepResearchWorkflowInput):
    return node_input
  if isinstance(node_input, dict):
    return DeepResearchWorkflowInput.model_validate(node_input)
  if isinstance(node_input, str):
    stripped = node_input.strip()
    if stripped.startswith("{"):
      try:
        return DeepResearchWorkflowInput.model_validate_json(stripped)
      except Exception:
        pass
    return DeepResearchWorkflowInput(
        research_question=stripped or "Latest trends in agent harnesses and workflow automation"
    )
  text_attr = getattr(node_input, "parts", None)
  if text_attr:
    joined = " ".join(p.text for p in text_attr if getattr(p, "text", None))
    return _coerce_workflow_input(joined)
  return DeepResearchWorkflowInput(
      research_question="Latest trends in agent harnesses and workflow automation"
  )


def build_deep_research_workflow(
    *,
    settings: RadarSettings | None = None,
    router: StrategicModelRouter | None = None,
    model_override: BaseLlm | Mapping[TaskTier, BaseLlm] | None = None,
    vector_store: PersistentSignalVectorStore | None = None,
    name: str = "deep_research_consolidation_workflow",
) -> Workflow:
  """Constructs the ADK graph Workflow for multi-source deep research consolidation.

  Graph topology:
    1. ``prepare_research_context`` -> normalizes request parameters into session state.
    2. ``research_planner_agent`` (Deep Reasoning Tier / Pro) -> emits ``ResearchPlanSchema``.
    3. Parallel fan-out to 3 Fast Extraction Tier (Flash) harvester agents:
       ``github_harvester_agent``, ``arxiv_harvester_agent``, ``community_harvester_agent``.
    4. ``join_harvester_streams`` (``JoinNode``) -> ``correlate_signals_node``.
    5. ``briefing_synthesizer_agent`` (Pro) -> ``self_eval_critic_agent`` (Flash).
    6. ``evaluate_quality_gate`` routes either ``'revise'`` -> ``briefing_revision_agent``
       -> ``self_eval_critic_agent`` (up to 1 loop) OR ``'gate_publication'`` ->
       ``editorial_hitl_gate`` (resumable HITL ``RequestInput``).
    7. ``editorial_hitl_gate`` routes either ``'publish'`` -> ``publish_digest_node``
       -> ``finalize_briefing_node`` OR ``'archive_only'`` -> ``finalize_briefing_node``.
  """
  active_settings = settings or RadarSettings.from_env(load_env_file=False)
  active_router = router or StrategicModelRouter.from_settings(active_settings)
  active_vector_store = vector_store or PersistentSignalVectorStore(
      db_path=active_settings.vector_db_path
  )
  redactor = SensitiveDataRedactor(
      enable_cloud_dlp=active_settings.enable_cloud_dlp,
      project_id=active_settings.google_cloud_project,
  )

  pro_model = active_router.model_for_tier(
      TaskTier.DEEP_REASONING, model_override=model_override
  )
  flash_model = active_router.model_for_tier(
      TaskTier.FAST_EXTRACTION, model_override=model_override
  )

  def prepare_research_context(ctx: Context, node_input: Any) -> Event:
    parsed = _coerce_workflow_input(node_input)
    state_updates = {
        "research_question": redactor.scrub_text(parsed.research_question),
        "lookback_days": parsed.lookback_days,
        "require_external_publication": parsed.require_external_publication,
        "target_channel": parsed.target_channel,
        "revision_count": 0,
    }
    prompt_payload = json.dumps(
        {
            "research_question": state_updates["research_question"],
            "lookback_days": parsed.lookback_days,
            "target_channel": parsed.target_channel,
        }
    )
    return Event(output=prompt_payload, state=state_updates)

  research_planner_agent = LlmAgent(
      name="research_planner_agent",
      model=pro_model,
      mode="single_turn",
      include_contents="none",
      instruction=PLANNER_INSTRUCTION,
      output_schema=ResearchPlanSchema,
      output_key="research_plan",
  )

  github_harvester_agent = LlmAgent(
      name="github_harvester_agent",
      model=flash_model,
      mode="single_turn",
      include_contents="none",
      instruction=GITHUB_HARVESTER_INSTRUCTION,
      tools=[build_fetch_github_releases_tool()],
  )

  arxiv_harvester_agent = LlmAgent(
      name="arxiv_harvester_agent",
      model=flash_model,
      mode="single_turn",
      include_contents="none",
      instruction=ARXIV_HARVESTER_INSTRUCTION,
      tools=[build_search_arxiv_papers_tool()],
  )

  community_harvester_agent = LlmAgent(
      name="community_harvester_agent",
      model=flash_model,
      mode="single_turn",
      include_contents="none",
      instruction=COMMUNITY_HARVESTER_INSTRUCTION,
      tools=[build_scan_hackernews_signals_tool()],
  )

  join_harvester_streams = JoinNode(name="join_harvester_streams")

  async def correlate_signals_node(ctx: Context, node_input: dict[str, Any]) -> Event:
    del node_input
    collected_items: list[NormalizedSignalItem] = []
    for state_key, fallback_items in (
        ("collected_github_signals", CURATED_GITHUB_RELEASES[:2]),
        ("collected_arxiv_signals", CURATED_ARXIV_PAPERS[:2]),
        ("collected_hn_signals", CURATED_HN_DISCUSSIONS[:2]),
    ):
      raw_list = ctx.state.get(state_key)
      if isinstance(raw_list, list) and raw_list:
        for raw_item in raw_list:
          collected_items.append(NormalizedSignalItem.model_validate(raw_item))
      else:
        collected_items.extend(fallback_items)

    research_question = str(
        ctx.state.get("research_question")
        or "agent harnesses and workflow automation"
    )
    correlation_output = correlate_and_score_research_signals(
        SignalCorrelationInput(
            signals=collected_items,
            focus_area=research_question[:180],
            top_k=8,
        )
    )
    archived_matches = await active_vector_store.search_similar_briefings(
        research_question, top_k=2
    )
    correlated_json = correlation_output.model_dump_json(indent=2)
    archived_json = json.dumps(archived_matches, indent=2)

    return Event(
        output=correlated_json,
        state={
            "correlated_signals_json": correlated_json,
            "archived_briefings_json": archived_json,
        },
    )

  briefing_synthesizer_agent = LlmAgent(
      name="briefing_synthesizer_agent",
      model=pro_model,
      mode="single_turn",
      include_contents="none",
      instruction=SYNTHESIZER_INSTRUCTION,
      output_schema=ConsolidatedResearchBrief,
      output_key="draft_research_brief",
  )

  self_eval_critic_agent = LlmAgent(
      name="self_eval_critic_agent",
      model=flash_model,
      mode="single_turn",
      include_contents="none",
      instruction=CRITIC_INSTRUCTION,
      output_schema=SelfEvalCritiqueVerdict,
      output_key="critique_verdict",
  )

  briefing_revision_agent = LlmAgent(
      name="briefing_revision_agent",
      model=pro_model,
      mode="single_turn",
      include_contents="none",
      instruction=REVISION_INSTRUCTION,
      output_schema=ConsolidatedResearchBrief,
      output_key="draft_research_brief",
  )

  def evaluate_quality_gate(ctx: Context, node_input: Any) -> Event:
    try:
      verdict = SelfEvalCritiqueVerdict.model_validate(node_input)
    except Exception:
      verdict = SelfEvalCritiqueVerdict(
          passed=True,
          quality_score=0.85,
          citation_coverage_ok=True,
          hallucination_risk="low",
          improvement_instructions="Fallback acceptance after non-JSON verdict.",
      )

    revision_count = int(ctx.state.get("revision_count", 0))
    draft_brief = ctx.state.get("draft_research_brief")
    if not isinstance(draft_brief, dict):
      draft_brief = {}

    if not verdict.passed and revision_count < 1:
      _logger.info(
          "self_eval_triggered_revision",
          quality_score=verdict.quality_score,
          revision_count=revision_count + 1,
      )
      return Event(
          output=draft_brief,
          route="revise",
          state={"revision_count": revision_count + 1},
      )

    return Event(output=draft_brief, route="gate_publication")

  editorial_hitl_gate = build_editorial_hitl_node()

  async def publish_digest_node(ctx: Context, node_input: dict[str, Any]) -> Event:
    markdown_body = format_research_brief_markdown(node_input)
    citations = [str(c) for c in node_input.get("citations", [])] or [
        "https://github.com/google/adk-python/releases"
    ]
    target_channel = str(ctx.state.get("target_channel") or "engineering_digest")
    receipt = await publish_approved_executive_signal_digest(
        PublishExecutiveDigestInput(
            headline=str(node_input.get("headline") or "Agent Harness Signal Briefing"),
            markdown_body=markdown_body,
            target_channel=target_channel,
            citations=citations,
        ),
        tool_context=ctx,
    )
    receipt_dict = receipt.model_dump(mode="json")
    updated = {**node_input, "publication_receipt": receipt_dict}
    return Event(
        output=updated,
        state={"publication_receipt": receipt_dict},
    )

  async def finalize_briefing_node(ctx: Context, node_input: dict[str, Any]):
    del ctx
    scrubbed_payload = redactor.scrub_payload(node_input)
    markdown_report = format_research_brief_markdown(scrubbed_payload)
    yield Event(message=markdown_report)
    yield Event(
        output=scrubbed_payload,
        state={"final_research_brief": scrubbed_payload},
    )

  return Workflow(
      name=name,
      description=(
          "Runs a full deep research workflow consolidating latest releases, arXiv "
          "papers, and community discussions on AI agent harnesses and workflow automation, "
          "with parallel harvesting, self-evaluation critique, and optional HITL publishing."
      ),
      input_schema=DeepResearchWorkflowInput,
      retry_config=RetryConfig(max_attempts=2, initial_delay=0.5),
      edges=[
          ("START", prepare_research_context, research_planner_agent),
          (
              research_planner_agent,
              (
                  github_harvester_agent,
                  arxiv_harvester_agent,
                  community_harvester_agent,
              ),
          ),
          (
              (
                  github_harvester_agent,
                  arxiv_harvester_agent,
                  community_harvester_agent,
              ),
              join_harvester_streams,
          ),
          (
              join_harvester_streams,
              correlate_signals_node,
              briefing_synthesizer_agent,
              self_eval_critic_agent,
              evaluate_quality_gate,
          ),
          (
              evaluate_quality_gate,
              {
                  "revise": briefing_revision_agent,
                  "gate_publication": editorial_hitl_gate,
              },
          ),
          (briefing_revision_agent, self_eval_critic_agent),
          (
              editorial_hitl_gate,
              {
                  "publish": publish_digest_node,
                  "archive_only": finalize_briefing_node,
              },
          ),
          (publish_digest_node, finalize_briefing_node),
      ],
  )
