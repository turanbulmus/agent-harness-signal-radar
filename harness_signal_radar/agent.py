"""ADK root_agent and App definition for Agent Harness & Workflow Automation Signal Radar."""

from __future__ import annotations

from collections.abc import Mapping

from google.adk import Agent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin

from .config.settings import RadarSettings
from .guardrails.policy_plugin import SecurityAndPolicyGuardrailPlugin
from .memory.async_consolidator import AsyncMemoryConsolidator, AsyncMemoryLifecyclePlugin
from .memory.compaction import (
  TokenBudgetCompactionPlugin,
  build_context_cache_config,
  build_events_compaction_config,
)
from .memory.persistence import PersistentSignalVectorStore
from .observability.intent_outcome import IntentOutcomeAuditPlugin
from .observability.logging_config import configure_structured_logging
from .observability.pii_redactor import SensitiveDataRedactor
from .observability.tracing import ensure_tracer_provider
from .prompts.constitution import (
  ARCHIVE_ANALYST_SUBAGENT_INSTRUCTION,
  COORDINATOR_INSTRUCTION,
  TRIAGE_SUBAGENT_INSTRUCTION,
)
from .routing.model_router import DynamicModelRoutingPlugin, StrategicModelRouter, TaskTier
from .tools import (
  build_correlate_signals_tool,
  build_editorial_approval_tool,
  build_fetch_github_releases_tool,
  build_publish_digest_tool,
  build_retrieve_archived_briefings_tool,
  build_scan_hackernews_signals_tool,
  build_search_arxiv_papers_tool,
)
from .workflows import build_deep_research_workflow


def build_root_agent(
    *,
    settings: RadarSettings | None = None,
    model_override: BaseLlm | Mapping[TaskTier, BaseLlm] | None = None,
    vector_store: PersistentSignalVectorStore | None = None,
) -> Agent:
  """Constructs the Coordinator + Sub-Agents + Workflow Tool multi-agent hierarchy."""
  active_settings = settings or RadarSettings.from_env(load_env_file=False)
  router = StrategicModelRouter.from_settings(active_settings)

  pro_model = router.model_for_tier(TaskTier.DEEP_REASONING, model_override=model_override)
  flash_model = router.model_for_tier(TaskTier.FAST_EXTRACTION, model_override=model_override)

  fetch_github_tool = build_fetch_github_releases_tool()
  search_arxiv_tool = build_search_arxiv_papers_tool()
  scan_hn_tool = build_scan_hackernews_signals_tool()
  correlate_tool = build_correlate_signals_tool()
  archive_tool = build_retrieve_archived_briefings_tool()
  approval_tool = build_editorial_approval_tool()
  publish_tool = build_publish_digest_tool()

  deep_research_workflow = build_deep_research_workflow(
      settings=active_settings,
      router=router,
      model_override=model_override,
      vector_store=vector_store,
  )

  quick_signal_triage_agent = Agent(
      name="QuickSignalTriageAgent",
      model=flash_model,
      description=(
          "Fast-tier specialist for quick single-source lookups on GitHub releases, "
          "arXiv papers, or Hacker News threads."
      ),
      instruction=TRIAGE_SUBAGENT_INSTRUCTION,
      tools=[fetch_github_tool, search_arxiv_tool, scan_hn_tool],
  )

  historical_memory_analyst_agent = Agent(
      name="HistoricalMemoryAnalystAgent",
      model=flash_model,
      description=(
          "Archive specialist that retrieves and compares historical signal briefings "
          "from persistent vector memory."
      ),
      instruction=ARCHIVE_ANALYST_SUBAGENT_INSTRUCTION,
      tools=[archive_tool],
  )

  return Agent(
      name="ResearchCoordinatorAgent",
      model=pro_model,
      description=(
          "Principal deep research coordinator consolidating signals on AI agent "
          "harnesses and workflow automation across GitHub, arXiv, and engineering "
          "communities with HITL publication governance."
      ),
      instruction=COORDINATOR_INSTRUCTION,
      sub_agents=[
          quick_signal_triage_agent,
          historical_memory_analyst_agent,
      ],
      tools=[
          deep_research_workflow,
          fetch_github_tool,
          search_arxiv_tool,
          scan_hn_tool,
          correlate_tool,
          archive_tool,
          approval_tool,
          publish_tool,
      ],
  )


def build_radar_app(
    *,
    settings: RadarSettings | None = None,
    model_override: BaseLlm | Mapping[TaskTier, BaseLlm] | None = None,
    consolidator: AsyncMemoryConsolidator | None = None,
    vector_store: PersistentSignalVectorStore | None = None,
) -> App:
  """Constructs the production ADK App with compaction, caching, plugins, and resumability."""
  active_settings = settings or RadarSettings.from_env(load_env_file=False)
  redactor = SensitiveDataRedactor(
      enable_cloud_dlp=active_settings.enable_cloud_dlp,
      project_id=active_settings.google_cloud_project,
  )
  configure_structured_logging(level=active_settings.log_level, redactor=redactor)
  ensure_tracer_provider(
      service_name=active_settings.app_name,
      enable_cloud_trace=active_settings.enable_cloud_trace,
      project_id=active_settings.google_cloud_project,
  )

  active_vector_store = vector_store or PersistentSignalVectorStore(
      db_path=active_settings.vector_db_path,
      redactor=redactor,
  )
  active_consolidator = consolidator or AsyncMemoryConsolidator(
      vector_store=active_vector_store,
      redactor=redactor,
  )
  router = StrategicModelRouter.from_settings(active_settings)

  coordinator = build_root_agent(
      settings=active_settings,
      model_override=model_override,
      vector_store=active_vector_store,
  )
  compaction_llm = (
      model_override[TaskTier.FAST_EXTRACTION]
      if isinstance(model_override, Mapping) and TaskTier.FAST_EXTRACTION in model_override
      else (model_override if isinstance(model_override, BaseLlm) else None)
  )

  return App(
      name=active_settings.app_name,
      root_agent=coordinator,
      plugins=[
          SecurityAndPolicyGuardrailPlugin(
              redactor=redactor,
              model_armor_template_id=active_settings.model_armor_template_id,
          ),
          DynamicModelRoutingPlugin(router=router),
          TokenBudgetCompactionPlugin(
              max_prompt_tokens=active_settings.max_prompt_tokens,
              redactor=redactor,
          ),
          ContextFilterPlugin(num_invocations_to_keep=8),
          ReflectAndRetryToolPlugin(
              max_retries=2,
              throw_exception_if_retry_exceeded=False,
          ),
          IntentOutcomeAuditPlugin(redactor=redactor),
          AutoTracingPlugin(extra_scope_prefixes=("harness_signal_radar",)),
          AsyncMemoryLifecyclePlugin(consolidator=active_consolidator),
      ],
      events_compaction_config=build_events_compaction_config(llm=compaction_llm),
      context_cache_config=build_context_cache_config(),
      resumability_config=ResumabilityConfig(is_resumable=True),
  )


default_settings = RadarSettings.from_env(load_env_file=False)
default_consolidator = AsyncMemoryConsolidator(
    vector_store=PersistentSignalVectorStore(db_path=default_settings.vector_db_path)
)
app = build_radar_app(settings=default_settings, consolidator=default_consolidator)
root_agent = app.root_agent
