"""Strict Pydantic v2 schemas for tools, workflow nodes, and structured LLM outputs."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SignalSourceCategory = Literal[
    "github_release",
    "arxiv_paper",
    "hackernews_discussion",
    "archived_briefing",
]

ToolExecutionStatus = Literal["success", "partial", "error", "blocked"]

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GuidedRecoveryAdvice(BaseModel):
  """Actionable recovery instructions returned to the LLM when a tool call fails."""

  model_config = ConfigDict(extra="forbid")

  error_code: str = Field(
      description="Machine-readable error classification (e.g. INVALID_ARGUMENTS, RATE_LIMITED)."
  )
  error_message: str = Field(
      description="Human-readable summary of what failed during validation or execution."
  )
  recovery_instructions: str = Field(
      description="Step-by-step instructions explaining how the LLM should fix arguments or recover."
  )
  suggested_parameters: dict[str, Any] = Field(
      default_factory=dict,
      description="Example valid parameters that satisfy the tool contract.",
  )
  retryable: bool = Field(
      default=True,
      description="Whether retrying the tool with corrected parameters or backoff can succeed.",
  )


class NormalizedSignalItem(BaseModel):
  """Canonical cross-source technical signal item for agent harnesses & workflow automation."""

  model_config = ConfigDict(extra="forbid")

  signal_id: str = Field(description="Stable unique identifier for deduplication.")
  title: str = Field(description="Concise headline or release title.")
  canonical_url: str = Field(description="Authoritative public source URL.")
  source_category: SignalSourceCategory = Field(
      description="Origin stream classification."
  )
  published_at: str = Field(description="ISO-8601 publication timestamp or date.")
  summary: str = Field(description="Technical summary of the harness/workflow capability.")
  authority_score: float = Field(
      ge=0.0,
      le=1.0,
      description="Normalized source credibility weight in [0.0, 1.0].",
  )
  engagement_velocity: float = Field(
      ge=0.0,
      le=1.0,
      description="Normalized adoption/discussion momentum score in [0.0, 1.0].",
  )
  harness_or_workflow_tags: list[str] = Field(
      default_factory=list,
      description="Taxonomy tags such as 'graph_orchestration', 'mcp_sandbox', 'compaction'.",
  )


class GitHubReleaseQueryInput(BaseModel):
  """Strict input schema for fetching GitHub releases of agent harness repositories."""

  model_config = ConfigDict(extra="forbid")

  repositories: list[str] = Field(
      min_length=1,
      max_length=10,
      examples=[["google/adk-python", "langchain-ai/langgraph"]],
      description="Repositories in strict 'owner/repo' format (e.g. ['google/adk-python']).",
  )
  lookback_days: int = Field(
      default=30,
      ge=1,
      le=365,
      description="Number of days back to inspect release notes.",
  )
  include_prereleases: bool = Field(
      default=False,
      description="Whether to include alpha/beta/release-candidate builds.",
  )

  @field_validator("repositories")
  @classmethod
  def _validate_repo_slugs(cls, repos: list[str]) -> list[str]:
    cleaned: list[str] = []
    for repo in repos:
      stripped = repo.strip()
      if not _REPO_PATTERN.match(stripped):
        raise ValueError(
            f"Repository '{repo}' must follow 'owner/repo' format, e.g. 'google/adk-python'."
        )
      cleaned.append(stripped)
    return cleaned


class GitHubReleaseQueryOutput(BaseModel):
  """Strict output schema returned by fetch_github_agent_framework_releases."""

  model_config = ConfigDict(extra="forbid")

  status: ToolExecutionStatus = Field(description="Execution outcome status.")
  source_mode: Literal["live_api", "curated_fixture", "error"] = Field(
      default="curated_fixture",
      description="Indicates whether data came from live GitHub API or curated fallback.",
  )
  releases: list[NormalizedSignalItem] = Field(
      default_factory=list,
      description="Normalized GitHub release signals.",
  )
  warnings: list[str] = Field(
      default_factory=list,
      description="Non-fatal warnings such as rate-limit fallbacks for specific repos.",
  )
  recovery_advice: GuidedRecoveryAdvice | None = Field(
      default=None,
      description="Present when status is 'error' or 'partial' to guide LLM recovery.",
  )


class ArxivPaperQueryInput(BaseModel):
  """Strict input schema for querying arXiv papers on agent harnesses & workflows."""

  model_config = ConfigDict(extra="forbid")

  search_query: str = Field(
      min_length=3,
      max_length=300,
      examples=["agent harness workflow orchestration"],
      description="Keyword or phrase query covering agent harnesses, DAG orchestration, or evals.",
  )
  max_results: int = Field(
      default=5,
      ge=1,
      le=20,
      description="Maximum number of arXiv abstracts to retrieve.",
  )
  categories: list[str] = Field(
      default_factory=lambda: ["cs.AI", "cs.SE", "cs.MA", "cs.DC"],
      min_length=1,
      max_length=8,
      description="arXiv subject category codes to scope the search.",
  )


class ArxivPaperQueryOutput(BaseModel):
  """Strict output schema returned by search_arxiv_agent_orchestration_papers."""

  model_config = ConfigDict(extra="forbid")

  status: ToolExecutionStatus = Field(description="Execution outcome status.")
  source_mode: Literal["live_api", "curated_fixture", "error"] = Field(
      default="curated_fixture",
      description="Indicates live arXiv Atom API vs curated fallback.",
  )
  papers: list[NormalizedSignalItem] = Field(
      default_factory=list,
      description="Normalized arXiv paper signals.",
  )
  warnings: list[str] = Field(default_factory=list)
  recovery_advice: GuidedRecoveryAdvice | None = Field(default=None)


class HackerNewsSignalQueryInput(BaseModel):
  """Strict input schema for scanning community practitioner discussions."""

  model_config = ConfigDict(extra="forbid")

  keywords: list[str] = Field(
      min_length=1,
      max_length=8,
      examples=[["agent harness", "workflow automation"]],
      description="Keywords for practitioner discussions (e.g. ['agent harness', 'workflow automation']).",
  )
  min_points: int = Field(
      default=25,
      ge=0,
      le=5000,
      description="Minimum community upvotes required to include a story.",
  )
  lookback_days: int = Field(
      default=30,
      ge=1,
      le=180,
      description="Number of days back to filter community threads.",
  )


class HackerNewsSignalQueryOutput(BaseModel):
  """Strict output schema returned by scan_hackernews_workflow_automation_signals."""

  model_config = ConfigDict(extra="forbid")

  status: ToolExecutionStatus = Field(description="Execution outcome status.")
  source_mode: Literal["live_api", "curated_fixture", "error"] = Field(
      default="curated_fixture",
      description="Indicates live Algolia HN API vs curated fallback.",
  )
  discussions: list[NormalizedSignalItem] = Field(
      default_factory=list,
      description="Normalized community discussion signals.",
  )
  warnings: list[str] = Field(default_factory=list)
  recovery_advice: GuidedRecoveryAdvice | None = Field(default=None)


class SignalCorrelationInput(BaseModel):
  """Strict input schema for deduplicating, scoring, and clustering collected signals."""

  model_config = ConfigDict(extra="forbid")

  signals: list[NormalizedSignalItem] = Field(
      min_length=1,
      max_length=100,
      description="Non-empty list of normalized signal items from harvester streams.",
  )
  focus_area: str = Field(
      default="agent_harness_and_workflow_automation",
      min_length=3,
      max_length=200,
      description="Topic lens used to boost relevance during correlation.",
  )
  top_k: int = Field(
      default=8,
      ge=1,
      le=25,
      description="Maximum number of top-ranked signals to retain.",
  )


class SignalCorrelationOutput(BaseModel):
  """Strict output schema returned by correlate_and_score_research_signals."""

  model_config = ConfigDict(extra="forbid")

  status: ToolExecutionStatus = Field(description="Execution outcome status.")
  ranked_signals: list[NormalizedSignalItem] = Field(default_factory=list)
  theme_clusters: dict[str, list[str]] = Field(default_factory=dict)
  convergence_highlights: list[str] = Field(default_factory=list)
  deduplicated_count: int = Field(default=0, ge=0)
  recovery_advice: GuidedRecoveryAdvice | None = Field(default=None)


class ArchivedBriefingQueryInput(BaseModel):
  """Strict input schema for querying persistent briefing memory."""

  model_config = ConfigDict(extra="forbid")

  query: str = Field(
      min_length=3,
      max_length=300,
      examples=["ADK workflow compaction"],
      description="Semantic query for retrieving archived research briefings.",
  )
  top_k: int = Field(
      default=3,
      ge=1,
      le=10,
      description="Maximum number of historical briefings to retrieve.",
  )


class ArchivedBriefingQueryOutput(BaseModel):
  """Strict output schema returned by retrieve_archived_signal_briefings."""

  model_config = ConfigDict(extra="forbid")

  status: ToolExecutionStatus = Field(description="Execution outcome status.")
  matches: list[dict[str, Any]] = Field(default_factory=list)
  recovery_advice: GuidedRecoveryAdvice | None = Field(default=None)


class EditorialApprovalRequestInput(BaseModel):
  """Input schema for requesting human editorial confirmation before external publication."""

  model_config = ConfigDict(extra="forbid")

  brief_headline: str = Field(min_length=5, max_length=200)
  target_channel: str = Field(min_length=2, max_length=80)
  citation_count: int = Field(ge=0)
  executive_preview: str = Field(min_length=10, max_length=1500)


class EditorialApprovalDecision(BaseModel):
  """Structured human-in-the-loop response payload for publication approval."""

  model_config = ConfigDict(extra="forbid")

  approved: bool = Field(description="True to authorize external digest publication.")
  reviewer_note: str = Field(
      default="",
      description="Optional human reviewer rationale or revision note.",
  )
  approved_channel: str = Field(
      default="engineering_digest",
      description="Authorized destination channel for the digest.",
  )


class PublishExecutiveDigestInput(BaseModel):
  """Strict input schema for publishing an approved executive signal digest."""

  model_config = ConfigDict(extra="forbid")

  headline: str = Field(min_length=5, max_length=200)
  markdown_body: str = Field(min_length=30)
  target_channel: str = Field(min_length=2, max_length=80)
  citations: list[str] = Field(min_length=1)


class PublishExecutiveDigestOutput(BaseModel):
  """Strict output schema returned by publish_approved_executive_signal_digest."""

  model_config = ConfigDict(extra="forbid")

  status: ToolExecutionStatus = Field(description="Execution outcome status.")
  delivery_id: str | None = Field(default=None)
  target_channel: str = Field(default="")
  published_at: str | None = Field(default=None)
  delivery_mode: Literal["webhook", "dry_run_archive", "blocked", "error"] = Field(
      default="dry_run_archive"
  )
  recovery_advice: GuidedRecoveryAdvice | None = Field(default=None)


class DeepResearchWorkflowInput(BaseModel):
  """Typed input schema for the DeepResearchConsolidationWorkflow node/tool."""

  model_config = ConfigDict(extra="forbid")

  research_question: str = Field(
      min_length=5,
      max_length=500,
      description="Deep research objective on agent harnesses and workflow automation.",
  )
  lookback_days: int = Field(
      default=30,
      ge=1,
      le=365,
      description="Window in days for collecting release, paper, and community signals.",
  )
  require_external_publication: bool = Field(
      default=False,
      description="When True, pauses at the Human-in-the-Loop gate prior to publishing.",
  )
  target_channel: str = Field(
      default="engineering_digest",
      min_length=2,
      max_length=80,
      description="Destination channel if external publication is approved.",
  )


class ResearchPlanSchema(BaseModel):
  """Structured research plan emitted by ResearchPlannerAgent."""

  model_config = ConfigDict(extra="forbid")

  core_thesis_question: str = Field(
      description="Focused synthesis question guiding downstream harvesters."
  )
  github_repositories: list[str] = Field(
      min_length=1,
      description="Target 'owner/repo' slugs to inspect.",
  )
  arxiv_search_queries: list[str] = Field(
      min_length=1,
      description="Queries for scholarly agent harness and orchestration papers.",
  )
  community_keywords: list[str] = Field(
      min_length=1,
      description="Practitioner discussion terms for community trend scanning.",
  )
  priority_themes: list[str] = Field(
      min_length=1,
      description="High-value architectural themes to prioritize.",
  )


class ConsolidatedResearchBrief(BaseModel):
  """Structured executive research brief produced by BriefingSynthesizerAgent."""

  model_config = ConfigDict(extra="forbid")

  headline: str = Field(description="Concise intelligence briefing headline.")
  executive_summary: str = Field(
      description="High-density executive synthesis anchored in verified citations."
  )
  agent_harness_trends: list[str] = Field(
      min_length=1,
      description="Verified shifts in agent harnesses (tool schemas, sandboxing, evals, memory).",
  )
  workflow_automation_trends: list[str] = Field(
      min_length=1,
      description="Verified shifts in workflow automation (DAG orchestration, durable state, HITL).",
  )
  architectural_implications: list[str] = Field(
      min_length=1,
      description="Concrete engineering recommendations derived from the consolidated signals.",
  )
  citations: list[str] = Field(
      min_length=1,
      description="Canonical URLs backing the synthesized claims.",
  )
  confidence_score: float = Field(
      ge=0.0,
      le=1.0,
      description="Evidence-backed confidence score in [0.0, 1.0].",
  )


class SelfEvalCritiqueVerdict(BaseModel):
  """Quality & grounding verdict produced by SelfEvalCriticAgent."""

  model_config = ConfigDict(extra="forbid")

  passed: bool = Field(
      description="True if citations, schema completeness, and analytical rigor pass."
  )
  quality_score: float = Field(ge=0.0, le=1.0)
  citation_coverage_ok: bool
  hallucination_risk: Literal["low", "medium", "high"]
  improvement_instructions: str = Field(
      description="Specific revision guidance if passed=False, or confirmation summary."
  )
