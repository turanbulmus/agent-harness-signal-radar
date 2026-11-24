"""arXiv scholarly paper harvester for agent orchestration and harness engineering."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import httpx
from google.adk.tools.tool_context import ToolContext

from ..config.settings import RadarSettings
from ..observability.logging_config import get_logger
from ..schemas import (
  ArxivPaperQueryInput,
  ArxivPaperQueryOutput,
  GuidedRecoveryAdvice,
  NormalizedSignalItem,
)
from .curated_fixtures import CURATED_ARXIV_PAPERS
from .strict_tool import StrictSchemaFunctionTool

_logger = get_logger("harness_signal_radar.tools.arxiv")
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _filter_curated_papers(query: str, max_results: int) -> list[NormalizedSignalItem]:
  terms = [t.strip().lower() for t in query.split() if len(t.strip()) >= 3]
  scored: list[tuple[int, NormalizedSignalItem]] = []
  for paper in CURATED_ARXIV_PAPERS:
    haystack = f"{paper.title} {paper.summary} {' '.join(paper.harness_or_workflow_tags)}".lower()
    hits = sum(1 for term in terms if term in haystack)
    scored.append((hits, paper))
  scored.sort(key=lambda pair: pair[0], reverse=True)
  return [paper for _, paper in scored[:max_results]]


async def search_arxiv_agent_orchestration_papers(
    params: ArxivPaperQueryInput,
    tool_context: ToolContext | None = None,
) -> ArxivPaperQueryOutput:
  """Search arXiv research papers covering agent harnesses, memory compaction, and DAG workflows.

  Queries the arXiv Atom search endpoint across specified computer science categories
  (default ``cs.AI``, ``cs.SE``, ``cs.MA``, ``cs.DC``) and returns normalized paper
  metadata with citations. Falls back to curated benchmark papers in offline mode or
  if arXiv throttles requests.

  Args:
    params: Validated ``ArxivPaperQueryInput`` specifying ``search_query``,
      ``max_results``, and ``categories``.
    tool_context: Optional ADK tool context for recording collected signals in state.

  Returns:
    A strict ``ArxivPaperQueryOutput`` containing normalized paper items and any
    warnings or guided recovery advice on failure.
  """
  settings = RadarSettings.from_env(load_env_file=False)
  force_live = os.environ.get("RADAR_FORCE_LIVE_HTTP", "").lower() == "true"

  if settings.offline_fixture_fallback and not force_live:
    papers = _filter_curated_papers(params.search_query, params.max_results)
    if tool_context is not None:
      tool_context.state["collected_arxiv_signals"] = [
          item.model_dump(mode="json") for item in papers
      ]
    return ArxivPaperQueryOutput(
        status="success",
        source_mode="curated_fixture",
        papers=papers,
        warnings=[
            "Using curated arXiv research baseline (set RADAR_FORCE_LIVE_HTTP=true for live Atom queries)."
        ],
    )

  cat_filter = " OR ".join(f"cat:{cat}" for cat in params.categories)
  full_query = f"({cat_filter}) AND all:{params.search_query}"
  query_params = {
      "search_query": full_query,
      "start": 0,
      "max_results": params.max_results,
      "sortBy": "submittedDate",
      "sortOrder": "descending",
  }

  try:
    async with httpx.AsyncClient(timeout=10.0) as client:
      response = await client.get(
          "https://export.arxiv.org/api/query", params=query_params
      )
      response.raise_for_status()
    root = ET.fromstring(response.text)
    papers: list[NormalizedSignalItem] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
      raw_id = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
      short_id = raw_id.rsplit("/", 1)[-1] or raw_id
      title = " ".join(
          (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").split()
      )
      summary = " ".join(
          (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").split()
      )[:360]
      published = (
          entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
      ).strip()
      if title and raw_id:
        papers.append(
            NormalizedSignalItem(
                signal_id=f"arxiv:{short_id}",
                title=title,
                canonical_url=raw_id,
                source_category="arxiv_paper",
                published_at=published,
                summary=summary,
                authority_score=0.90,
                engagement_velocity=0.72,
                harness_or_workflow_tags=["agent_harness", "research_paper"],
            )
        )
    if not papers and settings.offline_fixture_fallback:
      fallback = _filter_curated_papers(params.search_query, params.max_results)
      if tool_context is not None:
        tool_context.state["collected_arxiv_signals"] = [
            item.model_dump(mode="json") for item in fallback
        ]
      return ArxivPaperQueryOutput(
          status="partial",
          source_mode="curated_fixture",
          papers=fallback,
          warnings=["Live arXiv query returned zero hits; returned curated baseline papers."],
      )
    if tool_context is not None:
      tool_context.state["collected_arxiv_signals"] = [
          item.model_dump(mode="json") for item in papers
      ]
    return ArxivPaperQueryOutput(
        status="success",
        source_mode="live_api",
        papers=papers,
    )
  except Exception as exc:
    _logger.warning("arxiv_search_error", error=str(exc))
    if settings.offline_fixture_fallback:
      fallback = _filter_curated_papers(params.search_query, params.max_results)
      if tool_context is not None:
        tool_context.state["collected_arxiv_signals"] = [
            item.model_dump(mode="json") for item in fallback
        ]
      return ArxivPaperQueryOutput(
          status="partial",
          source_mode="curated_fixture",
          papers=fallback,
          warnings=[f"arXiv API unavailable ({type(exc).__name__}); used curated baseline."],
      )
    return ArxivPaperQueryOutput(
        status="error",
        source_mode="error",
        papers=[],
        warnings=[str(exc)],
        recovery_advice=GuidedRecoveryAdvice(
            error_code="ARXIV_QUERY_FAILED",
            error_message=f"arXiv search failed: {type(exc).__name__}",
            recovery_instructions=(
                "Simplify search_query terms, verify cs.* category codes, or retry with "
                "max_results <= 5."
            ),
            suggested_parameters={
                "search_query": "agent workflow orchestration",
                "max_results": 3,
                "categories": ["cs.AI", "cs.SE"],
            },
            retryable=True,
        ),
    )


def build_search_arxiv_papers_tool() -> StrictSchemaFunctionTool:
  """Builds the strict-schema arXiv scholarly signal harvester tool."""
  return StrictSchemaFunctionTool(
      func=search_arxiv_agent_orchestration_papers,
      input_schema=ArxivPaperQueryInput,
      output_schema=ArxivPaperQueryOutput,
  )
