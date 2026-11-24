"""Hacker News engineering practitioner signal harvester."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from google.adk.tools.tool_context import ToolContext

from ..config.settings import RadarSettings
from ..observability.logging_config import get_logger
from ..schemas import (
  GuidedRecoveryAdvice,
  HackerNewsSignalQueryInput,
  HackerNewsSignalQueryOutput,
  NormalizedSignalItem,
)
from .curated_fixtures import CURATED_HN_DISCUSSIONS
from .strict_tool import StrictSchemaFunctionTool

_logger = get_logger("harness_signal_radar.tools.hackernews")


def _filter_curated_discussions(keywords: list[str]) -> list[NormalizedSignalItem]:
  lowered = [kw.lower() for kw in keywords]
  matched: list[NormalizedSignalItem] = []
  for item in CURATED_HN_DISCUSSIONS:
    haystack = f"{item.title} {item.summary} {' '.join(item.harness_or_workflow_tags)}".lower()
    if any(kw in haystack for kw in lowered):
      matched.append(item)
  return matched or list(CURATED_HN_DISCUSSIONS)


async def scan_hackernews_workflow_automation_signals(
    params: HackerNewsSignalQueryInput,
    tool_context: ToolContext | None = None,
) -> HackerNewsSignalQueryOutput:
  """Scan Hacker News discussions for production engineering lessons on agent harnesses.

  Queries the Algolia HN Search API filtered by minimum point threshold and recency
  window, surfacing architectural debates, postmortems, and MCP/DAG adoption trends.
  Degrades gracefully to curated engineering discussions when running offline.

  Args:
    params: Validated ``HackerNewsSignalQueryInput`` specifying ``keywords``,
      ``min_points``, and ``lookback_days``.
    tool_context: Optional ADK tool context for recording collected signals in state.

  Returns:
    A strict ``HackerNewsSignalQueryOutput`` containing normalized discussion signals.
  """
  settings = RadarSettings.from_env(load_env_file=False)
  force_live = os.environ.get("RADAR_FORCE_LIVE_HTTP", "").lower() == "true"

  if settings.offline_fixture_fallback and not force_live:
    discussions = _filter_curated_discussions(params.keywords)
    if tool_context is not None:
      tool_context.state["collected_hn_signals"] = [
          item.model_dump(mode="json") for item in discussions
      ]
    return HackerNewsSignalQueryOutput(
        status="success",
        source_mode="curated_fixture",
        discussions=discussions,
        warnings=[
            "Using curated community signal baseline (set RADAR_FORCE_LIVE_HTTP=true for live HN Algolia API)."
        ],
    )

  cutoff_ts = int(
      (datetime.now(tz=UTC) - timedelta(days=params.lookback_days)).timestamp()
  )
  query_str = " OR ".join(params.keywords)
  api_params = {
      "query": query_str,
      "tags": "story",
      "numericFilters": f"points>={params.min_points},created_at_i>={cutoff_ts}",
      "hitsPerPage": 10,
  }

  try:
    async with httpx.AsyncClient(timeout=8.0) as client:
      response = await client.get(
          "https://hn.algolia.com/api/v1/search", params=api_params
      )
      response.raise_for_status()
    hits: list[dict[str, Any]] = response.json().get("hits", [])
    discussions: list[NormalizedSignalItem] = []
    for hit in hits:
      object_id = str(hit.get("objectID") or "")
      title = str(hit.get("title") or "").strip()
      if not object_id or not title:
        continue
      points = int(hit.get("points") or 0)
      comments = int(hit.get("num_comments") or 0)
      url = str(hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}")
      created_at = str(
          hit.get("created_at") or datetime.now(tz=UTC).isoformat()
      )
      velocity = min(1.0, round((points + comments * 1.5) / 400.0, 3))
      discussions.append(
          NormalizedSignalItem(
              signal_id=f"hn:{object_id}",
              title=title,
              canonical_url=url,
              source_category="hackernews_discussion",
              published_at=created_at,
              summary=(
                  f"Community engineering thread ({points} points, {comments} comments) "
                  f"discussing {', '.join(params.keywords[:3])}."
              ),
              authority_score=0.80,
              engagement_velocity=velocity,
              harness_or_workflow_tags=["community_signal", "workflow_automation"],
          )
      )
    if not discussions and settings.offline_fixture_fallback:
      fallback = _filter_curated_discussions(params.keywords)
      if tool_context is not None:
        tool_context.state["collected_hn_signals"] = [
            item.model_dump(mode="json") for item in fallback
        ]
      return HackerNewsSignalQueryOutput(
          status="partial",
          source_mode="curated_fixture",
          discussions=fallback,
          warnings=["Live HN search returned 0 stories above min_points; served baseline."],
      )
    if tool_context is not None:
      tool_context.state["collected_hn_signals"] = [
          item.model_dump(mode="json") for item in discussions
      ]
    return HackerNewsSignalQueryOutput(
        status="success",
        source_mode="live_api",
        discussions=discussions,
    )
  except Exception as exc:
    _logger.warning("hn_scan_error", error=str(exc))
    if settings.offline_fixture_fallback:
      fallback = _filter_curated_discussions(params.keywords)
      if tool_context is not None:
        tool_context.state["collected_hn_signals"] = [
            item.model_dump(mode="json") for item in fallback
        ]
      return HackerNewsSignalQueryOutput(
          status="partial",
          source_mode="curated_fixture",
          discussions=fallback,
          warnings=[f"HN API unavailable ({type(exc).__name__}); used curated baseline."],
      )
    return HackerNewsSignalQueryOutput(
        status="error",
        source_mode="error",
        discussions=[],
        warnings=[str(exc)],
        recovery_advice=GuidedRecoveryAdvice(
            error_code="HN_SCAN_FAILED",
            error_message=f"Community signal scan failed: {type(exc).__name__}",
            recovery_instructions=(
                "Lower min_points (e.g. 10) or broaden keywords and retry the scan."
            ),
            suggested_parameters={
                "keywords": ["agent harness", "workflow automation"],
                "min_points": 10,
                "lookback_days": 30,
            },
            retryable=True,
        ),
    )


def build_scan_hackernews_signals_tool() -> StrictSchemaFunctionTool:
  """Builds the strict-schema Hacker News community signal scanner tool."""
  return StrictSchemaFunctionTool(
      func=scan_hackernews_workflow_automation_signals,
      input_schema=HackerNewsSignalQueryInput,
      output_schema=HackerNewsSignalQueryOutput,
  )
