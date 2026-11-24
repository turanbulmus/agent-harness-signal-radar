"""GitHub release harvester tool for AI agent harness & workflow repositories."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from google.adk.tools.tool_context import ToolContext

from ..config.secrets import SecretManagerResolver, resolve_optional_secret
from ..config.settings import RadarSettings
from ..observability.logging_config import get_logger
from ..schemas import (
  GitHubReleaseQueryInput,
  GitHubReleaseQueryOutput,
  GuidedRecoveryAdvice,
  NormalizedSignalItem,
)
from .curated_fixtures import CURATED_GITHUB_RELEASES
from .strict_tool import StrictSchemaFunctionTool

_logger = get_logger("harness_signal_radar.tools.github")


def _parse_iso_dt(raw: str) -> datetime | None:
  try:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
  except ValueError:
    return None


def _infer_tags(text: str) -> list[str]:
  lowered = text.lower()
  tags: list[str] = ["agent_harness"]
  tag_keywords = {
      "workflow": "workflow_automation",
      "graph": "graph_orchestration",
      "mcp": "mcp_sandbox",
      "compact": "history_compaction",
      "memory": "async_memory",
      "checkpoint": "durable_checkpointing",
      "interrupt": "hitl_checkpointing",
      "a2a": "multi_agent_protocol",
      "trace": "distributed_tracing",
  }
  for keyword, tag in tag_keywords.items():
    if keyword in lowered and tag not in tags:
      tags.append(tag)
  return tags


def _match_curated_releases(
    repositories: list[str], *, include_prereleases: bool
) -> list[NormalizedSignalItem]:
  normalized_repos = {repo.lower() for repo in repositories}
  matched: list[NormalizedSignalItem] = []
  for item in CURATED_GITHUB_RELEASES:
    repo_part = item.signal_id.split(":")[1].lower() if ":" in item.signal_id else ""
    if repo_part in normalized_repos or any(
        repo_part.startswith(r.split("/")[0]) for r in normalized_repos
    ):
      if include_prereleases or (
          "alpha" not in item.title.lower() and "rc" not in item.title.lower()
      ):
        matched.append(item)
  if not matched:
    matched = list(CURATED_GITHUB_RELEASES[:3])
  return matched


async def fetch_github_agent_framework_releases(
    params: GitHubReleaseQueryInput,
    tool_context: ToolContext | None = None,
) -> GitHubReleaseQueryOutput:
  """Fetch recent GitHub releases for AI agent harness and workflow automation frameworks.

  Queries the GitHub REST API (authenticating via Secret Manager when configured)
  for release tags within ``lookback_days`` and normalizes release changelogs into
  scored ``NormalizedSignalItem`` records. Gracefully degrades to curated offline
  fixtures if offline mode is enabled or rate limits block live calls.

  Args:
    params: Validated ``GitHubReleaseQueryInput`` specifying ``repositories`` in
      ``owner/repo`` syntax, ``lookback_days``, and ``include_prereleases``.
    tool_context: Optional ADK tool context for recording collected signals in state.

  Returns:
    A strict ``GitHubReleaseQueryOutput`` containing normalized release signals,
    source provenance (``live_api`` or ``curated_fixture``), and recovery advice
    if partial failures or rate limits occurred.
  """
  settings = RadarSettings.from_env(load_env_file=False)
  resolver = settings.build_secret_resolver()
  github_token = resolve_optional_secret(
      resolver,
      settings.github_token_secret_id,
      env_fallback_var="GITHUB_TOKEN",
  )

  # Use fixtures immediately when explicit offline mode is enabled without a live flag.
  force_live = os.environ.get("RADAR_FORCE_LIVE_HTTP", "").lower() == "true"
  if settings.offline_fixture_fallback and not force_live:
    fixtures = _match_curated_releases(
        params.repositories, include_prereleases=params.include_prereleases
    )
    if tool_context is not None:
      tool_context.state["collected_github_signals"] = [
          item.model_dump(mode="json") for item in fixtures
      ]
    return GitHubReleaseQueryOutput(
        status="success",
        source_mode="curated_fixture",
        releases=fixtures,
        warnings=[
            "Using curated offline release baseline (set RADAR_FORCE_LIVE_HTTP=true for live GitHub API calls)."
        ],
    )

  cutoff = datetime.now(tz=UTC) - timedelta(days=params.lookback_days)
  headers: dict[str, str] = {
      "Accept": "application/vnd.github+json",
      "User-Agent": "agent-harness-signal-radar/0.1",
  }
  if github_token:
    headers["Authorization"] = f"Bearer {github_token}"

  collected: list[NormalizedSignalItem] = []
  warnings: list[str] = []

  async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
    for repo in params.repositories:
      url = f"https://api.github.com/repos/{repo}/releases?per_page=5"
      try:
        response = await client.get(url)
        if response.status_code in {403, 429}:
          warnings.append(
              f"GitHub API rate limit ({response.status_code}) on {repo}; verify Secret Manager token."
          )
          continue
        if response.status_code == 404:
          warnings.append(f"Repository '{repo}' was not found or has no public releases.")
          continue
        response.raise_for_status()
        payload: list[dict[str, Any]] = response.json()
        for rel in payload:
          if rel.get("draft"):
            continue
          if rel.get("prerelease") and not params.include_prereleases:
            continue
          published_raw = str(rel.get("published_at") or rel.get("created_at") or "")
          published_dt = _parse_iso_dt(published_raw)
          if published_dt and published_dt < cutoff:
            continue
          tag = str(rel.get("tag_name") or "untagged")
          name = str(rel.get("name") or tag)
          body = str(rel.get("body") or "Release notes omitted.").strip()
          summary = " ".join(body.split())[:340]
          collected.append(
              NormalizedSignalItem(
                  signal_id=f"gh:{repo}:{tag}",
                  title=f"{repo} {tag}: {name}",
                  canonical_url=str(
                      rel.get("html_url") or f"https://github.com/{repo}/releases"
                  ),
                  source_category="github_release",
                  published_at=published_raw or datetime.now(tz=UTC).isoformat(),
                  summary=summary,
                  authority_score=0.92,
                  engagement_velocity=0.80,
                  harness_or_workflow_tags=_infer_tags(f"{name} {summary}"),
              )
          )
      except httpx.HTTPError as exc:
        _logger.warning("github_release_fetch_error", repo=repo, error=str(exc))
        warnings.append(f"Network error querying {repo}: {type(exc).__name__}")

  if not collected:
    if settings.offline_fixture_fallback:
      fallback = _match_curated_releases(
          params.repositories, include_prereleases=params.include_prereleases
      )
      if tool_context is not None:
        tool_context.state["collected_github_signals"] = [
            item.model_dump(mode="json") for item in fallback
        ]
      return GitHubReleaseQueryOutput(
          status="partial" if warnings else "success",
          source_mode="curated_fixture",
          releases=fallback,
          warnings=warnings
          or ["No live releases matched window; served curated offline baseline."],
      )
    return GitHubReleaseQueryOutput(
        status="error",
        source_mode="error",
        releases=[],
        warnings=warnings,
        recovery_advice=GuidedRecoveryAdvice(
            error_code="GITHUB_FETCH_FAILED",
            error_message="Unable to fetch live GitHub releases for the requested repositories.",
            recovery_instructions=(
                "Verify repository slugs ('owner/repo'), configure GITHUB_TOKEN via Secret Manager "
                "to avoid anonymous rate limits, widen lookback_days, or enable OFFLINE_FIXTURE_FALLBACK=true."
            ),
            suggested_parameters={
                "repositories": ["google/adk-python", "langchain-ai/langgraph"],
                "lookback_days": 60,
                "include_prereleases": False,
            },
            retryable=True,
        ),
    )

  if tool_context is not None:
    tool_context.state["collected_github_signals"] = [
        item.model_dump(mode="json") for item in collected
    ]
  return GitHubReleaseQueryOutput(
      status="partial" if warnings else "success",
      source_mode="live_api",
      releases=collected,
      warnings=warnings,
  )


def build_fetch_github_releases_tool(
    resolver: SecretManagerResolver | None = None,
) -> StrictSchemaFunctionTool:
  """Builds the strict-schema GitHub release harvester tool."""
  del resolver  # Resolver resolved dynamically via settings to respect env/mocking.
  return StrictSchemaFunctionTool(
      func=fetch_github_agent_framework_releases,
      input_schema=GitHubReleaseQueryInput,
      output_schema=GitHubReleaseQueryOutput,
  )
