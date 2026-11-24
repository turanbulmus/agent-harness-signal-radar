"""Cross-source signal correlation, deduplication, and theme clustering tool."""

from __future__ import annotations

from collections import defaultdict

from ..schemas import NormalizedSignalItem, SignalCorrelationInput, SignalCorrelationOutput
from .strict_tool import StrictSchemaFunctionTool


def correlate_and_score_research_signals(
    params: SignalCorrelationInput,
) -> SignalCorrelationOutput:
  """Deduplicate, score, and cluster normalized signals across GitHub, arXiv, and community feeds.

  Combines source authority, engagement momentum, and relevance to ``focus_area``
  to eliminate duplicate announcements, group signals by architectural theme, and
  highlight cross-source convergence (topics confirmed by multiple independent streams).

  Args:
    params: Validated ``SignalCorrelationInput`` containing ``signals``,
      ``focus_area``, and ``top_k``.

  Returns:
    A strict ``SignalCorrelationOutput`` with ranked signals, theme clusters, and
    cross-source convergence highlights.
  """
  focus_terms = {
      t.strip().lower()
      for t in params.focus_area.replace("_", " ").split()
      if len(t.strip()) >= 3
  }

  deduped_by_key: dict[str, NormalizedSignalItem] = {}
  for item in params.signals:
    key = item.canonical_url.rstrip("/").lower() or item.signal_id.lower()
    existing = deduped_by_key.get(key)
    if existing is None or item.authority_score > existing.authority_score:
      deduped_by_key[key] = item

  def _composite_score(item: NormalizedSignalItem) -> float:
    text = f"{item.title} {item.summary} {' '.join(item.harness_or_workflow_tags)}".lower()
    overlap = sum(1 for term in focus_terms if term in text)
    relevance_bonus = min(0.20, overlap * 0.04)
    return (0.55 * item.authority_score) + (0.35 * item.engagement_velocity) + relevance_bonus

  ranked = sorted(
      deduped_by_key.values(),
      key=_composite_score,
      reverse=True,
  )[: params.top_k]

  theme_clusters: dict[str, list[str]] = defaultdict(list)
  tag_sources: dict[str, set[str]] = defaultdict(set)
  for item in ranked:
    tags = item.harness_or_workflow_tags or ["general_agent_engineering"]
    for tag in tags:
      theme_clusters[tag].append(f"{item.title} ({item.canonical_url})")
      tag_sources[tag].add(item.source_category)

  convergence_highlights: list[str] = []
  for tag, sources in sorted(
      tag_sources.items(), key=lambda pair: len(pair[1]), reverse=True
  ):
    if len(sources) >= 2:
      convergence_highlights.append(
          f"Cross-source convergence on '{tag}' across {', '.join(sorted(sources))} "
          f"({len(theme_clusters[tag])} signal(s))."
      )
  if not convergence_highlights and ranked:
    convergence_highlights.append(
        f"Primary concentration around '{ranked[0].harness_or_workflow_tags[0] if ranked[0].harness_or_workflow_tags else 'agent_harness'}' "
        f"led by {ranked[0].title}."
    )

  return SignalCorrelationOutput(
      status="success",
      ranked_signals=ranked,
      theme_clusters=dict(theme_clusters),
      convergence_highlights=convergence_highlights,
      deduplicated_count=max(0, len(params.signals) - len(deduped_by_key)),
  )


def build_correlate_signals_tool() -> StrictSchemaFunctionTool:
  """Builds the strict-schema cross-source signal correlation tool."""
  return StrictSchemaFunctionTool(
      func=correlate_and_score_research_signals,
      input_schema=SignalCorrelationInput,
      output_schema=SignalCorrelationOutput,
  )
