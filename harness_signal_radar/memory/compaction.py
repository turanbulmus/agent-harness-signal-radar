"""Context bloat management: ADK event compaction, context caching, and sliding-window trimming."""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai.types import Content, Part
from typing_extensions import override

from ..observability.logging_config import get_logger
from ..observability.pii_redactor import SensitiveDataRedactor

_logger = get_logger("harness_signal_radar.memory.compaction")


class SignalAwareEventsSummarizer(BaseEventsSummarizer):
  """PII-scrubbing event summarizer supporting LLM and deterministic offline compaction."""

  _RESEARCH_COMPACTION_TEMPLATE = (
      "Summarize the preceding multi-agent deep research interaction concisely. "
      "Preserve: (1) core research question, (2) exact tool names invoked, "
      "(3) verified GitHub releases/arXiv IDs/URLs retrieved, and (4) unresolved "
      "questions or pending HITL approvals. Omit raw PII or credentials.\n\n"
      "{conversation_history}"
  )

  def __init__(
      self,
      *,
      llm: BaseLlm | None = None,
      redactor: SensitiveDataRedactor | None = None,
      max_summary_chars: int = 1600,
  ) -> None:
    self._llm = llm
    self._redactor = redactor or SensitiveDataRedactor()
    self._max_summary_chars = max_summary_chars
    self._llm_summarizer = (
        LlmEventSummarizer(
            llm=llm,
            prompt_template=self._RESEARCH_COMPACTION_TEMPLATE,
        )
        if llm is not None
        else None
    )

  def _deterministic_summary(self, events: list[Event]) -> Event:
    lines: list[str] = []
    tools_used: list[str] = []
    for event in events:
      if not (event.content and event.content.parts):
        continue
      for part in event.content.parts:
        if part.function_call and part.function_call.name:
          if part.function_call.name not in tools_used:
            tools_used.append(part.function_call.name)
        elif part.text:
          cleaned = " ".join(part.text.split())
          if cleaned:
            lines.append(f"- {event.author}: {cleaned[:180]}")

    summary_parts = ["Compacted Research Context Summary:"]
    if tools_used:
      summary_parts.append(f"Tools invoked: {', '.join(tools_used)}")
    summary_parts.extend(lines[-8:])
    summary_text = self._redactor.scrub_text("\n".join(summary_parts))[
        : self._max_summary_chars
    ]

    return Event(
        author="user",
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=events[0].timestamp,
                end_timestamp=events[-1].timestamp,
                compacted_content=Content(
                    role="model",
                    parts=[Part.from_text(text=summary_text)],
                ),
            )
        ),
        invocation_id=Event.new_id(),
    )

  @override
  async def maybe_summarize_events(
      self, *, events: list[Event]
  ) -> Event | None:
    if not events:
      return None

    if self._llm_summarizer is not None:
      try:
        compacted = await self._llm_summarizer.maybe_summarize_events(events=events)
        if (
            compacted is not None
            and compacted.actions.compaction is not None
            and compacted.actions.compaction.compacted_content
            and compacted.actions.compaction.compacted_content.parts
        ):
          for part in compacted.actions.compaction.compacted_content.parts:
            if part.text:
              part.text = self._redactor.scrub_text(part.text)
          return compacted
      except Exception as exc:
        _logger.warning(
            "llm_compaction_fallback_to_deterministic",
            error=type(exc).__name__,
        )

    return self._deterministic_summary(events)


def build_events_compaction_config(
    *,
    llm: BaseLlm | None = None,
    compaction_interval: int = 4,
    overlap_size: int = 1,
    token_threshold: int = 12000,
    event_retention_size: int = 6,
) -> EventsCompactionConfig:
  """Builds ADK's EventsCompactionConfig combining sliding-window & token thresholds."""
  return EventsCompactionConfig(
      summarizer=SignalAwareEventsSummarizer(llm=llm),
      compaction_interval=compaction_interval,
      overlap_size=overlap_size,
      token_threshold=token_threshold,
      event_retention_size=event_retention_size,
  )


def build_context_cache_config(
    *,
    cache_intervals: int = 10,
    ttl_seconds: int = 1800,
    min_tokens: int = 4096,
) -> ContextCacheConfig:
  """Builds Google Cloud / Gemini ContextCacheConfig for multi-turn transfer efficiency."""
  return ContextCacheConfig(
      cache_intervals=cache_intervals,
      ttl_seconds=ttl_seconds,
      min_tokens=min_tokens,
  )


class TokenBudgetCompactionPlugin(BasePlugin):
  """Enforces a hard prompt token budget via sliding-window middle summarization."""

  def __init__(
      self,
      *,
      name: str = "token_budget_compactor",
      max_prompt_tokens: int = 16000,
      preserve_recent_turns: int = 4,
      redactor: SensitiveDataRedactor | None = None,
  ) -> None:
    super().__init__(name=name)
    self.max_prompt_tokens = max(256, max_prompt_tokens)
    self.preserve_recent_turns = max(1, preserve_recent_turns)
    self._redactor = redactor or SensitiveDataRedactor()

  @staticmethod
  def _content_chars(content: Content) -> int:
    total = 0
    for part in content.parts or []:
      if part.text:
        total += len(part.text)
      if part.function_call:
        total += len(str(part.function_call.name or "")) + len(
            str(part.function_call.args or "")
        )
      if part.function_response:
        total += len(str(part.function_response.name or "")) + len(
            str(part.function_response.response or "")
        )
    return total

  @classmethod
  def estimate_tokens(cls, contents: list[Content]) -> int:
    total_chars = sum(cls._content_chars(c) for c in contents)
    return max(1, total_chars // 4) if total_chars else 0

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    contents = list(llm_request.contents or [])
    estimated = self.estimate_tokens(contents)
    if (
        estimated <= self.max_prompt_tokens
        or len(contents) <= self.preserve_recent_turns + 1
    ):
      return None

    head = contents[:1]
    tail = contents[-self.preserve_recent_turns :]
    middle = contents[1 : -self.preserve_recent_turns]

    middle_snippets: list[str] = []
    for item in middle:
      role = item.role or "unknown"
      for part in item.parts or []:
        if part.text:
          snippet = " ".join(part.text.split())[:120]
          middle_snippets.append(f"[{role}] {snippet}")
        elif part.function_call:
          middle_snippets.append(f"[{role}:call] {part.function_call.name}")
        elif part.function_response:
          middle_snippets.append(f"[{role}:result] {part.function_response.name}")

    compacted_text = self._redactor.scrub_text(
        "Compacted earlier conversation turns (token budget guard):\n"
        + "\n".join(middle_snippets[-10:])
    )
    summary_content = Content(
        role="user",
        parts=[Part.from_text(text=compacted_text)],
    )
    compacted_contents: list[Any] = head + [summary_content] + tail
    new_estimate = self.estimate_tokens(compacted_contents)
    llm_request.contents = compacted_contents

    callback_context.state["temp:compaction_triggered"] = True
    callback_context.state["temp:compaction_tokens_before"] = estimated
    callback_context.state["temp:compaction_tokens_after"] = new_estimate

    _logger.info(
        "prompt_history_compacted",
        agent=callback_context.agent_name,
        tokens_before=estimated,
        tokens_after=new_estimate,
        removed_turns=len(middle),
    )
    return None
