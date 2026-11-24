"""Strategic Model Router dispatching fast extraction vs. deep reasoning tiers."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from typing_extensions import override

from ..config.settings import RadarSettings
from ..observability.logging_config import get_logger

_logger = get_logger("harness_signal_radar.routing")

_DEEP_REASONING_KEYWORDS = frozenset({
    "architecture",
    "architectural",
    "compare",
    "comparison",
    "consolidate",
    "correlation",
    "deep research",
    "design",
    "implications",
    "multi-source",
    "plan",
    "synthesize",
    "synthesis",
    "trade-off",
    "tradeoff",
    "trend",
})

_FAST_EXTRACTION_KEYWORDS = frozenset({
    "check",
    "latest tag",
    "list",
    "lookup",
    "quick",
    "status",
    "triage",
    "version",
    "what is",
})


class TaskTier(StrEnum):
  """Task complexity tiers mapped to cost/latency/reasoning-optimal Gemini models."""

  FAST_EXTRACTION = "fast_extraction"
  DEEP_REASONING = "deep_reasoning"


class StrategicModelRouter:
  """Routes ADK agent nodes and requests to the optimal Gemini model tier."""

  def __init__(
      self,
      *,
      reasoning_model: str = "gemini-2.5-pro",
      fast_model: str = "gemini-2.5-flash",
  ) -> None:
    self.reasoning_model = reasoning_model
    self.fast_model = fast_model

  @classmethod
  def from_settings(cls, settings: RadarSettings) -> StrategicModelRouter:
    return cls(
        reasoning_model=settings.reasoning_model,
        fast_model=settings.fast_model,
    )

  def model_for_tier(
      self,
      tier: TaskTier,
      *,
      model_override: BaseLlm | Mapping[TaskTier, BaseLlm] | None = None,
  ) -> str | BaseLlm:
    """Returns either the injected test BaseLlm or configured production model name."""
    if model_override is not None:
      if isinstance(model_override, Mapping):
        if tier in model_override:
          return model_override[tier]
      else:
        return model_override
    return self.reasoning_model if tier == TaskTier.DEEP_REASONING else self.fast_model

  @staticmethod
  def classify_prompt_complexity(prompt_text: str) -> TaskTier:
    """Classifies whether a request needs deep multi-step reasoning or fast extraction."""
    lowered = prompt_text.strip().lower()
    if not lowered:
      return TaskTier.FAST_EXTRACTION

    if any(kw in lowered for kw in _DEEP_REASONING_KEYWORDS):
      return TaskTier.DEEP_REASONING

    word_count = len(lowered.split())
    if word_count <= 14 and any(kw in lowered for kw in _FAST_EXTRACTION_KEYWORDS):
      return TaskTier.FAST_EXTRACTION

    return TaskTier.DEEP_REASONING if word_count >= 20 else TaskTier.FAST_EXTRACTION


class DynamicModelRoutingPlugin(BasePlugin):
  """Records model tier selections and optionally downgrades simple coordinator turns."""

  def __init__(
      self,
      *,
      router: StrategicModelRouter | None = None,
      allow_dynamic_downgrade: bool = True,
      name: str = "dynamic_model_router",
  ) -> None:
    super().__init__(name=name)
    self._router = router or StrategicModelRouter()
    self._allow_dynamic_downgrade = allow_dynamic_downgrade

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    latest_text = ""
    if llm_request.contents:
      for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
          texts = [p.text for p in content.parts if p.text]
          if texts:
            latest_text = " ".join(texts)
            break

    tier = self._router.classify_prompt_complexity(latest_text)
    selected_model = str(llm_request.model or self._router.reasoning_model)

    if (
        self._allow_dynamic_downgrade
        and callback_context.agent_name == "ResearchCoordinatorAgent"
        and tier == TaskTier.FAST_EXTRACTION
        and isinstance(llm_request.model, str)
        and llm_request.model == self._router.reasoning_model
    ):
      llm_request.model = self._router.fast_model
      selected_model = self._router.fast_model

    callback_context.state["temp:selected_model_tier"] = tier.value
    callback_context.state["temp:selected_model_name"] = selected_model

    _logger.info(
        "model_routing_evaluated",
        agent=callback_context.agent_name,
        classified_tier=tier.value,
        selected_model=selected_model,
    )
    return None
