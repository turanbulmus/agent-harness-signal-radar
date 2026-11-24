"""Vertex AI Agent Engine adapter supporting streaming queries and Secret Manager init."""

from __future__ import annotations

import os
from typing import Any

from .agent import root_agent
from .config.secrets import resolve_optional_secret
from .config.settings import RadarSettings


def create_agent_engine_app(settings: RadarSettings | None = None) -> Any:
  """Builds a Vertex AI AdkApp wrapper with streaming support and secure secret loading."""
  active_settings = settings or RadarSettings.from_env(load_env_file=True)
  resolver = active_settings.build_secret_resolver()

  # Resolve optional API credentials via Secret Manager when deployed outside ADC.
  api_key = resolve_optional_secret(
      resolver,
      active_settings.gemini_api_key_secret_id,
      env_fallback_var="GOOGLE_API_KEY",
  )
  if api_key and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key

  try:
    from vertexai.preview.reasoning_engines import AdkApp
  except ImportError as exc:
    raise RuntimeError(
        "Install optional '[gcp]' dependencies (google-cloud-aiplatform[agent_engines]) "
        "to deploy or instantiate Vertex AI AdkApp."
    ) from exc

  class HarnessSignalRadarAdkApp(AdkApp):
    """Streaming-enabled AdkApp forcing global model routing when required."""

    def query(self, *args: Any, **kwargs: Any) -> Any:
      os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
      return super().query(*args, **kwargs)

    def stream_query(self, *args: Any, **kwargs: Any):
      os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
      yield from super().stream_query(*args, **kwargs)

  return HarnessSignalRadarAdkApp(agent=root_agent)
