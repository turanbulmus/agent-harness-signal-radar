"""Runtime configuration settings for Agent Harness Signal Radar."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from .secrets import SecretManagerResolver


def _parse_bool(raw: str | None, default: bool) -> bool:
  if raw is None:
    return default
  normalized = raw.strip().lower()
  if normalized in {"1", "true", "yes", "on"}:
    return True
  if normalized in {"0", "false", "no", "off"}:
    return False
  return default


def _parse_int(raw: str | None, default: int) -> int:
  if raw is None or not raw.strip():
    return default
  try:
    return int(raw.strip())
  except ValueError:
    return default


class RadarSettings(BaseModel):
  """Typed non-secret configuration for the deep research signal consolidator."""

  model_config = ConfigDict(extra="forbid")

  app_name: str = Field(
      default="harness_signal_radar",
      description="ADK application identifier used for sessions, memory, and tracing.",
  )
  google_cloud_project: str | None = Field(
      default=None,
      description="Google Cloud project ID for Secret Manager, Vertex AI, and Cloud Trace.",
  )
  google_cloud_location: str = Field(
      default="us-central1",
      description="Google Cloud region for Vertex AI and Agent Engine.",
  )
  use_enterprise_genai: bool = Field(
      default=False,
      description="Whether to route google-genai SDK calls through Vertex AI Enterprise.",
  )
  reasoning_model: str = Field(
      default="gemini-2.5-pro",
      description="High-capability model for planning, coordination, and cross-source synthesis.",
  )
  fast_model: str = Field(
      default="gemini-2.5-flash",
      description="Low-latency model for parallel extraction, compaction, and self-eval.",
  )
  use_secret_manager: bool = Field(
      default=False,
      description="Enable Google Cloud Secret Manager resolution for runtime credentials.",
  )
  gemini_api_key_secret_id: str = Field(
      default="gemini-api-key",
      description="Secret Manager ID holding the Gemini API key when not using ADC.",
  )
  github_token_secret_id: str = Field(
      default="github-pat-token",
      description="Secret Manager ID holding the GitHub token for higher API rate limits.",
  )
  webhook_url_secret_id: str = Field(
      default="digest-webhook-url",
      description="Secret Manager ID holding the external Slack/Teams/Webhook endpoint.",
  )
  secret_version: str = Field(
      default="latest",
      description="Secret Manager version alias or numeric version.",
  )
  session_db_url: str = Field(
      default="sqlite+aiosqlite:///./.radar_data/sessions.db",
      description="SQLAlchemy async URL for ADK DatabaseSessionService.",
  )
  memory_db_path: str = Field(
      default="./.radar_data/memory.db",
      description="SQLite database path for ADK SqliteMemoryService.",
  )
  vector_db_path: str = Field(
      default="./.radar_data/signal_vectors.db",
      description="SQLite database path for persistent signal & briefing vector search.",
  )
  vertex_ai_agent_engine_id: str | None = Field(
      default=None,
      description="Optional Agent Engine ID enabling VertexAiSessionService and Memory Bank.",
  )
  offline_fixture_fallback: bool = Field(
      default=True,
      description="Fallback to curated signal fixtures if external network calls fail or are offline.",
  )
  enable_cloud_dlp: bool = Field(
      default=False,
      description="Enable Google Cloud DLP API in addition to local deterministic PII scrubbing.",
  )
  enable_cloud_trace: bool = Field(
      default=False,
      description="Enable OpenTelemetry Google Cloud Trace exporter when running in GCP.",
  )
  model_armor_template_id: str = Field(
      default="",
      description="Optional Google Cloud Model Armor template resource name.",
  )
  max_prompt_tokens: int = Field(
      default=16000,
      ge=1024,
      description="Token ceiling before sliding-window history compaction trims old context.",
  )
  log_level: str = Field(
      default="INFO",
      description="Structured JSON logging threshold.",
  )

  @classmethod
  def from_env(cls, *, load_env_file: bool = True) -> RadarSettings:
    """Constructs settings from environment variables and optional .env file."""
    if load_env_file:
      load_dotenv(override=False)

    return cls(
        app_name=os.environ.get("RADAR_APP_NAME", "harness_signal_radar"),
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
        google_cloud_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        use_enterprise_genai=_parse_bool(
            os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE"), False
        ),
        reasoning_model=os.environ.get("REASONING_MODEL", "gemini-2.5-pro"),
        fast_model=os.environ.get("FAST_MODEL", "gemini-2.5-flash"),
        use_secret_manager=_parse_bool(os.environ.get("USE_SECRET_MANAGER"), False),
        gemini_api_key_secret_id=os.environ.get(
            "GEMINI_API_KEY_SECRET_ID", "gemini-api-key"
        ),
        github_token_secret_id=os.environ.get(
            "GITHUB_TOKEN_SECRET_ID", "github-pat-token"
        ),
        webhook_url_secret_id=os.environ.get(
            "WEBHOOK_URL_SECRET_ID", "digest-webhook-url"
        ),
        secret_version=os.environ.get("SECRET_VERSION", "latest"),
        session_db_url=os.environ.get(
            "SESSION_DB_URL", "sqlite+aiosqlite:///./.radar_data/sessions.db"
        ),
        memory_db_path=os.environ.get("MEMORY_DB_PATH", "./.radar_data/memory.db"),
        vector_db_path=os.environ.get(
            "VECTOR_DB_PATH", "./.radar_data/signal_vectors.db"
        ),
        vertex_ai_agent_engine_id=os.environ.get("VERTEX_AI_AGENT_ENGINE_ID") or None,
        offline_fixture_fallback=_parse_bool(
            os.environ.get("OFFLINE_FIXTURE_FALLBACK"), True
        ),
        enable_cloud_dlp=_parse_bool(os.environ.get("ENABLE_CLOUD_DLP"), False),
        enable_cloud_trace=_parse_bool(os.environ.get("ENABLE_CLOUD_TRACE"), False),
        model_armor_template_id=os.environ.get("MODEL_ARMOR_TEMPLATE_ID", ""),
        max_prompt_tokens=_parse_int(os.environ.get("MAX_PROMPT_TOKENS"), 16000),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

  def build_secret_resolver(
      self, *, client: object | None = None
  ) -> SecretManagerResolver:
    """Creates a configured SecretManagerResolver for this settings profile."""
    return SecretManagerResolver(
        project_id=self.google_cloud_project,
        default_version=self.secret_version,
        use_secret_manager=self.use_secret_manager,
        client=client,
    )

  def ensure_local_data_dirs(self) -> None:
    """Creates parent directories for local SQLite session, memory, and vector files."""
    for path_str in (self.memory_db_path, self.vector_db_path):
      if path_str and path_str != ":memory:":
        Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if self.session_db_url.startswith("sqlite+aiosqlite:///"):
      raw_path = self.session_db_url.removeprefix("sqlite+aiosqlite:///")
      if raw_path and raw_path != ":memory:":
        Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
