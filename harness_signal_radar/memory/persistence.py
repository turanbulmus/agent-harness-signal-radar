"""Persistent session state, PII-redacted long-term memory, and SQLite vector storage."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from google.adk.events.event import Event
from google.adk.memory import BaseMemoryService, SqliteMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.sessions import BaseSessionService, DatabaseSessionService, Session
from google.genai import types
from typing_extensions import override

from ..config.settings import RadarSettings
from ..observability.pii_redactor import SensitiveDataRedactor

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
_VECTOR_DIMS = 128


def _compute_text_embedding(text: str) -> list[float]:
  """Builds a deterministic L2-normalized feature-hashing embedding vector."""
  vec = [0.0] * _VECTOR_DIMS
  tokens = _TOKEN_RE.findall(text.lower())
  if not tokens:
    return vec
  for token in tokens:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "big") % _VECTOR_DIMS
    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
    vec[bucket] += sign
  norm = math.sqrt(sum(v * v for v in vec))
  if norm > 0.0:
    vec = [round(v / norm, 6) for v in vec]
  return vec


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
  if not vec_a or not vec_b or len(vec_a) != len(vec_b):
    return 0.0
  return round(sum(a * b for a, b in zip(vec_a, vec_b, strict=True)), 6)


class PersistentSignalVectorStore:
  """Persistent SQLite-backed vector store for consolidated research briefings."""

  def __init__(
      self,
      db_path: str | Path = "./.radar_data/signal_vectors.db",
      *,
      redactor: SensitiveDataRedactor | None = None,
  ) -> None:
    self.db_path = str(db_path)
    self._redactor = redactor or SensitiveDataRedactor()
    if self.db_path != ":memory:":
      Path(self.db_path).expanduser().resolve().parent.mkdir(
          parents=True, exist_ok=True
      )

  async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS archived_briefings (
            briefing_id TEXT PRIMARY KEY,
            headline TEXT NOT NULL,
            summary TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    await conn.commit()

  async def upsert_briefing(
      self,
      *,
      briefing_id: str,
      headline: str,
      summary: str,
      citations: list[str],
      created_at: str | None = None,
  ) -> None:
    """Scrubs PII/secrets and persists a consolidated briefing vector record."""
    clean_headline = self._redactor.scrub_text(headline)
    clean_summary = self._redactor.scrub_text(summary)
    clean_citations = [self._redactor.scrub_text(c) for c in citations]
    embedding = _compute_text_embedding(f"{clean_headline} {clean_summary}")
    timestamp = created_at or datetime.now(tz=UTC).isoformat()

    async with aiosqlite.connect(self.db_path) as conn:
      await self._ensure_schema(conn)
      await conn.execute(
          """
          INSERT INTO archived_briefings (
              briefing_id, headline, summary, citations_json, vector_json, created_at
          ) VALUES (?, ?, ?, ?, ?, ?)
          ON CONFLICT(briefing_id) DO UPDATE SET
              headline = excluded.headline,
              summary = excluded.summary,
              citations_json = excluded.citations_json,
              vector_json = excluded.vector_json,
              created_at = excluded.created_at
          """,
          (
              briefing_id,
              clean_headline,
              clean_summary,
              json.dumps(clean_citations),
              json.dumps(embedding),
              timestamp,
          ),
      )
      await conn.commit()

  async def search_similar_briefings(
      self, query: str, *, top_k: int = 3
  ) -> list[dict[str, Any]]:
    """Retrieves top-k semantically similar archived research briefings."""
    if self.db_path != ":memory:" and not Path(self.db_path).exists():
      return []

    query_vec = _compute_text_embedding(self._redactor.scrub_text(query))
    async with aiosqlite.connect(self.db_path) as conn:
      await self._ensure_schema(conn)
      cursor = await conn.execute(
          """
          SELECT briefing_id, headline, summary, citations_json, vector_json, created_at
          FROM archived_briefings
          """
      )
      rows = await cursor.fetchall()

    scored: list[tuple[float, dict[str, Any]]] = []
    for briefing_id, headline, summary, citations_json, vector_json, created_at in rows:
      stored_vec = json.loads(vector_json)
      sim = _cosine_similarity(query_vec, stored_vec)
      scored.append((
          sim,
          {
              "briefing_id": briefing_id,
              "headline": headline,
              "summary": summary,
              "citations": json.loads(citations_json),
              "similarity_score": sim,
              "created_at": created_at,
          },
      ))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


class PiiRedactingMemoryService(BaseMemoryService):
  """Decorator around ADK BaseMemoryService that scrubs PII/secrets prior to storage."""

  def __init__(
      self,
      backend: BaseMemoryService,
      *,
      redactor: SensitiveDataRedactor | None = None,
  ) -> None:
    self._backend = backend
    self._redactor = redactor or SensitiveDataRedactor()

  def _sanitize_event(self, event: Event) -> Event:
    if not event.content or not event.content.parts:
      return event
    new_parts: list[types.Part] = []
    changed = False
    for part in event.content.parts:
      if part.text:
        scrubbed = self._redactor.scrub_text(part.text)
        if scrubbed != part.text:
          changed = True
        new_parts.append(types.Part.from_text(text=scrubbed))
      else:
        new_parts.append(part)
    if not changed:
      return event
    sanitized_content = types.Content(role=event.content.role, parts=new_parts)
    return event.model_copy(update={"content": sanitized_content})

  def _sanitize_session(self, session: Session) -> Session:
    sanitized_events = [self._sanitize_event(event) for event in session.events]
    sanitized_state = self._redactor.scrub_payload(dict(session.state))
    return session.model_copy(
        update={"events": sanitized_events, "state": sanitized_state}
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    await self._backend.add_session_to_memory(self._sanitize_session(session))

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    sanitized_events = [self._sanitize_event(e) for e in events]
    await self._backend.add_events_to_memory(
        app_name=app_name,
        user_id=user_id,
        events=sanitized_events,
        session_id=session_id,
        custom_metadata=custom_metadata,
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    scrubbed_query = self._redactor.scrub_text(query)
    return await self._backend.search_memory(
        app_name=app_name,
        user_id=user_id,
        query=scrubbed_query,
    )

  async def close(self) -> None:
    close_fn = getattr(self._backend, "close", None)
    if callable(close_fn):
      result = close_fn()
      if hasattr(result, "__await__"):
        await result


def create_persistent_session_service(settings: RadarSettings) -> BaseSessionService:
  """Creates a persistent session service (Vertex AI in GCP, SQLAlchemy DB locally)."""
  if settings.vertex_ai_agent_engine_id and settings.google_cloud_project:
    from google.adk.sessions import VertexAiSessionService

    return VertexAiSessionService(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        agent_engine_id=settings.vertex_ai_agent_engine_id,
    )

  settings.ensure_local_data_dirs()
  return DatabaseSessionService(db_url=settings.session_db_url)


def create_persistent_memory_service(
    settings: RadarSettings,
    *,
    redactor: SensitiveDataRedactor | None = None,
) -> PiiRedactingMemoryService:
  """Creates a PII-redacting persistent memory service (Vertex AI or SQLite FTS)."""
  active_redactor = redactor or SensitiveDataRedactor(
      enable_cloud_dlp=settings.enable_cloud_dlp,
      project_id=settings.google_cloud_project,
  )
  if settings.vertex_ai_agent_engine_id and settings.google_cloud_project:
    from google.adk.memory import VertexAiMemoryBankService

    backend: BaseMemoryService = VertexAiMemoryBankService(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        agent_engine_id=settings.vertex_ai_agent_engine_id,
    )
  else:
    settings.ensure_local_data_dirs()
    backend = SqliteMemoryService(db_path=settings.memory_db_path)

  return PiiRedactingMemoryService(backend=backend, redactor=active_redactor)
