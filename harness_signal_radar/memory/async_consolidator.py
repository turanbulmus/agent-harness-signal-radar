"""Non-blocking background memory and vector-store consolidation tasks."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.memory import BaseMemoryService
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions import Session
from typing_extensions import override

from ..observability.logging_config import get_logger
from ..observability.pii_redactor import SensitiveDataRedactor
from ..observability.tracing import start_span
from .persistence import PersistentSignalVectorStore

_logger = get_logger("harness_signal_radar.memory.async")


@dataclass(slots=True)
class _ConsolidationJob:
  job_id: str
  session: Session
  brief_payload: dict[str, Any] | None


class AsyncMemoryConsolidator:
  """Executes expensive memory & vector indexing asynchronously without blocking UI turns."""

  def __init__(
      self,
      *,
      memory_service: BaseMemoryService | None = None,
      vector_store: PersistentSignalVectorStore | None = None,
      redactor: SensitiveDataRedactor | None = None,
  ) -> None:
    self._memory_service = memory_service
    self._vector_store = vector_store
    self._redactor = redactor or SensitiveDataRedactor()
    self._background_tasks: set[asyncio.Task[Any]] = set()
    self._pending_jobs = 0
    self._idle_event: asyncio.Event | None = None
    self._completed_count = 0

  def configure_backends(
      self,
      *,
      memory_service: BaseMemoryService | None = None,
      vector_store: PersistentSignalVectorStore | None = None,
  ) -> None:
    """Binds or updates storage backends lazily (useful when initialized per-runner)."""
    if memory_service is not None:
      self._memory_service = memory_service
    if vector_store is not None:
      self._vector_store = vector_store

  @property
  def completed_count(self) -> int:
    return self._completed_count

  @property
  def active_task_count(self) -> int:
    return len(self._background_tasks)

  def _ensure_idle_event(self) -> asyncio.Event:
    if self._idle_event is None:
      self._idle_event = asyncio.Event()
      if self._pending_jobs == 0:
        self._idle_event.set()
    return self._idle_event

  async def _run_job(self, job: _ConsolidationJob) -> None:
    try:
      with start_span(
          "memory.async_consolidation",
          attributes={
              "memory.job_id": job.job_id,
              "session.id": job.session.id,
          },
      ):
        if self._memory_service is not None and job.session.events:
          await self._memory_service.add_session_to_memory(job.session)

        payload = job.brief_payload
        if payload is None:
          raw_brief = job.session.state.get("final_research_brief") or job.session.state.get(
              "draft_research_brief"
          )
          if isinstance(raw_brief, dict):
            payload = raw_brief

        if self._vector_store is not None and isinstance(payload, dict):
          headline = str(payload.get("headline") or "").strip()
          summary = str(payload.get("executive_summary") or "").strip()
          citations = [str(c) for c in payload.get("citations", [])]
          if headline and summary:
            await self._vector_store.upsert_briefing(
                briefing_id=f"{job.session.id}:{job.job_id}",
                headline=self._redactor.scrub_text(headline),
                summary=self._redactor.scrub_text(summary),
                citations=citations,
            )

        self._completed_count += 1
        _logger.info(
            "async_memory_consolidation_completed",
            job_id=job.job_id,
            session_id=job.session.id,
            indexed_briefing=bool(payload),
        )
    except Exception as exc:
      _logger.warning(
          "async_memory_consolidation_failed",
          job_id=job.job_id,
          session_id=job.session.id,
          error=self._redactor.scrub_text(str(exc)),
      )
    finally:
      self._pending_jobs = max(0, self._pending_jobs - 1)
      if self._pending_jobs == 0 and self._idle_event is not None:
        self._idle_event.set()

  def enqueue_session_consolidation(
      self,
      session: Session,
      brief_payload: dict[str, Any] | None = None,
  ) -> asyncio.Task[Any] | None:
    """Schedules background memory & vector indexing, returning immediately."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      return None

    idle_event = self._ensure_idle_event()
    self._pending_jobs += 1
    idle_event.clear()

    job = _ConsolidationJob(
        job_id=uuid.uuid4().hex[:10],
        session=session.model_copy(deep=True),
        brief_payload=self._redactor.scrub_payload(brief_payload)
        if brief_payload is not None
        else None,
    )
    task = loop.create_task(self._run_job(job), name=f"memory-consolidate-{job.job_id}")
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)
    return task

  async def wait_until_idle(self, *, timeout: float = 5.0) -> None:
    """Awaits completion of all scheduled background consolidation tasks."""
    if self._pending_jobs == 0 and not self._background_tasks:
      return
    idle_event = self._ensure_idle_event()
    await asyncio.wait_for(idle_event.wait(), timeout=timeout)
    if self._background_tasks:
      await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

  async def shutdown(self) -> None:
    """Drains remaining consolidation tasks cleanly."""
    await self.wait_until_idle(timeout=5.0)


class AsyncMemoryLifecyclePlugin(BasePlugin):
  """Triggers non-blocking background memory consolidation after each ADK run."""

  def __init__(
      self,
      *,
      consolidator: AsyncMemoryConsolidator,
      name: str = "async_memory_lifecycle",
  ) -> None:
    super().__init__(name=name)
    self._consolidator = consolidator

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    if invocation_context.session is None:
      return
    if invocation_context.memory_service is not None:
      self._consolidator.configure_backends(
          memory_service=invocation_context.memory_service
      )
    self._consolidator.enqueue_session_consolidation(invocation_context.session)
