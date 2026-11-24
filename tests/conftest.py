"""Shared pytest fixtures for isolated test execution and OpenTelemetry capture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from harness_signal_radar.config.settings import RadarSettings
from harness_signal_radar.observability.tracing import (
  clear_registered_exporters,
  register_span_exporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def tmp_settings(tmp_path: Path) -> RadarSettings:
  return RadarSettings(
      session_db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}",
      memory_db_path=str(tmp_path / "memory.db"),
      vector_db_path=str(tmp_path / "signal_vectors.db"),
      offline_fixture_fallback=True,
      use_secret_manager=False,
      log_level="INFO",
  )


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
  exporter = InMemorySpanExporter()
  register_span_exporter(exporter)
  try:
    yield exporter
  finally:
    clear_registered_exporters()
