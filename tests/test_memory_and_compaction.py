"""Tests for constitutional prompts, history compaction, persistent state, and async memory."""

from __future__ import annotations

from google.adk import Agent
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.genai import types
from harness_signal_radar.memory.async_consolidator import AsyncMemoryConsolidator
from harness_signal_radar.memory.compaction import (
  SignalAwareEventsSummarizer,
  TokenBudgetCompactionPlugin,
  build_context_cache_config,
  build_events_compaction_config,
)
from harness_signal_radar.memory.persistence import (
  PersistentSignalVectorStore,
  create_persistent_memory_service,
  create_persistent_session_service,
)
from harness_signal_radar.prompts.constitution import AGENT_CONSTITUTION


class _SimpleEchoLlm(BaseLlm):
  def __init__(self) -> None:
    super().__init__(model="gemini-2.5-flash")

  async def generate_content_async(self, llm_request, stream=False):
    del llm_request, stream
    yield LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text="Memory verified.")]
        )
    )


def test_constitution_and_compaction_configs():
  for required_section in (
      "<CONSTITUTION>",
      "PERSONA & MISSION",
      "EVIDENCE HIERARCHY",
      "ANTI-HALLUCINATION",
      "SECURITY, PRIVACY & HITL GOVERNANCE",
  ):
    assert required_section in AGENT_CONSTITUTION

  compaction_cfg = build_events_compaction_config()
  assert compaction_cfg.compaction_interval == 4
  assert compaction_cfg.overlap_size == 1
  assert compaction_cfg.token_threshold == 12000
  assert compaction_cfg.event_retention_size == 6

  cache_cfg = build_context_cache_config()
  assert cache_cfg.ttl_seconds == 1800
  assert cache_cfg.min_tokens == 4096


async def test_event_summarizer_and_token_budget_plugin_scrub_and_compact():
  summarizer = SignalAwareEventsSummarizer()
  fake_gh_token = "ghp_" + ("1234abcd" * 3)
  events = [
      Event(
          author="user",
          timestamp=100.0,
          content=types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text=f"Research ADK compaction for alice@example.com with {fake_gh_token}"
                  )
              ],
          ),
      ),
      Event(
          author="model",
          timestamp=105.0,
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text="Retrieved https://github.com/google/adk-python")],
          ),
      ),
  ]

  compacted = await summarizer.maybe_summarize_events(events=events)
  assert compacted is not None
  assert compacted.actions.compaction is not None
  compacted_text = compacted.actions.compaction.compacted_content.parts[0].text or ""
  assert "alice@example.com" not in compacted_text
  assert fake_gh_token not in compacted_text
  assert "[REDACTED_EMAIL]" in compacted_text
  assert "[REDACTED_SECRET]" in compacted_text

  # Exercise TokenBudgetCompactionPlugin on bloated LlmRequest contents
  plugin = TokenBudgetCompactionPlugin(max_prompt_tokens=300, preserve_recent_turns=2)
  agent = Agent(name="compactor_probe", model=_SimpleEchoLlm(), instruction="Probe.")
  app = App(name="compaction_app", root_agent=agent, plugins=[plugin])
  from google.adk.runners import InMemoryRunner

  runner = InMemoryRunner(app=app)
  session = await runner.session_service.create_session(app_name="compaction_app", user_id="u1")
  inv_ctx = runner._new_invocation_context(session=session)
  from google.adk.agents.callback_context import CallbackContext

  cb_ctx = CallbackContext(inv_ctx)

  bloated_contents = [
      types.Content(
          role="user" if i % 2 == 0 else "model",
          parts=[types.Part.from_text(text=f"Turn {i}: " + ("agent_harness " * 40))],
      )
      for i in range(10)
  ]
  req = LlmRequest(model="gemini-2.5-flash", contents=bloated_contents)
  res = await plugin.before_model_callback(callback_context=cb_ctx, llm_request=req)

  assert res is None
  assert len(req.contents) == 4  # 1 head + 1 summary + 2 recent tail
  assert cb_ctx.state["temp:compaction_triggered"] is True
  assert cb_ctx.state["temp:compaction_tokens_after"] < cb_ctx.state["temp:compaction_tokens_before"]


async def test_persistent_session_state_across_runner_restarts(tmp_settings):
  session_service_1 = create_persistent_session_service(tmp_settings)
  agent_1 = Agent(name="persist_probe", model=_SimpleEchoLlm(), instruction="Probe.")
  runner_1 = Runner(
      app_name=tmp_settings.app_name,
      agent=agent_1,
      session_service=session_service_1,
  )
  session = await session_service_1.create_session(
      app_name=tmp_settings.app_name,
      user_id="engineer_1",
      state={"preferred_focus": "mcp_sandbox"},
  )
  msg = types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
  _ = [
      e
      async for e in runner_1.run_async(
          user_id="engineer_1", session_id=session.id, new_message=msg
      )
  ]

  # Instantiate a brand-new DatabaseSessionService pointing to the same SQLite DB
  session_service_2 = create_persistent_session_service(tmp_settings)
  reloaded = await session_service_2.get_session(
      app_name=tmp_settings.app_name,
      user_id="engineer_1",
      session_id=session.id,
  )
  assert reloaded is not None
  assert reloaded.state["preferred_focus"] == "mcp_sandbox"
  assert len(reloaded.events) >= 2


async def test_async_memory_consolidator_nonblocking_and_pii_redacted(tmp_settings):
  session_service = create_persistent_session_service(tmp_settings)
  memory_service = create_persistent_memory_service(tmp_settings)
  vector_store = PersistentSignalVectorStore(db_path=tmp_settings.vector_db_path)
  consolidator = AsyncMemoryConsolidator(
      memory_service=memory_service,
      vector_store=vector_store,
  )

  session = await session_service.create_session(
      app_name=tmp_settings.app_name,
      user_id="u_async",
  )
  session.events.append(
      Event(
          author="user",
          timestamp=200.0,
          content=types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text="ADK workflow graph compaction brief for bob@example.com"
                  )
              ],
          ),
      )
  )

  task = consolidator.enqueue_session_consolidation(
      session,
      brief_payload={
          "headline": "ADK Graph Workflow Compaction Trend",
          "executive_summary": "Sliding-window compaction verified with contact bob@example.com.",
          "citations": ["https://github.com/google/adk-python/releases"],
      },
  )
  assert task is not None
  await consolidator.wait_until_idle(timeout=5.0)
  assert consolidator.completed_count == 1

  # Verify PersistentSignalVectorStore retrieved scrubbed briefing
  vec_hits = await vector_store.search_similar_briefings(
      "ADK graph workflow compaction", top_k=2
  )
  assert len(vec_hits) == 1
  assert "bob@example.com" not in vec_hits[0]["summary"]
  assert "[REDACTED_EMAIL]" in vec_hits[0]["summary"]

  # Verify SqliteMemoryService also persisted scrubbed memory
  mem_hits = await memory_service.search_memory(
      app_name=tmp_settings.app_name,
      user_id="u_async",
      query="compaction",
  )
  assert len(mem_hits.memories) >= 1
  raw_dump = str(mem_hits.memories)
  assert "bob@example.com" not in raw_dump
  await memory_service.close()
