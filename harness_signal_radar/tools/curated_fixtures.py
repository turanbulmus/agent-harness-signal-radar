"""High-signal curated fixtures for offline verification and graceful API degradation."""

from __future__ import annotations

from ..schemas import NormalizedSignalItem

CURATED_GITHUB_RELEASES: tuple[NormalizedSignalItem, ...] = (
    NormalizedSignalItem(
        signal_id="gh:google/adk-python:v2.15.0",
        title="google/adk-python v2.15.0: Graph Workflow Engine, EventsCompactionConfig & Context Caching",
        canonical_url="https://github.com/google/adk-python/releases/tag/v2.15.0",
        source_category="github_release",
        published_at="2026-11-18T10:00:00Z",
        summary=(
            "Introduces declarative graph Workflow orchestration with JoinNode fan-in, "
            "parallel_worker execution, token-threshold EventsCompactionConfig, "
            "ContextCacheConfig across multi-agent transfers, and RequestInput HITL checkpoints."
        ),
        authority_score=0.96,
        engagement_velocity=0.91,
        harness_or_workflow_tags=[
            "agent_harness",
            "graph_orchestration",
            "history_compaction",
            "hitl_checkpointing",
        ],
    ),
    NormalizedSignalItem(
        signal_id="gh:langchain-ai/langgraph:v1.0.10",
        title="langchain-ai/langgraph v1.0.10: Durable Subgraph Checkpointing & Interrupt Replay",
        canonical_url="https://github.com/langchain-ai/langgraph/releases/tag/v1.0.10",
        source_category="github_release",
        published_at="2026-11-14T16:30:00Z",
        summary=(
            "Hardens durable state checkpointing for cyclic agent workflows, time-travel "
            "branch replay, and streaming partial state updates across nested subgraphs."
        ),
        authority_score=0.93,
        engagement_velocity=0.88,
        harness_or_workflow_tags=[
            "workflow_automation",
            "durable_checkpointing",
            "hitl_checkpointing",
        ],
    ),
    NormalizedSignalItem(
        signal_id="gh:modelcontextprotocol/specification:2026-11",
        title="modelcontextprotocol/specification: Scoped Tool Elicitation & Sandboxed MCP Tasks",
        canonical_url="https://github.com/modelcontextprotocol/modelcontextprotocol/releases",
        source_category="github_release",
        published_at="2026-11-12T09:15:00Z",
        summary=(
            "Standardizes structured elicitation schemas for human confirmation, long-running "
            "task polling over Streamable HTTP, and granular tool permission boundaries."
        ),
        authority_score=0.95,
        engagement_velocity=0.94,
        harness_or_workflow_tags=[
            "agent_harness",
            "mcp_sandbox",
            "strict_tool_schemas",
            "hitl_checkpointing",
        ],
    ),
    NormalizedSignalItem(
        signal_id="gh:a2aproject/A2A:v0.3.4",
        title="a2aproject/A2A v0.3.4: Signed AgentCards & Streaming Task Artifact Handoffs",
        canonical_url="https://github.com/a2aproject/A2A/releases",
        source_category="github_release",
        published_at="2026-11-10T14:00:00Z",
        summary=(
            "Adds verifiable capability cards and asynchronous task artifact streaming for "
            "cross-framework agent-to-agent workflow delegation."
        ),
        authority_score=0.90,
        engagement_velocity=0.84,
        harness_or_workflow_tags=[
            "agent_harness",
            "multi_agent_protocol",
            "workflow_automation",
        ],
    ),
    NormalizedSignalItem(
        signal_id="gh:temporalio/sdk-python:v1.18.0",
        title="temporalio/sdk-python v1.18.0: Deterministic Saga Retries for Long-Running AI Agents",
        canonical_url="https://github.com/temporalio/sdk-python/releases",
        source_category="github_release",
        published_at="2026-11-08T18:45:00Z",
        summary=(
            "Provides durable activity heartbeats, bounded retry backoff, and deterministic "
            "replay wrappers for LLM tool calls and multi-step enterprise automations."
        ),
        authority_score=0.91,
        engagement_velocity=0.82,
        harness_or_workflow_tags=[
            "workflow_automation",
            "durable_checkpointing",
            "fault_tolerance",
        ],
    ),
)

CURATED_ARXIV_PAPERS: tuple[NormalizedSignalItem, ...] = (
    NormalizedSignalItem(
        signal_id="arxiv:2510.14201",
        title="Deterministic Harnesses for Stochastic Agents: Bounded Verification via Typed DAGs",
        canonical_url="https://arxiv.org/abs/2510.14201",
        source_category="arxiv_paper",
        published_at="2026-11-15T00:00:00Z",
        summary=(
            "Demonstrates that replacing unbounded ReAct loops with schema-typed workflow "
            "graphs and explicit verification gates cuts multi-step tool failure rates by 43%."
        ),
        authority_score=0.92,
        engagement_velocity=0.79,
        harness_or_workflow_tags=[
            "agent_harness",
            "graph_orchestration",
            "self_evaluation",
        ],
    ),
    NormalizedSignalItem(
        signal_id="arxiv:2511.03184",
        title="Sliding-Window Event Compaction & Episodic Memory Consolidation in Long-Horizon Agents",
        canonical_url="https://arxiv.org/abs/2511.03184",
        source_category="arxiv_paper",
        published_at="2026-11-11T00:00:00Z",
        summary=(
            "Evaluates token-threshold event summarization combined with asynchronous background "
            "memory banks, reducing prompt token cost by 58% without recall degradation."
        ),
        authority_score=0.90,
        engagement_velocity=0.76,
        harness_or_workflow_tags=[
            "agent_harness",
            "history_compaction",
            "async_memory",
        ],
    ),
    NormalizedSignalItem(
        signal_id="arxiv:2511.07892",
        title="Policy-as-Code Guardrails and Human-in-the-Loop Interruption for Enterprise Workflows",
        canonical_url="https://arxiv.org/abs/2511.07892",
        source_category="arxiv_paper",
        published_at="2026-11-06T00:00:00Z",
        summary=(
            "Proposes pre-tool policy plugins paired with resumable interrupt gates for "
            "high-stakes external side effects, achieving zero unauthorized state mutations."
        ),
        authority_score=0.89,
        engagement_velocity=0.74,
        harness_or_workflow_tags=[
            "workflow_automation",
            "policy_guardrails",
            "hitl_checkpointing",
        ],
    ),
)

CURATED_HN_DISCUSSIONS: tuple[NormalizedSignalItem, ...] = (
    NormalizedSignalItem(
        signal_id="hn:42901821",
        title="Show HN: Why We Replaced Single-Prompt Agents with Typed Workflow Graphs and Strict Schemas",
        canonical_url="https://news.ycombinator.com/item?id=42901821",
        source_category="hackernews_discussion",
        published_at="2026-11-19T13:20:00Z",
        summary=(
            "Production postmortem comparing monolithic chat loops against hybrid Pro-for-planning "
            "and Flash-for-extraction DAG harnesses with OpenTelemetry trace correlation."
        ),
        authority_score=0.84,
        engagement_velocity=0.93,
        harness_or_workflow_tags=[
            "agent_harness",
            "model_routing",
            "distributed_tracing",
            "graph_orchestration",
        ],
    ),
    NormalizedSignalItem(
        signal_id="hn:42894410",
        title="Lessons Running MCP Tool Sandboxes in Multi-Tenant Workflow Automation",
        canonical_url="https://news.ycombinator.com/item?id=42894410",
        source_category="hackernews_discussion",
        published_at="2026-11-16T08:45:00Z",
        summary=(
            "Engineers detail guided tool-error recovery payloads, PII scrubbing before log "
            "persistence, and Secret Manager credential injection for isolated tool execution."
        ),
        authority_score=0.82,
        engagement_velocity=0.89,
        harness_or_workflow_tags=[
            "mcp_sandbox",
            "pii_redaction",
            "guided_error_recovery",
            "workflow_automation",
        ],
    ),
    NormalizedSignalItem(
        signal_id="hn:42871204",
        title="Ask HN: How Are Teams Evaluating Agent Regressions in CI Before Deploy?",
        canonical_url="https://news.ycombinator.com/item?id=42871204",
        source_category="hackernews_discussion",
        published_at="2026-11-09T17:10:00Z",
        summary=(
            "Consensus favors deterministic FakeLlm golden-dataset trajectory suites in PR CI "
            "coupled with Intent-vs-Outcome structured JSON logs in staging."
        ),
        authority_score=0.80,
        engagement_velocity=0.85,
        harness_or_workflow_tags=[
            "agent_evals",
            "observability",
            "agent_harness",
        ],
    ),
)
