"""Domain tool builders with strict schemas and guided recovery."""

from .arxiv_papers import (
    build_search_arxiv_papers_tool,
    search_arxiv_agent_orchestration_papers,
)
from .github_releases import (
    build_fetch_github_releases_tool,
    fetch_github_agent_framework_releases,
)
from .hackernews_signals import (
    build_scan_hackernews_signals_tool,
    scan_hackernews_workflow_automation_signals,
)
from .memory_and_publishing import (
    build_editorial_approval_tool,
    build_publish_digest_tool,
    build_retrieve_archived_briefings_tool,
    publish_approved_executive_signal_digest,
    request_human_editorial_publication_approval,
    retrieve_archived_signal_briefings,
)
from .signal_correlator import (
    build_correlate_signals_tool,
    correlate_and_score_research_signals,
)
from .strict_tool import StrictSchemaFunctionTool

__all__ = [
    "StrictSchemaFunctionTool",
    "build_correlate_signals_tool",
    "build_editorial_approval_tool",
    "build_fetch_github_releases_tool",
    "build_publish_digest_tool",
    "build_retrieve_archived_briefings_tool",
    "build_scan_hackernews_signals_tool",
    "build_search_arxiv_papers_tool",
    "correlate_and_score_research_signals",
    "fetch_github_agent_framework_releases",
    "publish_approved_executive_signal_digest",
    "request_human_editorial_publication_approval",
    "retrieve_archived_signal_briefings",
    "scan_hackernews_workflow_automation_signals",
    "search_arxiv_agent_orchestration_papers",
]
