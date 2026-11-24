"""Tools for persistent briefing recall, human editorial approval, and digest publishing."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
from google.adk.tools import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

from ..config.secrets import resolve_optional_secret
from ..config.settings import RadarSettings
from ..memory.persistence import PersistentSignalVectorStore
from ..observability.logging_config import get_logger
from ..observability.pii_redactor import SensitiveDataRedactor
from ..schemas import (
  ArchivedBriefingQueryInput,
  ArchivedBriefingQueryOutput,
  EditorialApprovalRequestInput,
  GuidedRecoveryAdvice,
  PublishExecutiveDigestInput,
  PublishExecutiveDigestOutput,
)
from .strict_tool import StrictSchemaFunctionTool

_logger = get_logger("harness_signal_radar.tools.publishing")


async def retrieve_archived_signal_briefings(
    params: ArchivedBriefingQueryInput,
) -> ArchivedBriefingQueryOutput:
  """Retrieve historically consolidated research briefings from persistent vector storage.

  Searches the local persistent SQLite signal vector store (backed by scrubbed
  cross-session intelligence summaries) to surface longitudinal context, compare
  trend evolution over time, and avoid duplicating prior research.

  Args:
    params: Validated ``ArchivedBriefingQueryInput`` with semantic ``query`` and ``top_k``.

  Returns:
    A strict ``ArchivedBriefingQueryOutput`` listing top matching prior briefings.
  """
  settings = RadarSettings.from_env(load_env_file=False)
  store = PersistentSignalVectorStore(db_path=settings.vector_db_path)
  matches = await store.search_similar_briefings(params.query, top_k=params.top_k)
  return ArchivedBriefingQueryOutput(
      status="success",
      matches=matches,
  )


def request_human_editorial_publication_approval(
    brief_headline: str,
    target_channel: str,
    citation_count: int,
    executive_preview: str,
    tool_context: ToolContext | None = None,
) -> dict[str, object]:
  """Pause execution to obtain mandatory human editorial confirmation before publishing.

  High-stakes external communication MUST NOT bypass this Human-in-the-Loop checkpoint.
  When invoked as an ADK ``LongRunningFunctionTool``, this signals the client/operator
  to review the headline, citation count, target channel, and executive preview and
  respond with an approval decision before calling ``publish_approved_executive_signal_digest``.

  Args:
    brief_headline: Headline of the consolidated research brief pending review.
    target_channel: Intended external publication target (e.g. ``engineering_digest``).
    citation_count: Count of verified canonical source URLs supporting the brief.
    executive_preview: Short PII-scrubbed excerpt of the executive summary for sign-off.
    tool_context: ADK tool execution context used to persist pending approval metadata.

  Returns:
    A pending HITL ticket payload referencing the ``brief_headline`` and ``target_channel``.
  """
  validated = EditorialApprovalRequestInput(
      brief_headline=brief_headline,
      target_channel=target_channel,
      citation_count=citation_count,
      executive_preview=executive_preview,
  )
  redactor = SensitiveDataRedactor()
  ticket_id = f"hitl-{uuid.uuid4().hex[:10]}"
  if tool_context is not None:
    tool_context.state["editorial_approval_status"] = "PENDING"
    tool_context.state["editorial_approval_ticket_id"] = ticket_id
    tool_context.state["editorial_target_channel"] = validated.target_channel

  return {
      "status": "pending_human_approval",
      "ticket_id": ticket_id,
      "brief_headline": redactor.scrub_text(validated.brief_headline),
      "target_channel": validated.target_channel,
      "citation_count": validated.citation_count,
      "executive_preview": redactor.scrub_text(validated.executive_preview),
      "instructions_for_operator": (
          "Reply with FunctionResponse {'approved': true, 'approved_channel': '<channel>'} "
          "to authorize external digest publication."
      ),
  }


async def publish_approved_executive_signal_digest(
    params: PublishExecutiveDigestInput,
    tool_context: ToolContext | None = None,
) -> PublishExecutiveDigestOutput:
  """Publish an approved, PII-scrubbed executive signal digest to the configured channel.

  Enforces Human-in-the-Loop authorization: refuses execution unless
  ``editorial_approval_status == 'APPROVED'`` in session state. Uses Google Cloud
  Secret Manager to inject webhook credentials when configured, or records an
  idempotent dry-run delivery receipt locally.

  Args:
    params: Validated ``PublishExecutiveDigestInput`` containing ``headline``,
      ``markdown_body``, ``target_channel``, and ``citations``.
    tool_context: ADK tool execution context carrying HITL approval state.

  Returns:
    A strict ``PublishExecutiveDigestOutput`` with delivery receipt or guided
    blocking instructions if human approval has not been granted yet.
  """
  approval_status = (
      str(tool_context.state.get("editorial_approval_status", "UNAPPROVED"))
      if tool_context is not None
      else "UNAPPROVED"
  )
  if approval_status != "APPROVED":
    return PublishExecutiveDigestOutput(
        status="blocked",
        target_channel=params.target_channel,
        delivery_mode="blocked",
        recovery_advice=GuidedRecoveryAdvice(
            error_code="HITL_APPROVAL_REQUIRED",
            error_message=(
                "External digest publication is blocked because human editorial approval "
                f"status is '{approval_status}' (must be 'APPROVED')."
            ),
            recovery_instructions=(
                "Invoke 'request_human_editorial_publication_approval' first and wait for "
                "explicit human approval before calling 'publish_approved_executive_signal_digest'."
            ),
            suggested_parameters={
                "brief_headline": params.headline,
                "target_channel": params.target_channel,
                "citation_count": len(params.citations),
                "executive_preview": params.markdown_body[:240],
            },
            retryable=True,
        ),
    )

  redactor = SensitiveDataRedactor()
  clean_headline = redactor.scrub_text(params.headline)
  clean_body = redactor.scrub_text(params.markdown_body)
  settings = RadarSettings.from_env(load_env_file=False)
  resolver = settings.build_secret_resolver()
  webhook_url = resolve_optional_secret(
      resolver,
      settings.webhook_url_secret_id,
      env_fallback_var="DIGEST_WEBHOOK_URL",
  )

  published_at = datetime.now(tz=UTC).isoformat()
  delivery_id = f"digest-{uuid.uuid4().hex[:12]}"

  if webhook_url:
    try:
      async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(
            webhook_url,
            json={
                "delivery_id": delivery_id,
                "channel": params.target_channel,
                "headline": clean_headline,
                "markdown_body": clean_body,
                "citations": params.citations,
                "published_at": published_at,
            },
        )
        response.raise_for_status()
      return PublishExecutiveDigestOutput(
          status="success",
          delivery_id=delivery_id,
          target_channel=params.target_channel,
          published_at=published_at,
          delivery_mode="webhook",
      )
    except Exception as exc:
      _logger.warning("digest_webhook_publish_failed", error=redactor.scrub_text(str(exc)))
      return PublishExecutiveDigestOutput(
          status="error",
          target_channel=params.target_channel,
          delivery_mode="error",
          recovery_advice=GuidedRecoveryAdvice(
              error_code="WEBHOOK_DELIVERY_FAILED",
              error_message=f"Webhook post failed: {type(exc).__name__}",
              recovery_instructions=(
                  "Verify DIGEST_WEBHOOK_URL in Secret Manager or archive locally without "
                  "external webhook delivery."
              ),
              retryable=True,
          ),
      )

  _logger.info(
      "executive_digest_archived_dry_run",
      delivery_id=delivery_id,
      target_channel=params.target_channel,
      citation_count=len(params.citations),
  )
  return PublishExecutiveDigestOutput(
      status="success",
      delivery_id=delivery_id,
      target_channel=params.target_channel,
      published_at=published_at,
      delivery_mode="dry_run_archive",
  )


def build_retrieve_archived_briefings_tool() -> StrictSchemaFunctionTool:
  """Builds the strict-schema archived briefing retrieval tool."""
  return StrictSchemaFunctionTool(
      func=retrieve_archived_signal_briefings,
      input_schema=ArchivedBriefingQueryInput,
      output_schema=ArchivedBriefingQueryOutput,
  )


def build_editorial_approval_tool() -> LongRunningFunctionTool:
  """Builds the ADK LongRunningFunctionTool for HITL editorial sign-off."""
  return LongRunningFunctionTool(func=request_human_editorial_publication_approval)


def build_publish_digest_tool() -> StrictSchemaFunctionTool:
  """Builds the strict-schema executive digest publication tool."""
  return StrictSchemaFunctionTool(
      func=publish_approved_executive_signal_digest,
      input_schema=PublishExecutiveDigestInput,
      output_schema=PublishExecutiveDigestOutput,
  )
