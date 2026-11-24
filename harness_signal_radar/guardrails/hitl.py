"""Human-in-the-Loop (HITL) pause/resume gate for high-stakes external publishing."""

from __future__ import annotations

from typing import Any

from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow import FunctionNode

from ..observability.logging_config import get_logger
from ..schemas import EditorialApprovalDecision

_logger = get_logger("harness_signal_radar.guardrails.hitl")


def _parse_decision(raw_response: Any, default_channel: str) -> EditorialApprovalDecision:
  if isinstance(raw_response, dict):
    candidate = (
        raw_response["result"]
        if "result" in raw_response and "approved" not in raw_response
        else raw_response
    )
    if isinstance(candidate, dict):
      return EditorialApprovalDecision.model_validate(
          {
              "approved": bool(candidate.get("approved", False)),
              "reviewer_note": str(candidate.get("reviewer_note", "")),
              "approved_channel": str(
                  candidate.get("approved_channel") or default_channel
              ),
          }
      )
    if isinstance(candidate, bool):
      return EditorialApprovalDecision(
          approved=candidate, approved_channel=default_channel
      )
    if isinstance(candidate, str):
      normalized = candidate.strip().lower()
      approved = normalized in {"yes", "y", "approve", "approved", "true", "publish"}
      return EditorialApprovalDecision(
          approved=approved,
          reviewer_note=candidate,
          approved_channel=default_channel,
      )
  return EditorialApprovalDecision(approved=False, approved_channel=default_channel)


async def editorial_hitl_gate(ctx: Context, node_input: dict[str, Any]):
  """Interrupts workflow execution before external publishing until a human confirms.

  Routes to:
    - ``"publish"`` when human editorial sign-off explicitly approves external release.
    - ``"archive_only"`` when external publication is not requested or was rejected.
  """
  require_publication = bool(ctx.state.get("require_external_publication", False))
  target_channel = str(ctx.state.get("target_channel") or "engineering_digest")

  if not require_publication:
    yield Event(
        output=node_input,
        route="archive_only",
        state={"editorial_approval_status": "NOT_REQUIRED"},
    )
    return

  approval_round = int(ctx.state.get("editorial_approval_round", 0))
  interrupt_id = f"editorial_approval_{approval_round}"
  raw_response = ctx.resume_inputs.get(interrupt_id)

  if raw_response is None:
    headline = str(node_input.get("headline") or "Agent Harness Signal Briefing")
    citations = node_input.get("citations", [])
    _logger.info(
        "hitl_editorial_interrupt_raised",
        interrupt_id=interrupt_id,
        target_channel=target_channel,
        citation_count=len(citations) if isinstance(citations, list) else 0,
    )
    yield RequestInput(
        interrupt_id=interrupt_id,
        message=(
            f"Human editorial approval required before publishing '{headline}' "
            f"to '{target_channel}'. Confirm approval to proceed."
        ),
        payload={
            "headline": headline,
            "target_channel": target_channel,
            "citations_count": len(citations) if isinstance(citations, list) else 0,
            "executive_summary": str(node_input.get("executive_summary", ""))[:400],
        },
        response_schema=EditorialApprovalDecision,
    )
    return

  decision = _parse_decision(raw_response, default_channel=target_channel)
  status = "APPROVED" if decision.approved else "REJECTED"
  route = "publish" if decision.approved else "archive_only"
  _logger.info(
      "hitl_editorial_decision_recorded",
      interrupt_id=interrupt_id,
      status=status,
      route=route,
      channel=decision.approved_channel,
  )
  yield Event(
      output={
          **node_input,
          "editorial_decision": decision.model_dump(mode="json"),
      },
      route=route,
      state={
          "editorial_approval_status": status,
          "editorial_approval_round": approval_round + 1,
          "editorial_reviewer_note": decision.reviewer_note,
          "target_channel": decision.approved_channel,
      },
  )


def build_editorial_hitl_node() -> FunctionNode:
  """Constructs the resumable ADK FunctionNode for HITL publication gating."""
  return FunctionNode(
      func=editorial_hitl_gate,
      name="editorial_hitl_gate",
      rerun_on_resume=True,
  )
