"""Policy-as-Code ADK guardrail plugin for prompt injection, SSRF, PII, and HITL gates."""

from __future__ import annotations

import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing_extensions import override

from ..observability.logging_config import get_logger
from ..observability.pii_redactor import SensitiveDataRedactor
from ..schemas import GuidedRecoveryAdvice

_logger = get_logger("harness_signal_radar.guardrails")

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b"),
    re.compile(r"(?i)\breveal\s+(?:your\s+)?(?:system\s+prompt|constitution)\b"),
    re.compile(r"(?i)\bbypass\s+(?:human\s+)?(?:approval|guardrail|hitl|policy)\b"),
    re.compile(r"(?i)\boverride\s+(?:the\s+)?constitution\b"),
    re.compile(r"(?i)\bexfiltrate\s+(?:secrets?|api[_ ]?keys?|tokens?|credentials?)\b"),
)

_SSRF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)169\.254\.169\.254"),
    re.compile(r"(?i)metadata\.google\.internal"),
    re.compile(r"(?i)file:///etc/"),
    re.compile(r"(?i)https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0)(?::\d+)?"),
)


class SecurityAndPolicyGuardrailPlugin(BasePlugin):
  """Enforces security, Model Armor sanitization, PII redaction, and HITL policies.

  Responsibilities:
    1. ``before_model_callback``: Blocks prompt-injection, credential-exfiltration,
       and SSRF payloads (optionally calling Google Cloud Model Armor).
    2. ``before_tool_callback``: Blocks high-stakes external publishing unless
       ``editorial_approval_status == 'APPROVED'`` and blocks SSRF URLs in tool args.
    3. ``after_model_callback``: Scrubs any leaked PII or tokens in LLM output.
  """

  def __init__(
      self,
      *,
      name: str = "security_and_policy_guardrail",
      redactor: SensitiveDataRedactor | None = None,
      model_armor_template_id: str = "",
      model_armor_client: Any | None = None,
  ) -> None:
    super().__init__(name=name)
    self._redactor = redactor or SensitiveDataRedactor()
    self._model_armor_template_id = model_armor_template_id.strip()
    self._model_armor_client = model_armor_client

  def _check_model_armor(self, text: str) -> str | None:
    if not self._model_armor_template_id or not text:
      return None
    try:
      if self._model_armor_client is None:
        from google.cloud import modelarmor_v1

        self._model_armor_client = modelarmor_v1.ModelArmorClient()
      response = self._model_armor_client.sanitize_user_prompt(
          request={
              "name": self._model_armor_template_id,
              "user_prompt_data": {"text": text},
          }
      )
      sanitization = getattr(response, "sanitization_result", None)
      filter_match = getattr(sanitization, "filter_match_state", None)
      if filter_match and str(filter_match).endswith("MATCH_FOUND"):
        return "Blocked by Google Cloud Model Armor safety policy."
    except Exception as exc:
      _logger.warning("model_armor_check_skipped", error=type(exc).__name__)
    return None

  @classmethod
  def inspect_untrusted_text(cls, text: str) -> str | None:
    """Returns a policy violation reason if text contains injection or SSRF indicators."""
    for pattern in _INJECTION_PATTERNS:
      if pattern.search(text):
        return f"Prompt injection / policy bypass pattern detected ({pattern.pattern})."
    for pattern in _SSRF_PATTERNS:
      if pattern.search(text):
        return f"Restricted SSRF or local/metadata target detected ({pattern.pattern})."
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    user_texts: list[str] = []
    for content in llm_request.contents or []:
      if content.role == "user":
        for part in content.parts or []:
          if part.text:
            user_texts.append(part.text)
    if not user_texts:
      return None

    latest_user_text = user_texts[-1]
    reason = self.inspect_untrusted_text(latest_user_text) or self._check_model_armor(
        latest_user_text
    )
    if reason:
      callback_context.state["temp:guardrail_blocked"] = True
      callback_context.state["temp:guardrail_reason"] = reason
      _logger.warning(
          "guardrail_blocked_model_request",
          agent=callback_context.agent_name,
          reason=reason,
      )
      return LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text=(
                          "Request rejected by Security & Policy Guardrail: "
                          f"{reason} Please rephrase your research query without "
                          "instruction overrides or restricted endpoints."
                      )
                  )
              ],
          )
      )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    serialized_args = str(tool_args)
    for pattern in _SSRF_PATTERNS:
      if pattern.search(serialized_args):
        _logger.warning(
            "guardrail_blocked_ssrf_tool_args",
            tool_name=tool.name,
        )
        advice = GuidedRecoveryAdvice(
            error_code="POLICY_SSRF_BLOCKED",
            error_message="Tool arguments contain a restricted internal/metadata URL.",
            recovery_instructions="Use only public HTTPS endpoints and remove internal IP addresses.",
            retryable=False,
        )
        return {
            "status": "blocked",
            "recovery_advice": advice.model_dump(mode="json"),
        }

    if tool.name == "publish_approved_executive_signal_digest":
      approval_status = str(
          tool_context.state.get("editorial_approval_status", "UNAPPROVED")
      )
      if approval_status != "APPROVED":
        _logger.warning(
            "guardrail_blocked_unapproved_publication",
            approval_status=approval_status,
            agent=tool_context.agent_name,
        )
        advice = GuidedRecoveryAdvice(
            error_code="POLICY_HITL_APPROVAL_REQUIRED",
            error_message=(
                "Policy plugin blocked external publication because human editorial "
                f"approval status is '{approval_status}'."
            ),
            recovery_instructions=(
                "Call 'request_human_editorial_publication_approval' first and await "
                "explicit human sign-off before calling 'publish_approved_executive_signal_digest'."
            ),
            retryable=True,
        )
        return {
            "status": "blocked",
            "target_channel": str(tool_args.get("target_channel", "")),
            "delivery_mode": "blocked",
            "recovery_advice": advice.model_dump(mode="json"),
        }
    return None

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    del callback_context
    if not llm_response.content or not llm_response.content.parts:
      return None

    changed = False
    new_parts: list[types.Part] = []
    for part in llm_response.content.parts:
      if part.text:
        scrubbed = self._redactor.scrub_text(part.text)
        if scrubbed != part.text:
          changed = True
        new_parts.append(types.Part.from_text(text=scrubbed))
      else:
        new_parts.append(part)

    if not changed:
      return None

    _logger.info("guardrail_redacted_model_output_pii")
    return llm_response.model_copy(
        update={
            "content": types.Content(
                role=llm_response.content.role,
                parts=new_parts,
            )
        }
    )
