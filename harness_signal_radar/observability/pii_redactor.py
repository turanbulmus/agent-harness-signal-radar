"""Active PII, credential, and internal network redaction utilities."""

from __future__ import annotations

import re
from typing import Any


def _passes_luhn(digits: str) -> bool:
  total = 0
  reverse_digits = digits[::-1]
  for idx, ch in enumerate(reverse_digits):
    n = ord(ch) - ord("0")
    if idx % 2 == 1:
      n *= 2
      if n > 9:
        n -= 9
    total += n
  return total % 10 == 0


_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Google API keys, OAuth tokens, GitHub tokens, OpenAI/Anthropic keys
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\bya29\.[0-9A-Za-z\-_]{20,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{16,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9\-_]{16,}\b"), "[REDACTED_SECRET]"),
    # Authorization Bearer tokens
    (
        re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9\-._~+/]{12,}=*"),
        r"\1[REDACTED_SECRET]",
    ),
    # Generic secret key-value assignments
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
            r"secret|password)\s*([:=])\s*(['\"]?)([A-Za-z0-9\-._~+/]{8,})\3"
        ),
        r"\1\2\3[REDACTED_SECRET]\3",
    ),
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
)
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)")
_INTERNAL_HOST_RE = re.compile(
    r"\b(?:169\.254\.169\.254|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"
)

_SENSITIVE_FIELD_KEYS = frozenset({
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "auth_token",
    "client_secret",
    "password",
    "secret",
    "token",
    "ssn",
    "phone",
    "phone_number",
    "email",
    "email_address",
    "x-api-key",
})


class SensitiveDataRedactor:
  """Actively scrubs PII, secrets, and sensitive metadata before logging or storage.

  Supports deterministic local pattern scrubbing in all environments and optional
  Google Cloud Sensitive Data Protection (Cloud DLP) de-identification when
  configured in GCP.
  """

  def __init__(
      self,
      *,
      enable_cloud_dlp: bool = False,
      project_id: str | None = None,
      dlp_client: Any | None = None,
  ) -> None:
    self.enable_cloud_dlp = enable_cloud_dlp
    self.project_id = project_id
    self._dlp_client = dlp_client

  def _redact_with_cloud_dlp(self, text: str) -> str:
    if not self.enable_cloud_dlp or not self.project_id or not text:
      return text
    try:
      if self._dlp_client is None:
        from google.cloud import dlp_v2

        self._dlp_client = dlp_v2.DlpServiceClient()
      parent = f"projects/{self.project_id}/locations/global"
      inspect_config = {
          "info_types": [
              {"name": "EMAIL_ADDRESS"},
              {"name": "PHONE_NUMBER"},
              {"name": "US_SOCIAL_SECURITY_NUMBER"},
              {"name": "CREDIT_CARD_NUMBER"},
              {"name": "AUTH_TOKEN"},
          ]
      }
      deidentify_config = {
          "info_type_transformations": {
              "transformations": [
                  {
                      "primitive_transformation": {
                          "replace_with_info_type_config": {}
                      }
                  }
              ]
          }
      }
      response = self._dlp_client.deidentify_content(
          request={
              "parent": parent,
              "deidentify_config": deidentify_config,
              "inspect_config": inspect_config,
              "item": {"value": text},
          }
      )
      return response.item.value
    except Exception:
      # Never fail logging or persistence if Cloud DLP is temporarily unreachable.
      return text

  @staticmethod
  def _scrub_card_numbers(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
      raw = match.group(0)
      digits = "".join(ch for ch in raw if ch.isdigit())
      if 13 <= len(digits) <= 16 and _passes_luhn(digits):
        return "[REDACTED_CARD]"
      return raw

    return _CARD_CANDIDATE_RE.sub(_replace, text)

  def scrub_text(self, text: str) -> str:
    """Redacts secrets, PII, and internal IP addresses from arbitrary text."""
    if not text:
      return text

    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
      redacted = pattern.sub(replacement, redacted)

    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _SSN_RE.sub("[REDACTED_SSN]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = self._scrub_card_numbers(redacted)
    redacted = _INTERNAL_HOST_RE.sub("[REDACTED_INTERNAL_IP]", redacted)
    redacted = self._redact_with_cloud_dlp(redacted)
    return redacted

  def scrub_payload(self, value: Any, *, key_hint: str | None = None) -> Any:
    """Recursively scrubs nested dictionaries, lists, tuples, and strings."""
    if key_hint and key_hint.strip().lower() in _SENSITIVE_FIELD_KEYS:
      if isinstance(value, str) and value:
        lowered = key_hint.strip().lower()
        if "email" in lowered:
          return "[REDACTED_EMAIL]"
        if "phone" in lowered:
          return "[REDACTED_PHONE]"
        if "ssn" in lowered:
          return "[REDACTED_SSN]"
        return "[REDACTED_SECRET]"

    if isinstance(value, str):
      return self.scrub_text(value)
    if isinstance(value, dict):
      return {
          str(k): self.scrub_payload(v, key_hint=str(k))
          for k, v in value.items()
      }
    if isinstance(value, list):
      return [self.scrub_payload(item) for item in value]
    if isinstance(value, tuple):
      return [self.scrub_payload(item) for item in value]
    return value
