"""Secure secret injection backed by Google Cloud Secret Manager."""

from __future__ import annotations

import os
import threading
import time
from typing import Any


class SecretResolutionError(RuntimeError):
  """Raised when a required secret cannot be securely resolved."""


class SecretManagerResolver:
  """Resolves runtime credentials from Google Cloud Secret Manager.

  Secrets are cached in memory for a bounded TTL and never serialized into
  string representations or structured logs. When running locally with
  ``use_secret_manager=False``, environment variables act as the runtime
  injection source without hardcoding any fallback credentials in code.
  """

  def __init__(
      self,
      *,
      project_id: str | None = None,
      default_version: str = "latest",
      ttl_seconds: int = 300,
      use_secret_manager: bool = False,
      client: Any | None = None,
  ) -> None:
    self.project_id = project_id
    self.default_version = default_version
    self.ttl_seconds = max(1, ttl_seconds)
    self.use_secret_manager = use_secret_manager
    self._client = client
    self._cache: dict[tuple[str, str], tuple[str, float]] = {}
    self._lock = threading.Lock()

  def __repr__(self) -> str:
    return (
        f"SecretManagerResolver(project_id={self.project_id!r}, "
        f"use_secret_manager={self.use_secret_manager!r}, "
        f"cached_secret_count={len(self._cache)})"
    )

  def _get_client(self) -> Any:
    if self._client is None:
      try:
        from google.cloud import secretmanager
      except ImportError as exc:
        raise SecretResolutionError(
            "google-cloud-secret-manager is required when USE_SECRET_MANAGER=true."
        ) from exc
      self._client = secretmanager.SecretManagerServiceClient()
    return self._client

  def clear_cache(self) -> None:
    """Clears all cached secret values from memory."""
    with self._lock:
      self._cache.clear()

  def access_secret(
      self,
      secret_id: str,
      *,
      env_fallback_var: str | None = None,
      version: str | None = None,
      required: bool = False,
  ) -> str | None:
    """Retrieves a secret value from Secret Manager or runtime environment.

    Args:
      secret_id: The Google Cloud Secret Manager secret identifier.
      env_fallback_var: Environment variable name checked when Secret Manager is
        disabled or when running in local development mode.
      version: Secret version identifier (defaults to ``self.default_version``).
      required: Whether missing credentials should raise ``SecretResolutionError``.

    Returns:
      The resolved secret string, or ``None`` if optional and unavailable.
    """
    resolved_version = version or self.default_version
    cache_key = (secret_id, resolved_version)
    now = time.monotonic()

    with self._lock:
      cached = self._cache.get(cache_key)
      if cached is not None:
        secret_value, expires_at = cached
        if now < expires_at:
          return secret_value
        self._cache.pop(cache_key, None)

    secret_value: str | None = None
    if self.use_secret_manager:
      if not self.project_id:
        if env_fallback_var and not required:
          secret_value = os.environ.get(env_fallback_var) or None
        else:
          raise SecretResolutionError(
              "GOOGLE_CLOUD_PROJECT must be set to access Secret Manager secret "
              f"'{secret_id}'."
          )
      else:
        secret_path = (
            f"projects/{self.project_id}/secrets/{secret_id}/versions/{resolved_version}"
        )
        try:
          client = self._get_client()
          response = client.access_secret_version(request={"name": secret_path})
          payload = response.payload.data
          secret_value = (
              payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
          ).strip()
        except Exception as exc:
          env_val = os.environ.get(env_fallback_var) if env_fallback_var else None
          if env_val:
            secret_value = env_val.strip()
          elif required:
            raise SecretResolutionError(
                f"Failed to access Secret Manager secret '{secret_path}': {exc}"
            ) from exc
          else:
            secret_value = None
    elif env_fallback_var:
      raw_env = os.environ.get(env_fallback_var, "").strip()
      secret_value = raw_env or None

    if not secret_value:
      if required:
        hint = f" or environment variable '{env_fallback_var}'" if env_fallback_var else ""
        raise SecretResolutionError(
            f"Missing required secret '{secret_id}'. Provide it via Google Cloud "
            f"Secret Manager{hint}."
        )
      return None

    with self._lock:
      self._cache[cache_key] = (secret_value, now + self.ttl_seconds)
    return secret_value


def resolve_required_secret(
    resolver: SecretManagerResolver,
    secret_id: str,
    *,
    env_fallback_var: str | None = None,
    version: str | None = None,
) -> str:
  """Resolves a mandatory secret or raises ``SecretResolutionError``."""
  value = resolver.access_secret(
      secret_id,
      env_fallback_var=env_fallback_var,
      version=version,
      required=True,
  )
  assert value is not None
  return value


def resolve_optional_secret(
    resolver: SecretManagerResolver,
    secret_id: str,
    *,
    env_fallback_var: str | None = None,
    version: str | None = None,
) -> str | None:
  """Resolves an optional secret, returning ``None`` when unconfigured."""
  return resolver.access_secret(
      secret_id,
      env_fallback_var=env_fallback_var,
      version=version,
      required=False,
  )
