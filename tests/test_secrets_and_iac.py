"""Tests for Secret Manager injection, public-repo hygiene, and Terraform IaC."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from harness_signal_radar.config.secrets import (
  SecretManagerResolver,
  SecretResolutionError,
  resolve_required_secret,
)


def test_secret_manager_resolver_accesses_version_caches_and_redacts_repr():
  mock_client = MagicMock()
  mock_client.access_secret_version.return_value = MagicMock(
      payload=MagicMock(data=b"super-secret-token-value\n")
  )

  resolver = SecretManagerResolver(
      project_id="my-gcp-project",
      default_version="latest",
      ttl_seconds=60,
      use_secret_manager=True,
      client=mock_client,
  )

  first = resolve_required_secret(resolver, "github-pat-token")
  second = resolve_required_secret(resolver, "github-pat-token")

  assert first == "super-secret-token-value"
  assert second == "super-secret-token-value"
  mock_client.access_secret_version.assert_called_once_with(
      request={"name": "projects/my-gcp-project/secrets/github-pat-token/versions/latest"}
  )
  assert "super-secret-token-value" not in repr(resolver)


def test_secret_manager_resolver_raises_on_missing_required_secret(
    monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.delenv("MISSING_SECRET_ENV", raising=False)
  resolver = SecretManagerResolver(use_secret_manager=False)
  with pytest.raises(SecretResolutionError):
    resolve_required_secret(
        resolver,
        "missing-secret",
        env_fallback_var="MISSING_SECRET_ENV",
    )


def test_repo_contains_no_hardcoded_secrets_or_internal_corp_references():
  repo_root = Path(__file__).resolve().parent.parent
  forbidden_patterns = [
      re.compile(r"AIza[0-9A-Za-z\-_]{30,}"),
      re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
      re.compile(r"gho_[A-Za-z0-9_]{20,}"),
      re.compile(r"corp\.google\.com"),
      re.compile(r"googleplex\.com"),
      re.compile(r"/google/src/"),
  ]
  for path in repo_root.rglob("*"):
    if not path.is_file():
      continue
    if any(
        part in {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        for part in path.parts
    ):
      continue
    # Skip self-reference in this test file
    if path.name == "test_secrets_and_iac.py":
      continue
    content = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in forbidden_patterns:
      assert not pattern.search(content), f"Forbidden pattern {pattern.pattern} found in {path}"


def test_terraform_iac_and_ci_files_exist_with_expected_resources():
  repo_root = Path(__file__).resolve().parent.parent
  tf_dir = repo_root / "infra" / "terraform"
  for filename in (
      "versions.tf",
      "variables.tf",
      "main.tf",
      "iam.tf",
      "secrets.tf",
      "outputs.tf",
      "terraform.tfvars.example",
  ):
    assert (tf_dir / filename).is_file(), f"Missing Terraform file: {filename}"

  main_tf = (tf_dir / "main.tf").read_text(encoding="utf-8")
  secrets_tf = (tf_dir / "secrets.tf").read_text(encoding="utf-8")
  assert "google_cloud_run_v2_service" in main_tf
  assert "google_storage_bucket" in main_tf
  assert "google_secret_manager_secret" in secrets_tf
  assert "roles/secretmanager.secretAccessor" in secrets_tf

  ci_yml = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
  assert "pytest" in ci_yml
  assert "ruff check" in ci_yml
  assert "eval/run_evaluation.py" in ci_yml
  assert "terraform validate" in ci_yml
