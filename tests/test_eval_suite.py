"""Tests executing the golden dataset evaluation regression harness."""

from __future__ import annotations

from eval.run_evaluation import evaluate_golden_dataset


async def test_golden_dataset_regression_suite_passes_all_cases():
  summary = await evaluate_golden_dataset()
  assert summary.total_cases >= 8
  assert summary.passed_cases == summary.total_cases
  assert summary.overall_score == 1.0
