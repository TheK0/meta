from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fsmol_cliff.release_artifacts import (
    build_failure_taxonomy_rows,
    build_main_table_rows,
    build_paired_model_comparison_rows,
)


def _write_aggregate(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def test_build_main_table_rows_extracts_requested_metrics(tmp_path: Path) -> None:
    path = tmp_path / "knn.aggregate.json"
    _write_aggregate(
        path,
        [
            {"profile": "relaxed", "result_tier": "final", "split_type": "standard", "metric": "q_psr", "score": 0.5, "ci_low": 0.4, "ci_high": 0.6, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "sq_psr", "score": 0.6, "ci_low": 0.5, "ci_high": 0.7, "num_tasks": 8},
        ],
    )

    rows = build_main_table_rows(
        model_to_aggregate_path={"kNN": path},
        profile="relaxed",
        metrics=[("standard", "q_psr", "std_q_psr"), ("adversarial", "sq_psr", "adv_sq_psr")],
    )

    assert rows == [{"model": "kNN", "std_q_psr": 0.5, "adv_sq_psr": 0.6}]


def test_build_failure_taxonomy_rows_emits_interpretation_fields(tmp_path: Path) -> None:
    path = tmp_path / "rf.aggregate.json"
    _write_aggregate(
        path,
        [
            {"profile": "relaxed", "result_tier": "final", "split_type": "standard", "metric": "q_psr", "score": 0.7, "ci_low": 0.6, "ci_high": 0.8, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "standard", "metric": "c_bacc", "score": 0.51, "ci_low": 0.5, "ci_high": 0.52, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "sq_psr", "score": 0.86, "ci_low": 0.8, "ci_high": 0.9, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": 0.50, "ci_low": 0.49, "ci_high": 0.51, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.93, "ci_low": 0.9, "ci_high": 0.95, "num_tasks": 8},
        ],
    )

    rows = build_failure_taxonomy_rows(
        model_to_aggregate_path={"RF": path},
        profile="relaxed",
    )

    assert rows[0]["model"] == "RF"
    assert rows[0]["taxonomy_label"] == "ranking-competent but decision-collapsed"
    assert rows[0]["ranking_signal"] == "high"
    assert rows[0]["decision_signal"] == "weak"
    assert rows[0]["collapse_signal"] == "high"


def test_build_paired_model_comparison_rows_computes_delta_and_ci() -> None:
    rows = build_paired_model_comparison_rows(
        model_to_task_result_rows={
            "kNN": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.9},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.8},
            ],
            "cliff-aware": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.7},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.6},
            ],
        },
        profile="relaxed",
        comparisons=[("kNN", "cliff-aware")],
        metrics=[("adversarial", "scr")],
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    assert rows[0]["baseline_model"] == "kNN"
    assert rows[0]["treatment_model"] == "cliff-aware"
    assert rows[0]["split_type"] == "adversarial"
    assert rows[0]["metric"] == "scr"
    assert rows[0]["delta_mean"] < 0


def test_build_paired_model_comparison_rows_ignores_nan_seed_scores_within_task() -> None:
    rows = build_paired_model_comparison_rows(
        model_to_task_result_rows={
            "baseline": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": 0.4},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": float("nan")},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": 0.5},
            ],
            "treatment": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": 0.6},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": float("nan")},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": 0.7},
            ],
        },
        profile="relaxed",
        comparisons=[("baseline", "treatment")],
        metrics=[("adversarial", "c_bacc")],
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    assert rows[0]["num_tasks"] == 2
    assert rows[0]["delta_mean"] == pytest.approx(0.2)
