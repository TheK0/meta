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


def test_build_main_table_rows_filters_by_result_tier_and_defaults_to_final(tmp_path: Path) -> None:
    path = tmp_path / "knn.aggregate.json"
    _write_aggregate(
        path,
        [
            {"profile": "relaxed", "result_tier": "final", "split_type": "standard", "metric": "q_psr", "score": 0.5, "ci_low": 0.4, "ci_high": 0.6, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "standard", "metric": "q_psr", "score": 0.9, "ci_low": 0.8, "ci_high": 1.0, "num_tasks": 8},
        ],
    )

    final_rows = build_main_table_rows(
        model_to_aggregate_path={"kNN": path},
        profile="relaxed",
        metrics=[("standard", "q_psr", "std_q_psr")],
    )
    intermediate_rows = build_main_table_rows(
        model_to_aggregate_path={"kNN": path},
        profile="relaxed",
        result_tier="intermediate",
        metrics=[("standard", "q_psr", "std_q_psr")],
    )

    assert final_rows == [{"model": "kNN", "std_q_psr": 0.5}]
    assert intermediate_rows == [{"model": "kNN", "std_q_psr": 0.9}]


def test_build_main_table_rows_treats_missing_result_tier_as_final(tmp_path: Path) -> None:
    path = tmp_path / "knn.aggregate.json"
    _write_aggregate(
        path,
        [
            {"profile": "relaxed", "split_type": "standard", "metric": "q_psr", "score": 0.4, "ci_low": 0.3, "ci_high": 0.5, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "standard", "metric": "q_psr", "score": 0.9, "ci_low": 0.8, "ci_high": 1.0, "num_tasks": 8},
        ],
    )

    rows = build_main_table_rows(
        model_to_aggregate_path={"kNN": path},
        profile="relaxed",
        metrics=[("standard", "q_psr", "std_q_psr")],
    )

    assert rows == [{"model": "kNN", "std_q_psr": 0.4}]


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


def test_build_failure_taxonomy_rows_filters_by_result_tier(tmp_path: Path) -> None:
    path = tmp_path / "rf.aggregate.json"
    _write_aggregate(
        path,
        [
            {"profile": "relaxed", "result_tier": "final", "split_type": "standard", "metric": "q_psr", "score": 0.7, "ci_low": 0.6, "ci_high": 0.8, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "standard", "metric": "c_bacc", "score": 0.51, "ci_low": 0.5, "ci_high": 0.52, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "sq_psr", "score": 0.86, "ci_low": 0.8, "ci_high": 0.9, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "c_bacc", "score": 0.5, "ci_low": 0.49, "ci_high": 0.51, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.93, "ci_low": 0.9, "ci_high": 0.95, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "standard", "metric": "q_psr", "score": 0.4, "ci_low": 0.3, "ci_high": 0.5, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "standard", "metric": "c_bacc", "score": 0.6, "ci_low": 0.59, "ci_high": 0.61, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "sq_psr", "score": 0.45, "ci_low": 0.4, "ci_high": 0.5, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "c_bacc", "score": 0.61, "ci_low": 0.6, "ci_high": 0.62, "num_tasks": 8},
            {"profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.7, "ci_low": 0.65, "ci_high": 0.75, "num_tasks": 8},
        ],
    )

    final_rows = build_failure_taxonomy_rows(
        model_to_aggregate_path={"baseline": path},
        profile="relaxed",
    )
    intermediate_rows = build_failure_taxonomy_rows(
        model_to_aggregate_path={"baseline": path},
        profile="relaxed",
        result_tier="intermediate",
    )

    assert final_rows[0]["taxonomy_label"] == "ranking-competent but decision-collapsed"
    assert final_rows[0]["ranking_signal"] == "high"
    assert final_rows[0]["decision_signal"] == "weak"
    assert final_rows[0]["collapse_signal"] == "high"

    assert intermediate_rows[0]["taxonomy_label"] == "mixed"
    assert intermediate_rows[0]["ranking_signal"] == "low"
    assert intermediate_rows[0]["decision_signal"] == "strong"
    assert intermediate_rows[0]["collapse_signal"] == "low"


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


def test_build_paired_model_comparison_rows_filters_by_result_tier() -> None:
    rows = build_paired_model_comparison_rows(
        model_to_task_result_rows={
            "baseline": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.9},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.8},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.2},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.1},
            ],
            "treatment": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.7},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.6},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.5},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.4},
            ],
        },
        profile="relaxed",
        comparisons=[("baseline", "treatment")],
        metrics=[("adversarial", "scr")],
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    intermediate_rows = build_paired_model_comparison_rows(
        model_to_task_result_rows={
            "baseline": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.9},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.8},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.2},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.1},
            ],
            "treatment": [
                {"task_id": "t1", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.7},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "final", "split_type": "adversarial", "metric": "scr", "score": 0.6},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.5},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.4},
            ],
        },
        profile="relaxed",
        result_tier="intermediate",
        comparisons=[("baseline", "treatment")],
        metrics=[("adversarial", "scr")],
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    assert rows[0]["result_tier"] == "final"
    assert rows[0]["delta_mean"] == pytest.approx(-0.2)
    assert intermediate_rows[0]["result_tier"] == "intermediate"
    assert intermediate_rows[0]["delta_mean"] == pytest.approx(0.3)


def test_build_paired_model_comparison_rows_treats_missing_result_tier_as_final() -> None:
    rows = build_paired_model_comparison_rows(
        model_to_task_result_rows={
            "baseline": [
                {"task_id": "t1", "profile": "relaxed", "split_type": "adversarial", "metric": "scr", "score": 0.9},
                {"task_id": "t2", "profile": "relaxed", "split_type": "adversarial", "metric": "scr", "score": 0.8},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.2},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.1},
            ],
            "treatment": [
                {"task_id": "t1", "profile": "relaxed", "split_type": "adversarial", "metric": "scr", "score": 0.7},
                {"task_id": "t2", "profile": "relaxed", "split_type": "adversarial", "metric": "scr", "score": 0.6},
                {"task_id": "t1", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.5},
                {"task_id": "t2", "profile": "relaxed", "result_tier": "intermediate", "split_type": "adversarial", "metric": "scr", "score": 0.4},
            ],
        },
        profile="relaxed",
        comparisons=[("baseline", "treatment")],
        metrics=[("adversarial", "scr")],
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    assert rows[0]["result_tier"] == "final"
    assert rows[0]["delta_mean"] == pytest.approx(-0.2)
