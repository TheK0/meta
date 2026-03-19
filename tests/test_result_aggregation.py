from __future__ import annotations

import pytest

from fsmol_cliff.aggregate import aggregate_task_result_rows


def test_aggregate_task_result_rows_averages_across_seeds_then_tasks() -> None:
    rows = [
        {"task_id": "t1", "seed": 0, "split_type": "standard", "metric": "q_psr", "score": 0.5},
        {"task_id": "t1", "seed": 1, "split_type": "standard", "metric": "q_psr", "score": 0.7},
        {"task_id": "t2", "seed": 0, "split_type": "standard", "metric": "q_psr", "score": 0.3},
        {"task_id": "t2", "seed": 1, "split_type": "standard", "metric": "q_psr", "score": 0.5},
    ]

    aggregated = aggregate_task_result_rows(rows, bootstrap_iterations=200, bootstrap_seed=3)
    result = aggregated[0]

    assert result["split_type"] == "standard"
    assert result["metric"] == "q_psr"
    assert result["score"] == pytest.approx(0.5)
    assert result["num_tasks"] == 2
    assert result["ci_low"] <= result["score"] <= result["ci_high"]
