from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsmol_cliff.protocol_compare import write_protocol_comparison_json


def test_protocol_compare_writes_paired_rows(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.parquet"
    treatment_path = tmp_path / "treatment.parquet"
    output_path = tmp_path / "comparison.json"

    import pandas as pd

    pd.DataFrame(
        [
            {
                "task_id": "t1",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "c_bacc",
                "score": 0.50,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t2",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "c_bacc",
                "score": 0.52,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t1",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "scr",
                "score": 0.90,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t2",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "scr",
                "score": 0.91,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t1",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "c_bacc",
                "score": 0.40,
                "profile": "relaxed_covext_10_10",
                "result_tier": "final",
            },
        ]
    ).to_parquet(baseline_path, index=False)
    pd.DataFrame(
        [
            {
                "task_id": "t1",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "c_bacc",
                "score": 0.60,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t2",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "c_bacc",
                "score": 0.62,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t1",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "scr",
                "score": 0.80,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
            {
                "task_id": "t2",
                "seed": 0,
                "split_type": "adversarial",
                "metric": "scr",
                "score": 0.81,
                "profile": "relaxed_covext_10_10",
                "result_tier": "intermediate",
            },
        ]
    ).to_parquet(treatment_path, index=False)

    write_protocol_comparison_json(
        output_path=output_path,
        model_to_path={
            "baseline": baseline_path,
            "treatment": treatment_path,
        },
        comparisons=[("baseline", "treatment")],
        profile="relaxed_covext_10_10",
        result_tier="intermediate",
        metrics=[("adversarial", "c_bacc"), ("adversarial", "scr")],
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    rows = json.loads(output_path.read_text())
    assert len(rows) == 2
    assert rows[0]["baseline_model"] == "baseline"
    assert rows[0]["treatment_model"] == "treatment"
    assert rows[0]["profile"] == "relaxed_covext_10_10"
    assert rows[0]["result_tier"] == "intermediate"
    assert rows[0]["metric"] == "c_bacc"
    assert rows[0]["delta_mean"] == pytest.approx(0.1)
    assert rows[1]["metric"] == "scr"
    assert rows[1]["delta_mean"] < 0
