from __future__ import annotations

import pytest

from fsmol_cliff.aggregate import paired_bootstrap_delta_ci, task_bootstrap_ci


def test_task_bootstrap_ci_returns_ordered_interval() -> None:
    ci = task_bootstrap_ci([0.2, 0.6, 0.8], iterations=500, seed=7)

    assert ci["mean"] == 0.5333333333333333
    assert ci["low"] <= ci["mean"] <= ci["high"]


def test_paired_bootstrap_delta_ci_uses_task_level_deltas() -> None:
    ci = paired_bootstrap_delta_ci(
        baseline=[0.4, 0.5, 0.6],
        treatment=[0.6, 0.7, 0.8],
        iterations=500,
        seed=11,
    )

    assert ci["delta_mean"] == pytest.approx(0.2)
    assert ci["low"] <= ci["delta_mean"] <= ci["high"]
