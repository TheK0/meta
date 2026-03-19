from __future__ import annotations

import pytest

from fsmol_cliff.hypotheses import (
    compute_cliff_gap_metrics,
    validate_h1_model_set,
    validate_h2_shortcut_reliance,
    validate_h3_intervention,
)


def test_compute_cliff_gap_metrics_derives_gap_values() -> None:
    result = compute_cliff_gap_metrics(
        {
            "c_bacc": {"mean": 0.42},
            "nc_bacc": {"mean": 0.71},
            "q_psr": {"mean": 0.38},
            "nc_psr": {"mean": 0.66},
        }
    )

    assert result["cliffgap_bacc"] == pytest.approx(0.29)
    assert result["cliffgap_psr"] == pytest.approx(0.28)


def test_validate_h1_model_set_accepts_with_significant_gaps_and_ranking_disagreement() -> None:
    result = validate_h1_model_set(
        {
            "model_a": {
                "official": {"task_values": [0.80, 0.82, 0.81]},
                "c_bacc": {"task_values": [0.45, 0.47, 0.44]},
                "nc_bacc": {"task_values": [0.71, 0.74, 0.72]},
                "q_psr": {"task_values": [0.40, 0.42, 0.41]},
                "nc_psr": {"task_values": [0.69, 0.71, 0.70]},
            },
            "model_b": {
                "official": {"task_values": [0.75, 0.74, 0.76]},
                "c_bacc": {"task_values": [0.60, 0.59, 0.61]},
                "nc_bacc": {"task_values": [0.70, 0.69, 0.71]},
                "q_psr": {"task_values": [0.58, 0.57, 0.59]},
                "nc_psr": {"task_values": [0.68, 0.67, 0.69]},
            },
            "model_c": {
                "official": {"task_values": [0.70, 0.69, 0.71]},
                "c_bacc": {"task_values": [0.54, 0.53, 0.55]},
                "nc_bacc": {"task_values": [0.69, 0.68, 0.70]},
                "q_psr": {"task_values": [0.52, 0.51, 0.53]},
                "nc_psr": {"task_values": [0.67, 0.66, 0.68]},
            },
        },
        bootstrap_iterations=300,
        bootstrap_seed=3,
    )

    assert result["accepted"] is True
    assert result["gap_supported"] is True
    assert result["ranking_disagreement"] is True


def test_validate_h1_model_set_rejects_when_model_set_is_too_small() -> None:
    result = validate_h1_model_set(
        {
            "model_a": {
                "official": {"task_values": [0.80, 0.82]},
                "c_bacc": {"task_values": [0.45, 0.47]},
                "nc_bacc": {"task_values": [0.71, 0.74]},
                "q_psr": {"task_values": [0.40, 0.42]},
                "nc_psr": {"task_values": [0.69, 0.71]},
            },
            "model_b": {
                "official": {"task_values": [0.75, 0.74]},
                "c_bacc": {"task_values": [0.60, 0.59]},
                "nc_bacc": {"task_values": [0.70, 0.69]},
                "q_psr": {"task_values": [0.58, 0.57]},
                "nc_psr": {"task_values": [0.68, 0.67]},
            },
        },
        bootstrap_iterations=200,
        bootstrap_seed=1,
    )

    assert result["accepted"] is False
    assert "requires at least 3 models" in result["reason"]


def test_validate_h3_intervention_accepts_when_cliff_improves_and_controls_hold() -> None:
    result = validate_h3_intervention(
        baseline={
            "official": {"task_values": [0.60, 0.62, 0.61]},
            "c_bacc": {"task_values": [0.40, 0.42, 0.41]},
            "q_psr": {"task_values": [0.45, 0.44, 0.46]},
            "sq_psr": {"task_values": [0.35, 0.34, 0.36]},
            "nc_bacc": {"task_values": [0.70, 0.69, 0.71]},
            "nc_psr": {"task_values": [0.68, 0.69, 0.67]},
            "scr": {"task_values": [0.40, 0.41, 0.39]},
            "ss_scr": {"task_values": [0.46, 0.47, 0.45]},
            "ss_q_psr": {"task_values": [0.30, 0.29, 0.31]},
            "ss_sq_psr": {"task_values": [0.25, 0.24, 0.26]},
        },
        treatment={
            "official": {"task_values": [0.62, 0.63, 0.64]},
            "c_bacc": {"task_values": [0.55, 0.56, 0.57]},
            "q_psr": {"task_values": [0.58, 0.57, 0.59]},
            "sq_psr": {"task_values": [0.49, 0.48, 0.50]},
            "nc_bacc": {"task_values": [0.71, 0.72, 0.70]},
            "nc_psr": {"task_values": [0.69, 0.70, 0.68]},
            "scr": {"task_values": [0.22, 0.21, 0.23]},
            "ss_scr": {"task_values": [0.26, 0.25, 0.27]},
            "ss_q_psr": {"task_values": [0.44, 0.45, 0.43]},
            "ss_sq_psr": {"task_values": [0.39, 0.40, 0.38]},
        },
        bootstrap_iterations=300,
        bootstrap_seed=7,
    )

    assert result["accepted"] is True
    assert result["cliff_improved"] is True
    assert result["collapse_reduced"] is True
    assert result["controls_preserved"] is True


def test_validate_h3_intervention_rejects_when_noncliff_degrades() -> None:
    result = validate_h3_intervention(
        baseline={
            "official": {"task_values": [0.60, 0.62, 0.61]},
            "c_bacc": {"task_values": [0.40, 0.42, 0.41]},
            "q_psr": {"task_values": [0.45, 0.44, 0.46]},
            "nc_bacc": {"task_values": [0.70, 0.69, 0.71]},
            "nc_psr": {"task_values": [0.68, 0.69, 0.67]},
            "scr": {"task_values": [0.40, 0.41, 0.39]},
            "ss_scr": {"task_values": [0.46, 0.47, 0.45]},
            "ss_q_psr": {"task_values": [0.30, 0.29, 0.31]},
        },
        treatment={
            "official": {"task_values": [0.62, 0.63, 0.64]},
            "c_bacc": {"task_values": [0.55, 0.56, 0.57]},
            "q_psr": {"task_values": [0.58, 0.57, 0.59]},
            "nc_bacc": {"task_values": [0.50, 0.49, 0.51]},
            "nc_psr": {"task_values": [0.48, 0.49, 0.47]},
            "scr": {"task_values": [0.22, 0.21, 0.23]},
            "ss_scr": {"task_values": [0.26, 0.25, 0.27]},
            "ss_q_psr": {"task_values": [0.44, 0.45, 0.43]},
        },
        bootstrap_iterations=300,
        bootstrap_seed=9,
    )

    assert result["accepted"] is False
    assert result["controls_preserved"] is False


def test_validate_h2_shortcut_reliance_accepts_when_two_of_three_evidences_hold() -> None:
    result = validate_h2_shortcut_reliance(
        {
            "scr": {"mean": 0.45},
            "ss_scr": {"mean": 0.60},
            "q_psr": {"mean": 0.55},
            "ss_q_psr": {"mean": 0.40},
            "baseline": {
                "scr": {"task_values": [0.55, 0.56, 0.54]},
                "q_psr": {"task_values": [0.40, 0.41, 0.39]},
            },
            "treatment": {
                "scr": {"task_values": [0.30, 0.31, 0.29]},
                "q_psr": {"task_values": [0.58, 0.57, 0.59]},
            },
        },
        bootstrap_iterations=300,
        bootstrap_seed=5,
    )

    assert result["accepted"] is True
    assert result["num_supported_conditions"] >= 2


def test_validate_h2_shortcut_reliance_rejects_when_evidence_is_weak() -> None:
    result = validate_h2_shortcut_reliance(
        {
            "scr": {"mean": 0.0},
            "ss_scr": {"mean": 0.0},
            "q_psr": {"mean": 0.55},
            "ss_q_psr": {"mean": 0.56},
        },
        bootstrap_iterations=200,
        bootstrap_seed=2,
    )

    assert result["accepted"] is False
