from __future__ import annotations

from pathlib import Path

import pandas as pd

from fsmol_cliff.protonet_boundary_calibration import apply_boundary_uncertainty_calibration


def _make_episode() -> dict:
    return {
        "task_id": "CHEMBL1",
        "split_type": "adversarial",
        "seed": 0,
        "episode_id": 0,
        "support_pos_ids": ["a1", "a2"],
        "support_neg_ids": ["n1", "n2"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
    }


def _make_assay_context() -> dict:
    return {
        "labels": {
            "a1": 1, "a2": 1, "n1": 0, "n2": 0,
            "qa": 1, "qn": 0,
        },
        "cliff_pairs": [
            {"anchor_id": "a1", "neg_id": "n1", "sim": 0.92, "gap_abs": 1.5, "pair_type": "cliff"},
        ],
        "noncliff_pairs": [
            {"anchor_id": "a2", "neg_id": "n2", "sim": 0.88, "gap_abs": 0.5, "pair_type": "highsim_noncliff"},
        ],
    }


def _make_raw_scores_and_margins() -> tuple[dict[str, float], dict[str, float]]:
    raw_scores = {"a1": 0.80, "a2": 0.75, "n1": 0.20, "n2": 0.30, "qa": 0.52, "qn": 0.48}
    raw_margins = {mid: score - 0.5 for mid, score in raw_scores.items()}
    return raw_scores, raw_margins


def test_boundary_uncertainty_calibration_returns_expected_bundle_keys() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    raw_scores, raw_margins = _make_raw_scores_and_margins()
    bundle = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
    )
    assert set(bundle) == {
        "raw_scores",
        "calibrated_scores",
        "raw_margins",
        "calibrated_margins",
        "uncertainty_summary",
    }
    assert len(bundle["calibrated_scores"]) == len(raw_scores)
    assert len(bundle["uncertainty_summary"]) == len(raw_scores)


def test_boundary_uncertainty_calibration_is_identity_at_zero_uncertainty() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    raw_scores = {"a1": 0.90, "a2": 0.85, "n1": 0.10, "n2": 0.15, "qa": 0.55, "qn": 0.45}
    raw_margins = {mid: score - 0.5 for mid, score in raw_scores.items()}
    bundle = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
        uncertainty_scale=0.0,
    )
    for mid in raw_margins:
        assert abs(bundle["calibrated_margins"][mid] - bundle["raw_margins"][mid]) < 1e-10
        assert abs(bundle["calibrated_scores"][mid] - bundle["raw_scores"][mid]) < 1e-10


def test_boundary_uncertainty_calibration_only_shrinks_margin_magnitude() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    raw_scores, raw_margins = _make_raw_scores_and_margins()
    bundle = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
        uncertainty_scale=0.5,
    )
    for mid, raw_margin in raw_margins.items():
        calibrated_margin = bundle["calibrated_margins"][mid]
        assert abs(calibrated_margin) <= abs(raw_margin) + 1e-10
        if raw_margin > 0:
            assert calibrated_margin >= 0, f"Positive margin sign not preserved for {mid}"
        elif raw_margin < 0:
            assert calibrated_margin <= 0, f"Negative margin sign not preserved for {mid}"


def test_boundary_uncertainty_calibration_uncertainty_is_bounded() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    raw_scores, raw_margins = _make_raw_scores_and_margins()
    bundle = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
    )
    for mid, summary in bundle["uncertainty_summary"].items():
        assert 0.0 <= summary["composite_uncertainty"] <= 1.0
        assert 0.0 <= summary["local_ambiguity"] <= 1.0
        assert 0.0 <= summary["neighborhood_disagreement"] <= 1.0
        assert 0.0 <= summary["shrink_factor"] <= 1.0


def test_boundary_uncertainty_accepts_top_k_and_uncertainty_scale() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    raw_scores, raw_margins = _make_raw_scores_and_margins()
    bundle_low = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
        top_k=1,
        uncertainty_scale=0.05,
    )
    bundle_high = apply_boundary_uncertainty_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores=raw_scores,
        raw_margins=raw_margins,
        top_k=4,
        uncertainty_scale=0.5,
    )
    assert bundle_low["uncertainty_summary"]
    assert bundle_high["uncertainty_summary"]
    for mid in raw_margins:
        assert (
            bundle_high["uncertainty_summary"][mid]["shrink_factor"]
            <= bundle_low["uncertainty_summary"][mid]["shrink_factor"] + 1e-10
        )
