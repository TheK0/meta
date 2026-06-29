from __future__ import annotations


def test_identity_calibration_preserves_raw_scores() -> None:
    from fsmol_cliff.protonet_local_calibrated import apply_identity_local_calibration

    bundle = apply_identity_local_calibration(
        raw_scores={"qa": 0.8},
        raw_margins={"qa": 0.3},
    )

    assert bundle["raw_scores"] == {"qa": 0.8}
    assert bundle["calibrated_scores"] == {"qa": 0.8}
    assert bundle["raw_margins"] == {"qa": 0.3}
    assert bundle["calibrated_margins"] == {"qa": 0.3}


def test_query_only_local_calibration_uses_local_features() -> None:
    from fsmol_cliff.protonet_local_calibrated import apply_query_only_local_calibration

    episode = {
        "support_pos_ids": ["a1", "a2"],
        "support_neg_ids": ["n1", "n2"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
    }
    assay_context = {
        "labels": {"a1": 1, "a2": 1, "n1": 0, "n2": 0, "qa": 1, "qn": 0},
        "cliff_pairs": [
            {"assay_id": "CHEMBL1", "anchor_id": "a1", "neg_id": "n1", "sim": 0.95, "gap_abs": 1.4, "same_scaffold": False, "pair_type": "cliff"},
            {"assay_id": "CHEMBL1", "anchor_id": "a2", "neg_id": "n2", "sim": 0.91, "gap_abs": 1.2, "same_scaffold": False, "pair_type": "cliff"},
            {"assay_id": "CHEMBL1", "anchor_id": "qa", "neg_id": "n1", "sim": 0.94, "gap_abs": 1.3, "same_scaffold": True, "pair_type": "cliff"},
        ],
        "noncliff_pairs": [
            {"assay_id": "CHEMBL1", "anchor_id": "qa", "neg_id": "n2", "sim": 0.89, "gap_abs": 0.4, "same_scaffold": False, "pair_type": "highsim_noncliff"},
        ],
    }

    bundle = apply_query_only_local_calibration(
        episode=episode,
        assay_context=assay_context,
        raw_scores={"a1": 0.82, "a2": 0.74, "n1": 0.24, "n2": 0.31, "qa": 0.52, "qn": 0.48},
        raw_margins={"a1": 0.32, "a2": 0.24, "n1": -0.26, "n2": -0.19, "qa": 0.02, "qn": -0.02},
    )

    assert bundle["raw_scores"]["qa"] == 0.52
    assert bundle["calibrated_scores"]["qa"] != 0.52
    assert set(bundle["local_features"]["qa"]) == {
        "raw_score",
        "raw_margin",
        "prototype_gap",
        "support_dispersion",
        "cross_class_density",
        "cross_class_cliff_fraction",
    }
