from __future__ import annotations

import numpy as np

from fsmol_cliff.case_adapter import (
    compute_pair_features,
    predict_relation_probs,
    train_relation_head,
)
from fsmol_cliff.case_runner import score_case_net_episode
from fsmol_cliff.signed_relations import build_pair_relation_dataset


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
            {"anchor_id": "a1", "neg_id": "n1", "sim": 0.92, "gap_abs": 1.5,
             "same_scaffold": True, "pair_type": "cliff",
             "anchor_label": 1, "neg_label": 0},
        ],
        "noncliff_pairs": [
            {"anchor_id": "a2", "neg_id": "n2", "sim": 0.88, "gap_abs": 0.5,
             "same_scaffold": False, "pair_type": "highsim_noncliff",
             "anchor_label": 1, "neg_label": 0},
        ],
    }


def test_build_pair_relation_dataset_uses_support_only() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    labels, anchors, neighbors, features = build_pair_relation_dataset(
        episode=episode, assay_context=assay_context
    )
    # a1-n1 is a cliff pair → "flip", a2-n2 is noncliff with diff labels → "flip"
    assert len(labels) == 2
    assert labels[0] == "flip"  # cliff
    assert labels[1] == "flip"  # discordant noncliff
    assert set(anchors + neighbors) <= {"a1", "a2", "n1", "n2"}


def test_build_pair_relation_dataset_skips_non_support_pairs() -> None:
    episode = _make_episode()
    ctx = _make_assay_context()
    ctx["cliff_pairs"].append({
        "anchor_id": "qa", "neg_id": "qn", "sim": 0.95, "gap_abs": 2.0,
        "same_scaffold": False, "pair_type": "cliff",
        "anchor_label": 1, "neg_label": 0,
    })
    labels, anchors, neighbors, features = build_pair_relation_dataset(
        episode=episode, assay_context=ctx
    )
    # qa-qn is NOT in support → should be excluded
    assert len(labels) == 2  # still only the two support pairs


def test_compute_pair_features_returns_valid_vector() -> None:
    ctx = _make_assay_context()
    feat = compute_pair_features(
        anchor_id="a1", neighbor_id="n1",
        pair_info={"sim": 0.92, "gap_abs": 1.5, "same_scaffold": True},
        assay_context=ctx,
    )
    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert feat.shape[0] >= 4  # at least 4 scalar features
    assert feat[0] == 0.92  # sim
    assert feat[2] == 1.0  # same_scaffold


def test_train_relation_head_produces_usable_model() -> None:
    # Create synthetic features: flip pairs have high sim, same pairs have low sim
    features = [
        np.array([0.95, 2.0, 1.0, 1.0], dtype=np.float32),
        np.array([0.92, 1.8, 1.0, 1.0], dtype=np.float32),
        np.array([0.82, 0.3, 0.0, 1.0], dtype=np.float32),
        np.array([0.80, 0.2, 0.0, 0.0], dtype=np.float32),
    ]
    labels = ["flip", "flip", "same", "same"]
    head = train_relation_head(features=features, labels=labels)

    test_features = [
        np.array([0.93, 1.7, 1.0, 1.0], dtype=np.float32),  # should be flip
        np.array([0.81, 0.4, 0.0, 0.0], dtype=np.float32),  # should be same
    ]
    p_same, p_flip = predict_relation_probs(head, test_features)
    assert len(p_same) == 2
    assert p_flip[0] > p_same[0]  # first test pair → flip
    assert p_same[1] > p_flip[1]  # second test pair → same
    assert all(0.0 <= p <= 1.0 for p in p_same + p_flip)


def test_train_relation_head_handles_empty_input() -> None:
    head = train_relation_head(features=[], labels=[])
    p_same, p_flip = predict_relation_probs(
        head, [np.array([0.9, 1.0, 1.0, 1.0], dtype=np.float32)]
    )
    assert len(p_same) == 1
    assert 0.0 <= p_same[0] <= 1.0


def test_score_case_net_episode_produces_calibrated_scores() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    proto_scores = {"qa": 0.52, "qn": 0.48, "a1": 0.80, "a2": 0.75, "n1": 0.20, "n2": 0.30}

    calibrated = score_case_net_episode(
        episode=episode,
        assay_context=assay_context,
        proto_scores=proto_scores,
        fusion_lambda=0.5,
    )

    assert set(calibrated.keys()) == {"qa", "qn"}
    for score in calibrated.values():
        assert 0.0 <= score <= 1.0
    # Closer to support structure: qa should be closer to active side
    assert calibrated["qa"] >= calibrated["qn"]


def test_score_case_net_episode_pure_evidence() -> None:
    episode = _make_episode()
    assay_context = _make_assay_context()
    proto_scores = {"qa": 0.52, "qn": 0.48, "a1": 0.80, "a2": 0.75, "n1": 0.20, "n2": 0.30}

    calibrated = score_case_net_episode(
        episode=episode,
        assay_context=assay_context,
        proto_scores=proto_scores,
        fusion_lambda=0.0,  # pure evidence
    )
    assert set(calibrated.keys()) == {"qa", "qn"}
    for score in calibrated.values():
        assert 0.0 <= score <= 1.0
