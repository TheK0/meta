from __future__ import annotations

import pytest

from fsmol_cliff.evaluation import evaluate_episode_manifest, summarize_task_metric_rows


def test_evaluate_episode_manifest_computes_core_metrics_from_manifest_context() -> None:
    episode = {
        "task_id": "CHEMBL1",
        "seed": 0,
        "split_type": "standard",
        "episode_id": 0,
        "support_pos_ids": ["a1"],
        "support_neg_ids": ["n1"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
        "injected_pairs": [],
    }
    assay_context = {
        "labels": {"a1": 1, "n1": 0, "qa": 1, "qn": 0},
        "cliff_pairs": [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.2,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
        "noncliff_pairs": [],
    }

    result = evaluate_episode_manifest(
        episode=episode,
        assay_context=assay_context,
        score_fn=lambda _: {"qa": 0.9, "qn": 0.1},
    )

    assert result["metrics"]["c_bacc"] == 1.0
    assert result["metrics"]["q_psr"] == 1.0
    assert result["pair_counts"]["q_psr"] == 1
    assert result["metrics"]["average_precision_score"] == 1.0
    assert result["metrics"]["delta_auprc"] == 0.5
    assert result["episode_context"]["num_support_molecules"] == 2
    assert result["episode_context"]["fraction_positive_support"] == 0.5
    assert result["episode_context"]["num_query_molecules"] == 2
    assert result["episode_context"]["fraction_positive_query"] == 0.5
    assert result["episode_context"]["num_train_requested"] == 2
    assert result["episode_context"]["fraction_positive_test"] == 0.5


def test_evaluate_episode_manifest_uses_structured_decision_threshold_for_predictions_only() -> None:
    episode = {
        "task_id": "CHEMBL1",
        "seed": 0,
        "split_type": "standard",
        "episode_id": 0,
        "support_pos_ids": ["a1"],
        "support_neg_ids": ["n1"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
        "injected_pairs": [],
    }
    assay_context = {
        "labels": {"a1": 1, "n1": 0, "qa": 1, "qn": 0},
        "cliff_pairs": [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.2,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
        "noncliff_pairs": [],
    }

    default_result = evaluate_episode_manifest(
        episode=episode,
        assay_context=assay_context,
        score_fn=lambda _: {"qa": 0.6, "qn": 0.4},
    )
    threshold_result = evaluate_episode_manifest(
        episode=episode,
        assay_context=assay_context,
        score_fn=lambda _: {"scores": {"qa": 0.6, "qn": 0.4}, "decision_threshold": 0.7},
    )

    assert default_result["metrics"]["q_psr"] == 1.0
    assert threshold_result["metrics"]["q_psr"] == 1.0
    assert default_result["metrics"]["average_precision_score"] == 1.0
    assert threshold_result["metrics"]["average_precision_score"] == 1.0
    assert default_result["metrics"]["c_bacc"] == 1.0
    assert threshold_result["metrics"]["c_bacc"] == 0.5
    assert default_result["metrics"]["scr"] == 0.0
    assert threshold_result["metrics"]["scr"] == 1.0


def test_evaluate_episode_manifest_accepts_structured_payload_with_extra_metadata() -> None:
    episode = {
        "task_id": "CHEMBL1",
        "seed": 0,
        "split_type": "standard",
        "episode_id": 0,
        "support_pos_ids": ["a1"],
        "support_neg_ids": ["n1"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
        "injected_pairs": [],
    }
    assay_context = {
        "labels": {"a1": 1, "n1": 0, "qa": 1, "qn": 0},
        "cliff_pairs": [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.2,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
        "noncliff_pairs": [],
    }

    result = evaluate_episode_manifest(
        episode=episode,
        assay_context=assay_context,
        score_fn=lambda _: {
            "scores": {"qa": 0.6, "qn": 0.4},
            "decision_threshold": 0.7,
            "support_scores": {"a1": 0.8, "n1": 0.4},
        },
    )

    assert result["metrics"]["q_psr"] == 1.0
    assert result["metrics"]["c_bacc"] == 0.5


def test_evaluate_episode_manifest_keeps_raw_scores_mapping_compatible_with_scores_key() -> None:
    episode = {
        "task_id": "CHEMBL1",
        "seed": 0,
        "split_type": "standard",
        "episode_id": 0,
        "support_pos_ids": [1],
        "support_neg_ids": [2],
        "query_pos_ids": ["scores"],
        "query_neg_ids": ["qn"],
        "injected_pairs": [],
    }
    assay_context = {
        "labels": {1: 1, 2: 0, "scores": 1, "qn": 0},
        "cliff_pairs": [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "scores",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.2,
                "same_scaffold": False,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
        "noncliff_pairs": [],
    }

    result = evaluate_episode_manifest(
        episode=episode,
        assay_context=assay_context,
        score_fn=lambda _: {"scores": 0.9, "qn": 0.1},
    )

    assert result["metrics"]["q_psr"] == 1.0
    assert result["metrics"]["c_bacc"] == 1.0


def test_summarize_task_metric_rows_outputs_required_table_shape() -> None:
    rows = summarize_task_metric_rows(
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "metrics": {
                    "c_bacc": 1.0,
                    "q_psr": 1.0,
                    "average_precision_score": 1.0,
                    "delta_auprc": 0.5,
                },
                "pair_counts": {"c_bacc": 2, "q_psr": 1},
                "episode_context": {
                    "num_support_molecules": 2,
                    "fraction_positive_support": 0.5,
                    "num_query_molecules": 2,
                    "fraction_positive_query": 0.5,
                },
            },
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "metrics": {
                    "c_bacc": None,
                    "q_psr": 0.5,
                    "average_precision_score": 0.25,
                    "delta_auprc": 0.0,
                },
                "pair_counts": {"c_bacc": 0, "q_psr": 2},
                "episode_context": {
                    "num_support_molecules": 4,
                    "fraction_positive_support": 0.25,
                    "num_query_molecules": 4,
                    "fraction_positive_query": 0.25,
                },
            },
        ]
    )

    c_bacc_row = next(row for row in rows if row["metric"] == "c_bacc")
    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")
    ap_row = next(row for row in rows if row["metric"] == "average_precision_score")
    delta_row = next(row for row in rows if row["metric"] == "delta_auprc")

    assert c_bacc_row["task_id"] == "CHEMBL1"
    assert c_bacc_row["seed"] == 0
    assert c_bacc_row["split_type"] == "standard"
    assert c_bacc_row["metric"] == "c_bacc"
    assert c_bacc_row["score"] == 1.0
    assert c_bacc_row["coverage"] == 0.5
    assert c_bacc_row["num_valid_episodes"] == 1
    assert c_bacc_row["mean_num_valid_pairs_per_episode"] == 2.0
    assert c_bacc_row["num_episodes"] == 2
    assert c_bacc_row["num_support_molecules"] == 3.0
    assert c_bacc_row["fraction_positive_support"] == pytest.approx(0.375)
    assert c_bacc_row["num_query_molecules"] == 3.0
    assert c_bacc_row["fraction_positive_query"] == pytest.approx(0.375)
    assert q_psr_row["score"] == 0.75
    assert q_psr_row["coverage"] == 1.0
    assert ap_row["score"] == pytest.approx(0.625)
    assert delta_row["score"] == pytest.approx(0.25)
