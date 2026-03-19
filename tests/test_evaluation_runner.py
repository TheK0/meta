from __future__ import annotations

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


def test_summarize_task_metric_rows_outputs_required_table_shape() -> None:
    rows = summarize_task_metric_rows(
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "metrics": {"c_bacc": 1.0, "q_psr": 1.0},
                "pair_counts": {"c_bacc": 2, "q_psr": 1},
            },
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "standard",
                "metrics": {"c_bacc": None, "q_psr": 0.5},
                "pair_counts": {"c_bacc": 0, "q_psr": 2},
            },
        ]
    )

    c_bacc_row = next(row for row in rows if row["metric"] == "c_bacc")
    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")

    assert c_bacc_row == {
        "task_id": "CHEMBL1",
        "seed": 0,
        "split_type": "standard",
        "metric": "c_bacc",
        "score": 1.0,
        "coverage": 0.5,
        "num_valid_episodes": 1,
        "mean_num_valid_pairs_per_episode": 2.0,
    }
    assert q_psr_row["score"] == 0.75
    assert q_psr_row["coverage"] == 1.0
