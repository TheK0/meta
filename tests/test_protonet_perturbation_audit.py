from __future__ import annotations

import json
from pathlib import Path

from fsmol_cliff.protonet_perturbation_audit import (
    derive_query_id_slices,
    summarize_perturbation_report,
    summarize_query_score_variance,
    support_subset_dropout_episode,
)


def test_support_subset_dropout_episode_removes_support_examples_but_keeps_episode_legal() -> None:
    episode = {
        "support_pos_ids": ["a1", "a2", "a3"],
        "support_neg_ids": ["n1", "n2", "n3"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
    }

    dropped = support_subset_dropout_episode(episode, drop_per_class=1)

    assert dropped["support_pos_ids"] == ["a1", "a2"]
    assert dropped["support_neg_ids"] == ["n1", "n2"]
    assert dropped["query_pos_ids"] == ["qa"]
    assert dropped["query_neg_ids"] == ["qn"]


def test_summarize_query_score_variance_reports_cliff_control_and_same_scaffold_gaps() -> None:
    summary = summarize_query_score_variance(
        baseline_scores={"qa": 0.8, "qn": 0.2, "ca": 0.7, "cn": 0.3},
        perturbed_score_runs=[
            {"qa": 0.6, "qn": 0.4, "ca": 0.68, "cn": 0.32},
            {"qa": 0.9, "qn": 0.1, "ca": 0.69, "cn": 0.31},
        ],
        cliff_query_ids=["qa", "qn"],
        control_query_ids=["ca", "cn"],
        same_scaffold_cliff_query_ids=["qa", "qn"],
    )

    assert set(summary) == {
        "per_query_variance",
        "cliff_variance_mean",
        "control_variance_mean",
        "same_scaffold_cliff_variance_mean",
        "cliff_control_variance_gap",
        "same_scaffold_cliff_control_variance_gap",
    }
    assert summary["cliff_variance_mean"] > summary["control_variance_mean"]


def test_derive_query_id_slices_separates_cliff_control_and_same_scaffold_ids() -> None:
    slices = derive_query_id_slices(
        cliff_pairs=[
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.95,
                "gap_abs": 1.2,
                "same_scaffold": True,
                "pair_type": "cliff",
            }
        ],
        noncliff_pairs=[
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "ca",
                "neg_id": "cn",
                "sim": 0.91,
                "gap_abs": 0.2,
                "same_scaffold": False,
                "pair_type": "highsim_noncliff",
            }
        ],
    )

    assert slices["cliff_query_ids"] == ["qa", "qn"]
    assert slices["control_query_ids"] == ["ca", "cn"]
    assert slices["same_scaffold_cliff_query_ids"] == ["qa", "qn"]


def test_summarize_perturbation_report_writes_expected_top_level_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "audit.json"
    report = summarize_perturbation_report(
        output_path=output_path,
        rows=[
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "episode_id": 0,
                "cliff_control_variance_gap": 0.1,
                "same_scaffold_cliff_control_variance_gap": 0.2,
            },
            {
                "task_id": "CHEMBL2",
                "seed": 1,
                "episode_id": 1,
                "cliff_control_variance_gap": 0.3,
                "same_scaffold_cliff_control_variance_gap": 0.4,
            },
        ],
        profile="relaxed_covext_10_10",
        split_type="adversarial",
        seeds=[0, 1],
        episodes_per_task=5,
        dropout_strengths=[1, 2],
        views_per_strength=4,
    )

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["profile"] == "relaxed_covext_10_10"
    assert saved["split_type"] == "adversarial"
    assert saved["episodes_analyzed"] == 2
    assert saved["tasks_analyzed"] == ["CHEMBL1", "CHEMBL2"]
    assert saved["cliff_control_variance_gap_mean"] == 0.2
    assert saved["same_scaffold_cliff_control_variance_gap_mean"] == 0.30000000000000004
    assert report == saved
