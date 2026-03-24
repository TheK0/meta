from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fsmol_cliff.io import write_jsonl, write_parquet
from fsmol_cliff.runner import evaluate_release_with_sklearn_baseline


def test_evaluate_release_with_sklearn_baseline_writes_task_metric_rows(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {
                "molecule_id": "a1",
                "canonical_isomeric_smiles": "CCO",
                "label": 1,
                "r": 8.0,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 1.0, 0.0, 0.0],
            },
            {
                "molecule_id": "n1",
                "canonical_isomeric_smiles": "CCN",
                "label": 0,
                "r": 6.0,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 1.0, 1.0],
            },
            {
                "molecule_id": "qa",
                "canonical_isomeric_smiles": "CCF",
                "label": 1,
                "r": 8.2,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 0.8, 0.0, 0.0],
            },
            {
                "molecule_id": "qn",
                "canonical_isomeric_smiles": "CCC",
                "label": 0,
                "r": 6.1,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 0.8, 1.0],
            },
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {
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
        ],
    )
    write_parquet(release_dir / "episodes_adversarial.parquet", [])

    output_path = tmp_path / "task_results.parquet"
    rows = evaluate_release_with_sklearn_baseline(
        release_dir=release_dir,
        output_path=output_path,
        split_types=("standard",),
        model_name="kNN",
        model_params={"n_neighbors": 1},
    )

    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    c_bacc_row = next(row for row in rows if row["metric"] == "c_bacc")
    ap_row = next(row for row in rows if row["metric"] == "average_precision_score")
    delta_row = next(row for row in rows if row["metric"] == "delta_auprc")
    assert c_bacc_row["score"] == 1.0
    assert ap_row["score"] == 1.0
    assert delta_row["score"] == 0.5
    assert delta_row["fraction_positive_query"] == 0.5
    assert set(saved["metric"]) >= {"average_precision_score", "delta_auprc", "c_bacc", "q_psr"}
    assert len(saved) == 11


def test_evaluate_release_with_sklearn_baseline_emits_profile_and_result_tier(
    tmp_path: Path,
) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {
                "molecule_id": "a1",
                "canonical_isomeric_smiles": "CCO",
                "label": 1,
                "r": 8.0,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 1.0, 0.0, 0.0],
            },
            {
                "molecule_id": "n1",
                "canonical_isomeric_smiles": "CCN",
                "label": 0,
                "r": 6.0,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 1.0, 1.0],
            },
            {
                "molecule_id": "qa",
                "canonical_isomeric_smiles": "CCF",
                "label": 1,
                "r": 8.2,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 0.8, 0.0, 0.0],
            },
            {
                "molecule_id": "qn",
                "canonical_isomeric_smiles": "CCC",
                "label": 0,
                "r": 6.1,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 0.8, 1.0],
            },
        ],
    )
    write_jsonl(
        assay_dir / "pairs_strict.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard_strict.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "profile": "strict",
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ],
    )
    write_parquet(release_dir / "episodes_adversarial_strict.parquet", [])

    output_path = tmp_path / "task_results_schema.parquet"
    rows = evaluate_release_with_sklearn_baseline(
        release_dir=release_dir,
        output_path=output_path,
        split_types=("standard",),
        profile="strict",
        result_tier="final",
        model_name="kNN",
        model_params={"n_neighbors": 1},
    )

    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    assert {row["profile"] for row in rows} == {"strict"}
    assert {row["result_tier"] for row in rows} == {"final"}
    assert set(saved["profile"]) == {"strict"}
    assert set(saved["result_tier"]) == {"final"}


def test_evaluate_release_with_official_baseline_backend(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {
                "molecule_id": "a1",
                "canonical_isomeric_smiles": "CCO",
                "label": 1,
                "r": 8.0,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 1.0, 0.0, 0.0],
            },
            {
                "molecule_id": "n1",
                "canonical_isomeric_smiles": "CCN",
                "label": 0,
                "r": 6.0,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 1.0, 1.0],
            },
            {
                "molecule_id": "qa",
                "canonical_isomeric_smiles": "CCF",
                "label": 1,
                "r": 8.2,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 0.8, 0.0, 0.0],
            },
            {
                "molecule_id": "qn",
                "canonical_isomeric_smiles": "CCC",
                "label": 0,
                "r": 6.1,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 0.8, 1.0],
            },
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qa",
                "neg_id": "qn",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {
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
        ],
    )
    write_parquet(release_dir / "episodes_adversarial.parquet", [])

    rows = evaluate_release_with_sklearn_baseline(
        release_dir=release_dir,
        output_path=tmp_path / "task_results_official.parquet",
        split_types=("standard",),
        model_name="kNN",
        model_params={"n_neighbors": 1},
        backend="official",
    )

    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")
    ap_row = next(row for row in rows if row["metric"] == "average_precision_score")
    delta_row = next(row for row in rows if row["metric"] == "delta_auprc")
    assert q_psr_row["score"] == 1.0
    assert ap_row["score"] == 1.0
    assert delta_row["score"] == pytest.approx(0.5)


def test_evaluate_release_emits_sq_psr_for_adversarial_injected_pairs(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {
                "molecule_id": "a1",
                "canonical_isomeric_smiles": "CCO",
                "label": 1,
                "r": 8.0,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 1.0, 0.0, 0.0],
            },
            {
                "molecule_id": "a2",
                "canonical_isomeric_smiles": "CCCl",
                "label": 1,
                "r": 8.1,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 0.9, 0.0, 0.0],
            },
            {
                "molecule_id": "n1",
                "canonical_isomeric_smiles": "CCN",
                "label": 0,
                "r": 6.0,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 1.0, 1.0],
            },
            {
                "molecule_id": "n2",
                "canonical_isomeric_smiles": "CCC",
                "label": 0,
                "r": 6.1,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 0.8, 1.0],
            },
            {
                "molecule_id": "qp",
                "canonical_isomeric_smiles": "CCF",
                "label": 1,
                "r": 8.2,
                "scaffold_smiles": "CC",
                "fingerprint": [1.0, 0.8, 0.0, 0.0],
            },
            {
                "molecule_id": "sn",
                "canonical_isomeric_smiles": "CCBr",
                "label": 0,
                "r": 6.2,
                "scaffold_smiles": "CC",
                "fingerprint": [0.0, 0.0, 0.7, 1.0],
            },
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "qp",
                "neg_id": "n2",
                "sim": 0.9,
                "gap_abs": 1.1,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    )
    write_parquet(release_dir / "episodes_standard.parquet", [])
    write_parquet(
        release_dir / "episodes_adversarial.parquet",
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "split_type": "adversarial",
                "episode_id": 0,
                "support_pos_ids": ["a1", "a2"],
                "support_neg_ids": ["sn"],
                "query_pos_ids": ["qp"],
                "query_neg_ids": ["n1", "n2"],
                "injected_pairs": [
                    {
                        "assay_id": "CHEMBL1",
                        "anchor_id": "a1",
                        "neg_id": "n1",
                        "sim": 0.95,
                        "gap_abs": 1.4,
                        "same_scaffold": True,
                        "pair_type": "cliff",
                        "anchor_label": 1,
                        "neg_label": 0,
                    }
                ],
            }
        ],
    )

    rows = evaluate_release_with_sklearn_baseline(
        release_dir=release_dir,
        output_path=tmp_path / "task_results_sq.parquet",
        split_types=("adversarial",),
        model_name="kNN",
        model_params={"n_neighbors": 1},
    )

    sq_row = next(row for row in rows if row["metric"] == "sq_psr")
    assert sq_row["num_valid_episodes"] == 1
    assert sq_row["score"] is not None


def test_cliff_aware_baseline_backend_runs_with_extra_hard_negatives(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC", "fingerprint": [1.0, 1.0, 0.0, 0.0]},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 1.0, 1.0]},
            {"molecule_id": "hn1", "canonical_isomeric_smiles": "CCCl", "label": 0, "r": 5.9, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 0.9, 1.0]},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC", "fingerprint": [1.0, 0.8, 0.0, 0.0]},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 0.8, 1.0]},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {"assay_id": "CHEMBL1", "anchor_id": "qa", "neg_id": "qn", "sim": 0.9, "gap_abs": 1.1, "same_scaffold": True, "pair_type": "cliff", "anchor_label": 1, "neg_label": 0}
        ],
    )
    (assay_dir / "anchor_to_hardnegs.json").write_text('{"a1": ["hn1"]}')
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {"task_id": "CHEMBL1", "seed": 0, "split_type": "standard", "episode_id": 0, "support_pos_ids": ["a1"], "support_neg_ids": ["n1"], "query_pos_ids": ["qa"], "query_neg_ids": ["qn"], "injected_pairs": []}
        ],
    )
    write_parquet(release_dir / "episodes_adversarial.parquet", [])

    rows = evaluate_release_with_sklearn_baseline(
        release_dir=release_dir,
        output_path=tmp_path / "task_results_cliff_aware.parquet",
        split_types=("standard",),
        model_name="kNN",
        model_params={"n_neighbors": 1},
        backend="cliff-aware",
    )

    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")
    assert q_psr_row["score"] is not None


def test_evaluate_release_with_decision_aware_backend_uses_structured_threshold_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC", "fingerprint": [1.0, 1.0, 0.0, 0.0]},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 1.0, 1.0]},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC", "fingerprint": [1.0, 0.8, 0.0, 0.0]},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 0.8, 1.0]},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {"assay_id": "CHEMBL1", "anchor_id": "qa", "neg_id": "qn", "sim": 0.9, "gap_abs": 1.1, "same_scaffold": True, "pair_type": "cliff", "anchor_label": 1, "neg_label": 0}
        ],
    )
    (assay_dir / "anchor_to_hardnegs.json").write_text('{"a1": ["hn1"]}')
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {"task_id": "CHEMBL1", "seed": 0, "split_type": "standard", "episode_id": 0, "support_pos_ids": ["a1"], "support_neg_ids": ["n1"], "query_pos_ids": ["qa"], "query_neg_ids": ["qn"], "injected_pairs": []}
        ],
    )
    write_parquet(release_dir / "episodes_adversarial.parquet", [])

    calls: list[dict] = []

    def fake_score_decision_aware_sklearn_episode(**kwargs):
        calls.append(kwargs)
        return {
            "scores": {"a1": 0.8, "n1": 0.4, "qa": 0.6, "qn": 0.4},
            "decision_threshold": 0.7,
        }

    monkeypatch.setattr(
        "fsmol_cliff.runner.score_decision_aware_sklearn_episode",
        fake_score_decision_aware_sklearn_episode,
        raising=False,
    )

    rows = evaluate_release_with_sklearn_baseline(
        release_dir=release_dir,
        output_path=tmp_path / "task_results_decision_aware.parquet",
        split_types=("standard",),
        model_name="kNN",
        model_params={"n_neighbors": 1},
        backend="decision-aware",
    )

    assert calls[0]["support_pos_ids"] == ["a1"]
    assert calls[0]["support_neg_ids"] == ["n1"]
    assert calls[0]["anchor_to_hardnegs"] == {"a1": ["hn1"]}
    assert calls[0]["query_ids"] == ["a1", "n1", "qa", "qn"]
    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")
    c_bacc_row = next(row for row in rows if row["metric"] == "c_bacc")
    assert q_psr_row["score"] == 1.0
    assert c_bacc_row["score"] == 0.5


def test_decision_aware_backend_rejects_non_knn_model_name(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    write_parquet(
        assay_dir / "molecule_annotations.parquet",
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1, "r": 8.0, "scaffold_smiles": "CC", "fingerprint": [1.0, 1.0, 0.0, 0.0]},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0, "r": 6.0, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 1.0, 1.0]},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1, "r": 8.2, "scaffold_smiles": "CC", "fingerprint": [1.0, 0.8, 0.0, 0.0]},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0, "r": 6.1, "scaffold_smiles": "CC", "fingerprint": [0.0, 0.0, 0.8, 1.0]},
        ],
    )
    write_jsonl(
        assay_dir / "pairs.jsonl",
        [
            {"assay_id": "CHEMBL1", "anchor_id": "qa", "neg_id": "qn", "sim": 0.9, "gap_abs": 1.1, "same_scaffold": True, "pair_type": "cliff", "anchor_label": 1, "neg_label": 0}
        ],
    )
    (assay_dir / "anchor_to_hardnegs.json").write_text("{}")
    write_parquet(
        release_dir / "episodes_standard.parquet",
        [
            {"task_id": "CHEMBL1", "seed": 0, "split_type": "standard", "episode_id": 0, "support_pos_ids": ["a1"], "support_neg_ids": ["n1"], "query_pos_ids": ["qa"], "query_neg_ids": ["qn"], "injected_pairs": []}
        ],
    )
    write_parquet(release_dir / "episodes_adversarial.parquet", [])

    with pytest.raises(ValueError, match="model_name='kNN'"):
        evaluate_release_with_sklearn_baseline(
            release_dir=release_dir,
            output_path=tmp_path / "task_results_invalid_decision_aware.parquet",
            split_types=("standard",),
            model_name="randomForest",
            backend="decision-aware",
        )
