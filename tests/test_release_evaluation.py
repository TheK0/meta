from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
    assert c_bacc_row["score"] == 1.0
    assert list(saved["task_id"]) == ["CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL1", "CHEMBL1"]
