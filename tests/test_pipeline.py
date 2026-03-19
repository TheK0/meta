from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from fsmol_cliff.pipeline import build_assay_asset_bundle, load_task_records


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_task_records_reads_jsonl_gz_task_file(tmp_path: Path) -> None:
    task_file = tmp_path / "CHEMBL1.jsonl.gz"
    _write_jsonl_gz(
        task_file,
        [
            {
                "Assay_ID": "CHEMBL1",
                "compound_id": "mol-1",
                "Y": 1,
                "Relation": "=",
                "LogRegressionProperty": 8.0,
                "SMILES": "CCO",
            }
        ],
    )

    records = load_task_records(task_file)

    assert records[0]["Assay_ID"] == "CHEMBL1"
    assert records[0]["compound_id"] == "mol-1"


def test_build_assay_asset_bundle_writes_expected_release_files(tmp_path: Path) -> None:
    task_file = tmp_path / "CHEMBL1.jsonl.gz"
    out_dir = tmp_path / "release" / "CHEMBL1"
    _write_jsonl_gz(
        task_file,
        [
            {
                "Assay_ID": "CHEMBL1",
                "compound_id": "a1",
                "Y": 1,
                "Relation": "=",
                "LogRegressionProperty": 8.0,
                "CanonicalIsomericSmiles": "CCO",
            },
            {
                "Assay_ID": "CHEMBL1",
                "compound_id": "n1",
                "Y": 0,
                "Relation": "=",
                "LogRegressionProperty": 6.5,
                "CanonicalIsomericSmiles": "CCN",
            },
        ],
    )

    bundle = build_assay_asset_bundle(task_file=task_file, output_dir=out_dir)

    assert bundle["assay_id"] == "CHEMBL1"
    assert (out_dir / "pairs.jsonl").exists()
    assert (out_dir / "anchor_to_hardnegs.json").exists()
    assert (out_dir / "molecule_annotations.parquet").exists()
    assert (out_dir / "diagnostics.json").exists()

    annotations = pd.read_parquet(out_dir / "molecule_annotations.parquet")
    assert list(annotations["molecule_id"]) == ["a1", "n1"]
