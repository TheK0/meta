from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from fsmol_cliff.constants import EpisodeConfig
from fsmol_cliff.release import build_release_bundle


def _write_task(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _eligible_records() -> list[dict]:
    records = []
    for index in range(1, 26):
        records.append(
            {
                "Assay_ID": "CHEMBL_ELIGIBLE",
                "compound_id": f"p{index:02d}",
                "Y": 1,
                "Relation": "=",
                "LogRegressionProperty": 8.0 if index <= 15 else 7.0,
                "CanonicalIsomericSmiles": f"P{index:02d}",
            }
        )
    for index in range(1, 26):
        records.append(
            {
                "Assay_ID": "CHEMBL_ELIGIBLE",
                "compound_id": f"n{index:02d}",
                "Y": 0,
                "Relation": "=",
                "LogRegressionProperty": 6.5 if index <= 15 else 6.4,
                "CanonicalIsomericSmiles": f"N{index:02d}",
            }
        )
    return records


def _small_records() -> list[dict]:
    records = []
    for index in range(1, 6):
        records.append(
            {
                "Assay_ID": "CHEMBL_SMALL",
                "compound_id": f"sp{index:02d}",
                "Y": 1,
                "Relation": "=",
                "LogRegressionProperty": 8.0,
                "CanonicalIsomericSmiles": f"SP{index:02d}",
            }
        )
        records.append(
            {
                "Assay_ID": "CHEMBL_SMALL",
                "compound_id": f"sn{index:02d}",
                "Y": 0,
                "Relation": "=",
                "LogRegressionProperty": 6.0,
                "CanonicalIsomericSmiles": f"SN{index:02d}",
            }
        )
    return records


def test_build_release_bundle_writes_task_lists_and_manifests(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "fsmol"
    test_dir = data_dir / "test"
    output_dir = tmp_path / "release"
    task_list_file = tmp_path / "tasks.json"

    _write_task(test_dir / "CHEMBL_ELIGIBLE.jsonl.gz", _eligible_records())
    _write_task(test_dir / "CHEMBL_SMALL.jsonl.gz", _small_records())
    task_list_file.write_text(json.dumps({"test": ["CHEMBL_ELIGIBLE", "CHEMBL_SMALL"]}))

    def fake_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
        if not smiles_a or not smiles_b:
            return None
        if smiles_a.startswith("P") and smiles_b.startswith("N"):
            pos_index = int(smiles_a[1:])
            neg_index = int(smiles_b[1:])
        elif smiles_a.startswith("N") and smiles_b.startswith("P"):
            pos_index = int(smiles_b[1:])
            neg_index = int(smiles_a[1:])
        else:
            return 0.1
        if pos_index <= 15 and neg_index <= 15 and abs(pos_index - neg_index) <= 1:
            return 0.9
        if 16 <= pos_index <= 25 and pos_index == neg_index:
            return 0.9
        return 0.1

    monkeypatch.setattr("fsmol_cliff.assets.tanimoto_similarity", fake_similarity)

    release = build_release_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        task_list_file=task_list_file,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        fsmol_data_version="fsmol-test",
    )

    assert release["eligible_tasks"] == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_all.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_30.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_adv_eligible.json").read_text()) == ["CHEMBL_ELIGIBLE"]

    benchmark_manifest = json.loads((output_dir / "benchmark_manifest.json").read_text())
    assert benchmark_manifest["fsmol_data_version"] == "fsmol-test"
    assert benchmark_manifest["episode_config"]["query_per_class"] == 4

    assert (output_dir / "assays" / "CHEMBL_ELIGIBLE" / "pairs.jsonl").exists()
    assert (output_dir / "episodes_standard.parquet").exists()
    assert (output_dir / "episodes_adversarial.parquet").exists()

    standard = pd.read_parquet(output_dir / "episodes_standard.parquet")
    adversarial = pd.read_parquet(output_dir / "episodes_adversarial.parquet")
    assert list(standard["task_id"]) == ["CHEMBL_ELIGIBLE"]
    assert list(adversarial["task_id"]) == ["CHEMBL_ELIGIBLE"]
