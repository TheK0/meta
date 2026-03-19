from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

from fsmol_cliff.cli import main


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_build_assets_command_writes_asset_bundle(tmp_path: Path, monkeypatch) -> None:
    task_file = tmp_path / "CHEMBL1.jsonl.gz"
    output_dir = tmp_path / "out"
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

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsmol-cliff", "build-assets", "--task-file", str(task_file), "--output-dir", str(output_dir)],
    )

    assert main() == 0
    assert (output_dir / "pairs.jsonl").exists()


def test_adapter_status_command_writes_availability_report(tmp_path: Path, monkeypatch) -> None:
    output_file = tmp_path / "adapter_status.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsmol-cliff", "adapter-status", "--output", str(output_file)],
    )

    assert main() == 0
    payload = json.loads(output_file.read_text())
    assert payload["baseline"]["available"] is True


def test_build_episodes_command_writes_adversarial_manifest(tmp_path: Path, monkeypatch) -> None:
    spec_file = tmp_path / "episode_spec.json"
    out_file = tmp_path / "episode.json"
    spec_file.write_text(
        json.dumps(
            {
                "support_pos_ids": ["a1", "a2"],
                "support_neg_ids": ["sn1", "sn2"],
                "query_pos_ids": ["qp1", "qp2"],
                "query_neg_ids": ["n1", "n2", "n3", "n4"],
                "cliff_pairs": [
                    {
                        "assay_id": "CHEMBL1",
                        "anchor_id": "a1",
                        "neg_id": "n1",
                        "sim": 0.9,
                        "gap_abs": 1.2,
                        "same_scaffold": False,
                        "pair_type": "cliff",
                        "anchor_label": 1,
                        "neg_label": 0,
                    },
                    {
                        "assay_id": "CHEMBL1",
                        "anchor_id": "a2",
                        "neg_id": "n2",
                        "sim": 0.88,
                        "gap_abs": 1.1,
                        "same_scaffold": True,
                        "pair_type": "cliff",
                        "anchor_label": 1,
                        "neg_label": 0,
                    },
                ],
                "anchor_to_hardnegs": {"a1": ["n1"], "a2": ["n2"]},
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsmol-cliff", "build-episodes", "--input", str(spec_file), "--output", str(out_file)],
    )

    assert main() == 0
    payload = json.loads(out_file.read_text())
    assert len(payload["injected_pairs"]) == 2


def test_build_release_command_writes_release_bundle(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "fsmol"
    task_dir = data_dir / "test"
    task_list = tmp_path / "tasks.json"
    output_dir = tmp_path / "release"

    _write_jsonl_gz(
        task_dir / "CHEMBL1.jsonl.gz",
        [
            {
                "Assay_ID": "CHEMBL1",
                "compound_id": f"p{i:02d}",
                "Y": 1,
                "Relation": "=",
                "LogRegressionProperty": 8.0 if i <= 15 else 7.0,
                "CanonicalIsomericSmiles": f"P{i:02d}",
            }
            for i in range(1, 26)
        ]
        + [
            {
                "Assay_ID": "CHEMBL1",
                "compound_id": f"n{i:02d}",
                "Y": 0,
                "Relation": "=",
                "LogRegressionProperty": 6.5 if i <= 15 else 6.4,
                "CanonicalIsomericSmiles": f"N{i:02d}",
            }
            for i in range(1, 26)
        ],
    )
    task_list.write_text(json.dumps({"test": ["CHEMBL1"]}))

    monkeypatch.setattr(
        "fsmol_cliff.assets.tanimoto_similarity",
        lambda a, b: 0.9 if a and b and a[0] != b[0] else 0.1,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsmol-cliff",
            "build-release",
            "--data-dir",
            str(data_dir),
            "--task-list-file",
            str(task_list),
            "--output-dir",
            str(output_dir),
            "--support-per-class",
            "2",
            "--query-per-class",
            "4",
            "--episodes-per-split",
            "1",
            "--seeds",
            "[0]",
            "--fsmol-data-version",
            "fsmol-test",
        ],
    )

    assert main() == 0
    assert (output_dir / "benchmark_manifest.json").exists()
    assert (output_dir / "episodes_standard.parquet").exists()


def test_evaluate_command_writes_metric_summary(tmp_path: Path, monkeypatch) -> None:
    input_file = tmp_path / "eval.json"
    output_file = tmp_path / "metrics.json"
    input_file.write_text(
        json.dumps(
            {
                "labels": {"qa": 1, "qn": 0},
                "scores": {"qa": 0.8, "qn": 0.2},
                "predictions": {"qa": 1, "qn": 0},
                "cliff_query_ids": ["qa", "qn"],
                "noncliff_query_ids": ["qa", "qn"],
                "query_pairs": [
                    {
                        "assay_id": "CHEMBL1",
                        "anchor_id": "qa",
                        "neg_id": "qn",
                        "sim": 0.92,
                        "gap_abs": 1.4,
                        "same_scaffold": True,
                        "pair_type": "cliff",
                        "anchor_label": 1,
                        "neg_label": 0,
                    }
                ],
                "noncliff_pairs": [],
                "support_query_pairs": [],
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsmol-cliff", "evaluate", "--input", str(input_file), "--output", str(output_file)],
    )

    assert main() == 0
    payload = json.loads(output_file.read_text())
    assert payload["c_bacc"] == 1.0
    assert payload["q_psr"] == 1.0


def test_evaluate_command_can_run_release_mode(tmp_path: Path, monkeypatch) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)
    output_file = tmp_path / "task_results.parquet"

    from fsmol_cliff.io import write_jsonl, write_parquet

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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsmol-cliff",
            "evaluate",
            "--release-dir",
            str(release_dir),
            "--output",
            str(output_file),
            "--split-types",
            '["standard"]',
            "--model-name",
            "kNN",
            "--model-params",
            '{"n_neighbors": 1}',
        ],
    )

    assert main() == 0
    saved = pd.read_parquet(output_file)
    assert set(saved["metric"]) >= {"c_bacc", "q_psr"}


def test_evaluate_command_can_run_release_mode_with_official_backend(tmp_path: Path, monkeypatch) -> None:
    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)
    output_file = tmp_path / "task_results.parquet"

    from fsmol_cliff.io import write_jsonl, write_parquet

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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsmol-cliff",
            "evaluate",
            "--release-dir",
            str(release_dir),
            "--output",
            str(output_file),
            "--split-types",
            '["standard"]',
            "--model-name",
            "kNN",
            "--model-params",
            '{"n_neighbors": 1}',
            "--backend",
            "official",
        ],
    )

    assert main() == 0
    saved = pd.read_parquet(output_file)
    assert set(saved["metric"]) >= {"c_bacc", "q_psr"}


def test_aggregate_and_validate_commands_write_json_outputs(tmp_path: Path, monkeypatch) -> None:
    aggregate_input = tmp_path / "aggregate.json"
    aggregate_output = tmp_path / "aggregate_out.json"
    validate_output = tmp_path / "validate_out.json"
    aggregate_input.write_text(
        json.dumps(
            {
                "task-a": {"mean": 0.4, "valid_count": 1, "total_count": 1},
                "task-b": {"mean": 0.6, "valid_count": 1, "total_count": 1},
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsmol-cliff", "aggregate", "--input", str(aggregate_input), "--output", str(aggregate_output)],
    )
    assert main() == 0
    aggregate_payload = json.loads(aggregate_output.read_text())
    assert aggregate_payload["mean"] == 0.5

    validate_input = tmp_path / "validate.json"
    validate_input.write_text(
        json.dumps(
            {
                "c_bacc": {"mean": 0.5},
                "nc_bacc": {"mean": 0.8},
                "q_psr": {"mean": 0.4},
                "nc_psr": {"mean": 0.9},
                "sq_psr": {"mean": 0.3},
                "ss_sq_psr": {"mean": 0.2},
                "scr": {"mean": 0.2},
                "ss_scr": {"mean": 0.5},
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsmol-cliff",
            "validate-hypotheses",
            "--input",
            str(validate_input),
            "--output",
            str(validate_output),
        ],
    )
    assert main() == 0
    validate_payload = json.loads(validate_output.read_text())
    assert validate_payload["h1"]["accepted"] is True
    assert validate_payload["h2"]["accepted"] is True
    assert validate_payload["h3"]["accepted"] is True


def test_validate_hypotheses_command_supports_model_set_analysis(tmp_path: Path, monkeypatch) -> None:
    validate_input = tmp_path / "validate_models.json"
    validate_output = tmp_path / "validate_models_out.json"
    validate_input.write_text(
        json.dumps(
            {
                "models": {
                    "model_a": {
                        "official": {"task_values": [0.8, 0.82, 0.81]},
                        "c_bacc": {"task_values": [0.45, 0.47, 0.44]},
                        "nc_bacc": {"task_values": [0.71, 0.74, 0.72]},
                        "q_psr": {"task_values": [0.40, 0.42, 0.41]},
                        "nc_psr": {"task_values": [0.69, 0.71, 0.70]},
                    },
                    "model_b": {
                        "official": {"task_values": [0.75, 0.74, 0.76]},
                        "c_bacc": {"task_values": [0.60, 0.59, 0.61]},
                        "nc_bacc": {"task_values": [0.70, 0.69, 0.71]},
                        "q_psr": {"task_values": [0.58, 0.57, 0.59]},
                        "nc_psr": {"task_values": [0.68, 0.67, 0.69]},
                    },
                    "model_c": {
                        "official": {"task_values": [0.70, 0.69, 0.71]},
                        "c_bacc": {"task_values": [0.54, 0.53, 0.55]},
                        "nc_bacc": {"task_values": [0.69, 0.68, 0.70]},
                        "q_psr": {"task_values": [0.52, 0.51, 0.53]},
                        "nc_psr": {"task_values": [0.67, 0.66, 0.68]},
                    },
                }
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsmol-cliff",
            "validate-hypotheses",
            "--input",
            str(validate_input),
            "--output",
            str(validate_output),
        ],
    )

    assert main() == 0
    payload = json.loads(validate_output.read_text())
    assert payload["h1"]["accepted"] is True


def test_validate_hypotheses_command_supports_intervention_analysis(tmp_path: Path, monkeypatch) -> None:
    validate_input = tmp_path / "validate_intervention.json"
    validate_output = tmp_path / "validate_intervention_out.json"
    validate_input.write_text(
        json.dumps(
            {
                "baseline": {
                    "official": {"task_values": [0.60, 0.62, 0.61]},
                    "c_bacc": {"task_values": [0.40, 0.42, 0.41]},
                    "q_psr": {"task_values": [0.45, 0.44, 0.46]},
                    "sq_psr": {"task_values": [0.35, 0.34, 0.36]},
                    "nc_bacc": {"task_values": [0.70, 0.69, 0.71]},
                    "nc_psr": {"task_values": [0.68, 0.69, 0.67]},
                    "scr": {"task_values": [0.40, 0.41, 0.39]},
                    "ss_scr": {"task_values": [0.46, 0.47, 0.45]},
                    "ss_q_psr": {"task_values": [0.30, 0.29, 0.31]},
                    "ss_sq_psr": {"task_values": [0.25, 0.24, 0.26]},
                },
                "treatment": {
                    "official": {"task_values": [0.62, 0.63, 0.64]},
                    "c_bacc": {"task_values": [0.55, 0.56, 0.57]},
                    "q_psr": {"task_values": [0.58, 0.57, 0.59]},
                    "sq_psr": {"task_values": [0.49, 0.48, 0.50]},
                    "nc_bacc": {"task_values": [0.71, 0.72, 0.70]},
                    "nc_psr": {"task_values": [0.69, 0.70, 0.68]},
                    "scr": {"task_values": [0.22, 0.21, 0.23]},
                    "ss_scr": {"task_values": [0.26, 0.25, 0.27]},
                    "ss_q_psr": {"task_values": [0.44, 0.45, 0.43]},
                    "ss_sq_psr": {"task_values": [0.39, 0.40, 0.38]},
                },
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fsmol-cliff",
            "validate-hypotheses",
            "--input",
            str(validate_input),
            "--output",
            str(validate_output),
        ],
    )

    assert main() == 0
    payload = json.loads(validate_output.read_text())
    assert payload["h2"]["accepted"] is True
    assert payload["h3"]["accepted"] is True


def test_aggregate_command_can_read_task_result_parquet(tmp_path: Path, monkeypatch) -> None:
    parquet_input = tmp_path / "task_results.parquet"
    output_file = tmp_path / "aggregate.json"

    from fsmol_cliff.io import write_parquet

    write_parquet(
        parquet_input,
        [
            {"task_id": "t1", "seed": 0, "split_type": "standard", "metric": "q_psr", "score": 0.4},
            {"task_id": "t1", "seed": 1, "split_type": "standard", "metric": "q_psr", "score": 0.6},
            {"task_id": "t2", "seed": 0, "split_type": "standard", "metric": "q_psr", "score": 0.2},
            {"task_id": "t2", "seed": 1, "split_type": "standard", "metric": "q_psr", "score": 0.4},
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["fsmol-cliff", "aggregate", "--input", str(parquet_input), "--output", str(output_file)],
    )

    assert main() == 0
    payload = json.loads(output_file.read_text())
    assert payload[0]["metric"] == "q_psr"
    assert payload[0]["score"] == 0.4
