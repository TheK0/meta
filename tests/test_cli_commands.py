from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from fsmol_cliff.cli import main


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
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
