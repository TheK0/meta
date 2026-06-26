from __future__ import annotations

from pathlib import Path
import json

from fsmol_cliff.runner import (
    build_maml_legacy_smoke_command,
    convert_legacy_maml_outputs_to_episode_results,
    evaluate_release_with_maml_legacy,
)


def test_build_maml_legacy_smoke_command_contains_expected_env_and_paths() -> None:
    command = build_maml_legacy_smoke_command(
        release_dir=Path("/tmp/release"),
        data_dir=Path("/tmp/data"),
        checkpoint_path=Path("/tmp/checkpoints/maml.pkl"),
        output_path=Path("/tmp/out.json"),
        profile="relaxed",
        split_type="standard",
        task_id="CHEMBL1119333",
        seed=2,
        legacy_env_name="fsmol-maml-legacy",
        max_episodes=3,
    )

    assert command[:4] == ["conda", "run", "-n", "fsmol-maml-legacy"]
    assert "fsmol_cliff.maml_legacy_runner" in command
    assert "--task-id" in command
    assert "CHEMBL1119333" in command
    assert "--seed" in command
    assert "2" in command
    assert "--profile" in command
    assert "relaxed" in command
    assert str(Path("/tmp/checkpoints/maml.pkl")) in command


def test_convert_legacy_maml_outputs_to_episode_results_maps_scores_back_into_main_schema() -> None:
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
    legacy_outputs = [
        {
            "task_id": "CHEMBL1",
            "split_type": "standard",
            "episode_id": 0,
            "scores": {"qa": 0.8, "qn": 0.2},
        }
    ]

    results = convert_legacy_maml_outputs_to_episode_results(
        episodes=[episode],
        assay_context=assay_context,
        legacy_outputs=legacy_outputs,
    )

    assert len(results) == 1
    assert results[0]["metrics"]["q_psr"] == 1.0
    assert results[0]["metrics"]["average_precision_score"] == 1.0


def test_convert_legacy_maml_outputs_to_episode_results_uses_support_scores_for_sq_psr() -> None:
    episode = {
        "task_id": "CHEMBL1",
        "seed": 0,
        "split_type": "adversarial",
        "episode_id": 0,
        "support_pos_ids": ["a1"],
        "support_neg_ids": ["n1"],
        "query_pos_ids": ["qa"],
        "query_neg_ids": ["qn"],
        "injected_pairs": [
            {
                "assay_id": "CHEMBL1",
                "anchor_id": "a1",
                "neg_id": "qn",
                "sim": 0.95,
                "gap_abs": 1.4,
                "same_scaffold": True,
                "pair_type": "cliff",
                "anchor_label": 1,
                "neg_label": 0,
            }
        ],
    }
    assay_context = {
        "labels": {"a1": 1, "n1": 0, "qa": 1, "qn": 0},
        "cliff_pairs": [],
        "noncliff_pairs": [],
    }
    legacy_outputs = [
        {
            "task_id": "CHEMBL1",
            "split_type": "adversarial",
            "episode_id": 0,
            "scores": {"a1": 0.9, "qa": 0.8, "qn": 0.2},
        }
    ]

    results = convert_legacy_maml_outputs_to_episode_results(
        episodes=[episode],
        assay_context=assay_context,
        legacy_outputs=legacy_outputs,
    )

    assert results[0]["metrics"]["sq_psr"] == 1.0


def test_evaluate_release_with_maml_legacy_invokes_legacy_outputs_and_writes_rows(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    pd.DataFrame(
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
        ]
    ).to_parquet(release_dir / "episodes_standard.parquet", index=False)
    pd.DataFrame([]).to_parquet(release_dir / "episodes_adversarial.parquet", index=False)
    pd.DataFrame(
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0},
        ]
    ).to_parquet(assay_dir / "molecule_annotations.parquet", index=False)
    (assay_dir / "pairs.jsonl").write_text(
        json.dumps(
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
        )
        + "\n"
    )

    def fake_run_smoke(**kwargs):
        return {
            "standard": [
                {
                    "task_id": "CHEMBL1",
                    "split_type": "standard",
                    "episode_id": 0,
                    "scores": {"qa": 0.8, "qn": 0.2},
                }
            ]
        }

    monkeypatch.setattr("fsmol_cliff.runner.run_maml_legacy_smoke", fake_run_smoke)

    rows = evaluate_release_with_maml_legacy(
        release_dir=release_dir,
        data_dir=Path("/tmp/data"),
        checkpoint_path=Path("/tmp/maml.pkl"),
        output_path=tmp_path / "maml_rows.parquet",
        split_types=("standard",),
        profile="strict",
        task_ids=("CHEMBL1",),
    )

    assert len(rows) > 0
    q_psr_row = next(row for row in rows if row["metric"] == "q_psr")
    assert q_psr_row["score"] == 1.0


def test_evaluate_release_with_maml_legacy_reads_profile_specific_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    import pandas as pd

    release_dir = tmp_path / "release"
    assay_dir = release_dir / "assays" / "CHEMBL1"
    assay_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "task_id": "CHEMBL1",
                "seed": 0,
                "profile": "relaxed",
                "split_type": "standard",
                "episode_id": 0,
                "support_pos_ids": ["a1"],
                "support_neg_ids": ["n1"],
                "query_pos_ids": ["qa"],
                "query_neg_ids": ["qn"],
                "injected_pairs": [],
            }
        ]
    ).to_parquet(release_dir / "episodes_standard_relaxed.parquet", index=False)
    pd.DataFrame([]).to_parquet(release_dir / "episodes_adversarial_relaxed.parquet", index=False)
    pd.DataFrame(
        [
            {"molecule_id": "a1", "canonical_isomeric_smiles": "CCO", "label": 1},
            {"molecule_id": "n1", "canonical_isomeric_smiles": "CCN", "label": 0},
            {"molecule_id": "qa", "canonical_isomeric_smiles": "CCF", "label": 1},
            {"molecule_id": "qn", "canonical_isomeric_smiles": "CCC", "label": 0},
        ]
    ).to_parquet(assay_dir / "molecule_annotations.parquet", index=False)
    (assay_dir / "pairs_relaxed.jsonl").write_text(
        json.dumps(
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
        )
        + "\n"
    )

    def fake_run_smoke(**kwargs):
        assert kwargs["profile"] == "relaxed"
        return {
            "standard": [
                {
                    "task_id": "CHEMBL1",
                    "split_type": "standard",
                    "episode_id": 0,
                    "scores": {"qa": 0.8, "qn": 0.2},
                }
            ]
        }

    monkeypatch.setattr("fsmol_cliff.runner.run_maml_legacy_smoke", fake_run_smoke)

    rows = evaluate_release_with_maml_legacy(
        release_dir=release_dir,
        data_dir=Path("/tmp/data"),
        checkpoint_path=Path("/tmp/maml.pkl"),
        output_path=tmp_path / "maml_rows_relaxed.parquet",
        split_types=("standard",),
        profile="relaxed",
        task_ids=("CHEMBL1",),
    )

    assert len(rows) > 0
    assert {row["profile"] for row in rows} == {"relaxed"}
