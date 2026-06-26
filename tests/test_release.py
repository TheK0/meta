from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from fsmol_cliff.constants import EpisodeConfig, PROFILE_SPECS, RELAXED_PROFILE
from fsmol_cliff.release import build_episode_variant_release, build_release_bundle


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
                "CanonicalIsomericSmiles": "C" * index,
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
                "CanonicalIsomericSmiles": ("C" * (index - 1)) + "N",
            }
        )
    return records


def _normalize(value):
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


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
                "CanonicalIsomericSmiles": "C" * index,
            }
        )
        records.append(
            {
                "Assay_ID": "CHEMBL_SMALL",
                "compound_id": f"sn{index:02d}",
                "Y": 0,
                "Relation": "=",
                "LogRegressionProperty": 6.0,
                "CanonicalIsomericSmiles": ("C" * (index - 1)) + "N",
            }
        )
    return records


def _patch_release_test_chemistry(monkeypatch, fake_similarity) -> None:
    monkeypatch.setattr("fsmol_cliff.assets.tanimoto_similarity", fake_similarity)
    monkeypatch.setattr(
        "fsmol_cliff.release.default_benchmark_manifest",
        lambda: {
            "profiles": {
                "strict": PROFILE_SPECS["strict"].to_dict(),
                "relaxed": PROFILE_SPECS["relaxed"].to_dict(),
            },
            "built_profiles": [],
        },
    )


def test_profile_specs_register_auxiliary_relaxed_coverage_extension_profiles() -> None:
    base_profile = RELAXED_PROFILE.to_dict()
    relaxed_covext_10_10 = PROFILE_SPECS["relaxed_covext_10_10"].to_dict()
    relaxed_covext_10_5 = PROFILE_SPECS["relaxed_covext_10_5"].to_dict()

    assert relaxed_covext_10_10["similarity_threshold"] == 0.80
    assert relaxed_covext_10_10["activity_gap_threshold"] == 1.0
    assert relaxed_covext_10_10["min_cliff_pairs"] == 10
    assert relaxed_covext_10_10["min_noncliff_pairs"] == 10
    assert relaxed_covext_10_5["similarity_threshold"] == 0.80
    assert relaxed_covext_10_5["activity_gap_threshold"] == 1.0
    assert relaxed_covext_10_5["min_cliff_pairs"] == 10
    assert relaxed_covext_10_5["min_noncliff_pairs"] == 5

    for key, value in base_profile.items():
        if key in {"name", "min_cliff_pairs", "min_noncliff_pairs"}:
            continue
        assert relaxed_covext_10_10[key] == value
        assert relaxed_covext_10_5[key] == value


def test_build_release_bundle_adds_auxiliary_profile_to_release_manifest(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "fsmol"
    test_dir = data_dir / "test"
    output_dir = tmp_path / "release"

    _write_task(test_dir / "CHEMBL_ELIGIBLE.jsonl.gz", _eligible_records())

    def fake_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
        if not smiles_a or not smiles_b:
            return None
        return 0.9 if smiles_a[0] != smiles_b[0] else 0.1

    _patch_release_test_chemistry(monkeypatch, fake_similarity)

    build_release_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        profile="relaxed_covext_10_10",
        fsmol_data_version="fsmol-test",
    )

    benchmark_manifest = json.loads((output_dir / "benchmark_manifest.json").read_text())
    assert set(benchmark_manifest["profiles"]) == {"strict", "relaxed", "relaxed_covext_10_10"}
    assert benchmark_manifest["profiles"]["relaxed_covext_10_10"]["min_cliff_pairs"] == 10
    assert benchmark_manifest["profiles"]["relaxed_covext_10_10"]["min_noncliff_pairs"] == 10
    assert benchmark_manifest["built_profiles"] == ["relaxed_covext_10_10"]


def test_build_release_bundle_writes_profile_aware_task_lists_and_manifests(
    tmp_path: Path, monkeypatch
) -> None:
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

        def positive_index(smiles: str) -> int | None:
            return len(smiles) if set(smiles) == {"C"} else None

        def negative_index(smiles: str) -> int | None:
            if not smiles.endswith("N"):
                return None
            if set(smiles[:-1]) <= {"C"}:
                return len(smiles)
            return None

        pos_index = positive_index(smiles_a)
        neg_index = negative_index(smiles_b)
        if pos_index is None or neg_index is None:
            pos_index = positive_index(smiles_b)
            neg_index = negative_index(smiles_a)
        if pos_index is None or neg_index is None:
            return 0.1
        if pos_index <= 15 and neg_index <= 15 and abs(pos_index - neg_index) <= 1:
            return 0.9
        if 16 <= pos_index <= 25 and pos_index == neg_index:
            return 0.9
        return 0.1

    _patch_release_test_chemistry(monkeypatch, fake_similarity)

    strict_release = build_release_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        task_list_file=task_list_file,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        profile="strict",
        fsmol_data_version="fsmol-test",
    )
    relaxed_release = build_release_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        task_list_file=task_list_file,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        profile="relaxed",
        fsmol_data_version="fsmol-test",
    )

    assert strict_release["eligible_tasks"] == ["CHEMBL_ELIGIBLE"]
    assert relaxed_release["eligible_tasks"] == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_strict_all.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_strict_30.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_strict_adv_eligible.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_relaxed_all.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_relaxed_30.json").read_text()) == ["CHEMBL_ELIGIBLE"]
    assert json.loads((output_dir / "fsmol_cliff_relaxed_adv_eligible.json").read_text()) == ["CHEMBL_ELIGIBLE"]

    benchmark_manifest = json.loads((output_dir / "benchmark_manifest.json").read_text())
    assert benchmark_manifest["benchmark_version"] == "v4.0"
    assert benchmark_manifest["fsmol_data_version"] == "fsmol-test"
    assert benchmark_manifest["episode_config"]["query_per_class"] == 4
    assert benchmark_manifest["built_profiles"] == ["relaxed", "strict"]
    assert benchmark_manifest["profiles"]["strict"]["similarity_threshold"] == 0.85
    assert benchmark_manifest["profiles"]["relaxed"]["similarity_threshold"] == 0.8

    model_execution_metadata = json.loads((output_dir / "model_execution_metadata.json").read_text())
    assert model_execution_metadata["benchmark_version"] == "v4.0"
    assert model_execution_metadata["models"]["kNN"]["support_side_scoring"] is not None
    assert model_execution_metadata["models"]["RF"]["support_side_scoring"] is not None
    assert model_execution_metadata["models"]["kNN-cliff-aware"]["support_side_scoring"] is not None

    reproducibility_note = (output_dir / "release_reproducibility.md").read_text()
    assert "python -m fsmol_cliff.cli build-release" in reproducibility_note
    assert "src/fsmol_cliff/release.py" in reproducibility_note

    assert (output_dir / "assays" / "CHEMBL_ELIGIBLE" / "pairs_strict.jsonl").exists()
    assert (output_dir / "assays" / "CHEMBL_ELIGIBLE" / "pairs_relaxed.jsonl").exists()
    assert (output_dir / "assays" / "CHEMBL_ELIGIBLE" / "anchor_to_hardnegs_strict.json").exists()
    assert (output_dir / "assays" / "CHEMBL_ELIGIBLE" / "anchor_to_hardnegs_relaxed.json").exists()
    assert not (output_dir / "fsmol_cliff_all.json").exists()
    assert not (output_dir / "episodes_standard.parquet").exists()
    assert not (output_dir / "episodes_adversarial.parquet").exists()

    standard = pd.read_parquet(output_dir / "episodes_standard_strict.parquet")
    adversarial = pd.read_parquet(output_dir / "episodes_adversarial_strict.parquet")
    assert list(standard["task_id"]) == ["CHEMBL_ELIGIBLE"]
    assert list(adversarial["task_id"]) == ["CHEMBL_ELIGIBLE"]
    assert set(standard["profile"]) == {"strict"}
    assert set(adversarial["profile"]) == {"strict"}

    relaxed_standard = pd.read_parquet(output_dir / "episodes_standard_relaxed.parquet")
    relaxed_adversarial = pd.read_parquet(output_dir / "episodes_adversarial_relaxed.parquet")
    assert list(relaxed_standard["task_id"]) == ["CHEMBL_ELIGIBLE"]
    assert list(relaxed_adversarial["task_id"]) == ["CHEMBL_ELIGIBLE"]
    assert set(relaxed_standard["profile"]) == {"relaxed"}
    assert set(relaxed_adversarial["profile"]) == {"relaxed"}

    strict_task_summaries = pd.read_parquet(output_dir / "task_summaries_strict.parquet")
    relaxed_task_summaries = pd.read_parquet(output_dir / "task_summaries_relaxed.parquet")
    assert set(strict_task_summaries["assay_id"]) == {"CHEMBL_ELIGIBLE", "CHEMBL_SMALL"}
    assert set(relaxed_task_summaries["assay_id"]) == {"CHEMBL_ELIGIBLE", "CHEMBL_SMALL"}
    assert strict_task_summaries["anchor_to_hardnegs"].map(type).eq(str).all()
    assert relaxed_task_summaries["anchor_to_hardnegs"].map(type).eq(str).all()


def test_build_release_bundle_can_write_query_targeted_episode_variant(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "fsmol"
    test_dir = data_dir / "test"
    baseline_output_dir = tmp_path / "release_baseline"
    variant_output_dir = tmp_path / "release_variant"

    _write_task(test_dir / "CHEMBL_ELIGIBLE.jsonl.gz", _eligible_records())

    def fake_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
        if not smiles_a or not smiles_b:
            return None

        def positive_index(smiles: str) -> int | None:
            return len(smiles) if set(smiles) == {"C"} else None

        def negative_index(smiles: str) -> int | None:
            if not smiles.endswith("N"):
                return None
            if set(smiles[:-1]) <= {"C"}:
                return len(smiles)
            return None

        pos_index = positive_index(smiles_a)
        neg_index = negative_index(smiles_b)
        if pos_index is None or neg_index is None:
            pos_index = positive_index(smiles_b)
            neg_index = negative_index(smiles_a)
        if pos_index is None or neg_index is None:
            return 0.1
        if pos_index <= 15 and neg_index <= 15 and abs(pos_index - neg_index) <= 1:
            return 0.9
        if 16 <= pos_index <= 25 and pos_index == neg_index:
            return 0.9
        return 0.1

    _patch_release_test_chemistry(monkeypatch, fake_similarity)

    build_release_bundle(
        data_dir=data_dir,
        output_dir=baseline_output_dir,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        profile="relaxed",
        fsmol_data_version="fsmol-test",
    )
    build_release_bundle(
        data_dir=data_dir,
        output_dir=variant_output_dir,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        profile="relaxed",
        fsmol_data_version="fsmol-test",
        adversarial_episode_variant="query_targeted_support_neg",
    )

    benchmark_manifest = json.loads((variant_output_dir / "benchmark_manifest.json").read_text())
    assert benchmark_manifest["adversarial_episode_variant"] == "query_targeted_support_neg"

    note = (variant_output_dir / "episode_protocol_note.md").read_text()
    assert "Query-Targeted Support Negatives" in note

    baseline_standard = pd.read_parquet(baseline_output_dir / "episodes_standard_relaxed.parquet").to_dict(orient="records")
    variant_standard = pd.read_parquet(variant_output_dir / "episodes_standard_relaxed.parquet").to_dict(orient="records")
    assert [_normalize(row) for row in variant_standard] == [_normalize(row) for row in baseline_standard]

    baseline_episode = pd.read_parquet(
        baseline_output_dir / "episodes_adversarial_relaxed.parquet"
    ).to_dict(orient="records")[0]
    variant_episode = pd.read_parquet(
        variant_output_dir / "episodes_adversarial_relaxed.parquet"
    ).to_dict(orient="records")[0]

    for key in ("support_pos_ids", "query_pos_ids", "query_neg_ids", "injected_pairs"):
        assert _normalize(variant_episode[key]) == _normalize(baseline_episode[key])
    assert _normalize(variant_episode["support_neg_ids"]) != _normalize(baseline_episode["support_neg_ids"])


def test_build_episode_variant_release_rewrites_only_adversarial_manifests(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "fsmol"
    test_dir = data_dir / "test"
    base_output_dir = tmp_path / "release_base"
    variant_output_dir = tmp_path / "release_variant"

    _write_task(test_dir / "CHEMBL_ELIGIBLE.jsonl.gz", _eligible_records())

    def fake_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
        if not smiles_a or not smiles_b:
            return None

        def positive_index(smiles: str) -> int | None:
            return len(smiles) if set(smiles) == {"C"} else None

        def negative_index(smiles: str) -> int | None:
            if not smiles.endswith("N"):
                return None
            if set(smiles[:-1]) <= {"C"}:
                return len(smiles)
            return None

        pos_index = positive_index(smiles_a)
        neg_index = negative_index(smiles_b)
        if pos_index is None or neg_index is None:
            pos_index = positive_index(smiles_b)
            neg_index = negative_index(smiles_a)
        if pos_index is None or neg_index is None:
            return 0.1
        if pos_index <= 15 and neg_index <= 15 and abs(pos_index - neg_index) <= 1:
            return 0.9
        if 16 <= pos_index <= 25 and pos_index == neg_index:
            return 0.9
        return 0.1

    _patch_release_test_chemistry(monkeypatch, fake_similarity)

    build_release_bundle(
        data_dir=data_dir,
        output_dir=base_output_dir,
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_split=1,
        profile="relaxed",
        fsmol_data_version="fsmol-test",
    )

    build_episode_variant_release(
        base_release_dir=base_output_dir,
        output_dir=variant_output_dir,
        profile="relaxed",
        adversarial_episode_variant="query_targeted_support_neg",
    )

    base_manifest = json.loads((base_output_dir / "benchmark_manifest.json").read_text())
    variant_manifest = json.loads((variant_output_dir / "benchmark_manifest.json").read_text())
    assert variant_manifest["profiles"] == base_manifest["profiles"]
    assert variant_manifest["built_profiles"] == base_manifest["built_profiles"]
    assert variant_manifest["adversarial_episode_variant"] == "query_targeted_support_neg"

    base_standard = pd.read_parquet(base_output_dir / "episodes_standard_relaxed.parquet").to_dict(orient="records")
    variant_standard = pd.read_parquet(variant_output_dir / "episodes_standard_relaxed.parquet").to_dict(orient="records")
    assert [_normalize(row) for row in variant_standard] == [_normalize(row) for row in base_standard]

    base_adversarial = pd.read_parquet(base_output_dir / "episodes_adversarial_relaxed.parquet").to_dict(orient="records")[0]
    variant_adversarial = pd.read_parquet(variant_output_dir / "episodes_adversarial_relaxed.parquet").to_dict(orient="records")[0]
    for key in ("support_pos_ids", "query_pos_ids", "query_neg_ids", "injected_pairs"):
        assert _normalize(variant_adversarial[key]) == _normalize(base_adversarial[key])
    assert _normalize(variant_adversarial["support_neg_ids"]) != _normalize(base_adversarial["support_neg_ids"])

    assert (variant_output_dir / "assays" / "CHEMBL_ELIGIBLE" / "molecule_annotations.parquet").exists()
    assert "build-episode-variant-release" in (variant_output_dir / "release_reproducibility.md").read_text()
