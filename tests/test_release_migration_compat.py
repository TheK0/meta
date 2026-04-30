from __future__ import annotations

from pathlib import Path

import pandas as pd

from fsmol_cliff.io import resolve_assay_path, resolve_manifest_path


def testresolve_manifest_path_prefers_profile_name_but_falls_back_to_legacy(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    legacy = release_dir / "episodes_standard.parquet"
    legacy.write_text("legacy")
    assert resolve_manifest_path(release_dir, split_type="standard", profile="relaxed") == legacy

    profile_path = release_dir / "episodes_standard_relaxed.parquet"
    profile_path.write_text("profile")
    assert resolve_manifest_path(release_dir, split_type="standard", profile="relaxed") == profile_path


def testresolve_assay_path_prefers_profile_name_but_falls_back_to_legacy(tmp_path: Path) -> None:
    assay_dir = tmp_path / "assay"
    assay_dir.mkdir()

    legacy = assay_dir / "pairs.jsonl"
    legacy.write_text("{}\n")
    assert resolve_assay_path(assay_dir, stem="pairs", suffix=".jsonl", profile="relaxed") == legacy

    profile_path = assay_dir / "pairs_relaxed.jsonl"
    profile_path.write_text("{}\n")
    assert resolve_assay_path(assay_dir, stem="pairs", suffix=".jsonl", profile="relaxed") == profile_path
