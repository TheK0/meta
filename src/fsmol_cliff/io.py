from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def resolve_manifest_path(release_dir: Path, *, split_type: str, profile: str) -> Path:
    profile_path = release_dir / f"episodes_{split_type}_{profile}.parquet"
    legacy_path = release_dir / f"episodes_{split_type}.parquet"
    return profile_path if profile_path.exists() else legacy_path


def resolve_assay_path(assay_dir: Path, *, stem: str, suffix: str, profile: str) -> Path:
    profile_path = assay_dir / f"{stem}_{profile}{suffix}"
    legacy_path = assay_dir / f"{stem}{suffix}"
    return profile_path if profile_path.exists() else legacy_path


def load_assay_context(release_dir: Path, task_id: str, *, profile: str = "strict") -> dict:
    assay_dir = release_dir / "assays" / task_id
    annotations = pd.read_parquet(assay_dir / "molecule_annotations.parquet").to_dict(orient="records")
    pairs_path = resolve_assay_path(assay_dir, stem="pairs", suffix=".jsonl", profile=profile)
    cliff_pairs = []
    noncliff_pairs = []
    with pairs_path.open() as handle:
        for line in handle:
            pair = json.loads(line)
            if pair["pair_type"] == "cliff":
                cliff_pairs.append(pair)
            else:
                noncliff_pairs.append(pair)
    records_by_id = {record["molecule_id"]: record for record in annotations}
    labels = {record["molecule_id"]: int(record["label"]) for record in annotations}
    anchor_path = resolve_assay_path(assay_dir, stem="anchor_to_hardnegs", suffix=".json", profile=profile)
    anchor_to_hardnegs = json.loads(anchor_path.read_text()) if anchor_path.exists() else {}
    return {
        "records_by_id": records_by_id,
        "labels": labels,
        "cliff_pairs": cliff_pairs,
        "noncliff_pairs": noncliff_pairs,
        "anchor_to_hardnegs": anchor_to_hardnegs,
    }
