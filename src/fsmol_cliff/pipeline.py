from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .assets import build_assay_assets
from .chem import murcko_scaffold_smiles
from .constants import DEFAULT_PROTOCOL_CONSTANTS
from .io import write_json, write_jsonl, write_parquet


def load_task_records(task_file: Path) -> list[dict[str, Any]]:
    opener = gzip.open if task_file.suffix == ".gz" else open
    with opener(task_file, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_assay_asset_bundle(task_file: Path, output_dir: Path) -> dict[str, Any]:
    return build_assay_asset_bundle_for_profile(task_file, output_dir)


def build_assay_asset_bundle_for_profile(
    task_file: Path,
    output_dir: Path,
    *,
    profile: str | None = None,
    tau: float | None = None,
    delta: float | None = None,
    hard_negative_pool_size: int | None = None,
) -> dict[str, Any]:
    records = load_task_records(task_file)
    assay_id = _assay_id_for_records(task_file, records)
    bundle = build_assay_assets(
        assay_id,
        records,
        tau=tau if tau is not None else DEFAULT_PROTOCOL_CONSTANTS.similarity_threshold,
        delta=delta if delta is not None else DEFAULT_PROTOCOL_CONSTANTS.activity_gap_threshold,
        hard_negative_pool_size=hard_negative_pool_size if hard_negative_pool_size is not None else DEFAULT_PROTOCOL_CONSTANTS.hard_negative_pool_size,
    )

    molecules = [
        {
            "molecule_id": record["molecule_id"],
            "canonical_isomeric_smiles": record["canonical_isomeric_smiles"],
            "label": record["label"],
            "r": record["r"],
            "scaffold_smiles": murcko_scaffold_smiles(record["canonical_isomeric_smiles"])
            or "EMPTY_SCAFFOLD",
        }
        for record in bundle["molecules"]
    ]
    pairs = bundle["pairs"]["cliff"] + bundle["pairs"]["highsim_noncliff"]
    hard_negatives = {
        anchor_id: [pair["neg_id"] for pair in pool]
        for anchor_id, pool in bundle["hard_negative_pools"].items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_name = "pairs.jsonl" if profile is None else f"pairs_{profile}.jsonl"
    hardnegs_name = "anchor_to_hardnegs.json" if profile is None else f"anchor_to_hardnegs_{profile}.json"
    diagnostics_name = "diagnostics.json" if profile is None else f"diagnostics_{profile}.json"
    write_jsonl(output_dir / pairs_name, pairs)
    write_json(output_dir / hardnegs_name, hard_negatives)
    write_parquet(output_dir / "molecule_annotations.parquet", molecules)
    write_json(output_dir / diagnostics_name, bundle["diagnostics"])

    return {
        "assay_id": assay_id,
        "profile": profile,
        "molecules": molecules,
        "pairs": pairs,
        "pair_groups": bundle["pairs"],
        "hard_negatives": hard_negatives,
        "hard_negative_pools": bundle["hard_negative_pools"],
        "diagnostics": bundle["diagnostics"],
    }


def _assay_id_for_records(task_file: Path, records: list[dict[str, Any]]) -> str:
    if records and records[0].get("Assay_ID"):
        return str(records[0]["Assay_ID"])
    name = task_file.name
    if name.endswith(".jsonl.gz"):
        return name[: -len(".jsonl.gz")]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return task_file.stem
