from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Sequence

import pandas as pd

from .benchmark import default_benchmark_manifest
from .constants import DEFAULT_EPISODES_PER_SPLIT, DEFAULT_SEEDS, EpisodeConfig
from .episodes import compute_m_avail
from .io import write_json
from .manifests import build_adversarial_episode_manifests, build_standard_episode_manifests
from .models import PairRecord
from .pipeline import build_assay_asset_bundle
from .task_selection import cliff_richness_score, is_adv_eligible, is_benchmark_eligible, rank_tasks_for_topk


def build_release_bundle(
    *,
    data_dir: Path,
    output_dir: Path,
    task_list_file: Path | None = None,
    episode_config: EpisodeConfig = EpisodeConfig(),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    episodes_per_split: int = DEFAULT_EPISODES_PER_SPLIT,
    benchmark_version: str = "v3.0",
    fsmol_data_version: str = "<fixed_version>",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    assay_root = output_dir / "assays"
    assay_root.mkdir(exist_ok=True)

    task_files = discover_task_files(data_dir, task_list_file=task_list_file)
    task_summaries = []
    for task_file in task_files:
        bundle = build_assay_asset_bundle(task_file=task_file, output_dir=assay_root / assay_id_from_path(task_file))
        summary = summarize_assay_bundle(bundle)
        task_summaries.append(summary)

    eligible_tasks = [summary["assay_id"] for summary in task_summaries if is_benchmark_eligible(summary)]
    top_30 = [summary["assay_id"] for summary in rank_tasks_for_topk([summary for summary in task_summaries if is_benchmark_eligible(summary)], limit=30)]
    adv_eligible = [summary["assay_id"] for summary in task_summaries if is_benchmark_eligible(summary) and is_adv_eligible(summary)]

    standard_manifests = []
    adversarial_manifests = []
    bundles_by_id = {summary["assay_id"]: summary for summary in task_summaries}
    for summary in task_summaries:
        if summary["assay_id"] not in eligible_tasks:
            continue
        standard_manifests.extend(
            build_standard_episode_manifests(
                task_id=summary["assay_id"],
                positive_ids=summary["positive_ids"],
                negative_ids=summary["negative_ids"],
                episode_config=episode_config,
                seeds=seeds,
                episodes_per_seed=episodes_per_split,
            )
        )
        if summary["assay_id"] in adv_eligible:
            adversarial_manifests.extend(
                build_adversarial_episode_manifests(
                    task_id=summary["assay_id"],
                    positive_ids=summary["positive_ids"],
                    negative_ids=summary["negative_ids"],
                    cliff_pairs=[PairRecord(**pair) for pair in summary["cliff_pairs"]],
                    anchor_to_hardnegs=summary["anchor_to_hardnegs"],
                    episode_config=episode_config,
                    seeds=seeds,
                    episodes_per_seed=episodes_per_split,
                )
            )

    write_json(output_dir / "fsmol_cliff_all.json", eligible_tasks)
    write_json(output_dir / "fsmol_cliff_30.json", top_30)
    write_json(output_dir / "fsmol_cliff_adv_eligible.json", adv_eligible)
    _write_manifest(
        output_dir / "benchmark_manifest.json",
        benchmark_version=benchmark_version,
        fsmol_data_version=fsmol_data_version,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_split=episodes_per_split,
    )
    pd.DataFrame(task_summaries).to_parquet(output_dir / "task_summaries.parquet", index=False)
    pd.DataFrame(standard_manifests).to_parquet(output_dir / "episodes_standard.parquet", index=False)
    pd.DataFrame(adversarial_manifests).to_parquet(output_dir / "episodes_adversarial.parquet", index=False)

    return {
        "eligible_tasks": eligible_tasks,
        "top_30": top_30,
        "adv_eligible": adv_eligible,
        "num_standard_episodes": len(standard_manifests),
        "num_adversarial_episodes": len(adversarial_manifests),
    }


def discover_task_files(data_dir: Path, *, task_list_file: Path | None = None) -> list[Path]:
    if (data_dir / "test").exists():
        search_root = data_dir / "test"
    else:
        search_root = data_dir
    files = sorted([*search_root.glob("*.jsonl.gz"), *search_root.glob("*.jsonl")])
    if task_list_file is None:
        return files

    payload = json.loads(task_list_file.read_text())
    allowed = set(payload.get("test", []))
    return [task_file for task_file in files if assay_id_from_path(task_file) in allowed]


def summarize_assay_bundle(bundle: dict) -> dict:
    cliff_pairs = bundle["pair_groups"]["cliff"]
    noncliff_pairs = bundle["pair_groups"]["highsim_noncliff"]
    all_highsim = bundle["pair_groups"]["highsim_discordant"]
    cliff_pair_records = [PairRecord(**pair) for pair in cliff_pairs]
    anchor_ids = sorted({pair["anchor_id"] for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair["neg_id"] for pair in cliff_pairs})
    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pair_records) if cliff_pairs else 0
    similarities = [pair["sim"] for pair in all_highsim]
    return {
        "assay_id": bundle["assay_id"],
        "num_valid_molecules": len(bundle["molecules"]),
        "num_positive_molecules": sum(mol["label"] == 1 for mol in bundle["molecules"]),
        "num_negative_molecules": sum(mol["label"] == 0 for mol in bundle["molecules"]),
        "num_cliff_pairs": len(cliff_pairs),
        "num_anchor_molecules": len(anchor_ids),
        "num_cliff_negatives": len(cliff_neg_ids),
        "num_noncliff_highsim_pairs": len(noncliff_pairs),
        "median_sim": float(median(similarities)) if similarities else 0.0,
        "m_avail": m_avail,
        "positive_ids": [mol["molecule_id"] for mol in bundle["molecules"] if mol["label"] == 1],
        "negative_ids": [mol["molecule_id"] for mol in bundle["molecules"] if mol["label"] == 0],
        "cliff_pairs": cliff_pairs,
        "anchor_to_hardnegs": bundle["hard_negatives"],
        "cliff_richness_score": cliff_richness_score(
            {
                "num_cliff_pairs": len(cliff_pairs),
                "num_positive_molecules": sum(mol["label"] == 1 for mol in bundle["molecules"]),
                "num_negative_molecules": sum(mol["label"] == 0 for mol in bundle["molecules"]),
                "num_anchor_molecules": len(anchor_ids),
            }
        ) if cliff_pairs else 0.0,
    }


def assay_id_from_path(task_file: Path) -> str:
    name = task_file.name
    if name.endswith(".jsonl.gz"):
        return name[: -len(".jsonl.gz")]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return task_file.stem


def _write_manifest(
    path: Path,
    *,
    benchmark_version: str,
    fsmol_data_version: str,
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_split: int,
) -> None:
    payload = default_benchmark_manifest()
    payload["benchmark_version"] = benchmark_version
    payload["fsmol_data_version"] = fsmol_data_version
    payload["episode_config"] = episode_config.to_dict()
    payload["seeds"] = list(seeds)
    payload["episodes_per_split"] = episodes_per_split
    write_json(path, payload)
