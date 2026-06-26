from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import median
from typing import Sequence

import pandas as pd

from .benchmark import default_benchmark_manifest
from .constants import DEFAULT_EPISODES_PER_SPLIT, DEFAULT_SEEDS, EpisodeConfig, PROFILE_SPECS
from .episodes import compute_m_avail
from .io import write_json
from .manifests import (
    build_anchor_coverage_first_adversarial_episode_manifests,
    build_adversarial_episode_manifests,
    build_paired_hardness_balanced_adversarial_episode_manifests,
    build_query_cluster_separation_by_anchor_neg_mix_adversarial_episode_manifests,
    build_query_cluster_separation_by_neg_diversity_adversarial_episode_manifests,
    build_query_targeted_adversarial_episode_manifests,
    build_same_scaffold_query_targeted_adversarial_episode_manifests,
    build_standard_episode_manifests,
)
from .models import PairRecord
from .pipeline import build_assay_asset_bundle_for_profile
from .task_selection import cliff_richness_score, is_adv_eligible, is_benchmark_eligible, rank_tasks_for_topk

ADVERSARIAL_EPISODE_VARIANTS = {
    "anchor_coverage_first": build_anchor_coverage_first_adversarial_episode_manifests,
    "baseline": build_adversarial_episode_manifests,
    "paired_hardness_balanced": build_paired_hardness_balanced_adversarial_episode_manifests,
    "query_cluster_separation_by_anchor_neg_mix": build_query_cluster_separation_by_anchor_neg_mix_adversarial_episode_manifests,
    "query_cluster_separation_by_neg_diversity": build_query_cluster_separation_by_neg_diversity_adversarial_episode_manifests,
    "query_targeted_support_neg": build_query_targeted_adversarial_episode_manifests,
    "same_scaffold_query_targeted": build_same_scaffold_query_targeted_adversarial_episode_manifests,
}


def build_release_bundle(
    *,
    data_dir: Path,
    output_dir: Path,
    task_list_file: Path | None = None,
    episode_config: EpisodeConfig = EpisodeConfig(),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    episodes_per_split: int = DEFAULT_EPISODES_PER_SPLIT,
    profile: str = "strict",
    benchmark_version: str = "v4.0",
    fsmol_data_version: str = "<fixed_version>",
    adversarial_episode_variant: str = "baseline",
) -> dict:
    profile_spec = PROFILE_SPECS[profile]
    adversarial_manifest_builder = ADVERSARIAL_EPISODE_VARIANTS[adversarial_episode_variant]
    output_dir.mkdir(parents=True, exist_ok=True)
    assay_root = output_dir / "assays"
    assay_root.mkdir(exist_ok=True)

    task_files = discover_task_files(data_dir, task_list_file=task_list_file)
    task_summaries = []
    for task_file in task_files:
        bundle = build_assay_asset_bundle_for_profile(
            task_file=task_file,
            output_dir=assay_root / assay_id_from_path(task_file),
            profile=profile,
            tau=profile_spec.constants.similarity_threshold,
            delta=profile_spec.constants.activity_gap_threshold,
            hard_negative_pool_size=profile_spec.constants.hard_negative_pool_size,
        )
        summary = summarize_assay_bundle(bundle)
        task_summaries.append(summary)

    eligible_tasks = [summary["assay_id"] for summary in task_summaries if is_benchmark_eligible(summary, profile=profile_spec)]
    top_30 = [
        summary["assay_id"]
        for summary in rank_tasks_for_topk(
            [summary for summary in task_summaries if is_benchmark_eligible(summary, profile=profile_spec)],
            limit=30,
        )
    ]
    adv_eligible = [
        summary["assay_id"]
        for summary in task_summaries
        if is_benchmark_eligible(summary, profile=profile_spec) and is_adv_eligible(summary, profile=profile_spec)
    ]

    standard_manifests = []
    adversarial_manifests = []
    for summary in task_summaries:
        if summary["assay_id"] not in eligible_tasks:
            continue
        standard_manifests.extend(
            _tag_profile(
                build_standard_episode_manifests(
                task_id=summary["assay_id"],
                positive_ids=summary["positive_ids"],
                negative_ids=summary["negative_ids"],
                episode_config=episode_config,
                seeds=seeds,
                episodes_per_seed=episodes_per_split,
                ),
                profile=profile,
            )
        )
        if summary["assay_id"] in adv_eligible:
            adversarial_manifests.extend(
                _tag_profile(
                    adversarial_manifest_builder(
                        task_id=summary["assay_id"],
                        positive_ids=summary["positive_ids"],
                        negative_ids=summary["negative_ids"],
                        cliff_pairs=[PairRecord(**pair) for pair in summary["cliff_pairs"]],
                        anchor_to_hardnegs=summary["anchor_to_hardnegs"],
                        episode_config=episode_config,
                        seeds=seeds,
                        episodes_per_seed=episodes_per_split,
                    ),
                    profile=profile,
                )
            )

    write_json(output_dir / f"fsmol_cliff_{profile}_all.json", eligible_tasks)
    write_json(output_dir / f"fsmol_cliff_{profile}_30.json", top_30)
    write_json(output_dir / f"fsmol_cliff_{profile}_adv_eligible.json", adv_eligible)
    _write_manifest(
        output_dir / "benchmark_manifest.json",
        benchmark_version=benchmark_version,
        fsmol_data_version=fsmol_data_version,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_split=episodes_per_split,
        built_profile=profile,
        adversarial_episode_variant=adversarial_episode_variant,
    )
    write_json(output_dir / "model_execution_metadata.json", build_model_execution_metadata(benchmark_version=benchmark_version))
    (output_dir / "episode_protocol_note.md").write_text(
        render_episode_protocol_note(adversarial_episode_variant=adversarial_episode_variant)
    )
    (output_dir / "release_reproducibility.md").write_text(
        render_release_reproducibility_markdown(
            benchmark_version=benchmark_version,
            adversarial_episode_variant=adversarial_episode_variant,
        )
    )
    pd.DataFrame(_task_summaries_for_parquet(task_summaries)).to_parquet(
        output_dir / f"task_summaries_{profile}.parquet", index=False
    )
    pd.DataFrame(standard_manifests).to_parquet(output_dir / f"episodes_standard_{profile}.parquet", index=False)
    pd.DataFrame(adversarial_manifests).to_parquet(output_dir / f"episodes_adversarial_{profile}.parquet", index=False)

    return {
        "profile": profile,
        "eligible_tasks": eligible_tasks,
        "top_30": top_30,
        "adv_eligible": adv_eligible,
        "num_standard_episodes": len(standard_manifests),
        "num_adversarial_episodes": len(adversarial_manifests),
        "adversarial_episode_variant": adversarial_episode_variant,
    }


def build_episode_variant_release(
    *,
    base_release_dir: Path,
    output_dir: Path,
    profile: str,
    adversarial_episode_variant: str,
) -> dict:
    if adversarial_episode_variant == "baseline":
        raise ValueError("build_episode_variant_release requires a non-baseline adversarial_episode_variant.")

    benchmark_manifest = json.loads((base_release_dir / "benchmark_manifest.json").read_text())
    task_summary_path = base_release_dir / f"task_summaries_{profile}.parquet"
    if not task_summary_path.exists():
        raise FileNotFoundError(f"Missing task summaries for profile {profile}: {task_summary_path}")

    summary_rows = pd.read_parquet(task_summary_path).to_dict(orient="records")
    adv_eligible = set(json.loads((base_release_dir / f"fsmol_cliff_{profile}_adv_eligible.json").read_text()))
    episode_config = EpisodeConfig(
        support_per_class=int(benchmark_manifest["episode_config"]["support_per_class"]),
        query_per_class=int(benchmark_manifest["episode_config"]["query_per_class"]),
    )
    adversarial_manifest_builder = ADVERSARIAL_EPISODE_VARIANTS[adversarial_episode_variant]

    shutil.copytree(base_release_dir, output_dir)
    _remove_stale_task_results(output_dir)

    adversarial_manifests = []
    for row in summary_rows:
        assay_id = row["assay_id"]
        if assay_id not in adv_eligible:
            continue
        adversarial_manifests.extend(
            _tag_profile(
                adversarial_manifest_builder(
                    task_id=assay_id,
                    positive_ids=json.loads(row["positive_ids"]),
                    negative_ids=json.loads(row["negative_ids"]),
                    cliff_pairs=[PairRecord(**pair) for pair in json.loads(row["cliff_pairs"])],
                    anchor_to_hardnegs=json.loads(row["anchor_to_hardnegs"]),
                    episode_config=episode_config,
                    seeds=benchmark_manifest["seeds"],
                    episodes_per_seed=int(benchmark_manifest["episodes_per_split"]),
                ),
                profile=profile,
            )
        )

    pd.DataFrame(adversarial_manifests).to_parquet(output_dir / f"episodes_adversarial_{profile}.parquet", index=False)
    benchmark_manifest["adversarial_episode_variant"] = adversarial_episode_variant
    write_json(output_dir / "benchmark_manifest.json", benchmark_manifest)
    (output_dir / "episode_protocol_note.md").write_text(
        render_episode_protocol_note(adversarial_episode_variant=adversarial_episode_variant)
    )
    (output_dir / "release_reproducibility.md").write_text(
        render_release_reproducibility_markdown(
            benchmark_version=benchmark_manifest["benchmark_version"],
            adversarial_episode_variant=adversarial_episode_variant,
            build_command_name="build-episode-variant-release",
            base_release_dir=base_release_dir,
        )
    )
    return {
        "profile": profile,
        "base_release_dir": str(base_release_dir),
        "output_dir": str(output_dir),
        "num_adversarial_episodes": len(adversarial_manifests),
        "adversarial_episode_variant": adversarial_episode_variant,
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
        "profile": bundle.get("profile"),
        "num_valid_molecules": len(bundle["molecules"]),
        "num_positive_molecules": sum(mol["label"] == 1 for mol in bundle["molecules"]),
        "num_negative_molecules": sum(mol["label"] == 0 for mol in bundle["molecules"]),
        "num_cliff_pairs": len(cliff_pairs),
        "num_anchor_molecules": len(anchor_ids),
        "num_cliff_negatives": len(cliff_neg_ids),
        "num_noncliff_highsim_pairs": len(noncliff_pairs),
        "num_same_scaffold_cliff_pairs": len(bundle["pair_groups"]["same_scaffold_cliff"]),
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
    built_profile: str,
    adversarial_episode_variant: str,
) -> None:
    payload = default_benchmark_manifest()
    if path.exists():
        payload = json.loads(path.read_text())
    payload["benchmark_version"] = benchmark_version
    payload["fsmol_data_version"] = fsmol_data_version
    payload["episode_config"] = episode_config.to_dict()
    payload["seeds"] = list(seeds)
    payload["episodes_per_split"] = episodes_per_split
    payload.setdefault("profiles", {})
    if built_profile in PROFILE_SPECS:
        payload["profiles"][built_profile] = PROFILE_SPECS[built_profile].to_dict()
    payload["built_profiles"] = sorted({*payload.get("built_profiles", []), built_profile})
    payload["adversarial_episode_variant"] = adversarial_episode_variant
    write_json(path, payload)


def _task_summaries_for_parquet(task_summaries: Sequence[dict]) -> list[dict]:
    serialized = []
    for summary in task_summaries:
        row = dict(summary)
        for key in ("positive_ids", "negative_ids", "cliff_pairs", "anchor_to_hardnegs"):
            row[key] = json.dumps(row.get(key, [] if key != "anchor_to_hardnegs" else {}), sort_keys=True)
        serialized.append(row)
    return serialized


def _tag_profile(manifests: Sequence[dict], *, profile: str) -> list[dict]:
    return [{**manifest, "profile": profile} for manifest in manifests]


def _remove_stale_task_results(release_dir: Path) -> None:
    for path in release_dir.glob("task_results_*"):
        if path.is_file():
            path.unlink()


def build_model_execution_metadata(*, benchmark_version: str = "v4.0") -> dict:
    return {
        "benchmark_version": benchmark_version,
        "profiles": sorted(PROFILE_SPECS),
        "models": {
            "kNN": {
                "family": "sklearn",
                "support_valid_compatibility_enabled": False,
                "support_side_scoring": "predict_proba on support molecules from the episode-trained kNN classifier",
            },
            "RF": {
                "family": "sklearn",
                "support_valid_compatibility_enabled": False,
                "support_side_scoring": "predict_proba on support molecules from the episode-trained random forest classifier",
            },
            "ProtoNet": {
                "family": "metric-based",
                "support_valid_compatibility_enabled": False,
                "support_side_scoring": "support forward pass",
            },
            "MAML": {
                "family": "meta-learning",
                "support_valid_compatibility_enabled": True,
                "support_valid_rule": (
                    "deterministic holdout of the last k=1 positive and last k=1 negative "
                    "support samples into validation"
                ),
                "query_unchanged": True,
                "valid_pair_metrics_unchanged": True,
                "support_side_scoring": "post-adaptation forward pass on support molecules",
            },
            "kNN-cliff-aware": {
                "family": "intervention",
                "support_valid_compatibility_enabled": False,
                "support_side_scoring": (
                    "predict_proba on support molecules from the hard-negative-augmented episode-trained "
                    "kNN classifier"
                ),
            },
        },
    }


def render_release_reproducibility_markdown(
    *,
    benchmark_version: str = "v4.0",
    adversarial_episode_variant: str = "baseline",
    build_command_name: str = "build-release",
    base_release_dir: Path | None = None,
) -> str:
    if build_command_name == "build-episode-variant-release":
        build_release_command = (
            "- `PYTHONPATH=src python -m fsmol_cliff.cli build-episode-variant-release "
            f"--base-release-dir {base_release_dir or '<base_release_dir>'} --output-dir <out> "
            f"--profile relaxed --adversarial-episode-variant {adversarial_episode_variant}`"
        )
    else:
        build_release_command = (
            "- `PYTHONPATH=src python -m fsmol_cliff.cli build-release --data-dir <fsmol_dir> "
            f"--output-dir <out> --profile relaxed --adversarial-episode-variant {adversarial_episode_variant}`"
        )
    lines = [
        f"# FS-Mol-Cliff {benchmark_version} Release Reproducibility",
        "",
        "This release directory is a frozen artifact bundle, not a self-contained source snapshot.",
        "Use the current repository source files below to rebuild assets, rerun evaluation, and regenerate aggregate tables.",
        "",
        "## CLI Entry Points",
        "",
        build_release_command,
        "- `PYTHONPATH=src python -m fsmol_cliff.cli audit-attrition --release-dir <release_dir> --data-dir <fsmol_dir> --output-dir <audit_out> --profile relaxed`",
        "- `PYTHONPATH=src python -m fsmol_cliff.cli evaluate --release-dir <release_dir> --output <task_results.parquet> --profile relaxed --model-name RF`",
        "- `PYTHONPATH=src python -m fsmol_cliff.cli aggregate --input <task_results.parquet> --output <aggregate.json>`",
        "",
        "## Source File Entry Points",
        "",
        "- `src/fsmol_cliff/cli.py`",
        "- `src/fsmol_cliff/release.py`",
        "- `src/fsmol_cliff/audit.py`",
        "- `src/fsmol_cliff/runner.py`",
        "- `src/fsmol_cliff/aggregate.py`",
        "- `src/fsmol_cliff/release_artifacts.py`",
        "",
        "## Notes",
        "",
        f"- `benchmark_manifest.json` records `adversarial_episode_variant={adversarial_episode_variant}` for release reconstruction.",
        "- `episode_protocol_note.md` summarizes the adversarial episode construction rule used in this release.",
        "- `model_execution_metadata.json` records the scoring and compatibility policy for the supported model families.",
        "- If chemistry-level implementation details change, previously generated release metrics may require a rebuild and reevaluation to stay numerically aligned with the updated code.",
        "",
    ]
    return "\n".join(lines)


def render_episode_protocol_note(*, adversarial_episode_variant: str = "baseline") -> str:
    if adversarial_episode_variant == "query_targeted_support_neg":
        return "\n".join(
            [
                "# Query-Targeted Support Negatives",
                "",
                "This release variant keeps the benchmark substrate fixed and only rewrites the adversarial episode support negatives.",
                "Support positives, query composition, and injected cliff pairs are preserved from the baseline adversarial manifests.",
                "Support negatives are chosen preferentially from hard-negative candidates aligned to the injected support-query anchors.",
                "",
            ]
        )
    if adversarial_episode_variant == "same_scaffold_query_targeted":
        return "\n".join(
            [
                "# Same-Scaffold Query-Targeted Adversarial Episodes",
                "",
                "This release variant keeps the benchmark substrate fixed and rewrites the adversarial episode injection rule.",
                "When enough same-scaffold cliff pairs exist, injected cliff pairs are chosen from same-scaffold pairs first.",
                "If same-scaffold pairs are insufficient for the target injection count, the variant falls back to the baseline adversarial builder.",
                "",
            ]
        )
    if adversarial_episode_variant == "anchor_coverage_first":
        return "\n".join(
            [
                "# Anchor-Coverage-First Adversarial Episodes",
                "",
                "This release variant keeps the benchmark substrate fixed and rewrites only the adversarial injection priority.",
                "Injected cliff pairs prefer anchors with larger available cliff-negative coverage before falling back to lexical anchor order.",
                "After injected pairs are chosen, the rest of the adversarial episode skeleton follows the baseline builder.",
                "",
            ]
        )
    if adversarial_episode_variant == "paired_hardness_balanced":
        return "\n".join(
            [
                "# Paired-Hardness-Balanced Adversarial Episodes",
                "",
                "This release variant keeps the benchmark substrate fixed and rewrites only the adversarial pair-priority rule.",
                "Injected cliff pairs prefer more moderate but still valid cliff gaps before the most extreme cliff pairs.",
                "After injected pairs are chosen, the rest of the adversarial episode skeleton follows the baseline builder.",
                "",
            ]
        )
    if adversarial_episode_variant == "query_cluster_separation_by_neg_diversity":
        return "\n".join(
            [
                "# Query-Cluster Separation by Negative Diversity",
                "",
                "This release variant keeps the benchmark substrate fixed and rewrites the adversarial hard-negative priority rule.",
                "Injected query negatives prefer less hub-like negatives so the injected query side is spread across more distinct local neighborhoods.",
                "After injected pairs are chosen, the rest of the adversarial episode skeleton follows the baseline builder.",
                "",
            ]
        )
    if adversarial_episode_variant == "query_cluster_separation_by_anchor_neg_mix":
        return "\n".join(
            [
                "# Query-Cluster Separation by Anchor-Negative Mix",
                "",
                "This release variant combines negative-diversity ordering with an interleaved high-coverage/low-coverage anchor priority order.",
                "The goal is to avoid adversarial episodes being dominated by either hub negatives or only the highest-coverage anchors.",
                "After injected pairs are chosen, the rest of the adversarial episode skeleton follows the baseline builder.",
                "",
            ]
        )
    return "\n".join(
        [
            "# Baseline Adversarial Episode Protocol",
            "",
            "This release uses the default adversarial episode builder.",
            "Adversarial manifests are sampled directly from the profile-specific release substrate without any extra support-negative rewrite.",
            "",
        ]
    )
