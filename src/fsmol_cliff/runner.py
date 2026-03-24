from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pandas as pd

from .adapters import (
    score_cliff_aware_sklearn_episode,
    score_decision_aware_sklearn_episode,
    score_official_baseline_episode,
    score_sklearn_episode,
)
from .evaluation import evaluate_episode_manifest, summarize_task_metric_rows
from .io import write_parquet
from .protonet_runner import (
    load_protonet_model as _load_protonet_model,
    load_task_sample_map as _load_task_sample_map,
    score_protonet_manifest_episode as _score_protonet_manifest_episode,
)


def evaluate_release_with_sklearn_baseline(
    *,
    release_dir: Path,
    output_path: Path,
    split_types: Sequence[str] = ("standard", "adversarial"),
    profile: str = "strict",
    result_tier: str = "final",
    model_name: str = "kNN",
    model_params: dict | None = None,
    backend: str = "local",
) -> list[dict]:
    assay_context_cache: dict[str, dict] = {}
    episode_results: list[dict] = []
    for split_type in split_types:
        manifest_path = _resolve_manifest_path(release_dir, split_type=split_type, profile=profile)
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        for episode in frame.to_dict(orient="records"):
            task_id = episode["task_id"]
            assay_context = assay_context_cache.setdefault(task_id, _load_assay_context(release_dir, task_id, profile=profile))
            if backend == "local":
                scorer = score_sklearn_episode
            elif backend == "official":
                scorer = score_official_baseline_episode
            elif backend == "cliff-aware":
                scorer = score_cliff_aware_sklearn_episode
            elif backend == "decision-aware":
                scorer = score_decision_aware_sklearn_episode
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            episode_result = evaluate_episode_manifest(
                    episode=episode,
                    assay_context=assay_context,
                    score_fn=lambda current_episode, *, context=assay_context, scorer=scorer: scorer(
                        model_name=model_name,
                        assay_id=current_episode["task_id"],
                        records_by_id=context["records_by_id"],
                        **(
                            {
                                "support_ids": [*current_episode["support_pos_ids"], *current_episode["support_neg_ids"]],
                                "query_ids": [
                                    *current_episode["support_pos_ids"],
                                    *current_episode["support_neg_ids"],
                                    *current_episode["query_pos_ids"],
                                    *current_episode["query_neg_ids"],
                                ],
                                "use_grid_search": False,
                                "model_params": model_params,
                            }
                            if backend in {"local", "official"}
                            else {
                                "support_pos_ids": list(current_episode["support_pos_ids"]),
                                "support_neg_ids": list(current_episode["support_neg_ids"]),
                                "query_ids": [
                                    *current_episode["support_pos_ids"],
                                    *current_episode["support_neg_ids"],
                                    *current_episode["query_pos_ids"],
                                    *current_episode["query_neg_ids"],
                                ],
                                "anchor_to_hardnegs": context.get("anchor_to_hardnegs", {}),
                                "use_grid_search": False,
                                "model_params": model_params,
                            }
                        ),
                    ),
                )
            episode_results.append({**episode_result, "profile": profile, "result_tier": result_tier})
    rows = summarize_task_metric_rows(episode_results)
    write_parquet(output_path, rows)
    return rows


def build_maml_legacy_smoke_command(
    *,
    release_dir: Path,
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    profile: str,
    split_type: str,
    task_id: str,
    seed: int,
    legacy_env_name: str,
    max_episodes: int,
) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        legacy_env_name,
        "env",
        f"PYTHONPATH={Path(__file__).resolve().parents[1]}:/Volumes/macplus/project/meta/external/FS-Mol",
        "python",
        "-m",
        "fsmol_cliff.maml_legacy_runner",
        "--release-dir",
        str(release_dir),
        "--data-dir",
        str(data_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--task-id",
        task_id,
        "--profile",
        profile,
        "--split-type",
        split_type,
        "--seed",
        str(seed),
        "--output",
        str(output_path),
        "--max-episodes",
        str(max_episodes),
    ]


def run_maml_legacy_smoke(
    *,
    release_dir: Path,
    data_dir: Path,
    checkpoint_path: Path,
    task_id: str,
    seed: int,
    profile: str = "strict",
    split_types: Sequence[str] = ("standard", "adversarial"),
    legacy_env_name: str = "fsmol-maml-legacy",
    max_episodes: int = 3,
) -> dict[str, list[dict]]:
    outputs = {}
    for split_type in split_types:
        output_path = Path("/tmp") / f"maml_legacy_smoke_{task_id}_{split_type}.json"
        command = build_maml_legacy_smoke_command(
            release_dir=release_dir,
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            profile=profile,
            split_type=split_type,
            task_id=task_id,
            seed=seed,
            legacy_env_name=legacy_env_name,
            max_episodes=max_episodes,
        )
        subprocess.run(command, check=True)
        outputs[split_type] = json.loads(output_path.read_text())
    return outputs


def convert_legacy_maml_outputs_to_episode_results(
    *,
    episodes: list[dict],
    assay_context: dict,
    legacy_outputs: list[dict],
) -> list[dict]:
    by_episode = {int(row["episode_id"]): row for row in legacy_outputs}
    results = []
    for episode in episodes:
        output = by_episode[int(episode["episode_id"])]
        results.append(
            evaluate_episode_manifest(
                episode=episode,
                assay_context=assay_context,
                score_fn=lambda _, scores=output["scores"]: scores,
            )
        )
    return results


def evaluate_release_with_maml_legacy(
    *,
    release_dir: Path,
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    split_types: Sequence[str] = ("standard", "adversarial"),
    profile: str = "strict",
    result_tier: str = "final",
    task_ids: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    legacy_env_name: str = "fsmol-maml-legacy",
    max_episodes: int = 3,
) -> list[dict]:
    assay_context_cache: dict[str, dict] = {}
    episode_results: list[dict] = []
    for split_type in split_types:
        manifest_path = _resolve_manifest_path(release_dir, split_type=split_type, profile=profile)
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        if task_ids is not None:
            frame = frame[frame["task_id"].isin(task_ids)]
        if seeds is not None:
            frame = frame[frame["seed"].isin(seeds)]
        for task_id in frame["task_id"].drop_duplicates().tolist():
            assay_context = assay_context_cache.setdefault(task_id, _load_assay_context(release_dir, task_id, profile=profile))
            for seed in sorted(frame[frame["task_id"] == task_id]["seed"].drop_duplicates().tolist()):
                episodes = (
                    frame[(frame["task_id"] == task_id) & (frame["seed"] == seed)]
                    .head(max_episodes)
                    .to_dict(orient="records")
                )
                legacy_outputs = run_maml_legacy_smoke(
                    release_dir=release_dir,
                    data_dir=data_dir,
                    checkpoint_path=checkpoint_path,
                    task_id=task_id,
                    seed=seed,
                    profile=profile,
                    split_types=(split_type,),
                    legacy_env_name=legacy_env_name,
                    max_episodes=max_episodes,
                )[split_type]
                episode_results.extend(
                    {
                        **result,
                        "profile": profile,
                        "result_tier": result_tier,
                    }
                    for result in convert_legacy_maml_outputs_to_episode_results(
                        episodes=episodes,
                        assay_context=assay_context,
                        legacy_outputs=legacy_outputs,
                    )
                )
    rows = summarize_task_metric_rows(episode_results)
    write_parquet(output_path, rows)
    return rows


load_protonet_model = _load_protonet_model
load_task_sample_map = _load_task_sample_map
load_fsmol_task_sample_map = _load_task_sample_map
score_protonet_episode = _score_protonet_manifest_episode


def score_protonet_episode_with_model(
    *,
    model,
    task_id: str,
    sample_map: dict[str, object],
    episode: dict,
    batch_size: int = 320,
    support_score_mode: str = "forward",
) -> dict[str, float]:
    return _score_protonet_manifest_episode(
        model=model,
        sample_map=sample_map,
        episode=episode,
        batch_size=batch_size,
        support_score_mode=support_score_mode,
    )


score_protonet_episode = score_protonet_episode_with_model


def evaluate_release_with_protonet(
    *,
    release_dir: Path,
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    split_types: Sequence[str] = ("standard", "adversarial"),
    profile: str = "strict",
    result_tier: str = "final",
    task_ids: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    batch_size: int = 320,
    max_episodes: int | None = None,
    device: str | None = None,
    support_score_mode: str = "forward",
) -> list[dict]:
    assay_context_cache: dict[str, dict] = {}
    sample_map_cache: dict[str, dict[str, object]] = {}
    episode_results: list[dict] = []
    model = load_protonet_model(checkpoint_path=checkpoint_path, device=device)

    for split_type in split_types:
        manifest_path = _resolve_manifest_path(release_dir, split_type=split_type, profile=profile)
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        if task_ids is not None:
            frame = frame[frame["task_id"].isin(task_ids)]
        if seeds is not None:
            frame = frame[frame["seed"].isin(seeds)]
        for task_id in frame["task_id"].drop_duplicates().tolist():
            assay_context = assay_context_cache.setdefault(task_id, _load_assay_context(release_dir, task_id, profile=profile))
            sample_map = sample_map_cache.setdefault(task_id, load_fsmol_task_sample_map(data_dir, task_id))
            for seed in sorted(frame[frame["task_id"] == task_id]["seed"].drop_duplicates().tolist()):
                seed_frame = frame[(frame["task_id"] == task_id) & (frame["seed"] == seed)]
                if max_episodes is not None:
                    seed_frame = seed_frame.head(max_episodes)
                for episode in seed_frame.to_dict(orient="records"):
                    episode_result = evaluate_episode_manifest(
                            episode=episode,
                            assay_context=assay_context,
                            score_fn=lambda current_episode, *, current_task_id=task_id, current_sample_map=sample_map: score_protonet_episode(
                                model=model,
                                task_id=current_task_id,
                                sample_map=current_sample_map,
                                episode=current_episode,
                                batch_size=batch_size,
                                support_score_mode=support_score_mode,
                            ),
                        )
                    episode_results.append({**episode_result, "profile": profile, "result_tier": result_tier})

    rows = summarize_task_metric_rows(episode_results)
    write_parquet(output_path, rows)
    return rows


def _load_assay_context(release_dir: Path, task_id: str, *, profile: str = "strict") -> dict:
    assay_dir = release_dir / "assays" / task_id
    annotations = pd.read_parquet(assay_dir / "molecule_annotations.parquet").to_dict(orient="records")
    pairs_path = _resolve_assay_path(assay_dir, stem="pairs", suffix=".jsonl", profile=profile)
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
    return {
        "records_by_id": records_by_id,
        "labels": labels,
        "cliff_pairs": cliff_pairs,
        "noncliff_pairs": noncliff_pairs,
        "anchor_to_hardnegs": json.loads(_resolve_assay_path(assay_dir, stem="anchor_to_hardnegs", suffix=".json", profile=profile).read_text())
        if _resolve_assay_path(assay_dir, stem="anchor_to_hardnegs", suffix=".json", profile=profile).exists()
        else {},
    }


def _resolve_manifest_path(release_dir: Path, *, split_type: str, profile: str) -> Path:
    profile_path = release_dir / f"episodes_{split_type}_{profile}.parquet"
    legacy_path = release_dir / f"episodes_{split_type}.parquet"
    return profile_path if profile_path.exists() else legacy_path


def _resolve_assay_path(assay_dir: Path, *, stem: str, suffix: str, profile: str) -> Path:
    profile_path = assay_dir / f"{stem}_{profile}{suffix}"
    legacy_path = assay_dir / f"{stem}{suffix}"
    return profile_path if profile_path.exists() else legacy_path
