from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pandas as pd

from .adapters import score_cliff_aware_sklearn_episode, score_official_baseline_episode, score_sklearn_episode
from .evaluation import evaluate_episode_manifest, summarize_task_metric_rows
from .fsmol_bridge import default_external_fsmol_root
from .io import load_assay_context, resolve_assay_path, resolve_manifest_path, write_parquet
from . import protonet_runner as _pn  # noqa: F401 — used via _pn.xxx lookups for monkeypatch compatibility


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
        manifest_path = resolve_manifest_path(release_dir, split_type=split_type, profile=profile)
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        for episode in frame.to_dict(orient="records"):
            task_id = episode["task_id"]
            assay_context = assay_context_cache.setdefault(task_id, load_assay_context(release_dir, task_id, profile=profile))
            if backend == "local":
                scorer = score_sklearn_episode
            elif backend == "official":
                scorer = score_official_baseline_episode
            elif backend == "cliff-aware":
                scorer = score_cliff_aware_sklearn_episode
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
        f"PYTHONPATH={Path(__file__).resolve().parents[1]}:{default_external_fsmol_root()}",
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
        manifest_path = resolve_manifest_path(release_dir, split_type=split_type, profile=profile)
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        if task_ids is not None:
            frame = frame[frame["task_id"].isin(task_ids)]
        if seeds is not None:
            frame = frame[frame["seed"].isin(seeds)]
        for task_id in frame["task_id"].drop_duplicates().tolist():
            assay_context = assay_context_cache.setdefault(task_id, load_assay_context(release_dir, task_id, profile=profile))
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


# Module-lookup wrappers (not direct aliases) so that monkeypatching
# fsmol_cliff.protonet_runner.XXX propagates to callers via these names.
def load_protonet_model(*args, **kwargs):
    return _pn.load_protonet_model(*args, **kwargs)


def load_task_sample_map(*args, **kwargs):
    return _pn.load_task_sample_map(*args, **kwargs)


load_fsmol_task_sample_map = load_task_sample_map


def score_protonet_episode(**kwargs):
    return _pn.score_protonet_manifest_episode(**kwargs)


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
    calibration_mode: str = "identity",
    calibration_top_k: int = 2,
    calibration_uncertainty_scale: float = 0.1,
    calibration_margin_floor: float = 0.1,
    case_net_fusion_lambda: float = 0.5,
) -> list[dict]:
    assay_context_cache: dict[str, dict] = {}
    sample_map_cache: dict[str, dict[str, object]] = {}
    episode_results: list[dict] = []
    model = load_protonet_model(checkpoint_path=checkpoint_path, device=device)

    for split_type in split_types:
        manifest_path = resolve_manifest_path(release_dir, split_type=split_type, profile=profile)
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        if task_ids is not None:
            frame = frame[frame["task_id"].isin(task_ids)]
        if seeds is not None:
            frame = frame[frame["seed"].isin(seeds)]
        if max_episodes is not None:
            frame = frame.groupby(["task_id", "seed"], sort=False).head(max_episodes)
        for episode in frame.to_dict(orient="records"):
            task_id = episode["task_id"]
            assay_context = assay_context_cache.setdefault(task_id, load_assay_context(release_dir, task_id, profile=profile))
            sample_map = sample_map_cache.setdefault(task_id, load_fsmol_task_sample_map(data_dir, task_id))
            episode_result = evaluate_episode_manifest(
                    episode=episode,
                    assay_context=assay_context,
                    score_fn=lambda current_episode, *, current_sample_map=sample_map: score_protonet_episode(
                        model=model,
                        sample_map=current_sample_map,
                        episode=current_episode,
                        assay_context=assay_context,
                        batch_size=batch_size,
                        support_score_mode=support_score_mode,
                        calibration_mode=calibration_mode,
                        calibration_top_k=calibration_top_k,
                        calibration_uncertainty_scale=calibration_uncertainty_scale,
                        calibration_margin_floor=calibration_margin_floor,
                        case_net_fusion_lambda=case_net_fusion_lambda,
                    ),
                )
            episode_results.append({**episode_result, "profile": profile, "result_tier": result_tier})

    rows = summarize_task_metric_rows(episode_results)
    write_parquet(output_path, rows)
    return rows
