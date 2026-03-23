from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .evaluation import evaluate_episode_manifest, summarize_task_metric_rows
from .fsmol_bridge import install_fs_mol_compat_patches
from .io import write_parquet

FSMolTask = None
FSMolTaskSample = None
RichPath = None
get_protonet_batcher = None
task_sample_to_pn_task_sample = None
torchify = None


def _ensure_fs_mol_symbols() -> None:
    global FSMolTask
    global FSMolTaskSample
    global RichPath
    global get_protonet_batcher
    global task_sample_to_pn_task_sample
    global torchify

    if all(
        symbol is not None
        for symbol in (
            FSMolTask,
            FSMolTaskSample,
            RichPath,
            get_protonet_batcher,
            task_sample_to_pn_task_sample,
            torchify,
        )
    ):
        return

    install_fs_mol_compat_patches()

    from dpu_utils.utils import RichPath as imported_rich_path
    from fs_mol.data.fsmol_task import FSMolTask as imported_task
    from fs_mol.data.fsmol_task import FSMolTaskSample as imported_task_sample
    from fs_mol.data.protonet import get_protonet_batcher as imported_batcher
    from fs_mol.data.protonet import task_sample_to_pn_task_sample as imported_converter
    from fs_mol.utils.torch_utils import torchify as imported_torchify

    if FSMolTask is None:
        FSMolTask = imported_task
    if FSMolTaskSample is None:
        FSMolTaskSample = imported_task_sample
    if RichPath is None:
        RichPath = imported_rich_path
    if get_protonet_batcher is None:
        get_protonet_batcher = imported_batcher
    if task_sample_to_pn_task_sample is None:
        task_sample_to_pn_task_sample = imported_converter
    if torchify is None:
        torchify = imported_torchify


def _get_protonet_trainer_class():
    install_fs_mol_compat_patches()
    from fs_mol.utils.protonet_utils import PrototypicalNetworkTrainer

    return PrototypicalNetworkTrainer


def load_protonet_model(
    checkpoint_path: Path,
    *,
    device: torch.device | None = None,
):
    trainer_cls = _get_protonet_trainer_class()
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = trainer_cls(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model


def load_task_sample_map(data_dir: Path, task_id: str) -> dict[str, Any]:
    _ensure_fs_mol_symbols()
    task_path = RichPath.create(str(data_dir / "test" / f"{task_id}.jsonl.gz"))
    task = FSMolTask.load_from_file(task_path)
    return {f"{task_id}:{idx}": sample for idx, sample in enumerate(task.samples)}


def score_protonet_target_ids(
    *,
    model,
    task_id: str,
    sample_map: dict[str, Any],
    support_ids: list[str],
    target_ids: list[str],
    batch_size: int = 320,
) -> dict[str, float]:
    _ensure_fs_mol_symbols()
    task_sample = FSMolTaskSample(
        name=task_id,
        train_samples=[sample_map[molecule_id] for molecule_id in support_ids],
        valid_samples=[],
        test_samples=[sample_map[molecule_id] for molecule_id in target_ids],
    )
    batcher = get_protonet_batcher(max_num_graphs=batch_size)
    pn_task_sample = torchify(task_sample_to_pn_task_sample(task_sample, batcher), device=model.device)

    scores: dict[str, float] = {}
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in pn_task_sample.batches:
            probabilities = (
                torch.nn.functional.softmax(model(batch), dim=1).detach().cpu().numpy()[:, 1].tolist()
            )
            batch_ids = target_ids[offset : offset + batch.num_query_samples]
            scores.update(
                {
                    molecule_id: float(score)
                    for molecule_id, score in zip(batch_ids, probabilities, strict=True)
                }
            )
            offset += batch.num_query_samples
    if offset != len(target_ids):
        raise ValueError(
            f"ProtoNet score count mismatch for {task_id}: produced {offset} scores for {len(target_ids)} targets."
        )
    return scores


def score_protonet_manifest_episode(
    *,
    model,
    sample_map: dict[str, Any],
    episode: dict,
    batch_size: int = 320,
    support_score_mode: str = "forward",
) -> dict[str, float]:
    task_id = episode["task_id"]
    support_ids = [*episode["support_pos_ids"], *episode["support_neg_ids"]]
    query_ids = [*episode["query_pos_ids"], *episode["query_neg_ids"]]

    if support_score_mode != "forward":
        raise ValueError(f"Unsupported ProtoNet support_score_mode: {support_score_mode}")

    scores = score_protonet_target_ids(
        model=model,
        task_id=task_id,
        sample_map=sample_map,
        support_ids=support_ids,
        target_ids=query_ids,
        batch_size=batch_size,
    )
    support_scores = score_protonet_target_ids(
        model=model,
        task_id=task_id,
        sample_map=sample_map,
        support_ids=support_ids,
        target_ids=support_ids,
        batch_size=batch_size,
    )
    return {**support_scores, **scores}


def evaluate_release_with_protonet(
    *,
    release_dir: Path,
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    split_types: tuple[str, ...] = ("standard", "adversarial"),
    profile: str = "strict",
    result_tier: str = "final",
    task_ids: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    batch_size: int = 320,
    max_episodes: int | None = None,
    support_score_mode: str = "forward",
    device: torch.device | None = None,
) -> list[dict]:
    model = load_protonet_model(checkpoint_path, device=device)
    assay_context_cache: dict[str, dict] = {}
    sample_map_cache: dict[str, dict[str, Any]] = {}
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
        if max_episodes is not None:
            frame = frame.groupby(["task_id", "seed"], sort=False).head(max_episodes)
        for episode in frame.to_dict(orient="records"):
            task_id = episode["task_id"]
            assay_context = assay_context_cache.setdefault(task_id, _load_assay_context(release_dir, task_id, profile=profile))
            sample_map = sample_map_cache.setdefault(task_id, load_task_sample_map(data_dir, task_id))
            episode_result = evaluate_episode_manifest(
                    episode=episode,
                    assay_context=assay_context,
                    score_fn=lambda _, current_episode=episode, current_sample_map=sample_map: score_protonet_manifest_episode(
                        model=model,
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
    cliff_pairs = []
    noncliff_pairs = []
    with _resolve_assay_path(assay_dir, stem="pairs", suffix=".jsonl", profile=profile).open() as handle:
        for line in handle:
            pair = json.loads(line)
            if pair["pair_type"] == "cliff":
                cliff_pairs.append(pair)
            else:
                noncliff_pairs.append(pair)
    labels = {record["molecule_id"]: int(record["label"]) for record in annotations}
    return {
        "labels": labels,
        "cliff_pairs": cliff_pairs,
        "noncliff_pairs": noncliff_pairs,
    }


def _resolve_manifest_path(release_dir: Path, *, split_type: str, profile: str) -> Path:
    profile_path = release_dir / f"episodes_{split_type}_{profile}.parquet"
    legacy_path = release_dir / f"episodes_{split_type}.parquet"
    return profile_path if profile_path.exists() else legacy_path


def _resolve_assay_path(assay_dir: Path, *, stem: str, suffix: str, profile: str) -> Path:
    profile_path = assay_dir / f"{stem}_{profile}{suffix}"
    legacy_path = assay_dir / f"{stem}{suffix}"
    return profile_path if profile_path.exists() else legacy_path
