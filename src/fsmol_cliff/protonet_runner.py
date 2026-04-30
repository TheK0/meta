from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .fsmol_bridge import install_fs_mol_compat_patches
from .protonet_base import build_raw_score_bundle
from .protonet_local_calibrated import (
    apply_identity_local_calibration,
    apply_query_only_local_calibration,
)

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
    _warm_runtime_for_device(resolved_device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = trainer_cls(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model


def _warm_runtime_for_device(device: torch.device | str) -> None:
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type != "mps":
        return
    # Force MPS stream/allocator initialization before checkpoint deserialization.
    torch.ones(1, device=device)
    torch.mps.empty_cache()


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
            batch = _coerce_protonet_batch_float_features(batch)
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


def _coerce_protonet_batch_float_features(batch):
    if not hasattr(batch, "support_features") or not hasattr(batch, "query_features"):
        return batch

    def _coerce_feature_block(feature_block):
        fingerprints = getattr(feature_block, "fingerprints", None)
        descriptors = getattr(feature_block, "descriptors", None)
        return replace(
            feature_block,
            fingerprints=fingerprints.float() if hasattr(fingerprints, "float") else fingerprints,
            descriptors=descriptors.float() if hasattr(descriptors, "float") else descriptors,
        )

    return replace(
        batch,
        support_features=_coerce_feature_block(batch.support_features),
        query_features=_coerce_feature_block(batch.query_features),
    )


def score_protonet_manifest_episode(
    *,
    model,
    sample_map: dict[str, Any],
    episode: dict,
    assay_context: dict | None = None,
    batch_size: int = 320,
    support_score_mode: str = "forward",
    calibration_mode: str = "identity",
) -> dict[str, object]:
    task_id = episode["task_id"]
    support_ids = [*episode["support_pos_ids"], *episode["support_neg_ids"]]
    query_ids = [*episode["query_pos_ids"], *episode["query_neg_ids"]]

    if support_score_mode != "forward":
        raise ValueError(f"Unsupported ProtoNet support_score_mode: {support_score_mode}")

    query_scores = score_protonet_target_ids(
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
    raw_scores = {**support_scores, **query_scores}
    raw_bundle = build_raw_score_bundle(raw_scores=raw_scores)
    if calibration_mode == "identity":
        return apply_identity_local_calibration(
            raw_scores=raw_bundle["raw_scores"],
            raw_margins=raw_bundle["raw_margins"],
        )
    if calibration_mode == "query_only":
        if assay_context is None:
            raise ValueError("query_only ProtoNet calibration requires assay_context.")
        return apply_query_only_local_calibration(
            episode=episode,
            assay_context=assay_context,
            raw_scores=raw_bundle["raw_scores"],
            raw_margins=raw_bundle["raw_margins"],
        )
    raise ValueError(f"Unsupported ProtoNet calibration_mode: {calibration_mode}")
