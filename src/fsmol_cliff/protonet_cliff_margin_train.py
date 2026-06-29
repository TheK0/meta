from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy as np
import torch

from .fsmol_bridge import install_fs_mol_compat_patches

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliffProtoNetBatch:
    support_features: Any
    support_labels: np.ndarray
    support_log_r: np.ndarray
    query_features: Any
    query_log_r: np.ndarray
    task_name: str

    @property
    def num_query_samples(self) -> int:
        return int(self.query_features.num_graphs)


@dataclass(frozen=True)
class FeaturisedCliffPNTaskSample:
    task_name: str
    num_support_samples: int
    num_positive_support_samples: int
    num_query_samples: int
    num_positive_query_samples: int
    batches: list[CliffProtoNetBatch]
    batch_labels: list[np.ndarray]


def build_cliff_margin_trainer_class(base_trainer_cls):
    class CliffMarginPrototypicalNetworkTrainer(base_trainer_cls):
        def __init__(
            self,
            config,
            *,
            margin_gamma: float,
            lambda_cliff: float,
            control_preservation: bool,
            tau: float = 0.8,
            delta: float = 1.0,
        ):
            super().__init__(config)
            self.margin_gamma = float(margin_gamma)
            self.lambda_cliff = float(lambda_cliff)
            self.control_preservation = bool(control_preservation)
            self.tau = float(tau)
            self.delta = float(delta)
            self._last_support_embeddings = None
            self._last_query_embeddings = None
            self._last_support_labels = None

        def forward(self, input_batch):
            support_features: list[torch.Tensor] = []
            query_features: list[torch.Tensor] = []

            if "gnn" in self.config.used_features:
                support_features.append(self.graph_feature_extractor(input_batch.support_features))
                query_features.append(self.graph_feature_extractor(input_batch.query_features))
            if "ecfp" in self.config.used_features:
                support_features.append(input_batch.support_features.fingerprints.float())
                query_features.append(input_batch.query_features.fingerprints.float())
            if "pc-descs" in self.config.used_features:
                support_features.append(input_batch.support_features.descriptors.float())
                query_features.append(input_batch.query_features.descriptors.float())

            support_features_flat = torch.cat(support_features, dim=1)
            query_features_flat = torch.cat(query_features, dim=1)

            if self.use_fc:
                support_features_flat = self.fc(support_features_flat)
                query_features_flat = self.fc(query_features_flat)

            self._last_support_embeddings = support_features_flat
            self._last_query_embeddings = query_features_flat
            self._last_support_labels = input_batch.support_labels

            if self.config.distance_metric == "mahalanobis":
                class_means, class_precision_matrices = self.compute_class_means_and_precisions(
                    support_features_flat, input_batch.support_labels
                )
                number_of_classes = class_means.size(0)
                number_of_targets = query_features_flat.size(0)
                repeated_target = query_features_flat.repeat(1, number_of_classes).view(
                    -1, class_means.size(1)
                )
                repeated_class_means = class_means.repeat(number_of_targets, 1)
                repeated_difference = repeated_class_means - repeated_target
                repeated_difference = repeated_difference.view(
                    number_of_targets, number_of_classes, repeated_difference.size(1)
                ).permute(1, 0, 2)
                first_half = torch.matmul(repeated_difference, class_precision_matrices)
                return torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

            return self._protonets_euclidean_classifier(
                support_features_flat,
                query_features_flat,
                input_batch.support_labels,
            )

        def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, batch_features=None) -> torch.Tensor:
            label_loss = torch.nn.functional.cross_entropy(logits, labels.long())
            if batch_features is None:
                return label_loss
            cliff_penalty = cliff_margin_penalty_torch(
                support_embeddings=self._last_support_embeddings,
                support_labels=self._last_support_labels,
                query_embeddings=self._last_query_embeddings,
                query_labels=labels,
                support_fingerprints=batch_features.support_features.fingerprints,
                query_fingerprints=batch_features.query_features.fingerprints,
                support_log_r=batch_features.support_log_r,
                query_log_r=batch_features.query_log_r,
                margin_gamma=self.margin_gamma,
                tau=self.tau,
                delta=self.delta,
            )
            loss = label_loss + self.lambda_cliff * cliff_penalty
            if self.control_preservation:
                loss = loss + 0.1 * control_preservation_penalty_torch(
                    support_embeddings=self._last_support_embeddings,
                    support_labels=self._last_support_labels,
                    query_embeddings=self._last_query_embeddings,
                    query_labels=labels,
                    support_fingerprints=batch_features.support_features.fingerprints,
                    query_fingerprints=batch_features.query_features.fingerprints,
                    support_log_r=batch_features.support_log_r,
                    query_log_r=batch_features.query_log_r,
                    tau=self.tau,
                    delta=self.delta,
                )
            return loss

        def train_loop(self, out_dir: str, dataset, device: torch.device, aml_run=None, data_root: Path | None = None):
            _, _, _, _, _, _, DataFold, _, get_protonet_batcher, torchify, validate_by_finetuning_on_tasks, MetricLogger = _load_training_runtime_symbols()
            self.save_model(os.path.join(out_dir, "best_validation.pt"))
            train_task_sample_iterator = iter(
                _get_cliff_protonet_task_sample_iterable(
                    dataset=dataset,
                    data_fold=DataFold.TRAIN,
                    data_root=data_root,
                    num_samples=1,
                    support_size=self.config.support_set_size,
                    query_size=self.config.query_set_size,
                    max_num_graphs=self.config.batch_size,
                    repeat=True,
                )
            )
            best_validation_avg_prec = 0.0
            metric_logger = MetricLogger(
                log_fn=lambda msg: logger.info(msg),
                aml_run=aml_run,
                window_size=max(10, self.config.validate_every_num_steps / 5),
            )

            for step in range(1, self.config.num_train_steps + 1):
                torch.set_grad_enabled(True)
                self.optimizer.zero_grad()

                task_batch_losses: list[float] = []
                task_batch_metrics = []
                for _ in range(self.config.tasks_per_batch):
                    task_sample = next(train_task_sample_iterator)
                    train_task_sample = torchify(task_sample, device=device)
                    task_loss, task_metrics = run_cliff_batches(
                        self,
                        batches=train_task_sample.batches,
                        batch_labels=train_task_sample.batch_labels,
                        train=True,
                        tasks_per_batch=self.config.tasks_per_batch,
                    )
                    task_batch_losses.append(task_loss)
                    task_batch_metrics.append(task_metrics)

                if self.config.clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_value)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                task_batch_mean_loss = np.mean(task_batch_losses)
                metric_logger.log_metrics(loss=task_batch_mean_loss)

                if step % self.config.validate_every_num_steps == 0:
                    valid_metric = validate_by_finetuning_on_tasks(self, dataset, aml_run=aml_run)
                    if valid_metric > best_validation_avg_prec:
                        best_validation_avg_prec = valid_metric
                        self.save_model(os.path.join(out_dir, "best_validation.pt"))

            self.save_model(os.path.join(out_dir, "fully_trained.pt"))

    return CliffMarginPrototypicalNetworkTrainer


def cliff_margin_penalty_torch(
    *,
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    support_fingerprints: torch.Tensor,
    query_fingerprints: torch.Tensor,
    support_log_r: torch.Tensor,
    query_log_r: torch.Tensor,
    margin_gamma: float,
    tau: float,
    delta: float,
) -> torch.Tensor:
    cliff_mask, _ = identify_cliff_and_control_query_masks(
        support_labels=support_labels,
        query_labels=query_labels,
        support_fingerprints=support_fingerprints,
        query_fingerprints=query_fingerprints,
        support_log_r=support_log_r,
        query_log_r=query_log_r,
        tau=tau,
        delta=delta,
    )
    if not torch.any(cliff_mask):
        return torch.zeros((), dtype=query_embeddings.dtype, device=query_embeddings.device)
    prototypes = _compute_class_prototypes_torch(support_embeddings, support_labels)
    losses = []
    for index in torch.nonzero(cliff_mask, as_tuple=False).flatten():
        query_embedding = query_embeddings[index]
        label = int(query_labels[index].item())
        distance_to_positive = torch.sum((query_embedding - prototypes[1]) ** 2)
        distance_to_negative = torch.sum((query_embedding - prototypes[0]) ** 2)
        if label == 1:
            losses.append(torch.relu(distance_to_positive + margin_gamma - distance_to_negative))
        else:
            losses.append(torch.relu(distance_to_negative + margin_gamma - distance_to_positive))
    return torch.stack(losses).mean()


def control_preservation_penalty_torch(
    *,
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    support_fingerprints: torch.Tensor,
    query_fingerprints: torch.Tensor,
    support_log_r: torch.Tensor,
    query_log_r: torch.Tensor,
    tau: float,
    delta: float,
) -> torch.Tensor:
    _, control_mask = identify_cliff_and_control_query_masks(
        support_labels=support_labels,
        query_labels=query_labels,
        support_fingerprints=support_fingerprints,
        query_fingerprints=query_fingerprints,
        support_log_r=support_log_r,
        query_log_r=query_log_r,
        tau=tau,
        delta=delta,
    )
    if not torch.any(control_mask):
        return torch.zeros((), dtype=query_embeddings.dtype, device=query_embeddings.device)
    prototypes = _compute_class_prototypes_torch(support_embeddings, support_labels)
    penalties = []
    for index in torch.nonzero(control_mask, as_tuple=False).flatten():
        query_embedding = query_embeddings[index]
        label = int(query_labels[index].item())
        distance_to_positive = torch.sum((query_embedding - prototypes[1]) ** 2)
        distance_to_negative = torch.sum((query_embedding - prototypes[0]) ** 2)
        if label == 1:
            penalties.append(torch.relu(distance_to_positive - distance_to_negative))
        else:
            penalties.append(torch.relu(distance_to_negative - distance_to_positive))
    return torch.stack(penalties).mean()


def identify_cliff_and_control_query_masks(
    *,
    support_labels: torch.Tensor,
    query_labels: torch.Tensor,
    support_fingerprints: torch.Tensor,
    query_fingerprints: torch.Tensor,
    support_log_r: torch.Tensor,
    query_log_r: torch.Tensor,
    tau: float,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_bin = (support_fingerprints > 0).float()
    query_bin = (query_fingerprints > 0).float()
    intersection = torch.matmul(query_bin, support_bin.T)
    query_sum = query_bin.sum(dim=1, keepdim=True)
    support_sum = support_bin.sum(dim=1).unsqueeze(0)
    union = torch.clamp(query_sum + support_sum - intersection, min=1.0)
    similarities = intersection / union

    opposite_class = query_labels.unsqueeze(1) != support_labels.unsqueeze(0)
    high_sim = similarities >= tau
    cliff_gap = torch.abs(query_log_r.unsqueeze(1) - support_log_r.unsqueeze(0)) >= delta
    cliff_mask = torch.any(opposite_class & high_sim & cliff_gap, dim=1)
    control_mask = torch.any(opposite_class & high_sim & (~cliff_gap), dim=1)
    return cliff_mask, control_mask


def make_trainer_config(args: argparse.Namespace):
    add_graph_feature_extractor_arguments, make_graph_feature_extractor_config_from_args, _, _, PrototypicalNetworkTrainerConfig, _ = _load_external_training_symbols()
    return PrototypicalNetworkTrainerConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        used_features=args.features,
        distance_metric=args.distance_metric,
        batch_size=args.batch_size,
        tasks_per_batch=args.tasks_per_batch,
        support_set_size=args.support_set_size,
        query_set_size=args.query_set_size,
        validate_every_num_steps=args.validate_every,
        validation_support_set_sizes=tuple(args.validation_support_set_sizes),
        validation_query_set_size=args.validation_query_set_size,
        validation_num_samples=args.validation_num_samples,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
    )


def parse_command_line() -> argparse.Namespace:
    add_graph_feature_extractor_arguments, _, add_train_cli_args, _, _, _ = _load_external_training_symbols()
    parser = argparse.ArgumentParser(
        description="Train a ProtoNet model with cliff-margin loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_train_cli_args(parser)
    parser.add_argument("--features", type=str, default="gnn+ecfp+fc")
    parser.add_argument("--distance_metric", type=str, choices=["mahalanobis", "euclidean"], default="mahalanobis")
    add_graph_feature_extractor_arguments(parser)
    parser.add_argument("--support_set_size", type=int, default=64)
    parser.add_argument("--query_set_size", type=int, default=256)
    parser.add_argument("--tasks_per_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--validate_every", type=int, default=50)
    parser.add_argument("--validation-support-set-sizes", type=json.loads, default=[16, 128])
    parser.add_argument("--validation-query-set-size", type=int, default=512)
    parser.add_argument("--validation-num-samples", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--pretrained_gnn", type=str, default=None)
    parser.add_argument("--margin-gamma", type=float, default=0.1)
    parser.add_argument("--lambda-cliff", type=float, default=0.3)
    parser.add_argument("--control-preservation", action="store_true")
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--delta", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    _, _, _, set_up_train_run, _, base_trainer_cls = _load_external_training_symbols()
    args = parse_command_line()
    config = make_trainer_config(args)
    out_dir, dataset, aml_run = set_up_train_run("ProtoNetCliffMargin", args, torch=True)
    dataset._num_workers = 0
    trainer_cls = build_cliff_margin_trainer_class(base_trainer_cls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = trainer_cls(
        config=config,
        margin_gamma=args.margin_gamma,
        lambda_cliff=args.lambda_cliff,
        control_preservation=args.control_preservation,
        tau=args.tau,
        delta=args.delta,
    ).to(device)
    if args.pretrained_gnn is not None:
        model_trainer.load_model_gnn_weights(path=args.pretrained_gnn, device=device)
    model_trainer.train_loop(out_dir, dataset, device, aml_run, data_root=Path(args.DATA_PATH))


def _compute_class_prototypes_torch(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
) -> dict[int, torch.Tensor]:
    prototypes: dict[int, torch.Tensor] = {}
    for label in torch.unique(support_labels):
        mask = support_labels == label
        prototypes[int(label.item())] = support_embeddings[mask].mean(dim=0)
    return prototypes


def _load_external_training_symbols():
    install_fs_mol_compat_patches()
    from fs_mol.modules.graph_feature_extractor import (
        add_graph_feature_extractor_arguments,
        make_graph_feature_extractor_config_from_args,
    )
    from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run
    from fs_mol.utils.protonet_utils import PrototypicalNetworkTrainer, PrototypicalNetworkTrainerConfig

    return (
        add_graph_feature_extractor_arguments,
        make_graph_feature_extractor_config_from_args,
        add_train_cli_args,
        set_up_train_run,
        PrototypicalNetworkTrainerConfig,
        PrototypicalNetworkTrainer,
    )


def run_cliff_batches(
    model,
    *,
    batches: list[CliffProtoNetBatch],
    batch_labels: list[torch.Tensor],
    train: bool = False,
    tasks_per_batch: int = 1,
):
    _, _, _, _, _, _, _, compute_binary_task_metrics, _, _, _, _ = _load_training_runtime_symbols()
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_num_samples = 0.0, 0
    task_preds = []
    task_labels = []
    num_gradient_accumulation_steps = len(batches) * tasks_per_batch
    for batch_features, labels in zip(batches, batch_labels, strict=True):
        logits = model(batch_features)
        batch_loss = model.compute_loss(logits, labels, batch_features) / num_gradient_accumulation_steps
        if train:
            batch_loss.backward()
        total_loss += batch_loss.detach() * batch_features.num_query_samples * num_gradient_accumulation_steps
        total_num_samples += batch_features.num_query_samples
        batch_preds = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        task_preds.append(batch_preds[:, 1])
        task_labels.append(labels.detach().cpu().numpy())

    metrics = compute_binary_task_metrics(
        predictions=np.concatenate(task_preds, axis=0),
        labels=np.concatenate(task_labels, axis=0),
    )
    return total_loss.cpu().item() / total_num_samples, metrics


def _get_cliff_protonet_task_sample_iterable(
    *,
    dataset,
    data_fold,
    data_root: Path | None,
    num_samples: int,
    support_size: int,
    query_size: int,
    max_num_graphs: int,
    repeat: bool,
):
    _, _, _, _, _, _, _, _, get_protonet_batcher, _, _, _ = _load_training_runtime_symbols()
    FSMolTask, StratifiedTaskSampler, _ = _load_task_sampling_symbols()
    task_sampler = StratifiedTaskSampler(train_size_or_ratio=support_size, test_size_or_ratio=query_size)
    batcher = get_protonet_batcher(max_num_graphs=max_num_graphs)

    def path_to_batches_pipeline(paths, idx):
        if len(paths) > 1:
            raise ValueError("Expected a single task path per batch pipeline.")
        task_path = paths[0]
        task = FSMolTask.load_from_file(task_path)
        sample_to_log_r = _load_log_r_by_sample_id(task_path, task.samples)
        num_task_samples = 0
        for _ in range(num_samples):
            try:
                task_sample = task_sampler.sample(task, seed=idx + num_task_samples)
                num_task_samples += 1
            except Exception as exc:
                logger.debug("%s: cliff-margin sampling failed: %s", task.name, exc)
                continue
            yield task_sample_to_cliff_pn_task_sample(
                task_sample=task_sample,
                batcher=batcher,
                sample_to_log_r=sample_to_log_r,
            )

    return dataset.get_task_reading_iterable(
        data_fold=data_fold,
        task_reader_fn=path_to_batches_pipeline,
        repeat=repeat,
    )


def task_sample_to_cliff_pn_task_sample(
    *,
    task_sample,
    batcher,
    sample_to_log_r: dict[int, float],
) -> FeaturisedCliffPNTaskSample:
    support_batches = list(batcher.batch(task_sample.train_samples))
    if len(support_batches) > 1:
        raise ValueError("Support set too large to fit into a single batch.")
    support_features, support_labels = support_batches[0]
    support_log_r = np.array([sample_to_log_r[id(sample)] for sample in task_sample.train_samples], dtype=np.float32)

    try:
        orig_max_num_graphs = batcher._max_num_graphs
        batcher._max_num_graphs = orig_max_num_graphs - support_features.num_graphs
        sample_batches = []
        batch_labels = []
        query_offset = 0
        for query_features, query_labels in batcher.batch(task_sample.test_samples):
            batch_size = int(len(query_labels))
            query_samples = task_sample.test_samples[query_offset : query_offset + batch_size]
            query_log_r = np.array([sample_to_log_r[id(sample)] for sample in query_samples], dtype=np.float32)
            sample_batches.append(
                CliffProtoNetBatch(
                    support_features=support_features,
                    support_labels=support_labels,
                    support_log_r=support_log_r,
                    query_features=query_features,
                    query_log_r=query_log_r,
                    task_name=task_sample.name,
                )
            )
            batch_labels.append(query_labels)
            query_offset += batch_size
    finally:
        batcher._max_num_graphs = orig_max_num_graphs

    return FeaturisedCliffPNTaskSample(
        task_name=task_sample.name,
        num_support_samples=len(task_sample.train_samples),
        num_positive_support_samples=sum(sample.bool_label for sample in task_sample.train_samples),
        num_query_samples=len(task_sample.test_samples),
        num_positive_query_samples=sum(sample.bool_label for sample in task_sample.test_samples),
        batches=sample_batches,
        batch_labels=batch_labels,
    )


def _load_log_r_by_sample_id(task_path, task_samples) -> dict[int, float]:
    rows = list(task_path.read_by_file_suffix())
    return {
        id(sample): float(rows[index].get("LogRegressionProperty") or "nan")
        for index, sample in enumerate(task_samples)
    }


def _load_task_sampling_symbols():
    install_fs_mol_compat_patches()
    from fs_mol.data.fsmol_task import FSMolTask
    from fs_mol.data import StratifiedTaskSampler, DataFold

    return FSMolTask, StratifiedTaskSampler, DataFold


def _load_training_runtime_symbols():
    install_fs_mol_compat_patches()
    from fs_mol.data import DataFold
    from fs_mol.utils.metrics import compute_binary_task_metrics
    from fs_mol.utils.metric_logger import MetricLogger
    from fs_mol.utils.protonet_utils import validate_by_finetuning_on_tasks
    from fs_mol.data.protonet import get_protonet_batcher
    from fs_mol.utils.torch_utils import torchify

    return (
        None,
        None,
        None,
        None,
        None,
        None,
        DataFold,
        compute_binary_task_metrics,
        get_protonet_batcher,
        torchify,
        validate_by_finetuning_on_tasks,
        MetricLogger,
    )


if __name__ == "__main__":
    main()
