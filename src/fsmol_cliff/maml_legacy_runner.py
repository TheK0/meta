from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .maml_legacy import enable_legacy_adam_optimizer, maml_checkpoint_args, split_support_for_validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--profile", default="strict")
    parser.add_argument("--split-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--holdout-per-class", type=int, default=1)
    parser.add_argument("--max-num-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    args = parser.parse_args()

    enable_legacy_adam_optimizer()

    from dpu_utils.utils import RichPath
    from fs_mol.data.fsmol_task import FSMolTask, FSMolTaskSample
    from fs_mol.maml_test import load_model_for_eval
    from fs_mol.utils.maml_utils import train_loop, validate_on_data_iterable
    from fs_mol.data.maml import TFGraphBatchIterable
    from tf2_gnn.cli_utils.model_utils import load_weights_verbosely
    from functools import partial

    release_dir = Path(args.release_dir)
    data_dir = Path(args.data_dir)
    manifest_path = release_dir / f"episodes_{args.split_type}_{args.profile}.parquet"
    if not manifest_path.exists():
        manifest_path = release_dir / f"episodes_{args.split_type}.parquet"
    manifest = pd.read_parquet(manifest_path)
    episodes = manifest[
        (manifest["task_id"] == args.task_id) & (manifest["seed"] == args.seed)
    ].head(args.max_episodes).to_dict(orient="records")

    task_path = RichPath.create(str(data_dir / "test" / f"{args.task_id}.jsonl.gz"))
    task = FSMolTask.load_from_file(task_path)
    sample_map = {f"{args.task_id}:{idx}": sample for idx, sample in enumerate(task.samples)}

    model = load_model_for_eval(maml_checkpoint_args(args.checkpoint))
    base_model_weights = {var.name: var.value() for var in model.trainable_variables}

    output_rows = []
    for episode in episodes:
        train_ids, valid_ids = split_support_for_validation(
            support_pos_ids=list(episode["support_pos_ids"]),
            support_neg_ids=list(episode["support_neg_ids"]),
            holdout_per_class=args.holdout_per_class,
        )
        support_ids = list(episode["support_pos_ids"]) + list(episode["support_neg_ids"])
        test_ids = list(episode["query_pos_ids"]) + list(episode["query_neg_ids"])
        task_sample = FSMolTaskSample(
            name=args.task_id,
            train_samples=[sample_map[molecule_id] for molecule_id in train_ids],
            valid_samples=[sample_map[molecule_id] for molecule_id in valid_ids],
            test_samples=[sample_map[molecule_id] for molecule_id in test_ids],
        )
        temp_dir = Path("/tmp") / f"maml_legacy_runner_{args.split_type}_{episode['episode_id']}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        model_save_file = str(temp_dir / "best_model.pkl")
        for var in model.trainable_variables:
            model_var_name = var.name.split("/", 1)[1] if var.name.startswith("valid/") else var.name
            var.assign(base_model_weights[model_var_name])
        model.reset_optimizer_state_to_initial()
        train_loop(
            model=model,
            train_data=TFGraphBatchIterable(samples=task_sample.train_samples, max_num_nodes=10000),
            valid_fn=partial(
                validate_on_data_iterable,
                data_iterable=TFGraphBatchIterable(samples=task_sample.valid_samples, max_num_nodes=10000),
                metric_to_use="loss",
                quiet=True,
            ),
            model_save_file=model_save_file,
            metric_to_use="avg_precision",
            max_num_epochs=args.max_num_epochs,
            patience=args.patience,
            quiet=True,
        )
        load_weights_verbosely(model_save_file, model)
        support_iter = TFGraphBatchIterable(samples=[sample_map[molecule_id] for molecule_id in support_ids], max_num_nodes=10000)
        test_iter = TFGraphBatchIterable(samples=task_sample.test_samples, max_num_nodes=10000)
        support_predictions = model.predict(support_iter).numpy().tolist()
        query_predictions = model.predict(test_iter).numpy().tolist()
        output_rows.append(
            {
                "task_id": args.task_id,
                "seed": args.seed,
                "split_type": args.split_type,
                "episode_id": episode["episode_id"],
                "scores": {
                    **{molecule_id: float(score) for molecule_id, score in zip(support_ids, support_predictions)},
                    **{molecule_id: float(score) for molecule_id, score in zip(test_ids, query_predictions)},
                },
                "num_test": len(test_ids),
            }
        )

    Path(args.output).write_text(json.dumps(output_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
