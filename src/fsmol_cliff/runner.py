from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .adapters import score_official_baseline_episode, score_sklearn_episode
from .evaluation import evaluate_episode_manifest, summarize_task_metric_rows
from .io import write_parquet


def evaluate_release_with_sklearn_baseline(
    *,
    release_dir: Path,
    output_path: Path,
    split_types: Sequence[str] = ("standard", "adversarial"),
    model_name: str = "kNN",
    model_params: dict | None = None,
    backend: str = "local",
) -> list[dict]:
    assay_context_cache: dict[str, dict] = {}
    episode_results: list[dict] = []
    for split_type in split_types:
        manifest_path = release_dir / f"episodes_{split_type}.parquet"
        if not manifest_path.exists():
            continue
        frame = pd.read_parquet(manifest_path)
        for episode in frame.to_dict(orient="records"):
            task_id = episode["task_id"]
            assay_context = assay_context_cache.setdefault(task_id, _load_assay_context(release_dir, task_id))
            scorer = score_sklearn_episode if backend == "local" else score_official_baseline_episode
            episode_results.append(
                evaluate_episode_manifest(
                    episode=episode,
                    assay_context=assay_context,
                    score_fn=lambda current_episode, *, context=assay_context, scorer=scorer: scorer(
                        model_name=model_name,
                        assay_id=current_episode["task_id"],
                        records_by_id=context["records_by_id"],
                        support_ids=[*current_episode["support_pos_ids"], *current_episode["support_neg_ids"]],
                        query_ids=[*current_episode["query_pos_ids"], *current_episode["query_neg_ids"]],
                        use_grid_search=False,
                        model_params=model_params,
                    ),
                )
            )
    rows = summarize_task_metric_rows(episode_results)
    write_parquet(output_path, rows)
    return rows


def _load_assay_context(release_dir: Path, task_id: str) -> dict:
    assay_dir = release_dir / "assays" / task_id
    annotations = pd.read_parquet(assay_dir / "molecule_annotations.parquet").to_dict(orient="records")
    pairs_path = assay_dir / "pairs.jsonl"
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
    }
