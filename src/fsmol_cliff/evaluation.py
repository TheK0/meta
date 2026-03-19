from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping

from .aggregate import task_mean
from .metrics import c_bacc, nc_bacc, nc_psr, q_psr, scr, sq_psr, ss_q_psr, ss_scr, ss_sq_psr
from .models import PairRecord


def evaluate_episode_manifest(
    *,
    episode: dict,
    assay_context: dict,
    score_fn: Callable[[dict], dict[str, float]],
) -> dict:
    labels = assay_context["labels"]
    scores = score_fn(episode)
    predictions = {molecule_id: 1 if score >= 0.5 else 0 for molecule_id, score in scores.items()}
    query_pairs = [
        PairRecord(**pair)
        for pair in assay_context.get("cliff_pairs", [])
        if pair["anchor_id"] in set(episode["query_pos_ids"]) and pair["neg_id"] in set(episode["query_neg_ids"])
    ]
    noncliff_pairs = [
        PairRecord(**pair)
        for pair in assay_context.get("noncliff_pairs", [])
        if pair["anchor_id"] in set(episode["query_pos_ids"]) and pair["neg_id"] in set(episode["query_neg_ids"])
    ]
    support_query_pairs = [PairRecord(**pair) for pair in episode.get("injected_pairs", [])]
    cliff_query_ids = _cliff_query_ids(query_pairs)
    noncliff_query_ids = _cliff_query_ids(noncliff_pairs)
    metrics = {
        "c_bacc": c_bacc(None, labels, predictions, cliff_query_ids),
        "nc_bacc": nc_bacc(None, labels, predictions, noncliff_query_ids),
        "q_psr": q_psr(None, query_pairs, scores),
        "nc_psr": nc_psr(None, noncliff_pairs, scores),
        "sq_psr": sq_psr(None, support_query_pairs, scores),
        "scr": scr(None, query_pairs + noncliff_pairs, predictions),
        "ss_q_psr": ss_q_psr(None, query_pairs, scores),
        "ss_scr": ss_scr(None, query_pairs + noncliff_pairs, predictions),
        "ss_sq_psr": ss_sq_psr(None, support_query_pairs, scores),
    }
    pair_counts = {
        "c_bacc": len(cliff_query_ids),
        "nc_bacc": len(noncliff_query_ids),
        "q_psr": len(query_pairs),
        "nc_psr": len(noncliff_pairs),
        "sq_psr": len(support_query_pairs),
        "scr": len(query_pairs) + len(noncliff_pairs),
        "ss_q_psr": sum(pair.same_scaffold for pair in query_pairs),
        "ss_scr": sum(pair.same_scaffold for pair in [*query_pairs, *noncliff_pairs]),
        "ss_sq_psr": sum(pair.same_scaffold for pair in support_query_pairs),
    }
    return {
        "task_id": episode["task_id"],
        "seed": episode["seed"],
        "split_type": episode["split_type"],
        "episode_id": episode["episode_id"],
        "metrics": metrics,
        "pair_counts": pair_counts,
    }


def summarize_task_metric_rows(episode_results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for result in episode_results:
        for metric, value in result["metrics"].items():
            grouped[(result["task_id"], result["seed"], result["split_type"], metric)].append(
                {
                    "value": value,
                    "pair_count": result["pair_counts"].get(metric, 0),
                }
            )

    rows = []
    for (task_id, seed, split_type, metric), entries in grouped.items():
        summary = task_mean(entry["value"] for entry in entries)
        valid_pair_counts = [entry["pair_count"] for entry in entries if entry["value"] is not None]
        rows.append(
            {
                "task_id": task_id,
                "seed": seed,
                "split_type": split_type,
                "metric": metric,
                "score": summary["mean"],
                "coverage": summary["coverage"],
                "num_valid_episodes": summary["valid_count"],
                "mean_num_valid_pairs_per_episode": (
                    sum(valid_pair_counts) / len(valid_pair_counts) if valid_pair_counts else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["task_id"], row["seed"], row["split_type"], row["metric"]))


def _cliff_query_ids(pairs: list[PairRecord]) -> list[str]:
    ids = []
    seen = set()
    for pair in pairs:
        for molecule_id in (pair.anchor_id, pair.neg_id):
            if molecule_id not in seen:
                ids.append(molecule_id)
                seen.add(molecule_id)
    return ids
