from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Callable, Mapping

from sklearn.metrics import average_precision_score

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
    support_ids = [*episode["support_pos_ids"], *episode["support_neg_ids"]]
    query_ids = [*episode["query_pos_ids"], *episode["query_neg_ids"]]
    query_pos_ids = set(episode["query_pos_ids"])
    query_neg_ids = set(episode["query_neg_ids"])
    query_pairs = [
        PairRecord(**pair)
        for pair in assay_context.get("cliff_pairs", [])
        if pair["anchor_id"] in query_pos_ids and pair["neg_id"] in query_neg_ids
    ]
    noncliff_pairs = [
        PairRecord(**pair)
        for pair in assay_context.get("noncliff_pairs", [])
        if pair["anchor_id"] in query_pos_ids and pair["neg_id"] in query_neg_ids
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
        "average_precision_score": average_precision_for_query_predictions(labels, scores, query_ids),
        "delta_auprc": delta_auprc_for_query_predictions(labels, scores, query_ids),
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
        "average_precision_score": 0,
        "delta_auprc": 0,
    }
    episode_context = {
        "num_train_requested": len(support_ids),
        "num_train": len(support_ids),
        "fraction_positive_train": fraction_positive_for_query_ids(labels, support_ids),
        "num_test": len(query_ids),
        "fraction_positive_test": fraction_positive_for_query_ids(labels, query_ids),
        "num_support_molecules": len(support_ids),
        "fraction_positive_support": fraction_positive_for_query_ids(labels, support_ids),
        "num_query_molecules": len(query_ids),
        "fraction_positive_query": fraction_positive_for_query_ids(labels, query_ids),
    }
    return {
        "task_id": episode["task_id"],
        "seed": episode["seed"],
        "split_type": episode["split_type"],
        "episode_id": episode["episode_id"],
        "metrics": metrics,
        "pair_counts": pair_counts,
        "episode_context": episode_context,
    }


def summarize_task_metric_rows(episode_results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int, str, str], list[dict]] = defaultdict(list)
    for result in episode_results:
        for metric, value in result["metrics"].items():
            grouped[
                (
                    str(result.get("profile", "strict")),
                    str(result.get("result_tier", "final")),
                    result["task_id"],
                    result["seed"],
                    result["split_type"],
                    metric,
                )
            ].append(
                {
                    "value": value,
                    "pair_count": result["pair_counts"].get(metric, 0),
                    "episode_context": result.get("episode_context", {}),
                }
            )

    rows = []
    for (profile, result_tier, task_id, seed, split_type, metric), entries in grouped.items():
        summary = task_mean(entry["value"] for entry in entries)
        valid_pair_counts = [entry["pair_count"] for entry in entries if entry["value"] is not None]
        episode_context = _summarize_episode_context(entry["episode_context"] for entry in entries)
        rows.append(
            {
                "profile": profile,
                "result_tier": result_tier,
                "task_id": task_id,
                "seed": seed,
                "split_type": split_type,
                "metric": metric,
                "score": summary["mean"],
                "coverage": summary["coverage"],
                "num_episodes": summary["total_count"],
                "num_valid_episodes": summary["valid_count"],
                "mean_num_valid_pairs_per_episode": (
                    sum(valid_pair_counts) / len(valid_pair_counts) if valid_pair_counts else 0.0
                ),
                **episode_context,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["profile"],
            row["result_tier"],
            row["task_id"],
            row["seed"],
            row["split_type"],
            row["metric"],
        ),
    )


def fraction_positive_for_query_ids(
    labels: Mapping[str, int],
    query_ids: Iterable[str],
) -> float | None:
    valid_ids = [molecule_id for molecule_id in query_ids if molecule_id in labels]
    if not valid_ids:
        return None
    positives = sum(int(labels[molecule_id]) for molecule_id in valid_ids)
    return positives / len(valid_ids)


def average_precision_for_query_predictions(
    labels: Mapping[str, int],
    scores: Mapping[str, float],
    query_ids: Iterable[str],
) -> float | None:
    valid_ids = _scored_query_ids(labels, scores, query_ids)
    if not valid_ids:
        return None
    y_true = [int(labels[molecule_id]) for molecule_id in valid_ids]
    y_score = [float(scores[molecule_id]) for molecule_id in valid_ids]
    return float(average_precision_score(y_true, y_score))


def delta_auprc_for_query_predictions(
    labels: Mapping[str, int],
    scores: Mapping[str, float],
    query_ids: Iterable[str],
) -> float | None:
    valid_ids = _scored_query_ids(labels, scores, query_ids)
    if not valid_ids:
        return None
    ap = average_precision_for_query_predictions(labels, scores, valid_ids)
    if ap is None:
        return None
    fraction_positive = fraction_positive_for_query_ids(labels, valid_ids)
    if fraction_positive is None:
        return None
    return ap - fraction_positive


def _cliff_query_ids(pairs: list[PairRecord]) -> list[str]:
    ids = []
    seen = set()
    for pair in pairs:
        for molecule_id in (pair.anchor_id, pair.neg_id):
            if molecule_id not in seen:
                ids.append(molecule_id)
                seen.add(molecule_id)
    return ids


def _scored_query_ids(
    labels: Mapping[str, int],
    scores: Mapping[str, float],
    query_ids: Iterable[str],
) -> list[str]:
    return [molecule_id for molecule_id in query_ids if molecule_id in labels and molecule_id in scores]


def _summarize_episode_context(episode_contexts: Iterable[Mapping[str, float | int | None]]) -> dict[str, float | None]:
    context_list = list(episode_contexts)
    num_support_summary = task_mean(context.get("num_support_molecules") for context in context_list)
    fraction_positive_support_summary = task_mean(context.get("fraction_positive_support") for context in context_list)
    num_query_summary = task_mean(context.get("num_query_molecules") for context in context_list)
    fraction_positive_query_summary = task_mean(context.get("fraction_positive_query") for context in context_list)
    return {
        "num_train_requested": num_support_summary["mean"],
        "num_train": num_support_summary["mean"],
        "fraction_positive_train": fraction_positive_support_summary["mean"],
        "num_test": num_query_summary["mean"],
        "fraction_positive_test": fraction_positive_query_summary["mean"],
        "num_support_molecules": num_support_summary["mean"],
        "fraction_positive_support": fraction_positive_support_summary["mean"],
        "num_query_molecules": num_query_summary["mean"],
        "fraction_positive_query": fraction_positive_query_summary["mean"],
    }
