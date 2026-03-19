from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from typing import Any


def _is_valid_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isnan(float(value))
    return False


def task_mean(episode_values: Iterable[float | None]) -> dict[str, float | int | None]:
    values = list(episode_values)
    valid_values = [float(value) for value in values if _is_valid_number(value)]
    total_count = len(values)
    valid_count = len(valid_values)
    mean = None if not valid_values else sum(valid_values) / valid_count
    coverage = 0.0 if total_count == 0 else valid_count / total_count
    return {
        "mean": mean,
        "coverage": coverage,
        "valid_count": valid_count,
        "total_count": total_count,
    }


task_level_mean = task_mean


def _iter_task_summaries(
    task_summaries: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(task_summaries, Mapping):
        return list(task_summaries.values())
    return list(task_summaries)


def macro_mean(
    task_summaries: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    summaries = _iter_task_summaries(task_summaries)
    valid_means = [
        float(summary["mean"])
        for summary in summaries
        if _is_valid_number(summary.get("mean")) and int(summary.get("valid_count", 0)) > 0
    ]
    total_count = len(summaries)
    valid_count = len(valid_means)
    mean = None if not valid_means else sum(valid_means) / valid_count
    coverage = 0.0 if total_count == 0 else valid_count / total_count
    return {
        "mean": mean,
        "coverage": coverage,
        "valid_count": valid_count,
        "total_count": total_count,
    }


macro_average = macro_mean


def task_bootstrap_ci(
    task_values: Iterable[float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    values = [float(value) for value in task_values]
    if not values:
        raise ValueError("task_values must not be empty")

    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "low": means[int(0.025 * (iterations - 1))],
        "high": means[int(0.975 * (iterations - 1))],
    }


def paired_bootstrap_delta_ci(
    *,
    baseline: Iterable[float],
    treatment: Iterable[float],
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    baseline_values = [float(value) for value in baseline]
    treatment_values = [float(value) for value in treatment]
    if len(baseline_values) != len(treatment_values) or not baseline_values:
        raise ValueError("baseline and treatment must be non-empty and have matching lengths")

    deltas = [treatment_value - baseline_value for baseline_value, treatment_value in zip(baseline_values, treatment_values)]
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample = [rng.choice(deltas) for _ in range(len(deltas))]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    return {
        "delta_mean": sum(deltas) / len(deltas),
        "low": samples[int(0.025 * (iterations - 1))],
        "high": samples[int(0.975 * (iterations - 1))],
    }


def aggregate_task_result_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        score = row.get("score")
        if not _is_valid_number(score):
            continue
        split_metric = (str(row["split_type"]), str(row["metric"]))
        task_scores = grouped.setdefault(split_metric, {})
        task_scores.setdefault(str(row["task_id"]), []).append(float(score))

    aggregated = []
    for (split_type, metric), task_scores in sorted(grouped.items()):
        per_task_means = [sum(scores) / len(scores) for _, scores in sorted(task_scores.items())]
        ci = task_bootstrap_ci(
            per_task_means,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        aggregated.append(
            {
                "split_type": split_type,
                "metric": metric,
                "score": ci["mean"],
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "num_tasks": len(per_task_means),
            }
        )
    return aggregated


__all__ = [
    "aggregate_task_result_rows",
    "macro_average",
    "macro_mean",
    "paired_bootstrap_delta_ci",
    "task_bootstrap_ci",
    "task_level_mean",
    "task_mean",
]
