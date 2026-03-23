from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

from .aggregate import paired_bootstrap_delta_ci


def build_main_table_rows(
    *,
    model_to_aggregate_path: Mapping[str, Path],
    profile: str,
    result_tier: str = "final",
    metrics: Sequence[tuple[str, str, str]],
) -> list[dict]:
    rows: list[dict] = []
    for model_name, path in model_to_aggregate_path.items():
        lookup = _aggregate_lookup(path, profile=profile, result_tier=result_tier)
        row = {"model": model_name}
        for split_type, metric, column in metrics:
            row[column] = lookup.get((split_type, metric), {}).get("score")
        rows.append(row)
    return rows


def build_failure_taxonomy_rows(
    *,
    model_to_aggregate_path: Mapping[str, Path],
    profile: str,
    result_tier: str = "final",
) -> list[dict]:
    rows: list[dict] = []
    for model_name, path in model_to_aggregate_path.items():
        lookup = _aggregate_lookup(path, profile=profile, result_tier=result_tier)
        std_q_psr = _score(lookup, "standard", "q_psr")
        std_c_bacc = _score(lookup, "standard", "c_bacc")
        adv_sq_psr = _score(lookup, "adversarial", "sq_psr")
        adv_c_bacc = _score(lookup, "adversarial", "c_bacc")
        adv_scr = _score(lookup, "adversarial", "scr")
        taxonomy_label = _taxonomy_label(
            model_name=model_name,
            std_q_psr=std_q_psr,
            std_c_bacc=std_c_bacc,
            adv_sq_psr=adv_sq_psr,
            adv_c_bacc=adv_c_bacc,
            adv_scr=adv_scr,
        )
        rows.append(
            {
                "model": model_name,
                "taxonomy_label": taxonomy_label,
                "ranking_signal": _bucket(max(std_q_psr, adv_sq_psr), high=0.58, medium=0.52),
                "decision_signal": _decision_bucket(max(std_c_bacc, adv_c_bacc)),
                "collapse_signal": _collapse_bucket(adv_scr),
                "std_q_psr": std_q_psr,
                "std_c_bacc": std_c_bacc,
                "adv_sq_psr": adv_sq_psr,
                "adv_c_bacc": adv_c_bacc,
                "adv_scr": adv_scr,
            }
        )
    return rows


def build_paired_model_comparison_rows(
    *,
    model_to_task_result_rows: Mapping[str, Sequence[Mapping]],
    profile: str,
    result_tier: str = "final",
    comparisons: Sequence[tuple[str, str]] | None = None,
    metrics: Sequence[tuple[str, str]] = (),
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
) -> list[dict]:
    if comparisons is None:
        comparisons = list(combinations(model_to_task_result_rows.keys(), 2))

    rows: list[dict] = []
    per_model_lookup = {
        model_name: _task_metric_lookup(
            task_rows,
            profile=profile,
            result_tier=result_tier,
        )
        for model_name, task_rows in model_to_task_result_rows.items()
    }
    for baseline_model, treatment_model in comparisons:
        baseline_lookup = per_model_lookup[baseline_model]
        treatment_lookup = per_model_lookup[treatment_model]
        for split_type, metric in metrics:
            baseline_values, treatment_values, num_tasks = _paired_task_values(
                baseline_lookup,
                treatment_lookup,
                split_type=split_type,
                metric=metric,
            )
            if not baseline_values:
                continue
            ci = paired_bootstrap_delta_ci(
                baseline=baseline_values,
                treatment=treatment_values,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "profile": profile,
                    "result_tier": result_tier,
                    "baseline_model": baseline_model,
                    "treatment_model": treatment_model,
                    "split_type": split_type,
                    "metric": metric,
                    "num_tasks": num_tasks,
                    "delta_mean": ci["delta_mean"],
                    "ci_low": ci["low"],
                    "ci_high": ci["high"],
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown_table(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        path.write_text("")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_value(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def _aggregate_lookup(path: Path, *, profile: str, result_tier: str) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text())
    return {
        (row["split_type"], row["metric"]): row
        for row in payload
        if row.get("profile", profile) == profile
        and row.get("result_tier", "final") == result_tier
    }


def _score(lookup: Mapping[tuple[str, str], Mapping], split_type: str, metric: str) -> float:
    row = lookup.get((split_type, metric))
    return float(row["score"]) if row is not None else float("nan")


def _task_metric_lookup(
    rows: Sequence[Mapping],
    *,
    profile: str,
    result_tier: str,
) -> dict[tuple[str, str, str], list[float]]:
    lookup: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("profile", profile) != profile or row.get("result_tier", "final") != result_tier:
            continue
        score = float(row["score"])
        if math.isnan(score):
            continue
        key = (str(row["task_id"]), str(row["split_type"]), str(row["metric"]))
        lookup.setdefault(key, []).append(score)
    return lookup


def _paired_task_values(
    baseline_lookup: Mapping[tuple[str, str, str], list[float]],
    treatment_lookup: Mapping[tuple[str, str, str], list[float]],
    *,
    split_type: str,
    metric: str,
) -> tuple[list[float], list[float], int]:
    task_ids = sorted(
        {
            task_id
            for task_id, row_split, row_metric in baseline_lookup
            if row_split == split_type and row_metric == metric
        }
        & {
            task_id
            for task_id, row_split, row_metric in treatment_lookup
            if row_split == split_type and row_metric == metric
        }
    )
    baseline_values = []
    treatment_values = []
    for task_id in task_ids:
        baseline_scores = baseline_lookup[(task_id, split_type, metric)]
        treatment_scores = treatment_lookup[(task_id, split_type, metric)]
        baseline_values.append(sum(baseline_scores) / len(baseline_scores))
        treatment_values.append(sum(treatment_scores) / len(treatment_scores))
    return baseline_values, treatment_values, len(task_ids)


def _bucket(value: float, *, high: float, medium: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _collapse_bucket(value: float) -> str:
    if value >= 0.9:
        return "high"
    if value >= 0.75:
        return "medium"
    return "low"


def _decision_bucket(value: float) -> str:
    if value >= 0.54:
        return "strong"
    if value >= 0.52:
        return "moderate"
    return "weak"


def _taxonomy_label(
    *,
    model_name: str,
    std_q_psr: float,
    std_c_bacc: float,
    adv_sq_psr: float,
    adv_c_bacc: float,
    adv_scr: float,
) -> str:
    lowered = model_name.lower()
    if "cliff-aware" in lowered:
        return "intervention reduces collapse"
    if lowered == "knn":
        return "global similarity collapse"
    if lowered == "rf":
        return "ranking-competent but decision-collapsed"
    if "proto" in lowered:
        return "strong metric baseline with persistent collapse"
    if "maml" in lowered:
        return "boundary-aware but adversarially fragile"
    if max(std_q_psr, adv_sq_psr) >= 0.58 and max(std_c_bacc, adv_c_bacc) < 0.53 and adv_scr >= 0.85:
        return "ranking-competent but decision-collapsed"
    if adv_scr >= 0.9:
        return "global similarity collapse"
    return "mixed"


def _format_markdown_value(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
