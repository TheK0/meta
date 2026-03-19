from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scipy.stats import spearmanr

from .aggregate import paired_bootstrap_delta_ci


def _extract_mean(metric_summary: Any) -> float | None:
    if metric_summary is None:
        return None
    if isinstance(metric_summary, Mapping):
        if "mean" in metric_summary:
            metric_summary = metric_summary.get("mean")
        elif "task_values" in metric_summary:
            values = [float(value) for value in metric_summary.get("task_values", [])]
            return None if not values else sum(values) / len(values)
        else:
            return None
    if isinstance(metric_summary, bool):
        return None
    if isinstance(metric_summary, (int, float)):
        value = float(metric_summary)
        return None if math.isnan(value) else value
    return None


def _decision(
    summary: Mapping[str, Any],
    *,
    rules: Mapping[str, tuple[str, str, str]],
) -> dict[str, Any]:
    missing_metrics = sorted(
        {
            metric_name
            for left_metric, _, right_metric in rules.values()
            for metric_name in (left_metric, right_metric)
            if _extract_mean(summary.get(metric_name)) is None
        }
    )
    checks = {}
    for check_name, (left_metric, operator, right_metric) in rules.items():
        left_value = _extract_mean(summary.get(left_metric))
        right_value = _extract_mean(summary.get(right_metric))
        if left_value is None or right_value is None:
            checks[check_name] = False
            continue
        if operator == "<":
            checks[check_name] = left_value < right_value
        elif operator == ">":
            checks[check_name] = left_value > right_value
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    return {
        "accepted": not missing_metrics and all(checks.values()),
        "checks": checks,
        "missing_metrics": tuple(missing_metrics),
    }


def _extract_task_values(metric_summary: Any) -> list[float]:
    if metric_summary is None:
        return []
    if isinstance(metric_summary, Mapping):
        values = metric_summary.get("task_values", [])
        return [float(value) for value in values]
    return []


def compute_cliff_gap_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    c_bacc = _extract_mean(summary.get("c_bacc"))
    nc_bacc = _extract_mean(summary.get("nc_bacc"))
    q_psr = _extract_mean(summary.get("q_psr"))
    nc_psr = _extract_mean(summary.get("nc_psr"))
    return {
        "cliffgap_bacc": None if c_bacc is None or nc_bacc is None else nc_bacc - c_bacc,
        "cliffgap_psr": None if q_psr is None or nc_psr is None else nc_psr - q_psr,
    }


def validate_h1_model_set(
    model_set: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if len(model_set) < 3:
        return {"accepted": False, "reason": "requires at least 3 models", "gap_supported": False, "ranking_disagreement": False}

    gap_supported = True
    official_scores: list[tuple[str, float]] = []
    cliff_scores: list[tuple[str, float]] = []
    for model_name, summary in model_set.items():
        gap_metrics = compute_cliff_gap_metrics(summary)
        gap_bacc_values = _paired_gap_values(summary, "nc_bacc", "c_bacc")
        gap_psr_values = _paired_gap_values(summary, "nc_psr", "q_psr")
        supported = False
        for values in (gap_bacc_values, gap_psr_values):
            if values:
                ci = paired_bootstrap_delta_ci(
                    baseline=[0.0] * len(values),
                    treatment=values,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                if ci["low"] > 0:
                    supported = True
                    break
        gap_supported = gap_supported and supported

        official_mean = _extract_mean(summary.get("official"))
        cliff_mean = _extract_mean(summary.get("q_psr")) or _extract_mean(summary.get("c_bacc"))
        if official_mean is not None and cliff_mean is not None:
            official_scores.append((model_name, official_mean))
            cliff_scores.append((model_name, cliff_mean))

    official_order = [name for name, _ in sorted(official_scores, key=lambda item: item[1], reverse=True)]
    cliff_order = [name for name, _ in sorted(cliff_scores, key=lambda item: item[1], reverse=True)]
    ranking_disagreement = official_order != cliff_order
    correlation = None
    if len(official_scores) >= 3 and len(cliff_scores) >= 3:
        official_rank_values = [score for _, score in official_scores]
        cliff_rank_values = [dict(cliff_scores)[name] for name, _ in official_scores]
        correlation = float(spearmanr(official_rank_values, cliff_rank_values).statistic)
        if correlation <= 0.5:
            ranking_disagreement = True

    return {
        "accepted": gap_supported and ranking_disagreement,
        "gap_supported": gap_supported,
        "ranking_disagreement": ranking_disagreement,
        "spearman": correlation,
    }


def validate_h3_intervention(
    *,
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    cliff_improved = _positive_delta_support(
        baseline,
        treatment,
        metrics=("c_bacc", "q_psr", "sq_psr"),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    collapse_reduced = _negative_delta_support(
        baseline,
        treatment,
        metrics=("scr", "ss_scr"),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    ) or _positive_delta_support(
        baseline,
        treatment,
        metrics=("ss_q_psr", "ss_sq_psr"),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    controls_preserved = not _negative_delta_support(
        baseline,
        treatment,
        metrics=("official", "nc_bacc", "nc_psr"),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        strict=True,
    )
    return {
        "accepted": cliff_improved and collapse_reduced and controls_preserved,
        "cliff_improved": cliff_improved,
        "collapse_reduced": collapse_reduced,
        "controls_preserved": controls_preserved,
    }


def validate_h2_shortcut_reliance(
    summary: Mapping[str, Any],
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    evidence = {
        "scr_above_zero": (_extract_mean(summary.get("scr")) or 0.0) > 0.0,
        "same_scaffold_harder": (
            ((_extract_mean(summary.get("ss_scr")) or float("-inf")) > (_extract_mean(summary.get("scr")) or float("inf")))
            or ((_extract_mean(summary.get("ss_q_psr")) or float("inf")) < (_extract_mean(summary.get("q_psr")) or float("-inf")))
        ),
        "intervention_linked": False,
    }
    baseline = summary.get("baseline")
    treatment = summary.get("treatment")
    if isinstance(baseline, Mapping) and isinstance(treatment, Mapping):
        qpsr_improved = _positive_delta_support(
            baseline,
            treatment,
            metrics=("q_psr", "sq_psr"),
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        scr_reduced = _negative_delta_support(
            baseline,
            treatment,
            metrics=("scr", "ss_scr"),
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        evidence["intervention_linked"] = qpsr_improved and scr_reduced

    num_supported_conditions = sum(bool(value) for value in evidence.values())
    return {
        "accepted": num_supported_conditions >= 2,
        "conditions": evidence,
        "num_supported_conditions": num_supported_conditions,
    }


def _paired_gap_values(summary: Mapping[str, Any], left_metric: str, right_metric: str) -> list[float]:
    left_values = _extract_task_values(summary.get(left_metric))
    right_values = _extract_task_values(summary.get(right_metric))
    if len(left_values) != len(right_values) or not left_values:
        return []
    return [left - right for left, right in zip(left_values, right_values)]


def _positive_delta_support(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    metrics: tuple[str, ...],
    iterations: int,
    seed: int,
) -> bool:
    for metric in metrics:
        baseline_values = _extract_task_values(baseline.get(metric))
        treatment_values = _extract_task_values(treatment.get(metric))
        if len(baseline_values) != len(treatment_values) or not baseline_values:
            continue
        ci = paired_bootstrap_delta_ci(
            baseline=baseline_values,
            treatment=treatment_values,
            iterations=iterations,
            seed=seed,
        )
        if ci["low"] > 0:
            return True
    return False


def _negative_delta_support(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    metrics: tuple[str, ...],
    iterations: int,
    seed: int,
    strict: bool = False,
) -> bool:
    found = False
    for metric in metrics:
        baseline_values = _extract_task_values(baseline.get(metric))
        treatment_values = _extract_task_values(treatment.get(metric))
        if len(baseline_values) != len(treatment_values) or not baseline_values:
            continue
        ci = paired_bootstrap_delta_ci(
            baseline=baseline_values,
            treatment=treatment_values,
            iterations=iterations,
            seed=seed,
        )
        if strict:
            if ci["high"] < 0:
                return True
        else:
            if ci["high"] < 0:
                return True
            found = found or (ci["delta_mean"] < 0)
    return found if not strict else False


def validate_h1(summary: Mapping[str, Any]) -> dict[str, Any]:
    return _decision(
        summary,
        rules={
            "c_bacc_below_nc_bacc": ("c_bacc", "<", "nc_bacc"),
            "q_psr_below_nc_psr": ("q_psr", "<", "nc_psr"),
        },
    )


def validate_h2(summary: Mapping[str, Any]) -> dict[str, Any]:
    return _decision(
        summary,
        rules={
            "sq_psr_below_q_psr": ("sq_psr", "<", "q_psr"),
            "ss_sq_psr_below_sq_psr": ("ss_sq_psr", "<", "sq_psr"),
        },
    )


def validate_h3(summary: Mapping[str, Any]) -> dict[str, Any]:
    return _decision(
        summary,
        rules={
            "ss_scr_above_scr": ("ss_scr", ">", "scr"),
        },
    )


check_h1 = validate_h1
check_h2 = validate_h2
check_h3 = validate_h3


__all__ = [
    "check_h1",
    "check_h2",
    "check_h3",
    "validate_h1",
    "validate_h2",
    "validate_h3",
]
