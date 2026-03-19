from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _extract_mean(metric_summary: Any) -> float | None:
    if metric_summary is None:
        return None
    if isinstance(metric_summary, Mapping):
        metric_summary = metric_summary.get("mean")
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
