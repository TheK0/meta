from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .release_artifacts import build_paired_model_comparison_rows

DEFAULT_PROTOCOL_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("standard", "c_bacc"),
    ("standard", "scr"),
    ("adversarial", "c_bacc"),
    ("adversarial", "sq_psr"),
    ("adversarial", "scr"),
    ("adversarial", "ss_scr"),
    ("adversarial", "nc_bacc"),
    ("adversarial", "nc_psr"),
)


def write_protocol_comparison_json(
    *,
    output_path: Path,
    model_to_path: Mapping[str, Path],
    comparisons: Sequence[tuple[str, str]],
    profile: str,
    result_tier: str,
    metrics: Sequence[tuple[str, str]] = DEFAULT_PROTOCOL_COMPARISON_METRICS,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
) -> list[dict]:
    rows = build_paired_model_comparison_rows(
        model_to_task_result_rows={
            model_name: pd.read_parquet(path).to_dict(orient="records")
            for model_name, path in model_to_path.items()
        },
        profile=profile,
        result_tier=result_tier,
        comparisons=comparisons,
        metrics=metrics,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, sep, raw_path = value.partition("=")
        if not sep or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        parsed[name] = Path(raw_path)
    return parsed


def parse_comparisons(values: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        baseline, sep, treatment = value.partition(":")
        if not sep or not baseline or not treatment:
            raise ValueError(f"Expected BASELINE:TREATMENT, got: {value}")
        parsed.append((baseline, treatment))
    return parsed
