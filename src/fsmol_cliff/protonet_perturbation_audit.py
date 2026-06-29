from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import random
from pathlib import Path

from .models import PairRecord


def support_subset_dropout_episode(
    episode: Mapping[str, object],
    *,
    drop_per_class: int,
) -> dict:
    support_pos_ids = list(episode["support_pos_ids"])
    support_neg_ids = list(episode["support_neg_ids"])
    if len(support_pos_ids) <= drop_per_class or len(support_neg_ids) <= drop_per_class:
        raise ValueError("drop_per_class leaves an empty support class.")

    return {
        **episode,
        "support_pos_ids": support_pos_ids[:-drop_per_class],
        "support_neg_ids": support_neg_ids[:-drop_per_class],
    }


def summarize_query_score_variance(
    *,
    baseline_scores: Mapping[str, float],
    perturbed_score_runs: Sequence[Mapping[str, float]],
    cliff_query_ids: Sequence[str],
    control_query_ids: Sequence[str],
    same_scaffold_cliff_query_ids: Sequence[str],
) -> dict:
    per_query_variance: dict[str, float] = {}
    for molecule_id, baseline_score in baseline_scores.items():
        values = [float(baseline_score)]
        values.extend(float(run[molecule_id]) for run in perturbed_score_runs if molecule_id in run)
        mean_value = sum(values) / len(values)
        per_query_variance[molecule_id] = sum((value - mean_value) ** 2 for value in values) / len(values)

    return {
        "per_query_variance": per_query_variance,
        "cliff_variance_mean": _mean_for_ids(per_query_variance, cliff_query_ids),
        "control_variance_mean": _mean_for_ids(per_query_variance, control_query_ids),
        "same_scaffold_cliff_variance_mean": _mean_for_ids(per_query_variance, same_scaffold_cliff_query_ids),
        "cliff_control_variance_gap": _mean_for_ids(per_query_variance, cliff_query_ids) - _mean_for_ids(per_query_variance, control_query_ids),
        "same_scaffold_cliff_control_variance_gap": _mean_for_ids(per_query_variance, same_scaffold_cliff_query_ids) - _mean_for_ids(per_query_variance, control_query_ids),
    }


def derive_query_id_slices(
    *,
    cliff_pairs: Sequence[Mapping],
    noncliff_pairs: Sequence[Mapping],
) -> dict[str, list[str]]:
    cliff_ids = _pair_query_ids(PairRecord(**pair) for pair in cliff_pairs)
    control_ids = _pair_query_ids(PairRecord(**pair) for pair in noncliff_pairs)
    same_scaffold_cliff_ids = _pair_query_ids(
        PairRecord(**pair) for pair in cliff_pairs if pair.get("same_scaffold")
    )
    return {
        "cliff_query_ids": cliff_ids,
        "control_query_ids": control_ids,
        "same_scaffold_cliff_query_ids": same_scaffold_cliff_ids,
    }


def _pair_query_ids(pairs: Sequence[PairRecord]) -> list[str]:
    seen = set()
    ordered = []
    for pair in pairs:
        for molecule_id in (pair.anchor_id, pair.neg_id):
            if molecule_id not in seen:
                seen.add(molecule_id)
                ordered.append(molecule_id)
    return ordered


def _mean_for_ids(values: Mapping[str, float], ids: Sequence[str]) -> float:
    selected = [values[molecule_id] for molecule_id in ids if molecule_id in values]
    return 0.0 if not selected else sum(selected) / len(selected)


def summarize_perturbation_report(
    *,
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
    profile: str,
    split_type: str,
    seeds: Sequence[int],
    episodes_per_task: int,
    dropout_strengths: Sequence[int],
    views_per_strength: int,
) -> dict:
    report = {
        "profile": profile,
        "split_type": split_type,
        "seeds": list(seeds),
        "episodes_per_task": episodes_per_task,
        "dropout_strengths": list(dropout_strengths),
        "views_per_strength": views_per_strength,
        "episodes_analyzed": len(rows),
        "tasks_analyzed": sorted({str(row["task_id"]) for row in rows}),
        "cliff_control_variance_gap_mean": _mean_for_ids(
            {str(index): float(row["cliff_control_variance_gap"]) for index, row in enumerate(rows)},
            [str(index) for index in range(len(rows))],
        ),
        "same_scaffold_cliff_control_variance_gap_mean": _mean_for_ids(
            {str(index): float(row["same_scaffold_cliff_control_variance_gap"]) for index, row in enumerate(rows)},
            [str(index) for index in range(len(rows))],
        ),
        "episode_rows": list(rows),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
