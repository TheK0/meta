from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .assets import build_assay_assets
from .episodes import compute_m_avail
from .io import write_json
from .pipeline import load_task_records
from .release import assay_id_from_path, discover_task_files
from .models import PairRecord


@dataclass(frozen=True)
class AuditThresholds:
    min_valid_molecules: int = 50
    min_positive_molecules: int = 15
    min_negative_molecules: int = 15
    min_cliff_pairs: int = 25
    min_noncliff_pairs: int = 10
    min_anchor_molecules: int = 10
    min_cliff_negatives: int = 10
    min_m_avail: int = 2

    @property
    def min_highsim_discordant_pairs(self) -> int:
        return self.min_cliff_pairs + self.min_noncliff_pairs


_FUNNEL_STAGES = [
    ("raw_assay_present", lambda row, t: bool(row.get("raw_assay_present", True))),
    ("legal_samples", lambda row, t: row["num_valid_molecules"] >= t.min_valid_molecules),
    (
        "active_inactive_minimums",
        lambda row, t: (
            row["num_positive_molecules"] >= t.min_positive_molecules
            and row["num_negative_molecules"] >= t.min_negative_molecules
        ),
    ),
    (
        "highsim_discordant_support",
        lambda row, t: row["num_highsim_discordant_pairs"] >= t.min_highsim_discordant_pairs,
    ),
    ("c_t", lambda row, t: row["num_cliff_pairs"] >= t.min_cliff_pairs),
    ("d_t", lambda row, t: row["num_noncliff_highsim_pairs"] >= t.min_noncliff_pairs),
    ("a_t", lambda row, t: row["num_anchor_molecules"] >= t.min_anchor_molecules),
    ("n_t_cliff", lambda row, t: row["num_cliff_negatives"] >= t.min_cliff_negatives),
    ("m_avail", lambda row, t: row["m_avail"] >= t.min_m_avail),
]


def build_attrition_rows(
    summaries: Sequence[dict],
    *,
    thresholds: AuditThresholds,
) -> list[dict]:
    rows = []
    for summary in summaries:
        failure_stage = None
        benchmark_eligible = True
        for stage_name, predicate in _FUNNEL_STAGES[:-1]:
            if not predicate(summary, thresholds):
                failure_stage = stage_name
                benchmark_eligible = False
                break

        adversarial_eligible = False
        if benchmark_eligible:
            if not _FUNNEL_STAGES[-1][1](summary, thresholds):
                failure_stage = _FUNNEL_STAGES[-1][0]
                adversarial_eligible = False
            else:
                adversarial_eligible = True

        rows.append(
            {
                **summary,
                "failure_stage": failure_stage,
                "benchmark_eligible": benchmark_eligible,
                "adversarial_eligible": adversarial_eligible,
            }
        )
    return rows


def build_attrition_funnel(rows: Sequence[dict]) -> list[dict]:
    funnel = []
    remaining = list(rows)
    for stage_name, _ in _FUNNEL_STAGES:
        failed_here = [row for row in remaining if row["failure_stage"] == stage_name]
        passed_here = [row for row in remaining if row["failure_stage"] != stage_name]
        funnel.append(
            {
                "stage": stage_name,
                "passed_count": len(passed_here),
                "failed_count": len(failed_here),
            }
        )
        remaining = passed_here
    return funnel


def sweep_attrition_thresholds(
    summaries: Sequence[dict],
    *,
    taus: Sequence[float],
    deltas: Sequence[float],
    min_cliff_pairs: Sequence[int],
    min_noncliff_pairs: Sequence[int],
) -> list[dict]:
    results = []
    for tau in taus:
        for delta in deltas:
            subset = [summary for summary in summaries if summary["tau"] == tau and summary["delta"] == delta]
            for cliff_threshold in min_cliff_pairs:
                for noncliff_threshold in min_noncliff_pairs:
                    thresholds = AuditThresholds(
                        min_cliff_pairs=cliff_threshold,
                        min_noncliff_pairs=noncliff_threshold,
                    )
                    rows = build_attrition_rows(subset, thresholds=thresholds)
                    eligible = [row for row in rows if row["benchmark_eligible"]]
                    adversarial = [row for row in rows if row["adversarial_eligible"]]
                    results.append(
                        {
                            "tau": tau,
                            "delta": delta,
                            "min_cliff_pairs": cliff_threshold,
                            "min_noncliff_pairs": noncliff_threshold,
                            "eligible_assay_count": len(eligible),
                            "total_cliff_pairs": sum(row["num_cliff_pairs"] for row in eligible),
                            "total_anchors": sum(row["num_anchor_molecules"] for row in eligible),
                            "adversarial_eligible_assay_count": len(adversarial),
                            "same_scaffold_assay_count": sum(
                                row["num_same_scaffold_cliff_pairs"] > 0 for row in eligible
                            ),
                            "same_scaffold_assay_fraction": (
                                sum(row["num_same_scaffold_cliff_pairs"] > 0 for row in eligible) / len(eligible)
                                if eligible
                                else 0.0
                            ),
                            "same_scaffold_cliff_pair_count": sum(
                                row["num_same_scaffold_cliff_pairs"] for row in eligible
                            ),
                        }
                    )
    return results


def build_real_audit_summaries(
    *,
    data_dir: Path,
    taus: Sequence[float],
    deltas: Sequence[float],
    task_list_file: Path | None = None,
) -> list[dict]:
    summaries = []
    task_files = discover_task_files(data_dir, task_list_file=task_list_file)
    for task_file in task_files:
        assay_id = assay_id_from_path(task_file)
        records = load_task_records(task_file)
        for tau in taus:
            for delta in deltas:
                bundle = build_assay_assets(assay_id, records, tau=tau, delta=delta)
                cliff_pairs = bundle["pairs"]["cliff"]
                anchor_ids = sorted({pair["anchor_id"] for pair in cliff_pairs})
                cliff_neg_ids = sorted({pair["neg_id"] for pair in cliff_pairs})
                m_avail = (
                    compute_m_avail(anchor_ids, cliff_neg_ids, [PairRecord(**pair) for pair in cliff_pairs])
                    if cliff_pairs
                    else 0
                )
                summaries.append(
                    {
                        "assay_id": assay_id,
                        "raw_assay_present": True,
                        "tau": tau,
                        "delta": delta,
                        "num_valid_molecules": len(bundle["molecules"]),
                        "num_positive_molecules": len(bundle["actives"]),
                        "num_negative_molecules": len(bundle["inactives"]),
                        "num_highsim_discordant_pairs": len(bundle["pairs"]["highsim_discordant"]),
                        "num_cliff_pairs": len(bundle["pairs"]["cliff"]),
                        "num_noncliff_highsim_pairs": len(bundle["pairs"]["highsim_noncliff"]),
                        "num_anchor_molecules": len(anchor_ids),
                        "num_cliff_negatives": len(cliff_neg_ids),
                        "num_same_scaffold_cliff_pairs": len(bundle["pairs"]["same_scaffold_cliff"]),
                        "m_avail": m_avail,
                    }
                )
    return summaries


def write_attrition_audit(
    *,
    release_dir: Path,
    data_dir: Path,
    output_dir: Path,
    profile: str = "strict",
    taus: Sequence[float] = (0.8, 0.85),
    deltas: Sequence[float] = (0.5, 1.0),
    min_cliff_pairs: Sequence[int] = (10, 25),
    min_noncliff_pairs: Sequence[int] = (5, 10),
    task_list_file: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_summaries_path = _resolve_task_summaries_path(release_dir, profile=profile)
    strict_rows = pd.read_parquet(task_summaries_path).to_dict(orient="records")
    strict_summaries = [
        {
            **row,
            "raw_assay_present": True,
            "tau": row.get("tau", 0.85 if profile == "strict" else 0.8),
            "delta": 1.0,
            "num_highsim_discordant_pairs": row.get("num_cliff_pairs", 0) + row.get("num_noncliff_highsim_pairs", 0),
            "num_same_scaffold_cliff_pairs": row.get("num_same_scaffold_cliff_pairs", 0),
        }
        for row in strict_rows
    ]
    thresholds = AuditThresholds()
    attrition_rows = build_attrition_rows(strict_summaries, thresholds=thresholds)
    funnel = build_attrition_funnel(attrition_rows)

    sweep_summaries = build_real_audit_summaries(
        data_dir=data_dir,
        taus=taus,
        deltas=deltas,
        task_list_file=task_list_file,
    )
    threshold_sweep = sweep_attrition_thresholds(
        sweep_summaries,
        taus=taus,
        deltas=deltas,
        min_cliff_pairs=min_cliff_pairs,
        min_noncliff_pairs=min_noncliff_pairs,
    )

    pd.DataFrame(attrition_rows).to_parquet(output_dir / "attrition_by_assay.parquet", index=False)
    pd.DataFrame(threshold_sweep).to_parquet(output_dir / "threshold_sensitivity.parquet", index=False)
    summary = {
        "profile": profile,
        "raw_assays": len(attrition_rows),
        "eligible_assays": sum(row["benchmark_eligible"] for row in attrition_rows),
        "adversarial_eligible_assays": sum(row["adversarial_eligible"] for row in attrition_rows),
        "funnel": funnel,
    }
    write_json(output_dir / "attrition_summary.json", summary)
    return summary


def _resolve_task_summaries_path(release_dir: Path, *, profile: str) -> Path:
    profile_path = release_dir / f"task_summaries_{profile}.parquet"
    legacy_path = release_dir / "task_summaries.parquet"
    return profile_path if profile_path.exists() else legacy_path
