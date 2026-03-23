from __future__ import annotations

from .constants import BenchmarkProfile, STRICT_PROFILE


def is_benchmark_eligible(stats: dict, *, profile: BenchmarkProfile = STRICT_PROFILE) -> bool:
    return (
        stats["num_valid_molecules"] >= profile.min_valid_molecules
        and stats["num_positive_molecules"] >= profile.min_positive_molecules
        and stats["num_negative_molecules"] >= profile.min_negative_molecules
        and stats["num_cliff_pairs"] >= profile.min_cliff_pairs
        and stats["num_anchor_molecules"] >= profile.min_anchor_molecules
        and stats["num_cliff_negatives"] >= profile.min_cliff_negatives
        and stats["num_noncliff_highsim_pairs"] >= profile.min_noncliff_pairs
    )


def is_adv_eligible(stats: dict, *, profile: BenchmarkProfile = STRICT_PROFILE) -> bool:
    return stats["m_avail"] >= profile.min_m_avail


def cliff_richness_score(stats: dict) -> float:
    cliff_density = stats["num_cliff_pairs"] / (
        stats["num_positive_molecules"] * stats["num_negative_molecules"]
    )
    anchor_coverage = stats["num_anchor_molecules"] / stats["num_positive_molecules"]
    return cliff_density * anchor_coverage


def rank_tasks_for_topk(tasks: list[dict], limit: int) -> list[dict]:
    return sorted(
        tasks,
        key=lambda task: (
            -cliff_richness_score(task),
            -task["num_cliff_pairs"],
            -task["median_sim"],
            task["assay_id"],
        ),
    )[:limit]
