from __future__ import annotations


def is_benchmark_eligible(stats: dict) -> bool:
    return (
        stats["num_valid_molecules"] >= 50
        and stats["num_positive_molecules"] >= 15
        and stats["num_negative_molecules"] >= 15
        and stats["num_cliff_pairs"] >= 25
        and stats["num_anchor_molecules"] >= 10
        and stats["num_cliff_negatives"] >= 10
        and stats["num_noncliff_highsim_pairs"] >= 10
    )


def is_adv_eligible(stats: dict) -> bool:
    return stats["m_avail"] >= 2


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
