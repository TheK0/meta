from __future__ import annotations

from fsmol_cliff.task_selection import (
    cliff_richness_score,
    is_adv_eligible,
    is_benchmark_eligible,
    rank_tasks_for_topk,
)


def test_benchmark_eligibility_requires_protocol_minimum_counts() -> None:
    stats = {
        "num_valid_molecules": 50,
        "num_positive_molecules": 15,
        "num_negative_molecules": 15,
        "num_cliff_pairs": 25,
        "num_anchor_molecules": 10,
        "num_cliff_negatives": 10,
        "num_noncliff_highsim_pairs": 10,
    }

    assert is_benchmark_eligible(stats) is True
    assert is_benchmark_eligible({**stats, "num_cliff_pairs": 24}) is False


def test_adv_eligibility_requires_matching_capacity_of_two() -> None:
    assert is_adv_eligible({"m_avail": 2}) is True
    assert is_adv_eligible({"m_avail": 1}) is False


def test_cliff_richness_score_multiplies_density_and_anchor_coverage() -> None:
    stats = {
        "num_cliff_pairs": 30,
        "num_positive_molecules": 15,
        "num_negative_molecules": 20,
        "num_anchor_molecules": 10,
    }

    assert cliff_richness_score(stats) == (30 / (15 * 20)) * (10 / 15)


def test_rank_tasks_for_topk_uses_protocol_tie_break_order() -> None:
    tasks = [
        {
            "assay_id": "CHEMBL_B",
            "num_cliff_pairs": 40,
            "num_positive_molecules": 20,
            "num_negative_molecules": 20,
            "num_anchor_molecules": 10,
            "median_sim": 0.91,
        },
        {
            "assay_id": "CHEMBL_A",
            "num_cliff_pairs": 40,
            "num_positive_molecules": 20,
            "num_negative_molecules": 20,
            "num_anchor_molecules": 10,
            "median_sim": 0.91,
        },
        {
            "assay_id": "CHEMBL_C",
            "num_cliff_pairs": 30,
            "num_positive_molecules": 20,
            "num_negative_molecules": 20,
            "num_anchor_molecules": 10,
            "median_sim": 0.99,
        },
    ]

    ranked = rank_tasks_for_topk(tasks, limit=2)
    assert [item["assay_id"] for item in ranked] == ["CHEMBL_A", "CHEMBL_B"]
