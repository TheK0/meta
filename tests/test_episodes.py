from __future__ import annotations

import pytest

from fsmol_cliff.episodes import (
    build_adversarial_episode,
    compute_adversarial_injection_count,
    compute_m_avail,
)
from fsmol_cliff.models import PairRecord


def _pair(anchor_id: str, neg_id: str, *, sim: float = 0.9, gap_abs: float = 1.2) -> PairRecord:
    return PairRecord(
        assay_id="CHEMBL1",
        anchor_id=anchor_id,
        neg_id=neg_id,
        sim=sim,
        gap_abs=gap_abs,
        same_scaffold=False,
        pair_type="cliff",
    )


def test_compute_m_avail_finds_maximum_matching_size_on_nontrivial_graph() -> None:
    cliff_pairs = [
        _pair("a1", "n1"),
        _pair("a1", "n2"),
        _pair("a2", "n1"),
        _pair("a3", "n2"),
        _pair("a3", "n3"),
        _pair("a4", "n3"),
    ]

    m_avail = compute_m_avail(
        support_pos_ids=["a1", "a2", "a3", "a4"],
        query_neg_ids=["n1", "n2", "n3", "n4"],
        cliff_pairs=cliff_pairs,
    )

    assert m_avail == 3


def test_compute_adversarial_injection_count_caps_by_ratio_support_and_matching() -> None:
    assert compute_adversarial_injection_count(
        query_neg_count=5,
        support_pos_count=4,
        m_avail=3,
    ) == 2


def test_build_adversarial_episode_selects_injected_pairs_deterministically() -> None:
    cliff_pairs = [
        _pair("a1", "n1", sim=0.98),
        _pair("a1", "n2", sim=0.97),
        _pair("a2", "n1", sim=0.96),
        _pair("a2", "n3", sim=0.95),
        _pair("a3", "n2", sim=0.94),
        _pair("a3", "n3", sim=0.93),
    ]
    anchor_to_hardnegs = {
        "a1": ["n1", "n2"],
        "a2": ["n1", "n3"],
        "a3": ["n2", "n3"],
    }

    episode = build_adversarial_episode(
        support_pos_ids=["a1", "a2", "a3"],
        support_neg_ids=["sneg1", "sneg2", "sneg3"],
        query_pos_ids=["qpos1", "qpos2"],
        query_neg_ids=["n1", "n2", "n3", "n4"],
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )

    assert episode is not None
    assert episode.to_dict()["injected_pairs"] == [
        _pair("a1", "n1", sim=0.98).to_dict(),
        _pair("a2", "n3", sim=0.95).to_dict(),
    ]


def test_build_adversarial_episode_excludes_tasks_with_fewer_than_two_injections() -> None:
    episode = build_adversarial_episode(
        support_pos_ids=["a1", "a2"],
        support_neg_ids=["sneg1", "sneg2"],
        query_pos_ids=["qpos1", "qpos2"],
        query_neg_ids=["n1", "n2", "n3", "n4"],
        cliff_pairs=[_pair("a1", "n1")],
        anchor_to_hardnegs={"a1": ["n1"], "a2": []},
    )

    assert episode is None


def test_build_adversarial_episode_rejects_molecule_reuse_across_episode_splits() -> None:
    with pytest.raises(ValueError, match="reused"):
        build_adversarial_episode(
            support_pos_ids=["a1", "a2"],
            support_neg_ids=["shared-neg", "sneg2"],
            query_pos_ids=["qpos1", "qpos2"],
            query_neg_ids=["shared-neg", "n2", "n3", "n4"],
            cliff_pairs=[_pair("a1", "n2"), _pair("a2", "n3")],
            anchor_to_hardnegs={"a1": ["n2"], "a2": ["n3"]},
        )
