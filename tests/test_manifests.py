from __future__ import annotations

from fsmol_cliff.constants import EpisodeConfig
from fsmol_cliff.manifests import (
    build_adversarial_episode_manifests,
    build_standard_episode_manifests,
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


def test_build_standard_episode_manifests_is_deterministic_and_balanced() -> None:
    manifests = build_standard_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["p1", "p2", "p3", "p4", "p5", "p6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6"],
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=2),
        seeds=[0],
        episodes_per_seed=2,
    )

    assert len(manifests) == 2
    first = manifests[0]
    assert first["split_type"] == "standard"
    assert len(first["support_pos_ids"]) == 2
    assert len(first["support_neg_ids"]) == 2
    assert len(first["query_pos_ids"]) == 2
    assert len(first["query_neg_ids"]) == 2
    assert set(first["support_pos_ids"]).isdisjoint(first["query_pos_ids"])
    assert set(first["support_neg_ids"]).isdisjoint(first["query_neg_ids"])
    assert build_standard_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["p1", "p2", "p3", "p4", "p5", "p6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6"],
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=2),
        seeds=[0],
        episodes_per_seed=2,
    ) == manifests


def test_build_adversarial_episode_manifests_injects_pairs_and_respects_sizes() -> None:
    manifests = build_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"],
        cliff_pairs=[
            _pair("a1", "n1", sim=0.99),
            _pair("a1", "n2", sim=0.95),
            _pair("a2", "n2", sim=0.98),
            _pair("a2", "n3", sim=0.96),
            _pair("a3", "n3", sim=0.97),
            _pair("a4", "n4", sim=0.94),
        ],
        anchor_to_hardnegs={
            "a1": ["n1", "n2"],
            "a2": ["n2", "n3"],
            "a3": ["n3"],
            "a4": ["n4"],
        },
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=1,
    )

    assert len(manifests) == 1
    episode = manifests[0]
    assert episode["split_type"] == "adversarial"
    assert len(episode["support_pos_ids"]) == 2
    assert len(episode["support_neg_ids"]) == 2
    assert len(episode["query_pos_ids"]) == 4
    assert len(episode["query_neg_ids"]) == 4
    assert len(episode["injected_pairs"]) == 2
    injected = {(pair["anchor_id"], pair["neg_id"]) for pair in episode["injected_pairs"]}
    assert injected == {("a1", "n1"), ("a2", "n2")}
    assert {"a1", "a2"} == set(episode["support_pos_ids"])
    assert {"n1", "n2"}.issubset(set(episode["query_neg_ids"]))


def test_build_adversarial_episode_manifests_skips_tasks_without_minimum_matching() -> None:
    manifests = build_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4"],
        negative_ids=["n1", "n2", "n3", "n4"],
        cliff_pairs=[_pair("a1", "n1")],
        anchor_to_hardnegs={"a1": ["n1"]},
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=2,
    )

    assert manifests == []
