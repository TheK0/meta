from __future__ import annotations

from fsmol_cliff.constants import EpisodeConfig
from fsmol_cliff.manifests import (
    build_anchor_coverage_first_adversarial_episode_manifests,
    build_adversarial_episode_manifests,
    build_paired_hardness_balanced_adversarial_episode_manifests,
    build_query_cluster_separation_by_anchor_neg_mix_adversarial_episode_manifests,
    build_query_cluster_separation_by_neg_diversity_adversarial_episode_manifests,
    build_query_targeted_adversarial_episode_manifests,
    build_same_scaffold_query_targeted_adversarial_episode_manifests,
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


def test_build_query_targeted_adversarial_episode_manifests_prefers_anchor_guided_support_negatives() -> None:
    manifests = build_query_targeted_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6", "hn1", "hn2", "hn3", "hn4"],
        cliff_pairs=[
            _pair("a1", "n1", sim=0.99),
            _pair("a1", "n2", sim=0.95),
            _pair("a2", "n2", sim=0.98),
            _pair("a2", "n3", sim=0.96),
            _pair("a3", "n3", sim=0.97),
            _pair("a4", "n4", sim=0.94),
        ],
        anchor_to_hardnegs={
            "a1": ["n1", "hn1", "hn2"],
            "a2": ["n2", "hn3", "hn4"],
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
    assert set(episode["support_pos_ids"]) == {"a1", "a2"}
    assert len(episode["support_neg_ids"]) == 2
    assert episode["support_neg_ids"] == ["hn2", "hn3"]
    assert set(episode["support_neg_ids"]).isdisjoint(set(episode["query_neg_ids"]))


def test_build_query_targeted_adversarial_episode_manifests_preserves_baseline_episode_skeleton() -> None:
    kwargs = {
        "task_id": "CHEMBL1",
        "positive_ids": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "negative_ids": ["n1", "n2", "n3", "n4", "n5", "n6", "hn1", "hn2", "hn3", "hn4"],
        "cliff_pairs": [
            _pair("a1", "n1", sim=0.99),
            _pair("a1", "n2", sim=0.95),
            _pair("a2", "n2", sim=0.98),
            _pair("a2", "n3", sim=0.96),
            _pair("a3", "n3", sim=0.97),
            _pair("a4", "n4", sim=0.94),
        ],
        "anchor_to_hardnegs": {
            "a1": ["n1", "hn1", "hn2"],
            "a2": ["n2", "hn3", "hn4"],
            "a3": ["n3"],
            "a4": ["n4"],
        },
        "episode_config": EpisodeConfig(support_per_class=2, query_per_class=4),
        "seeds": [0],
        "episodes_per_seed": 1,
    }

    baseline = build_adversarial_episode_manifests(**kwargs)[0]
    variant = build_query_targeted_adversarial_episode_manifests(**kwargs)[0]

    assert variant["support_pos_ids"] == baseline["support_pos_ids"]
    assert variant["query_pos_ids"] == baseline["query_pos_ids"]
    assert variant["query_neg_ids"] == baseline["query_neg_ids"]
    assert variant["injected_pairs"] == baseline["injected_pairs"]
    assert variant["support_neg_ids"] != baseline["support_neg_ids"]


def test_build_same_scaffold_query_targeted_episode_manifests_prefers_same_scaffold_pairs() -> None:
    manifests = build_same_scaffold_query_targeted_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6"],
        cliff_pairs=[
            PairRecord(
                assay_id="CHEMBL1",
                anchor_id="a1",
                neg_id="n1",
                sim=0.99,
                gap_abs=1.2,
                same_scaffold=True,
                pair_type="cliff",
            ),
            PairRecord(
                assay_id="CHEMBL1",
                anchor_id="a2",
                neg_id="n2",
                sim=0.98,
                gap_abs=1.2,
                same_scaffold=False,
                pair_type="cliff",
            ),
            PairRecord(
                assay_id="CHEMBL1",
                anchor_id="a3",
                neg_id="n3",
                sim=0.97,
                gap_abs=1.2,
                same_scaffold=True,
                pair_type="cliff",
            ),
            PairRecord(
                assay_id="CHEMBL1",
                anchor_id="a4",
                neg_id="n4",
                sim=0.96,
                gap_abs=1.2,
                same_scaffold=False,
                pair_type="cliff",
            ),
        ],
        anchor_to_hardnegs={
            "a1": ["n1"],
            "a2": ["n2"],
            "a3": ["n3"],
            "a4": ["n4"],
        },
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=1,
    )

    assert len(manifests) == 1
    episode = manifests[0]
    injected = {(pair["anchor_id"], pair["neg_id"]) for pair in episode["injected_pairs"]}
    assert injected == {("a1", "n1"), ("a3", "n3")}


def test_build_anchor_coverage_first_adversarial_episode_manifests_prefers_higher_coverage_anchors() -> None:
    manifests = build_anchor_coverage_first_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6"],
        cliff_pairs=[
            _pair("a1", "n1", sim=0.99),
            _pair("a2", "n1", sim=0.98),
            _pair("a2", "n2", sim=0.97),
            _pair("a2", "n3", sim=0.96),
            _pair("a3", "n2", sim=0.95),
            _pair("a4", "n4", sim=0.94),
        ],
        anchor_to_hardnegs={
            "a1": ["n1"],
            "a2": ["n1", "n2", "n3"],
            "a3": ["n2"],
            "a4": ["n4"],
        },
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=1,
    )

    assert len(manifests) == 1
    episode = manifests[0]
    injected = {(pair["anchor_id"], pair["neg_id"]) for pair in episode["injected_pairs"]}
    assert injected == {("a2", "n1"), ("a3", "n2")}


def test_build_paired_hardness_balanced_adversarial_episode_manifests_avoids_only_extreme_pairs() -> None:
    manifests = build_paired_hardness_balanced_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6"],
        cliff_pairs=[
            _pair("a1", "n1", sim=0.99, gap_abs=3.0),
            _pair("a1", "n2", sim=0.98, gap_abs=1.2),
            _pair("a2", "n3", sim=0.97, gap_abs=2.8),
            _pair("a2", "n4", sim=0.96, gap_abs=1.3),
            _pair("a3", "n5", sim=0.95, gap_abs=1.1),
            _pair("a4", "n6", sim=0.94, gap_abs=1.0),
        ],
        anchor_to_hardnegs={
            "a1": ["n1", "n2"],
            "a2": ["n3", "n4"],
            "a3": ["n5"],
            "a4": ["n6"],
        },
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=1,
    )

    assert len(manifests) == 1
    episode = manifests[0]
    injected = {(pair["anchor_id"], pair["neg_id"]) for pair in episode["injected_pairs"]}
    assert injected == {("a1", "n2"), ("a2", "n4")}


def test_build_query_cluster_separation_by_neg_diversity_adversarial_episode_manifests_avoids_hub_negatives() -> None:
    manifests = build_query_cluster_separation_by_neg_diversity_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6"],
        cliff_pairs=[
            _pair("a1", "n1", sim=0.99),
            _pair("a1", "n2", sim=0.98),
            _pair("a2", "n1", sim=0.97),
            _pair("a2", "n3", sim=0.96),
            _pair("a3", "n1", sim=0.95),
            _pair("a3", "n4", sim=0.94),
            _pair("a4", "n5", sim=0.93),
        ],
        anchor_to_hardnegs={
            "a1": ["n1", "n2"],
            "a2": ["n1", "n3"],
            "a3": ["n1", "n4"],
            "a4": ["n5"],
        },
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=1,
    )

    assert len(manifests) == 1
    episode = manifests[0]
    injected = {(pair["anchor_id"], pair["neg_id"]) for pair in episode["injected_pairs"]}
    assert injected == {("a1", "n2"), ("a2", "n3")}


def test_build_query_cluster_separation_by_anchor_neg_mix_adversarial_episode_manifests_interleaves_anchor_scales() -> None:
    manifests = build_query_cluster_separation_by_anchor_neg_mix_adversarial_episode_manifests(
        task_id="CHEMBL1",
        positive_ids=["a1", "a2", "a3", "a4", "a5", "a6"],
        negative_ids=["n1", "n2", "n3", "n4", "n5", "n6", "n7"],
        cliff_pairs=[
            _pair("a1", "n1", sim=0.99),
            _pair("a1", "n2", sim=0.98),
            _pair("a1", "n3", sim=0.97),
            _pair("a2", "n1", sim=0.96),
            _pair("a2", "n4", sim=0.95),
            _pair("a2", "n5", sim=0.94),
            _pair("a3", "n6", sim=0.93),
            _pair("a4", "n7", sim=0.92),
        ],
        anchor_to_hardnegs={
            "a1": ["n1", "n2", "n3"],
            "a2": ["n1", "n4", "n5"],
            "a3": ["n6"],
            "a4": ["n7"],
        },
        episode_config=EpisodeConfig(support_per_class=2, query_per_class=4),
        seeds=[0],
        episodes_per_seed=1,
    )

    assert len(manifests) == 1
    episode = manifests[0]
    injected = {(pair["anchor_id"], pair["neg_id"]) for pair in episode["injected_pairs"]}
    assert injected == {("a1", "n2"), ("a3", "n6")}
