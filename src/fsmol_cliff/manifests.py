from __future__ import annotations

import random
from typing import Iterable, Sequence

from .constants import EpisodeConfig
from .episodes import (
    compute_adversarial_injection_count,
    compute_m_avail,
    select_injected_pairs,
    select_injected_pairs_with_anchor_priority,
)
from .models import PairRecord


def build_standard_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    manifests: list[dict] = []
    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            episode_rng = random.Random(f"{task_id}:standard:{seed}:{episode_index}")
            support_pos_ids, query_pos_ids = _sample_balanced_class_split(
                positive_ids,
                episode_config.support_per_class,
                episode_config.query_per_class,
                episode_rng,
            )
            support_neg_ids, query_neg_ids = _sample_balanced_class_split(
                negative_ids,
                episode_config.support_per_class,
                episode_config.query_per_class,
                episode_rng,
            )
            manifests.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "split_type": "standard",
                    "episode_id": episode_index,
                    "support_pos_ids": support_pos_ids,
                    "support_neg_ids": support_neg_ids,
                    "query_pos_ids": query_pos_ids,
                    "query_neg_ids": query_neg_ids,
                    "injected_pairs": [],
                }
            )
    return manifests


def build_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    anchor_ids = sorted({pair.anchor_id for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair.neg_id for pair in cliff_pairs})
    if not anchor_ids or not cliff_neg_ids:
        return []

    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pairs)
    injection_count = compute_adversarial_injection_count(
        query_neg_count=episode_config.query_per_class,
        support_pos_count=episode_config.support_per_class,
        m_avail=m_avail,
    )
    if injection_count < 2:
        return []

    injected_pairs = select_injected_pairs(
        support_pos_ids=anchor_ids,
        query_neg_ids=cliff_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        injection_count=injection_count,
    )
    if len(injected_pairs) != injection_count:
        return []

    return _build_adversarial_manifests_from_injected_pairs(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        injected_pairs=injected_pairs,
        rng_label="adversarial",
    )


def build_same_scaffold_query_targeted_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    anchor_ids = sorted({pair.anchor_id for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair.neg_id for pair in cliff_pairs})
    if not anchor_ids or not cliff_neg_ids:
        return []

    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pairs)
    injection_count = compute_adversarial_injection_count(
        query_neg_count=episode_config.query_per_class,
        support_pos_count=episode_config.support_per_class,
        m_avail=m_avail,
    )
    if injection_count < 2:
        return []

    same_scaffold_pairs = [pair for pair in cliff_pairs if pair.same_scaffold]
    if same_scaffold_pairs:
        same_scaffold_injected_pairs = select_injected_pairs(
            support_pos_ids=sorted({pair.anchor_id for pair in same_scaffold_pairs}),
            query_neg_ids=sorted({pair.neg_id for pair in same_scaffold_pairs}),
            cliff_pairs=same_scaffold_pairs,
            anchor_to_hardnegs=anchor_to_hardnegs,
            injection_count=injection_count,
        )
        if len(same_scaffold_injected_pairs) == injection_count:
            return _build_adversarial_manifests_from_injected_pairs(
                task_id=task_id,
                positive_ids=positive_ids,
                negative_ids=negative_ids,
                episode_config=episode_config,
                seeds=seeds,
                episodes_per_seed=episodes_per_seed,
                injected_pairs=same_scaffold_injected_pairs,
                rng_label="adversarial:same-scaffold-query-targeted",
            )

    return build_adversarial_episode_manifests(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
    )


def build_anchor_coverage_first_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    anchor_ids = sorted({pair.anchor_id for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair.neg_id for pair in cliff_pairs})
    if not anchor_ids or not cliff_neg_ids:
        return []

    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pairs)
    injection_count = compute_adversarial_injection_count(
        query_neg_count=episode_config.query_per_class,
        support_pos_count=episode_config.support_per_class,
        m_avail=m_avail,
    )
    if injection_count < 2:
        return []

    pair_lookup = {(pair.anchor_id, pair.neg_id) for pair in cliff_pairs}
    anchor_priority_order = sorted(
        anchor_ids,
        key=lambda anchor_id: (
            -sum(
                1
                for neg_id in anchor_to_hardnegs.get(anchor_id, ())
                if neg_id in set(cliff_neg_ids) and (anchor_id, neg_id) in pair_lookup
            ),
            anchor_id,
        ),
    )
    injected_pairs = select_injected_pairs_with_anchor_priority(
        support_pos_ids=anchor_ids,
        query_neg_ids=cliff_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        injection_count=injection_count,
        anchor_priority_order=anchor_priority_order,
    )
    if len(injected_pairs) != injection_count:
        return []

    return _build_adversarial_manifests_from_injected_pairs(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        injected_pairs=injected_pairs,
        rng_label="adversarial:anchor-coverage-first",
    )


def build_paired_hardness_balanced_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    anchor_ids = sorted({pair.anchor_id for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair.neg_id for pair in cliff_pairs})
    if not anchor_ids or not cliff_neg_ids:
        return []

    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pairs)
    injection_count = compute_adversarial_injection_count(
        query_neg_count=episode_config.query_per_class,
        support_pos_count=episode_config.support_per_class,
        m_avail=m_avail,
    )
    if injection_count < 2:
        return []

    injected_pairs = select_injected_pairs(
        support_pos_ids=anchor_ids,
        query_neg_ids=cliff_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs={
            anchor_id: [
                pair.neg_id
                for pair in sorted(
                    [pair for pair in cliff_pairs if pair.anchor_id == anchor_id],
                    key=lambda pair: (pair.gap_abs, -pair.sim, pair.neg_id),
                )
            ]
            for anchor_id in anchor_ids
        },
        injection_count=injection_count,
    )
    if len(injected_pairs) != injection_count:
        return []

    return _build_adversarial_manifests_from_injected_pairs(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        injected_pairs=injected_pairs,
        rng_label="adversarial:paired-hardness-balanced",
    )


def build_query_cluster_separation_by_neg_diversity_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    anchor_ids = sorted({pair.anchor_id for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair.neg_id for pair in cliff_pairs})
    if not anchor_ids or not cliff_neg_ids:
        return []

    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pairs)
    injection_count = compute_adversarial_injection_count(
        query_neg_count=episode_config.query_per_class,
        support_pos_count=episode_config.support_per_class,
        m_avail=m_avail,
    )
    if injection_count < 2:
        return []

    injected_pairs = select_injected_pairs(
        support_pos_ids=anchor_ids,
        query_neg_ids=cliff_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=_build_neg_diversity_ordered_anchor_to_hardnegs(
            anchor_ids=anchor_ids,
            cliff_pairs=cliff_pairs,
            anchor_to_hardnegs=anchor_to_hardnegs,
        ),
        injection_count=injection_count,
    )
    if len(injected_pairs) != injection_count:
        return []

    return _build_adversarial_manifests_from_injected_pairs(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        injected_pairs=injected_pairs,
        rng_label="adversarial:query-cluster-neg-diversity",
    )


def build_query_cluster_separation_by_anchor_neg_mix_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    anchor_ids = sorted({pair.anchor_id for pair in cliff_pairs})
    cliff_neg_ids = sorted({pair.neg_id for pair in cliff_pairs})
    if not anchor_ids or not cliff_neg_ids:
        return []

    m_avail = compute_m_avail(anchor_ids, cliff_neg_ids, cliff_pairs)
    injection_count = compute_adversarial_injection_count(
        query_neg_count=episode_config.query_per_class,
        support_pos_count=episode_config.support_per_class,
        m_avail=m_avail,
    )
    if injection_count < 2:
        return []

    anchor_to_hardnegs_ordered = _build_neg_diversity_ordered_anchor_to_hardnegs(
        anchor_ids=anchor_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    injected_pairs = select_injected_pairs_with_anchor_priority(
        support_pos_ids=anchor_ids,
        query_neg_ids=cliff_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs_ordered,
        injection_count=injection_count,
        anchor_priority_order=_build_anchor_mix_priority_order(
            anchor_ids=anchor_ids,
            ordered_anchor_to_hardnegs=anchor_to_hardnegs_ordered,
        ),
    )
    if len(injected_pairs) != injection_count:
        return []

    return _build_adversarial_manifests_from_injected_pairs(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        injected_pairs=injected_pairs,
        rng_label="adversarial:query-cluster-anchor-neg-mix",
    )


def _build_adversarial_manifests_from_injected_pairs(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
    injected_pairs: Sequence[PairRecord],
    rng_label: str,
) -> list[dict]:
    injected_anchor_ids = [pair.anchor_id for pair in injected_pairs]
    injected_neg_ids = [pair.neg_id for pair in injected_pairs]
    manifests: list[dict] = []
    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            episode_rng = random.Random(f"{task_id}:{rng_label}:{seed}:{episode_index}")
            support_pos_fill = _sample_without_replacement(
                [molecule_id for molecule_id in positive_ids if molecule_id not in set(injected_anchor_ids)],
                episode_config.support_per_class - len(injected_anchor_ids),
                episode_rng,
            )
            remaining_pos_after_support = [
                molecule_id
                for molecule_id in positive_ids
                if molecule_id not in set(injected_anchor_ids) and molecule_id not in set(support_pos_fill)
            ]
            query_pos_ids = _sample_without_replacement(
                remaining_pos_after_support,
                episode_config.query_per_class,
                episode_rng,
            )
            support_neg_ids = _sample_without_replacement(
                [molecule_id for molecule_id in negative_ids if molecule_id not in set(injected_neg_ids)],
                episode_config.support_per_class,
                episode_rng,
            )
            query_neg_fill = _sample_without_replacement(
                [
                    molecule_id
                    for molecule_id in negative_ids
                    if molecule_id not in set(injected_neg_ids) and molecule_id not in set(support_neg_ids)
                ],
                episode_config.query_per_class - len(injected_neg_ids),
                episode_rng,
            )
            support_pos_ids = list(injected_anchor_ids) + support_pos_fill
            query_neg_ids = list(injected_neg_ids) + query_neg_fill
            _validate_episode_sizes(
                support_pos_ids=support_pos_ids,
                support_neg_ids=support_neg_ids,
                query_pos_ids=query_pos_ids,
                query_neg_ids=query_neg_ids,
                episode_config=episode_config,
            )
            manifests.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "split_type": "adversarial",
                    "episode_id": episode_index,
                    "support_pos_ids": support_pos_ids,
                    "support_neg_ids": support_neg_ids,
                    "query_pos_ids": query_pos_ids,
                    "query_neg_ids": query_neg_ids,
                    "injected_pairs": [pair.to_dict() for pair in injected_pairs],
                }
            )
    return manifests


def _build_neg_diversity_ordered_anchor_to_hardnegs(
    *,
    anchor_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
) -> dict[str, list[str]]:
    neg_degree: dict[str, int] = {}
    for pair in cliff_pairs:
        neg_degree[pair.neg_id] = neg_degree.get(pair.neg_id, 0) + 1

    ordered: dict[str, list[str]] = {}
    for anchor_id in anchor_ids:
        original = list(anchor_to_hardnegs.get(anchor_id, ()))
        original_order = {neg_id: index for index, neg_id in enumerate(original)}
        available = [pair.neg_id for pair in cliff_pairs if pair.anchor_id == anchor_id]
        available_set = set(available)
        ordered[anchor_id] = sorted(
            [neg_id for neg_id in original if neg_id in available_set],
            key=lambda neg_id: (
                neg_degree.get(neg_id, 0),
                original_order.get(neg_id, len(original)),
                neg_id,
            ),
        )
    return ordered


def _build_anchor_mix_priority_order(
    *,
    anchor_ids: Sequence[str],
    ordered_anchor_to_hardnegs: dict[str, Sequence[str]],
) -> list[str]:
    high_first = sorted(
        anchor_ids,
        key=lambda anchor_id: (-len(ordered_anchor_to_hardnegs.get(anchor_id, ())), anchor_id),
    )
    low_first = sorted(
        anchor_ids,
        key=lambda anchor_id: (len(ordered_anchor_to_hardnegs.get(anchor_id, ())), anchor_id),
    )
    mixed: list[str] = []
    seen: set[str] = set()
    for index in range(max(len(high_first), len(low_first))):
        if index < len(high_first) and high_first[index] not in seen:
            mixed.append(high_first[index])
            seen.add(high_first[index])
        if index < len(low_first) and low_first[index] not in seen:
            mixed.append(low_first[index])
            seen.add(low_first[index])
    return mixed


def build_query_targeted_adversarial_episode_manifests(
    *,
    task_id: str,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    episode_config: EpisodeConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    baseline_manifests = build_adversarial_episode_manifests(
        task_id=task_id,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        episode_config=episode_config,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
    )
    if not baseline_manifests:
        return []

    manifests: list[dict] = []
    for manifest in baseline_manifests:
        support_anchor_ids = [pair["anchor_id"] for pair in manifest["injected_pairs"]]
        support_neg_rng = random.Random(
            f"{task_id}:adversarial:query-targeted:support-neg:{manifest['seed']}:{manifest['episode_id']}"
        )
        support_neg_ids = _build_query_targeted_support_neg_ids(
            support_anchor_ids=support_anchor_ids,
            negative_ids=negative_ids,
            query_neg_ids=manifest["query_neg_ids"],
            anchor_to_hardnegs=anchor_to_hardnegs,
            support_size=episode_config.support_per_class,
            episode_rng=support_neg_rng,
        )
        _validate_episode_sizes(
            support_pos_ids=manifest["support_pos_ids"],
            support_neg_ids=support_neg_ids,
            query_pos_ids=manifest["query_pos_ids"],
            query_neg_ids=manifest["query_neg_ids"],
            episode_config=episode_config,
        )
        manifests.append({**manifest, "support_neg_ids": support_neg_ids})
    return manifests


def _sample_balanced_class_split(
    molecule_ids: Sequence[str],
    support_size: int,
    query_size: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    shuffled = list(molecule_ids)
    rng.shuffle(shuffled)
    needed = support_size + query_size
    if len(shuffled) < needed:
        raise ValueError("Not enough molecules to sample a balanced episode.")
    selected = shuffled[:needed]
    return selected[:support_size], selected[support_size:]


def _sample_without_replacement(
    molecule_ids: Sequence[str],
    sample_size: int,
    rng: random.Random,
) -> list[str]:
    if sample_size == 0:
        return []
    pool = list(molecule_ids)
    rng.shuffle(pool)
    if len(pool) < sample_size:
        raise ValueError("Not enough molecules to fill episode slots.")
    return pool[:sample_size]


def _validate_episode_sizes(
    *,
    support_pos_ids: Sequence[str],
    support_neg_ids: Sequence[str],
    query_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    episode_config: EpisodeConfig,
) -> None:
    if len(support_pos_ids) != episode_config.support_per_class:
        raise ValueError("support_pos_ids size mismatch")
    if len(support_neg_ids) != episode_config.support_per_class:
        raise ValueError("support_neg_ids size mismatch")
    if len(query_pos_ids) != episode_config.query_per_class:
        raise ValueError("query_pos_ids size mismatch")
    if len(query_neg_ids) != episode_config.query_per_class:
        raise ValueError("query_neg_ids size mismatch")
    all_ids = [*support_pos_ids, *support_neg_ids, *query_pos_ids, *query_neg_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Episode manifest reuses molecules across slots.")


def _build_query_targeted_support_neg_ids(
    *,
    support_anchor_ids: Sequence[str],
    negative_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    anchor_to_hardnegs: dict[str, Sequence[str]],
    support_size: int,
    episode_rng: random.Random,
) -> list[str]:
    query_neg_set = set(query_neg_ids)
    chosen: list[str] = []
    seen = set(query_neg_ids)
    for anchor_id in support_anchor_ids:
        for neg_id in anchor_to_hardnegs.get(anchor_id, ()):
            if neg_id in seen:
                continue
            chosen.append(neg_id)
            seen.add(neg_id)
            break
        if len(chosen) >= support_size:
            return chosen[:support_size]

    fill_pool = [
        molecule_id
        for molecule_id in negative_ids
        if molecule_id not in seen and molecule_id not in query_neg_set
    ]
    episode_rng.shuffle(fill_pool)
    return [*chosen, *fill_pool[: max(0, support_size - len(chosen))]]
