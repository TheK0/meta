from __future__ import annotations

import random
from typing import Iterable, Sequence

from .constants import EpisodeConfig
from .episodes import compute_adversarial_injection_count, compute_m_avail, select_injected_pairs
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

    injected_anchor_ids = [pair.anchor_id for pair in injected_pairs]
    injected_neg_ids = [pair.neg_id for pair in injected_pairs]

    manifests: list[dict] = []
    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            episode_rng = random.Random(f"{task_id}:adversarial:{seed}:{episode_index}")
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
