from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

from .models import PairRecord


@dataclass(frozen=True)
class AdversarialEpisode:
    support_pos_ids: tuple[str, ...]
    support_neg_ids: tuple[str, ...]
    query_pos_ids: tuple[str, ...]
    query_neg_ids: tuple[str, ...]
    injected_pairs: tuple[PairRecord, ...]

    def to_dict(self) -> dict:
        return {
            "support_pos_ids": list(self.support_pos_ids),
            "support_neg_ids": list(self.support_neg_ids),
            "query_pos_ids": list(self.query_pos_ids),
            "query_neg_ids": list(self.query_neg_ids),
            "injected_pairs": [pair.to_dict() for pair in self.injected_pairs],
        }


def compute_m_avail(
    support_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
) -> int:
    adjacency = _build_adjacency(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=cliff_pairs,
    )
    return _maximum_matching_size(adjacency)


def compute_adversarial_injection_count(
    query_neg_count: int,
    support_pos_count: int,
    m_avail: int,
) -> int:
    return min(query_neg_count // 2, support_pos_count, m_avail)


def select_injected_pairs(
    support_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: Mapping[str, Sequence[str]],
    injection_count: int,
) -> list[PairRecord]:
    ordered_pairs = _filter_pairs_by_hardneg_order(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    return _select_injected_pairs(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=ordered_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        injection_count=injection_count,
    )


def select_injected_pairs_with_anchor_priority(
    support_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: Mapping[str, Sequence[str]],
    injection_count: int,
    anchor_priority_order: Sequence[str],
) -> list[PairRecord]:
    ordered_pairs = _filter_pairs_by_hardneg_order(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    anchor_order = [
        anchor_id
        for anchor_id in anchor_priority_order
        if anchor_id in set(support_pos_ids)
    ]
    for anchor_id in sorted(set(support_pos_ids)):
        if anchor_id not in anchor_order:
            anchor_order.append(anchor_id)
    return _select_injected_pairs(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=ordered_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        injection_count=injection_count,
        anchor_order=anchor_order,
    )


def build_adversarial_episode(
    support_pos_ids: Sequence[str],
    support_neg_ids: Sequence[str],
    query_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: Mapping[str, Sequence[str]],
) -> AdversarialEpisode | None:
    _validate_no_reuse(
        support_pos_ids=support_pos_ids,
        support_neg_ids=support_neg_ids,
        query_pos_ids=query_pos_ids,
        query_neg_ids=query_neg_ids,
    )

    ordered_pairs = _filter_pairs_by_hardneg_order(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=cliff_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
    )
    m_avail = compute_m_avail(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=ordered_pairs,
    )
    injection_count = compute_adversarial_injection_count(
        query_neg_count=len(query_neg_ids),
        support_pos_count=len(support_pos_ids),
        m_avail=m_avail,
    )
    if injection_count < 2:
        return None

    injected_pairs = select_injected_pairs(
        support_pos_ids=support_pos_ids,
        query_neg_ids=query_neg_ids,
        cliff_pairs=ordered_pairs,
        anchor_to_hardnegs=anchor_to_hardnegs,
        injection_count=injection_count,
    )
    if len(injected_pairs) != injection_count:
        raise RuntimeError("Failed to realize deterministic adversarial matching.")

    return AdversarialEpisode(
        support_pos_ids=tuple(support_pos_ids),
        support_neg_ids=tuple(support_neg_ids),
        query_pos_ids=tuple(query_pos_ids),
        query_neg_ids=tuple(query_neg_ids),
        injected_pairs=tuple(injected_pairs),
    )


def _select_injected_pairs(
    support_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: Mapping[str, Sequence[str]],
    injection_count: int,
    anchor_order: Sequence[str] | None = None,
) -> list[PairRecord]:
    support_anchor_order = (
        list(dict.fromkeys(anchor_order))
        if anchor_order is not None
        else sorted(dict.fromkeys(support_pos_ids))
    )
    query_neg_set = set(query_neg_ids)
    pair_lookup = _build_pair_lookup(cliff_pairs)
    adjacency = {
        anchor_id: [
            neg_id
            for neg_id in anchor_to_hardnegs.get(anchor_id, ())
            if neg_id in query_neg_set and (anchor_id, neg_id) in pair_lookup
        ]
        for anchor_id in support_anchor_order
    }

    @lru_cache(maxsize=65536)
    def remaining_capacity(start_index: int, used_negs: frozenset[str]) -> int:
        remaining_adjacency = {
            anchor_id: [neg_id for neg_id in adjacency[anchor_id] if neg_id not in used_negs]
            for anchor_id in support_anchor_order[start_index:]
        }
        return _maximum_matching_size(remaining_adjacency)

    # Try anchors in lexical order and each anchor's hard negatives in fixed order,
    # but only keep choices that still allow the target cardinality.
    def search(
        start_index: int,
        used_negs: frozenset[str],
        pairs_needed: int,
    ) -> list[PairRecord] | None:
        if pairs_needed == 0:
            return []
        if start_index >= len(support_anchor_order):
            return None
        if remaining_capacity(start_index, used_negs) < pairs_needed:
            return None

        anchor_id = support_anchor_order[start_index]
        for neg_id in adjacency[anchor_id]:
            if neg_id in used_negs:
                continue
            next_used_negs = used_negs | {neg_id}
            if remaining_capacity(start_index + 1, next_used_negs) < pairs_needed - 1:
                continue
            remainder = search(
                start_index=start_index + 1,
                used_negs=next_used_negs,
                pairs_needed=pairs_needed - 1,
            )
            if remainder is not None:
                return [pair_lookup[(anchor_id, neg_id)], *remainder]

        if remaining_capacity(start_index + 1, used_negs) < pairs_needed:
            return None
        return search(
            start_index=start_index + 1,
            used_negs=used_negs,
            pairs_needed=pairs_needed,
        )

    return search(0, frozenset(), injection_count) or []


def _filter_pairs_by_hardneg_order(
    support_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
    anchor_to_hardnegs: Mapping[str, Sequence[str]],
) -> list[PairRecord]:
    support_pos_set = set(support_pos_ids)
    query_neg_set = set(query_neg_ids)
    allowed_edges = {
        (anchor_id, neg_id)
        for anchor_id in support_pos_set
        for neg_id in anchor_to_hardnegs.get(anchor_id, ())
        if neg_id in query_neg_set
    }
    filtered_pairs = [
        pair
        for pair in cliff_pairs
        if pair.anchor_id in support_pos_set
        and pair.neg_id in query_neg_set
        and (pair.anchor_id, pair.neg_id) in allowed_edges
    ]
    return sorted(filtered_pairs, key=lambda pair: (pair.anchor_id, pair.sort_key()))


def _build_pair_lookup(cliff_pairs: Iterable[PairRecord]) -> dict[tuple[str, str], PairRecord]:
    pair_lookup: dict[tuple[str, str], PairRecord] = {}
    for pair in cliff_pairs:
        pair_lookup.setdefault((pair.anchor_id, pair.neg_id), pair)
    return pair_lookup


def _build_adjacency(
    support_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
    cliff_pairs: Sequence[PairRecord],
) -> dict[str, list[str]]:
    support_anchor_order = list(dict.fromkeys(sorted(support_pos_ids)))
    query_neg_set = set(query_neg_ids)
    adjacency: dict[str, list[str]] = {anchor_id: [] for anchor_id in support_anchor_order}
    seen_edges: set[tuple[str, str]] = set()
    for pair in sorted(cliff_pairs, key=lambda record: (record.anchor_id, record.sort_key())):
        edge = (pair.anchor_id, pair.neg_id)
        if (
            pair.anchor_id not in adjacency
            or pair.neg_id not in query_neg_set
            or edge in seen_edges
        ):
            continue
        adjacency[pair.anchor_id].append(pair.neg_id)
        seen_edges.add(edge)
    return adjacency


def _maximum_matching_size(adjacency: Mapping[str, Sequence[str]]) -> int:
    match_by_neg: dict[str, str] = {}

    def augment(anchor_id: str, seen_negs: set[str]) -> bool:
        for neg_id in adjacency.get(anchor_id, ()):
            if neg_id in seen_negs:
                continue
            seen_negs.add(neg_id)
            if neg_id not in match_by_neg or augment(match_by_neg[neg_id], seen_negs):
                match_by_neg[neg_id] = anchor_id
                return True
        return False

    size = 0
    for anchor_id in adjacency:
        if augment(anchor_id, set()):
            size += 1
    return size


def _validate_no_reuse(
    support_pos_ids: Sequence[str],
    support_neg_ids: Sequence[str],
    query_pos_ids: Sequence[str],
    query_neg_ids: Sequence[str],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for molecule_id in (
        *support_pos_ids,
        *support_neg_ids,
        *query_pos_ids,
        *query_neg_ids,
    ):
        if molecule_id in seen:
            duplicates.add(molecule_id)
        seen.add(molecule_id)
    if duplicates:
        raise ValueError(
            "Episode molecule IDs are reused across support/query splits: "
            + ", ".join(sorted(duplicates))
        )
