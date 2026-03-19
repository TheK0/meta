from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .models import PairRecord


MetricValue = float | None


def _valid_ids(
    labels: Mapping[str, int],
    values: Mapping[str, float | int],
    query_ids: Iterable[str] | None = None,
) -> list[str]:
    ids = labels.keys() if query_ids is None else query_ids
    return [molecule_id for molecule_id in ids if molecule_id in labels and molecule_id in values]


def _filtered_pairs(
    pairs: Iterable[PairRecord],
    *,
    same_scaffold: bool | None = None,
) -> list[PairRecord]:
    selected = []
    for pair in pairs:
        if same_scaffold is not None and pair.same_scaffold is not same_scaffold:
            continue
        selected.append(pair)
    return selected


def _safe_mean(values: Iterable[float]) -> MetricValue:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def rank_1(
    manifest: Any,
    labels: Mapping[str, int],
    scores: Mapping[str, float],
    query_ids: Iterable[str] | None = None,
) -> MetricValue:
    _ = manifest
    valid_ids = _valid_ids(labels, scores, query_ids)
    if not valid_ids:
        return None

    top_score = max(scores[molecule_id] for molecule_id in valid_ids)
    top_labels = {labels[molecule_id] for molecule_id in valid_ids if scores[molecule_id] == top_score}

    if top_labels == {1}:
        return 1.0
    if top_labels == {0}:
        return 0.0
    return 0.5


def balanced_accuracy_for_subset(
    manifest: Any,
    labels: Mapping[str, int],
    predictions: Mapping[str, int],
    query_ids: Iterable[str],
) -> MetricValue:
    _ = manifest
    valid_ids = _valid_ids(labels, predictions, query_ids)
    if not valid_ids:
        return None

    positives = [molecule_id for molecule_id in valid_ids if labels[molecule_id] == 1]
    negatives = [molecule_id for molecule_id in valid_ids if labels[molecule_id] == 0]
    if not positives or not negatives:
        return None

    true_positive_rate = sum(predictions[molecule_id] == 1 for molecule_id in positives) / len(positives)
    true_negative_rate = sum(predictions[molecule_id] == 0 for molecule_id in negatives) / len(negatives)
    return (true_positive_rate + true_negative_rate) / 2.0


def c_bacc(
    manifest: Any,
    labels: Mapping[str, int],
    predictions: Mapping[str, int],
    query_ids: Iterable[str],
) -> MetricValue:
    return balanced_accuracy_for_subset(manifest, labels, predictions, query_ids)


def nc_bacc(
    manifest: Any,
    labels: Mapping[str, int],
    predictions: Mapping[str, int],
    query_ids: Iterable[str],
) -> MetricValue:
    return balanced_accuracy_for_subset(manifest, labels, predictions, query_ids)


def cliff_balanced_accuracy(
    manifest: Any,
    labels: Mapping[str, int],
    predictions: Mapping[str, int],
    query_ids: Iterable[str],
) -> MetricValue:
    return c_bacc(manifest, labels, predictions, query_ids)


def noncliff_balanced_accuracy(
    manifest: Any,
    labels: Mapping[str, int],
    predictions: Mapping[str, int],
    query_ids: Iterable[str],
) -> MetricValue:
    return nc_bacc(manifest, labels, predictions, query_ids)


def _pair_score(
    pair: PairRecord,
    scores: Mapping[str, float],
) -> MetricValue:
    if pair.anchor_id not in scores or pair.neg_id not in scores:
        return None
    if pair.anchor_label == pair.neg_label:
        return None

    anchor_score = scores[pair.anchor_id]
    neg_score = scores[pair.neg_id]
    if math.isclose(anchor_score, neg_score):
        return 0.5

    expected_direction = 1 if pair.anchor_label > pair.neg_label else -1
    observed_direction = 1 if anchor_score > neg_score else -1
    return 1.0 if observed_direction == expected_direction else 0.0


def pair_success_rate(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
    *,
    same_scaffold: bool | None = None,
) -> MetricValue:
    _ = manifest
    pair_scores = [
        score
        for pair in _filtered_pairs(pairs, same_scaffold=same_scaffold)
        if (score := _pair_score(pair, scores)) is not None
    ]
    return _safe_mean(pair_scores)


def q_psr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
) -> MetricValue:
    return pair_success_rate(manifest, pairs, scores)


def nc_psr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
) -> MetricValue:
    return pair_success_rate(manifest, pairs, scores)


def sq_psr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
) -> MetricValue:
    return pair_success_rate(manifest, pairs, scores)


def ss_q_psr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
) -> MetricValue:
    return pair_success_rate(manifest, pairs, scores, same_scaffold=True)


def ss_nc_psr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
) -> MetricValue:
    return pair_success_rate(manifest, pairs, scores, same_scaffold=True)


def ss_sq_psr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    scores: Mapping[str, float],
) -> MetricValue:
    return pair_success_rate(manifest, pairs, scores, same_scaffold=True)


same_scaffold_q_psr = ss_q_psr
same_scaffold_nc_psr = ss_nc_psr
same_scaffold_sq_psr = ss_sq_psr


def collapse_rate(
    manifest: Any,
    pairs: Iterable[PairRecord],
    predictions: Mapping[str, int],
    *,
    same_scaffold: bool | None = None,
) -> MetricValue:
    _ = manifest
    collapsed = []
    for pair in _filtered_pairs(pairs, same_scaffold=same_scaffold):
        if pair.anchor_label == pair.neg_label:
            continue
        if pair.anchor_id not in predictions or pair.neg_id not in predictions:
            continue
        collapsed.append(1.0 if predictions[pair.anchor_id] == predictions[pair.neg_id] else 0.0)
    return _safe_mean(collapsed)


def scr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    predictions: Mapping[str, int],
) -> MetricValue:
    return collapse_rate(manifest, pairs, predictions)


def ss_scr(
    manifest: Any,
    pairs: Iterable[PairRecord],
    predictions: Mapping[str, int],
) -> MetricValue:
    return collapse_rate(manifest, pairs, predictions, same_scaffold=True)


same_scaffold_scr = ss_scr


__all__ = [
    "MetricValue",
    "balanced_accuracy_for_subset",
    "c_bacc",
    "cliff_balanced_accuracy",
    "collapse_rate",
    "nc_bacc",
    "nc_psr",
    "noncliff_balanced_accuracy",
    "pair_success_rate",
    "q_psr",
    "rank_1",
    "same_scaffold_nc_psr",
    "same_scaffold_q_psr",
    "same_scaffold_scr",
    "same_scaffold_sq_psr",
    "scr",
    "sq_psr",
    "ss_nc_psr",
    "ss_q_psr",
    "ss_scr",
    "ss_sq_psr",
]
