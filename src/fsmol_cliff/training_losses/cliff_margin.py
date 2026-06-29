from __future__ import annotations

from collections.abc import Iterable, Mapping


def cliff_margin_loss(
    *,
    positive_rows: Iterable[Mapping[str, float]],
    negative_rows: Iterable[Mapping[str, float]],
    margin: float,
) -> float:
    losses = []
    for row in positive_rows:
        distance_to_positive = float(row["distance_to_positive"])
        distance_to_negative = float(row["distance_to_negative"])
        losses.append(max(0.0, distance_to_positive + margin - distance_to_negative))
    for row in negative_rows:
        distance_to_negative = float(row["distance_to_negative"])
        distance_to_positive = float(row["distance_to_positive"])
        losses.append(max(0.0, distance_to_negative + margin - distance_to_positive))
    return sum(losses) / len(losses) if losses else 0.0


def control_preservation_penalty(
    *,
    baseline_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
    control_pairs: Iterable[tuple[str, str]],
) -> float:
    penalties = []
    for anchor_id, neg_id in control_pairs:
        baseline_gap = float(baseline_scores[anchor_id]) - float(baseline_scores[neg_id])
        candidate_gap = float(candidate_scores[anchor_id]) - float(candidate_scores[neg_id])
        penalties.append(max(0.0, baseline_gap - candidate_gap))
    return sum(penalties) / len(penalties) if penalties else 0.0
