from __future__ import annotations

from collections.abc import Mapping


def build_raw_score_bundle(
    *,
    raw_scores: Mapping[str, float],
    raw_margins: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    scores = {molecule_id: float(score) for molecule_id, score in raw_scores.items()}
    margins = (
        {molecule_id: float(margin) for molecule_id, margin in raw_margins.items()}
        if raw_margins is not None
        else {molecule_id: float(score) - 0.5 for molecule_id, score in scores.items()}
    )
    return {
        "raw_scores": scores,
        "raw_margins": margins,
    }
