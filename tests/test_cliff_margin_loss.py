from __future__ import annotations

import pytest

from fsmol_cliff.training_losses.cliff_margin import cliff_margin_loss, control_preservation_penalty


def test_cliff_margin_loss_is_zero_when_margin_is_satisfied() -> None:
    loss = cliff_margin_loss(
        positive_rows=[
            {
                "distance_to_positive": 0.2,
                "distance_to_negative": 0.5,
            }
        ],
        negative_rows=[
            {
                "distance_to_positive": 0.6,
                "distance_to_negative": 0.2,
            }
        ],
        margin=0.1,
    )

    assert loss == 0.0


def test_cliff_margin_loss_is_positive_when_margin_is_violated() -> None:
    loss = cliff_margin_loss(
        positive_rows=[
            {
                "distance_to_positive": 0.3,
                "distance_to_negative": 0.35,
            }
        ],
        negative_rows=[
            {
                "distance_to_positive": 0.45,
                "distance_to_negative": 0.4,
            }
        ],
        margin=0.1,
    )

    assert loss == pytest.approx((0.05 + 0.05) / 2)


def test_control_preservation_penalty_is_zero_when_gap_is_preserved() -> None:
    penalty = control_preservation_penalty(
        baseline_scores={"a": 0.7, "n": 0.3},
        candidate_scores={"a": 0.75, "n": 0.35},
        control_pairs=[("a", "n")],
    )

    assert penalty == 0.0


def test_control_preservation_penalty_is_positive_when_gap_shrinks_past_baseline() -> None:
    penalty = control_preservation_penalty(
        baseline_scores={"a": 0.7, "n": 0.3},
        candidate_scores={"a": 0.55, "n": 0.5},
        control_pairs=[("a", "n")],
    )

    assert penalty == pytest.approx(0.35)
