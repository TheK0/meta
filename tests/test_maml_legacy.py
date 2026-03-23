from __future__ import annotations

from fsmol_cliff.maml_legacy import split_support_for_validation


def test_split_support_for_validation_holds_out_one_per_class() -> None:
    train_ids, valid_ids = split_support_for_validation(
        support_pos_ids=["p1", "p2", "p3"],
        support_neg_ids=["n1", "n2", "n3"],
        holdout_per_class=1,
    )

    assert train_ids == ["p1", "p2", "n1", "n2"]
    assert valid_ids == ["p3", "n3"]


def test_split_support_for_validation_requires_enough_support_examples() -> None:
    try:
        split_support_for_validation(
            support_pos_ids=["p1"],
            support_neg_ids=["n1"],
            holdout_per_class=1,
        )
    except ValueError as exc:
        assert "Not enough support samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError for undersized support sets")
