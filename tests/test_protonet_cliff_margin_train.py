from __future__ import annotations

import torch

from fsmol_cliff.protonet_cliff_margin_train import (
    build_cliff_margin_trainer_class,
    cliff_margin_penalty_torch,
    control_preservation_penalty_torch,
    identify_cliff_and_control_query_masks,
)


def test_identify_cliff_and_control_query_masks_separates_cliff_and_control_queries() -> None:
    support_labels = torch.tensor([1, 0], dtype=torch.long)
    query_labels = torch.tensor([1, 1], dtype=torch.long)
    support_fingerprints = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
    query_fingerprints = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
    support_log_r = torch.tensor([8.0, 6.0], dtype=torch.float32)
    query_log_r = torch.tensor([7.5, 6.3], dtype=torch.float32)

    cliff_mask, control_mask = identify_cliff_and_control_query_masks(
        support_labels=support_labels,
        query_labels=query_labels,
        support_fingerprints=support_fingerprints,
        query_fingerprints=query_fingerprints,
        support_log_r=support_log_r,
        query_log_r=query_log_r,
        tau=0.8,
        delta=1.0,
    )

    assert cliff_mask.tolist() == [True, False]
    assert control_mask.tolist() == [False, True]


def test_cliff_margin_penalty_torch_is_positive_for_margin_violation() -> None:
    penalty = cliff_margin_penalty_torch(
        support_embeddings=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        support_labels=torch.tensor([1, 0], dtype=torch.long),
        query_embeddings=torch.tensor([[0.49, 0.49]], dtype=torch.float32),
        query_labels=torch.tensor([1], dtype=torch.long),
        support_fingerprints=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        query_fingerprints=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        support_log_r=torch.tensor([8.0, 6.0], dtype=torch.float32),
        query_log_r=torch.tensor([7.4], dtype=torch.float32),
        margin_gamma=0.1,
        tau=0.8,
        delta=1.0,
    )

    assert penalty.item() > 0.0


def test_control_preservation_penalty_torch_is_zero_when_no_control_queries() -> None:
    penalty = control_preservation_penalty_torch(
        support_embeddings=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        support_labels=torch.tensor([1, 0], dtype=torch.long),
        query_embeddings=torch.tensor([[0.45, 0.45]], dtype=torch.float32),
        query_labels=torch.tensor([1], dtype=torch.long),
        support_fingerprints=torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        query_fingerprints=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        support_log_r=torch.tensor([8.0, 6.0], dtype=torch.float32),
        query_log_r=torch.tensor([7.4], dtype=torch.float32),
        tau=0.8,
        delta=1.0,
    )

    assert penalty.item() == 0.0


def test_build_cliff_margin_trainer_class_adds_margin_term_to_loss() -> None:
    class FakeBaseTrainer:
        def __init__(self, config):
            self.config = config
            self.use_fc = False

    trainer_cls = build_cliff_margin_trainer_class(FakeBaseTrainer)
    trainer = trainer_cls(
        config=type("Config", (), {"used_features": "ecfp", "distance_metric": "euclidean"})(),
        margin_gamma=0.1,
        lambda_cliff=0.3,
        control_preservation=False,
    )
    trainer._last_support_embeddings = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    trainer._last_query_embeddings = torch.tensor([[0.45, 0.45]], dtype=torch.float32)
    trainer._last_support_labels = torch.tensor([1, 0], dtype=torch.long)

    loss = trainer.compute_loss(
        torch.tensor([[0.2, 0.8]], dtype=torch.float32),
        torch.tensor([1], dtype=torch.long),
        type(
            "Batch",
            (),
            {
                "support_features": type("Features", (), {"fingerprints": torch.tensor([[1.0, 1.0], [1.0, 1.0]])})(),
                "query_features": type("Features", (), {"fingerprints": torch.tensor([[1.0, 1.0]])})(),
                "support_log_r": torch.tensor([8.0, 6.0], dtype=torch.float32),
                "query_log_r": torch.tensor([7.4], dtype=torch.float32),
            },
        )(),
    )

    assert loss.item() > 0.0
