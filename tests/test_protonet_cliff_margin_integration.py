from __future__ import annotations

import torch


def test_protonet_cliff_margin_loss_can_be_added_on_top_of_base_loss() -> None:
    from fsmol_cliff.protonet_cliff_margin import build_cliff_margin_loss_bundle

    logits = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    labels = torch.tensor([1, 0], dtype=torch.long)
    support_embeddings = torch.tensor(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [1.0, 1.0],
            [1.1, 1.1],
        ],
        dtype=torch.float32,
    )
    support_labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    query_embeddings = torch.tensor([[0.3, 0.3], [0.95, 0.95]], dtype=torch.float32)

    bundle = build_cliff_margin_loss_bundle(
        logits=logits,
        labels=labels,
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        query_embeddings=query_embeddings,
        margin_gamma=0.1,
        lambda_cliff=0.3,
    )

    assert set(bundle) == {"label_loss", "cliff_margin_loss", "total_loss"}
    assert bundle["label_loss"] > 0
    assert bundle["cliff_margin_loss"] >= 0
    assert bundle["total_loss"] == bundle["label_loss"] + 0.3 * bundle["cliff_margin_loss"]
