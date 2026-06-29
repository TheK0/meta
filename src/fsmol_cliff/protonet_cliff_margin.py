from __future__ import annotations

import torch

from .training_losses.cliff_margin import cliff_margin_loss


def build_cliff_margin_loss_bundle(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    margin_gamma: float,
    lambda_cliff: float,
) -> dict[str, float]:
    label_loss = torch.nn.functional.cross_entropy(logits, labels.long())
    class_prototypes = _compute_class_prototypes(support_embeddings, support_labels)
    positive_rows = []
    negative_rows = []
    for query_embedding, label in zip(query_embeddings, labels, strict=True):
        distance_to_positive = torch.sum((query_embedding - class_prototypes[1]) ** 2).item()
        distance_to_negative = torch.sum((query_embedding - class_prototypes[0]) ** 2).item()
        row = {
            "distance_to_positive": distance_to_positive,
            "distance_to_negative": distance_to_negative,
        }
        if int(label) == 1:
            positive_rows.append(row)
        else:
            negative_rows.append(row)
    margin_loss = cliff_margin_loss(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        margin=margin_gamma,
    )
    return {
        "label_loss": float(label_loss.item()),
        "cliff_margin_loss": float(margin_loss),
        "total_loss": float(label_loss.item() + lambda_cliff * margin_loss),
    }


def _compute_class_prototypes(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
) -> dict[int, torch.Tensor]:
    prototypes: dict[int, torch.Tensor] = {}
    for label in torch.unique(support_labels):
        mask = support_labels == label
        prototypes[int(label.item())] = support_embeddings[mask].mean(dim=0)
    return prototypes
