from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrototypicalNetworkTrainerConfig:
    used_features: str = "ecfp"
    distance_metric: str = "euclidean"


class PrototypicalNetworkTrainer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, batch):
        num_query = int(getattr(batch, "num_query_samples", 1))
        return torch.zeros((num_query, 2), dtype=torch.float32)


def validate_by_finetuning_on_tasks(*args, **kwargs):
    return {}
