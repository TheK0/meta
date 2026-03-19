from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeConfig:
    n_way: int = 2
    support_per_class: int = 16
    query_per_class: int = 16
    class_balance: str = "balanced"

    def to_dict(self) -> dict:
        return {
            "n_way": self.n_way,
            "support_per_class": self.support_per_class,
            "query_per_class": self.query_per_class,
            "class_balance": self.class_balance,
        }


@dataclass(frozen=True)
class ProtocolConstants:
    similarity_threshold: float = 0.85
    activity_gap_threshold: float = 1.0
    hard_negative_pool_size: int = 32
    adversarial_injection_ratio: float = 0.5

    def to_dict(self) -> dict:
        return {
            "similarity_threshold": self.similarity_threshold,
            "activity_gap_threshold": self.activity_gap_threshold,
            "hard_negative_pool_size": self.hard_negative_pool_size,
            "adversarial_injection_ratio": self.adversarial_injection_ratio,
        }


DEFAULT_EPISODE_CONFIG = EpisodeConfig()
DEFAULT_PROTOCOL_CONSTANTS = ProtocolConstants()
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_EPISODES_PER_SPLIT = 400
