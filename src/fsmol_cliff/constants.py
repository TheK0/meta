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


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    constants: ProtocolConstants
    min_cliff_pairs: int = 25
    min_noncliff_pairs: int = 10
    min_valid_molecules: int = 50
    min_positive_molecules: int = 15
    min_negative_molecules: int = 15
    min_anchor_molecules: int = 10
    min_cliff_negatives: int = 10
    min_m_avail: int = 2

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            **self.constants.to_dict(),
            "min_cliff_pairs": self.min_cliff_pairs,
            "min_noncliff_pairs": self.min_noncliff_pairs,
            "min_valid_molecules": self.min_valid_molecules,
            "min_positive_molecules": self.min_positive_molecules,
            "min_negative_molecules": self.min_negative_molecules,
            "min_anchor_molecules": self.min_anchor_molecules,
            "min_cliff_negatives": self.min_cliff_negatives,
            "min_m_avail": self.min_m_avail,
        }


STRICT_PROFILE = BenchmarkProfile(
    name="strict",
    constants=ProtocolConstants(
        similarity_threshold=0.85,
        activity_gap_threshold=1.0,
        hard_negative_pool_size=32,
        adversarial_injection_ratio=0.5,
    ),
)

RELAXED_PROFILE = BenchmarkProfile(
    name="relaxed",
    constants=ProtocolConstants(
        similarity_threshold=0.80,
        activity_gap_threshold=1.0,
        hard_negative_pool_size=32,
        adversarial_injection_ratio=0.5,
    ),
)

RELAXED_COVEXT_10_10_PROFILE = BenchmarkProfile(
    name="relaxed_covext_10_10",
    constants=RELAXED_PROFILE.constants,
    min_cliff_pairs=10,
    min_noncliff_pairs=10,
)

RELAXED_COVEXT_10_5_PROFILE = BenchmarkProfile(
    name="relaxed_covext_10_5",
    constants=RELAXED_PROFILE.constants,
    min_cliff_pairs=10,
    min_noncliff_pairs=5,
)

PROFILE_SPECS = {
    STRICT_PROFILE.name: STRICT_PROFILE,
    RELAXED_PROFILE.name: RELAXED_PROFILE,
    RELAXED_COVEXT_10_10_PROFILE.name: RELAXED_COVEXT_10_10_PROFILE,
    RELAXED_COVEXT_10_5_PROFILE.name: RELAXED_COVEXT_10_5_PROFILE,
}
