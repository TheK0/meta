from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import (
    DEFAULT_EPISODE_CONFIG,
    DEFAULT_EPISODES_PER_SPLIT,
    DEFAULT_PROTOCOL_CONSTANTS,
    DEFAULT_SEEDS,
    EpisodeConfig,
    ProtocolConstants,
)


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_version: str
    fsmol_data_version: str
    fsmol_episode_generator_commit: str
    fsmol_metric_commit: str
    episode_config: EpisodeConfig
    seeds: tuple[int, ...]
    episodes_per_split: int
    constants: ProtocolConstants

    @classmethod
    def default(
        cls,
        *,
        benchmark_version: str = "v3.0",
        fsmol_data_version: str = "<fixed_version>",
        fsmol_episode_generator_commit: str = "<commit_hash>",
        fsmol_metric_commit: str = "<commit_hash>",
    ) -> "BenchmarkManifest":
        return cls(
            benchmark_version=benchmark_version,
            fsmol_data_version=fsmol_data_version,
            fsmol_episode_generator_commit=fsmol_episode_generator_commit,
            fsmol_metric_commit=fsmol_metric_commit,
            episode_config=DEFAULT_EPISODE_CONFIG,
            seeds=DEFAULT_SEEDS,
            episodes_per_split=DEFAULT_EPISODES_PER_SPLIT,
            constants=DEFAULT_PROTOCOL_CONSTANTS,
        )

    def to_dict(self) -> dict:
        return {
            "benchmark_version": self.benchmark_version,
            "fsmol_data_version": self.fsmol_data_version,
            "fsmol_episode_generator_commit": self.fsmol_episode_generator_commit,
            "fsmol_metric_commit": self.fsmol_metric_commit,
            "episode_config": self.episode_config.to_dict(),
            "seeds": list(self.seeds),
            "episodes_per_split": self.episodes_per_split,
            "constants": self.constants.to_dict(),
        }


@dataclass(frozen=True)
class PairRecord:
    assay_id: str
    anchor_id: str
    neg_id: str
    sim: float
    gap_abs: float
    same_scaffold: bool
    pair_type: str
    anchor_label: int = 1
    neg_label: int = 0

    def sort_key(self) -> tuple[float, float, str]:
        return (-self.sim, -self.gap_abs, self.neg_id)

    def to_dict(self) -> dict:
        return asdict(self)
