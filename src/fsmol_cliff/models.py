from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import (
    DEFAULT_EPISODE_CONFIG,
    DEFAULT_EPISODES_PER_SPLIT,
    DEFAULT_SEEDS,
    FINAL_PROFILE_SPECS,
    PROFILE_SPECS,
    EpisodeConfig,
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
    profiles: dict[str, dict]
    built_profiles: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def default(
        cls,
        *,
        benchmark_version: str = "v4.0",
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
            profiles={name: profile.to_dict() for name, profile in FINAL_PROFILE_SPECS.items()},
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
            "profiles": self.profiles,
            "built_profiles": list(self.built_profiles),
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
