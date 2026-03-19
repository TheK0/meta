from __future__ import annotations


def default_benchmark_manifest() -> dict:
    """Return the protocol defaults for FS-Mol-Cliff v3.0."""
    return {
        "benchmark_version": "v3.0",
        "fsmol_data_version": "<fixed_version>",
        "fsmol_episode_generator_commit": "<commit_hash>",
        "fsmol_metric_commit": "<commit_hash>",
        "episode_config": {
            "n_way": 2,
            "support_per_class": 16,
            "query_per_class": 16,
            "class_balance": "balanced",
        },
        "seeds": [0, 1, 2, 3, 4],
        "episodes_per_split": 400,
        "constants": {
            "similarity_threshold": 0.85,
            "activity_gap_threshold": 1.0,
            "hard_negative_pool_size": 32,
            "adversarial_injection_ratio": 0.5,
        },
    }
