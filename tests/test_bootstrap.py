from __future__ import annotations

import json

from fsmol_cliff.benchmark import default_benchmark_manifest
from fsmol_cliff.cli import build_parser


def test_default_manifest_matches_v3_protocol_defaults() -> None:
    manifest = default_benchmark_manifest()

    assert manifest["benchmark_version"] == "v3.0"
    assert manifest["episode_config"] == {
        "n_way": 2,
        "support_per_class": 16,
        "query_per_class": 16,
        "class_balance": "balanced",
    }
    assert manifest["seeds"] == [0, 1, 2, 3, 4]
    assert manifest["episodes_per_split"] == 400
    assert manifest["constants"] == {
        "similarity_threshold": 0.85,
        "activity_gap_threshold": 1.0,
        "hard_negative_pool_size": 32,
        "adversarial_injection_ratio": 0.5,
    }


def test_manifest_is_json_serializable() -> None:
    manifest = default_benchmark_manifest()
    assert json.loads(json.dumps(manifest)) == manifest


def test_cli_exposes_expected_top_level_subcommands() -> None:
    parser = build_parser()

    subparsers_action = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
    )

    assert set(subparsers_action.choices) == {
        "fetch-fsmol",
        "adapter-status",
        "build-assets",
        "build-release",
        "build-episodes",
        "evaluate",
        "aggregate",
        "validate-hypotheses",
    }
