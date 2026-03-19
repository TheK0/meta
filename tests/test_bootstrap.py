from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fsmol_cliff.benchmark import default_benchmark_manifest
from fsmol_cliff.cli import build_parser


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True, capture_output=True, text=True)
    (path / "README.md").write_text("# FS-Mol\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial"], check=True, capture_output=True, text=True)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def test_default_manifest_uses_checkout_commit_for_episode_and_metric_fields(tmp_path: Path) -> None:
    checkout_root = tmp_path / "FS-Mol"
    checkout_root.mkdir()
    commit = _init_git_repo(checkout_root)

    manifest = default_benchmark_manifest(fsmol_checkout_root=checkout_root)

    assert manifest["fsmol_episode_generator_commit"] == commit
    assert manifest["fsmol_metric_commit"] == commit


def test_default_manifest_falls_back_to_placeholder_commit_without_checkout(tmp_path: Path) -> None:
    manifest = default_benchmark_manifest(fsmol_checkout_root=tmp_path / "missing")

    assert manifest["fsmol_episode_generator_commit"] == "<commit_hash>"
    assert manifest["fsmol_metric_commit"] == "<commit_hash>"


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
