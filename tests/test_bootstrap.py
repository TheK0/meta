from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fsmol_cliff.aggregate import aggregate_task_result_rows
from fsmol_cliff.aggregate import paired_bootstrap_delta_ci, task_bootstrap_ci
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


def test_default_manifest_matches_v4_protocol_defaults() -> None:
    manifest = default_benchmark_manifest()

    assert manifest["benchmark_version"] == "v4.0"
    assert manifest["episode_config"] == {
        "n_way": 2,
        "support_per_class": 16,
        "query_per_class": 16,
        "class_balance": "balanced",
    }
    assert manifest["seeds"] == [0, 1, 2, 3, 4]
    assert manifest["episodes_per_split"] == 400
    assert manifest["profiles"] == {
        "strict": {
            "name": "strict",
            "similarity_threshold": 0.85,
            "activity_gap_threshold": 1.0,
            "hard_negative_pool_size": 32,
            "adversarial_injection_ratio": 0.5,
            "min_cliff_pairs": 25,
            "min_noncliff_pairs": 10,
            "min_valid_molecules": 50,
            "min_positive_molecules": 15,
            "min_negative_molecules": 15,
            "min_anchor_molecules": 10,
            "min_cliff_negatives": 10,
            "min_m_avail": 2,
        },
        "relaxed": {
            "name": "relaxed",
            "similarity_threshold": 0.8,
            "activity_gap_threshold": 1.0,
            "hard_negative_pool_size": 32,
            "adversarial_injection_ratio": 0.5,
            "min_cliff_pairs": 25,
            "min_noncliff_pairs": 10,
            "min_valid_molecules": 50,
            "min_positive_molecules": 15,
            "min_negative_molecules": 15,
            "min_anchor_molecules": 10,
            "min_cliff_negatives": 10,
            "min_m_avail": 2,
        },
    }
    assert manifest["built_profiles"] == []


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
        "audit-attrition",
        "build-assets",
        "build-release",
        "build-episodes",
        "evaluate",
        "aggregate",
        "validate-hypotheses",
    }


def test_aggregate_task_result_rows_preserves_profile_and_result_tier() -> None:
    rows = [
        {
            "task_id": "CHEMBL1",
            "profile": "relaxed",
            "result_tier": "final",
            "seed": 0,
            "split_type": "standard",
            "metric": "q_psr",
            "score": 0.6,
        },
        {
            "task_id": "CHEMBL1",
            "profile": "relaxed",
            "result_tier": "final",
            "seed": 1,
            "split_type": "standard",
            "metric": "q_psr",
            "score": 0.4,
        },
        {
            "task_id": "CHEMBL2",
            "profile": "relaxed",
            "result_tier": "final",
            "seed": 0,
            "split_type": "standard",
            "metric": "q_psr",
            "score": 0.8,
        },
    ]

    aggregated = aggregate_task_result_rows(rows)

    assert aggregated == [
        {
            "profile": "relaxed",
            "result_tier": "final",
            "split_type": "standard",
            "metric": "q_psr",
            "score": 0.65,
            "ci_low": 0.5,
            "ci_high": 0.8,
            "num_tasks": 2,
        }
    ]


def test_task_bootstrap_ci_returns_ordered_interval() -> None:
    ci = task_bootstrap_ci([0.2, 0.6, 0.8], iterations=500, seed=7)

    assert ci["mean"] == 0.5333333333333333
    assert ci["low"] <= ci["mean"] <= ci["high"]


def test_paired_bootstrap_delta_ci_uses_task_level_deltas() -> None:
    ci = paired_bootstrap_delta_ci(
        baseline=[0.4, 0.5, 0.6],
        treatment=[0.6, 0.7, 0.8],
        iterations=500,
        seed=11,
    )

    assert ci["delta_mean"] == pytest.approx(0.2)
    assert ci["low"] <= ci["delta_mean"] <= ci["high"]


def test_aggregate_task_result_rows_averages_across_seeds_then_tasks() -> None:
    rows = [
        {"task_id": "t1", "seed": 0, "split_type": "standard", "metric": "q_psr", "score": 0.5},
        {"task_id": "t1", "seed": 1, "split_type": "standard", "metric": "q_psr", "score": 0.7},
        {"task_id": "t2", "seed": 0, "split_type": "standard", "metric": "q_psr", "score": 0.3},
        {"task_id": "t2", "seed": 1, "split_type": "standard", "metric": "q_psr", "score": 0.5},
    ]

    aggregated = aggregate_task_result_rows(rows, bootstrap_iterations=200, bootstrap_seed=3)
    result = aggregated[0]

    assert result["split_type"] == "standard"
    assert result["metric"] == "q_psr"
    assert result["score"] == pytest.approx(0.5)
    assert result["num_tasks"] == 2
    assert result["ci_low"] <= result["score"] <= result["ci_high"]
