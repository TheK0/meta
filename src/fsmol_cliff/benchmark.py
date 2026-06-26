from __future__ import annotations

import subprocess
from pathlib import Path

from .fsmol_bridge import default_external_fsmol_root
from .models import BenchmarkManifest


def resolve_git_commit(checkout_root: Path | None) -> str | None:
    if checkout_root is None or not checkout_root.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def resolve_external_fsmol_commit(fsmol_checkout_root: Path | None = None) -> str | None:
    if fsmol_checkout_root is not None:
        return resolve_git_commit(fsmol_checkout_root)
    try:
        default_root = default_external_fsmol_root()
    except FileNotFoundError:
        return None
    return resolve_git_commit(default_root)


def default_benchmark_manifest(*, fsmol_checkout_root: Path | None = None) -> dict:
    """Return the protocol defaults for FS-Mol-Cliff v4.0."""
    commit = resolve_external_fsmol_commit(fsmol_checkout_root)
    return BenchmarkManifest.default(
        fsmol_episode_generator_commit=commit or "<commit_hash>",
        fsmol_metric_commit=commit or "<commit_hash>",
    ).to_dict()
