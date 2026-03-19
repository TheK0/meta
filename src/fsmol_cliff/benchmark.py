from __future__ import annotations

from .models import BenchmarkManifest


def default_benchmark_manifest() -> dict:
    """Return the protocol defaults for FS-Mol-Cliff v3.0."""
    return BenchmarkManifest.default().to_dict()
