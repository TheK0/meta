from __future__ import annotations

from pathlib import Path

from fsmol_cliff.io import write_json
from fsmol_cliff.models import BenchmarkManifest


def test_write_json_persists_sorted_manifest(tmp_path: Path) -> None:
    target = tmp_path / "benchmark_manifest.json"

    write_json(target, BenchmarkManifest.default().to_dict())

    text = target.read_text()
    assert '"benchmark_version": "v4.0"' in text
    assert '"profiles"' in text
    assert '"built_profiles": []' in text
    assert text.endswith("\n")
