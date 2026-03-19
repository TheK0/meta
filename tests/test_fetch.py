from __future__ import annotations

import json
from pathlib import Path

from fsmol_cliff.fetch import build_source_manifest, sha256_file, write_source_manifest


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "tasks.json"
    target.write_text('{"test":["CHEMBL1"]}')

    digest = sha256_file(target)

    assert digest == sha256_file(target)
    assert len(digest) == 64


def test_build_source_manifest_records_local_data_dir_and_task_list_hash(tmp_path: Path) -> None:
    data_dir = tmp_path / "fsmol"
    data_dir.mkdir()
    task_list = tmp_path / "tasks.json"
    task_list.write_text('{"test":["CHEMBL1"]}')

    manifest = build_source_manifest(
        data_dir=data_dir,
        source_url=None,
        task_list_file=task_list,
        fsmol_data_version="fsmol-0.1",
    )

    assert manifest["fsmol_data_version"] == "fsmol-0.1"
    assert manifest["data_dir"] == str(data_dir.resolve())
    assert manifest["task_list_file"]["sha256"] == sha256_file(task_list)


def test_write_source_manifest_persists_json_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "fsmol"
    data_dir.mkdir()
    out_file = tmp_path / "raw_manifest.json"

    write_source_manifest(
        output_path=out_file,
        data_dir=data_dir,
        source_url="https://example.test/fsmol.zip",
        task_list_file=None,
        fsmol_data_version="fsmol-0.1",
    )

    payload = json.loads(out_file.read_text())
    assert payload["source_url"] == "https://example.test/fsmol.zip"
