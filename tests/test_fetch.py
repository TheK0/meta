from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
import zipfile

import pytest

from fsmol_cliff.fetch import (
    build_source_manifest,
    populate_data_dir,
    sha256_file,
    write_source_manifest,
)


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test User"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial"], check=True, capture_output=True, text=True)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    assert manifest["source_commit"] is None
    assert manifest["task_list_file"]["sha256"] == sha256_file(task_list)


def test_write_source_manifest_persists_json_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "fsmol"
    data_dir.mkdir()
    out_file = tmp_path / "raw_manifest.json"

    write_source_manifest(
        output_path=out_file,
        data_dir=data_dir,
        source_url=None,
        task_list_file=None,
        fsmol_data_version="fsmol-0.1",
    )

    payload = json.loads(out_file.read_text())
    assert payload["source_url"] is None
    assert payload["source_commit"] is None


def test_write_source_manifest_copies_local_source_directory_and_records_commit(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "CHEMBL1.jsonl"
    source_file.write_text('{"molecule_id":"mol-1"}\n')
    commit = _init_git_repo(source_dir)
    data_dir = tmp_path / "fetched"
    manifest_path = tmp_path / "benchmark_manifest.source.json"

    write_source_manifest(
        output_path=manifest_path,
        data_dir=data_dir,
        source_url=str(source_dir),
        task_list_file=None,
        fsmol_data_version="fsmol-0.1",
    )

    assert (data_dir / "CHEMBL1.jsonl").read_text() == source_file.read_text()
    payload = json.loads(manifest_path.read_text())
    assert payload["source_url"] == str(source_dir)
    assert payload["source_commit"] == commit


def test_populate_data_dir_extracts_remote_archive_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("test/CHEMBL1.jsonl", '{"molecule_id":"mol-1"}\n')

    class FakeResponse(io.BytesIO):
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

    def fake_urlopen(url: str) -> FakeResponse:
        assert url == "https://example.test/fsmol.zip"
        return FakeResponse(archive_bytes.getvalue())

    monkeypatch.setattr("fsmol_cliff.fetch.request.urlopen", fake_urlopen)

    data_dir = tmp_path / "downloaded"
    populate_data_dir(data_dir, "https://example.test/fsmol.zip")

    assert (data_dir / "test" / "CHEMBL1.jsonl").read_text() == '{"molecule_id":"mol-1"}\n'
