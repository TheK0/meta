from __future__ import annotations

import hashlib
from pathlib import Path

from .io import write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(
    data_dir: Path,
    source_url: str | None,
    task_list_file: Path | None,
    fsmol_data_version: str,
) -> dict:
    manifest = {
        "fsmol_data_version": fsmol_data_version,
        "data_dir": str(data_dir.resolve()),
        "source_url": source_url,
        "task_list_file": None,
    }
    if task_list_file is not None:
        manifest["task_list_file"] = {
            "path": str(task_list_file.resolve()),
            "sha256": sha256_file(task_list_file),
        }
    return manifest


def write_source_manifest(
    output_path: Path,
    data_dir: Path,
    source_url: str | None,
    task_list_file: Path | None,
    fsmol_data_version: str,
) -> None:
    payload = build_source_manifest(
        data_dir=data_dir,
        source_url=source_url,
        task_list_file=task_list_file,
        fsmol_data_version=fsmol_data_version,
    )
    write_json(output_path, payload)
