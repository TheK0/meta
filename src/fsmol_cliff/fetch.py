from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib import parse, request

from .benchmark import resolve_git_commit
from .io import write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_remote_url(source: str) -> bool:
    return parse.urlsplit(source).scheme in {"http", "https", "ftp", "file"}


def _prepare_target_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        return
    if any(data_dir.iterdir()):
        raise ValueError(f"Target data directory must be empty: {data_dir}")


def _copy_local_source_directory(source_dir: Path, data_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Local source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Local source must be a directory: {source_dir}")
    if source_dir.resolve() == data_dir.resolve():
        data_dir.mkdir(parents=True, exist_ok=True)
        return
    _prepare_target_dir(data_dir)
    for child in sorted(source_dir.iterdir(), key=lambda entry: entry.name):
        if child.name == ".git":
            continue
        destination = data_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _download_remote_archive(source_url: str, data_dir: Path) -> None:
    _prepare_target_dir(data_dir)
    suffix = "".join(Path(parse.urlsplit(source_url).path).suffixes) or ".zip"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            with request.urlopen(source_url) as response:
                shutil.copyfileobj(response, handle)
        shutil.unpack_archive(str(temp_path), str(data_dir))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def populate_data_dir(data_dir: Path, source_url: str | Path) -> None:
    source = str(source_url)
    if _is_remote_url(source):
        _download_remote_archive(source, data_dir)
        return
    _copy_local_source_directory(Path(source), data_dir)


def infer_source_commit(source_url: str | Path | None) -> str | None:
    if source_url is None:
        return None
    source = str(source_url)
    if _is_remote_url(source):
        return None
    return resolve_git_commit(Path(source))


def build_source_manifest(
    data_dir: Path,
    source_url: str | Path | None,
    task_list_file: Path | None,
    fsmol_data_version: str,
) -> dict:
    manifest = {
        "fsmol_data_version": fsmol_data_version,
        "data_dir": str(data_dir.resolve()),
        "source_url": None if source_url is None else str(source_url),
        "source_commit": infer_source_commit(source_url),
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
    source_url: str | Path | None,
    task_list_file: Path | None,
    fsmol_data_version: str,
) -> None:
    if source_url is not None:
        populate_data_dir(data_dir, source_url)
    payload = build_source_manifest(
        data_dir=data_dir,
        source_url=source_url,
        task_list_file=task_list_file,
        fsmol_data_version=fsmol_data_version,
    )
    write_json(output_path, payload)
