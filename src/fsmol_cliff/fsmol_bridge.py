from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

from .adapters import AdapterSpec


def default_external_fsmol_root() -> Path:
    override = os.environ.get("FSMOL_EXTERNAL_ROOT")
    if override:
        return Path(override)
    return Path("/Volumes/macplus/project/meta/external/FS-Mol")


def resolve_script_path(spec: AdapterSpec, root: Path | None = None) -> Path:
    base = root or default_external_fsmol_root()
    return base / spec.script_path


def load_module_from_script(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_callable_from_spec(spec: AdapterSpec, root: Path | None = None):
    module = load_module_from_script(resolve_script_path(spec, root=root))
    return getattr(module, spec.callable_name)
