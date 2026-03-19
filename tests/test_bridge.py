from __future__ import annotations

from pathlib import Path

from fsmol_cliff.adapters import default_adapter_registry
from fsmol_cliff.fsmol_bridge import default_external_fsmol_root, load_callable_from_spec, resolve_script_path


def test_default_external_fsmol_root_matches_local_reference_checkout() -> None:
    root = default_external_fsmol_root()
    assert root.name == "FS-Mol"


def test_resolve_script_path_points_to_existing_official_script() -> None:
    spec = default_adapter_registry()["baseline"]
    script_path = resolve_script_path(spec)

    assert script_path == default_external_fsmol_root() / "fs_mol" / "baseline_test.py"
    assert script_path.exists()


def test_load_callable_from_official_baseline_script() -> None:
    callable_obj = load_callable_from_spec(default_adapter_registry()["baseline"])

    assert callable(callable_obj)
    assert callable_obj.__name__ == "test"
