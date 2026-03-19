from __future__ import annotations

from fsmol_cliff.adapters import AdapterSpec, default_adapter_registry


def test_default_adapter_registry_exposes_all_official_families() -> None:
    registry = default_adapter_registry()

    assert set(registry) == {
        "baseline",
        "mat",
        "maml",
        "multitask",
        "protonet",
    }


def test_baseline_adapter_points_to_official_script_and_entry_function() -> None:
    spec = default_adapter_registry()["baseline"]

    assert spec == AdapterSpec(
        name="baseline",
        script_path="fs_mol/baseline_test.py",
        callable_name="test",
        requires_checkpoint=False,
        family="sklearn",
    )
