from __future__ import annotations

from fsmol_cliff.adapters import AdapterSpec, default_adapter_registry, select_cliff_aware_hard_negatives


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


def test_select_cliff_aware_hard_negatives_picks_unique_candidates_in_order() -> None:
    selected = select_cliff_aware_hard_negatives(
        support_pos_ids=["a1", "a2"],
        excluded_ids={"a1", "a2", "n1", "q1"},
        anchor_to_hardnegs={
            "a1": ["n1", "hn1", "hn2"],
            "a2": ["hn1", "hn3"],
        },
        max_per_anchor=1,
    )

    assert selected == ["hn1", "hn3"]
