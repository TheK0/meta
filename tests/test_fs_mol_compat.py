from __future__ import annotations

from fsmol_cliff.fsmol_bridge import install_fs_mol_compat_patches, load_callable_from_spec
from fsmol_cliff.adapters import default_adapter_registry


def test_install_fs_mol_compat_patches_allows_multitask_and_protonet_imports() -> None:
    install_fs_mol_compat_patches()

    multitask = load_callable_from_spec(default_adapter_registry()["multitask"])
    protonet = load_callable_from_spec(default_adapter_registry()["protonet"])

    assert multitask.__name__ == "eval_model_by_finetuning_on_task"
    assert protonet.__name__ == "evaluate_protonet_model"
