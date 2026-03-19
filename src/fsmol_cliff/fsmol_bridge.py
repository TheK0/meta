from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
import importlib

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
    root = script_path.parent.parent
    path_entries = [str(root)]
    mat_src = root / "third_party" / "MAT" / "src"
    vendor_mat_src = Path(__file__).resolve().parents[2] / "vendor" / "MAT" / "src"
    if mat_src.exists():
        path_entries.append(str(mat_src))
    elif vendor_mat_src.exists():
        path_entries.append(str(vendor_mat_src))
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    previous_sys_path = list(sys.path)
    for entry in reversed(path_entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        install_fs_mol_compat_patches(root)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_sys_path
    return module


def load_callable_from_spec(spec: AdapterSpec, root: Path | None = None):
    module = load_module_from_script(resolve_script_path(spec, root=root))
    return getattr(module, spec.callable_name)


def install_fs_mol_compat_patches(root: Path | None = None) -> None:
    base = root or default_external_fsmol_root()
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    vendor_mat_src = Path(__file__).resolve().parents[2] / "vendor" / "MAT" / "src"
    if vendor_mat_src.exists() and str(vendor_mat_src) not in sys.path:
        sys.path.insert(0, str(vendor_mat_src))
    patches = {
        "fs_mol.modules.graph_feature_extractor": base / "fs_mol" / "modules" / "graph_feature_extractor.py",
        "fs_mol.models.protonet": base / "fs_mol" / "models" / "protonet.py",
    }
    for module_name, path in patches.items():
        if module_name in sys.modules or not path.exists():
            continue
        _load_patched_module(module_name, path)


def _load_patched_module(module_name: str, path: Path) -> ModuleType:
    source = path.read_text()
    if module_name == "fs_mol.modules.graph_feature_extractor":
        source = source.replace("from dataclasses import dataclass", "from dataclasses import dataclass, field")
        source = source.replace("gnn_config: GNNConfig = GNNConfig()", "gnn_config: GNNConfig = field(default_factory=GNNConfig)")
        source = source.replace(
            "readout_config: GraphReadoutConfig = GraphReadoutConfig()",
            "readout_config: GraphReadoutConfig = field(default_factory=GraphReadoutConfig)",
        )
    elif module_name == "fs_mol.models.protonet":
        source = source.replace("from dataclasses import dataclass", "from dataclasses import dataclass, field")
        source = source.replace(
            "graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()",
            "graph_feature_extractor_config: GraphFeatureExtractorConfig = field(default_factory=GraphFeatureExtractorConfig)",
        )

    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(path))
    if spec is None:
        raise ImportError(f"Could not create spec for patched module {module_name}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    module.__package__ = module_name.rsplit(".", 1)[0]
    sys.modules[module_name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module
