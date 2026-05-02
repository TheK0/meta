from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from .adapters import AdapterSpec

# ---------------------------------------------------------------------------
# Upstream source hash-lock registry
#
# Each entry maps a module name to (expected_sha256, upstream_relative_path).
# If the upstream source changes, the hash check will fail with a clear error
# instead of silently producing a broken patch.
# ---------------------------------------------------------------------------
_UPSTREAM_HASH_REGISTRY: dict[str, tuple[str, str]] = {
    # Order matters: leaf modules must be patched before modules that import them.
    # gnn and graph_readout are leaf dependencies (only import torch_scatter).
    # graph_feature_extractor imports gnn + graph_readout — must come after.
    "fs_mol.modules.gnn": (
        "40b162893b276f9642f071307169456872dc8b4a2e06f2c0ce5965974a7d1ca2",
        "fs_mol/modules/gnn.py",
    ),
    "fs_mol.modules.graph_readout": (
        "7c16408d41dc10137efc6eb4576feee80de8ffd8677df1bdfa5eb07300d6c82b",
        "fs_mol/modules/graph_readout.py",
    ),
    "fs_mol.modules.graph_feature_extractor": (
        "7ddb60879a15c348de9d0080f53b7fc1699dd1683377afe38c2421c320195a0b",
        "fs_mol/modules/graph_feature_extractor.py",
    ),
    "fs_mol.models.protonet": (
        "28905e0729885bbb77af8365377b28d658bc3620f262046c005055034a6bb218",
        "fs_mol/models/protonet.py",
    ),
}

# Per-module symbols that must exist after patching.
_REQUIRED_PATCHED_SYMBOLS: dict[str, list[str]] = {
    "fs_mol.modules.gnn": ["scatter_sum", "scatter_mean", "scatter_max", "scatter_log_softmax"],
    "fs_mol.modules.graph_readout": ["scatter_softmax", "scatter"],
    "fs_mol.modules.graph_feature_extractor": ["GraphFeatureExtractorConfig", "GraphFeatureExtractor"],
    "fs_mol.models.protonet": ["PrototypicalNetwork"],
}


def default_external_fsmol_root() -> Path:
    override = os.environ.get("FSMOL_EXTERNAL_ROOT")
    if override:
        path = Path(override)
    else:
        path = Path("/Volumes/macplus/project/meta/external/FS-Mol")
    if not path.exists():
        raise FileNotFoundError(
            f"FS-Mol external repository not found at: {path}\n"
            f"Set the FSMOL_EXTERNAL_ROOT environment variable to point to "
            f"a local checkout of the FS-Mol repository.\n"
            f"  export FSMOL_EXTERNAL_ROOT=/path/to/FS-Mol"
        )
    return path


def resolve_script_path(spec: AdapterSpec, root: Path | None = None) -> Path:
    base = root or default_external_fsmol_root()
    return base / spec.script_path


# ---------------------------------------------------------------------------
# sys.path context manager
# ---------------------------------------------------------------------------

@contextmanager
def _sys_path_entries(entries: list[str]) -> Iterator[None]:
    """Temporarily prepend *entries* to sys.path; restore on exit."""
    previous = list(sys.path)
    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        yield
    finally:
        sys.path[:] = previous


# ---------------------------------------------------------------------------
# Source-patch validation
# ---------------------------------------------------------------------------

def _verify_upstream_hash(root: Path, module_name: str, relative_path: str, expected_hash: str) -> None:
    """Raise RuntimeError if the upstream source hash does not match the expected value."""
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Upstream FS-Mol source file missing: {path}\n"
            f"Ensure the FS-Mol checkout is complete."
        )
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Upstream FS-Mol source has changed for {module_name}.\n"
            f"  File: {path}\n"
            f"  Expected SHA256: {expected_hash}\n"
            f"  Actual SHA256:   {actual_hash}\n\n"
            f"The vendored patches in fsmol_bridge._apply_source_patches may no longer "
            f"apply correctly. Update _UPSTREAM_HASH_REGISTRY with the new hash after "
            f"verifying the patches still work.\n"
        )


def _validate_patched_module(module_name: str, module: ModuleType) -> None:
    """Check that key symbols exist on the patched module."""
    required = _REQUIRED_PATCHED_SYMBOLS.get(module_name, [])
    missing = [sym for sym in required if not hasattr(module, sym)]
    if missing:
        raise ImportError(
            f"Patched module {module_name} is missing expected symbols: {missing}\n"
            f"The upstream source may have renamed or removed these symbols. "
            f"Update _apply_source_patches and _REQUIRED_PATCHED_SYMBOLS."
        )


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

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
    with _sys_path_entries(path_entries):
        install_fs_mol_compat_patches(root)
        spec.loader.exec_module(module)
    return module


def load_callable_from_spec(spec: AdapterSpec, root: Path | None = None):
    module = load_module_from_script(resolve_script_path(spec, root=root))
    return getattr(module, spec.callable_name)


# ---------------------------------------------------------------------------
# Compatibility patching
# ---------------------------------------------------------------------------

def install_fs_mol_compat_patches(root: Path | None = None) -> None:
    base = root or default_external_fsmol_root()
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    vendor_mat_src = Path(__file__).resolve().parents[2] / "vendor" / "MAT" / "src"
    if vendor_mat_src.exists() and str(vendor_mat_src) not in sys.path:
        sys.path.insert(0, str(vendor_mat_src))

    for module_name, (expected_hash, relative_path) in _UPSTREAM_HASH_REGISTRY.items():
        if module_name in sys.modules:
            continue
        upstream_path = base / relative_path
        if not upstream_path.exists():
            continue
        _verify_upstream_hash(base, module_name, relative_path, expected_hash)
        _load_patched_module(module_name, upstream_path)


def _load_patched_module(module_name: str, path: Path) -> ModuleType:
    source = path.read_text()
    source = _apply_source_patches(module_name, source)
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(path))
    if spec is None:
        raise ImportError(f"Could not create spec for patched module {module_name}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    module.__package__ = module_name.rsplit(".", 1)[0]
    sys.modules[module_name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    _validate_patched_module(module_name, module)
    return module


def _cuda_scatter_available() -> bool:
    """Return True if native CUDA torch_scatter is usable (no compat patching needed)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import torch_scatter  # noqa: F401

        return True
    except ImportError:
        return False


def _apply_source_patches(module_name: str, source: str) -> str:
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
    elif module_name == "fs_mol.modules.gnn":
        if not _cuda_scatter_available():
            source = source.replace(
                "from torch_scatter import scatter_sum, scatter_log_softmax, scatter_mean, scatter_max",
                "from fsmol_cliff.torch_scatter_compat import "
                "scatter_sum, scatter_log_softmax, scatter_mean, scatter_max",
            )
    elif module_name == "fs_mol.modules.graph_readout":
        if not _cuda_scatter_available():
            source = source.replace(
                "from torch_scatter import scatter_softmax, scatter",
                "from fsmol_cliff.torch_scatter_compat import scatter_softmax, scatter",
            )
    return source
