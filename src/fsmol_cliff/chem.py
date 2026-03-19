from __future__ import annotations

from functools import lru_cache

import numpy as np


def rdkit_is_available() -> bool:
    try:
        _load_rdkit()
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_rdkit():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return Chem, DataStructs, AllChem, MurckoScaffold


@lru_cache(maxsize=8192)
def canonicalize_isomeric_smiles(smiles: str | None) -> str | None:
    value = _normalize_smiles(smiles)
    if value is None:
        return None
    if not rdkit_is_available():
        return value

    chem, _, _, _ = _load_rdkit()
    molecule = chem.MolFromSmiles(value)
    if molecule is None:
        return value
    return str(chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


@lru_cache(maxsize=8192)
def murcko_scaffold_smiles(smiles: str | None) -> str | None:
    value = canonicalize_isomeric_smiles(smiles)
    if value is None or not rdkit_is_available():
        return None

    _, _, _, murcko_scaffold = _load_rdkit()
    try:
        scaffold = murcko_scaffold.MurckoScaffoldSmilesFromSmiles(value)
    except ValueError:
        return None
    return scaffold or None


@lru_cache(maxsize=32768)
def tanimoto_similarity(smiles_a: str | None, smiles_b: str | None) -> float | None:
    canonical_a = canonicalize_isomeric_smiles(smiles_a)
    canonical_b = canonicalize_isomeric_smiles(smiles_b)
    if canonical_a is None or canonical_b is None or not rdkit_is_available():
        return None

    fingerprint_a = _morgan_fingerprint(canonical_a)
    fingerprint_b = _morgan_fingerprint(canonical_b)
    if fingerprint_a is None or fingerprint_b is None:
        return None

    _, data_structs, _, _ = _load_rdkit()
    return float(data_structs.TanimotoSimilarity(fingerprint_a, fingerprint_b))


def morgan_fingerprint_array(smiles: str | None) -> np.ndarray | None:
    canonical_smiles = canonicalize_isomeric_smiles(smiles)
    if canonical_smiles is None or not rdkit_is_available():
        return None
    fingerprint = _morgan_fingerprint(canonical_smiles)
    if fingerprint is None:
        return None
    _, data_structs, _, _ = _load_rdkit()
    array = np.zeros((2048,), dtype=np.int8)
    data_structs.ConvertToNumpyArray(fingerprint, array)
    return array


@lru_cache(maxsize=8192)
def _morgan_fingerprint(canonical_smiles: str):
    chem, _, all_chem, _ = _load_rdkit()
    molecule = chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        return None
    return all_chem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def _normalize_smiles(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    value = str(smiles).strip()
    return value or None


__all__ = [
    "canonicalize_isomeric_smiles",
    "morgan_fingerprint_array",
    "murcko_scaffold_smiles",
    "rdkit_is_available",
    "tanimoto_similarity",
]
