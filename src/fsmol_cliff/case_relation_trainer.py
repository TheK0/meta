"""CASE-Net v2: Pretrained cross-task cliff-vs-noncliff relation head (NO-GO).

Builds a global relation dataset by sampling active-inactive pairs from
raw FS-Mol train/valid task files.  Trains a RandomForest to distinguish
cliff (gap >= delta) from highsim_noncliff (gap < delta) pairs.

Label convention (v2, corrected):
  y = 1: cliff
  y = 0: highsim_noncliff

Historical note: earlier iterations used "same" / "flip" naming inherited
from an incorrect assumption that highsim_noncliff pairs could be concordant.
Those names are retired.  All variables use cliff / noncliff.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import numpy as np

from .chem import (
    morgan_fingerprint_array,
    murcko_scaffold_smiles,
    require_rdkit,
)


# ---------------------------------------------------------------------------
# Dataset builder (fast — no full assay pipeline)
# ---------------------------------------------------------------------------

def build_global_relation_dataset(
    data_dir: Path,
    split: str = "train",
    *,
    tau: float = 0.8,
    delta: float = 1.0,
    max_assays: int | None = None,
    max_pairs_per_assay: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample active-inactive pairs, compute Tanimoto, label cliff/noncliff.

    Returns (X, y): X (n_pairs, ~4104) float32, y (n_pairs,) int64.
    y: 0 = highsim_noncliff (gap < delta), 1 = cliff (gap >= delta).
    """
    require_rdkit()

    task_dir = data_dir / split
    task_files = sorted(task_dir.glob("*.jsonl.gz"))
    if max_assays is not None:
        task_files = task_files[:max_assays]
    random.shuffle(task_files)

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    assay_count = 0
    pair_count = 0

    for task_file in task_files:
        try:
            feat, lab, n = _process_one_assay(
                task_file, tau=tau, delta=delta, max_pairs=max_pairs_per_assay
            )
            if feat:
                features_list.extend(feat)
                labels_list.extend(lab)
                assay_count += 1
                pair_count += n
        except Exception:
            continue

    print(f"  Processed {assay_count} assays, {pair_count} pairs total")
    if not features_list:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(features_list, axis=0)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def _process_one_assay(
    task_file: Path, *, tau: float, delta: float, max_pairs: int,
) -> tuple[list[np.ndarray], list[int], int]:
    """Process one FS-Mol assay file into pair features and labels."""
    with gzip.open(task_file, "rt") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    # Filter to valid molecules with fingerprints
    valid: list[dict] = []
    for i, rec in enumerate(records):
        smi = _extract_smiles(rec)
        label = _extract_label(rec)
        if not smi or label is None:
            continue
        fp = morgan_fingerprint_array(smi)
        if fp is None:
            continue
        # Extract activity value (use LogRegressionProperty for FS-Mol train format)
        act_raw = rec.get("r", rec.get("standardised_activity",
                         rec.get("LogRegressionProperty",
                         rec.get("RegressionProperty", 0))))
        try:
            activity = float(act_raw)
        except (ValueError, TypeError):
            activity = 0.0
        valid.append({
            "id": str(rec.get("Compound_ID", rec.get("molecule_id", str(i)))),
            "smiles": smi, "label": label, "fp": fp, "activity": activity,
        })

    actives = [m for m in valid if m["label"] == 1]
    inactives = [m for m in valid if m["label"] == 0]
    if len(actives) < 2 or len(inactives) < 2:
        return [], [], 0

    # Scaffold map
    scf_map: dict[str, str] = {}
    for m in actives + inactives:
        try:
            scf = murcko_scaffold_smiles(m["smiles"])
            scf_map[m["id"]] = scf or "NOSCAFFOLD"
        except Exception:
            scf_map[m["id"]] = "NOSCAFFOLD"

    feats: list[np.ndarray] = []
    labels: list[int] = []
    count = 0
    for active in actives:
        for inactive in inactives:
            if count >= max_pairs:
                break
            sim = _tanimoto(active["fp"], inactive["fp"])
            if sim < tau:
                continue
            gap = abs(active["activity"] - inactive["activity"])
            ss = scf_map.get(active["id"]) == scf_map.get(inactive["id"])
            feat = _pair_fingerprint_features(
                active["fp"], inactive["fp"],
                sim=float(sim), same_scaffold=ss,
            )
            feats.append(feat)
            labels.append(1 if gap >= delta else 0)
            count += 1
        if count >= max_pairs:
            break

    return feats, labels, count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_smiles(rec: dict) -> str | None:
    for key in ("canonical_isomeric_smiles", "SMILES", "smiles", "Canonical_SMILES"):
        val = rec.get(key)
        if val and isinstance(val, str) and len(val) > 1:
            return val
    return None


def _extract_label(rec: dict) -> int | None:
    # FS-Mol train/valid uses "Property", test uses "label"
    for key in ("label", "Label", "Property", "reg_label", "binary_label"):
        val = rec.get(key)
        if val is not None:
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
    # Fallback: threshold activity
    for key in ("r", "standardised_activity", "LogRegressionProperty"):
        act = rec.get(key)
        if act is not None:
            try:
                return 1 if float(act) >= 6.5 else 0
            except (ValueError, TypeError):
                pass
    return None


def _tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = float((fp_a & fp_b).sum())
    union = float((fp_a | fp_b).sum())
    return inter / union if union > 0 else 0.0


def _pair_fingerprint_features(
    fp_a: np.ndarray,
    fp_b: np.ndarray,
    *,
    sim: float,
    same_scaffold: bool,
) -> np.ndarray:
    """Build feature vector: 4 scalars + |fp_a - fp_b| + min(fp_a, fp_b)."""
    f_a = fp_a.astype(np.float32)
    f_b = fp_b.astype(np.float32)
    abs_diff = np.abs(f_a - f_b)
    inter = np.minimum(f_a, f_b)
    bit_diff = float((abs_diff > 0).sum())
    shared = float((inter > 0).sum())
    scalar = np.array([sim, float(same_scaffold), bit_diff, shared], dtype=np.float32)
    return np.concatenate([scalar, abs_diff, inter])


# ---------------------------------------------------------------------------
# Model training & inference
# ---------------------------------------------------------------------------

def train_global_relation_head(X: np.ndarray, y: np.ndarray) -> object:
    from sklearn.ensemble import RandomForestClassifier

    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Insufficient training data")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def predict_global_relations(
    head: object, X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_noncliff, p_cliff) probabilities for each pair."""
    proba = head.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 0], proba[:, 1]
    return proba[:, 0], 1.0 - proba[:, 0]
