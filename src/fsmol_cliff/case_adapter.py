# CASE-Net v1: per-episode logistic-regression relation head (NO-GO).
# Historical note: internal variable names p_same / p_flip are
# retained for v1 code only.  The corrected v2 naming uses
# p_cliff / p_noncliff.  See case_relation_trainer.py for v2.
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def compute_pair_features(
    *,
    anchor_id: str,
    neighbor_id: str,
    pair_info: Mapping[str, object] | None,
    assay_context: Mapping[str, object],
    proto_embeddings: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build a fixed-length feature vector for a molecule pair.

    Features (order-stable):
      0: Tanimoto similarity (from pair metadata or 0.0)
      1: |gap_abs| (from pair metadata or 0.0)
      2: same_scaffold (1.0 if same Murcko scaffold, else 0.0)
      3: support_label_anchor (label if anchor in support, else -1)

    If proto_embeddings is provided, appends:
      |emb(anchor) - emb(neighbor)| (element-wise absolute diff)

    Returns a 1-D float32 array.
    """
    sim = float(pair_info.get("sim", 0.0)) if pair_info else 0.0
    gap = float(pair_info.get("gap_abs", 0.0)) if pair_info else 0.0
    ss = 1.0 if (pair_info and pair_info.get("same_scaffold")) else 0.0

    labels = assay_context.get("labels", {})
    anchor_label = int(labels.get(anchor_id, -1))
    neighbor_label = int(labels.get(neighbor_id, -1))
    # Use anchor's label as a feature; -1 means not in assay labels
    anchor_lbl_feat = float(anchor_label) if anchor_label in (0, 1) else -1.0

    feats = [sim, gap, ss, anchor_lbl_feat]

    if proto_embeddings is not None:
        emb_a = proto_embeddings.get(anchor_id)
        emb_n = proto_embeddings.get(neighbor_id)
        if emb_a is not None and emb_n is not None:
            feats.extend(np.abs(emb_a - emb_n).tolist())
        else:
            feats.extend([0.0] * 8)  # placeholder; adjust dim as needed

    return np.array(feats, dtype=np.float32)


def train_relation_head(
    *,
    features: list[np.ndarray],
    labels: list[str],
) -> object:
    """Train a lightweight per-episode relation classifier.

    Returns a scikit-learn LogisticRegression model.
    Labels "same" → 0, "flip" → 1.
    """
    if not features:
        return _FallbackRelationHead()

    from sklearn.linear_model import LogisticRegression

    X = np.stack(features, axis=0)
    y = np.array([0 if lab == "same" else 1 for lab in labels], dtype=np.int64)

    if len(set(y)) < 2:
        return _FallbackRelationHead()

    try:
        clf = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=200,
            class_weight="balanced",
        )
        clf.fit(X, y)
    except (ValueError, RuntimeError):
        return _FallbackRelationHead()

    return clf


def predict_relation_probs(
    head: object,
    features: list[np.ndarray],
) -> tuple[list[float], list[float]]:
    """Return (p_same, p_flip) for each pair feature vector."""
    if isinstance(head, _FallbackRelationHead):
        return head.predict(features)

    if not features:
        return [], []

    X = np.stack(features, axis=0)
    try:
        proba = head.predict_proba(X)
        if proba.shape[1] == 2:
            p_same = proba[:, 0].tolist()
            p_flip = proba[:, 1].tolist()
        else:
            p_same = proba[:, 0].tolist()
            p_flip = (1.0 - proba[:, 0]).tolist()
    except (ValueError, RuntimeError):
        p_same = [0.5] * len(features)
        p_flip = [0.5] * len(features)

    return p_same, p_flip


class _FallbackRelationHead:
    """Deterministic fallback: uses pair similarity to guess same/flip."""

    def predict(self, features: list[np.ndarray]) -> tuple[list[float], list[float]]:
        p_flip: list[float] = []
        p_same: list[float] = []
        for feat in features:
            sim = float(feat[0]) if len(feat) > 0 else 0.0
            pf = min(1.0, max(0.0, sim))
            ps = 1.0 - pf
            p_flip.append(pf)
            p_same.append(ps)
        return p_same, p_flip
