"""CASE-Net v2: pair-level training and validation.

Usage:
  PYTHONPATH=src python -m fsmol_cliff.case_train_v2 \
    --data-dir fs-mol --output-dir outputs/case_net_v2 \
    --max-train 500 --max-valid 100
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> dict:
    args = {}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:].replace("-", "_")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                args[key] = argv[i + 1]
                i += 2
            else:
                args[key] = "true"
                i += 1
        else:
            i += 1
    return args


def main() -> None:
    args = parse_args()
    data_dir = Path(args.get("data_dir", "fs-mol"))
    output_dir = Path(args.get("output_dir", "outputs/case_net_v2"))
    max_train = int(args.get("max_train", 500))
    max_valid = int(args.get("max_valid", 100))
    tau = float(args.get("tau", 0.8))
    delta = float(args.get("delta", 1.0))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Build training data from FS-Mol train ----
    print("Building training dataset from FS-Mol train...")
    from .case_relation_trainer import build_global_relation_dataset

    X_train, y_train = build_global_relation_dataset(
        data_dir, "train", tau=tau, delta=delta, max_assays=max_train, max_pairs_per_assay=200
    )
    print(f"  Training pairs: {len(X_train)} (flip={int(y_train.sum())}, same={len(y_train)-int(y_train.sum())})")

    if len(X_train) < 50 or len(np.unique(y_train)) < 2:
        print("  FATAL: Insufficient training data")
        sys.exit(1)

    # ---- Phase 2: Train relation head ----
    print("Training RandomForest relation head...")
    from .case_relation_trainer import train_global_relation_head

    head = train_global_relation_head(X_train, y_train)

    # ---- Phase 3: Validate on FS-Mol valid ----
    print(f"Building validation dataset from FS-Mol valid (max {max_valid} assays)...")
    X_val, y_val = build_global_relation_dataset(
        data_dir, "valid", tau=tau, delta=delta, max_assays=max_valid
    )
    print(f"  Validation pairs: {len(X_val)} (flip={int(y_val.sum())}, same={len(y_val)-int(y_val.sum())})")

    if len(X_val) < 10:
        print("  FATAL: Insufficient validation data")
        sys.exit(1)

    # ---- Phase 4: Evaluate ----
    from .case_relation_trainer import predict_global_relations
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )

    p_same, p_flip = predict_global_relations(head, X_val)

    auprc = average_precision_score(y_val, p_flip)
    auc = roc_auc_score(y_val, p_flip)
    y_pred = (p_flip > 0.5).astype(np.int64)
    bacc = balanced_accuracy_score(y_val, y_pred)

    # Cliff probability by class
    cliff_mask = y_val == 1
    p_cliff_on_cliff = p_flip[cliff_mask].mean() if cliff_mask.sum() > 0 else 0.0
    p_cliff_on_noncliff = p_flip[~cliff_mask].mean() if (~cliff_mask).sum() > 0 else 0.0

    # ---- Phase 5: Report ----
    base_rate = float(y_val.mean())
    report = {
        "train_pairs": int(len(X_train)),
        "train_cliff_ratio": float(y_train.mean()),
        "val_pairs": int(len(X_val)),
        "val_cliff_ratio": base_rate,
        "auprc": float(auprc),
        "base_rate": base_rate,
        "auc_roc": float(auc),
        "balanced_accuracy": float(bacc),
        "p_cliff_mean_cliff": float(p_cliff_on_cliff),
        "p_cliff_mean_noncliff": float(p_cliff_on_noncliff),
        "gate_auprc_above_base": bool(auprc > base_rate),
        "gate_bacc_above_060": bool(bacc > 0.60),
    }
    report["gate_passed"] = bool(report["gate_auprc_above_base"] and report["gate_bacc_above_060"])

    print(f"\n{'='*50}")
    print(f"PAIR-LEVEL VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"  Training pairs:    {len(X_train)} (cliff ratio: {y_train.mean():.3f})")
    print(f"  Validation pairs:  {len(X_val)} (cliff ratio: {y_val.mean():.3f})")
    print(f"  AUPRC:              {auprc:.4f}  (base rate: {y_val.mean():.4f})")
    print(f"  AUC-ROC:            {auc:.4f}")
    print(f"  Balanced Accuracy:  {bacc:.4f}")
    print(f"  p_cliff on cliffs:  {p_cliff_on_cliff:.4f}")
    print(f"  p_cliff on noncliffs: {p_cliff_on_noncliff:.4f}")
    print(f"  Gate AUPRC>base:    {report['gate_auprc_above_base']}")
    print(f"  Gate BAcc>0.60:     {report['gate_bacc_above_060']}")
    print(f"  GATE PASSED:        {report['gate_passed']}")

    with open(output_dir / "pair_level_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save model for later use
    import pickle
    with open(output_dir / "relation_head.pkl", "wb") as f:
        pickle.dump(head, f)

    print(f"\nModel saved to {output_dir / 'relation_head.pkl'}")
    print(f"Report saved to {output_dir / 'pair_level_report.json'}")


if __name__ == "__main__":
    main()
