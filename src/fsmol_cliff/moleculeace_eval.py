"""MoleculeACE external validation: pair-level cliff diagnostics.

For each of the 30 MoleculeACE targets:
1. Compute Morgan fingerprints
2. Enumerate active-inactive pairs, filter by Tanimoto >= tau
3. Classify as cliff (gap >= delta) or noncliff
4. Build kNN/RF models on train split, score test split
5. Report cliff-specific metrics
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import morgan_fingerprint_array, require_rdkit


def evaluate_moleculeace(
    data_dir: Path,
    output_dir: Path,
    *,
    tau: float = 0.8,
    delta: float = 1.0,
    activity_col: str = "y [pEC50/pKi]",
) -> list[dict]:
    """Run pair-level cliff diagnostics on all MoleculeACE targets."""
    require_rdkit()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    csv_files = sorted(data_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "old" not in str(f.parent)]

    for csv_file in csv_files:
        target_name = csv_file.stem
        try:
            target_result = _evaluate_one_target(
                csv_file, target_name, tau=tau, delta=delta,
                activity_col=activity_col,
            )
            results.append(target_result)
        except Exception as exc:
            print(f"  {target_name}: FAILED — {type(exc).__name__}: {exc}")
            continue

    # Write aggregated results
    summary = _aggregate_results(results)
    with open(output_dir / "moleculeace_results.json", "w") as f:
        json.dump({"per_target": results, "summary": summary}, f, indent=2)

    _print_summary(results, summary)
    return results


def _evaluate_one_target(
    csv_file: Path,
    target_name: str,
    *,
    tau: float,
    delta: float,
    activity_col: str,
) -> dict:
    df = pd.read_csv(csv_file)
    print(f"  {target_name}: {len(df)} mols, cliff_mol={df.cliff_mol.sum()}")

    # Compute fingerprints
    smiles_list = df["smiles"].tolist()
    fps: dict[int, np.ndarray] = {}
    valid_indices: list[int] = []
    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or len(smi) < 2:
            continue
        fp = morgan_fingerprint_array(smi)
        if fp is not None:
            fps[i] = fp
            valid_indices.append(i)

    df_valid = df.iloc[valid_indices].copy()
    df_valid["fp"] = [fps[i] for i in valid_indices]
    activity = df_valid[activity_col].values

    # Binarize: top 50% active, bottom 50% inactive (median split)
    median_act = np.median(activity)
    df_valid["binary_label"] = (activity >= median_act).astype(int)

    # Build cliff/noncliff pairs
    train_mask = df_valid["split"] == "train"
    test_mask = df_valid["split"] == "test"

    # Compute all active-inactive pairs in train
    train_actives = df_valid[train_mask & (df_valid["binary_label"] == 1)]
    train_inactives = df_valid[train_mask & (df_valid["binary_label"] == 0)]

    n_cliff = n_noncliff = 0
    for _, active in train_actives.iterrows():
        for _, inactive in train_inactives.iterrows():
            sim = _tanimoto(active["fp"], inactive["fp"])
            if sim < tau:
                continue
            gap = abs(active[activity_col] - inactive[activity_col])
            if gap >= delta:
                n_cliff += 1
            else:
                n_noncliff += 1

    # Evaluate: kNN + RF on Morgan fingerprints
    X_train = np.stack(df_valid[train_mask]["fp"].values)
    y_train = df_valid[train_mask]["binary_label"].values
    X_test = np.stack(df_valid[test_mask]["fp"].values)
    y_test = df_valid[test_mask]["binary_label"].values

    test_activity = df_valid[test_mask][activity_col].values
    test_idx = df_valid[test_mask].index.tolist()

    models = {}
    for name, clf in [
        ("kNN", _build_knn()),
        ("RF", _build_rf()),
    ]:
        clf.fit(X_train.astype(np.float32), y_train)
        proba = clf.predict_proba(X_test.astype(np.float32))[:, 1]
        pred = (proba >= 0.5).astype(int)
        models[name] = {"proba": proba, "pred": pred}

    # Compute cliff-specific metrics on test set
    metrics = {}
    for model_name, model_data in models.items():
        pred = model_data["pred"]
        proba = model_data["proba"]

        # Enumerate test-set active-inactive pairs
        test_actives_mask = (y_test == 1)
        test_inactives_mask = (y_test == 0)

        c_bacc, nc_bacc, scr, q_psr, nc_psr = _compute_cliff_metrics(
            df_valid, test_mask, test_actives_mask, test_inactives_mask,
            pred, proba, tau, delta, activity_col,
        )
        metrics[model_name] = {
            "c_bacc": c_bacc, "nc_bacc": nc_bacc,
            "scr": scr, "q_psr": q_psr, "nc_psr": nc_psr,
        }

    return {
        "target": target_name,
        "n_molecules": len(df_valid),
        "n_cliff_pairs_train": n_cliff,
        "n_noncliff_pairs_train": n_noncliff,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "cliff_ratio": n_cliff / max(n_cliff + n_noncliff, 1),
        "metrics": metrics,
    }


def _build_knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5)


def _build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)


def _tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = float((fp_a & fp_b).sum())
    union = float((fp_a | fp_b).sum())
    return inter / union if union > 0 else 0.0


def _compute_cliff_metrics(
    df, test_mask, actives_mask, inactives_mask,
    pred, proba, tau, delta, activity_col,
) -> tuple:
    test_df = df[test_mask].copy()
    test_df["pred"] = pred
    test_df["proba"] = proba

    test_actives = test_df[actives_mask]
    test_inactives = test_df[inactives_mask]

    # Balanced accuracy
    tpr = (test_actives["pred"] == 1).mean() if len(test_actives) > 0 else 0.0
    tnr = (test_inactives["pred"] == 0).mean() if len(test_inactives) > 0 else 0.0
    bacc = (tpr + tnr) / 2.0

    # Cliff-specific BAcc
    cliff_correct = 0
    cliff_total = 0
    noncliff_correct = 0
    noncliff_total = 0
    pair_success = 0
    pair_total = 0
    collapse = 0
    collapse_total = 0
    nc_pair_success = 0
    nc_pair_total = 0

    for _, active in test_actives.iterrows():
        for _, inactive in test_inactives.iterrows():
            sim = _tanimoto(active["fp"], inactive["fp"])
            if sim < tau:
                continue
            gap = abs(active[activity_col] - inactive[activity_col])
            is_cliff = gap >= delta
            pair_correct = active["proba"] > inactive["proba"]
            both_same_pred = active["pred"] == inactive["pred"]

            if is_cliff:
                cliff_total += 1
                if active["pred"] == 1:
                    cliff_correct += 1

            pair_total += 1
            if pair_correct:
                pair_success += 1
            if both_same_pred:
                collapse_total += 1

            if not is_cliff:
                noncliff_total += 1
                if inactive["pred"] == 0:
                    noncliff_correct += 1
                if pair_correct:
                    nc_pair_success += 1

    c_bacc_val = cliff_correct / max(cliff_total, 1)
    nc_bacc_val = noncliff_correct / max(noncliff_total, 1)
    scr_val = collapse_total / max(pair_total, 1)
    q_psr_val = pair_success / max(pair_total, 1)
    nc_psr_val = nc_pair_success / max(noncliff_total, 1) if noncliff_total > 0 else 0.0

    return c_bacc_val, nc_bacc_val, scr_val, q_psr_val, nc_psr_val


def _aggregate_results(results: list[dict]) -> dict:
    """Compute macro-averaged metrics across all targets."""
    summary: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for model_name, m in r.get("metrics", {}).items():
            for metric, value in m.items():
                summary[f"{model_name}_{metric}"].append(value)
        summary["n_molecules"].append(r["n_molecules"])
        summary["n_cliff_pairs_train"].append(r["n_cliff_pairs_train"])

    agg = {}
    for key, values in summary.items():
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }
    agg["n_targets"] = len(results)
    agg["total_molecules"] = int(sum(summary["n_molecules"]))
    return agg


def _print_summary(results: list[dict], summary: dict) -> None:
    print(f"\n{'='*70}")
    print(f"MoleculeACE External Validation")
    print(f"{'='*70}")
    print(f"  Targets: {summary['n_targets']}")
    print(f"  Total molecules: {summary['total_molecules']}")
    print(f"\n  Per-model macro-averaged metrics:")
    print(f"  {'Metric':<15s} {'kNN':>10s} {'RF':>10s}")
    print(f"  {'-'*35}")
    for metric in ["c_bacc", "nc_bacc", "scr", "q_psr", "nc_psr"]:
        knn_v = summary.get(f"kNN_{metric}", {}).get("mean", 0)
        rf_v = summary.get(f"RF_{metric}", {}).get("mean", 0)
        print(f"  {metric:<15s} {knn_v:10.4f} {rf_v:10.4f}")

    # Ranking-decision mismatch check
    rf_cb = summary.get("RF_c_bacc", {}).get("mean", 0)
    rf_qp = summary.get("RF_q_psr", {}).get("mean", 0)
    rf_scr = summary.get("RF_scr", {}).get("mean", 0)
    knn_cb = summary.get("kNN_c_bacc", {}).get("mean", 0)
    knn_qp = summary.get("kNN_q_psr", {}).get("mean", 0)
    knn_scr = summary.get("kNN_scr", {}).get("mean", 0)

    print(f"\n  {'='*50}")
    print(f"  Mismatch diagnostics:")
    print(f"    RF:  Q-PSR={rf_qp:.4f}  C-BAcc={rf_cb:.4f}  SCR={rf_scr:.4f}")
    print(f"    kNN: Q-PSR={knn_qp:.4f}  C-BAcc={knn_cb:.4f}  SCR={knn_scr:.4f}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/tmp/MoleculeACE-main/MoleculeACE/Data/benchmark_data"
    )
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "outputs/moleculeace_validation"
    )
    evaluate_moleculeace(data_dir, output_dir)
