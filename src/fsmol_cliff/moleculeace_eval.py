"""MoleculeACE external pair-level diagnostic.

For each of the 30 MoleculeACE targets, computes cliff-specific metrics
using pair-balanced decision accuracy.  Follows the corrected definitions:

  C-BAcc  = mean(pair_decision_acc) over cliff pairs (gap >= delta)
  NC-BAcc = mean(pair_decision_acc) over noncliff pairs (gap < delta)
  SCR     = fraction of high-sim pairs where both molecules get the same
            binary prediction.
  Q-PSR   = fraction of high-sim pairs where active_score > inactive_score.
  NC-PSR  = Q-PSR restricted to noncliff pairs.

Legacy one-sided metrics are renamed:
  C-ActiveAcc      = fraction of cliff-pair active molecules predicted active.
  NC-InactiveAcc   = fraction of noncliff-pair inactive mols predicted inactive.

Targets with zero eligible test pairs are excluded from macro-aggregation
on a per-metric basis (not padded to zero).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

from .chem import morgan_fingerprint_array, murcko_scaffold_smiles, require_rdkit


def evaluate_moleculeace_v2(
    data_dir: Path,
    output_dir: Path,
    *,
    tau: float = 0.8,
    delta: float = 1.0,
    activity_col: str = "y [pEC50/pKi]",
    n_bootstrap: int = 2000,
) -> dict:
    """Run corrected pair-level diagnostics on all MoleculeACE targets."""
    require_rdkit()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "old" not in str(f.parent)]

    per_target: list[dict] = []
    for csv_file in csv_files:
        target_name = csv_file.stem
        row = _evaluate_one_target_v2(csv_file, target_name, tau=tau, delta=delta, activity_col=activity_col)
        if row is not None:
            per_target.append(row)

    summary = _build_summary_v2(per_target, n_bootstrap=n_bootstrap)

    # ---- Write outputs ----
    with open(output_dir / "moleculeace_results_v2.json", "w") as f:
        json.dump({"per_target": per_target, "summary": summary}, f, indent=2)

    df_tgt = pd.DataFrame(per_target)
    df_tgt.to_csv(output_dir / "moleculeace_per_target_v2.csv", index=False)

    rows_sm = []
    for model_name, metrics in summary.items():
        if model_name.startswith("_"):
            continue
        for metric_name, vals in metrics.items():
            rows_sm.append({
                "model": model_name,
                "metric": metric_name,
                **vals,
            })
    pd.DataFrame(rows_sm).to_csv(output_dir / "moleculeace_summary_v2.csv", index=False)

    with open(output_dir / "moleculeace_bootstrap_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary_v2(per_target, summary)
    return {"per_target": per_target, "summary": summary}


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def _evaluate_one_target_v2(
    csv_file: Path,
    target_name: str,
    *,
    tau: float,
    delta: float,
    activity_col: str,
) -> dict | None:
    df = pd.read_csv(csv_file)

    # --- fingerprints ---
    fp_list: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, smi in enumerate(df["smiles"]):
        if not isinstance(smi, str) or len(smi) < 2:
            continue
        fp = morgan_fingerprint_array(smi)
        if fp is not None:
            fp_list.append(fp)
            valid_idx.append(i)

    if len(valid_idx) < 10:
        return None

    fp_all = fp_list
    act = df.iloc[valid_idx][activity_col].values.astype(np.float64)
    split = df.iloc[valid_idx]["split"].values

    median_act = np.median(act)
    labels = (act >= median_act).astype(np.int64)

    train_mask = split == "train"
    test_mask = split == "test"

    if train_mask.sum() < 10 or test_mask.sum() < 5:
        return None

    X_tr = np.stack([f.astype(np.float32) for f, m in zip(fp_all, train_mask) if m])
    y_tr = labels[train_mask]
    X_te = np.stack([f.astype(np.float32) for f, m in zip(fp_all, test_mask) if m])
    y_te = labels[test_mask]
    act_te = act[test_mask]
    fp_te = [f for f, m in zip(fp_all, test_mask) if m]

    test_act_mask = y_te == 1
    test_inact_mask = y_te == 0

    # --- scaffolds ---
    scaffolds: set[str] = set()
    for smi in df["smiles"]:
        scf = murcko_scaffold_smiles(smi)
        if scf:
            scaffolds.add(scf)

    # --- train-set pairs (cliff + noncliff) ---
    train_act_mask = labels[train_mask] == 1
    train_inact_mask = labels[train_mask] == 0
    train_act_fp = [f for f, m in zip(fp_all, train_mask) if m and labels[train_mask][list(train_mask).index(True)] == 1]
    # Actually compute properly:
    fp_train_all = [f for f, m in zip(fp_all, train_mask) if m]
    y_tr_list = y_tr.tolist()
    train_act_fp_list = [fp_train_all[i] for i in range(len(y_tr)) if y_tr_list[i] == 1]
    train_inact_fp_list = [fp_train_all[i] for i in range(len(y_tr)) if y_tr_list[i] == 0]
    act_tr = act[train_mask]
    train_act_act_list = [act_tr[i] for i in range(len(y_tr)) if y_tr_list[i] == 1]
    train_inact_act_list = [act_tr[i] for i in range(len(y_tr)) if y_tr_list[i] == 0]

    n_train_cliff = 0
    n_train_noncliff = 0
    for ai, (fp_a, act_a) in enumerate(zip(train_act_fp_list, train_act_act_list)):
        for ii, (fp_i, act_i) in enumerate(zip(train_inact_fp_list, train_inact_act_list)):
            sim = _tanimoto(fp_a, fp_i)
            if sim < tau:
                continue
            gap = abs(act_a - act_i)
            if gap >= delta:
                n_train_cliff += 1
            else:
                n_train_noncliff += 1

    # --- test-set pairs ---
    test_act_indices = [i for i in range(len(y_te)) if test_act_mask[i]]
    test_inact_indices = [i for i in range(len(y_te)) if test_inact_mask[i]]

    n_test_cliff = 0
    n_test_noncliff = 0
    for ai in test_act_indices:
        for ii in test_inact_indices:
            sim = _tanimoto(fp_te[ai], fp_te[ii])
            if sim < tau:
                continue
            gap = abs(act_te[ai] - act_te[ii])
            if gap >= delta:
                n_test_cliff += 1
            else:
                n_test_noncliff += 1

    row: dict = {
        "target": target_name,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_molecules": len(valid_idx),
        "n_scaffolds": len(scaffolds),
        "train_cliff_pairs": n_train_cliff,
        "train_highsim_noncliff_pairs": n_train_noncliff,
        "test_cliff_pairs": n_test_cliff,
        "test_highsim_noncliff_pairs": n_test_noncliff,
        "test_highsim_pairs": n_test_cliff + n_test_noncliff,
    }

    # --- models ---
    for clf_name, clf in [
        ("kNN", KNeighborsClassifier(n_neighbors=5)),
        ("RF", RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)),
    ]:
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)

        _compute_pair_metrics_v2(
            row, clf_name, fp_te, act_te, y_te, pred, proba,
            tau=tau, delta=delta,
        )

    return row


def _compute_pair_metrics_v2(
    row: dict,
    clf_name: str,
    fp_te: list[np.ndarray],
    act_te: np.ndarray,
    y_te: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    *,
    tau: float,
    delta: float,
) -> None:
    test_act_mask = y_te == 1
    test_inact_mask = y_te == 0
    test_act_indices = [i for i in range(len(y_te)) if test_act_mask[i]]
    test_inact_indices = [i for i in range(len(y_te)) if test_inact_mask[i]]

    # Accumulators
    cliff_pd_acc: list[float] = []       # pair_decision_acc on cliff pairs
    noncliff_pd_acc: list[float] = []    # pair_decision_acc on noncliff pairs
    cliff_active_acc: list[int] = []     # 1(active_pred==1) on cliff pairs
    noncliff_inactive_acc: list[int] = [] # 1(inactive_pred==0) on noncliff pairs
    pair_success: list[int] = []         # 1 for all high-sim pairs
    nc_pair_success: list[int] = []      # 1 for noncliff pairs only
    collapse: list[int] = []             # 1(active_pred == inactive_pred) for all

    for ai in test_act_indices:
        for ii in test_inact_indices:
            sim = _tanimoto(fp_te[ai], fp_te[ii])
            if sim < tau:
                continue
            gap = abs(act_te[ai] - act_te[ii])
            is_cliff = gap >= delta

            pd = 0.5 * (float(pred[ai] == 1) + float(pred[ii] == 0))
            if is_cliff:
                cliff_pd_acc.append(pd)
                cliff_active_acc.append(int(pred[ai] == 1))
            else:
                noncliff_pd_acc.append(pd)
                noncliff_inactive_acc.append(int(pred[ii] == 0))
                if proba[ai] > proba[ii]:
                    nc_pair_success.append(1)
                else:
                    nc_pair_success.append(0)

            if proba[ai] > proba[ii]:
                pair_success.append(1)
            else:
                pair_success.append(0)

            if pred[ai] == pred[ii]:
                collapse.append(1)
            else:
                collapse.append(0)

    row[f"{clf_name}_C-BAcc"] = _safe_mean(cliff_pd_acc)
    row[f"{clf_name}_NC-BAcc"] = _safe_mean(noncliff_pd_acc)
    row[f"{clf_name}_SCR"] = _safe_mean(collapse)
    row[f"{clf_name}_Q-PSR"] = _safe_mean(pair_success)
    row[f"{clf_name}_NC-PSR"] = _safe_mean(nc_pair_success)
    row[f"{clf_name}_C-ActiveAcc"] = _safe_mean(cliff_active_acc)
    row[f"{clf_name}_NC-InactiveAcc"] = _safe_mean(noncliff_inactive_acc)
    row[f"{clf_name}_n_cliff_pairs"] = len(cliff_pd_acc)
    row[f"{clf_name}_n_noncliff_pairs"] = len(noncliff_pd_acc)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _build_summary_v2(per_target: list[dict], n_bootstrap: int = 2000) -> dict:
    summary: dict = {}
    model_names = ["kNN", "RF"]
    metrics = [
        "C-BAcc", "NC-BAcc", "SCR", "Q-PSR", "NC-PSR",
        "C-ActiveAcc", "NC-InactiveAcc",
    ]
    eligibility = {
        "C-BAcc": "test_cliff_pairs >= 1",
        "NC-BAcc": "test_highsim_noncliff_pairs >= 1",
        "SCR": "test_highsim_pairs >= 1",
        "Q-PSR": "test_highsim_pairs >= 1",
        "NC-PSR": "test_highsim_noncliff_pairs >= 1",
        "C-ActiveAcc": "test_cliff_pairs >= 1",
        "NC-InactiveAcc": "test_highsim_noncliff_pairs >= 1",
    }

    for model_name in model_names:
        summary[model_name] = {}
        for metric in metrics:
            col = f"{model_name}_{metric}"
            vals = np.array([r[col] for r in per_target if not np.isnan(r[col])])
            n_eligible = len(vals)
            if n_eligible == 0:
                summary[model_name][metric] = {
                    "eligible_targets": 0, "mean": None, "ci_low": None,
                    "ci_high": None, "aggregation": "macro_mean",
                    "eligibility_rule": eligibility.get(metric, ""),
                }
                continue

            mean_val = float(np.mean(vals))
            bs_means: list[float] = []
            for _ in range(n_bootstrap):
                idx = resample(range(n_eligible))
                bs_means.append(float(np.mean(vals[idx])))
            bs_means_arr = np.array(bs_means)
            ci_low = float(np.percentile(bs_means_arr, 2.5))
            ci_high = float(np.percentile(bs_means_arr, 97.5))

            summary[model_name][metric] = {
                "eligible_targets": n_eligible,
                "mean": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "aggregation": "macro_mean",
                "eligibility_rule": eligibility.get(metric, ""),
            }

    # sensitivity: count targets at different cliff-pair thresholds
    n_ge1 = sum(1 for r in per_target if r["test_cliff_pairs"] >= 1)
    n_ge3 = sum(1 for r in per_target if r["test_cliff_pairs"] >= 3)
    n_ge5 = sum(1 for r in per_target if r["test_cliff_pairs"] >= 5)
    summary["_sensitivity"] = {
        "total_targets": len(per_target),
        "with_ge1_test_cliff_pair": n_ge1,
        "with_ge3_test_cliff_pairs": n_ge3,
        "with_ge5_test_cliff_pairs": n_ge5,
        "with_ge1_test_highsim_pair": sum(1 for r in per_target if r["test_highsim_pairs"] >= 1),
    }

    return summary


def _print_summary_v2(per_target: list[dict], summary: dict) -> None:
    print(f"\n{'='*80}")
    print("MoleculeACE External Pair-Level Diagnostic (v2 — corrected metrics)")
    print(f"{'='*80}")
    print(f"  Targets: {len(per_target)}")
    print(f"  Sensitivity: {summary['_sensitivity']}")
    print()

    for model_name in ["kNN", "RF"]:
        print(f"  {model_name}:")
        print(f"  {'Metric':<18s} {'Eligible':>8s} {'Mean':>8s} {'95% CI':>24s}")
        print(f"  {'-'*58}")
        for metric, vals in summary[model_name].items():
            if vals.get("mean") is None:
                continue
            print(f"  {metric:<18s} {vals['eligible_targets']:8d} {vals['mean']:8.4f}  [{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]")
        print()

    # Mismatch diagnostics
    cb_knn = [r for r in per_target if not np.isnan(r.get("kNN_C-BAcc", np.nan))]
    cb_rf = [r for r in per_target if not np.isnan(r.get("RF_C-BAcc", np.nan))]
    if cb_knn and cb_rf:
        # Match targets
        common = [r for r in per_target if not np.isnan(r.get("kNN_C-BAcc", np.nan)) and not np.isnan(r.get("RF_C-BAcc", np.nan))]
        rf_q_better = sum(1 for r in common if r["RF_Q-PSR"] > r["kNN_Q-PSR"])
        knn_c_better = sum(1 for r in common if r["kNN_C-BAcc"] > r["RF_C-BAcc"])
        mismatch = sum(1 for r in common if r["RF_Q-PSR"] > r["kNN_Q-PSR"] and r["kNN_C-BAcc"] > r["RF_C-BAcc"])
        print(f"  Mismatch (targets with both metrics):")
        print(f"    RF Q-PSR > kNN Q-PSR:  {rf_q_better}/{len(common)}")
        print(f"    kNN C-BAcc > RF C-BAcc: {knn_c_better}/{len(common)}")
        print(f"    Simultaneous mismatch:   {mismatch}/{len(common)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    inter = float((fp_a & fp_b).sum())
    union = float((fp_a | fp_b).sum())
    return inter / union if union > 0 else 0.0


def _safe_mean(values: list) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/tmp/MoleculeACE-main/MoleculeACE/Data/benchmark_data"
    )
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "outputs/moleculeace_validation"
    )
    evaluate_moleculeace_v2(data_dir, output_dir)
