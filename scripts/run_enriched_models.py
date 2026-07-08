#!/usr/bin/env python3
"""Train full-cohort models on the enriched feature set with OOF stacking + calibration.

Improvements over `run_full_cohort_models.py`:
  * Uses `enriched_dataset.parquet` (206 features: multi-statistic per-signal
    aggregates + missingness flags + demographics + comorbidity burden) instead
    of the 25-feature median-only table.
  * Proper out-of-fold (OOF) stacking: base-learner meta-features come from
    cross_val_predict on the training fold, so the meta-learner never sees a
    base model's in-sample predictions (no stacking leakage).
  * Probability calibration (isotonic) fit on an internal calibration split, and
    calibration quality reported (Brier score, expected calibration error).
  * Bootstrap 95% CIs for AUROC.

Outputs to results/full_cohort_enriched/ (kept separate from the median-only
canonical results until validated).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
META_COLS = {"stay_id", "subject_id", "hadm_id", "los_hours", "primary_icd10", "feasible", "protocol"}


def load_config() -> dict:
    with open(ROOT / os.environ.get("THESIS_CONFIG", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def load_data(cfg: dict, dataset: str = "enriched_dataset.parquet"):
    d = pd.read_parquet(resolve(cfg, "processed_dir") / dataset)
    feat = [c for c in d.columns if c not in META_COLS]
    X = d[feat].to_numpy(dtype=np.float32)
    y = d["feasible"].to_numpy(dtype=int)
    return d, X, y, feat


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(probs[m].mean() - y_true[m].mean())
    return float(ece)


def tune_threshold(y_val, probs_val) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.25, 0.75, 101):
        f1 = f1_score(y_val, (probs_val >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def metric_row(name, y_true, probs, threshold=0.5, seed=42, extra=None) -> dict:
    pred = (probs >= threshold).astype(int)
    row = {
        "model": name,
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
    }
    rng = np.random.RandomState(seed)
    idx_all = np.arange(len(y_true))
    boots = []
    for _ in range(1000):
        idx = rng.choice(idx_all, len(idx_all), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], probs[idx]))
    if boots:
        row["auroc_ci_lo"] = float(np.percentile(boots, 2.5))
        row["auroc_ci_hi"] = float(np.percentile(boots, 97.5))
    if extra:
        row.update(extra)
    return row


def base_factories(seed: int) -> dict:
    """Base learners. Tree models get raw features (NaN handled natively where
    supported); linear/MLP get median-imputed + scaled features."""
    def imputed(est):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("clf", est)])

    def tree_imputed(est):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", est)])

    fac = {
        "Logistic Regression": lambda: imputed(
            LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=seed)),
        "Random Forest": lambda: tree_imputed(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        "Neural Baseline": lambda: imputed(MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300,
            early_stopping=True, random_state=seed)),
    }
    try:
        import xgboost as xgb
        fac["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.5,
            eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1)
    except Exception:
        pass
    try:
        import lightgbm as lgb
        fac["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.03, num_leaves=48, max_depth=-1,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.7, min_child_samples=30,
            reg_lambda=1.5, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)
    except Exception:
        pass
    try:
        import catboost as cb
        fac["CatBoost"] = lambda: cb.CatBoostClassifier(
            iterations=700, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
            auto_class_weights="Balanced", random_seed=seed, verbose=0,
            allow_writing_files=False, thread_count=-1)
    except Exception:
        pass
    return fac


def predict_pos(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    z = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/full_cohort_enriched")
    ap.add_argument("--dataset", default="enriched_dataset.parquet",
                    help="parquet under processed_dir to model (default: enriched)")
    ap.add_argument("--tag", default="enriched", help="filename tag for outputs")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    out_dir = ROOT / args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    data, X, y, feat = load_data(cfg, args.dataset)
    print(f"Dataset: {args.dataset} — {X.shape[0]} rows x {X.shape[1]} features; feasible={y.mean():.3f}")

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                           random_state=seed, stratify=y)
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    factories = base_factories(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows, statuses = [], []
    oof = {}      # OOF probs on train (for meta-learner)
    test_pf = {}  # test probs from full-train refit (for meta-learner + individual metrics)

    # small internal validation slice for threshold tuning
    fit_idx, val_idx = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                        random_state=seed, stratify=ytr)

    for name, factory in factories.items():
        t0 = time.time()
        try:
            oof_p = cross_val_predict(factory(), Xtr, ytr, cv=skf,
                                      method="predict_proba", n_jobs=1)[:, 1]
            oof[name] = oof_p
            model = factory(); model.fit(Xtr, ytr)
            test_pf[name] = predict_pos(model, Xte)
            thr = tune_threshold(ytr[val_idx], oof_p[val_idx])
            row = metric_row(name, yte, test_pf[name], thr, seed)
            row["fit_seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            statuses.append({"model": name, "status": "completed", "seconds": row["fit_seconds"]})
            print(f"{name:20s} AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} "
                  f"Brier={row['brier']:.4f} ECE={row['ece']:.4f}", flush=True)
        except Exception as exc:
            statuses.append({"model": name, "status": "failed", "error": str(exc)})
            print(f"{name}: FAILED {exc}", flush=True)

    # ---- OOF-stacked ensemble + isotonic calibration ----
    base_names = [n for n in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
                  if n in oof]
    if len(base_names) >= 2:
        t0 = time.time()
        Z_tr = np.column_stack([oof[n] for n in base_names])
        Z_te = np.column_stack([test_pf[n] for n in base_names])
        # split train meta-features for calibration
        m_fit, m_cal = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                        random_state=seed, stratify=ytr)
        meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        meta.fit(Z_tr[m_fit], ytr[m_fit])
        try:  # sklearn >= 1.6 removed cv="prefit" in favour of FrozenEstimator
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(meta), method="isotonic")
        except Exception:
            cal = CalibratedClassifierCV(meta, method="isotonic", cv="prefit")
        cal.fit(Z_tr[m_cal], ytr[m_cal])
        probs = cal.predict_proba(Z_te)[:, 1]
        thr = tune_threshold(ytr[m_cal], cal.predict_proba(Z_tr[m_cal])[:, 1])
        row = metric_row("Stacked Enriched (calibrated)", yte, probs, thr, seed,
                         extra={"base_models": "; ".join(base_names)})
        row["fit_seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        statuses.append({"model": "Stacked Enriched (calibrated)", "status": "completed",
                         "base_models": base_names, "seconds": row["fit_seconds"]})
        print(f"{'Stacked Enriched':20s} AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} "
              f"Brier={row['brier']:.4f} ECE={row['ece']:.4f}", flush=True)

    results = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    results.to_csv(out_dir / "tables" / f"{args.tag}_model_results.csv", index=False)
    with open(out_dir / f"{args.tag}_model_run.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_rows": int(len(data)),
            "n_features": int(X.shape[1]),
            "feature_names": feat,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "statuses": statuses,
            "results": results.to_dict(orient="records"),
        }, f, indent=2)
    print("\n" + results[["model", "auroc", "auprc", "f1_macro", "brier", "ece"]].to_string(index=False))
    print(f"\nSaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
