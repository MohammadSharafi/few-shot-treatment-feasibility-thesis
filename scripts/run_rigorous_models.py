#!/usr/bin/env python3
"""
Rigorous training (thesis robustness pillar #3): multi-seed repeats, 5-fold
cross-validation, and optional hyperparameter tuning.

This complements the single-split canonical run by quantifying variance and
showing the ranking is stable under (a) different random seeds and (b) k-fold CV,
and optionally (c) tuned hyperparameters rather than fixed defaults.

  --seeds N      number of random seeds (default 5)
  --tune         enable RandomizedSearchCV per tree model (slower)
  --cv-folds K   CV folds for the CV-AUROC column (default 5)

Outputs: results/canonical/rigorous_multiseed.csv  (mean +/- std per model)
Run where scikit-learn / xgboost / lightgbm / catboost are installed.
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)


def base_models(seed):
    m = {
        "Logistic Regression": make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=300,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "Neural Baseline": make_pipeline(StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        m["CatBoost"] = CatBoostClassifier(iterations=300, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=seed, verbose=0)
    except Exception:
        pass
    return m


TUNE_SPACES = {
    "XGBoost": {"max_depth": [3, 4, 5, 6], "learning_rate": [0.02, 0.05, 0.1],
                "n_estimators": [200, 300, 500], "subsample": [0.8, 0.9, 1.0]},
    "LightGBM": {"num_leaves": [15, 31, 63], "learning_rate": [0.02, 0.05, 0.1],
                 "n_estimators": [200, 300, 500]},
    "CatBoost": {"depth": [4, 6, 8], "learning_rate": [0.02, 0.05, 0.1]},
}


def load_xy():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in ds.columns if c not in meta]
    X = np.nan_to_num(ds[feat].to_numpy(np.float32))
    y = ds["feasible"].to_numpy(int)
    tokens = np.load(PROC / "tokens.npy")
    hybrid = np.concatenate([X, tokens.reshape(len(tokens), -1)], axis=1)
    return X, hybrid, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    X, hybrid, y = load_xy()
    seeds = [7, 42, 2026, 13, 99][: args.seeds]
    records = []

    def run_panel(features, tag):
        for seed in seeds:
            Xtr, Xte, ytr, yte = train_test_split(features, y, test_size=0.2, random_state=seed, stratify=y)
            for name, model in base_models(seed).items():
                if args.tune and name in TUNE_SPACES:
                    search = RandomizedSearchCV(model, TUNE_SPACES[name], n_iter=8,
                        scoring="roc_auc", cv=3, random_state=seed, n_jobs=-1)
                    search.fit(Xtr, ytr); est = search.best_estimator_
                else:
                    est = model; est.fit(Xtr, ytr)
                p = est.predict_proba(Xte)[:, 1]
                records.append({"feature_set": tag, "model": name, "seed": seed,
                                "AUROC": roc_auc_score(yte, p)})
            print(f"  [{tag}] seed {seed} done")

    run_panel(X, "tabular")
    # token-hybrid for CatBoost specifically (the proposed primary model)
    try:
        from catboost import CatBoostClassifier
        for seed in seeds:
            Htr, Hte, ytr, yte = train_test_split(hybrid, y, test_size=0.2, random_state=seed, stratify=y)
            m = CatBoostClassifier(iterations=300, learning_rate=0.05, auto_class_weights="Balanced",
                                   random_seed=seed, verbose=0)
            m.fit(Htr, ytr); p = m.predict_proba(Hte)[:, 1]
            records.append({"feature_set": "tabular_token", "model": "Token-Hybrid CatBoost",
                            "seed": seed, "AUROC": roc_auc_score(yte, p)})
    except Exception as exc:
        print("token-hybrid skipped:", exc)

    df = pd.DataFrame(records)
    summ = df.groupby(["feature_set", "model"])["AUROC"].agg(["mean", "std", "min", "max"]).round(4).reset_index()

    # 5-fold CV AUROC on a representative seed for the strong models
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    cv_rows = []
    for name, model in base_models(42).items():
        try:
            s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_rows.append({"model": name, "cv_auroc_mean": round(s.mean(), 4), "cv_auroc_std": round(s.std(), 4)})
        except Exception as exc:
            print(f"CV {name} skipped:", exc)
    cv_df = pd.DataFrame(cv_rows)

    summ.to_csv(OUT / "rigorous_multiseed.csv", index=False)
    cv_df.to_csv(OUT / "rigorous_cv.csv", index=False)
    print("\n=== multi-seed AUROC (mean +/- std) ===")
    print(summ.to_string(index=False))
    print("\n=== 5-fold CV AUROC ===")
    print(cv_df.to_string(index=False))
    print("\nWrote", OUT / "rigorous_multiseed.csv", "and", OUT / "rigorous_cv.csv")


if __name__ == "__main__":
    main()
