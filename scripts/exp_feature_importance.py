#!/usr/bin/env python3
"""
Feature-importance / explainability analysis (new results chapter section).

Trains the primary model on the canonical split and reports which first-24h
laboratory and vital-sign variables drive the prediction, using (a) the model's
built-in importance and (b) permutation importance (model-agnostic). If SHAP is
installed, also writes a SHAP summary. Clinically interpretable.

Outputs: results/canonical/feature_importance.csv (+ figure)
Needs: data/processed_full_cohort/thesis_dataset.parquet ; catboost ; sklearn.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; FIG = ROOT / "results/full_cohort/figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

FEATNAMES = ["Creatinine", "Potassium", "Sodium", "Chloride", "Glucose", "Calcium",
             "Magnesium", "Hemoglobin", "Platelets", "WBC", "Lactate", "ALT", "AST",
             "BUN", "Bicarbonate", "Heart Rate", "Systolic BP", "Diastolic BP",
             "Mean BP", "Temperature", "SpO2", "Resp Rate"]


def main():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in ds.columns if c not in meta]
    names = FEATNAMES[:len(feat)] if len(feat) >= len(FEATNAMES) else [f"f{i}" for i in range(len(feat))]
    X = np.nan_to_num(ds[feat].to_numpy(np.float32)); y = ds["feasible"].to_numpy(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, learning_rate=0.05, auto_class_weights="Balanced",
                           random_seed=SEED, verbose=0)
    m.fit(Xtr, ytr)
    print(f"Test AUROC = {roc_auc_score(yte, m.predict_proba(Xte)[:,1]):.4f}")

    builtin = np.asarray(m.get_feature_importance())
    perm = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
    df = pd.DataFrame({"feature": names[:len(feat)],
                       "builtin_importance": np.round(builtin[:len(feat)], 3),
                       "perm_importance_mean": np.round(perm.importances_mean[:len(feat)], 4),
                       "perm_importance_std": np.round(perm.importances_std[:len(feat)], 4)})
    df = df.sort_values("perm_importance_mean", ascending=False)
    df.to_csv(OUT / "feature_importance.csv", index=False)
    print(df.to_string(index=False))

    top = df.head(15).iloc[::-1]
    plt.figure(figsize=(7.5, 5.5))
    plt.barh(top.feature, top.perm_importance_mean, xerr=top.perm_importance_std,
             color="#3b6ea5", capsize=2)
    plt.xlabel("Permutation importance (Δ AUROC)")
    plt.title("Most predictive first-24h variables (primary model)")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=160); plt.close()

    try:
        import shap
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(Xte[:2000])
        shap.summary_plot(sv, Xte[:2000], feature_names=names[:len(feat)], show=False, max_display=15)
        plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()
        print("Wrote SHAP summary.")
    except Exception as e:
        print("(SHAP optional, skipped:", e, ")")
    print("Wrote", OUT / "feature_importance.csv")


if __name__ == "__main__":
    main()
