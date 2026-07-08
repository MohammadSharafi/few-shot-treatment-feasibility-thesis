#!/usr/bin/env python3
"""Generate thesis figures + LaTeX table for the enriched-feature section.

Produces (into results/full_cohort_enriched/figures and .../tables):
  * enriched_reliability.png     — calibration/reliability curve of stacked model
  * enriched_feature_importance.png — top-20 features (LightGBM gain)
  * enriched_results_table.tex   — LaTeX rows for the results chapter
Also copies the comparison figure into results/full_cohort/figures so the thesis
graphicspath resolves it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
ENR = ROOT / "results/full_cohort_enriched"
FIG = ENR / "figures"
TAB = ENR / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Human-readable names for the 22 signals (itemid -> label)
SIGNAL = {
    50912: "Creatinine", 50971: "Potassium", 50983: "Sodium", 50902: "Chloride",
    50931: "Glucose", 50893: "Calcium", 50960: "Magnesium", 50811: "Hemoglobin",
    51265: "Platelets", 51300: "WBC", 50813: "Lactate", 50861: "ALT", 50878: "AST",
    51006: "BUN", 50882: "Bicarbonate", 220045: "HeartRate", 220050: "SBP",
    220051: "DBP", 220052: "MBP", 223761: "Temp", 220277: "SpO2", 220210: "RespRate",
}


def pretty(col: str) -> str:
    parts = col.split("_")
    if parts[0].isdigit() and int(parts[0]) in SIGNAL:
        return f"{SIGNAL[int(parts[0])]} ({'_'.join(parts[1:])})"
    return col


def main() -> int:
    d = pd.read_parquet(ROOT / "data/processed_full_cohort/enriched_dataset.parquet")
    meta = {"stay_id", "subject_id", "hadm_id", "los_hours", "primary_icd10", "feasible"}
    feat = [c for c in d.columns if c not in meta]
    X = d[feat].to_numpy(np.float32)
    y = d["feasible"].to_numpy(int)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

    model = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=48,
                               subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
                               min_child_samples=30, reg_lambda=1.5, class_weight="balanced",
                               random_state=42, n_jobs=-1, verbose=-1, importance_type="gain")
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    auroc = roc_auc_score(y[te], p)

    # Reliability curve
    frac_pos, mean_pred = calibration_curve(y[te], p, n_bins=12, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", c="#2C6FBB", label=f"Enriched model (AUROC {auroc:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability curve — enriched model"); ax.legend(loc="upper left")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / "enriched_reliability.png", dpi=300); plt.close(fig)

    # Feature importance (top 20)
    imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False).head(20)
    labels = [pretty(c) for c in imp.index][::-1]
    vals = imp.values[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.barh(range(len(vals)), vals, color="#2C6FBB")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LightGBM gain importance")
    ax.set_title("Top-20 features — enriched pipeline"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "enriched_feature_importance.png", dpi=300); plt.close(fig)

    # LaTeX results table from the enriched run
    res = pd.read_csv(TAB / "enriched_model_results.csv")
    name_map = {
        "Stacked Enriched (calibrated)": r"Stacked (enriched)\,$\star$",
        "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
        "Random Forest": "Random Forest", "Logistic Regression": "Logistic Regression",
        "Neural Baseline": "Neural Baseline",
    }
    order = ["Stacked Enriched (calibrated)", "XGBoost", "LightGBM", "CatBoost",
             "Random Forest", "Logistic Regression", "Neural Baseline"]
    lines = []
    for m in order:
        r = res[res.model == m]
        if not len(r):
            continue
        r = r.iloc[0]
        lines.append(
            f"{name_map[m]} & {r.auroc:.3f} & {r.auroc_ci_lo:.3f}--{r.auroc_ci_hi:.3f} & "
            f"{r.auprc:.3f} & {r.f1_macro:.3f} & {r.mcc:.3f} & {r.brier:.3f} & {r.ece:.3f} \\\\"
        )
    (TAB / "enriched_results_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # Copy comparison + new figures into the canonical figures dir for graphicspath
    dst = ROOT / "results/full_cohort/figures"
    for f in ["enriched_vs_baseline_auroc.png", "enriched_reliability.png",
              "enriched_feature_importance.png"]:
        if (FIG / f).exists():
            shutil.copy(FIG / f, dst / f)

    print(f"Enriched LightGBM test AUROC: {auroc:.4f}")
    print("Wrote figures + enriched_results_table.tex; copied figures to", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
