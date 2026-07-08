#!/usr/bin/env python3
"""
ROC, precision-recall and calibration curves for the model panel (new figures).

Reads the saved test predictions of the canonical run and draws three multi-model
figures used in the results chapter. No retraining (fast).

Outputs: results/full_cohort/figures/{roc_curves,pr_curves,calibration_curves_all}.png
Needs: results/final_optimized/{tables/final_model_results.csv, predictions/*.csv}
"""
from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FOPT = ROOT / "results/final_optimized/tables/final_model_results.csv"
PRED = ROOT / "results/final_optimized/predictions"
FIG = ROOT / "results/full_cohort/figures"; FIG.mkdir(parents=True, exist_ok=True)

PANEL = [("Token-Hybrid CatBoost", "CatBoost", "tabular_token"),
         ("Stacked Ensemble", "Stacked Validation Meta-LR", "mixed"),
         ("XGBoost", "XGBoost", "tabular"),
         ("Random Forest", "Random Forest", "tabular"),
         ("LightGBM", "LightGBM", "tabular"),
         ("Logistic Regression", "Logistic Regression", "tabular")]
COLORS = ["#0b3d2e", "#3b6ea5", "#b5384d", "#d98a36", "#6b8fb5", "#9aa7b1"]


def roc_points(y, p):
    o = np.argsort(-p); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / tp[-1]; fpr = fp / fp[-1]
    return np.r_[0, fpr], np.r_[0, tpr]


def pr_points(y, p):
    o = np.argsort(-p); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / (tp + fp); rec = tp / y.sum()
    return rec, prec


def resolve(exp_id):
    for c in (exp_id, f"selected_{exp_id}"):
        p = PRED / f"{c}_test_predictions.csv"
        if p.exists():
            return p
    hits = glob.glob(str(PRED / f"*{exp_id}*_test_predictions.csv"))
    return Path(hits[0]) if hits else None


def main():
    fr = pd.read_csv(FOPT)
    loaded = []
    for disp, model, feat in PANEL:
        sub = fr[(fr.model == model) & (fr.feature_set == feat)].sort_values("AUROC", ascending=False)
        if sub.empty:
            continue
        pf = resolve(sub.iloc[0]["experiment_id"])
        if pf is None:
            continue
        d = pd.read_csv(pf)
        loaded.append((disp, d.y_true.values.astype(int), d.probability.values))

    # ROC
    plt.figure(figsize=(5.6, 5.4))
    for (disp, y, p), c in zip(loaded, COLORS):
        fpr, tpr = roc_points(y, p)
        plt.plot(fpr, tpr, color=c, lw=1.6, label=disp)
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "roc_curves.png", dpi=160); plt.close()

    # PR
    plt.figure(figsize=(5.6, 5.4))
    for (disp, y, p), c in zip(loaded, COLORS):
        rec, prec = pr_points(y, p)
        plt.plot(rec, prec, color=c, lw=1.6, label=disp)
    plt.axhline(loaded[0][1].mean(), ls="--", color="gray", label="prevalence")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-recall curves"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "pr_curves.png", dpi=160); plt.close()

    # Calibration
    plt.figure(figsize=(5.6, 5.4))
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect")
    for (disp, y, p), c in zip(loaded, COLORS):
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 20:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        plt.plot(xs, ys, "o-", color=c, lw=1.4, ms=3, label=disp)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration curves"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "calibration_curves_all.png", dpi=160); plt.close()
    print("Wrote roc_curves.png, pr_curves.png, calibration_curves_all.png to", FIG)


if __name__ == "__main__":
    main()
