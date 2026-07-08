#!/usr/bin/env python3
"""Generate robustness figures from the --full / --expanded result CSVs."""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "results/canonical"
FIG = ROOT / "results/full_cohort/figures"; FIG.mkdir(parents=True, exist_ok=True)

# 1) Label sensitivity: grouped bars (AUROC per model across labels)
ls = pd.read_csv(C / "label_sensitivity.csv")
labels = ["L1_primary", "L2_mortality", "L3_home_only", "L4_strict_los", "L5_no_los"]
models = ["Logistic Regression", "Random Forest", "CatBoost", "Token-Hybrid CatBoost"]
piv = ls.pivot_table(index="label", columns="model", values="AUROC").reindex(labels)[models]
fig, ax = plt.subplots(figsize=(8.4, 4.6)); x = np.arange(len(labels)); w = 0.2
colors = ["#9aa7b1", "#6b8fb5", "#3b6ea5", "#0b3d2e"]
for i, m in enumerate(models):
    ax.bar(x + (i - 1.5) * w, piv[m], w, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(["L1 primary", "L2 mortality", "L3 home-only", "L4 strict-LOS", "L5 no-LOS"], fontsize=9)
ax.set_ylabel("Test AUROC"); ax.set_ylim(0.65, 0.9)
ax.set_title("Label-definition sensitivity: ranking is stable across 5 labels")
ax.legend(fontsize=8, ncol=2); plt.tight_layout()
plt.savefig(FIG / "robustness_label_sensitivity.png", dpi=160); plt.close()

# 2) Multi-seed: bar with std error bars
ms = pd.read_csv(C / "rigorous_multiseed.csv").sort_values("mean")
fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(ms))
ax.barh(y, ms["mean"], xerr=ms["std"], color="#3b6ea5", capsize=3)
ax.set_yticks(y); ax.set_yticklabels(ms["model"], fontsize=9); ax.set_xlim(0.62, 0.78)
ax.set_xlabel("AUROC (mean ± std over 5 seeds)")
ax.set_title("Stability across random seeds")
plt.tight_layout(); plt.savefig(FIG / "robustness_multiseed.png", dpi=160); plt.close()

# 3) ICD-9 expansion: AUROC on 32k vs 64k (same basic runner)
try:
    a = pd.read_csv(ROOT / "results/full_cohort/tables/full_cohort_model_results.csv").set_index("model")["auroc"]
    b = pd.read_csv(ROOT / "results/full_cohort_icd9/tables/full_cohort_model_results.csv").set_index("model")["auroc"]
    common = [m for m in a.index if m in b.index]
    fig, ax = plt.subplots(figsize=(8.0, 4.6)); x = np.arange(len(common)); w = 0.38
    ax.bar(x - w/2, a.loc[common], w, label="ICD-10 cohort (32,399)", color="#6b8fb5")
    ax.bar(x + w/2, b.loc[common], w, label="ICD-9+10 cohort (64,444)", color="#0b3d2e")
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0.5, 0.8)
    ax.set_title("Generalization to the doubled (ICD-9+10) cohort")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(FIG / "robustness_icd9.png", dpi=160); plt.close()
    print("wrote robustness_icd9.png")
except Exception as e:
    print("icd9 fig skipped:", e)

print("Wrote robustness figures to", FIG)
