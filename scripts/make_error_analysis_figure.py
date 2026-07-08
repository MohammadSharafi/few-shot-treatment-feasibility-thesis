#!/usr/bin/env python3
"""Error-analysis figure for the primary model: predicted-probability distributions
by true outcome, with the ambiguous decision band highlighted. Real test predictions."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "full_cohort", "figures")
PRED = os.path.join(ROOT, "results", "full_cohort", "predictions",
                    "selected_opt_catboost_tabular_token_008_test_predictions.csv")
C = dict(feas="#2E7D32", infeas="#B23A3A", band="#9aa7b2", ink="#22303C")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

d = pd.read_csv(PRED)
y = d.y_true.to_numpy(); p = d.probability.to_numpy()
bins = np.linspace(0, 1, 41)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axvspan(0.40, 0.60, color=C["band"], alpha=0.22, label="ambiguous band (0.40–0.60)")
ax.hist(p[y == 1], bins=bins, color=C["feas"], alpha=0.62, label="Feasible (true)")
ax.hist(p[y == 0], bins=bins, color=C["infeas"], alpha=0.58, label="Infeasible (true)")
ax.axvline(0.535, color=C["ink"], lw=1.8, ls="--", label="operating threshold (0.535)")
ax.set_xlabel("Predicted probability of feasibility")
ax.set_ylabel("Number of ICU stays (test set)")
ax.set_title("Where the model is confident — and where it is not", color=C["ink"], fontweight="bold")
ax.legend(loc="upper center", fontsize=9.5)
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "error_analysis_distributions.png"), dpi=300, bbox_inches="tight")
print("saved error_analysis_distributions.png")
