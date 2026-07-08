#!/usr/bin/env python3
"""
Learning-curve / data-efficiency analysis (new results section).

Trains the primary model on increasing fractions of the training data and plots
test AUROC vs training-set size. This quantifies how much data the task needs and
directly motivates the few-shot question: does performance saturate early?

Outputs: results/canonical/learning_curve.csv (+ figure)
Needs: thesis_dataset.parquet, tokens.npy ; catboost.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; FIG = ROOT / "results/full_cohort/figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in ds.columns if c not in meta]
    X = np.nan_to_num(ds[feat].to_numpy(np.float32))
    tokens = np.load(PROC / "tokens.npy")
    Z = np.concatenate([X, tokens.reshape(len(tokens), -1)], axis=1)
    y = ds["feasible"].to_numpy(int)
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=SEED, stratify=y)

    from catboost import CatBoostClassifier
    rng = np.random.default_rng(SEED); n = len(ytr); rows = []
    for frac in FRACTIONS:
        k = max(200, int(frac * n))
        sub = rng.choice(np.arange(n), min(k, n), replace=False)
        if len(np.unique(ytr[sub])) < 2:
            continue
        m = CatBoostClassifier(iterations=300, learning_rate=0.05, auto_class_weights="Balanced",
                               random_seed=SEED, verbose=0)
        m.fit(Ztr[sub], ytr[sub])
        au = roc_auc_score(yte, m.predict_proba(Zte)[:, 1])
        rows.append({"fraction": frac, "n_train": len(sub), "AUROC": round(au, 4)})
        print(f"  frac={frac:<5} n={len(sub):<6} AUROC={au:.4f}")
    df = pd.DataFrame(rows); df.to_csv(OUT / "learning_curve.csv", index=False)

    plt.figure(figsize=(7, 4.4))
    plt.semilogx(df.n_train, df.AUROC, "o-", color="#0b3d2e")
    full = df.AUROC.iloc[-1]
    plt.axhline(full, ls="--", color="gray", label=f"full-data AUROC = {full:.3f}")
    plt.xlabel("Training-set size (log scale)"); plt.ylabel("Test AUROC")
    plt.title("Learning curve (primary model): performance saturates early")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "learning_curve.png", dpi=160); plt.close()
    print("Wrote", OUT / "learning_curve.csv")


if __name__ == "__main__":
    main()
