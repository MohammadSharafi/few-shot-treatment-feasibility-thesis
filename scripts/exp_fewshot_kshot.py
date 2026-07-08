#!/usr/bin/env python3
"""
Few-shot K-shot analysis (new results section; directly on the thesis title).

Evaluates a prototype-based few-shot classifier on the Symptom-Token embeddings as
a function of the number of support examples per class (K = 1, 5, 10, 20, 50, 100).
This characterises exactly how the few-shot approach behaves as more labelled
examples become available, and where (if anywhere) it is competitive.

Outputs: results/canonical/fewshot_kshot.csv (+ figure)
Needs: tokens.npy, thesis_dataset.parquet (for labels).
Pure numpy/sklearn — no deep learning required (raw-token prototype baseline).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; FIG = ROOT / "results/full_cohort/figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
KS = [1, 5, 10, 20, 50, 100]
N_EPISODES = 200


def prototype_auroc(emb_support, y_support, emb_query, y_query):
    """Distance-to-prototype classifier; returns AUROC on the query set."""
    protos = {c: emb_support[y_support == c].mean(0) for c in [0, 1]}
    d0 = np.linalg.norm(emb_query - protos[0], axis=1)
    d1 = np.linalg.norm(emb_query - protos[1], axis=1)
    score = d0 - d1                       # higher => closer to class 1
    return roc_auc_score(y_query, score)


def main():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    y = ds["feasible"].to_numpy(int)
    tokens = np.load(PROC / "tokens.npy").reshape(len(y), -1)
    emb = StandardScaler().fit_transform(tokens)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    Etr, ytr = emb[itr], y[itr]
    Ete, yte = emb[ite], y[ite]
    rng = np.random.default_rng(SEED)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]

    rows = []
    for K in KS:
        aucs = []
        for _ in range(N_EPISODES):
            sp = rng.choice(pos, K, replace=False); sn = rng.choice(neg, K, replace=False)
            sup = np.concatenate([sp, sn]); ys = ytr[sup]
            aucs.append(prototype_auroc(Etr[sup], ys, Ete, yte))
        rows.append({"K": K, "AUROC_mean": round(np.mean(aucs), 4),
                     "AUROC_std": round(np.std(aucs), 4)})
        print(f"  K={K:<4} AUROC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    df = pd.DataFrame(rows); df.to_csv(OUT / "fewshot_kshot.csv", index=False)

    plt.figure(figsize=(7, 4.4))
    plt.errorbar(df.K, df.AUROC_mean, yerr=df.AUROC_std, fmt="o-", color="#b5384d", capsize=3,
                 label="Few-shot prototype (tokens)")
    plt.axhline(0.756, ls="--", color="#0b3d2e", label="Token-Hybrid CatBoost (full data)")
    plt.xscale("log"); plt.xlabel("Support examples per class (K, log scale)")
    plt.ylabel("Test AUROC"); plt.title("Few-shot performance vs. number of shots")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "fewshot_kshot.png", dpi=160); plt.close()
    print("Wrote", OUT / "fewshot_kshot.csv")


if __name__ == "__main__":
    main()
