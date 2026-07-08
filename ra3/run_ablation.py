"""Ablation study: does each RA3 component actually contribute?

Compares full RA3 against three ablations at a fixed noise scale, across seeds:
  * uniform_lambda : same protection for every feature (removes risk-adaptivity)
  * no_exposure    : allocate by 1/utility only (removes the re-identification signal)
  * no_surrogate   : additive noise only (removes the class-conditional surrogate)

Reports mean +/- std of utility (both tasks) and the three attacks.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from data import build_broker_data, train_holdout_split
from mechanisms import RA3
from utility import downstream_utility
from attacks import run_all_attacks

OUT = Path("results/ra3"); OUT.mkdir(parents=True, exist_ok=True)
SENSITIVE_SOURCE = "50813_max"
NOISE = 1.0
SEEDS = (42, 7, 123)


def sensitive(bd, idx):
    v = bd.X[SENSITIVE_SOURCE].to_numpy()[idx]
    return (v > np.nanmedian(v)).astype(int)


def variants(seed):
    return [
        ("RA3-full", RA3(noise=NOISE, seed=seed)),
        ("RA3-uniform_lambda", RA3(noise=NOISE, seed=seed, ablate="uniform_lambda")),
        ("RA3-no_exposure", RA3(noise=NOISE, seed=seed, ablate="no_exposure")),
        ("RA3-no_surrogate", RA3(noise=NOISE, seed=seed, ablate="no_surrogate")),
    ]


def main(n_sample=12000):
    rows = []
    for seed in SEEDS:
        bd = build_broker_data(n_sample=n_sample, seed=seed)
        tr, te = train_holdout_split(bd, seed=seed)
        nonmem = np.where(~bd.member_mask)[0]
        Xtr = bd.X.iloc[tr].reset_index(drop=True); ytr = bd.y[tr]
        Xte = bd.X.iloc[te].reset_index(drop=True); yte = bd.y[te]
        y2tr, y2te = bd.y_secondary[tr], bd.y_secondary[te]
        mem = Xtr.copy(); non = bd.X.iloc[nonmem].reset_index(drop=True)
        m = min(len(mem), len(non))
        sens = sensitive(bd, tr)
        for name, mech in variants(seed):
            t0 = time.time()
            rel = mech.fit_transform(Xtr, bd.qi_cols, bd.clinical_cols, y=ytr)
            u = downstream_utility(rel, ytr, Xte, yte, bd.feature_names, seed=seed)
            u2 = downstream_utility(rel, y2tr, Xte, y2te, bd.feature_names, seed=seed)
            a = run_all_attacks(rel, mem.iloc[:m].reset_index(drop=True),
                                non.iloc[:m].reset_index(drop=True),
                                bd.qi_cols, bd.feature_names, sens[:len(rel)],
                                SENSITIVE_SOURCE, seed=seed)
            rows.append({"variant": name, "seed": seed, "auroc": u["auroc"],
                         "auroc_secondary": u2["auroc"], **a})
            print(f"  seed{seed} {name:22s} AUROC={u['auroc']:.3f} "
                  f"MIA={a['mia_auc']:.3f} reid={a['reid_rate']:.3f} "
                  f"attr={a['attr_bacc']:.3f} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    metrics = ["auroc", "auroc_secondary", "mia_auc", "reid_rate", "attr_bacc"]
    agg = df.groupby("variant")[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT / "ablation.csv", index=False)
    df.to_csv(OUT / "ablation_raw.csv", index=False)
    print("\nAblation aggregate -> results/ra3/ablation.csv")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
