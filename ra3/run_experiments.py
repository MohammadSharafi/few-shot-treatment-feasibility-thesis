"""End-to-end RA3 privacy-utility evaluation.

For each mechanism (and, for DP-style mechanisms, each epsilon) we:
  1. aggregate the broker's training records,
  2. measure downstream utility on real held-out records,
  3. run the full attack suite for privacy,
and write a tidy results table plus a machine-readable JSON for the figures.
"""
from __future__ import annotations

import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data import build_broker_data, train_holdout_split, quasi_identifier_uniqueness
from mechanisms import Raw, UniformDP, KAnonymityMicro, DPSynthetic, RA3, noise_to_epsilon
from utility import downstream_utility
from attacks import run_all_attacks

SEED = 42
OUT = Path("results/ra3"); OUT.mkdir(parents=True, exist_ok=True)


SENSITIVE_SOURCE = "50813_max"    # lactate max: proxy for critical illness (the secret)


def sensitive_attribute(bd, idx):
    """A sensitive binary clinical attribute the attacker should not learn: whether
    the patient's lactate (50813) max was elevated (a proxy for critical illness)."""
    col = SENSITIVE_SOURCE if SENSITIVE_SOURCE in bd.X.columns else bd.clinical_cols[0]
    v = bd.X[col].to_numpy()[idx]
    return (v > np.nanmedian(v)).astype(int)


NOISE_GRID = [0.25, 0.5, 1.0, 2.0]


def build_mechanisms():
    mechs = [("Raw", Raw())]
    for s in NOISE_GRID:
        mechs.append((f"UniformDP_s{s}", UniformDP(noise=s, seed=SEED)))
    for k in [5, 10, 25]:
        mechs.append((f"kAnon_k{k}", KAnonymityMicro(k=k, seed=SEED)))
    for s in NOISE_GRID:
        mechs.append((f"DPSynth_s{s}", DPSynthetic(noise=s, seed=SEED)))
    for s in NOISE_GRID:
        mechs.append((f"RA3_s{s}", RA3(noise=s, seed=SEED)))
    return mechs


def main(n_sample=None):
    t0 = time.time()
    print("Loading broker data ...")
    bd = build_broker_data(n_sample=n_sample, seed=SEED)
    print(f"  n={bd.n}, |QI|={len(bd.qi_cols)}, |clinical|={len(bd.clinical_cols)}")

    uq_fine = quasi_identifier_uniqueness(bd.X, bd.qi_cols)
    uq_coarse = quasi_identifier_uniqueness(bd.X, bd.qi_cols, bins=10)
    print("  QI uniqueness (raw):", uq_fine)

    tr_idx, te_idx = train_holdout_split(bd, seed=SEED)
    nonmem_idx = np.where(~bd.member_mask)[0]

    X_train = bd.X.iloc[tr_idx].reset_index(drop=True)
    y_train = bd.y[tr_idx]
    X_test = bd.X.iloc[te_idx].reset_index(drop=True)   # real, unprotected
    y_test = bd.y[te_idx]
    y2_train = bd.y_secondary[tr_idx]                   # secondary task labels
    y2_test = bd.y_secondary[te_idx]
    members_orig = X_train.copy()                       # original member records
    nonmembers_orig = bd.X.iloc[nonmem_idx].reset_index(drop=True)
    # match probe sizes
    m = min(len(members_orig), len(nonmembers_orig))
    members_probe = members_orig.iloc[:m].reset_index(drop=True)
    nonmembers_probe = nonmembers_orig.iloc[:m].reset_index(drop=True)
    sensitive = sensitive_attribute(bd, tr_idx)

    all_cols = bd.feature_names
    rows = []
    budget_records = {}
    for name, mech in build_mechanisms():
        ts = time.time()
        released = mech.fit_transform(X_train, bd.qi_cols, bd.clinical_cols, y=y_train)
        util = downstream_utility(released, y_train, X_test, y_test, all_cols, seed=SEED)
        # Secondary-task utility: general fidelity, not the task the release was tuned on.
        util2 = downstream_utility(released, y2_train, X_test, y2_test, all_cols, seed=SEED)
        util["auroc_secondary"] = util2["auroc"]
        atk = run_all_attacks(
            released, members_probe, nonmembers_probe,
            bd.qi_cols, all_cols, sensitive[:len(released)], SENSITIVE_SOURCE, seed=SEED,
        )
        row = {"mechanism": name, **util, **atk, "sec": round(time.time() - ts, 1)}
        rows.append(row)
        if isinstance(mech, RA3):
            budget_records[name] = {
                "budget": mech.budget_.tolist(),
                "exposure": mech.exposure_.tolist(),
                "utility": mech.utility_.tolist(),
                "features": all_cols,
            }
        print(f"  {name:18s} AUROC={util['auroc']:.3f} AUROC2={util['auroc_secondary']:.3f} "
              f"MIA={atk['mia_auc']:.3f} reid={atk['reid_rate']:.3f} "
              f"attr={atk['attr_bacc']:.3f} ({row['sec']}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_table.csv", index=False)

    summary = {
        "n_records": bd.n,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "qi_cols": bd.qi_cols,
        "n_clinical": len(bd.clinical_cols),
        "qi_uniqueness_raw": uq_fine,
        "qi_uniqueness_coarse_10bin": uq_coarse,
        "results": rows,
        "ra3_budget": budget_records,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']}s -> {OUT/'results.json'}")
    return summary


def main_multiseed(n_sample=None, seeds=(42, 7, 123)):
    """Run the sweep across several seeds and aggregate mean +/- std per mechanism."""
    global SEED
    all_runs = []
    for s in seeds:
        SEED = s
        print(f"\n===== SEED {s} =====")
        summ = main(n_sample=n_sample)
        for r in summ["results"]:
            all_runs.append({**r, "seed": s})
    df = pd.DataFrame(all_runs)
    metrics = ["auroc", "auroc_secondary", "mia_auc", "reid_rate", "attr_bacc"]
    agg = df.groupby("mechanism")[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["mechanism"] + [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]
    agg.to_csv(OUT / "results_multiseed.csv", index=False)
    df.to_csv(OUT / "results_multiseed_raw.csv", index=False)
    with open(OUT / "results_multiseed.json", "w") as f:
        json.dump({"seeds": list(seeds), "runs": all_runs,
                   "aggregate": agg.to_dict(orient="records")}, f, indent=2)
    print(f"\nMulti-seed aggregate -> {OUT/'results_multiseed.csv'}")
    return agg


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    n = None; multiseed = False
    for a in args:
        if a == "--multiseed":
            multiseed = True
        else:
            n = int(a)
    if multiseed:
        main_multiseed(n_sample=n)
    else:
        main(n_sample=n)
