"""Attack-in-the-loop certification demo.

Given a target membership-inference AUC, bisection-search the RA3 noise scale to find
the *least* protective (highest-utility) release that still meets the target, verifying
with the real MIA each iteration. This turns RA3 from 'pick a scale' into 'name a risk
budget, get the minimal-distortion certified release'.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from data import build_broker_data, train_holdout_split
from mechanisms import RA3
from utility import downstream_utility
from attacks import membership_inference_attack, linkage_reidentification, attribute_inference_attack

OUT = Path("results/ra3"); OUT.mkdir(parents=True, exist_ok=True)
SENSITIVE_SOURCE = "50813_max"


def evaluate(release, mem, non, qi, cols, sens_src, sens, seed):
    a = membership_inference_attack(release, mem, non, cols, seed)
    l = linkage_reidentification(release, mem, qi, seed=seed)
    att = attribute_inference_attack(release, sens[:len(release)],
                                     [c for c in cols if c != sens_src], seed)
    return {**a, **l, **att}


def certify(target_mia, bd, tr, te, mem, non, m, sens, seed, lo=0.2, hi=3.0, iters=7):
    """Bisection on noise scale to the smallest s with MIA <= target."""
    Xtr = bd.X.iloc[tr].reset_index(drop=True); ytr = bd.y[tr]
    Xte = bd.X.iloc[te].reset_index(drop=True); yte = bd.y[te]
    trace = []
    best = None
    for it in range(iters):
        s = 0.5 * (lo + hi)
        rel = RA3(noise=s, seed=seed).fit_transform(Xtr, bd.qi_cols, bd.clinical_cols, y=ytr)
        mia = membership_inference_attack(rel, mem.iloc[:m].reset_index(drop=True),
                                          non.iloc[:m].reset_index(drop=True),
                                          bd.feature_names, seed)["mia_auc"]
        u = downstream_utility(rel, ytr, Xte, yte, bd.feature_names, seed=seed)["auroc"]
        trace.append({"iter": it, "s": round(s, 3), "mia": round(mia, 3), "auroc": round(u, 3)})
        print(f"    it{it} s={s:.3f} MIA={mia:.3f} AUROC={u:.3f}")
        if mia <= target_mia:
            best = (s, rel); hi = s          # meets target -> try less protection
        else:
            lo = s                            # too leaky -> more protection
    # final full evaluation at the certified s
    s_final = best[0] if best else hi
    rel = RA3(noise=s_final, seed=seed).fit_transform(Xtr, bd.qi_cols, bd.clinical_cols, y=ytr)
    u = downstream_utility(rel, ytr, Xte, yte, bd.feature_names, seed=seed)
    u2 = downstream_utility(rel, bd.y_secondary[tr], Xte, bd.y_secondary[te],
                            bd.feature_names, seed=seed)
    atk = evaluate(rel, mem.iloc[:m].reset_index(drop=True), non.iloc[:m].reset_index(drop=True),
                   bd.qi_cols, bd.feature_names, SENSITIVE_SOURCE, sens, seed)
    return {"target_mia": target_mia, "certified_s": round(s_final, 3),
            "auroc": round(u["auroc"], 3), "auroc_secondary": round(u2["auroc"], 3),
            "mia_auc": round(atk["mia_auc"], 3), "reid_rate": round(atk["reid_rate"], 4),
            "attr_bacc": round(atk["attr_bacc"], 3), "iterations": len(trace),
            "trace": trace}


def main(n_sample=12000, seed=42):
    t0 = time.time()
    bd = build_broker_data(n_sample=n_sample, seed=seed)
    tr, te = train_holdout_split(bd, seed=seed)
    non_idx = np.where(~bd.member_mask)[0]
    mem = bd.X.iloc[tr].reset_index(drop=True)
    non = bd.X.iloc[non_idx].reset_index(drop=True)
    m = min(len(mem), len(non))
    v = bd.X[SENSITIVE_SOURCE].to_numpy()[tr]
    sens = (v > np.nanmedian(v)).astype(int)

    results = []
    for tgt in (0.60, 0.55):
        print(f"  certifying to MIA <= {tgt} ...")
        results.append(certify(tgt, bd, tr, te, mem, non, m, sens, seed))
    json.dump(results, open(OUT / "certify.json", "w"), indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s -> results/ra3/certify.json")
    for r in results:
        print(f"  target {r['target_mia']}: s*={r['certified_s']} "
              f"MIA={r['mia_auc']} AUROC={r['auroc']} sec={r['auroc_secondary']} "
              f"reid={r['reid_rate']} attr={r['attr_bacc']} in {r['iterations']} iters")


if __name__ == "__main__":
    main()
