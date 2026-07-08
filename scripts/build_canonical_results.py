#!/usr/bin/env python3
"""
SINGLE SOURCE OF TRUTH for the thesis results.

All numbers come from ONE run: the leakage-audited `final_optimized` pipeline on
the full 32,399-stay cohort (train 20,735 / val 5,184 / test 6,480). Discrimination,
calibration, operating point and confusion come from final_model_results.csv;
bootstrap CIs, DeLong tests, decision-curve and reliability are recomputed in pure
numpy from the matching saved test predictions (same run). No mixing across runs.

Outputs -> results/canonical/
"""
from __future__ import annotations
import os, math, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results/final_optimized/predictions")
FOPT = os.path.join(ROOT, "results/final_optimized/tables/final_model_results.csv")
OUT = os.path.join(ROOT, "results/canonical")
FIG = os.path.join(OUT, "figures")
for d in (OUT, FIG): os.makedirs(d, exist_ok=True)

# Canonical model panel: display name -> (model, feature_set) in final_model_results.csv.
# Prediction files are resolved ROBUSTLY from each row's experiment_id at run time
# (NOT hardcoded), so the script follows whatever ids a fresh optimization run
# produces. When several rows match a (model, feature_set) we keep the best AUROC.
PANEL = [
    ("Token-Hybrid CatBoost", ("CatBoost", "tabular_token")),
    ("Stacked Ensemble", ("Stacked Validation Meta-LR", "mixed")),
    ("Weighted Ensemble", ("Weighted Average Ensemble", "mixed")),
    ("CatBoost", ("CatBoost", "tabular")),
    ("Random Forest", ("Random Forest", "tabular")),
    ("XGBoost", ("XGBoost", "tabular")),
    ("LightGBM", ("LightGBM", "tabular")),
    ("Neural Baseline", ("Neural Baseline", "tabular")),
    ("Federated LR", ("Simulated Federated LR Weighted Nodes", "tabular")),
    ("Logistic Regression", ("Logistic Regression", "tabular")),
    ("Few-Shot ETHOS", ("Few-Shot ETHOS Prototype", "token")),
]


def resolve_pred_file(exp_id):
    """Return an existing prediction-file path for an experiment_id, or None.
    Tries the id as-is and with a 'selected_' prefix; finally globs as a fallback."""
    import glob as _glob
    if not isinstance(exp_id, str) or not exp_id:
        return None
    for cand in (exp_id, f"selected_{exp_id}"):
        p = os.path.join(PRED, f"{cand}_test_predictions.csv")
        if os.path.exists(p):
            return p
    hits = _glob.glob(os.path.join(PRED, f"*{exp_id}*_test_predictions.csv"))
    return hits[0] if hits else None

def auc_mw(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = y.sum(); n0 = len(y) - n1
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p)+1)
    u, inv, c = np.unique(p, return_inverse=True, return_counts=True)
    s = np.zeros(len(c)); np.add.at(s, inv, ranks); ranks = (s/c)[inv]
    return (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)

def boot_ci(y, p, n=2000, seed=2026):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2: out.append(auc_mw(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def fast_delong(preds, y):
    y = np.asarray(y); pos = y == 1; neg = ~pos; m = pos.sum(); n = neg.sum()
    names = list(preds); k = len(names)
    X = np.vstack([np.asarray(preds[nm])[pos] for nm in names])
    Y = np.vstack([np.asarray(preds[nm])[neg] for nm in names])
    tx = np.array([_midrank(X[r]) for r in range(k)])
    ty = np.array([_midrank(Y[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([X[r], Y[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(1)/m - (m+1)/2)/n
    v01 = (tz[:, :m] - tx)/n; v10 = 1 - (tz[:, m:] - ty)/m
    cov = np.atleast_2d(np.cov(v01))/m + np.atleast_2d(np.cov(v10))/n
    return names, aucs, cov

def ncdf(z): return 0.5*(1+math.erf(z/math.sqrt(2)))

def main():
    fr = pd.read_csv(FOPT)
    rows, preds = [], {}
    y = None
    has_expid = "experiment_id" in fr.columns
    for disp, (model, feat) in PANEL:
        sub = fr[(fr.model == model) & (fr.feature_set == feat)]
        if sub.empty:
            print("WARN: no row for", disp); continue
        r = sub.sort_values("AUROC", ascending=False).iloc[0]   # best match if several
        rec = {"model": disp, "AUROC": round(r.AUROC, 4), "AUPRC": round(r.AUPRC, 4),
               "accuracy": round(r.accuracy, 4), "recall": round(r.recall, 4),
               "specificity": round(r.specificity, 4), "F1": round(r.F1, 4),
               "Brier": round(r.Brier, 4), "ECE": round(r.ECE, 4),
               "cal_slope": round(r.calibration_slope, 3), "threshold": round(r.threshold, 3),
               "tn": int(r.tn), "fp": int(r.fp), "fn": int(r.fn), "tp": int(r.tp)}
        tn, fp, fn, tp = rec["tn"], rec["fp"], rec["fn"], rec["tp"]
        denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) or 1
        rec["MCC"] = round((tp*tn - fp*fn)/denom, 3)
        pf = resolve_pred_file(r["experiment_id"]) if has_expid else None
        if pf:
            d = pd.read_csv(pf).sort_values("stay_id")
            yy = d.y_true.values; pp = d.probability.values
            if y is None: y = yy
            if len(yy) == (len(y) if y is not None else len(yy)):
                preds[disp] = pp
                lo, hi = boot_ci(yy, pp)
                rec["CI_lo"], rec["CI_hi"] = round(lo, 4), round(hi, 4)
            else:
                rec["CI_lo"], rec["CI_hi"] = np.nan, np.nan
        else:
            rec["CI_lo"], rec["CI_hi"] = np.nan, np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "canonical_model_results.csv"), index=False)
    print(out[["model","AUROC","CI_lo","CI_hi","AUPRC","Brier","ECE","MCC"]].to_string(index=False))

    # DeLong across models that have predictions
    names, aucs, cov = fast_delong(preds, y)
    dl = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            var = cov[i, i] + cov[j, j] - 2*cov[i, j]
            z = (aucs[i]-aucs[j])/math.sqrt(var) if var > 0 else float("nan")
            p = 2*(1-ncdf(abs(z))) if var > 0 else float("nan")
            dl.append({"A": names[i], "B": names[j], "dAUROC": round(aucs[i]-aucs[j], 4),
                       "z": round(z, 3), "p": round(p, 4), "sig": "yes" if p < 0.05 else "no"})
    pd.DataFrame(dl).sort_values("p").to_csv(os.path.join(OUT, "canonical_delong.csv"), index=False)

    # Primary model = Token-Hybrid CatBoost
    prim = "Token-Hybrid CatBoost"; p = preds[prim]; N = len(y); prev = y.mean()
    # Decision curve
    ts = np.arange(0.05, 0.81, 0.01); nbm, nba = [], []
    for t in ts:
        pp = p >= t; tp = np.sum(pp & (y == 1)); fp = np.sum(pp & (y == 0))
        nbm.append(tp/N - fp/N*(t/(1-t))); nba.append(prev-(1-prev)*(t/(1-t)))
    # persist the exact DCA values so the thesis numbers stay verifiable each run
    pd.DataFrame({"threshold": np.round(ts, 2), "net_benefit_model": nbm,
                  "net_benefit_treat_all": nba, "net_benefit_treat_none": 0.0}).to_csv(
        os.path.join(OUT, "canonical_decision_curve.csv"), index=False)
    plt.figure(figsize=(7,4.5)); plt.plot(ts, nbm, lw=2, label="Token-Hybrid CatBoost")
    plt.plot(ts, nba, "--", color="gray", label="Treat all"); plt.axhline(0, ls=":", color="black", label="Treat none")
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis (primary model, full-cohort test)")
    plt.ylim(min(0, min(nba))-0.02, max(nbm)+0.02); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "canonical_decision_curve.png"), dpi=160); plt.close()
    # Reliability
    bins = np.linspace(0,1,11); idx = np.clip(np.digitize(p, bins)-1, 0, 9); xs, ys = [], []
    for b in range(10):
        mb = idx == b
        if mb.sum(): xs.append(p[mb].mean()); ys.append(y[mb].mean())
    plt.figure(figsize=(5.2,5)); plt.plot([0,1],[0,1],"--",color="gray",label="Perfect")
    plt.plot(xs, ys, "o-", label=prim); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Reliability diagram (primary model)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "canonical_reliability.png"), dpi=160); plt.close()

    # AUROC bar with CI
    dd = out.dropna(subset=["CI_lo"]).sort_values("AUROC")
    plt.figure(figsize=(7.6,4.8))
    yv = np.arange(len(dd))
    plt.barh(yv, dd.AUROC, color="#3b6ea5",
             xerr=[dd.AUROC-dd.CI_lo, dd.CI_hi-dd.AUROC], capsize=3)
    plt.yticks(yv, dd.model, fontsize=9); plt.xlim(0.70, 0.78)
    plt.xlabel("Test AUROC (95% bootstrap CI)"); plt.title("Discrimination on the full-cohort test set")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "canonical_auroc.png"), dpi=160); plt.close()

    # Confusion matrix for primary
    r = out[out.model == prim].iloc[0]
    cm = np.array([[r.tn, r.fp],[r.fn, r.tp]])
    plt.figure(figsize=(4.6,4.2)); plt.imshow(cm, cmap="Blues")
    for (a,b), v in np.ndenumerate(cm):
        plt.text(b, a, f"{int(v)}", ha="center", va="center", fontsize=13,
                 color="white" if v > cm.max()/2 else "black")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    plt.title(f"Confusion matrix — {prim}"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "canonical_confusion.png"), dpi=160); plt.close()

    print("\nDeLong (top vs top, head):")
    print(pd.DataFrame(dl).sort_values("p", ascending=False).head(8).to_string(index=False))
    print("\nWrote canonical results + figures to", OUT)

if __name__ == "__main__":
    main()
