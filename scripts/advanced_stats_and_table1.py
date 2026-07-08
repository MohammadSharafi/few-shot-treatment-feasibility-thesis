#!/usr/bin/env python3
"""
Real statistical hardening for the full-cohort thesis/paper:
  (1) AUROC + 95% bootstrap CI per model (from saved test predictions)
  (2) Pairwise DeLong test for AUROC differences (fast DeLong; Sun & Xu 2014)
  (3) Decision-curve / net-benefit analysis for the primary model
  (4) Reliability (calibration) curve for the primary model
  (5) Table 1 cohort characteristics (age, sex, ICU LOS) from raw MIMIC-IV files

Pure numpy/pandas/matplotlib (no scipy/sklearn). Outputs:
  results/full_cohort/stats/*.csv
  results/full_cohort/figures/full_cohort_decision_curve.png
  results/full_cohort/figures/full_cohort_reliability_xgboost.png
  results/full_cohort/tables/table1_characteristics.csv
"""
from __future__ import annotations
import os, glob, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "results/final_optimized/reproduced_baselines")
STATS_DIR = os.path.join(ROOT, "results/full_cohort/stats")
FIG_DIR = os.path.join(ROOT, "results/full_cohort/figures")
TAB_DIR = os.path.join(ROOT, "results/full_cohort/tables")
for d in (STATS_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

NICE = {
    "xgboost": "XGBoost", "catboost": "CatBoost", "lightgbm": "LightGBM",
    "random_forest": "Random Forest", "neural_baseline": "Neural Baseline",
    "logistic_regression": "Logistic Regression",
}

# ---------- AUROC + fast DeLong ----------
def auc_mw(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)]
    return (r_pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def fast_delong(preds, y):
    """preds: dict name->prob (same order/labels). Returns AUCs, covariance."""
    y = np.asarray(y)
    pos = y == 1; neg = ~pos
    m = pos.sum(); n = neg.sum()
    names = list(preds)
    k = len(names)
    X = np.vstack([np.asarray(preds[nm])[pos] for nm in names])  # k x m
    Y = np.vstack([np.asarray(preds[nm])[neg] for nm in names])  # k x n
    tx = np.array([_midrank(X[r]) for r in range(k)])
    ty = np.array([_midrank(Y[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([X[r], Y[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    s01 = np.atleast_2d(s01); s10 = np.atleast_2d(s10)
    cov = s01 / m + s10 / n
    return names, aucs, cov

def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def delong_pvalue(auc_i, auc_j, cov, i, j):
    var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    if var <= 0:
        return float("nan"), float("nan")
    z = (auc_i - auc_j) / math.sqrt(var)
    p = 2 * (1 - normal_cdf(abs(z)))
    return z, p

def boot_ci(y, p, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); idx = np.arange(len(y)); out = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        out.append(auc_mw(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*_test_predictions.csv")))
    preds = {}; y = None
    base = None
    for f in files:
        key = os.path.basename(f).replace("_test_predictions.csv", "")
        d = pd.read_csv(f).sort_values("stay_id").reset_index(drop=True)
        if base is None:
            base = d[["stay_id", "y_true"]].copy(); y = base.y_true.values
        preds[NICE.get(key, key)] = d.probability.values
    print(f"Loaded {len(preds)} models, n_test={len(y)}, prevalence={y.mean():.3f}")

    # (1) AUROC + bootstrap CI
    rows = []
    for nm, p in preds.items():
        lo, hi = boot_ci(y, p)
        rows.append({"model": nm, "AUROC": round(auc_mw(y, p), 4),
                     "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)})
    auc_df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    auc_df.to_csv(os.path.join(STATS_DIR, "auroc_bootstrap_ci.csv"), index=False)
    print("\nAUROC + 95% CI:\n", auc_df.to_string(index=False))

    # (2) DeLong pairwise
    names, aucs, cov = fast_delong(preds, y)
    pr = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            z, p = delong_pvalue(aucs[i], aucs[j], cov, i, j)
            pr.append({"model_A": names[i], "model_B": names[j],
                       "AUROC_A": round(aucs[i], 4), "AUROC_B": round(aucs[j], 4),
                       "delta": round(aucs[i] - aucs[j], 4),
                       "z": round(z, 3), "p_value": round(p, 4),
                       "significant_0.05": "yes" if p < 0.05 else "no"})
    dl = pd.DataFrame(pr).sort_values("p_value")
    dl.to_csv(os.path.join(STATS_DIR, "delong_pairwise.csv"), index=False)
    print("\nDeLong pairwise (head):\n", dl.head(12).to_string(index=False))

    # (3) Decision curve / net benefit for primary model (XGBoost)
    prim = "XGBoost"
    p = preds[prim]; N = len(y); prev = y.mean()
    ts = np.arange(0.05, 0.81, 0.01)
    nb_model, nb_all = [], []
    for t in ts:
        pred_pos = p >= t
        tp = np.sum((pred_pos) & (y == 1)); fp = np.sum((pred_pos) & (y == 0))
        nb_model.append(tp / N - fp / N * (t / (1 - t)))
        nb_all.append(prev - (1 - prev) * (t / (1 - t)))
    dca = pd.DataFrame({"threshold": ts, "net_benefit_model": nb_model,
                        "net_benefit_treat_all": nb_all, "net_benefit_treat_none": 0.0})
    dca.to_csv(os.path.join(STATS_DIR, "decision_curve_xgboost.csv"), index=False)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ts, nb_model, label="XGBoost model", lw=2)
    plt.plot(ts, nb_all, label="Treat all", ls="--", color="gray")
    plt.axhline(0, label="Treat none", ls=":", color="black")
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis (primary model, full-cohort test set)")
    plt.ylim(min(0, min(nb_all)) - 0.02, max(nb_model) + 0.02)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "full_cohort_decision_curve.png"), dpi=150)
    plt.close()

    # (4) Reliability curve for XGBoost (10 bins)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    rel = pd.DataFrame({"mean_pred": xs, "obs_freq": ys, "n": ns})
    rel.to_csv(os.path.join(STATS_DIR, "reliability_xgboost.csv"), index=False)
    plt.figure(figsize=(5.2, 5))
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect")
    plt.plot(xs, ys, "o-", color="C0", label="XGBoost")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Reliability diagram (primary model)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "full_cohort_reliability_xgboost.png"), dpi=150)
    plt.close()

    # (5) Table 1 from raw MIMIC-IV (join on stay_id)
    try:
        ic = pd.read_csv(os.path.join(ROOT, "mimic-iv-3.1/icu/icustays.csv.gz"),
                         usecols=["subject_id", "stay_id", "los", "first_careunit"])
        pa = pd.read_csv(os.path.join(ROOT, "mimic-iv-3.1/hosp/patients.csv.gz"),
                         usecols=["subject_id", "gender", "anchor_age"])
        df = base.merge(ic, on="stay_id", how="left").merge(pa, on="subject_id", how="left")
        df["los_hours"] = df["los"] * 24.0
        def summ(sub, name):
            r = {"group": name, "n": len(sub),
                 "feasible_rate": round(sub.y_true.mean(), 3),
                 "age_median": round(sub.anchor_age.median(), 1),
                 "age_q1": round(sub.anchor_age.quantile(.25), 0),
                 "age_q3": round(sub.anchor_age.quantile(.75), 0),
                 "female_pct": round(100 * (sub.gender == "F").mean(), 1),
                 "icu_los_h_median": round(sub.los_hours.median(), 1)}
            return r
        t1 = pd.DataFrame([summ(df, "Overall (test set)"),
                           summ(df[df.y_true == 1], "Feasible"),
                           summ(df[df.y_true == 0], "Infeasible")])
        t1.to_csv(os.path.join(TAB_DIR, "table1_characteristics.csv"), index=False)
        print("\nTable 1 (test set, real):\n", t1.to_string(index=False))
    except Exception as e:
        print("\n[Table 1] could not build from raw files:", e)

    print("\nWrote:",
          "\n -", os.path.join(STATS_DIR, "auroc_bootstrap_ci.csv"),
          "\n -", os.path.join(STATS_DIR, "delong_pairwise.csv"),
          "\n -", os.path.join(STATS_DIR, "decision_curve_xgboost.csv"),
          "\n -", os.path.join(FIG_DIR, "full_cohort_decision_curve.png"),
          "\n -", os.path.join(FIG_DIR, "full_cohort_reliability_xgboost.png"),
          "\n -", os.path.join(TAB_DIR, "table1_characteristics.csv"))

if __name__ == "__main__":
    main()
