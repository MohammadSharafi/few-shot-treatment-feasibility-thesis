#!/usr/bin/env python3
"""
Subgroup / fairness analysis (new results chapter section + TRIPOD-AI item).

Evaluates the primary model's discrimination and calibration within demographic
and clinical subgroups (age band, sex, disease node), to check for performance
disparities. Real, runnable; uses raw patient files for age/sex.

Outputs: results/canonical/subgroup_performance.csv (+ figure)
Needs: thesis_dataset.parquet, tokens.npy, raw patients/icustays ; catboost.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
H = ROOT / "mimic-iv-3.1/hosp"; I = ROOT / "mimic-iv-3.1/icu"
OUT = ROOT / "results/canonical"; FIG = ROOT / "results/full_cohort/figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
NODES = {"sepsis": ["A40", "A41", "R652"], "cardiac": ["I21", "I22", "I25", "I50", "I47", "I48", "I49"],
         "respiratory": ["J18", "J44", "J96", "J80", "J15"], "renal": ["N17", "N18", "N19"],
         "diabetes": ["E10", "E11", "E13"]}


def disease_of(icd):
    s = str(icd).upper()
    for node, codes in NODES.items():
        if any(s.startswith(c) for c in codes):
            return node
    return "other"


def metrics(y, p):
    if len(np.unique(y)) < 2 or len(y) < 30:
        return np.nan, np.nan, len(y)
    return roc_auc_score(y, p), brier_score_loss(y, p), len(y)


def main():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in ds.columns if c not in meta]
    X = np.nan_to_num(ds[feat].to_numpy(np.float32))
    tokens = np.load(PROC / "tokens.npy")
    Z = np.concatenate([X, tokens.reshape(len(tokens), -1)], axis=1)
    y = ds["feasible"].to_numpy(int)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, learning_rate=0.05, auto_class_weights="Balanced",
                           random_seed=SEED, verbose=0)
    m.fit(Z[itr], y[itr]); p = m.predict_proba(Z[ite])[:, 1]

    test = ds.iloc[ite].copy(); test["p"] = p; test["y"] = y[ite]
    test["disease"] = test["primary_icd10"].map(disease_of)
    # join age/sex
    pa = pd.read_csv(H / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    test = test.merge(pa, on="subject_id", how="left")
    test["age_band"] = pd.cut(test["anchor_age"], [0, 50, 65, 80, 200],
                              labels=["18-49", "50-64", "65-79", "80+"])

    rows = []
    def add(group, value, sub):
        au, br, n = metrics(sub["y"].values, sub["p"].values)
        rows.append({"group": group, "value": str(value), "n": n,
                     "feasible_rate": round(sub["y"].mean(), 3),
                     "AUROC": round(au, 4) if au == au else np.nan,
                     "Brier": round(br, 4) if br == br else np.nan})
    add("overall", "all", test)
    for v in ["F", "M"]:
        add("sex", v, test[test.gender == v])
    for v in ["18-49", "50-64", "65-79", "80+"]:
        add("age_band", v, test[test.age_band == v])
    for v in list(NODES) + ["other"]:
        add("disease", v, test[test.disease == v])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "subgroup_performance.csv", index=False)
    print(out.to_string(index=False))

    sub = out[out.group.isin(["sex", "age_band", "disease"])].dropna(subset=["AUROC"])
    plt.figure(figsize=(8, 4.6))
    colors = {"sex": "#6b8fb5", "age_band": "#3b6ea5", "disease": "#0b3d2e"}
    plt.bar(range(len(sub)), sub.AUROC, color=[colors[g] for g in sub.group])
    plt.axhline(out[out.group == "overall"].AUROC.iloc[0], ls="--", color="gray", label="overall")
    plt.xticks(range(len(sub)), sub.value, rotation=40, ha="right", fontsize=8)
    plt.ylabel("AUROC"); plt.ylim(0.6, 0.85); plt.legend()
    plt.title("Subgroup discrimination (primary model)")
    plt.tight_layout(); plt.savefig(FIG / "subgroup_auroc.png", dpi=160); plt.close()
    print("Wrote", OUT / "subgroup_performance.csv")


if __name__ == "__main__":
    main()
