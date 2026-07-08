#!/usr/bin/env python3
"""
Label-sensitivity experiment (thesis robustness pillar #1).

Re-derives FIVE clinically reasonable definitions of the discharge-feasibility
surrogate and trains a fixed model panel on each, to show that the central
finding (tree/hybrid models strong, raw few-shot weak, calibration matters) is
not an artefact of one arbitrary label choice.

Label variants:
  L1_primary   survive AND favourable discharge (home/rehab) AND ICU LOS < 720 h
  L2_mortality survive to hospital discharge (mortality only)
  L3_home_only survive AND discharge to HOME/HOME HEALTH CARE AND LOS < 720 h
  L4_strict_los survive AND favourable AND ICU LOS < 168 h (7 days)
  L5_no_los    survive AND favourable (drop the LOS condition)

Outputs: results/canonical/label_sensitivity.csv  (+ figure)

Needs: data/processed_full_cohort/{cohort.parquet, thesis_dataset.parquet, tokens.npy}
Run on a machine with scikit-learn / xgboost / lightgbm / catboost installed.
"""
from __future__ import annotations
import os, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
FAV = ["HOME", "HOME HEALTH CARE", "HOME WITH HOME IV PROVIDR", "REHAB/DISTINCT PART HOSP", "REHAB"]
SEED = 42


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return e


def boot_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2:
            out.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def make_models(seed=SEED):
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=300,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
    }
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(iterations=300, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=seed, verbose=0)
    except Exception:
        pass
    return models


def evaluate(X, y, name_prefix, hybrid_X=None):
    rows = []
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    panel = make_models()
    for name, model in panel.items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        lo, hi = boot_ci(yte, p)
        rows.append({"label": name_prefix, "model": name, "AUROC": round(roc_auc_score(yte, p), 4),
                     "CI_lo": round(lo, 4), "CI_hi": round(hi, 4),
                     "AUPRC": round(average_precision_score(yte, p), 4), "ECE": round(ece(yte, p), 4),
                     "n_pos": int(y.sum()), "n": int(len(y))})
    # Token-hybrid CatBoost
    if hybrid_X is not None:
        try:
            from catboost import CatBoostClassifier
            Htr, Hte, ytr2, yte2 = train_test_split(hybrid_X, y, test_size=0.2, random_state=SEED, stratify=y)
            m = CatBoostClassifier(iterations=300, learning_rate=0.05, auto_class_weights="Balanced",
                                   random_seed=SEED, verbose=0)
            m.fit(Htr, ytr2); p = m.predict_proba(Hte)[:, 1]
            lo, hi = boot_ci(yte2, p)
            rows.append({"label": name_prefix, "model": "Token-Hybrid CatBoost",
                         "AUROC": round(roc_auc_score(yte2, p), 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4),
                         "AUPRC": round(average_precision_score(yte2, p), 4), "ECE": round(ece(yte2, p), 4),
                         "n_pos": int(y.sum()), "n": int(len(y))})
        except Exception as exc:
            print("  (token-hybrid skipped:", exc, ")")
    return rows


def main():
    cohort = pd.read_parquet(PROC / "cohort.parquet")
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat_cols = [c for c in ds.columns if c not in meta]
    X = np.nan_to_num(ds[feat_cols].to_numpy(np.float32))
    tokens = np.load(PROC / "tokens.npy")
    hybrid_X = np.concatenate([X, tokens.reshape(len(tokens), -1)], axis=1)

    c = cohort.set_index("stay_id").loc[ds["stay_id"]].reset_index()
    surv = c.hospital_expire_flag == 0; los = c.los_hours
    labels = {
        "L1_primary":   (surv & c.discharge_location.isin(FAV) & (los < 720)).astype(int).to_numpy(),
        "L2_mortality": surv.astype(int).to_numpy(),
        "L3_home_only": (surv & c.discharge_location.isin(["HOME", "HOME HEALTH CARE"]) & (los < 720)).astype(int).to_numpy(),
        "L4_strict_los": (surv & c.discharge_location.isin(FAV) & (los < 168)).astype(int).to_numpy(),
        "L5_no_los":    (surv & c.discharge_location.isin(FAV)).astype(int).to_numpy(),
    }
    all_rows = []
    for name, y in labels.items():
        print(f"[{name}] prevalence={y.mean():.3f} (n_pos={int(y.sum())}/{len(y)})")
        all_rows += evaluate(X, y, name, hybrid_X=hybrid_X)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "label_sensitivity.csv", index=False)
    print("\nWrote", OUT / "label_sensitivity.csv")
    print(df.pivot_table(index="model", columns="label", values="AUROC").to_string())


if __name__ == "__main__":
    main()
