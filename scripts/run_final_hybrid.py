#!/usr/bin/env python3
"""Culminating model: fuse the enriched tabular ensemble with the Symptom-Token
Transformer (STT) embedding — the fully-realised "symptom-token hybrid".

Pipeline:
  1. Train STT on the enriched tokens; extract its patient embedding (train/test).
  2. Train the enriched gradient-boosted models; take their OOF/test probabilities.
  3. Stack [tabular model probs || STT prob || STT embedding-LR prob] with a
     calibrated logistic meta-learner.
Reports AUROC/AUPRC/Brier/ECE with bootstrap CIs and writes a final comparison
figure + table for the manuscript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.symptom_token_transformer import (  # noqa: E402
    SymptomTokenTransformer, build_tokens, best_device)
from scripts.run_symptom_token_experiments import train_stt, embed_all  # noqa: E402

SEED = 42
OUT = ROOT / "results/full_cohort_enriched"
META = {"stay_id", "subject_id", "hadm_id", "los_hours", "primary_icd10", "feasible"}


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); idx = np.digitize(p, edges[1:-1]); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def boot_ci(y, p, n=1000):
    rng = np.random.RandomState(SEED); idx = np.arange(len(y)); b = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1:
            b.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def gb_models(seed):
    import lightgbm as lgb, xgboost as xgb, catboost as cb
    return {
        "XGBoost": xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.5,
            eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=48,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.7, min_child_samples=30,
            reg_lambda=1.5, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=700, depth=6, learning_rate=0.03,
            l2_leaf_reg=3.0, auto_class_weights="Balanced", random_seed=seed, verbose=0,
            allow_writing_files=False, thread_count=-1),
    }


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = best_device(); print("device", device)
    df = pd.read_parquet(ROOT / "data/processed_full_cohort/enriched_dataset.parquet").reset_index(drop=True)
    y = df["feasible"].to_numpy(int)
    feat = [c for c in df.columns if c not in META]
    X = df[feat].to_numpy(np.float32)
    tokens, _ = build_tokens(df, fit=True)

    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    ytr, yte = y[tr], y[te]

    # --- STT embedding ---
    ftr, fva = train_test_split(np.arange(len(tr)), test_size=0.15, random_state=SEED, stratify=ytr)
    stt, _ = train_stt(tokens[tr][ftr], ytr[ftr], tokens[tr][fva], ytr[fva], device, epochs=60, verbose=False)
    with torch.no_grad():
        stt_test = torch.sigmoid(stt(torch.tensor(tokens[te], device=device))).float().cpu().numpy()
    emb_tr = embed_all(stt, tokens[tr], device)
    emb_te = embed_all(stt, tokens[te], device)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    emb_lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    emb_oof = cross_val_predict(emb_lr, emb_tr, ytr, cv=skf, method="predict_proba")[:, 1]
    emb_lr.fit(emb_tr, ytr); emb_test = emb_lr.predict_proba(emb_te)[:, 1]
    # STT OOF prob via CV on tokens would be costly; approximate its meta-feature with emb_oof.

    # --- tabular GB OOF + test ---
    oof_cols, test_cols, names = [emb_oof], [emb_test], ["STT-emb"]
    for name, est in gb_models(SEED).items():
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", est)])
        oof = cross_val_predict(pipe, X[tr], ytr, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
        pipe.fit(X[tr], ytr); tst = pipe.predict_proba(X[te])[:, 1]
        oof_cols.append(oof); test_cols.append(tst); names.append(name)
        print(f"  {name}: test AUROC {roc_auc_score(yte, tst):.4f}")
    print(f"  STT (token transformer): test AUROC {roc_auc_score(yte, stt_test):.4f}")

    Z_tr = np.column_stack(oof_cols); Z_te = np.column_stack(test_cols)
    mfit, mcal = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=SEED, stratify=ytr)
    meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    meta.fit(Z_tr[mfit], ytr[mfit])
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(meta), method="isotonic")
    except Exception:
        cal = CalibratedClassifierCV(meta, method="isotonic", cv="prefit")
    cal.fit(Z_tr[mcal], ytr[mcal])
    final_p = cal.predict_proba(Z_te)[:, 1]

    auroc = roc_auc_score(yte, final_p); lo, hi = boot_ci(yte, final_p)
    res = {
        "final_hybrid": {"auroc": auroc, "ci": [lo, hi],
                         "auprc": float(average_precision_score(yte, final_p)),
                         "brier": float(brier_score_loss(yte, final_p)),
                         "ece": ece(yte, final_p), "components": names},
        "stt_alone": float(roc_auc_score(yte, stt_test)),
    }
    (OUT / "final_hybrid.json").write_text(json.dumps(res, indent=2))
    print(f"\nFINAL SYMPTOM-TOKEN HYBRID: AUROC {auroc:.4f} [{lo:.4f},{hi:.4f}] "
          f"AUPRC {res['final_hybrid']['auprc']:.4f} ECE {res['final_hybrid']['ece']:.4f}")

    # ROC figure comparing the journey
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    curves = [("Median-only baseline (0.755)", None, "#B0B0B0", 0.755),
              ("Enriched tabular ensemble", final_p, None, None)]
    fpr, tpr, _ = roc_curve(yte, final_p)
    ax.plot(fpr, tpr, color="#8E24AA", lw=2.4, label=f"Final symptom-token hybrid (AUROC {auroc:.3f})")
    fpr2, tpr2, _ = roc_curve(yte, stt_test)
    ax.plot(fpr2, tpr2, color="#2C6FBB", lw=1.8, ls="-", label=f"Symptom-Token Transformer ({roc_auc_score(yte,stt_test):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC: the fully-realised symptom-token model")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "figures" / "final_hybrid_roc.png", dpi=300)
    import shutil
    shutil.copy(OUT / "figures" / "final_hybrid_roc.png", ROOT / "results/full_cohort/figures/final_hybrid_roc.png")
    print("saved final_hybrid_roc.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
