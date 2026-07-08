#!/usr/bin/env python3
"""
Tokenization-sensitivity experiment (thesis robustness pillar #2).

Builds SEVERAL Symptom-Token encodings and evaluates the proposed Token-Hybrid
CatBoost on each, to show that the hybrid's strong, well-calibrated result is not
tied to one arbitrary tokenization choice.

Token schemes (all derived from the same first-24h aggregated features):
  T_none        tabular features only (reference; no tokens)
  T_25x4        canonical scheme: 25 steps x [id, intensity, time, modality]
  T_seq15       shorter sequence (15 steps)
  T_seq35       longer sequence (35 steps)
  T_no_modality drop the modality channel (25 x 3)
  T_intensity   intensity-only flattened (compact)
  T_binned      intensity discretised into 5 bins (ordinal tokens)

Outputs: results/canonical/token_sensitivity.csv (+ figure)
Needs: data/processed_full_cohort/{thesis_dataset.parquet, tokens.npy}
Run where catboost is installed.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((np.sum(idx == b) / len(p)) * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if np.any(idx == b)))


def boot_ci(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2:
            out.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def rebuild_tokens(tokens, scheme):
    """Return a flat 2-D token matrix for the given scheme from the (N,25,4) tensor."""
    N = len(tokens)
    if scheme == "T_25x4":
        return tokens.reshape(N, -1)
    if scheme == "T_seq15":
        return tokens[:, :15, :].reshape(N, -1)
    if scheme == "T_seq35":
        pad = np.zeros((N, 35 - tokens.shape[1], tokens.shape[2]), dtype=tokens.dtype)
        return np.concatenate([tokens, pad], axis=1).reshape(N, -1)
    if scheme == "T_no_modality":
        return tokens[:, :, :3].reshape(N, -1)
    if scheme == "T_intensity":
        return tokens[:, :, 1]                       # intensity channel only
    if scheme == "T_binned":
        inten = tokens[:, :, 1]
        return np.clip((inten * 5).astype(int), 0, 4).astype(np.float32)
    raise ValueError(scheme)


def fit_eval(Z, y):
    from catboost import CatBoostClassifier
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.2, random_state=SEED, stratify=y)
    m = CatBoostClassifier(iterations=300, learning_rate=0.05, auto_class_weights="Balanced",
                           random_seed=SEED, verbose=0)
    m.fit(Ztr, ytr); p = m.predict_proba(Zte)[:, 1]
    lo, hi = boot_ci(yte, p)
    return {"AUROC": round(roc_auc_score(yte, p), 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4),
            "AUPRC": round(average_precision_score(yte, p), 4), "ECE": round(ece(yte, p), 4)}


def main():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in ds.columns if c not in meta]
    X = np.nan_to_num(ds[feat].to_numpy(np.float32))
    y = ds["feasible"].to_numpy(int)
    tokens = np.load(PROC / "tokens.npy")

    schemes = ["T_none", "T_25x4", "T_seq15", "T_seq35", "T_no_modality", "T_intensity", "T_binned"]
    rows = []
    for s in schemes:
        if s == "T_none":
            Z = X
        else:
            Z = np.concatenate([X, rebuild_tokens(tokens, s)], axis=1)
        r = fit_eval(Z, y); r.update({"scheme": s, "n_features": Z.shape[1]})
        rows.append(r); print(f"  {s:14} AUROC={r['AUROC']} ECE={r['ECE']} dims={Z.shape[1]}")
    df = pd.DataFrame(rows)[["scheme", "n_features", "AUROC", "CI_lo", "CI_hi", "AUPRC", "ECE"]]
    df.to_csv(OUT / "token_sensitivity.csv", index=False)
    print("\nWrote", OUT / "token_sensitivity.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
