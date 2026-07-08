#!/usr/bin/env python3
"""
Definitive few-shot fairness test: does proper tokenization + a stronger few-shot
method rescue few-shot learning on this task?

This gives few-shot its BEST fair chance, controlling the two things that could
unfairly handicap it:
  (1) Tokenization  -> compares the simple tokens (tokens.npy) against the
      ETHOS-aligned quantile/temporal tokens (tokens_ethos_bag.npy).
  (2) Few-shot method -> uses the stronger of Euclidean-prototype and
      cosine-prototype (cosine is better for high-dim sparse token vectors),
      averaged over many episodes, with standardisation.

It also trains the token-hybrid (tabular + ETHOS bag) to see whether the richer
tokens help the strong model. Honest result either way:
  - if ETHOS tokens lift few-shot meaningfully -> supports the title's premise;
  - if not -> the negative result holds even under ideal tokenisation.

Outputs:
  results/canonical/fewshot_ethos.csv       (K-sweep: simple vs ETHOS, best method)
  results/canonical/token_hybrid_ethos.csv  (tabular vs tabular+ETHOS bag)
Needs: tokens_ethos_bag.npy + tokens_ethos_meta.parquet (run 04_tokenize_ethos.py),
       tokens.npy + thesis_dataset.parquet ; catboost for the hybrid part.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42; KS = [1, 5, 10, 20, 50, 100]; N_EP = 300


def standardize(train, *others):
    mu = train.mean(0); sd = train.std(0) + 1e-8
    return [(a - mu) / sd for a in (train, *others)]


def proto_scores(Es, ys, Eq, metric):
    c0 = Es[ys == 0].mean(0); c1 = Es[ys == 1].mean(0)
    if metric == "euclidean":
        return np.linalg.norm(Eq - c0, axis=1) - np.linalg.norm(Eq - c1, axis=1)
    # cosine: higher similarity to class-1 prototype => higher score
    def cos(a, b):
        return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-8)
    return cos(Eq, c1) - cos(Eq, c0)


def kshot(E, y, itr, ite, metric):
    Etr, Ete = standardize(E[itr], E[ite]); ytr, yte = y[itr], y[ite]
    rng = np.random.default_rng(SEED)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    out = {}
    for K in KS:
        a = []
        for _ in range(N_EP):
            sup = np.concatenate([rng.choice(pos, K, replace=False), rng.choice(neg, K, replace=False)])
            a.append(roc_auc_score(yte, proto_scores(Etr[sup], ytr[sup], Ete, metric)))
        out[K] = (float(np.mean(a)), float(np.std(a)))
    return out


def main():
    meta = pd.read_parquet(PROC / "tokens_ethos_meta.parquet")
    if "feasible" not in meta.columns:
        raise SystemExit("run 04_tokenize_ethos.py after 03_labels.py (labels missing)")
    y = meta["feasible"].to_numpy(int)
    idx = np.arange(len(y)); itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    reps = {}
    reps["ETHOS_bag"] = np.load(PROC / "tokens_ethos_bag.npy")
    # simple tokens aligned to the same stay order via thesis_dataset
    try:
        simple = np.load(PROC / "tokens.npy").reshape(-1, 25 * 4)
        ds_order = pd.read_parquet(PROC / "thesis_dataset.parquet")["stay_id"].to_numpy()
        pos = {s: i for i, s in enumerate(ds_order)}
        sel = meta["stay_id"].map(pos).to_numpy()
        if not np.isnan(sel.astype(float)).any():
            reps["simple_25x4"] = simple[sel.astype(int)]
    except Exception as e:
        print("simple-token alignment skipped:", e)

    rows = []
    for rep_name, E in reps.items():
        for metric in ("euclidean", "cosine"):
            res = kshot(E, y, itr, ite, metric)
            for K, (m, s) in res.items():
                rows.append({"representation": rep_name, "method": metric, "K": K,
                             "AUROC_mean": round(m, 4), "AUROC_std": round(s, 4)})
            print(f"  {rep_name:12} {metric:9} K=100 AUROC={res[100][0]:.4f}")
    df = pd.DataFrame(rows)
    # keep best method per representation/K
    best = (df.sort_values("AUROC_mean", ascending=False)
              .drop_duplicates(["representation", "K"]).sort_values(["representation", "K"]))
    best.to_csv(OUT / "fewshot_ethos.csv", index=False)
    print("\nBest few-shot per representation/K:\n", best.to_string(index=False))

    # token-hybrid with ETHOS bag
    try:
        from catboost import CatBoostClassifier
        ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
        m = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
        feat = [c for c in ds.columns if c not in m]
        tab = np.nan_to_num(ds.set_index("stay_id").loc[meta["stay_id"]][feat].to_numpy(np.float32))
        bag = reps["ETHOS_bag"]; res = []
        for name, Z in [("tabular_only", tab), ("tabular+ETHOS_bag", np.concatenate([tab, bag], axis=1))]:
            c = CatBoostClassifier(iterations=400, learning_rate=0.05, auto_class_weights="Balanced",
                                   random_seed=SEED, verbose=0)
            c.fit(Z[itr], y[itr]); p = c.predict_proba(Z[ite])[:, 1]
            res.append({"feature_set": name, "AUROC": round(roc_auc_score(y[ite], p), 4),
                        "AUPRC": round(average_precision_score(y[ite], p), 4)})
            print(f"  [hybrid] {name}: AUROC={res[-1]['AUROC']}")
        pd.DataFrame(res).to_csv(OUT / "token_hybrid_ethos.csv", index=False)
    except Exception as e:
        print("hybrid part skipped:", e)
    print("Wrote fewshot_ethos.csv (+ token_hybrid_ethos.csv) to", OUT)


if __name__ == "__main__":
    main()
