#!/usr/bin/env python3
"""
Few-shot and probing on the GPT-2-PRETRAINED ETHOS embeddings (gap-filling test).

This is the true ETHOS-faithful evaluation: it uses the patient representation from
the autoregressively pretrained EthosGPT (scripts/pretrain_ethos_gpt.py) and asks
whether GPT-2-style pretraining rescues the few-shot model that failed without it.

Three downstream uses of the pretrained embedding:
  (1) few-shot prototype K-sweep (cosine) on pretrained embeddings,
  (2) linear probe (logistic regression) on pretrained embeddings,
  (3) hybrid (tabular + pretrained embedding) with CatBoost.

Outputs:
  results/canonical/fewshot_pretrained.csv
  results/canonical/probe_pretrained.csv
Needs: results/canonical/ethos_gpt.pt, tokens_ethos_seq.npy + meta, thesis_dataset.parquet.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))
from models.ethos_gpt import EthosGPT
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42; KS = [1, 5, 10, 20, 50, 100]; N_EP = 300


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embeddings():
    ckpt = torch.load(OUT / "ethos_gpt.pt", map_location="cpu", weights_only=False)
    c = ckpt["config"]
    m = EthosGPT(c["vocab_size"], max_len=c["max_len"], d_model=c["d_model"],
                 n_head=c["n_head"], n_layer=c["n_layer"])
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    seq = torch.from_numpy(np.load(PROC / "tokens_ethos_seq.npy").astype(np.int64))

    def _embed_on(dev):
        m.to(dev); out = []
        with torch.no_grad():
            for i in range(0, len(seq), 256):
                out.append(m.embed(seq[i:i + 256].to(dev)).cpu().numpy())
        return np.concatenate(out, 0)

    dev = _device()
    try:
        return _embed_on(dev)
    except Exception as e:                      # GPU/MPS quirk -> fall back to CPU
        if dev != "cpu":
            print(f"  ({dev} failed: {str(e)[:80]}; falling back to CPU)")
            return _embed_on("cpu")
        raise


def proto_cos(Es, ys, Eq):
    c0 = Es[ys == 0].mean(0); c1 = Es[ys == 1].mean(0)
    def cos(a, b): return (a @ b) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b)+1e-8)
    return cos(Eq, c1) - cos(Eq, c0)


def main():
    meta = pd.read_parquet(PROC / "tokens_ethos_meta.parquet")
    if "feasible" not in meta.columns:
        raise SystemExit("labels missing in meta; run 04_tokenize_ethos.py after 03_labels.py")
    y = meta["feasible"].to_numpy(int)
    E = embeddings()
    idx = np.arange(len(y)); itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(E[itr]); Etr, Ete = sc.transform(E[itr]), sc.transform(E[ite])
    ytr, yte = y[itr], y[ite]

    # (1) few-shot K-sweep on pretrained embeddings
    rng = np.random.default_rng(SEED); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    rows = []
    for K in KS:
        a = []
        for _ in range(N_EP):
            sup = np.concatenate([rng.choice(pos, K, replace=False), rng.choice(neg, K, replace=False)])
            a.append(roc_auc_score(yte, proto_cos(Etr[sup], ytr[sup], Ete)))
        rows.append({"K": K, "AUROC_mean": round(np.mean(a), 4), "AUROC_std": round(np.std(a), 4)})
        print(f"  [pretrained few-shot] K={K:<4} AUROC={np.mean(a):.4f}")
    pd.DataFrame(rows).to_csv(OUT / "fewshot_pretrained.csv", index=False)

    # (2) linear probe + (3) tabular+embedding hybrid
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    probe.fit(E[itr], ytr); pp = probe.predict_proba(E[ite])[:, 1]
    res = [{"model": "linear_probe(pretrained_emb)", "AUROC": round(roc_auc_score(yte, pp), 4),
            "AUPRC": round(average_precision_score(yte, pp), 4)}]
    try:
        from catboost import CatBoostClassifier
        ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
        m = ["stay_id","subject_id","hadm_id","feasible","protocol","primary_icd10","los_hours"]
        feat = [c for c in ds.columns if c not in m]
        tab = np.nan_to_num(ds.set_index("stay_id").loc[meta["stay_id"]][feat].to_numpy(np.float32))
        Z = np.concatenate([tab, E], axis=1)
        cb = CatBoostClassifier(iterations=400, learning_rate=0.05, auto_class_weights="Balanced",
                                random_seed=SEED, verbose=0)
        cb.fit(Z[itr], ytr); pz = cb.predict_proba(Z[ite])[:, 1]
        res.append({"model": "tabular+pretrained_emb(CatBoost)", "AUROC": round(roc_auc_score(yte, pz), 4),
                    "AUPRC": round(average_precision_score(yte, pz), 4)})
    except Exception as e:
        print("hybrid skipped:", e)
    pd.DataFrame(res).to_csv(OUT / "probe_pretrained.csv", index=False)
    print(pd.DataFrame(res).to_string(index=False))
    print("Wrote fewshot_pretrained.csv + probe_pretrained.csv to", OUT)


if __name__ == "__main__":
    main()
