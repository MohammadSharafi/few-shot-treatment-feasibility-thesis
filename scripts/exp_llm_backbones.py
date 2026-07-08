#!/usr/bin/env python3
"""
Benchmark of open-source clinical/biomedical transformer backbones.

Tests whether stronger pretrained backbones beat the current models. Each patient's
first-24h profile is serialised into a short clinical-note-style string (the
TabLLM/LIFT approach), embedded with each backbone, and the embeddings are
evaluated three ways for a fair, apples-to-apples comparison with CatBoost,
Token-Hybrid, ETHOS-token and GPT-pretrained:

  - linear probe (logistic) AUROC      -> embedding quality
  - few-shot prototype (cosine) AUROC  -> few-shot usefulness (K=5/20/100)
  - tabular + embedding (CatBoost)      -> hybrid usefulness (AUROC, ECE)

Backbones (open-source, auto-downloaded from HuggingFace; each skipped gracefully
if unavailable):
  emilyalsentzer/Bio_ClinicalBERT                      (clinical BERT)
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract (PubMedBERT, biomedical)
  dmis-lab/biobert-base-cased-v1.2                     (BioBERT)
  sentence-transformers/all-MiniLM-L6-v2              (general lightweight)

Output: results/canonical/llm_backbones.csv (one row per backbone) + per-backbone
        few-shot K-sweep in results/canonical/llm_fewshot_<name>.csv
Needs: transformers + torch + catboost; data/processed_full_cohort/thesis_dataset.parquet
Run:   pip install transformers && python scripts/exp_llm_backbones.py
       (set HF_BACKBONES env var to a comma list to override the model list)
"""
from __future__ import annotations
import os, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
FEAT = ["Creatinine", "Potassium", "Sodium", "Chloride", "Glucose", "Calcium",
        "Magnesium", "Hemoglobin", "Platelets", "WBC", "Lactate", "ALT", "AST",
        "BUN", "Bicarbonate", "HeartRate", "SystolicBP", "DiastolicBP", "MeanBP",
        "Temperature", "SpO2", "RespRate"]
DEFAULT_BACKBONES = [
    "emilyalsentzer/Bio_ClinicalBERT",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "dmis-lab/biobert-base-cased-v1.2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def serialize(X, names):
    """Turn each row into 'Feature is low/normal/high (level k/10)' clinical text."""
    # per-feature deciles for low/normal/high wording
    q = np.quantile(X, [0.2, 0.8], axis=0)
    texts = []
    for row in X:
        parts = []
        for j, nm in enumerate(names):
            v = row[j]
            lvl = "low" if v <= q[0, j] else ("high" if v >= q[1, j] else "normal")
            parts.append(f"{nm} {lvl}")
        texts.append("ICU first 24 hours. " + ", ".join(parts) + ".")
    return texts


def embed(texts, model_name, device):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i+64], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            h = mdl(**b).last_hidden_state           # (B, T, H)
            mask = b["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            outs.append(emb.cpu().numpy())
    return np.concatenate(outs, 0)


def cos_proto(Es, ys, Eq):
    c0 = Es[ys == 0].mean(0); c1 = Es[ys == 1].mean(0)
    f = lambda a, b: (a @ b) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b)+1e-8)
    return f(Eq, c1) - f(Eq, c0)


def fewshot(E, y, itr, ite, name):
    sc = StandardScaler().fit(E[itr])
    Etr, Ete = sc.transform(E[itr]), sc.transform(E[ite]); ytr, yte = y[itr], y[ite]
    rng = np.random.default_rng(SEED); pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    rows = {}
    for K in (5, 20, 100):
        a = []
        for _ in range(200):
            sup = np.concatenate([rng.choice(pos, K, replace=False),
                                  rng.choice(neg, K, replace=False)])   # drawn ONCE
            a.append(roc_auc_score(yte, cos_proto(Etr[sup], ytr[sup], Ete)))
        rows[K] = float(np.mean(a))
    return rows


def main():
    ds = pd.read_parquet(PROC / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in ds.columns if c not in meta]
    names = FEAT[:len(feat)]
    X = np.nan_to_num(ds[feat].to_numpy(np.float32)); y = ds["feasible"].to_numpy(int)
    texts = serialize(X, names)
    idx = np.arange(len(y)); itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    backbones = os.environ.get("HF_BACKBONES", ",".join(DEFAULT_BACKBONES)).split(",")
    summary = []
    for name in backbones:
        name = name.strip()
        try:
            print(f"\n=== {name} (device={device}) ===")
            try:
                E = embed(texts, name, device)
            except Exception as ge:                 # GPU/MPS quirk for this model -> CPU
                if device != "cpu":
                    print(f"  ({device} failed: {str(ge)[:80]}; retrying on CPU)")
                    E = embed(texts, name, "cpu")
                else:
                    raise
            # linear probe
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            probe.fit(E[itr], y[itr]); pp = probe.predict_proba(E[ite])[:, 1]
            probe_auc = roc_auc_score(y[ite], pp)
            # few-shot
            fs = fewshot(E, y, itr, ite, name)
            pd.DataFrame([{"K": k, "AUROC": round(v, 4)} for k, v in fs.items()]).to_csv(
                OUT / f"llm_fewshot_{re.sub('[^a-zA-Z0-9]', '_', name)}.csv", index=False)
            # hybrid
            hyb_auc = hyb_ece = np.nan
            try:
                from catboost import CatBoostClassifier
                Z = np.concatenate([X, E], axis=1)
                cb = CatBoostClassifier(iterations=400, learning_rate=0.05, auto_class_weights="Balanced",
                                        random_seed=SEED, verbose=0)
                cb.fit(Z[itr], y[itr]); pz = cb.predict_proba(Z[ite])[:, 1]
                hyb_auc = roc_auc_score(y[ite], pz); hyb_ece = brier_score_loss(y[ite], pz)
            except Exception as e:
                print("  hybrid skipped:", e)
            summary.append({"backbone": name, "emb_dim": E.shape[1],
                            "probe_AUROC": round(probe_auc, 4),
                            "fewshot_K100_AUROC": round(fs[100], 4),
                            "hybrid_AUROC": round(hyb_auc, 4) if hyb_auc == hyb_auc else np.nan,
                            "hybrid_Brier": round(hyb_ece, 4) if hyb_ece == hyb_ece else np.nan})
            print(f"  probe={probe_auc:.4f}  fewshotK100={fs[100]:.4f}  hybrid={hyb_auc:.4f}")
        except Exception as e:
            print(f"  SKIPPED {name}: {e}")
            summary.append({"backbone": name, "error": str(e)[:120]})

    df = pd.DataFrame(summary)
    # reference rows for fair comparison
    ref = pd.DataFrame([
        {"backbone": "CatBoost (tabular) [reference]", "hybrid_AUROC": 0.756},
        {"backbone": "Token-Hybrid CatBoost [reference]", "hybrid_AUROC": 0.756},
        {"backbone": "Few-Shot ETHOS raw [reference]", "fewshot_K100_AUROC": 0.607},
    ])
    out = pd.concat([df, ref], ignore_index=True)
    out.to_csv(OUT / "llm_backbones.csv", index=False)
    print("\n", out.to_string(index=False))
    print("\nKeep only backbones whose hybrid_AUROC or fewshot beats the references.")
    print("Wrote", OUT / "llm_backbones.csv")


if __name__ == "__main__":
    main()
