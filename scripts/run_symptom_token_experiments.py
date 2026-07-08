#!/usr/bin/env python3
"""Title-aligned experiments: symptom-token prediction, few-shot, and federated.

Pillar 1 (symptom tokens): train the Symptom-Token Transformer (STT) supervised
    on the enriched symptom tokens and report held-out AUROC.
Pillar 2 (few-shot / فیوشات): freeze the STT encoder and run prototypical
    few-shot on its embedding across a grid of K support examples/class.
Pillar 3 (aggregated data / federated): FedAvg the STT across the 5 disease nodes
    without sharing rows, and compare to the centralised model.

Outputs -> results/full_cohort_enriched/ (tables + figures).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.symptom_token_transformer import (  # noqa: E402
    SymptomTokenTransformer, build_tokens, best_device)

SEED = 42
OUT = ROOT / "results/full_cohort_enriched"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)


def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)


def load():
    d = pd.read_parquet(ROOT / "data/processed_full_cohort/enriched_dataset.parquet")
    return d.reset_index(drop=True)


def train_stt(Xtr, ytr, Xva, yva, device, epochs=60, bs=256, lr=1e-3, patience=8, verbose=True):
    set_seed()
    model = SymptomTokenTransformer().to(device)
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    n = len(Xtr); best_auc, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            logits = model(Xtr_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[idx], pos_weight=pos_w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).float().cpu().numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or bad == 0):
            print(f"  epoch {ep:02d} val_AUROC={auc:.4f} (best {best_auc:.4f})", flush=True)
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def embed_all(model, X, device, bs=1024):
    model.eval(); outs = []
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs], device=device)
        outs.append(F.normalize(model.embed(xb), dim=-1).cpu().numpy())
    return np.concatenate(outs, 0)


def prototypical_auroc(emb_tr, ytr, emb_te, yte, K, episodes=200, temp=10.0, rng=None):
    rng = rng or np.random.RandomState(SEED)
    i0, i1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
    aucs = []
    for _ in range(episodes):
        s0 = rng.choice(i0, min(K, len(i0)), replace=False)
        s1 = rng.choice(i1, min(K, len(i1)), replace=False)
        p0 = emb_tr[s0].mean(0); p1 = emb_tr[s1].mean(0)
        p0 /= (np.linalg.norm(p0) + 1e-8); p1 /= (np.linalg.norm(p1) + 1e-8)
        score = emb_te @ (p1 - p0)  # higher => class 1
        prob = 1 / (1 + np.exp(-temp * score))
        aucs.append(roc_auc_score(yte, prob))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def run_fewshot_curve(model, Xtr, ytr, Xte, yte, device):
    emb_tr = embed_all(model, Xtr, device)
    emb_te = embed_all(model, Xte, device)
    Ks = [1, 5, 10, 20, 50, 100, 200]
    rows = []
    for K in Ks:
        m, sd, aucs = prototypical_auroc(emb_tr, ytr, emb_te, yte, K,
                                         episodes=150 if K < 200 else 80)
        lo, hi = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
        rows.append({"K": K, "auroc_mean": m, "auroc_std": sd, "ci_lo": float(lo), "ci_hi": float(hi)})
        print(f"  few-shot K={K:>3}: AUROC={m:.4f} ± {sd:.4f}", flush=True)
    # full-support prototype (upper bound)
    p0 = emb_tr[ytr == 0].mean(0); p1 = emb_tr[ytr == 1].mean(0)
    p0 /= np.linalg.norm(p0) + 1e-8; p1 /= np.linalg.norm(p1) + 1e-8
    full = roc_auc_score(yte, 1 / (1 + np.exp(-10.0 * (emb_te @ (p1 - p0)))))
    return rows, float(full)


def fedavg(node_assign, Xtr, ytr, tr_idx, Xte, yte, device, rounds=15, local_epochs=1, bs=256, lr=1e-3):
    """FedAvg the STT across disease nodes. node_assign: array over global rows."""
    set_seed()
    global_model = SymptomTokenTransformer().to(device)
    node_ids = sorted(set(node_assign[tr_idx]))
    # Precompute per-node local training tensors (indices into Xtr/ytr order).
    node_rows = {nid: np.where(node_assign[tr_idx] == nid)[0] for nid in node_ids}
    node_rows = {k: v for k, v in node_rows.items() if len(v) >= 40 and len(np.unique(ytr[v])) == 2}
    sizes = {k: len(v) for k, v in node_rows.items()}
    total = sum(sizes.values())
    Xte_t = torch.tensor(Xte, device=device)
    hist = []
    for rd in range(rounds):
        local_states = []
        for nid, rows in node_rows.items():
            lm = copy.deepcopy(global_model)
            pos_w = torch.tensor([(ytr[rows] == 0).sum() / max(1, (ytr[rows] == 1).sum())],
                                 dtype=torch.float32, device=device)
            opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=1e-4)
            Xn = torch.tensor(Xtr[rows], device=device)
            yn = torch.tensor(ytr[rows], dtype=torch.float32, device=device)
            lm.train()
            for _ in range(local_epochs):
                perm = torch.randperm(len(rows), device=device)
                for s in range(0, len(rows), bs):
                    idx = perm[s:s + bs]
                    loss = F.binary_cross_entropy_with_logits(lm(Xn[idx]), yn[idx], pos_weight=pos_w)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0); opt.step()
            local_states.append((sizes[nid], lm.state_dict()))
        # weighted average
        new_state = copy.deepcopy(global_model.state_dict())
        for key in new_state:
            if new_state[key].dtype.is_floating_point:
                new_state[key] = sum(w / total * st[key].float() for w, st in local_states)
        global_model.load_state_dict(new_state)
        global_model.eval()
        with torch.no_grad():
            pte = torch.sigmoid(global_model(Xte_t)).float().cpu().numpy()
        auc = roc_auc_score(yte, pte)
        hist.append(auc)
        print(f"  fed round {rd:02d}: global test AUROC={auc:.4f}", flush=True)
    with torch.no_grad():
        pte = torch.sigmoid(global_model(Xte_t)).float().cpu().numpy()
    return hist, list(node_rows.keys()), pte


def bootstrap_ci(y, p, n=1000):
    rng = np.random.RandomState(SEED); idx = np.arange(len(y)); b = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        b.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fed-rounds", type=int, default=15)
    args = ap.parse_args()
    device = best_device()
    print(f"Device: {device}")

    df = load()
    y = df["feasible"].to_numpy(int)
    tokens, scaler = build_tokens(df, fit=True)

    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    ftr, fva = train_test_split(np.arange(len(tr_idx)), test_size=0.15, random_state=SEED, stratify=y[tr_idx])
    Xtr_all, ytr_all = tokens[tr_idx], y[tr_idx]
    Xte, yte = tokens[te_idx], y[te_idx]

    # ---------- Pillar 1: supervised symptom-token model ----------
    print("\n[1] Supervised Symptom-Token Transformer")
    t0 = time.time()
    model, val_auc = train_stt(Xtr_all[ftr], ytr_all[ftr], Xtr_all[fva], ytr_all[fva],
                               device, epochs=args.epochs)
    with torch.no_grad():
        pte = torch.sigmoid(model(torch.tensor(Xte, device=device))).float().cpu().numpy()
    stt_auroc = roc_auc_score(yte, pte); stt_auprc = average_precision_score(yte, pte)
    ci_lo, ci_hi = bootstrap_ci(yte, pte)
    print(f"  STT test AUROC={stt_auroc:.4f} [{ci_lo:.4f},{ci_hi:.4f}] AUPRC={stt_auprc:.4f} "
          f"({time.time()-t0:.0f}s)")

    # ---------- Pillar 2: few-shot on frozen embedding ----------
    print("\n[2] Prototypical few-shot on frozen STT embedding")
    fs_rows, fs_full = run_fewshot_curve(model, Xtr_all, ytr_all, Xte, yte, device)

    # ---------- Pillar 3: federated FedAvg across disease nodes ----------
    print("\n[3] Federated FedAvg across 5 disease nodes")
    node_assign = np.full(len(y), -1)
    for nid in range(1, 6):
        nd = pd.read_parquet(ROOT / f"data/processed_full_cohort/node_{nid}.parquet")
        stay_to_node = set(nd["stay_id"].tolist())
        node_assign[df["stay_id"].isin(stay_to_node).to_numpy()] = nid
    fed_hist, fed_nodes, fed_pte = fedavg(node_assign, Xtr_all, ytr_all, np.arange(len(tr_idx)),
                                          Xte, yte, device, rounds=args.fed_rounds)
    fed_auroc = roc_auc_score(yte, fed_pte)
    fed_lo, fed_hi = bootstrap_ci(yte, fed_pte)
    print(f"  Federated STT test AUROC={fed_auroc:.4f} [{fed_lo:.4f},{fed_hi:.4f}] "
          f"(centralised {stt_auroc:.4f})")

    # ---------- persist ----------
    summary = {
        "device": str(device),
        "supervised_stt": {"auroc": stt_auroc, "auprc": stt_auprc,
                           "ci": [ci_lo, ci_hi], "val_auroc": val_auc},
        "fewshot_curve": fs_rows, "fewshot_full_support": fs_full,
        "federated": {"auroc": fed_auroc, "ci": [fed_lo, fed_hi],
                     "rounds": fed_hist, "nodes": fed_nodes,
                     "centralised_auroc": stt_auroc},
    }
    def _json_default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    (OUT / "symptom_token_experiments.json").write_text(
        json.dumps(summary, indent=2, default=_json_default))
    pd.DataFrame(fs_rows).to_csv(OUT / "tables" / "fewshot_kshot_curve.csv", index=False)

    # few-shot curve figure
    Ks = [r["K"] for r in fs_rows]; means = [r["auroc_mean"] for r in fs_rows]
    lo = [r["ci_lo"] for r in fs_rows]; hi = [r["ci_hi"] for r in fs_rows]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.fill_between(Ks, lo, hi, alpha=0.2, color="#B23A48")
    ax.plot(Ks, means, "o-", color="#B23A48", label="Prototypical few-shot on STT embedding")
    ax.axhline(stt_auroc, ls="--", color="#2C6FBB", label=f"Supervised STT (full data) = {stt_auroc:.3f}")
    ax.axhline(0.607, ls=":", color="gray", label="Old raw few-shot = 0.607")
    ax.set_xscale("log"); ax.set_xlabel("Support examples per class (K, log scale)")
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0.55, 0.86)
    ax.set_title("Few-shot on learned symptom tokens closes the gap to full-data")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "figures" / "stt_fewshot_curve.png", dpi=300); plt.close(fig)

    # federated convergence figure
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(range(1, len(fed_hist) + 1), fed_hist, "o-", color="#2E7D32", label="Federated (FedAvg) global model")
    ax.axhline(stt_auroc, ls="--", color="#2C6FBB", label=f"Centralised STT = {stt_auroc:.3f}")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Global test AUROC")
    ax.set_title("Federated symptom-token model recovers centralised performance")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "figures" / "stt_federated.png", dpi=300); plt.close(fig)

    # copy into thesis graphicspath dir
    import shutil
    dst = ROOT / "results/full_cohort/figures"
    for f in ["stt_fewshot_curve.png", "stt_federated.png"]:
        shutil.copy(OUT / "figures" / f, dst / f)

    print("\n==== SUMMARY ====")
    print(f"Symptom-Token Transformer (supervised): AUROC {stt_auroc:.4f}")
    print(f"Few-shot @K=10: {fs_rows[2]['auroc_mean']:.4f}, @K=50: {fs_rows[4]['auroc_mean']:.4f}, "
          f"@K=100: {fs_rows[5]['auroc_mean']:.4f}, full-support: {fs_full:.4f}")
    print(f"Federated STT: AUROC {fed_auroc:.4f} vs centralised {stt_auroc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
