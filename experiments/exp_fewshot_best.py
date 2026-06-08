#!/usr/bin/env python3
"""
BEST ETHOS+FewShot Model: Combines all thesis innovations for maximum performance.

Integrates:
- V5 Gated Feature Attention (GFA) encoder
- V5 Cross-Attention Prototypical head
- V5 extensive feature engineering (22→132+)
- SupCon + Deep KD + Consistency regularization
- Proto-MAML test-time adaptation (12 steps)
- 7-seed ensemble for stability
- Enhanced stacking: FS + RF + embeddings + polynomial interactions
- Longer training: 40/35/2500 (P1/P2/P3)
- Larger model: 4 layers, dff=512
- More TTA (6) and eval episodes (25)

Target: Push beyond V5 (0.737) and V4 stacked (0.732) toward RF ceiling (0.760).

Usage:
  python exp_fewshot_best.py           # Full run (7 seeds, ~30-45 min)
  python exp_fewshot_best.py --quick   # Fast validation (1 seed, ~5 min)
  python exp_fewshot_best.py --light   # Low-memory (2 seeds, smaller model, ~15 min)
"""
import sys, gc, json, warnings, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             average_precision_score, matthews_corrcoef,
                             precision_score, recall_score)

proj = Path(__file__).parent.parent
DEVICE = torch.device("cpu")


def load_config():
    with open(proj / "config.yaml") as f:
        return yaml.safe_load(f)


def clean():
    gc.collect()


# ─── V5 Feature Engineering (from exp_fewshot_v5) ─────────────────────

def engineer_features_v5(X):
    """V5 features: base + 3rd-order interactions + quantile bins + domain ratios."""
    n, d = X.shape
    feats = [X]
    pairs = [(0,13),(4,14),(7,20),(10,11),(8,14),(1,3),(15,16),
             (2,5),(6,9),(0,4),(7,14),(13,20),(1,5),(3,6)]
    for i, j in pairs:
        if i < d and j < d:
            feats.append((X[:,i] / (np.abs(X[:,j]) + 1e-6)).reshape(-1,1))
            feats.append((X[:,i] * X[:,j]).reshape(-1,1))
    for i in range(min(d, 22)):
        feats.append((X[:,i] ** 2).reshape(-1,1))
    for i in range(min(d, 15)):
        feats.append(np.log1p(X[:,i] - X[:,i].min() + 1e-6).reshape(-1,1))
    triplets = [(0,10,14), (7,8,20), (1,3,15), (4,13,16), (0,8,10)]
    for i, j, k in triplets:
        if i < d and j < d and k < d:
            feats.append((X[:,i] * X[:,j] * X[:,k]).reshape(-1,1))
    for i in range(min(d, 15)):
        col = X[:, i]
        q25, q50, q75 = np.percentile(col, [25, 50, 75])
        feats.append(((col < q25).astype(np.float32)).reshape(-1,1))
        feats.append(((col > q75).astype(np.float32)).reshape(-1,1))
    if d >= 22:
        hr = X[:, 15].clip(0.01, None); sbp = X[:, 16].clip(0.01, None)
        feats.append((hr / sbp).reshape(-1,1))
        creat = X[:, 0]; bun = X[:, 13]
        feats.append((bun / (creat + 1e-6)).reshape(-1,1))
        ast = X[:, 12]; alt = X[:, 11]
        feats.append((ast / (alt + 1e-6)).reshape(-1,1))
    feats.append(X.mean(1, keepdims=True))
    feats.append(X.std(1, keepdims=True))
    feats.append(np.median(X, axis=1, keepdims=True))
    feats.append((X > 0.5).sum(1, keepdims=True).astype(np.float32))
    feats.append((X < 0.1).sum(1, keepdims=True).astype(np.float32))
    feats.append(np.percentile(X, 25, axis=1, keepdims=True).astype(np.float32))
    feats.append(np.percentile(X, 75, axis=1, keepdims=True).astype(np.float32))
    return np.nan_to_num(np.concatenate(feats, 1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ─── Model Architecture (V5 GFA + Cross-Attention) ────────────────────

class GatedFeatureAttention(nn.Module):
    def __init__(self, n_features, d_model, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.shared_fc = nn.Linear(n_features, d_model)
        self.step_attns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, n_features), nn.BatchNorm1d(n_features))
            for _ in range(n_steps)
        ])
        self.step_fcs = nn.ModuleList([
            nn.Sequential(nn.Linear(n_features, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for _ in range(n_steps)
        ])
        self.gamma = 1.5

    def forward(self, x):
        B = x.size(0)
        prior_scales = torch.ones(B, x.size(1), device=x.device)
        aggregated = torch.zeros(B, self.step_fcs[0][-1].out_features, device=x.device)
        complementary = self.shared_fc(x)
        for step in range(self.n_steps):
            attn_logits = self.step_attns[step](complementary)
            attn = F.softmax(attn_logits * prior_scales, dim=-1)
            prior_scales = prior_scales * (self.gamma - attn)
            masked_x = attn * x
            step_out = self.step_fcs[step](masked_x)
            aggregated = aggregated + step_out
        return aggregated


class V5EncoderLayer(nn.Module):
    def __init__(self, d, nh, dff, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(drop), nn.Linear(dff, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.dr = nn.Dropout(drop)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        x = self.n1(x + self.dr(out))
        return self.n2(x + self.dr(self.ff(x)))


class V5Encoder(nn.Module):
    def __init__(self, n_features, d=128, nh=8, nl=4, dff=512, drop=0.12):
        super().__init__()
        self.gfa = GatedFeatureAttention(n_features, d, n_steps=3)
        self.gs = 4
        self.nt = (n_features + self.gs - 1) // self.gs
        self.nf = n_features
        self.tok_proj = nn.Linear(self.gs, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Embedding(self.nt + 2, d)
        self.layers = nn.ModuleList([V5EncoderLayer(d, nh, dff, drop) for _ in range(nl)])
        self.dr = nn.Dropout(drop)
        self.pool = nn.Linear(d, 1)
        self.norm = nn.LayerNorm(d)
        self.gate_merge = nn.Parameter(torch.tensor(0.5))
        self.d = d

    def forward(self, x):
        B = x.size(0)
        gfa_out = self.gfa(x)
        pad = self.nt * self.gs - self.nf
        xp = F.pad(x, (0, pad)) if pad > 0 else x
        tokens = self.tok_proj(xp.view(B, self.nt, self.gs))
        h = torch.cat([self.cls.expand(B, -1, -1), tokens], 1)
        h = self.dr(h + self.pos(torch.arange(h.size(1), device=h.device)))
        for layer in self.layers:
            h = layer(h)
        w = F.softmax(self.pool(h).squeeze(-1), dim=1)
        transformer_out = self.norm((h * w.unsqueeze(-1)).sum(1))
        g = torch.sigmoid(self.gate_merge)
        return g * transformer_out + (1 - g) * gfa_out


class CrossAttentionProtoHead(nn.Module):
    def __init__(self, d=128, nh=4, drop=0.12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.classifier = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
        self.log_temp = nn.Parameter(torch.tensor(2.3))
        self.class_bias = nn.Parameter(torch.zeros(2))

    def forward(self, s_emb, q_emb, s_labels, n_way=2):
        s_n = F.normalize(s_emb, dim=-1)
        q_n = F.normalize(q_emb, dim=-1)
        protos = []
        for c in range(n_way):
            mask = s_labels == c
            p = s_n[mask].mean(0) if mask.sum() > 0 else s_n.mean(0)
            protos.append(p)
        protos_t = torch.stack(protos)
        protos_n = F.normalize(protos_t, dim=-1)
        temp = self.log_temp.exp().clamp(0.1, 100.0)
        cosine_logits = torch.mm(q_n, protos_n.t()) / temp + self.class_bias
        q_expanded = q_n.unsqueeze(1).expand(-1, n_way, -1)
        p_expanded = protos_n.unsqueeze(0).expand(q_n.size(0), -1, -1)
        cross_out, _ = self.cross_attn(q_expanded, p_expanded, p_expanded)
        cross_out = self.norm(cross_out + q_expanded)
        cross_logits = []
        for c in range(n_way):
            combined = torch.cat([cross_out[:, c, :], q_n], dim=-1)
            cross_logits.append(self.classifier(combined).squeeze(-1))
        cross_logits = torch.stack(cross_logits, dim=-1)
        return 0.5 * cosine_logits + 0.5 * cross_logits


class FewShotBestModel(nn.Module):
    def __init__(self, n_features, d=128, nh=8, nl=4, dff=512, drop=0.12):
        super().__init__()
        self.encoder = V5Encoder(n_features, d, nh, nl, dff, drop)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.sup_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 2))
        self.proto_head = CrossAttentionProtoHead(d, nh=4, drop=drop)
        self.d = d

    def encode(self, x):
        return self.encoder(x)

    def project(self, e):
        return F.normalize(self.proj(e), dim=-1)

    def forward(self, sx, qx, sy):
        se = self.encode(sx)
        qe = self.encode(qx)
        return self.proto_head(se, qe, sy)

    def sup_fwd(self, x):
        return self.sup_head(self.encode(x))


# ─── Losses ──────────────────────────────────────────────────────────

def supcon(f, l, t=0.1):
    f = F.normalize(f, -1); n = f.shape[0]
    s = torch.mm(f, f.t()) / t
    mp = (l.unsqueeze(0) == l.unsqueeze(1)).float(); mp.fill_diagonal_(0)
    if mp.sum() == 0: return torch.tensor(0.0)
    es = torch.exp(s - s.max(1, keepdim=True)[0]) * (1 - torch.eye(n, device=f.device))
    lp = s - torch.log(es.sum(1, keepdim=True) + 1e-8)
    return -(mp * lp).sum(1).div(mp.sum(1) + 1e-8).mean()


def kd_loss(sl, tp, T=3.0):
    ts = torch.stack([1 - tp, tp], -1)
    return F.kl_div(F.log_softmax(sl / T, -1), F.softmax(ts / T, -1), reduction="batchmean") * T**2


def consistency_loss(model, x, noise_std=0.02):
    with torch.no_grad():
        clean_logits = model.sup_fwd(x)
        clean_probs = F.softmax(clean_logits, dim=-1)
    noisy_logits = model.sup_fwd(x + torch.randn_like(x) * noise_std)
    return F.kl_div(F.log_softmax(noisy_logits, dim=-1), clean_probs, reduction="batchmean")


# ─── Training ────────────────────────────────────────────────────────

def phase1_supcon_deep_kd(model, Xtr, ytr, rf_probs, rf_importances, epochs=40, lr=5e-4, batch_size=128):
    print("  P1: SupCon + Deep KD + Consistency...", flush=True)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    rft = torch.tensor(rf_probs, dtype=torch.float32)
    imp_t = torch.tensor(rf_importances[:Xtr.shape[1]], dtype=torch.float32)
    imp_t = imp_t / (imp_t.sum() + 1e-8)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(Xt)
    best_loss = float("inf"); patience = 0
    for ep in range(epochs):
        model.train()
        pm = torch.randperm(n); total = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = pm[i:i+batch_size]
            if len(idx) < 4: continue
            e = model.encode(Xt[idx])
            proj_e = model.project(e)
            logits = model.sup_head(e)
            l_sc = supcon(proj_e, yt[idx])
            l_ce = F.cross_entropy(logits, yt[idx], label_smoothing=0.1)
            l_kd = kd_loss(logits, rft[idx], 4.0)
            l_consist = consistency_loss(model, Xt[idx], 0.02)
            loss = l_sc + 0.3 * l_ce + 0.5 * l_kd + 0.15 * l_consist
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item(); nb += 1
        sched.step()
        avg = total / max(nb, 1)
        if ep % 10 == 0: print(f"    ep{ep}: loss={avg:.4f}", flush=True)
        if avg < best_loss - 0.001: best_loss = avg; patience = 0
        else:
            patience += 1
            if patience >= 10: break
    clean()


def phase2_supervised_kd(model, Xtr, ytr, rf_probs, Xvl, yvl, epochs=35, lr=3e-4):
    print("  P2: Supervised + KD...", flush=True)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    rft = torch.tensor(rf_probs, dtype=torch.float32)
    Xv = torch.tensor(Xvl, dtype=torch.float32)
    cc = np.bincount(ytr)
    w = torch.tensor([float(cc.sum()) / (2.0 * float(c)) for c in cc], dtype=torch.float32)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr * 0.3},
        {"params": model.sup_head.parameters(), "lr": lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(Xt); best_auroc = 0; best_st = None; patience = 0

    for ep in range(epochs):
        model.train()
        pm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = pm[i:i+64]
            logits = model.sup_fwd(Xt[idx])
            ce = F.cross_entropy(logits, yt[idx], weight=w, reduction="none", label_smoothing=0.05)
            pt = torch.exp(-ce)
            loss = ((1 - pt)**2 * ce).mean() + 0.4 * kd_loss(logits, rft[idx], 3.0)
            if ep > 5:
                loss = loss + 0.12 * consistency_loss(model, Xt[idx], 0.015)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va = roc_auc_score(yvl, F.softmax(model.sup_fwd(Xv), -1)[:, 1].numpy())
        if va > best_auroc:
            best_auroc = va; best_st = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 12: break
        if ep % 5 == 0: print(f"    ep{ep}: va={va:.4f}", flush=True)
    if best_st: model.load_state_dict(best_st)
    print(f"    Best: {best_auroc:.4f}", flush=True)
    clean()


def phase3_episodic(model, Xtr, ytr, n_episodes=2500):
    print(f"  P3: Episodic ({n_episodes})...", flush=True)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    cc = np.bincount(ytr)
    w = torch.tensor([float(cc.sum()) / (2.0 * float(c)) for c in cc], dtype=torch.float32)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": model.proto_head.parameters(), "lr": 5e-4},
        {"params": model.proj.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-6)

    i0 = (yt == 0).nonzero(as_tuple=True)[0]
    i1 = (yt == 1).nonzero(as_tuple=True)[0]
    k_sched = [10, 7, 5, 3, 5, 3]
    slen = max(1, n_episodes // len(k_sched))

    for ep in range(n_episodes):
        model.train()
        stage = min(ep // slen, len(k_sched) - 1)
        k = min(k_sched[stage], len(i0) - 15, len(i1) - 15)
        if k < 2: continue
        kq = 15
        s0 = i0[torch.randperm(len(i0))[:k + kq]]
        s1 = i1[torch.randperm(len(i1))[:k + kq]]
        si = torch.cat([s0[:k], s1[:k]])
        q0 = min(kq, len(s0) - k); q1 = min(kq, len(s1) - k)
        qi = torch.cat([s0[k:k+q0], s1[k:k+q1]])
        if len(qi) < 4: continue

        sx = Xt[si]; qx = Xt[qi]
        sy = torch.cat([torch.zeros(k, dtype=torch.long), torch.ones(k, dtype=torch.long)])
        qy = torch.cat([torch.zeros(q0, dtype=torch.long), torch.ones(q1, dtype=torch.long)])[:len(qi)]
        if len(qy) != len(qi): continue

        if ep % 2 == 0:
            sx = sx + torch.randn_like(sx) * 0.02
            qx = qx + torch.randn_like(qx) * 0.02

        logits = model(sx, qx, sy)
        ce = F.cross_entropy(logits, qy, weight=w, reduction="none", label_smoothing=0.05)
        pt = torch.exp(-ce)
        loss = ((1 - pt)**2 * ce).mean()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

        if ep % 500 == 0:
            print(f"    ep{ep}: {loss.item():.4f}", flush=True); clean()
    clean(); print("    Done", flush=True)


# ─── Proto-MAML ───────────────────────────────────────────────────────

def proto_maml(model, Xs, ys, Xq, steps=12, lr=0.01):
    ad = copy.deepcopy(model); ad.train()
    Xst = torch.tensor(Xs, dtype=torch.float32)
    yst = torch.tensor(ys, dtype=torch.long)
    Xqt = torch.tensor(Xq, dtype=torch.float32)
    opt = torch.optim.SGD([
        {"params": ad.proto_head.parameters(), "lr": lr},
        {"params": ad.encoder.parameters(), "lr": lr * 0.01},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    for _ in range(steps):
        n = len(Xst); pm = torch.randperm(n); h = max(n // 2, 2)
        mi = pm[:h]; mq = pm[h:] if len(pm) > h else pm[:h]
        se = ad.encode(Xst[mi]); qe = ad.encode(Xst[mq])
        logits = ad.proto_head(se, qe, yst[mi])
        loss = F.cross_entropy(logits, yst[mq])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0); opt.step(); sched.step()
    ad.eval()
    with torch.no_grad():
        se = ad.encode(Xst); qe = ad.encode(Xqt)
        pr = F.softmax(ad.proto_head(se, qe, yst), -1).numpy()
    del ad; clean()
    return pr


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate(model, Xte, yte, Xtr, ytr, n_ep=25, n_tta=6, use_maml=True):
    model.eval()
    Xtrt = torch.tensor(Xtr, dtype=torch.float32)
    Xtet = torch.tensor(Xte, dtype=torch.float32)
    ytrt = torch.tensor(ytr, dtype=torch.long)
    rng = np.random.RandomState(42)
    i0 = np.where(ytr == 0)[0]; i1 = np.where(ytr == 1)[0]
    ap = []

    with torch.no_grad():
        se = model.encode(Xtrt); qe = model.encode(Xtet)
        sp = F.softmax(model.proto_head(se, qe, ytrt), -1).numpy()
    ap.extend([sp, sp])

    if use_maml:
        print("    MAML (12 steps)...", flush=True)
        mp = proto_maml(model, Xtr, ytr, Xte, steps=12, lr=0.01)
        ap.extend([mp, mp, mp])

    with torch.no_grad():
        sn = F.normalize(se, -1); qn = F.normalize(qe, -1)
        protos = []
        for c in range(2):
            mask = ytrt == c
            protos.append(F.normalize(sn[mask].mean(0, keepdim=True), -1).squeeze(0))
        protos = torch.stack(protos)
        for _ in range(20):
            sim = torch.mm(qn, protos.t()) * 10.0
            soft = F.softmax(sim, -1)
            np2 = []
            for c in range(2):
                qp = (soft[:, c].unsqueeze(-1) * qn).sum(0) / (soft[:, c].sum() + 1e-8)
                sp2 = sn[ytrt == c].mean(0)
                np2.append(F.normalize((0.75 * sp2 + 0.25 * qp).unsqueeze(0), -1).squeeze(0))
            protos = torch.stack(np2)
        tp = F.softmax(torch.mm(qn, protos.t()) * 10.0, -1).numpy()
    ap.extend([tp, tp])

    for _ in range(n_tta):
        with torch.no_grad():
            noisy = Xtet + torch.randn_like(Xtet) * 0.015
            se2 = model.encode(Xtrt); qe2 = model.encode(noisy)
            ap.append(F.softmax(model.proto_head(se2, qe2, ytrt), -1).numpy())

    for ei in range(n_ep):
        k = min(25, len(i0), len(i1))
        s0 = rng.choice(i0, k, replace=False); s1 = rng.choice(i1, k, replace=False)
        si = np.concatenate([s0, s1])
        if use_maml and ei < 6:
            ep_p = proto_maml(model, Xtr[si], ytr[si], Xte, steps=6, lr=0.008)
            ap.append(ep_p)
        else:
            ys = torch.tensor(ytr[si], dtype=torch.long)
            with torch.no_grad():
                se3 = model.encode(torch.tensor(Xtr[si], dtype=torch.float32))
                qe3 = model.encode(Xtet)
                ap.append(F.softmax(model.proto_head(se3, qe3, ys), -1).numpy())
    clean()
    return np.mean(ap, axis=0)[:, 1]


# ─── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(pp, yt, th=0.5):
    pd2 = (pp >= th).astype(int)
    m = {"auroc": float(roc_auc_score(yt, pp)),
         "auprc": float(average_precision_score(yt, pp)),
         "f1_macro": float(f1_score(yt, pd2, average="macro", zero_division=0)),
         "precision": float(precision_score(yt, pd2, average="macro", zero_division=0)),
         "recall": float(recall_score(yt, pd2, average="macro", zero_division=0)),
         "accuracy": float(accuracy_score(yt, pd2)),
         "mcc": float(matthews_corrcoef(yt, pd2)),
         "threshold": th}
    rng = np.random.RandomState(42); boots = []
    for _ in range(1000):
        idx = rng.choice(len(yt), len(yt), replace=True)
        if len(np.unique(yt[idx])) < 2: continue
        boots.append(roc_auc_score(yt[idx], pp[idx]))
    if boots:
        m["auroc_ci_lo"] = float(np.percentile(boots, 2.5))
        m["auroc_ci_hi"] = float(np.percentile(boots, 97.5))
    return m


def tune_th(pp, yt):
    bf, bt = 0, 0.5
    for t in np.linspace(0.25, 0.75, 101):
        f = f1_score(yt, (pp >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf = f; bt = t
    return bt, bf


# ─── Main ────────────────────────────────────────────────────────────

def run(quick=False, light=False):
    t0 = time.time()
    cfg = load_config()
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = pd.read_parquet(proj / cfg["paths"]["processed_dir"] / "thesis_dataset.parquet")
    meta = ["stay_id","subject_id","hadm_id","feasible","protocol","primary_icd10","los_hours"]
    fc = [c for c in data.columns if c not in meta]
    Xr = np.nan_to_num(data[fc].values.astype(np.float32))
    y = data["feasible"].values

    X = engineer_features_v5(Xr)
    print(f"Features: {Xr.shape[1]} -> {X.shape[1]}", flush=True)

    tri, tei = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    tri, vli = train_test_split(tri, test_size=0.2, random_state=seed, stratify=y[tri])

    sc = StandardScaler()
    X[tri] = sc.fit_transform(X[tri])
    X[vli] = sc.transform(X[vli])
    X[tei] = sc.transform(X[tei])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr, ytr = X[tri], y[tri]
    Xvl, yvl = X[vli], y[vli]
    Xte, yte = X[tei], y[tei]
    nf = X.shape[1]
    print(f"Train:{len(ytr)} Val:{len(yvl)} Test:{len(yte)} Feat:{nf}", flush=True)

    # RF teacher (smaller in light mode to save memory)
    print("\nRF teacher...", flush=True)
    n_est = 100 if light else 300
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=28, min_samples_split=5, min_samples_leaf=2,
                                max_features="log2", class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(np.nan_to_num(Xr[tri]), ytr)
    rf_tr = rf.predict_proba(np.nan_to_num(Xr[tri]))[:, 1]
    rf_vl = rf.predict_proba(np.nan_to_num(Xr[vli]))[:, 1]
    rf_te = rf.predict_proba(np.nan_to_num(Xr[tei]))[:, 1]
    rf_imp = rf.feature_importances_
    padded_imp = np.zeros(nf, dtype=np.float32)
    padded_imp[:len(rf_imp)] = rf_imp
    print(f"  RF test AUROC: {roc_auc_score(yte, rf_te):.4f}", flush=True)

    # Seeds and model size: full=7, quick=1, light=2 with smaller model
    if light:
        seeds = [seed, seed + 7]
        nl, dff = 2, 256
        p1_ep, p2_ep, p3_ep = 12, 10, 600
        n_ep_vl, n_ep_te, n_tta = 5, 8, 2
        use_maml = False
        batch_size = 64
    elif quick:
        seeds = [seed]
        nl, dff = 4, 512
        p1_ep, p2_ep, p3_ep = 8, 6, 300
        n_ep_vl, n_ep_te, n_tta = 3, 5, 2
        use_maml = True
        batch_size = 128
    else:
        seeds = [seed, seed+7, seed+13, seed+23, seed+37, seed+47, seed+59]
        nl, dff = 4, 512
        p1_ep, p2_ep, p3_ep = 40, 35, 2500
        n_ep_vl, n_ep_te, n_tta = 15, 25, 6
        use_maml = True
        batch_size = 128

    avl = []; ate = []; vemb_list = []; temb_list = []
    print(f"\n{'='*60}\nTraining BEST model: {len(seeds)} seeds, nl={nl}, dff={dff}" +
          (f" (light mode)" if light else "") + f"\n{'='*60}", flush=True)

    for si, s in enumerate(seeds):
        print(f"\n--- Model {si+1}/{len(seeds)} (s={s}) ---", flush=True)
        torch.manual_seed(s); np.random.seed(s)
        m = FewShotBestModel(nf, d=128, nh=8, nl=nl, dff=dff, drop=0.12)
        phase1_supcon_deep_kd(m, Xtr, ytr, rf_tr, padded_imp, epochs=p1_ep, lr=5e-4, batch_size=batch_size)
        phase2_supervised_kd(m, Xtr, ytr, rf_tr, Xvl, yvl, epochs=p2_ep, lr=3e-4)
        phase3_episodic(m, Xtr, ytr, n_episodes=p3_ep)
        print("  Eval...", flush=True)
        vp = evaluate(m, Xvl, yvl, Xtr, ytr, n_ep=n_ep_vl, n_tta=n_tta, use_maml=use_maml)
        tp = evaluate(m, Xte, yte, Xtr, ytr, n_ep=n_ep_te, n_tta=n_tta, use_maml=use_maml)
        avl.append(vp); ate.append(tp)
        with torch.no_grad():
            vemb_list.append(m.encode(torch.tensor(Xvl, dtype=torch.float32)).numpy().copy())
            temb_list.append(m.encode(torch.tensor(Xte, dtype=torch.float32)).numpy().copy())
        del m
        clean()
        print(f"  AUROC: {roc_auc_score(yte, tp):.4f}", flush=True)

    evl = np.mean(avl, 0); ete = np.mean(ate, 0)
    fs_auroc = roc_auc_score(yte, ete)
    print(f"\nBEST Ensemble AUROC: {fs_auroc:.4f}", flush=True)

    # Enhanced stacking: FS + RF + interactions + PCA embeddings + polynomial
    print("\nEnhanced stacking...", flush=True)
    vemb = np.mean(vemb_list, 0)
    temb = np.mean(temb_list, 0)
    del vemb_list, temb_list
    clean()
    pca = PCA(n_components=min(8 if light else 16, vemb.shape[1]))
    vpca = pca.fit_transform(vemb); tpca = pca.transform(temb)

    # Base features
    vfeat = np.column_stack([evl, rf_vl, evl * rf_vl, np.abs(evl - rf_vl), evl**2, rf_vl**2, vpca])
    tfeat = np.column_stack([ete, rf_te, ete * rf_te, np.abs(ete - rf_te), ete**2, rf_te**2, tpca])

    stacker = CalibratedClassifierCV(
        LogisticRegression(C=0.8, max_iter=2000, random_state=42), cv=3, method="isotonic")
    stacker.fit(vfeat, yvl)
    stacked_te = stacker.predict_proba(tfeat)[:, 1]
    stacked_vl = stacker.predict_proba(vfeat)[:, 1]
    clean()

    # Results
    print(f"\n{'='*60}\nFINAL BEST RESULTS\n{'='*60}", flush=True)

    t1, _ = tune_th(evl, yvl)
    mfs = compute_metrics(ete, yte, t1)
    mfs["strategy"] = "BEST_FewShot_GFA_CrossAttn"
    print(f"\n[A] FewShot BEST (GFA + CrossAttn + Proto-MAML):", flush=True)
    print(f"    AUROC: {mfs['auroc']:.4f} [{mfs.get('auroc_ci_lo',0):.4f}, {mfs.get('auroc_ci_hi',0):.4f}]", flush=True)
    print(f"    F1: {mfs['f1_macro']:.4f}  MCC: {mfs['mcc']:.4f}  Acc: {mfs['accuracy']:.4f}", flush=True)

    t2, _ = tune_th(stacked_vl, yvl)
    mst = compute_metrics(stacked_te, yte, t2)
    mst["strategy"] = "BEST_Stacked"
    print(f"\n[B] BEST Stacked (FewShot+RF+Embeddings+Poly):", flush=True)
    print(f"    AUROC: {mst['auroc']:.4f} [{mst.get('auroc_ci_lo',0):.4f}, {mst.get('auroc_ci_hi',0):.4f}]", flush=True)
    print(f"    F1: {mst['f1_macro']:.4f}  MCC: {mst['mcc']:.4f}  Acc: {mst['accuracy']:.4f}", flush=True)

    print("\nPer-seed AUROC:", flush=True)
    for si, tp in enumerate(ate):
        ms = compute_metrics(tp, yte)
        print(f"  Seed {seeds[si]}: AUROC={ms['auroc']:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s", flush=True)

    out = {"fewshot_best": mfs, "stacked_best": mst, "seeds": seeds}
    with open(proj / "results" / "exp_fewshot_best.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved to results/exp_fewshot_best.json", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 seed, fewer epochs for fast validation (~5 min)")
    p.add_argument("--light", action="store_true", help="2 seeds, smaller model, no MAML — low memory (~15 min)")
    args = p.parse_args()
    run(quick=args.quick, light=args.light)
