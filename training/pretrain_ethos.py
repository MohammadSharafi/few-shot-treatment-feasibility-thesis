#!/usr/bin/env python3
"""
Self-supervised pretraining for ETHOS encoder.
Tasks: (a) Masked Symptom Modeling, (b) Contrastive Learning (same node = positive).
Saves: models/ethos_pretrained.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def assign_node(icd_str):
    icd = str(icd_str).replace(".", "").upper()
    for node_name, prefixes in [
        ("node_1_sepsis", ["A40", "A41", "R65"]),
        ("node_2_cardiac", ["I21", "I22", "I50", "I48"]),
        ("node_3_respiratory", ["J18", "J44", "J96", "J80"]),
        ("node_4_renal", ["N17", "N18", "N19"]),
        ("node_5_diabetes", ["E10", "E11", "E13"]),
    ]:
        for p in prefixes:
            if icd.startswith(p.replace(".", "")):
                return node_name
    return "node_0_other"

def to_tokens(X, device):
    """Convert (n, 25) to (n, 25, 4) tokens."""
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    n, d = X.shape
    tokens = torch.zeros(n, 25, 4, device=device)
    for j in range(min(25, d)):
        tokens[:, j, 0] = (j + 1) / 25.0
        tokens[:, j, 1] = X[:, j]
        tokens[:, j, 2] = 0.0
        tokens[:, j, 3] = 0.0 if j < 15 else (0.5 if 22 <= j < 25 else 1.0)
    return tokens

class MaskedMLMHead(nn.Module):
    """Predict masked symptom_id (classification) and intensity (regression)."""
    def __init__(self, d_model, vocab_size=23):
        super().__init__()
        self.sid_head = nn.Linear(d_model, vocab_size)
        self.int_head = nn.Linear(d_model, 1)
    def forward(self, h):
        return self.sid_head(h), self.int_head(h).squeeze(-1)

def main():
    cfg = load_config()
    proj = Path(__file__).parent.parent
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data - same split as baselines
    data = pd.read_parquet(Path(paths["processed_dir"]) / "thesis_dataset.parquet")
    from utils.feature_engineering import add_engineered_features
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in data.columns if c not in meta]
    X = data[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = add_engineered_features(X)
    y = data["feasible"].values
    
    train_idx, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    X_train = X[train_idx]
    
    data_df = data.iloc[train_idx].copy()
    data_df["node"] = data_df["primary_icd10"].apply(assign_node)
    node_to_id = {n: i for i, n in enumerate(sorted(data_df["node"].unique()))}
    node_ids = np.array([node_to_id[n] for n in data_df["node"]])
    
    tokens = to_tokens(X_train, device)
    enc_cfg = cfg["encoder"]
    
    from models.encoder import ETHOSEncoder
    encoder = ETHOSEncoder(
        vocab_size=enc_cfg["vocab_size"],
        d_model=enc_cfg["d_model"],
        nhead=enc_cfg["nhead"],
        num_layers=enc_cfg["num_layers"],
        d_ff=enc_cfg["d_ff"],
        dropout=enc_cfg["dropout"],
    ).to(device)
    mlm_head = MaskedMLMHead(enc_cfg["d_model"], enc_cfg["vocab_size"]).to(device)
    
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(mlm_head.parameters()), lr=1e-4, weight_decay=0.01)
    
    n_epochs = 30
    mask_ratio = 0.15
    batch_size = 64
    n_samples = len(tokens)
    
    encoder.train()
    mlm_head.train()
    best_loss = float("inf")
    patience = 5
    no_improve = 0
    
    for ep in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            x = tokens[idx].clone()
            B, S, D = x.shape
            
            # Mask 15% of tokens (x has 25 tokens)
            mask = torch.rand(B, 25, device=device) < mask_ratio
            
            x_masked = x.clone()
            x_masked[mask] = 0.0
            x_masked[mask, 0] = 0.0
            
            h_full = encoder(x_masked, return_full=True)
            h_masked = h_full[:, 1:, :][mask]
            
            sid_logits, int_pred = mlm_head(h_masked)
            sid_true = (x[mask, 0] * 24.99).long().clamp(0, enc_cfg["vocab_size"] - 1)
            int_true = x[mask, 1]
            
            loss_sid = F.cross_entropy(sid_logits, sid_true)
            loss_int = F.mse_loss(int_pred, int_true)
            loss_mlm = loss_sid + 0.5 * loss_int
            
            # Contrastive: same node = positive (InfoNCE)
            emb = encoder(x_masked)
            emb_norm = F.normalize(emb, p=2, dim=-1)
            node_batch = torch.tensor(node_ids[idx.cpu().numpy()], device=device, dtype=torch.long)
            sim = torch.mm(emb_norm, emb_norm.t()) / 0.1
            pos_mask = (node_batch.unsqueeze(0) == node_batch.unsqueeze(1)).float()
            pos_mask = pos_mask * (1 - torch.eye(B, device=device))
            n_pos = pos_mask.sum(1).clamp(min=1)
            loss_contr = 0.0
            for i in range(B):
                pos_idx = (pos_mask[i] > 0.5).nonzero(as_tuple=True)[0]
                if len(pos_idx) == 0:
                    continue
                logits_i = sim[i]
                logits_i = logits_i - logits_i[i].detach()
                log_prob_pos = F.log_softmax(logits_i, dim=0)[pos_idx].mean()
                loss_contr = loss_contr - log_prob_pos
            loss_contr = loss_contr / max(1, (pos_mask.sum(1) > 0).sum().item())
            
            loss = loss_mlm + 0.5 * loss_contr
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
        
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            out_path = proj / "models" / "ethos_pretrained.pt"
            torch.save({"encoder": encoder.state_dict()}, out_path)
        else:
            no_improve += 1
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1}/{n_epochs} loss={avg_loss:.4f}")
        if no_improve >= patience:
            print(f"Early stop at epoch {ep+1}")
            break
    
    print(f"Pretraining done. Saved to models/ethos_pretrained.pt")

if __name__ == "__main__":
    main()
