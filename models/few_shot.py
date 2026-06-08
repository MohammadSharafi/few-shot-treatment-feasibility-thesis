"""
Few-Shot Treatment Learning via Prototypical Networks
Uses ETHOS encoder embedding space. fit(), predict(), evaluate() interface.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml

from .encoder import ETHOSEncoder

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

class ProjectionMLP(nn.Module):
    """Non-linear projection from encoder space to metric space."""
    def __init__(self, d_in=128, d_hidden=256, d_out=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)

class PrototypicalHead(nn.Module):
    """Projects embedding to logits for few-shot classification with learnable temperature."""
    def __init__(self, d_model=128, n_way=2, dropout=0.3, temperature=10.0, use_projection=True):
        super().__init__()
        self.n_way = n_way
        self.use_projection = use_projection
        proj_dim = 64 if use_projection else d_model
        self.projection = ProjectionMLP(d_model, d_model * 2, proj_dim, dropout) if use_projection else None
        self.dropout = nn.Dropout(dropout)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        
    def forward(self, support_emb, query_emb, support_labels):
        if self.use_projection and self.projection is not None:
            support_emb = self.projection(support_emb)
            query_emb = self.projection(query_emb)
        else:
            support_emb = self.dropout(support_emb)
            query_emb = self.dropout(query_emb)
        support_emb = F.normalize(support_emb, p=2, dim=-1)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        prototypes = []
        for c in range(self.n_way):
            mask = support_labels == c
            if mask.sum() > 0:
                prototypes.append(support_emb[mask].mean(dim=0))
            else:
                prototypes.append(support_emb.mean(dim=0))
        prototypes = torch.stack(prototypes)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.1, max=100.0)
        logits = torch.mm(query_emb, prototypes.t()) / temperature
        return logits

class FewShotModel(nn.Module):
    """ETHOS Encoder + Prototypical Head."""
    def __init__(self, cfg=None, encoder_kwargs=None, pretrained_path=None, freeze_encoder_layers=0):
        super().__init__()
        cfg = cfg or load_config()
        enc_cfg = cfg["encoder"]
        fs_cfg = cfg["few_shot"]
        ek = encoder_kwargs or {}
        self.encoder = ETHOSEncoder(
            vocab_size=enc_cfg["vocab_size"],
            d_model=enc_cfg["d_model"],
            nhead=enc_cfg["nhead"],
            num_layers=enc_cfg["num_layers"],
            d_ff=enc_cfg["d_ff"],
            dropout=enc_cfg["dropout"],
            use_cooccur=ek.get("use_cooccur", True),
            use_intensity=ek.get("use_intensity", True),
            use_temporal=ek.get("use_temporal", True),
        )
        if pretrained_path and Path(pretrained_path).exists():
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "encoder" in state:
                self.encoder.load_state_dict(state["encoder"], strict=False)
            del state
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.encoder.layers):
                if i < freeze_encoder_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
        temp = fs_cfg.get("temperature", 10.0)
        use_proj = fs_cfg.get("use_projection", True)
        self.head = PrototypicalHead(d_model=enc_cfg["d_model"], n_way=fs_cfg["n_way"],
                                     temperature=temp, use_projection=use_proj)
        self.n_way = fs_cfg["n_way"]
        
    def forward(self, support_x, query_x, support_y):
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)
        return self.head(support_emb, query_emb, support_y)

class FewShotPredictor:
    """fit(), predict(), evaluate() interface for few-shot model."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            import warnings
            warnings.warn("GPU not available; running on CPU.")
    
    def fit(self, X, y, n_episodes=500, n_support=5, n_query=15, encoder_kwargs=None,
            pretrained_path=None, use_pretrained=True, freeze_encoder_layers=0, encoder_lr=None, head_lr=None,
            k_curriculum=None, node_ids=None, hard_node_weights=None):
        """Episodic training. k_curriculum: list of K values per stage, e.g. [10, 5, (3,5,7)]."""
        cfg = self.cfg["few_shot"]
        n_episodes = min(n_episodes, cfg["n_episodes_train"])
        n_support = cfg["n_support"]
        n_query = cfg["n_query"]
        
        _pt = Path(__file__).parent.parent / "models" / "ethos_pretrained.pt"
        load_pretrained = use_pretrained and (pretrained_path or _pt.exists())
        pretrained_path = pretrained_path if pretrained_path else (_pt if load_pretrained else None)
        self.model = FewShotModel(self.cfg, encoder_kwargs=encoder_kwargs,
            pretrained_path=pretrained_path, freeze_encoder_layers=freeze_encoder_layers).to(self.device)
        enc_lr = encoder_lr if encoder_lr is not None else cfg["lr"] * 0.1
        hd_lr = head_lr if head_lr is not None else cfg["lr"]
        opt = torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": enc_lr},
            {"params": self.model.head.parameters(), "lr": hd_lr},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, n_episodes), eta_min=1e-5
        )
        
        X = self._to_tokens(X)
        y = torch.tensor(np.asarray(y), dtype=torch.long, device=self.device)
        classes = torch.unique(y)
        if len(classes) < 2:
            return self
        
        k_schedule = k_curriculum if k_curriculum else [n_support]
        n_per_stage = max(1, n_episodes // len(k_schedule))
        
        for ep in range(n_episodes):
            self.model.train()
            stage = min(ep // max(1, n_per_stage), len(k_schedule) - 1)
            k_use = k_schedule[stage]
            if isinstance(k_use, (list, tuple)):
                k_use = int(np.random.choice(k_use))
            k_sup = min(k_use, n_support + n_query, 10)
            k_q = min(n_query, 20)
            
            c0, c1 = np.random.choice(classes.cpu().numpy(), 2, replace=False)
            idx0 = (y == c0).nonzero(as_tuple=True)[0]
            idx1 = (y == c1).nonzero(as_tuple=True)[0]
            if node_ids is not None and hard_node_weights is not None:
                w0 = torch.tensor([hard_node_weights.get(node_ids[i.item()], 1.0) for i in idx0.cpu()], device=self.device)
                w1 = torch.tensor([hard_node_weights.get(node_ids[i.item()], 1.0) for i in idx1.cpu()], device=self.device)
                p0 = (w0 / w0.sum()).cpu().numpy()
                p1 = (w1 / w1.sum()).cpu().numpy()
                sel0 = np.random.choice(len(idx0), min(k_sup + k_q, len(idx0)), replace=False, p=p0)
                sel1 = np.random.choice(len(idx1), min(k_sup + k_q, len(idx1)), replace=False, p=p1)
                idx0 = idx0[sel0]
                idx1 = idx1[sel1]
            else:
                idx0 = idx0[torch.randperm(len(idx0))[:min(k_sup + k_q, len(idx0))]]
                idx1 = idx1[torch.randperm(len(idx1))[:min(k_sup + k_q, len(idx1))]]
            n0, n1 = len(idx0), len(idx1)
            if n0 < k_sup or n1 < k_sup:
                continue
            support_idx = torch.cat([idx0[:k_sup], idx1[:k_sup]])
            q0 = min(k_q, n0 - k_sup)
            q1 = min(k_q, n1 - k_sup)
            if q0 < 1 and q1 < 1:
                continue
            query_idx = torch.cat([idx0[k_sup:k_sup+q0], idx1[k_sup:k_sup+q1]])
            if len(query_idx) < 2:
                continue
            
            support_x = X[support_idx]
            query_x = X[query_idx]
            support_y = torch.cat([torch.zeros(k_sup, device=self.device, dtype=torch.long),
                                 torch.ones(k_sup, device=self.device, dtype=torch.long)])
            q0_actual = min(q0, len(idx0) - k_sup)
            q1_actual = min(q1, len(idx1) - k_sup)
            query_y = torch.cat([torch.zeros(q0_actual, device=self.device, dtype=torch.long),
                                torch.ones(q1_actual, device=self.device, dtype=torch.long)])[:len(query_idx)]
            if len(query_y) != len(query_idx):
                continue
            
            logits = self.model(support_x, query_x, support_y)
            # Focal loss + class weighting for imbalance (58% infeasible, 42% feasible)
            use_focal = self.cfg["few_shot"].get("use_focal_loss", False)
            pos_weight = self.cfg["few_shot"].get("pos_weight", None)
            weight = torch.tensor([1.0, pos_weight], device=logits.device) if pos_weight else None
            if use_focal:
                gamma = self.cfg["few_shot"].get("focal_gamma", 2.0)
                ce = F.cross_entropy(logits, query_y, weight=weight, reduction="none")
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** gamma * ce).mean()
            elif weight is not None:
                loss = F.cross_entropy(logits, query_y, weight=weight)
            else:
                loss = F.cross_entropy(logits, query_y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
        # Optional: short supervised finetune (adapts to full-support eval)
        n_finetune = self.cfg["few_shot"].get("n_finetune_epochs", 0)
        if n_finetune > 0 and len(X) <= 3000:
            X_t = self._to_tokens(X)
            idx0 = (y == 0).nonzero(as_tuple=True)[0]
            idx1 = (y == 1).nonzero(as_tuple=True)[0]
            for _ in range(n_finetune):
                self.model.train()
                k = min(25, len(idx0), len(idx1))
                if k < 2:
                    break
                s0 = idx0[torch.randperm(len(idx0))[:k]]
                s1 = idx1[torch.randperm(len(idx1))[:k]]
                sup_idx = torch.cat([s0, s1])
                q0 = idx0[torch.randperm(len(idx0))[k:k+32]]
                q1 = idx1[torch.randperm(len(idx1))[k:k+32]]
                q_idx = torch.cat([q0, q1])[:64]
                if len(q_idx) < 4:
                    continue
                support_x, query_x = X_t[sup_idx], X_t[q_idx]
                support_y, query_y = y[sup_idx], y[q_idx]
                logits = self.model(support_x, query_x, support_y)
                loss = F.cross_entropy(logits, query_y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
        
        return self
    
    def _to_tokens(self, X):
        """Convert 2D flat features to 3D tokens (n, 25, 4). Supports 22 or 25 cols."""
        X = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        if X.ndim == 2:
            n, d = X.shape
            tokens = torch.zeros(n, 25, 4, device=self.device)
            for j in range(min(25, d)):
                tokens[:, j, 0] = (j + 1) / 25.0
                tokens[:, j, 1] = X[:, j]
                tokens[:, j, 2] = 0.0
                tokens[:, j, 3] = 0.0 if j < 15 else (0.5 if 22 <= j < 25 else 1.0)
            return tokens
        return X
    
    def predict(self, X, X_support=None, y_support=None):
        """Predict using prototypes from support set."""
        self.model.eval()
        with torch.no_grad():
            X = self._to_tokens(X)
            emb = self.model.encoder(X)
            if X_support is not None and y_support is not None:
                X_sup = self._to_tokens(X_support)
                y_sup = torch.tensor(y_support, dtype=torch.long, device=self.device)
                sup_emb = self.model.encoder(X_sup)
                logits = self.model.head(sup_emb, emb, y_sup)
            else:
                # No support: use training data or random (caller should provide support)
                logits = torch.randn(emb.shape[0], 2, device=self.device)
            pred = logits.argmax(dim=-1).cpu().numpy()
        return pred
    
    def evaluate(self, X, y, X_support=None, y_support=None, n_eval_episodes=20):
        """Return dict of metrics. Multi-episode averaging for stable results.
        Pass X_train,y_train as support for supervised evaluation."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
        y = np.asarray(y)
        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        
        if X_support is not None and y_support is not None:
            probs = self._predict_proba(X, X_support, y_support)
            pred = probs.argmax(axis=1)
            y_eval = y
        else:
            all_probs = []
            all_y = []
            all_query_idx = []
            rng = np.random.RandomState(self.cfg.get("seed", 42))
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            k = min(10, len(idx0) // 3, len(idx1) // 3)
            if k < 2:
                k = 2
            for ep in range(n_eval_episodes):
                sup0 = rng.choice(idx0, min(k, len(idx0)), replace=False)
                sup1 = rng.choice(idx1, min(k, len(idx1)), replace=False)
                support_idx = np.concatenate([sup0, sup1])
                query_idx = np.array([i for i in range(len(y)) if i not in support_idx])
                if len(query_idx) < 10:
                    continue
                X_sup = X[support_idx]
                y_sup = y[support_idx]
                X_q = X[query_idx]
                ep_probs = self._predict_proba(X_q, X_sup, y_sup)
                all_probs.append(ep_probs)
                all_y.append(y[query_idx])
                all_query_idx.append(query_idx)
            if not all_probs:
                return {"accuracy": 0.5, "f1_macro": 0.5, "auroc": 0.5, "auprc": 0.5, "mcc": 0.0}
            avg_probs = np.mean(all_probs, axis=0)
            probs = avg_probs
            pred = probs.argmax(axis=1)
            y_eval = all_y[0]
        
        if len(np.unique(y_eval)) < 2:
            auroc, auprc = 0.5, 0.5
        else:
            try:
                auroc = roc_auc_score(y_eval, probs[:, 1])
                auprc = average_precision_score(y_eval, probs[:, 1])
            except Exception:
                auroc, auprc = 0.5, 0.5
        return {
            "accuracy": accuracy_score(y_eval, pred),
            "f1_macro": f1_score(y_eval, pred, average="macro", zero_division=0),
            "auroc": auroc,
            "auprc": auprc,
            "mcc": matthews_corrcoef(y_eval, pred) if len(np.unique(y_eval)) >= 2 else 0.0,
        }
    
    def _predict_proba(self, X, X_support, y_support, query_chunk: int | None = None):
        """Encode support and queries in chunks to avoid huge CPU/GPU batches."""
        self.model.eval()
        if query_chunk is None:
            query_chunk = 48 if self.device.type == "cpu" else 512
        enc_chunk = 64 if self.device.type == "cpu" else 512
        with torch.no_grad():
            X_sup = self._to_tokens(X_support)
            y_sup = torch.tensor(y_support, dtype=torch.long, device=self.device)
            if X_sup.shape[0] <= enc_chunk:
                support_emb = self.model.encoder(X_sup)
            else:
                parts = []
                for s in range(0, X_sup.shape[0], enc_chunk):
                    parts.append(self.model.encoder(X_sup[s : s + enc_chunk]))
                support_emb = torch.cat(parts, dim=0)
            X = self._to_tokens(X)
            outs = []
            for start in range(0, X.shape[0], query_chunk):
                q = X[start : start + query_chunk]
                query_emb = self.model.encoder(q)
                logits = self.model.head(support_emb, query_emb, y_sup)
                outs.append(F.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
