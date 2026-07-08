"""
Symptom-Token Transformer (STT)
================================
A faithful realisation of the thesis title — "prediction of treatment feasibility
with symptom tokens" — on the *enriched* symptom panel.

Where the original tokenizer put a single median value in each token, here every
one of the 22 physiological signals becomes a **symptom token** carrying a small
vector of clinically meaningful channels (level, trend, variability, extremes,
measurement density, missingness). A learnable per-signal embedding is added, a
23rd *context token* injects demographics/comorbidity, and a Transformer encoder
learns cross-symptom interactions ("symptom syndromes"). A CLS token pools the
sequence into a patient embedding used for both supervised classification and
prototypical few-shot.

This module is self-contained (standard nn.TransformerEncoder) and runs on
Apple MPS / CUDA / CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 15 labs + 7 vitals, in the config order.
LAB_ITEMIDS = [50912, 50971, 50983, 50902, 50931, 50893, 50960, 50811, 51265,
               51300, 50813, 50861, 50878, 51006, 50882]
VITAL_ITEMIDS = [220045, 220050, 220051, 220052, 223761, 220277, 220210]
ALL_ITEMIDS = LAB_ITEMIDS + VITAL_ITEMIDS
# Per-signal token channels drawn from the enriched aggregates.
TOKEN_STATS = ["mean", "std", "min", "max", "slope", "last", "n", "missing"]
N_CHANNELS = len(TOKEN_STATS)          # 8
N_SIGNALS = len(ALL_ITEMIDS)           # 22
CONTEXT_COLS = ["age", "sex_male", "n_diagnoses", "adm_emergency",
                "adm_elective", "ins_medicare", "ins_medicaid", "married"]


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TokenScaler:
    """Robust per-channel standardisation fit on the training rows only."""
    center: np.ndarray = field(default=None)
    scale: np.ndarray = field(default=None)
    ctx_center: np.ndarray = field(default=None)
    ctx_scale: np.ndarray = field(default=None)


def build_tokens(df, scaler: TokenScaler | None = None, fit: bool = False):
    """Enriched dataframe -> (tokens[n,23,C], scaler).

    Token j (0..21) holds signal-level channels; token 22 is the context token.
    Continuous channels are median/IQR standardised; `missing` and binary context
    flags are passed through. `n` (measurement count) is log1p-compressed first.
    """
    n = len(df)
    raw = np.zeros((n, N_SIGNALS, N_CHANNELS), dtype=np.float32)
    for si, itemid in enumerate(ALL_ITEMIDS):
        for ci, stat in enumerate(TOKEN_STATS):
            col = f"{itemid}_{stat}"
            if col in df.columns:
                v = df[col].to_numpy(dtype=np.float32)
                if stat == "n":
                    v = np.log1p(np.nan_to_num(v, nan=0.0))
                raw[:, si, ci] = v

    cont_mask = np.array([s not in ("missing",) for s in TOKEN_STATS])
    if fit or scaler is None:
        flat = raw.reshape(-1, N_CHANNELS)
        center = np.nanmedian(flat, axis=0)
        q75 = np.nanpercentile(flat, 75, axis=0)
        q25 = np.nanpercentile(flat, 25, axis=0)
        scale = np.where((q75 - q25) > 1e-6, (q75 - q25), 1.0)
        scaler = TokenScaler(center=center, scale=scale)

    tok = raw.copy()
    for ci in range(N_CHANNELS):
        if cont_mask[ci]:
            tok[:, :, ci] = (raw[:, :, ci] - scaler.center[ci]) / scaler.scale[ci]
    tok = np.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)
    tok = np.clip(tok, -6, 6)

    # Context token
    ctx = np.zeros((n, N_CHANNELS), dtype=np.float32)
    for ci, col in enumerate(CONTEXT_COLS):
        if col in df.columns:
            ctx[:, ci] = np.nan_to_num(df[col].to_numpy(dtype=np.float32), nan=0.0)
    if fit or scaler.ctx_center is None:
        cont_ctx = np.array([c in ("age", "n_diagnoses") for c in CONTEXT_COLS])
        cc = np.where(cont_ctx, np.nanmedian(ctx, axis=0), 0.0)
        iqr = np.nanpercentile(ctx, 75, axis=0) - np.nanpercentile(ctx, 25, axis=0)
        cs = np.where(cont_ctx & (iqr > 1e-6), iqr, 1.0)
        scaler.ctx_center, scaler.ctx_scale = cc, cs
    ctx = (ctx - scaler.ctx_center) / scaler.ctx_scale
    ctx = np.clip(np.nan_to_num(ctx), -6, 6)

    tokens = np.concatenate([tok, ctx[:, None, :]], axis=1)  # (n, 23, C)
    return tokens.astype(np.float32), scaler


class SymptomTokenTransformer(nn.Module):
    def __init__(self, n_tokens: int = N_SIGNALS + 1, n_channels: int = N_CHANNELS,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.n_tokens = n_tokens
        self.value_proj = nn.Linear(n_channels, d_model)
        # Learnable identity embedding per token slot (each signal + context).
        self.token_embed = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_tokens, C) -> patient embedding (B, d_model)."""
        B = x.shape[0]
        h = self.value_proj(x) + self.token_embed.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        return self.norm(h[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)
