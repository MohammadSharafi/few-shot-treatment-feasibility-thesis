#!/usr/bin/env python3
"""
Minimal GPT-2-style causal transformer for Patient Health Timeline tokens.

This fills the key methodological gap relative to ETHOS (Renc et al., 2024), which
pretrains a GPT-2 autoregressive model on tokenized timelines. The earlier code in
this repo only had a BERT-style encoder and a denoising-autoencoder pretraining;
ETHOS's actual recipe is next-token (autoregressive) language-model pretraining.

This module provides:
  - EthosGPT: a small causal transformer (token + positional embeddings, pre-LN
    blocks with causal self-attention, tied LM head) for next-token pretraining.
  - .embed(seq): mean-pooled hidden state over non-pad tokens -> patient embedding
    for downstream few-shot / linear-probe / hybrid use.

Pure PyTorch; sized for the small PHT vocabulary (~50 tokens). Train with
scripts/pretrain_ethos_gpt.py and use with scripts/exp_fewshot_pretrained.py.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head; self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # .reshape (safe superset of .view): q/k/v are non-contiguous split views.
        q = q.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EthosGPT(nn.Module):
    def __init__(self, vocab_size, max_len=256, d_model=128, n_head=4, n_layer=4,
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id; self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def backbone(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)              # (B, T, d_model)

    def forward(self, idx, targets=None):
        h = self.backbone(idx)
        logits = self.head(h)
        loss = None
        if targets is not None:
            # .reshape (not .view): targets = xb[:, 1:] is a non-contiguous slice,
            # and view() requires contiguous memory; reshape copies if needed.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=self.pad_id)
        return logits, loss

    @torch.no_grad()
    def embed(self, idx):
        """Mean-pooled hidden state over non-pad tokens -> (B, d_model) patient embedding."""
        h = self.backbone(idx)
        mask = (idx != self.pad_id).float().unsqueeze(-1)
        summed = (h * mask).sum(1)
        cnt = mask.sum(1).clamp(min=1.0)
        return summed / cnt
