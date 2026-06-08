"""
ETHOS Encoder: Transformer with symptom-specific modifications
- Learnable co-occurrence bias matrix added to attention scores
- Attention weights * token intensity
- Attention weights * exp(-0.1 * time_delta)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ETHOSEncoderLayer(nn.Module):
    """Single transformer layer with ETHOS modifications."""
    def __init__(self, d_model, nhead, d_ff=512, dropout=0.1, vocab_size=23, use_cooccur=True, use_intensity=True, use_temporal=True):
        super().__init__()
        self.use_cooccur = use_cooccur
        self.use_intensity = use_intensity
        self.use_temporal = use_temporal
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cooccur_bias = nn.Parameter(torch.zeros(vocab_size, vocab_size))
        
    def forward(self, src, symptom_ids, intensity, time_delta):
        # src: (B, S, d_model)
        B, S, D = src.shape
        # Self-attention with co-occurrence bias
        # attn_weights from multihead: we need to add bias and scale
        # MultiheadAttention returns (out, attn_weights)
        attn_out, attn_weights = self.self_attn(src, src, src, need_weights=True)
        # attn_weights: (B, nhead, S, S)
        # Co-occurrence bias
        if self.use_cooccur:
            sid = symptom_ids.clamp(0, self.cooccur_bias.shape[0] - 1)
            bias_2d = self.cooccur_bias[sid[:, :, None], sid[:, None, :]]
            bias_proj = bias_2d.mean(dim=-1, keepdim=True).expand(B, S, D) * 0.01
            attn_out = attn_out + self.dropout(bias_proj)
        # Scale by intensity and time_delta
        if self.use_intensity and self.use_temporal:
            scale = (intensity * torch.exp(-0.1 * time_delta)).unsqueeze(-1)
        elif self.use_intensity:
            scale = intensity.unsqueeze(-1)
        elif self.use_temporal:
            scale = torch.exp(-0.1 * time_delta).unsqueeze(-1)
        else:
            scale = torch.ones(B, S, 1, device=src.device)
        attn_out = attn_out * scale
        src = src + self.dropout(attn_out)
        src = self.norm1(src)
        src = src + self.dropout(self.ff(src))
        src = self.norm2(src)
        return src

class ETHOSEncoder(nn.Module):
    """
    Transformer Encoder with ETHOS modifications.
    Output: CLS token -> [batch, d_model] patient representation
    """
    def __init__(self, vocab_size=23, d_model=128, nhead=8, num_layers=6, d_ff=512, dropout=0.1, use_cooccur=True, use_intensity=True, use_temporal=True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.input_proj = nn.Linear(4, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            ETHOSEncoderLayer(d_model, nhead, d_ff, dropout, vocab_size, use_cooccur, use_intensity, use_temporal)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, intensity=None, time_delta=None, return_full=False):
        B, S, _ = x.shape
        symptom_ids = x[:, :, 0].long().clamp(0, self.vocab_size - 1)
        if intensity is None:
            intensity = x[:, :, 1]
        if time_delta is None:
            time_delta = x[:, :, 2]
        
        h = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        intensity = torch.cat([torch.ones(B, 1, device=x.device), intensity], dim=1)
        time_delta = torch.cat([torch.zeros(B, 1, device=x.device), time_delta], dim=1)
        symptom_ids = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), symptom_ids], dim=1)
        
        for layer in self.layers:
            h = layer(h, symptom_ids, intensity, time_delta)
        
        if return_full:
            return h
        return h[:, 0, :]
