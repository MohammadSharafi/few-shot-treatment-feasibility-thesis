#!/usr/bin/env python3
"""
GPT-2-style autoregressive pretraining of EthosGPT on PHT token sequences.

This is the ETHOS recipe: next-token (language-model) pretraining over tokenized
patient timelines, UNSUPERVISED (no labels). The resulting representation is then
used downstream (scripts/exp_fewshot_pretrained.py).

Input : data/processed_full_cohort/tokens_ethos_seq.npy (+ vocab json)
Output: results/canonical/ethos_gpt.pt  (pretrained weights + config)

Run where torch is installed:
  python scripts/pretrain_ethos_gpt.py --epochs 8
GPU is used automatically if available.
"""
from __future__ import annotations
import os, json, argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))
from models.ethos_gpt import EthosGPT

PROC = Path(os.environ.get("PROC_DIR", ROOT / "data/processed_full_cohort"))
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)

    seq = np.load(PROC / "tokens_ethos_seq.npy")
    vocab = json.load(open(PROC / "tokens_ethos_vocab.json"))
    V = vocab["vocab_size"]; max_len = seq.shape[1]
    # train split by row (stay) — pretraining is unsupervised so any split is fine
    n = len(seq); idx = np.arange(n); rng = np.random.default_rng(SEED); rng.shuffle(idx)
    tr = idx[: int(0.9 * n)]
    X = torch.from_numpy(seq[tr].astype(np.int64))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EthosGPT(V, max_len=max_len, d_model=args.d_model, n_head=args.n_head,
                     n_layer=args.n_layer).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dl = DataLoader(TensorDataset(X), batch_size=args.batch, shuffle=True, drop_last=True)
    print(f"Pretraining EthosGPT: vocab={V}, seq_len={max_len}, params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M, device={dev}")

    model.train()
    for ep in range(args.epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for (xb,) in dl:
            xb = xb.to(dev)
            inp, tgt = xb[:, :-1], xb[:, 1:]
            _, loss = model(inp, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{args.epochs}  loss={tot/nb:.4f}  ({time.time()-t0:.1f}s)")

    torch.save({"state_dict": model.state_dict(),
                "config": {"vocab_size": V, "max_len": max_len, "d_model": args.d_model,
                           "n_head": args.n_head, "n_layer": args.n_layer}},
               OUT / "ethos_gpt.pt")
    print("Saved pretrained model to", OUT / "ethos_gpt.pt")


if __name__ == "__main__":
    main()
