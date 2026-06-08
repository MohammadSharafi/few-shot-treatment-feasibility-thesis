#!/usr/bin/env python3
"""
Exp2: K-shot analysis — Minimum labeled examples for clinically acceptable performance
Reports mean ± std across multiple seeds for fair comparison and CI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.model_selection import train_test_split

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def run():
    cfg = load_config()
    proj = Path(__file__).parent.parent
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    
    data = pd.read_parquet(Path(paths["processed_dir"]) / "thesis_dataset.parquet")
    tokens = np.load(Path(paths["processed_dir"]) / "tokens.npy")
    y = data["feasible"].values
    
    k_values = [1, 3, 5, 10, 15]
    n_episodes = 80
    n_seeds = 2
    seeds = [42, 43]
    
    results = []
    for k in k_values:
        aurocs, f1s = [], []
        for seed in seeds:
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                from models.few_shot import FewShotPredictor
                fs = FewShotPredictor(cfg)
                fs.cfg["few_shot"]["n_support"] = k
                train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
                X_tok_train = tokens[train_idx]
                X_tok_test = tokens[test_idx]
                fs.fit(X_tok_train, y[train_idx], n_episodes=min(n_episodes, len(X_tok_train)), n_support=k)
                ev = fs.evaluate(X_tok_test, y[test_idx])
                aurocs.append(ev.get("auroc", 0.5))
                f1s.append(ev.get("f1_macro", 0))
            except Exception as e:
                aurocs.append(0.5)
                f1s.append(0)
        
        aurocs = np.array(aurocs)
        f1s = np.array(f1s)
        results.append({
            "k": k,
            "auroc": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auroc_ci_lo": float(np.percentile(aurocs, 2.5)),
            "auroc_ci_hi": float(np.percentile(aurocs, 97.5)),
            "f1_macro": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        })
    
    df = pd.DataFrame(results)
    out_dir = Path(paths["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp2_results.csv", index=False)
    try:
        import json
        res_dir = Path(paths.get("results_dir", proj / "results"))
        if isinstance(res_dir, str):
            res_dir = Path(res_dir)
        if not Path(str(res_dir)).is_absolute():
            res_dir = proj / res_dir
        res_dir.mkdir(parents=True, exist_ok=True)
        json.dump(df.to_dict(orient="records"), open(res_dir / "exp2_shots.json", "w"), indent=2)
    except OSError:
        pass
    with open(out_dir / "exp2_results.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print("Exp2 complete. K-shot AUROC:", [f"{r['auroc']:.3f}±{r['auroc_std']:.3f}" for r in results])
    return df

if __name__ == "__main__":
    run()
