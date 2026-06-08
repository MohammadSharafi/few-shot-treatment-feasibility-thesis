#!/usr/bin/env python3
"""
Exp4: Ablation — Contribution of each innovation
Configs: full_model, no_symptom_tokens, no_clinical_bias, no_intensity_weighting, no_temporal_decay, no_few_shot
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.model_selection import train_test_split

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
except Exception:
    pass

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
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    X_flat = data[[c for c in data.columns if c not in meta]].values.astype(np.float32)
    
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=cfg["seed"], stratify=y)
    n_episodes = 100
    
    configs = [
        ("full_model", {"use_tokens": True, "encoder_kwargs": None}),
        ("no_symptom_tokens", {"use_tokens": False, "encoder_kwargs": None}),
        ("no_clinical_bias", {"use_tokens": True, "encoder_kwargs": {"use_cooccur": False, "use_intensity": True, "use_temporal": True}}),
        ("no_intensity_weighting", {"use_tokens": True, "encoder_kwargs": {"use_cooccur": True, "use_intensity": False, "use_temporal": True}}),
        ("no_temporal_decay", {"use_tokens": True, "encoder_kwargs": {"use_cooccur": True, "use_intensity": True, "use_temporal": False}}),
        ("no_few_shot", {"use_tokens": True, "encoder_kwargs": None, "use_nn": True}),
    ]
    
    results = []
    from sklearn.metrics import roc_auc_score, f1_score
    
    for name, c in configs:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.manual_seed(42)
            np.random.seed(42)
            
            if c.get("use_nn"):
                from models.baselines import get_baselines
                bl = get_baselines(cfg)[3]
                bl.fit(X_flat[train_idx], y[train_idx])
                pred = bl.predict(X_flat[test_idx])
                auroc = roc_auc_score(y[test_idx], bl.model.predict_proba(X_flat[test_idx])[:, 1]) if hasattr(bl.model, "predict_proba") else 0.5
                results.append({"variant": name, "auroc": auroc, "f1_macro": f1_score(y[test_idx], pred, average="macro", zero_division=0)})
            elif not c["use_tokens"]:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=1000, random_state=cfg["seed"])
                lr.fit(X_flat[train_idx], y[train_idx])
                pred = lr.predict(X_flat[test_idx])
                auroc = roc_auc_score(y[test_idx], lr.predict_proba(X_flat[test_idx])[:, 1])
                results.append({"variant": name, "auroc": auroc, "f1_macro": f1_score(y[test_idx], pred, average="macro", zero_division=0)})
            else:
                from models.few_shot import FewShotPredictor
                fs = FewShotPredictor(cfg)
                fs.fit(tokens[train_idx], y[train_idx], n_episodes=min(n_episodes, len(train_idx)), encoder_kwargs=c.get("encoder_kwargs"))
                ev = fs.evaluate(tokens[test_idx], y[test_idx])
                results.append({"variant": name, "auroc": ev.get("auroc", 0.5), "f1_macro": ev.get("f1_macro", 0)})
        except Exception as e:
            results.append({"variant": name, "auroc": None, "f1_macro": None, "error": str(e)[:50]})
    
    df = pd.DataFrame(results)
    out_dir = Path(paths["tables_dir"])
    res_dir = Path(paths.get("results_dir", proj / "results"))
    if not Path(str(res_dir)).is_absolute():
        res_dir = proj / res_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_dir / "exp4_results.csv", index=False)
        with open(out_dir / "exp4_results.tex", "w") as f:
            f.write(df.to_latex(index=False, float_format="%.3f"))
    except OSError:
        import json
        with open(res_dir / "exp4_results.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    try:
        import json
        json.dump(df.to_dict(orient="records"), open(res_dir / "exp4_ablation.json", "w"), indent=2, default=str)
    except OSError:
        pass
    print("Exp4 complete.")
    return df

if __name__ == "__main__":
    run()
