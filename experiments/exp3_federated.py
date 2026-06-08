#!/usr/bin/env python3
"""
Exp3: Federated vs centralized — Quantified accuracy cost
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def run():
    cfg = load_config()
    proj = Path(__file__).parent.parent
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    
    node_files = list(Path(paths["processed_dir"]).glob("node_*.parquet"))
    if not node_files:
        print("No node files. Run 05_federated_split first.")
        return pd.DataFrame()
    
    node_data = []
    for pf in sorted(node_files):
        df = pd.read_parquet(pf)
        meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
        feat = [c for c in df.columns if c not in meta]
        X = df[feat].values.astype(np.float32)
        y = df["feasible"].values
        if len(X) >= 10 and len(np.unique(y)) >= 2:
            node_data.append((X, y))
    
    if len(node_data) < 2:
        print("Insufficient nodes for federated experiment.")
        return pd.DataFrame()
    
    results = []
    try:
        from models.federated import FederatedPredictor
        from models.few_shot import FewShotPredictor
        
        fed = FederatedPredictor(cfg)
        fed.fit(node_data, n_rounds=10)
        
        X_all = np.vstack([n[0] for n in node_data])
        y_all = np.concatenate([n[1] for n in node_data])
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=cfg["seed"], stratify=y_all)
        
        cen = FewShotPredictor(cfg)
        cen.fit(X_train, y_train, n_episodes=min(50, len(X_train)))
        
        ev_fed = fed.evaluate(X_test, y_test)
        ev_cen = cen.evaluate(X_test, y_test)
        gap = ev_cen.get("auroc", 0.5) - ev_fed.get("auroc", 0.5)
        results = [
            {"setup": "Centralized", "auroc": ev_cen.get("auroc", 0.5), "f1_macro": ev_cen.get("f1_macro", 0)},
            {"setup": "Federated", "auroc": ev_fed.get("auroc", 0.5), "f1_macro": ev_fed.get("f1_macro", 0)},
            {"setup": "Gap", "auroc": gap, "f1_macro": ev_cen.get("f1_macro", 0) - ev_fed.get("f1_macro", 0)},
        ]
    except Exception as e:
        results = [{"setup": "error", "error": str(e)}]
    
    df = pd.DataFrame(results)
    out_dir = Path(paths["tables_dir"])
    res_dir = Path(paths.get("results_dir", proj / "results"))
    if not Path(str(res_dir)).is_absolute():
        res_dir = proj / res_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_dir / "exp3_results.csv", index=False)
        with open(out_dir / "exp3_results.tex", "w") as f:
            f.write(df.to_latex(index=False, float_format="%.3f"))
    except OSError:
        import json
        with open(res_dir / "exp3_results.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    try:
        import json
        json.dump(df.to_dict(orient="records"), open(res_dir / "exp3_federated.json", "w"), indent=2)
    except (OSError, KeyError):
        pass
    print("Exp3 complete.")
    return df

if __name__ == "__main__":
    run()
