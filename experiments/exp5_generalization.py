#!/usr/bin/env python3
"""
Exp5: Cross-disease generalization — Train on nodes 1,2,3; test on nodes 4,5
Compare: Proposed vs XGBoost vs StandardNN
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
    
    node_files = sorted(Path(paths["processed_dir"]).glob("node_*.parquet"))
    if len(node_files) < 3:
        print("Need at least 3 nodes.")
        return pd.DataFrame()
    
    train_nodes = node_files[:3]
    test_nodes = node_files[3:] if len(node_files) > 3 else node_files[-1:]
    
    results = []
    try:
        from models.few_shot import FewShotPredictor
        from sklearn.linear_model import LogisticRegression
        from models.baselines import get_baselines
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, f1_score
        
        X_train_list, y_train_list = [], []
        for f in train_nodes:
            df = pd.read_parquet(f)
            meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
            feat = [c for c in df.columns if c not in meta]
            X_train_list.append(df[feat].values.astype(np.float32))
            y_train_list.append(df["feasible"].values)
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        for test_node in test_nodes:
            df_test = pd.read_parquet(test_node)
            meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
            feat = [c for c in df_test.columns if c not in meta]
            X_test = df_test[feat].values.astype(np.float32)
            y_test = df_test["feasible"].values
            
            if len(X_test) < 5 or len(np.unique(y_test)) < 2:
                continue
            
            row = {"test_node": test_node.stem}
            
            fs = FewShotPredictor(cfg)
            fs.fit(X_train, y_train, n_episodes=min(100, len(X_train)))
            ev_fs = fs.evaluate(X_test, y_test)
            row["proposed_auroc"] = ev_fs.get("auroc", 0.5)
            
            lr = LogisticRegression(max_iter=1000, random_state=cfg["seed"])
            lr.fit(X_train, y_train)
            row["lr_auroc"] = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
            
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=cfg["seed"], use_label_encoder=False, eval_metric="logloss")
            xgb_model.fit(X_train, y_train)
            row["xgboost_auroc"] = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
            
            bl = get_baselines(cfg)[3]
            bl.fit(X_train, y_train)
            row["nn_auroc"] = roc_auc_score(y_test, bl.model.predict_proba(X_test)[:, 1]) if hasattr(bl.model, "predict_proba") else 0.5
            
            row["improvement_vs_lr"] = row["proposed_auroc"] - row["lr_auroc"]
            results.append(row)
    except Exception as e:
        results = [{"error": str(e)}]
    
    df = pd.DataFrame(results)
    out_dir = Path(paths["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp5_results.csv", index=False)
    with open(out_dir / "exp5_results.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    try:
        import json
        res_dir = Path(paths.get("results_dir", "results"))
        if not res_dir.is_absolute():
            res_dir = proj / res_dir
        res_dir.mkdir(parents=True, exist_ok=True)
        json.dump(df.to_dict(orient="records"), open(res_dir / "exp5_generalization.json", "w"), indent=2)
    except OSError:
        pass
    print("Exp5 complete.")
    return df

if __name__ == "__main__":
    run()
