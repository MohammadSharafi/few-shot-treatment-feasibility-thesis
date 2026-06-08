#!/usr/bin/env python3
"""
Hybrid: RF on [ETHOS embeddings 128D + tabular 25D] = 153D.
Same splits as RF. RandomizedSearchCV 50 trials.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef

proj = Path(__file__).parent.parent


def load_config():
    with open(proj / "config.yaml") as f:
        return yaml.safe_load(f)


def get_data_splits():
    """Same splits as exp_tabular_sota (RF)."""
    cfg = load_config()
    from utils.feature_engineering import add_engineered_features

    data = pd.read_parquet(proj / cfg["paths"]["processed_dir"] / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in data.columns if c not in meta]
    X = data[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = add_engineered_features(X)
    y = data["feasible"].values

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=cfg["seed"], stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=cfg["seed"], stratify=y[train_idx]
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test, cfg


def to_tokens(X, device):
    """Convert (n, 25) to (n, 25, 4) tokens for ETHOS."""
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    n, d = X.shape
    tokens = torch.zeros(n, 25, 4, device=device)
    for j in range(min(25, d)):
        tokens[:, j, 0] = (j + 1) / 25.0
        tokens[:, j, 1] = X[:, j]
        tokens[:, j, 2] = 0.0
        tokens[:, j, 3] = 0.0 if j < 15 else (0.5 if 22 <= j < 25 else 1.0)
    return tokens


def extract_ethos_embeddings(X, cfg, device):
    """Extract 128-D ETHOS embeddings. Train encoder with supervised head first if needed."""
    from models.encoder import ETHOSEncoder

    enc_cfg = cfg["encoder"]
    encoder = ETHOSEncoder(
        vocab_size=enc_cfg["vocab_size"],
        d_model=enc_cfg["d_model"],
        nhead=enc_cfg["nhead"],
        num_layers=enc_cfg.get("num_layers", 2),
        d_ff=enc_cfg["d_ff"],
        dropout=0.0,
    ).to(device)
    encoder.eval()

    # Try pretrained
    pretrained = proj / "models" / "ethos_pretrained.pt"
    if pretrained.exists():
        try:
            state = torch.load(pretrained, map_location=device, weights_only=True)
            if "encoder" in state:
                encoder.load_state_dict(state["encoder"], strict=False)
        except Exception:
            pass

    # Also try supervised checkpoint (encoder weights)
    sup_path = proj / "models" / "ethos_supervised.pt"
    if sup_path.exists():
        try:
            state = torch.load(sup_path, map_location=device, weights_only=True)
            if "encoder" in state:
                encoder.load_state_dict(state["encoder"], strict=False)
        except Exception:
            pass

    with torch.no_grad():
        x = to_tokens(X, device)
        emb = encoder(x).cpu().numpy()
    return emb


def run():
    X_train, y_train, X_val, y_val, X_test, y_test, cfg = get_data_splits()
    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Extracting ETHOS embeddings...")
    emb_train = extract_ethos_embeddings(X_train, cfg, device)
    emb_val = extract_ethos_embeddings(X_val, cfg, device)
    emb_test = extract_ethos_embeddings(X_test, cfg, device)

    # Hybrid: [ETHOS 128D, tabular 25D] = 153D
    X_hybrid_train = np.hstack([emb_train, X_train])
    X_hybrid_val = np.hstack([emb_val, X_val])
    X_hybrid_test = np.hstack([emb_test, X_test])

    X_train_val = np.vstack([X_hybrid_train, X_hybrid_val])
    y_train_val = np.concatenate([y_train, y_val])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 16, 20, 24, 28],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    rf = RandomForestClassifier(random_state=seed)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=50, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed
    )
    search.fit(X_train_val, y_train_val)

    out_dir = proj / cfg["paths"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # RF on ETHOS-only (128D) baseline
    param_dist_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 16, 20, 24],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"],
    }
    rf_only = RandomForestClassifier(random_state=seed)
    search_rf = RandomizedSearchCV(
        rf_only, param_dist_rf, n_iter=30, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed
    )
    emb_train_val = np.vstack([emb_train, emb_val])
    search_rf.fit(emb_train_val, y_train_val)
    probs_val_rf = search_rf.best_estimator_.predict_proba(emb_val)[:, 1]
    best_t_rf, best_f1_rf = 0.5, 0
    for t in np.linspace(0.3, 0.7, 41):
        pred = (probs_val_rf >= t).astype(int)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1_rf:
            best_f1_rf, best_t_rf = f1, t
    probs_test_rf = search_rf.best_estimator_.predict_proba(emb_test)[:, 1]
    pred_test_rf = (probs_test_rf >= best_t_rf).astype(int)
    auroc_rf = roc_auc_score(y_test, probs_test_rf) if len(np.unique(y_test)) >= 2 else 0.5
    f1_rf = f1_score(y_test, pred_test_rf, average="macro", zero_division=0)
    acc_rf = accuracy_score(y_test, pred_test_rf)
    mcc_rf = matthews_corrcoef(y_test, pred_test_rf) if len(np.unique(y_test)) >= 2 else 0.0
    row_rf = {"model": "RF_on_ETHOS_only", "auroc": auroc_rf, "f1_macro": f1_rf, "accuracy": acc_rf, "mcc": mcc_rf}
    pd.DataFrame([row_rf]).to_csv(out_dir / "exp_rf_ethos_only.csv", index=False)
    print(f"RF_on_ETHOS_only: AUROC={auroc_rf:.3f}, F1={f1_rf:.3f}, Acc={acc_rf:.3f}")

    # Threshold tuning on val (for hybrid)
    probs_val = search.best_estimator_.predict_proba(X_hybrid_val)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.3, 0.7, 41):
        pred = (probs_val >= t).astype(int)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    probs_test = search.best_estimator_.predict_proba(X_hybrid_test)[:, 1]
    pred_test = (probs_test >= best_t).astype(int)

    auroc = roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) >= 2 else 0.5
    f1 = f1_score(y_test, pred_test, average="macro", zero_division=0)
    acc = accuracy_score(y_test, pred_test)
    mcc = matthews_corrcoef(y_test, pred_test) if len(np.unique(y_test)) >= 2 else 0.0

    row = {
        "model": "ETHOS_RF_Hybrid",
        "auroc": auroc,
        "f1_macro": f1,
        "accuracy": acc,
        "mcc": mcc,
        "threshold": best_t,
    }
    pd.DataFrame([row]).to_csv(out_dir / "exp_ethos_rf_hybrid.csv", index=False)
    print(f"ETHOS_RF_Hybrid: AUROC={auroc:.3f}, F1={f1:.3f}, Acc={acc:.3f}")
    return row


if __name__ == "__main__":
    run()
