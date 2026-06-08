#!/usr/bin/env python3
"""
Performance breakdown diagnostics for treatment feasibility prediction.
Computes metrics per node, per protocol, calibration, learning curves,
and compares: XGBoost (raw), MLP (tokens), ETHOS+FewShot.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def get_feat_meta():
    return (
        [str(i) for i in range(22)],
        ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    )

def assign_node(icd_str):
    icd = str(icd_str).replace(".", "").upper()
    for node, prefixes in [
        ("sepsis", ["A40", "A41", "R65"]),
        ("cardiac", ["I21", "I22", "I50", "I48"]),
        ("respiratory", ["J18", "J44", "J96", "J80"]),
        ("renal", ["N17", "N18", "N19"]),
        ("diabetes", ["E10", "E11", "E13"]),
    ]:
        for p in prefixes:
            if icd.startswith(p.replace(".", "")):
                return node
    return "other"

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * np.abs(acc - conf)
    return ece / len(y_true) if len(y_true) > 0 else 0.0

def run_xgboost(X_train, y_train, X_test, y_test):
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return probs, pred

def run_mlp(X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPClassifier
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    X_te = scaler.transform(np.nan_to_num(X_test, nan=0.0))
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    model.fit(X_tr, y_train)
    probs = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    return probs, pred

def run_fewshot(tokens, y, train_idx, test_idx, n_episodes=50):
    from models.few_shot import FewShotPredictor
    cfg = load_config()
    fs = FewShotPredictor(cfg)
    X_tr = tokens[train_idx]
    y_tr = y[train_idx]
    X_te = tokens[test_idx]
    y_te = y[test_idx]
    fs.fit(X_tr, y_tr, n_episodes=min(n_episodes, len(train_idx)))
    ev = fs.evaluate(X_te, y_te)
    # Use episodic eval metrics; probs from support/query split inside evaluate
    auroc = ev.get("auroc", 0.5)
    pred = fs.predict(X_te)  # no support = uses internal
    probs = np.full(len(y_te), 0.5)
    try:
        k = min(5, (y_tr == 0).sum() // 2, (y_tr == 1).sum() // 2)
        if k >= 2:
            idx0 = np.where(y_tr == 0)[0]
            idx1 = np.where(y_tr == 1)[0]
            sup_idx = np.concatenate([np.random.RandomState(42).choice(idx0, k), np.random.RandomState(43).choice(idx1, k)])
            probs = fs._predict_proba(X_te, X_tr[sup_idx], y_tr[sup_idx])[:, 1]
    except Exception:
        pass
    return probs, pred, auroc

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Skip few-shot (slow)")
    args = p.parse_args()
    cfg = load_config()
    proj = Path(__file__).parent.parent
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    
    data = pd.read_parquet(Path(paths["processed_dir"]) / "thesis_dataset.parquet")
    tokens = np.load(Path(paths["processed_dir"]) / "tokens.npy")
    
    feat_cols, meta_cols = get_feat_meta()
    feat_cols = [c for c in feat_cols if c in data.columns]
    X = data[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = data["feasible"].values
    
    data = data.copy()
    data["node"] = data["primary_icd10"].apply(assign_node)
    
    print("=" * 70)
    print("PERFORMANCE BREAKDOWN DIAGNOSTICS")
    print("=" * 70)
    
    # Class imbalance
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    print(f"\nClass balance: feasible=1: {n_pos} ({100*n_pos/len(y):.1f}%), feasible=0: {n_neg} ({100*n_neg/len(y):.1f}%)")
    
    # Global train/test split
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=cfg["seed"], stratify=y)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # --- 1. Per disease node ---
    print("\n--- METRICS PER DISEASE NODE (XGBoost) ---")
    xgb_probs, xgb_pred = run_xgboost(X_train, y_train, X_test, y_test)
    best_auroc_global = roc_auc_score(y_test, xgb_probs)
    
    node_metrics = []
    for node in ["sepsis", "cardiac", "respiratory", "renal", "diabetes", "other"]:
        mask = (data.iloc[test_idx]["node"] == node).values
        if mask.sum() < 10:
            continue
        y_n = y_test[mask]
        p_n = xgb_probs[mask]
        pred_n = xgb_pred[mask]
        if len(np.unique(y_n)) < 2:
            auroc_n = 0.5
        else:
            auroc_n = roc_auc_score(y_n, p_n)
        node_metrics.append((node, auroc_n, f1_score(y_n, pred_n, average="macro", zero_division=0), accuracy_score(y_n, pred_n), mask.sum()))
    
    node_metrics.sort(key=lambda x: x[1])
    worst_node = node_metrics[0] if node_metrics else ("N/A", 0.5, 0, 0, 0)
    for node, auroc, f1, acc, n in node_metrics:
        print(f"  {node}: AUROC={auroc:.3f}, F1={f1:.3f}, Acc={acc:.3f}, n={n}")
    
    # --- 2. Per protocol ---
    print("\n--- METRICS PER PROTOCOL (XGBoost, top 5 protocols) ---")
    for proto in data.iloc[test_idx]["protocol"].value_counts().head(5).index:
        mask = (data.iloc[test_idx]["protocol"] == proto).values
        if mask.sum() < 10:
            continue
        y_p = y_test[mask]
        p_p = xgb_probs[mask]
        if len(np.unique(y_p)) >= 2:
            auroc_p = roc_auc_score(y_p, p_p)
        else:
            auroc_p = 0.5
        print(f"  protocol {proto}: AUROC={auroc_p:.3f}, n={mask.sum()}")
    
    # --- 3. Calibration ---
    print("\n--- CALIBRATION (XGBoost) ---")
    ece = compute_ece(y_test, xgb_probs)
    print(f"  ECE (10 bins): {ece:.4f}")
    
    # --- 4. Learning curves ---
    print("\n--- LEARNING CURVES (XGBoost) ---")
    fracs = [0.25, 0.5, 0.75, 1.0]
    lc_aurocs = []
    for frac in fracs:
        n_use = max(50, int(len(train_idx) * frac))
        idx_use = np.random.RandomState(42).choice(train_idx, n_use, replace=False)
        X_tr = X[idx_use]
        y_tr = y[idx_use]
        probs, _ = run_xgboost(X_tr, y_tr, X_test, y_test)
        auroc = roc_auc_score(y_test, probs)
        lc_aurocs.append(auroc)
        print(f"  {frac*100:.0f}% train ({n_use} samples): AUROC={auroc:.3f}")
    
    # --- 5. Model comparison ---
    print("\n--- MODEL COMPARISON ---")
    # a) XGBoost raw
    xgb_probs, xgb_pred = run_xgboost(X_train, y_train, X_test, y_test)
    a_xgb = roc_auc_score(y_test, xgb_probs)
    f1_xgb = f1_score(y_test, xgb_pred, average="macro", zero_division=0)
    acc_xgb = accuracy_score(y_test, xgb_pred)
    print(f"  a) XGBoost (raw 22 feat): AUROC={a_xgb:.3f}, F1={f1_xgb:.3f}, Acc={acc_xgb:.3f}")
    
    # b) MLP on raw
    mlp_probs, mlp_pred = run_mlp(X_train, y_train, X_test, y_test)
    a_mlp = roc_auc_score(y_test, mlp_probs)
    f1_mlp = f1_score(y_test, mlp_pred, average="macro", zero_division=0)
    acc_mlp = accuracy_score(y_test, mlp_pred)
    print(f"  b) MLP (raw 22 feat):     AUROC={a_mlp:.3f}, F1={f1_mlp:.3f}, Acc={acc_mlp:.3f}")
    
    # c) ETHOS + Few-Shot
    a_fs, f1_fs, acc_fs = 0.5, 0.0, 0.5
    if not args.quick:
        try:
            fs_probs, fs_pred, fs_auroc = run_fewshot(tokens, y, train_idx, test_idx, n_episodes=30)
            a_fs = roc_auc_score(y_test, fs_probs) if len(np.unique(y_test)) >= 2 else fs_auroc
            f1_fs = f1_score(y_test, fs_pred, average="macro", zero_division=0)
            acc_fs = accuracy_score(y_test, fs_pred)
            print(f"  c) ETHOS+FewShot:         AUROC={a_fs:.3f}, F1={f1_fs:.3f}, Acc={acc_fs:.3f}")
        except Exception as e:
            print(f"  c) ETHOS+FewShot:         FAILED ({e})")
    else:
        print("  c) ETHOS+FewShot:         (skipped --quick)")
    
    best_auroc = max(a_xgb, a_mlp, a_fs)
    
    # --- 6. Underfitting/overfitting signal ---
    train_probs_xgb, _ = run_xgboost(X_train, y_train, X_train, y_train)
    train_auroc = roc_auc_score(y_train, train_probs_xgb)
    gap = train_auroc - a_xgb
    if gap > 0.15:
        fit_signal = "Overfitting (train AUROC >> test AUROC)"
    elif gap < 0.02:
        fit_signal = "Possible underfitting (train ≈ test)"
    else:
        fit_signal = "Reasonable fit"
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best AUROC so far by any model: {best_auroc:.3f}")
    print(f"  Worst disease node: {worst_node[0]} (AUROC={worst_node[1]:.3f}, n={worst_node[4]})")
    print(f"  Evidence of underfitting/overfitting: {fit_signal}")
    print(f"  Train AUROC (XGBoost): {train_auroc:.3f}, Test: {a_xgb:.3f}, Gap: {gap:.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
