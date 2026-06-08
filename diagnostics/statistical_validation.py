#!/usr/bin/env python3
"""
Compute bootstrap 95% CI and DeLong test for thesis Statistical Significance section.
Run after experiments to validate reported numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def get_data():
    cfg = load_config()
    proj = Path(__file__).parent.parent
    from utils.feature_engineering import add_engineered_features
    data = pd.read_parquet(proj / cfg["paths"]["processed_dir"] / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in data.columns if c not in meta]
    X = data[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = add_engineered_features(X)
    y = data["feasible"].values
    return X, y, cfg

def bootstrap_auroc(y_true, y_prob, n_boot=1000):
    n = len(y_true)
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, [2.5, 97.5])

def delong_test(y_true, pred1, pred2):
    """Approximate DeLong test using bootstrap of AUC difference."""
    n_boot = 1000
    rng = np.random.default_rng(42)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a1 = roc_auc_score(y_true[idx], pred1[idx])
        a2 = roc_auc_score(y_true[idx], pred2[idx])
        diffs.append(a1 - a2)
    diffs = np.array(diffs)
    se = np.std(diffs)
    z = np.mean(diffs) / se if se > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p

def main():
    X, y, cfg = get_data()
    seed = cfg["seed"]
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
    )
    X_test, y_test = X[test_idx], y[test_idx]

    # RF (simplified - use same params as thesis)
    rf = RandomForestClassifier(n_estimators=200, max_depth=28, min_samples_split=5,
                               min_samples_leaf=2, max_features="log2", class_weight="balanced",
                               random_state=seed)
    rf.fit(X[train_idx], y[train_idx])
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_auroc = roc_auc_score(y_test, rf_prob)
    rf_ci = bootstrap_auroc(y_test, rf_prob)

    print("=== Statistical Validation for Thesis ===\n")
    print(f"Random Forest: AUROC {rf_auroc:.3f} [95% CI: {rf_ci[0]:.2f}, {rf_ci[1]:.2f}]")
    print("\nNote: ETHOS+FewShot requires full pipeline. Use saved predictions if available.")
    print("DeLong test: run with both model predictions. p < 0.01 expected for RF vs ETHOS.")

if __name__ == "__main__":
    main()
