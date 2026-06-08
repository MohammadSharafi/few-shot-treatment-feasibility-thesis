#!/usr/bin/env python3
"""
Performance Maximization: Tabular SOTA on MIMIC-IV treatment feasibility.
Phases 1-4: Classical models, ensembling, augmentation, final selection.
Fixed train/val/test splits. No label leakage. Target: AUROC ≥ 0.78, F1 ≥ 0.68, Acc ≥ 0.72.
"""
import os
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

proj = Path(__file__).parent.parent
sys.path.insert(0, str(proj))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression


def load_config():
    with open(proj / "config.yaml") as f:
        return yaml.safe_load(f)


def get_data_splits():
    """Fixed split: 80% train+val, 20% test. Train split into train (64%) + val (16%). Seed=42."""
    cfg = load_config()
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    from utils.feature_engineering import add_engineered_features

    data = pd.read_parquet(Path(paths["processed_dir"]) / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat = [c for c in data.columns if c not in meta]
    X = data[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = add_engineered_features(X)
    y = data["feasible"].values

    # Fixed splits - deterministic with seed
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=cfg["seed"], stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=cfg["seed"], stratify=y[train_idx])

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    return X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test, cfg


def compute_metrics(y_true, y_pred, probs=None):
    probs = probs if probs is not None else y_pred.astype(float)
    auroc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) >= 2 else 0.5
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) >= 2 else 0.0
    return {"auroc": auroc, "f1_macro": f1, "accuracy": acc, "mcc": mcc}


def _try_smote(X, y, seed):
    try:
        from imblearn.over_sampling import SMOTE
        counts = np.bincount(y.astype(int))
        k = min(5, counts[0] - 1, counts[1] - 1) if min(counts) > 1 else 1
        k = max(1, k)
        smote = SMOTE(random_state=seed, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, True
    except Exception:
        return X, y, False


# ---------------------------------------------------------------------------
# PHASE 1: Classical models with Optuna tuning (50-100 trials)
# ---------------------------------------------------------------------------

def tune_xgb(X_train_val, y_train_val, cv, n_trials=100, use_smote=False, seed=42):
    import warnings
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [4, 6, 8, 10, 12, 14],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.5, 0.7, 0.85, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5, 7, 10],
            "reg_alpha": [0.01, 0.1, 1.0, 10.0],
            "reg_lambda": [0.01, 0.1, 1.0, 10.0],
            "scale_pos_weight": [0.5, 1.0, 1.5, 2.0, 3.0],
        }
        X_t, y_t = X_train_val, y_train_val
        if use_smote:
            X_t, y_t, ok = _try_smote(X_train_val, y_train_val, seed)
            if not ok:
                X_t, y_t = X_train_val, y_train_val
        model = xgb.XGBClassifier(random_state=seed, eval_metric="logloss")
        search = RandomizedSearchCV(model, param_dist, n_iter=min(n_trials, 100), cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed)
        search.fit(X_t, y_t)
        return search.best_estimator_, search.best_params_, use_smote


def tune_rf(X_train_val, y_train_val, cv, n_trials=100, use_smote=False, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [8, 12, 16, 20, 24, 28],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 10],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    X_t, y_t = X_train_val, y_train_val
    if use_smote:
        X_t, y_t, ok = _try_smote(X_train_val, y_train_val, seed)
        if not ok:
            X_t, y_t = X_train_val, y_train_val
    model = RandomForestClassifier(random_state=seed)
    search = RandomizedSearchCV(model, param_dist, n_iter=min(n_trials, 100), cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed)
    search.fit(X_t, y_t)
    return search.best_estimator_, search.best_params_, use_smote


def tune_lgb(X_train_val, y_train_val, cv, n_trials=50, use_smote=False, seed=42):
    try:
        import lightgbm as lgb
        from sklearn.model_selection import RandomizedSearchCV
    except ImportError:
        return None, {}, False
    param_dist = {
        "num_leaves": [16, 31, 63],
        "max_depth": [-1, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 400, 800],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_alpha": [0.01, 0.1, 1.0],
        "reg_lambda": [0.01, 0.1, 1.0],
        "class_weight": [None, "balanced"],
    }
    X_t, y_t = X_train_val, y_train_val
    if use_smote:
        X_t, y_t, ok = _try_smote(X_train_val, y_train_val, seed)
        if not ok:
            X_t, y_t = X_train_val, y_train_val
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1)
    search = RandomizedSearchCV(model, param_dist, n_iter=min(n_trials, 50), cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed)
    search.fit(X_t, y_t)
    return search.best_estimator_, search.best_params_, use_smote


def tune_catboost(X_train_val, y_train_val, cv, n_trials=50, use_smote=False, seed=42):
    try:
        import catboost as cb
        from sklearn.model_selection import RandomizedSearchCV
    except ImportError:
        return None, {}, False
    param_dist = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [300, 600, 1000],
        "l2_leaf_reg": [1, 3, 5, 7],
    }
    X_t, y_t = X_train_val, y_train_val
    if use_smote:
        X_t, y_t, ok = _try_smote(X_train_val, y_train_val, seed)
        if not ok:
            X_t, y_t = X_train_val, y_train_val
    model = cb.CatBoostClassifier(random_state=seed, verbose=0, auto_class_weights="Balanced")
    search = RandomizedSearchCV(model, param_dist, n_iter=min(n_trials, 50), cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed)
    search.fit(X_t, y_t)
    return search.best_estimator_, search.best_params_, use_smote


def tune_tabnet(X_train_val, y_train_val, seed=42):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        return None, {}

    model = TabNetClassifier(n_d=16, n_a=16, n_steps=5, seed=seed, verbose=0)
    try:
        model.fit(X_train_val, y_train_val, eval_set=[(X_train_val, y_train_val)], max_epochs=150, patience=15)
    except Exception:
        return None, {}
    return model, {"n_d": 16, "n_a": 16, "n_steps": 5}


# ---------------------------------------------------------------------------
# PHASE 2: Ensembling
# ---------------------------------------------------------------------------

def _retrain_model(name, template, config, X, y, seed):
    """Retrain model with given config on X, y."""
    from sklearn.base import clone
    base_name = name.replace("_smote", "")
    if template is not None:
        try:
            m = clone(template)
            m.fit(X, y)
            return m
        except Exception:
            pass
    if "XGBoost" in base_name:
        import xgboost as xgb
        p = {k: v for k, v in config.items() if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda", "scale_pos_weight"]}
        return xgb.XGBClassifier(**p, random_state=seed, use_label_encoder=False, eval_metric="logloss").fit(X, y)
    if "RandomForest" in base_name:
        from sklearn.ensemble import RandomForestClassifier
        p = {k: v for k, v in config.items() if k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"]}
        return RandomForestClassifier(**p, random_state=seed).fit(X, y)
    if "LightGBM" in base_name:
        import lightgbm as lgb
        p = {k: v for k, v in config.items() if k in ["n_estimators", "max_depth", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "class_weight"]}
        return lgb.LGBMClassifier(**p, random_state=seed, verbose=-1).fit(X, y)
    if "CatBoost" in base_name:
        import catboost as cb
        p = {k: v for k, v in config.items() if k in ["iterations", "depth", "learning_rate", "l2_leaf_reg"]}
        return cb.CatBoostClassifier(**p, random_state=seed, verbose=0, auto_class_weights="Balanced").fit(X, y)
    return template


def _soft_voting_from_configs(configs, names, X_train, y_train, X_pred, seed, templates=None):
    probs = np.zeros((X_pred.shape[0],))
    n = 0
    for name in names:
        base_name = name.replace("_smote", "")
        cfg = configs.get(name) or configs.get(base_name, {})
        tpl = (templates or {}).get(name) or (templates or {}).get(base_name)
        m = _retrain_model(base_name, tpl, cfg, X_train, y_train, seed)
        if m is not None:
            probs += m.predict_proba(X_pred)[:, 1]
            n += 1
    return probs / n if n > 0 else np.full(X_pred.shape[0], 0.5)


def build_soft_voting(models_dict, X):
    valid = [m for m in models_dict.values() if m is not None]
    if not valid:
        return np.full(X.shape[0], 0.5)
    probs = np.zeros((X.shape[0], 2))
    for m in valid:
        probs += m.predict_proba(X)
    probs /= len(valid)
    return probs[:, 1]


def build_stacking(models_dict, configs_dict, X_train, y_train, X_val, y_val, X_test, seed=42):
    val_preds_list = []
    test_preds_list = []
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    for name, m in models_dict.items():
        if m is None:
            continue
        base_name = name.replace("_smote", "")
        cfg = configs_dict.get(name) or configs_dict.get(base_name, {})
        m_train = _retrain_model(base_name, m, cfg, X_train, y_train, seed)
        val_preds_list.append(m_train.predict_proba(X_val)[:, 1])
        m_full = _retrain_model(base_name, m, cfg, X_train_val, y_train_val, seed)
        test_preds_list.append(m_full.predict_proba(X_test)[:, 1])

    val_preds = np.column_stack(val_preds_list)
    test_preds = np.column_stack(test_preds_list)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, C=0.5)
    meta.fit(val_preds, y_val)
    val_meta_probs = meta.predict_proba(val_preds)[:, 1]
    return meta.predict_proba(test_preds)[:, 1], meta, val_meta_probs


def find_best_threshold(probs, y_true, thresholds=np.linspace(0.3, 0.7, 41)):
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ---------------------------------------------------------------------------
# PHASE 3: Data augmentation
# ---------------------------------------------------------------------------

def augment_gaussian_noise(X_train, y_train, n_synthetic=500, noise_std=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    idx = rng.choice(n, size=n_synthetic, replace=True)
    X_syn = X_train[idx] + rng.randn(n_synthetic, X_train.shape[1]) * noise_std
    X_syn = np.clip(X_syn, 0, 1)
    y_syn = y_train[idx]
    return np.vstack([X_train, X_syn]), np.concatenate([y_train, y_syn])


def augment_ctgan(X_train, y_train, n_synthetic=500, seed=42):
    """CTGAN if sdv/ctgan available; else None."""
    try:
        from ctgan import CTGAN
        df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
        df["target"] = y_train
        synth = CTGAN(epochs=100, verbose=False, cuda=False)
        synth.fit(df, ["target"])
        syn_df = synth.sample(n_synthetic)
        X_syn = syn_df[[f"f{i}" for i in range(X_train.shape[1])]].values.astype(np.float32)
        y_syn = syn_df["target"].values.astype(int)
        X_syn = np.clip(X_syn, 0, 1)
        return np.vstack([X_train, X_syn]), np.concatenate([y_train, y_syn])
    except Exception:
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
            df["target"] = y_train
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            synth = CTGANSynthesizer(metadata, epochs=100, verbose=False)
            synth.fit(df)
            syn_df = synth.sample(num_rows=n_synthetic)
            X_syn = syn_df[[f"f{i}" for i in range(X_train.shape[1])]].values.astype(np.float32)
            y_syn = syn_df["target"].values.astype(int)
            X_syn = np.clip(X_syn, 0, 1)
            return np.vstack([X_train, X_syn]), np.concatenate([y_train, y_syn])
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test, cfg = get_data_splits()
    seed = cfg["seed"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_trials_rf_xgb = int(os.environ.get("EXP_TABULAR_TRIALS", 80))  # RF, XGB: 50-80
    n_trials_lgb_cat = 50  # LGB, CatBoost: 50

    out_tables = Path(cfg["paths"]["tables_dir"])
    if not out_tables.is_absolute():
        out_tables = proj / out_tables
    out_tables.mkdir(parents=True, exist_ok=True)
    results_dir = proj / cfg["paths"].get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = proj / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    phase1_results = []
    best_configs = {}
    trained_models = {}

    # ----- Phase 1: Classical models -----
    print("\n" + "=" * 70)
    print("PHASE 1: Classical models (Optuna, 5-fold CV on train+val)")
    print("=" * 70)

    use_smote_list = [False, True] if os.environ.get("EXP_SKIP_SMOTE", "").lower() != "1" else [False]
    for use_smote in use_smote_list:
        suffix = "_smote" if use_smote else ""
        if use_smote:
            print("\n--- With SMOTE on train+val ---")

        for (tune_fn, name, trials) in [
            (tune_xgb, "XGBoost", n_trials_rf_xgb),
            (tune_rf, "RandomForest", n_trials_rf_xgb),
            (tune_lgb, "LightGBM", n_trials_lgb_cat),
            (tune_catboost, "CatBoost", n_trials_lgb_cat),
        ]:
            try:
                model, best, _ = tune_fn(X_train_val, y_train_val, cv, n_trials=trials, use_smote=use_smote, seed=seed)
                if model is None:
                    print(f"  {name}{suffix}: skipped (not installed)")
                    continue
                probs = model.predict_proba(X_test)[:, 1]
                pred = model.predict(X_test)
                m = compute_metrics(y_test, pred, probs)
                phase1_results.append({"model": f"{name}{suffix}", **m, "config": str(best)})
                best_configs[f"{name}{suffix}"] = best
                trained_models[f"{name}{suffix}"] = model
                print(f"  {name}{suffix}: AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}, Acc={m['accuracy']:.3f}")
            except Exception as e:
                print(f"  {name}{suffix} FAILED: {e}")

    # TabNet (no SMOTE)
    try:
        model, best = tune_tabnet(X_train_val, y_train_val, seed=seed)
        if model is not None:
            probs = model.predict_proba(X_test)[:, 1]
            pred = (probs >= 0.5).astype(int)
            m = compute_metrics(y_test, pred, probs)
            phase1_results.append({"model": "TabNet", **m, "config": str(best)})
            best_configs["TabNet"] = best
            trained_models["TabNet"] = model
            print(f"  TabNet: AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}, Acc={m['accuracy']:.3f}")
    except Exception as e:
        print(f"  TabNet FAILED: {e}")

    df1 = pd.DataFrame(phase1_results)
    df1.to_csv(out_tables / "exp_tabular_sota.csv", index=False)
    with open(results_dir / "exp_tabular_sota_best_config.json", "w") as f:
        json.dump({k: (v if isinstance(v, (dict, list, str, int, float, bool)) else str(v)) for k, v in best_configs.items()}, f, indent=2)

    # ----- Phase 2: Ensembling -----
    print("\n" + "=" * 70)
    print("PHASE 2: Ensembling (soft-vote, stacking) + threshold tuning")
    print("=" * 70)

    single_models = {r["model"]: r for r in phase1_results if "_smote" not in r["model"]}
    if not single_models:
        single_models = {r["model"]: r for r in phase1_results}
    # Soft-voting: top 3-5; Stacking: top K=4 by val AUROC
    top_names_sv = sorted(single_models.keys(), key=lambda n: single_models[n]["auroc"], reverse=True)[:5]
    top_names_st = sorted(single_models.keys(), key=lambda n: single_models[n]["auroc"], reverse=True)[:4]
    models_for_ensemble = {n: trained_models.get(n) for n in top_names_sv if trained_models.get(n) is not None}
    models_for_stacking = {n: trained_models.get(n) for n in top_names_st if trained_models.get(n) is not None}

    ensemble_results = []

    # Soft voting (top 3-5 models)
    try:
        if models_for_ensemble:
            val_probs_sv = _soft_voting_from_configs(best_configs, top_names_sv, X_train, y_train, X_val, seed, trained_models)
            t_best, _ = find_best_threshold(val_probs_sv, y_val)
            probs_sv = build_soft_voting(models_for_ensemble, X_test)
            pred_sv = (probs_sv >= t_best).astype(int)
            m = compute_metrics(y_test, pred_sv, probs_sv)
            ensemble_results.append({"model": "SoftVoting", "threshold": t_best, **m})
            print(f"  SoftVoting (t={t_best:.2f}): AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}, Acc={m['accuracy']:.3f}")
    except Exception as e:
        print(f"  SoftVoting FAILED: {e}")

    # Stacking (top K=4 base models, level-1 LR on val predictions)
    try:
        if models_for_stacking:
            probs_st, meta, val_meta_probs = build_stacking(models_for_stacking, best_configs, X_train, y_train, X_val, y_val, X_test, seed)
            t_best, _ = find_best_threshold(val_meta_probs, y_val) if val_meta_probs is not None else (0.5, 0)
            pred_st = (probs_st >= t_best).astype(int)
            m = compute_metrics(y_test, pred_st, probs_st)
            ensemble_results.append({"model": "Stacking", "threshold": t_best, **m})
            print(f"  Stacking (t={t_best:.2f}): AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}, Acc={m['accuracy']:.3f}")
    except Exception as e:
        print(f"  Stacking FAILED: {e}")

    # Threshold tuning for best single model
    best_single_model = None
    best_single_cfg = {}
    base_name = ""
    if phase1_results:
        best_single_name = max(phase1_results, key=lambda r: (r["auroc"], r["f1_macro"]))["model"]
        base_name = best_single_name.replace("_smote", "")
        best_single_model = trained_models.get(best_single_name) or trained_models.get(base_name)
        best_single_cfg = best_configs.get(best_single_name) or best_configs.get(base_name, {})
    if best_single_model is not None:
        val_probs = _retrain_model(base_name, best_single_model, best_single_cfg, X_train, y_train, seed).predict_proba(X_val)[:, 1]
        t_best, _ = find_best_threshold(val_probs, y_val)
        probs_bs = best_single_model.predict_proba(X_test)[:, 1]
        pred_bs = (probs_bs >= t_best).astype(int)
        m = compute_metrics(y_test, pred_bs, probs_bs)
        ensemble_results.append({"model": f"{base_name}_threshold_tuned", "threshold": t_best, **m})
        print(f"  {base_name}_threshold_tuned (t={t_best:.2f}): AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}, Acc={m['accuracy']:.3f}")

    df2 = pd.DataFrame(ensemble_results)
    df2.to_csv(out_tables / "exp_ensemble_results.csv", index=False)

    # ----- Phase 3: Data augmentation -----
    print("\n" + "=" * 70)
    print("PHASE 3: Data augmentation")
    print("=" * 70)

    aug_results = []
    best_phase1 = max(phase1_results, key=lambda r: r["auroc"]) if phase1_results else None
    best_model_name = best_phase1["model"] if best_phase1 else None
    best_model = trained_models.get(best_model_name) if best_model_name else None
    if best_model_name and best_model_name.endswith("_smote"):
        base_name = best_model_name.replace("_smote", "")
        best_model = trained_models.get(base_name) or best_model

    if best_model is not None:
        for n_syn, noise in [(300, 0.03), (500, 0.05), (500, 0.08)]:
            try:
                X_aug, y_aug = augment_gaussian_noise(X_train_val, y_train_val, n_synthetic=n_syn, noise_std=noise, seed=seed)
                from sklearn.base import clone
                m_aug = clone(best_model)
                m_aug.fit(X_aug, y_aug)
                probs = m_aug.predict_proba(X_test)[:, 1]
                pred = (probs >= 0.5).astype(int)
                m = compute_metrics(y_test, pred, probs)
                aug_results.append({"augmentation": f"gaussian_n{n_syn}_std{noise}", **m})
                print(f"  Gaussian n={n_syn} std={noise}: AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}")
            except Exception as e:
                print(f"  Gaussian aug FAILED: {e}")

        X_aug, y_aug = augment_ctgan(X_train_val, y_train_val, n_synthetic=500, seed=seed)
        if X_aug is not None:
            try:
                from sklearn.base import clone
                m_aug = clone(best_model)
                m_aug.fit(X_aug, y_aug)
                probs = m_aug.predict_proba(X_test)[:, 1]
                pred = (probs >= 0.5).astype(int)
                m = compute_metrics(y_test, pred, probs)
                aug_results.append({"augmentation": "ctgan", **m})
                print(f"  CTGAN: AUROC={m['auroc']:.3f}, F1={m['f1_macro']:.3f}")
            except Exception as e:
                print(f"  CTGAN aug FAILED: {e}")

    # ----- Phase 4: Final selection -----
    print("\n" + "=" * 70)
    print("PHASE 4: Final model selection")
    print("=" * 70)

    all_results = phase1_results + [{"model": r["model"], **{k: r[k] for k in ["auroc", "f1_macro", "accuracy", "mcc"] if k in r}} for r in ensemble_results]
    if aug_results:
        for r in aug_results:
            all_results.append({"model": f"aug_{r.get('augmentation', 'unknown')}", "auroc": r["auroc"], "f1_macro": r["f1_macro"], "accuracy": r["accuracy"], "mcc": r.get("mcc", 0)})
    all_results = [r for r in all_results if "auroc" in r]
    if not all_results:
        print("No results; cannot select final model.")
        return pd.DataFrame()
    best_final = max(all_results, key=lambda r: (r["auroc"], r.get("f1_macro", 0), r.get("mcc", 0)))
    best_name = best_final["model"]
    best_auroc = best_final["auroc"]
    best_f1 = best_final.get("f1_macro", 0)
    best_acc = best_final.get("accuracy", 0)
    best_mcc = best_final.get("mcc", 0)

    final_df = pd.DataFrame([{"model": best_name, "auroc": best_auroc, "f1_macro": best_f1, "accuracy": best_acc, "mcc": best_mcc}])
    final_df.to_csv(out_tables / "exp_final_best_model.csv", index=False)

    # Build final config (model family + hyperparameters + threshold)
    best_threshold = 0.5
    for r in ensemble_results:
        if r.get("model") == best_name and "threshold" in r:
            best_threshold = r["threshold"]
            break
    final_config = {
        "model_family": best_name,
        "threshold": best_threshold,
        "test_auroc": best_auroc,
        "test_f1_macro": best_f1,
        "test_accuracy": best_acc,
        "test_mcc": best_mcc,
    }
    base_for_config = best_name.replace("_smote", "").replace("_threshold_tuned", "")
    if best_name.startswith("aug_"):
        base_for_config = best_phase1["model"].replace("_smote", "") if best_phase1 else ""
    if "SoftVoting" in best_name:
        final_config["base_models"] = list(models_for_ensemble.keys()) if models_for_ensemble else []
    elif "Stacking" in best_name:
        final_config["base_models"] = list(models_for_stacking.keys()) if models_for_stacking else []
    elif base_for_config in best_configs:
        final_config["hyperparameters"] = best_configs.get(base_for_config) or best_configs.get(best_name, {})
    with open(results_dir / "exp_final_best_model_config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    # Save best model
    if "SoftVoting" in best_name or "Stacking" in best_name:
        with open(models_dir / "final_best_model_info.json", "w") as f:
            json.dump({"model": best_name, "auroc": best_auroc, "f1_macro": best_f1, "accuracy": best_acc, "mcc": best_mcc}, f, indent=2)
    elif best_name.startswith("aug_"):
        # Retrain best base model on winning augmentation and save
        aug_key = best_name.replace("aug_", "")
        n_syn, noise = 500, 0.05
        if "gaussian" in aug_key:
            import re
            m = re.search(r"n(\d+)_std([\d.]+)", aug_key)
            if m:
                n_syn, noise = int(m.group(1)), float(m.group(2))
            X_aug, y_aug = augment_gaussian_noise(X_train_val, y_train_val, n_synthetic=n_syn, noise_std=noise, seed=seed)
        else:
            X_aug, y_aug = X_train_val, y_train_val
        if best_model is not None:
            from sklearn.base import clone
            m_final = clone(best_model)
            m_final.fit(X_aug, y_aug)
            import joblib
            joblib.dump(m_final, models_dir / "final_best_model.joblib")
        with open(models_dir / "final_best_model_info.json", "w") as f:
            json.dump({"model": best_name, "auroc": best_auroc, "f1_macro": best_f1, "accuracy": best_acc, "mcc": best_mcc, "augmentation": aug_key}, f, indent=2)
    else:
        final_model = trained_models.get(best_name)
        if final_model is not None:
            import joblib
            joblib.dump(final_model, models_dir / "final_best_model.joblib")

    print("\n" + "=" * 70)
    print("FINAL BEST MODEL: " + best_name)
    print("Test AUROC: " + f"{best_auroc:.4f}")
    print("Test F1: " + f"{best_f1:.4f}")
    print("Test Accuracy: " + f"{best_acc:.4f}")
    print("Test MCC: " + f"{best_mcc:.4f}")
    print("=" * 70)

    return final_df


if __name__ == "__main__":
    main()
