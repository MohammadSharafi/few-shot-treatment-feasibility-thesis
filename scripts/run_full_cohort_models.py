#!/usr/bin/env python3
"""Run stable full-cohort models and save results under results/full_cohort."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, value in cfg["paths"].items():
        path = Path(value)
        paths[key] = path if path.is_absolute() else ROOT / path
    return paths


def load_data(cfg: dict, paths: dict[str, Path]):
    data = pd.read_parquet(paths["processed_dir"] / "thesis_dataset.parquet")
    meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat_cols = [c for c in data.columns if c not in meta]
    X = data[feat_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from utils.feature_engineering import add_engineered_features

        X = add_engineered_features(X)
    except Exception:
        pass
    y = data["feasible"].to_numpy(dtype=int)
    return data, X, y


def metric_row(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (probs >= threshold).astype(int)
    row = {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "auprc": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    rng = np.random.RandomState(42)
    boots = []
    idx_all = np.arange(len(y_true))
    for _ in range(1000):
        idx = rng.choice(idx_all, len(idx_all), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], probs[idx]))
    if boots:
        row["auroc_ci_lo"] = float(np.percentile(boots, 2.5))
        row["auroc_ci_hi"] = float(np.percentile(boots, 97.5))
    return row


def tune_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.25, 0.75, 101):
        f1 = f1_score(y_val, (probs_val >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    pred = model.predict(X)
    return pred.astype(float)


def model_factories(seed: int) -> dict:
    factories = {
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1),
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "Neural Baseline": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=160,
                early_stopping=True,
                random_state=seed,
            ),
        ),
    }
    try:
        import xgboost as xgb

        factories["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass
    try:
        import lightgbm as lgb

        factories["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass
    try:
        import catboost as cb

        factories["CatBoost"] = lambda: cb.CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
    except Exception:
        pass
    return factories


def run_standard_models(X, y, train_idx, test_idx, seed: int):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    rows, statuses, fitted = [], [], {}
    for name, factory in model_factories(seed).items():
        t0 = time.time()
        try:
            model = factory()
            model.fit(X_fit, y_fit)
            val_probs = predict_proba_positive(model, X_val)
            threshold = tune_threshold(y_val, val_probs)
            final_model = factory()
            final_model.fit(X_train, y_train)
            test_probs = predict_proba_positive(final_model, X_test)
            row = metric_row(name, y_test, test_probs, threshold)
            row["fit_seconds"] = round(time.time() - t0, 3)
            rows.append(row)
            fitted[name] = final_model
            statuses.append({"model": name, "status": "completed", "seconds": row["fit_seconds"]})
            print(f"{name}: AUROC={row['auroc']:.4f} F1={row['f1_macro']:.4f}", flush=True)
        except Exception as exc:
            statuses.append({"model": name, "status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 3)})
            print(f"{name}: failed: {exc}", flush=True)
    return rows, statuses, fitted


def run_stacked(X, y, train_idx, test_idx, seed: int, available_names: list[str]):
    names = [n for n in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"] if n in available_names]
    if len(names) < 2:
        return None, {"model": "Stacked Classical", "status": "skipped", "reason": "Fewer than two base models completed"}
    factories = model_factories(seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sub_idx, meta_idx = train_test_split(np.arange(len(y_train)), test_size=0.25, random_state=seed, stratify=y_train)
    meta_features, test_features = [], []
    t0 = time.time()
    try:
        for name in names:
            m_meta = factories[name]()
            m_meta.fit(X_train[sub_idx], y_train[sub_idx])
            meta_features.append(predict_proba_positive(m_meta, X_train[meta_idx]))
            m_full = factories[name]()
            m_full.fit(X_train, y_train)
            test_features.append(predict_proba_positive(m_full, X_test))
        Z_meta = np.column_stack(meta_features)
        Z_test = np.column_stack(test_features)
        meta_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        meta_model.fit(Z_meta, y_train[meta_idx])
        meta_val_probs = predict_proba_positive(meta_model, Z_meta)
        threshold = tune_threshold(y_train[meta_idx], meta_val_probs)
        probs = predict_proba_positive(meta_model, Z_test)
        row = metric_row("Stacked Classical", y_test, probs, threshold)
        row["fit_seconds"] = round(time.time() - t0, 3)
        row["base_models"] = "; ".join(names)
        return row, {"model": "Stacked Classical", "status": "completed", "seconds": row["fit_seconds"], "base_models": names}
    except Exception as exc:
        return None, {"model": "Stacked Classical", "status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 3)}


def run_token_hybrid(paths, X, y, train_idx, test_idx, seed: int):
    token_path = paths["processed_dir"] / "tokens.npy"
    if not token_path.exists():
        return None, {"model": "Token Hybrid RF", "status": "skipped", "reason": "tokens.npy missing"}
    t0 = time.time()
    try:
        tokens = np.load(token_path)
        flat_tokens = tokens.reshape(tokens.shape[0], -1).astype(np.float32)
        Xh = np.hstack([X, flat_tokens])
        X_train, X_test = Xh[train_idx], Xh[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
        )
        model = RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_fit, y_fit)
        threshold = tune_threshold(y_val, predict_proba_positive(model, X_val))
        final_model = clone(model)
        final_model.fit(X_train, y_train)
        probs = predict_proba_positive(final_model, X_test)
        row = metric_row("Token Hybrid RF", y_test, probs, threshold)
        row["fit_seconds"] = round(time.time() - t0, 3)
        return row, {"model": "Token Hybrid RF", "status": "completed", "seconds": row["fit_seconds"]}
    except Exception as exc:
        return None, {"model": "Token Hybrid RF", "status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 3)}


def run_fewshot(paths, y, train_idx, test_idx, cfg, max_episodes: int):
    token_path = paths["processed_dir"] / "tokens.npy"
    if not token_path.exists():
        return None, {"model": "Few-Shot ETHOS", "status": "skipped", "reason": "tokens.npy missing"}
    t0 = time.time()
    try:
        from models.few_shot import FewShotPredictor

        tokens = np.load(token_path)
        fs = FewShotPredictor(cfg)
        fs.fit(tokens[train_idx], y[train_idx], n_episodes=max_episodes)
        probs = fs._predict_proba(tokens[test_idx], tokens[train_idx], y[train_idx])
        threshold = 0.5
        row = metric_row("Few-Shot ETHOS", y[test_idx], probs[:, 1], threshold)
        row["fit_seconds"] = round(time.time() - t0, 3)
        row["episodes"] = int(max_episodes)
        return row, {"model": "Few-Shot ETHOS", "status": "completed", "seconds": row["fit_seconds"], "episodes": int(max_episodes)}
    except Exception as exc:
        return None, {"model": "Few-Shot ETHOS", "status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 3)}


def run_federated_lr(paths, X, y, train_idx, test_idx, seed: int):
    processed = paths["processed_dir"]
    node_files = sorted(processed.glob("node_*.parquet"))
    if len(node_files) < 2:
        return None, {"model": "Federated LR Node Ensemble", "status": "skipped", "reason": "node files missing"}
    t0 = time.time()
    try:
        data = pd.read_parquet(processed / "thesis_dataset.parquet")
        train_stays = set(data.iloc[train_idx]["stay_id"].tolist())
        X_test, y_test = X[test_idx], y[test_idx]
        probs_list, weights = [], []
        meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
        for node_file in node_files:
            node = pd.read_parquet(node_file)
            node = node[node["stay_id"].isin(train_stays)]
            if len(node) < 20 or node["feasible"].nunique() < 2:
                continue
            feat = [c for c in node.columns if c not in meta]
            Xn = np.nan_to_num(node[feat].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                from utils.feature_engineering import add_engineered_features

                Xn = add_engineered_features(Xn)
            except Exception:
                pass
            yn = node["feasible"].to_numpy(dtype=int)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1),
            )
            model.fit(Xn, yn)
            probs_list.append(predict_proba_positive(model, X_test))
            weights.append(len(node))
        if not probs_list:
            return None, {"model": "Federated LR Node Ensemble", "status": "skipped", "reason": "no trainable nodes"}
        weights_arr = np.asarray(weights, dtype=float)
        probs = np.average(np.vstack(probs_list), axis=0, weights=weights_arr)
        threshold = 0.5
        row = metric_row("Federated LR Node Ensemble", y_test, probs, threshold)
        row["fit_seconds"] = round(time.time() - t0, 3)
        row["nodes_used"] = len(probs_list)
        return row, {"model": "Federated LR Node Ensemble", "status": "completed", "seconds": row["fit_seconds"], "nodes_used": len(probs_list)}
    except Exception as exc:
        return None, {"model": "Federated LR Node Ensemble", "status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fewshot", action="store_true", help="Skip the full-split few-shot run")
    parser.add_argument("--fewshot-episodes", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config()
    paths = resolve_paths(cfg)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)

    data, X, y = load_data(cfg, paths)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=cfg["seed"], stratify=y
    )
    split = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_label_distribution": {str(int(k)): int(v) for k, v in pd.Series(y[train_idx]).value_counts().sort_index().items()},
        "test_label_distribution": {str(int(k)): int(v) for k, v in pd.Series(y[test_idx]).value_counts().sort_index().items()},
    }

    rows, statuses, fitted = run_standard_models(X, y, train_idx, test_idx, cfg["seed"])
    stacked_row, stacked_status = run_stacked(X, y, train_idx, test_idx, cfg["seed"], list(fitted.keys()))
    statuses.append(stacked_status)
    if stacked_row:
        rows.append(stacked_row)
    hybrid_row, hybrid_status = run_token_hybrid(paths, X, y, train_idx, test_idx, cfg["seed"])
    statuses.append(hybrid_status)
    if hybrid_row:
        rows.append(hybrid_row)
    fed_row, fed_status = run_federated_lr(paths, X, y, train_idx, test_idx, cfg["seed"])
    statuses.append(fed_status)
    if fed_row:
        rows.append(fed_row)
    if args.skip_fewshot:
        statuses.append({"model": "Few-Shot ETHOS", "status": "skipped", "reason": "requested via --skip-fewshot"})
    else:
        fs_row, fs_status = run_fewshot(paths, y, train_idx, test_idx, cfg, args.fewshot_episodes)
        statuses.append(fs_status)
        if fs_row:
            rows.append(fs_row)

    results = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    results.to_csv(paths["tables_dir"] / "full_cohort_model_results.csv", index=False)
    with open(paths["results_dir"] / "full_cohort_model_run.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_rows": int(len(data)),
                "features_after_engineering": int(X.shape[1]),
                "split": split,
                "statuses": statuses,
                "results": results.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
