#!/usr/bin/env python3
"""Calibration-aware final optimized experiments.

This runner keeps the test split sealed during hyperparameter search. Trial rows
are logged with validation metrics; test metrics are written only for selected
candidate models.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

META_COLS = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp",
    "dataset_path",
    "model_name",
    "feature_set",
    "token_variant",
    "split_seed",
    "model_seed",
    "hyperparameters",
    "validation_AUROC",
    "validation_AUPRC",
    "validation_Brier",
    "validation_ECE",
    "test_AUROC_if_selected",
    "test_AUPRC_if_selected",
    "test_Brier_if_selected",
    "test_ECE_if_selected",
    "runtime",
    "status",
    "failure_reason",
    "notes",
    "output_files",
]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict[str, Path]:
    paths = {}
    for key, value in cfg["paths"].items():
        path = Path(value)
        paths[key] = path if path.is_absolute() else ROOT / path
    return paths


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (probs >= left) & (probs <= right) if i == 0 else (probs > left) & (probs <= right)
        if np.any(mask):
            ece += float(mask.mean() * abs(probs[mask].mean() - y_true[mask].mean()))
    return ece


def calibration_intercept_slope(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2 or float(np.std(probs)) == 0.0:
        return math.nan, math.nan
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
        model.fit(logits, y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return math.nan, math.nan


def metric_row(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y_true, probs)
    return {
        "AUROC": float(roc_auc_score(y_true, probs)),
        "AUPRC": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall": float(tp / (tp + fn)) if (tp + fn) else math.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "F1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "Brier": float(brier_score_loss(y_true, probs)),
        "ECE": ece_score(y_true, probs),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 121):
        score = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t


def jsonable(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clinical_scale(raw: np.ndarray, cfg: dict) -> np.ndarray:
    lab_itemids = cfg["symptoms"]["lab_itemids"]
    vital_itemids = cfg["symptoms"]["vital_itemids"]
    ranges = {}
    ranges.update({int(k): tuple(v) for k, v in cfg["symptoms"]["lab_ranges"].items()})
    ranges.update({int(k): tuple(v) for k, v in cfg["symptoms"]["vital_ranges"].items()})
    out = raw.astype(np.float32, copy=True)
    for j, itemid in enumerate((lab_itemids + vital_itemids)[: out.shape[1]]):
        lo, hi = ranges[int(itemid)]
        out[:, j] = np.clip((out[:, j] - lo) / (hi - lo), 0.0, 1.0)
    return out


@dataclass
class FeatureStore:
    data: pd.DataFrame
    feature_cols: list[str]
    y: np.ndarray
    tabular: np.ndarray
    token_flat: np.ndarray | None


def load_feature_store(cfg: dict, paths: dict[str, Path], train_idx: np.ndarray) -> FeatureStore:
    dataset_path = paths["processed_dir"] / "thesis_dataset.parquet"
    data = pd.read_parquet(dataset_path)
    feature_cols = [c for c in data.columns if c not in META_COLS]
    raw = data[feature_cols].to_numpy(dtype=np.float32)
    scaled = clinical_scale(raw, cfg)
    missing = np.isnan(scaled).astype(np.float32)
    med = np.nanmedian(scaled[train_idx], axis=0)
    med = np.nan_to_num(med, nan=0.5)
    imputed = scaled.copy()
    inds = np.where(np.isnan(imputed))
    imputed[inds] = np.take(med, inds[1])
    imputed = np.nan_to_num(imputed, nan=0.5, posinf=0.5, neginf=0.5)
    try:
        from utils.feature_engineering import add_engineered_features

        engineered = add_engineered_features(imputed)
    except Exception:
        engineered = imputed
    tabular = np.hstack([engineered, missing]).astype(np.float32)
    token_path = paths["processed_dir"] / "tokens.npy"
    token_flat = None
    if token_path.exists():
        tokens = np.load(token_path).astype(np.float32)
        token_flat = tokens.reshape(tokens.shape[0], -1)
    return FeatureStore(data=data, feature_cols=feature_cols, y=data["feasible"].to_numpy(dtype=int), tabular=tabular, token_flat=token_flat)


def feature_matrix(store: FeatureStore, feature_set: str) -> np.ndarray:
    if feature_set == "tabular":
        return store.tabular
    if feature_set == "token":
        if store.token_flat is None:
            raise FileNotFoundError("tokens.npy is missing")
        return store.token_flat
    if feature_set == "tabular_token":
        if store.token_flat is None:
            raise FileNotFoundError("tokens.npy is missing")
        return np.hstack([store.tabular, store.token_flat]).astype(np.float32)
    raise ValueError(f"Unknown feature_set={feature_set}")


def predict_positive(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    z = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-z))


def platt_fit(y_val: np.ndarray, probs_val: np.ndarray):
    clipped = np.clip(probs_val, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
    model.fit(logits, y_val)
    return model


def platt_apply(model, probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def calibrate_choice(y_val: np.ndarray, probs_val: np.ndarray, probs_test: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    choices = [("none", probs_val, probs_test)]
    try:
        platt = platt_fit(y_val, probs_val)
        choices.append(("platt", platt_apply(platt, probs_val), platt_apply(platt, probs_test)))
    except Exception:
        pass
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs_val, y_val)
        choices.append(("isotonic", iso.predict(probs_val), iso.predict(probs_test)))
    except Exception:
        pass
    choices.sort(key=lambda item: (brier_score_loss(y_val, np.clip(item[1], 0, 1)), ece_score(y_val, np.clip(item[1], 0, 1))))
    name, val_cal, test_cal = choices[0]
    return name, np.clip(val_cal, 0, 1), np.clip(test_cal, 0, 1)


def baseline_factories(seed: int) -> dict[str, object]:
    factories: dict[str, object] = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, n_jobs=-1),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "Neural Baseline": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3, max_iter=180, early_stopping=True, random_state=seed),
        ),
    }
    try:
        import xgboost as xgb

        factories["XGBoost"] = xgb.XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass
    try:
        import lightgbm as lgb

        factories["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=350,
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

        factories["CatBoost"] = cb.CatBoostClassifier(
            iterations=350,
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


def candidate_from_params(model_name: str, params: dict, seed: int):
    if model_name == "Logistic Regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, random_state=seed, n_jobs=-1, **params),
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_name == "Neural Baseline":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=220, early_stopping=True, random_state=seed, **params),
        )
    if model_name == "XGBoost":
        import xgboost as xgb

        return xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1, **params)
    if model_name == "LightGBM":
        import lightgbm as lgb

        return lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if model_name == "CatBoost":
        import catboost as cb

        return cb.CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False, thread_count=-1, **params)
    raise ValueError(model_name)


def param_spaces() -> dict[str, dict]:
    return {
        "Logistic Regression": {
            "C": np.logspace(-2, 1.2, 12),
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 8, 12, 18],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
            "class_weight": ["balanced", "balanced_subsample", None],
        },
        "XGBoost": {
            "n_estimators": [250, 400, 650],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.02, 0.04, 0.06, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0, 8.0],
            "scale_pos_weight": [0.8, 1.0, 1.25],
        },
        "LightGBM": {
            "n_estimators": [250, 400, 650],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 4, 6, 8],
            "learning_rate": [0.02, 0.04, 0.06, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0, 8.0],
            "class_weight": [None, "balanced"],
        },
        "CatBoost": {
            "iterations": [300, 500, 700],
            "depth": [4, 5, 6, 8],
            "learning_rate": [0.02, 0.04, 0.06, 0.1],
            "l2_leaf_reg": [1, 3, 5, 8, 12],
            "auto_class_weights": ["Balanced", None],
        },
        "Neural Baseline": {
            "hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32), (256, 128)],
            "alpha": [1e-5, 1e-4, 1e-3, 3e-3],
            "learning_rate_init": [3e-4, 1e-3, 3e-3],
            "batch_size": [128, 256, 512],
        },
    }


def registry_row(
    exp_id: str,
    dataset_path: Path,
    model_name: str,
    feature_set: str,
    token_variant: str,
    split_seed: int,
    model_seed: int,
    params: dict,
    runtime: float,
    status: str,
    failure_reason: str = "",
    notes: str = "",
    val: dict | None = None,
    test: dict | None = None,
    output_files: list[str] | None = None,
) -> dict:
    val = val or {}
    test = test or {}
    return {
        "experiment_id": exp_id,
        "timestamp": now_iso(),
        "dataset_path": str(dataset_path),
        "model_name": model_name,
        "feature_set": feature_set,
        "token_variant": token_variant,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "hyperparameters": jsonable(params),
        "validation_AUROC": val.get("AUROC", ""),
        "validation_AUPRC": val.get("AUPRC", ""),
        "validation_Brier": val.get("Brier", ""),
        "validation_ECE": val.get("ECE", ""),
        "test_AUROC_if_selected": test.get("AUROC", ""),
        "test_AUPRC_if_selected": test.get("AUPRC", ""),
        "test_Brier_if_selected": test.get("Brier", ""),
        "test_ECE_if_selected": test.get("ECE", ""),
        "runtime": round(runtime, 3),
        "status": status,
        "failure_reason": failure_reason,
        "notes": notes,
        "output_files": ";".join(output_files or []),
    }


def run_baselines(store: FeatureStore, paths: dict[str, Path], train_idx, val_idx, test_idx, seed: int, registry: list[dict]) -> pd.DataFrame:
    out_dir = paths["results_dir"] / "reproduced_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    X = feature_matrix(store, "tabular")
    train_val_idx = np.concatenate([train_idx, val_idx])
    for name, model in baseline_factories(seed).items():
        t0 = time.time()
        try:
            model.fit(X[train_idx], store.y[train_idx])
            val_probs = predict_positive(model, X[val_idx])
            threshold = tune_threshold(store.y[val_idx], val_probs)
            final_model = clone(model)
            final_model.fit(X[train_val_idx], store.y[train_val_idx])
            test_probs = predict_positive(final_model, X[test_idx])
            row = {"model": name, "feature_set": "tabular", **metric_row(store.y[test_idx], test_probs, threshold)}
            rows.append(row)
            registry.append(registry_row(f"baseline_{name.lower().replace(' ', '_')}", paths["processed_dir"] / "thesis_dataset.parquet", name, "tabular", "none", seed, seed, model.get_params() if hasattr(model, "get_params") else {}, time.time() - t0, "completed", notes="Baseline reproduction; not used for hyperparameter selection.", val=metric_row(store.y[val_idx], val_probs, threshold), test=row))
            pd.DataFrame({"stay_id": store.data.iloc[test_idx]["stay_id"], "y_true": store.y[test_idx], "probability": test_probs}).to_csv(out_dir / f"{name.lower().replace(' ', '_')}_test_predictions.csv", index=False)
            print(f"baseline {name}: AUROC={row['AUROC']:.4f} AUPRC={row['AUPRC']:.4f}", flush=True)
        except Exception as exc:
            registry.append(registry_row(f"baseline_{name.lower().replace(' ', '_')}", paths["processed_dir"] / "thesis_dataset.parquet", name, "tabular", "none", seed, seed, {}, time.time() - t0, "failed", failure_reason=str(exc)))
            print(f"baseline {name} failed: {exc}", flush=True)
    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    df.to_csv(out_dir / "reproduced_baseline_results.csv", index=False)
    return df


def run_optimization(store: FeatureStore, paths: dict[str, Path], train_idx, val_idx, test_idx, seed: int, n_trials: int, registry: list[dict]) -> pd.DataFrame:
    trial_rows = []
    spaces = param_spaces()
    model_feature_sets = [
        ("Logistic Regression", "tabular"),
        ("Random Forest", "tabular"),
        ("XGBoost", "tabular"),
        ("LightGBM", "tabular"),
        ("CatBoost", "tabular"),
        ("Neural Baseline", "tabular"),
        ("Random Forest", "token"),
        ("XGBoost", "token"),
        ("LightGBM", "token"),
        ("CatBoost", "token"),
        ("Random Forest", "tabular_token"),
        ("XGBoost", "tabular_token"),
        ("LightGBM", "tabular_token"),
        ("CatBoost", "tabular_token"),
    ]
    for model_name, fset in model_feature_sets:
        if fset != "tabular" and store.token_flat is None:
            continue
        X = feature_matrix(store, fset)
        sampled = list(ParameterSampler(spaces[model_name], n_iter=n_trials, random_state=seed + len(trial_rows)))
        for i, params in enumerate(sampled, start=1):
            exp_id = f"opt_{model_name.lower().replace(' ', '_')}_{fset}_{i:03d}"
            t0 = time.time()
            try:
                model = candidate_from_params(model_name, params, seed)
                model.fit(X[train_idx], store.y[train_idx])
                val_probs = predict_positive(model, X[val_idx])
                threshold = tune_threshold(store.y[val_idx], val_probs)
                val = metric_row(store.y[val_idx], val_probs, threshold)
                row = {
                    "experiment_id": exp_id,
                    "model": model_name,
                    "feature_set": fset,
                    "token_variant": "flat_25x4" if fset != "tabular" else "none",
                    "params": params,
                    **{f"validation_{k}": v for k, v in val.items()},
                }
                trial_rows.append(row)
                registry.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", model_name, fset, row["token_variant"], seed, seed, params, time.time() - t0, "completed", val=val))
                print(f"{exp_id}: val AUROC={val['AUROC']:.4f} AUPRC={val['AUPRC']:.4f} Brier={val['Brier']:.4f}", flush=True)
            except Exception as exc:
                registry.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", model_name, fset, "flat_25x4" if fset != "tabular" else "none", seed, seed, params, time.time() - t0, "failed", failure_reason=str(exc)))
                print(f"{exp_id} failed: {exc}", flush=True)
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(paths["results_dir"] / "optimization" / "validation_trials.csv", index=False)
    return trials


def run_fewshot_prototypes(store: FeatureStore, paths: dict[str, Path], train_idx, val_idx, test_idx, seed: int, registry: list[dict]) -> pd.DataFrame:
    if store.token_flat is None:
        return pd.DataFrame()
    X = store.token_flat
    y = store.y
    rows = []
    rng = np.random.default_rng(seed)
    for k in [1, 2, 4, 8, 16, 32, 64]:
        val_probs_runs, test_probs_runs = [], []
        t0 = time.time()
        for rep in range(5):
            support = []
            for c in [0, 1]:
                pool = train_idx[y[train_idx] == c]
                support.extend(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
            support = np.asarray(support)
            means = []
            for c in [0, 1]:
                means.append(X[support[y[support] == c]].mean(axis=0))
            proto = np.vstack(means)
            proto = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8, None)
            for idx, bucket in [(val_idx, val_probs_runs), (test_idx, test_probs_runs)]:
                q = X[idx] / np.clip(np.linalg.norm(X[idx], axis=1, keepdims=True), 1e-8, None)
                sim = q @ proto.T
                logits = sim[:, 1] - sim[:, 0]
                bucket.append(1.0 / (1.0 + np.exp(-5.0 * logits)))
        val_probs = np.mean(val_probs_runs, axis=0)
        test_probs = np.mean(test_probs_runs, axis=0)
        threshold = tune_threshold(y[val_idx], val_probs)
        val = metric_row(y[val_idx], val_probs, threshold)
        test = metric_row(y[test_idx], test_probs, threshold)
        exp_id = f"fewshot_token_prototype_k{k}"
        row = {"experiment_id": exp_id, "model": "Few-Shot ETHOS Prototype", "feature_set": "token", "K": k, **{f"validation_{kk}": vv for kk, vv in val.items()}, **{f"test_{kk}": vv for kk, vv in test.items()}}
        rows.append(row)
        registry.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", "Few-Shot ETHOS Prototype", "token", "flat_25x4", seed, seed, {"K": k, "repeats": 5}, time.time() - t0, "completed", notes="Class-balanced token prototype baseline; test reported for prespecified few-shot K sweep.", val=val, test=test))
        print(f"few-shot prototype K={k}: val AUROC={val['AUROC']:.4f}, test AUROC={test['AUROC']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(paths["tables_dir"] / "ethos_fewshot_token_results.csv", index=False)
    return df


def select_candidates(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    df = trials.copy()
    df["selection_score"] = df["validation_AUROC"] + 0.15 * df["validation_AUPRC"] - 0.2 * df["validation_ECE"] - 0.1 * df["validation_Brier"]
    selected = []
    selected.append(df.sort_values("validation_AUROC", ascending=False).head(1))
    selected.append(df.sort_values("validation_AUPRC", ascending=False).head(1))
    selected.append(df.sort_values(["validation_Brier", "validation_ECE"], ascending=[True, True]).head(1))
    selected.append(df.sort_values("selection_score", ascending=False).head(1))
    for key in ["model", "feature_set"]:
        selected.append(df.sort_values("selection_score", ascending=False).groupby(key, as_index=False).head(1))
    out = pd.concat(selected, ignore_index=True).drop_duplicates("experiment_id")
    return out.sort_values("selection_score", ascending=False)


def final_evaluate_selected(store: FeatureStore, paths: dict[str, Path], selected: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, registry: list[dict]) -> pd.DataFrame:
    rows = []
    model_dir = paths["results_dir"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = paths["results_dir"] / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for _, sel in selected.iterrows():
        model_name = sel["model"]
        fset = sel["feature_set"]
        params = sel["params"]
        if isinstance(params, str):
            params = json.loads(params)
        X = feature_matrix(store, fset)
        t0 = time.time()
        try:
            model = candidate_from_params(model_name, params, seed)
            model.fit(X[train_idx], store.y[train_idx])
            val_probs_raw = predict_positive(model, X[val_idx])
            test_probs_raw = predict_positive(model, X[test_idx])
            calibration, val_probs, test_probs = calibrate_choice(store.y[val_idx], val_probs_raw, test_probs_raw)
            threshold = tune_threshold(store.y[val_idx], val_probs)
            val = metric_row(store.y[val_idx], val_probs, threshold)
            test = metric_row(store.y[test_idx], test_probs, threshold)
            exp_id = f"selected_{sel['experiment_id']}"
            pred_path = pred_dir / f"{exp_id}_test_predictions.csv"
            pd.DataFrame({"stay_id": store.data.iloc[test_idx]["stay_id"], "y_true": store.y[test_idx], "probability": test_probs}).to_csv(pred_path, index=False)
            joblib.dump({"model": model, "calibration": calibration, "params": params, "feature_set": fset}, model_dir / f"{exp_id}.joblib")
            row = {
                "experiment_id": exp_id,
                "model": model_name,
                "feature_set": fset,
                "token_variant": "flat_25x4" if fset != "tabular" else "none",
                "calibration": calibration,
                "hyperparameters": jsonable(params),
                **test,
                "validation_AUROC": val["AUROC"],
                "validation_AUPRC": val["AUPRC"],
                "validation_Brier": val["Brier"],
                "validation_ECE": val["ECE"],
            }
            rows.append(row)
            registry.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", model_name, fset, row["token_variant"], seed, seed, params, time.time() - t0, "selected_test_completed", notes=f"Calibration selected on validation: {calibration}", val=val, test=test, output_files=[str(pred_path)]))
            print(f"selected {model_name}/{fset}: test AUROC={test['AUROC']:.4f} AUPRC={test['AUPRC']:.4f} Brier={test['Brier']:.4f}", flush=True)
        except Exception as exc:
            registry.append(registry_row(f"selected_{sel['experiment_id']}", paths["processed_dir"] / "thesis_dataset.parquet", model_name, fset, "flat_25x4" if fset != "tabular" else "none", seed, seed, params, time.time() - t0, "selected_test_failed", failure_reason=str(exc)))
    final = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    final.to_csv(paths["tables_dir"] / "optimized_model_comparison.csv", index=False)
    return final


def save_curves_and_thresholds(final: pd.DataFrame, paths: dict[str, Path], y_test: np.ndarray) -> None:
    pred_dir = paths["results_dir"] / "predictions"
    fig_data = paths["results_dir"] / "calibration"
    fig_data.mkdir(parents=True, exist_ok=True)
    threshold_rows = []
    for _, row in final.head(5).iterrows():
        pred_path = pred_dir / f"{row['experiment_id']}_test_predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        probs = pred["probability"].to_numpy()
        y = pred["y_true"].to_numpy()
        fpr, tpr, _ = roc_curve(y, probs)
        precision, recall, _ = precision_recall_curve(y, probs)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(fig_data / f"{row['experiment_id']}_roc_curve.csv", index=False)
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(fig_data / f"{row['experiment_id']}_pr_curve.csv", index=False)
        for t in np.linspace(0.2, 0.8, 13):
            m = metric_row(y, probs, float(t))
            threshold_rows.append({"experiment_id": row["experiment_id"], "model": row["model"], "threshold": float(t), "recall": m["recall"], "specificity": m["specificity"], "F1": m["F1"], "accuracy": m["accuracy"]})
    pd.DataFrame(threshold_rows).to_csv(paths["tables_dir"] / "threshold_sensitivity_specificity.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=int(os.environ.get("FINAL_OPT_TRIALS", "10")))
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    paths = resolve_paths(cfg)
    for key in ["results_dir", "tables_dir", "figures_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    for sub in ["optimization", "models", "calibration", "logs"]:
        (paths["results_dir"] / sub).mkdir(parents=True, exist_ok=True)

    y_tmp = pd.read_parquet(paths["processed_dir"] / "thesis_dataset.parquet", columns=["feasible"])["feasible"].to_numpy(dtype=int)
    train_val_idx, test_idx = train_test_split(np.arange(len(y_tmp)), test_size=0.2, random_state=cfg["seed"], stratify=y_tmp)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=cfg["seed"], stratify=y_tmp[train_val_idx])
    np.savez(paths["results_dir"] / "optimization" / "split_indices_seed42.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    store = load_feature_store(cfg, paths, train_idx)

    split_summary = {
        "seed": cfg["seed"],
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_label_distribution": {str(k): int(v) for k, v in pd.Series(store.y[train_idx]).value_counts().sort_index().items()},
        "validation_label_distribution": {str(k): int(v) for k, v in pd.Series(store.y[val_idx]).value_counts().sort_index().items()},
        "test_label_distribution": {str(k): int(v) for k, v in pd.Series(store.y[test_idx]).value_counts().sort_index().items()},
        "feature_columns_raw": len(store.feature_cols),
        "feature_columns_model": int(store.tabular.shape[1]),
        "token_flat_columns": int(store.token_flat.shape[1]) if store.token_flat is not None else 0,
    }
    (paths["results_dir"] / "optimization" / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    registry: list[dict] = []
    baseline_df = pd.DataFrame()
    if not args.skip_baselines:
        baseline_df = run_baselines(store, paths, train_idx, val_idx, test_idx, cfg["seed"], registry)
    trials = run_optimization(store, paths, train_idx, val_idx, test_idx, cfg["seed"], args.trials, registry)
    fewshot = run_fewshot_prototypes(store, paths, train_idx, val_idx, test_idx, cfg["seed"], registry)
    selected = select_candidates(trials)
    selected.to_csv(paths["results_dir"] / "optimization" / "selected_candidates_validation_only.csv", index=False)
    final = final_evaluate_selected(store, paths, selected, train_idx, val_idx, test_idx, cfg["seed"], registry)

    if not fewshot.empty:
        best_fs = fewshot.sort_values("validation_AUROC", ascending=False).head(1)
        fs_test = {
            "experiment_id": best_fs.iloc[0]["experiment_id"],
            "model": best_fs.iloc[0]["model"],
            "feature_set": "token",
            "token_variant": "flat_25x4",
            "calibration": "none",
            "hyperparameters": jsonable({"K": int(best_fs.iloc[0]["K"])}),
            "AUROC": best_fs.iloc[0]["test_AUROC"],
            "AUPRC": best_fs.iloc[0]["test_AUPRC"],
            "accuracy": best_fs.iloc[0]["test_accuracy"],
            "recall": best_fs.iloc[0]["test_recall"],
            "specificity": best_fs.iloc[0]["test_specificity"],
            "F1": best_fs.iloc[0]["test_F1"],
            "Brier": best_fs.iloc[0]["test_Brier"],
            "ECE": best_fs.iloc[0]["test_ECE"],
            "calibration_intercept": best_fs.iloc[0]["test_calibration_intercept"],
            "calibration_slope": best_fs.iloc[0]["test_calibration_slope"],
            "threshold": best_fs.iloc[0]["test_threshold"],
            "tn": best_fs.iloc[0]["test_tn"],
            "fp": best_fs.iloc[0]["test_fp"],
            "fn": best_fs.iloc[0]["test_fn"],
            "tp": best_fs.iloc[0]["test_tp"],
            "validation_AUROC": best_fs.iloc[0]["validation_AUROC"],
            "validation_AUPRC": best_fs.iloc[0]["validation_AUPRC"],
            "validation_Brier": best_fs.iloc[0]["validation_Brier"],
            "validation_ECE": best_fs.iloc[0]["validation_ECE"],
        }
        final = pd.concat([final, pd.DataFrame([fs_test])], ignore_index=True).sort_values("AUROC", ascending=False)
        final.to_csv(paths["tables_dir"] / "optimized_model_comparison.csv", index=False)

    final_rank = final.copy()
    if not final_rank.empty:
        final_rank["overall_score"] = final_rank["AUROC"] + 0.15 * final_rank["AUPRC"] - 0.2 * final_rank["ECE"] - 0.1 * final_rank["Brier"]
        final_rank = final_rank.sort_values("overall_score", ascending=False)
    final_rank.to_csv(paths["tables_dir"] / "final_model_ranking.csv", index=False)
    save_curves_and_thresholds(final_rank, paths, store.y[test_idx])
    pd.DataFrame(registry, columns=REGISTRY_COLUMNS).to_csv(paths["results_dir"] / "optimization" / "experiment_registry.csv", index=False)

    summary = {
        "baseline_rows": int(len(baseline_df)),
        "validation_trials": int(len(trials)),
        "fewshot_rows": int(len(fewshot)),
        "selected_test_models": int(len(final)),
        "best_by_auroc": final.sort_values("AUROC", ascending=False).head(1).to_dict(orient="records") if not final.empty else [],
        "best_overall": final_rank.head(1).to_dict(orient="records") if not final_rank.empty else [],
    }
    (paths["results_dir"] / "optimization" / "final_optimized_experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
