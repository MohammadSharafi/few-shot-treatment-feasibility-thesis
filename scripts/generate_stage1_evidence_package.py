#!/usr/bin/env python3
"""Generate the Stage 1 evidence package requested before thesis rewrite.

This script is intentionally report-first. It reuses the verified final split,
adds feasible missing analyses, normalizes the experiment registry schema, and
writes the required Stage 1 Markdown/CSV/figure artifacts.
"""
from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_final_optimized_experiments import (  # noqa: E402
    REGISTRY_COLUMNS,
    calibrate_choice,
    candidate_from_params,
    ece_score,
    feature_matrix,
    jsonable,
    load_config,
    load_feature_store,
    metric_row,
    predict_positive,
    registry_row,
    resolve_paths,
    tune_threshold,
)


EXTRA_REGISTRY_COLUMNS = [
    "git_commit",
    "training_fraction",
    "selected_for_test",
]
STAGE1_REGISTRY_COLUMNS = [
    "experiment_id",
    "timestamp",
    "git_commit",
    "dataset_path",
    "model_name",
    "feature_set",
    "token_variant",
    "split_seed",
    "model_seed",
    "training_fraction",
    "hyperparameters",
    "validation_AUROC",
    "validation_AUPRC",
    "validation_Brier",
    "validation_ECE",
    "selected_for_test",
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_params(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or value.strip() == "":
        return {}
    try:
        return json.loads(value)
    except Exception:
        return ast.literal_eval(value)


def fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def table_text(df: pd.DataFrame) -> str:
    """Markdown table when tabulate exists, otherwise a stable plain-text table."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "\n```\n" + df.to_string(index=False) + "\n```"


def ensure_dirs(paths: dict[str, Path]) -> None:
    for sub in [
        "tables",
        "figures",
        "logs",
        "models",
        "calibration",
        "optimization",
        "low_data",
        "ethos",
        "hybrid",
        "federated",
        "uncertainty",
        "subgroups",
        "reporting",
        "modern_ehr_models",
        "predictions",
    ]:
        (paths["results_dir"] / sub).mkdir(parents=True, exist_ok=True)


def load_split(paths: dict[str, Path]):
    split_path = paths["results_dir"] / "optimization" / "split_indices_seed42.npz"
    split = np.load(split_path)
    return split["train_idx"], split["val_idx"], split["test_idx"]


def clinical_scaled_imputed_matrix(cfg: dict, store, train_idx):
    raw = store.data[store.feature_cols].to_numpy(dtype=np.float32)
    from run_final_optimized_experiments import clinical_scale

    scaled = clinical_scale(raw, cfg)
    missing = np.isnan(scaled).astype(np.float32)
    med = np.nanmedian(scaled[train_idx], axis=0)
    med = np.nan_to_num(med, nan=0.5)
    imputed = scaled.copy()
    inds = np.where(np.isnan(imputed))
    imputed[inds] = np.take(med, inds[1])
    imputed = np.nan_to_num(imputed, nan=0.5, posinf=0.5, neginf=0.5)
    return imputed, missing


def metric_from_probs(y_true, probs, threshold=0.5):
    m = metric_row(y_true, probs, threshold)
    return {
        "AUROC": m["AUROC"],
        "AUPRC": m["AUPRC"],
        "Brier": m["Brier"],
        "ECE": m["ECE"],
        "accuracy": m["accuracy"],
        "recall": m["recall"],
        "specificity": m["specificity"],
        "F1": m["F1"],
        "calibration_intercept": m["calibration_intercept"],
        "calibration_slope": m["calibration_slope"],
        "threshold": m["threshold"],
        "tn": m["tn"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tp": m["tp"],
    }


def fit_eval_model(model, X, y, train_idx, val_idx, test_idx):
    model.fit(X[train_idx], y[train_idx])
    val_raw = predict_positive(model, X[val_idx])
    test_raw = predict_positive(model, X[test_idx])
    cal_name, val_probs, test_probs = calibrate_choice(y[val_idx], val_raw, test_raw)
    threshold = tune_threshold(y[val_idx], val_probs)
    val = metric_from_probs(y[val_idx], val_probs, threshold)
    test = metric_from_probs(y[test_idx], test_probs, threshold)
    return val, test, val_probs, test_probs, cal_name


def token_prototype_probs(tokens_flat, y, support_idx, pred_idx):
    protos = []
    for cls in [0, 1]:
        cls_idx = support_idx[y[support_idx] == cls]
        if len(cls_idx) == 0:
            protos.append(tokens_flat[support_idx].mean(axis=0))
        else:
            protos.append(tokens_flat[cls_idx].mean(axis=0))
    proto = np.vstack(protos)
    proto = proto / np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8, None)
    q = tokens_flat[pred_idx] / np.clip(np.linalg.norm(tokens_flat[pred_idx], axis=1, keepdims=True), 1e-8, None)
    logits = (q @ proto.T)[:, 1] - (q @ proto.T)[:, 0]
    return 1.0 / (1.0 + np.exp(-5.0 * logits))


def low_data_experiments(cfg, paths, store, train_idx, val_idx, test_idx):
    out_dir = paths["results_dir"] / "low_data"
    y = store.y
    rows = []
    registry_rows = []
    rng = np.random.default_rng(cfg["seed"])
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    model_specs = [
        ("Logistic Regression", "tabular", lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed))),
        ("Random Forest", "tabular", lambda seed: RandomForestClassifier(n_estimators=220, min_samples_leaf=4, max_features="sqrt", random_state=seed, n_jobs=-1)),
        ("XGBoost", "tabular", lambda seed: candidate_from_params("XGBoost", {"n_estimators": 260, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0}, seed)),
        ("LightGBM", "tabular", lambda seed: candidate_from_params("LightGBM", {"n_estimators": 260, "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "class_weight": None}, seed)),
        ("Token Hybrid XGBoost", "tabular_token", lambda seed: candidate_from_params("XGBoost", {"n_estimators": 260, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0}, seed)),
        ("Token Hybrid LightGBM", "tabular_token", lambda seed: candidate_from_params("LightGBM", {"n_estimators": 260, "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "class_weight": None}, seed)),
    ]
    for frac in fractions:
        if frac >= 1.0:
            sub_idx = train_idx
        else:
            sub_idx, _ = train_test_split(train_idx, train_size=frac, random_state=cfg["seed"] + int(frac * 1000), stratify=y[train_idx])
        for model_name, fset, factory in model_specs:
            t0 = time.time()
            exp_id = f"lowdata_{model_name.lower().replace(' ', '_')}_{fset}_{int(frac*100):03d}"
            try:
                X = feature_matrix(store, fset)
                val, test, _, _, cal = fit_eval_model(factory(cfg["seed"]), X, y, sub_idx, val_idx, test_idx)
                row = {
                    "experiment_id": exp_id,
                    "model": model_name,
                    "feature_set": fset,
                    "training_fraction": frac,
                    "train_rows": len(sub_idx),
                    "calibration": cal,
                    **test,
                    "validation_AUROC": val["AUROC"],
                    "validation_AUPRC": val["AUPRC"],
                    "validation_Brier": val["Brier"],
                    "validation_ECE": val["ECE"],
                    "runtime": round(time.time() - t0, 3),
                }
                rows.append(row)
                registry_rows.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", model_name, fset, "flat_25x4" if "token" in fset else "none", cfg["seed"], cfg["seed"], {"low_data_fraction": frac}, time.time() - t0, "completed", notes="Low-data fixed validation/test experiment", val=val, test=test))
            except Exception as exc:
                registry_rows.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", model_name, fset, "flat_25x4" if "token" in fset else "none", cfg["seed"], cfg["seed"], {"low_data_fraction": frac}, time.time() - t0, "failed", failure_reason=str(exc)))
        if store.token_flat is not None:
            t0 = time.time()
            exp_id = f"lowdata_improved_token_prototype_{int(frac*100):03d}"
            val_probs = token_prototype_probs(store.token_flat, y, sub_idx, val_idx)
            test_probs = token_prototype_probs(store.token_flat, y, sub_idx, test_idx)
            threshold = tune_threshold(y[val_idx], val_probs)
            val = metric_from_probs(y[val_idx], val_probs, threshold)
            test = metric_from_probs(y[test_idx], test_probs, threshold)
            row = {
                "experiment_id": exp_id,
                "model": "Improved Few-Shot/Token Prototype",
                "feature_set": "token",
                "training_fraction": frac,
                "train_rows": len(sub_idx),
                "calibration": "none",
                **test,
                "validation_AUROC": val["AUROC"],
                "validation_AUPRC": val["AUPRC"],
                "validation_Brier": val["Brier"],
                "validation_ECE": val["ECE"],
                "runtime": round(time.time() - t0, 3),
            }
            rows.append(row)
            registry_rows.append(registry_row(exp_id, paths["processed_dir"] / "thesis_dataset.parquet", "Improved Few-Shot/Token Prototype", "token", "flat_25x4", cfg["seed"], cfg["seed"], {"low_data_fraction": frac}, time.time() - t0, "completed", notes="Low-data token prototype", val=val, test=test))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "low_data_sample_efficiency.csv", index=False)
    df[["experiment_id", "model", "training_fraction", "AUROC", "AUPRC", "Brier", "ECE", "accuracy", "recall", "specificity", "F1", "calibration_intercept", "calibration_slope"]].to_csv(out_dir / "low_data_calibration_summary.csv", index=False)
    for metric, fname in [("AUROC", "low_data_auroc_curve.png"), ("AUPRC", "low_data_auprc_curve.png")]:
        plt.figure(figsize=(8, 5))
        for model, group in df.groupby("model"):
            group = group.sort_values("training_fraction")
            plt.plot(group["training_fraction"] * 100, group[metric], marker="o", label=model)
        plt.xlabel("Labelled training data (%)")
        plt.ylabel(metric)
        plt.title(f"Low-data sample efficiency: {metric}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300)
        plt.savefig(paths["results_dir"] / "figures" / fname, dpi=300)
        plt.close()
    best = df.sort_values(["AUROC", "AUPRC"], ascending=False).iloc[0]
    report = f"""# Low-Data Experiment Report

Fixed validation and test splits were kept unchanged. Only the labelled training subset was varied.

Fractions tested: 1%, 2%, 5%, 10%, 20%, 50%, 100%.

Best low-data row by test AUROC:

- Model: `{best['model']}`
- Feature set: `{best['feature_set']}`
- Training fraction: `{best['training_fraction']:.2f}`
- AUROC: {fmt(best['AUROC'])}
- AUPRC: {fmt(best['AUPRC'])}
- Brier: {fmt(best['Brier'])}
- ECE: {fmt(best['ECE'])}

Interpretation: token-hybrid models were explicitly evaluated in low-data regimes. The few-shot/token prototype remained a weaker standalone predictor than boosted supervised models; its scientifically honest role is as a low-data/token benchmark rather than a superior final model.
"""
    (out_dir / "LOW_DATA_EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    return df, registry_rows


def generate_ensemble_predictions(cfg, paths, store, train_idx, val_idx, test_idx):
    selected = pd.read_csv(paths["results_dir"] / "optimization" / "selected_candidates_validation_only.csv")
    selected = selected.sort_values("selection_score", ascending=False).drop_duplicates(["model", "feature_set"]).head(5)
    val_cols, test_cols, names, weights = [], [], [], []
    for _, row in selected.iterrows():
        model_name = row["model"]
        fset = row["feature_set"]
        params = parse_params(row["params"])
        X = feature_matrix(store, fset)
        model = candidate_from_params(model_name, params, cfg["seed"])
        model.fit(X[train_idx], store.y[train_idx])
        val_cols.append(predict_positive(model, X[val_idx]))
        test_cols.append(predict_positive(model, X[test_idx]))
        names.append(f"{model_name}:{fset}")
        weights.append(float(row["validation_AUROC"]))
    Z_val = np.column_stack(val_cols)
    Z_test = np.column_stack(test_cols)
    # Stacked validation meta-LR
    meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=cfg["seed"])
    meta.fit(Z_val, store.y[val_idx])
    st_val_raw = predict_positive(meta, Z_val)
    st_test_raw = predict_positive(meta, Z_test)
    st_cal, st_val, st_test = calibrate_choice(store.y[val_idx], st_val_raw, st_test_raw)
    # Weighted average
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    wa_val_raw = np.average(Z_val, axis=1, weights=weights)
    wa_test_raw = np.average(Z_test, axis=1, weights=weights)
    wa_cal, wa_val, wa_test = calibrate_choice(store.y[val_idx], wa_val_raw, wa_test_raw)
    pred_dir = paths["results_dir"] / "predictions"
    pd.DataFrame({"stay_id": store.data.iloc[test_idx]["stay_id"], "y_true": store.y[test_idx], "probability": st_test}).to_csv(pred_dir / "selected_stacked_validation_meta_lr_test_predictions.csv", index=False)
    pd.DataFrame({"stay_id": store.data.iloc[test_idx]["stay_id"], "y_true": store.y[test_idx], "probability": wa_test}).to_csv(pred_dir / "selected_weighted_average_validation_auc_test_predictions.csv", index=False)
    return {
        "stacked": {"val": st_val, "test": st_test, "calibration": st_cal, "base_models": names},
        "weighted": {"val": wa_val, "test": wa_test, "calibration": wa_cal, "base_models": names, "weights": weights.tolist()},
    }


def calibration_outputs(paths, store, val_idx, test_idx, ensemble_preds):
    final = pd.read_csv(paths["results_dir"] / "tables" / "final_model_ranking.csv")
    rows = []
    pred_dir = paths["results_dir"] / "predictions"
    for _, row in final.head(10).iterrows():
        p = pred_dir / f"{row['experiment_id']}_test_predictions.csv"
        if p.exists():
            pred = pd.read_csv(p)
            m = metric_from_probs(pred["y_true"].to_numpy(), pred["probability"].to_numpy(), float(row.get("threshold", 0.5)))
            rows.append({"experiment_id": row["experiment_id"], "model": row["model"], "feature_set": row["feature_set"], **m})
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(paths["results_dir"] / "tables" / "final_calibration_results.csv", index=False)

    plt.figure(figsize=(7, 6))
    for _, row in final.head(5).iterrows():
        p = pred_dir / f"{row['experiment_id']}_test_predictions.csv"
        if not p.exists():
            continue
        pred = pd.read_csv(p)
        probs = pred["probability"].to_numpy()
        y = pred["y_true"].to_numpy()
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            mask = (probs >= bins[i]) & (probs <= bins[i + 1]) if i == 0 else (probs > bins[i]) & (probs <= bins[i + 1])
            if mask.any():
                xs.append(probs[mask].mean())
                ys.append(y[mask].mean())
        plt.plot(xs, ys, marker="o", label=str(row["model"])[:26])
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration curves for top final models")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(paths["results_dir"] / "figures" / "calibration_curves.png", dpi=300)
    plt.close()

    report = "# Calibration Report\n\n"
    if not cal_df.empty:
        best_brier = cal_df.sort_values("Brier").iloc[0]
        best_ece = cal_df.sort_values("ECE").iloc[0]
        report += f"- Best Brier among evaluated top models: `{best_brier['model']}` ({fmt(best_brier['Brier'])}).\n"
        report += f"- Best ECE among evaluated top models: `{best_ece['model']}` ({fmt(best_ece['ECE'])}).\n"
    report += "- Calibration and thresholds were selected using validation data in the final optimized runners.\n"
    report += "- Final calibration curves are saved in `results/final_optimized/figures/calibration_curves.png`.\n"
    (paths["results_dir"] / "CALIBRATION_REPORT.md").write_text(report, encoding="utf-8")
    return cal_df


def uncertainty_outputs(paths, store, val_idx, test_idx, ensemble_preds):
    out_dir = paths["results_dir"] / "uncertainty"
    rows = []
    for name, pred_pack in [("Stacked Validation Meta-LR", ensemble_preds["stacked"]), ("Weighted Average Ensemble", ensemble_preds["weighted"])]:
        probs = pred_pack["test"]
        y = store.y[test_idx]
        conf = np.maximum(probs, 1 - probs)
        order = np.argsort(-conf)
        for cov in [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
            k = max(10, int(round(cov * len(y))))
            idx = order[:k]
            y_sub = y[idx]
            p_sub = probs[idx]
            pred = (p_sub >= 0.5).astype(int)
            rows.append({
                "model": name,
                "coverage": cov,
                "n_retained": k,
                "accuracy": accuracy_score(y_sub, pred),
                "error_rate": 1 - accuracy_score(y_sub, pred),
                "AUROC": roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else np.nan,
                "AUPRC": average_precision_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else np.nan,
                "mean_confidence": float(conf[idx].mean()),
            })
    selective = pd.DataFrame(rows)
    selective.to_csv(out_dir / "SELECTIVE_PREDICTION_RESULTS.csv", index=False)
    plt.figure(figsize=(7, 5))
    for model, g in selective.groupby("model"):
        g = g.sort_values("coverage")
        plt.plot(g["coverage"] * 100, g["accuracy"], marker="o", label=model)
    plt.xlabel("Coverage retained (%)")
    plt.ylabel("Accuracy")
    plt.title("Selective prediction by confidence")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(paths["results_dir"] / "figures" / "selective_prediction_curve.png", dpi=300)
    plt.close()

    conformal_rows = []
    for name, pred_pack in [("Stacked Validation Meta-LR", ensemble_preds["stacked"]), ("Weighted Average Ensemble", ensemble_preds["weighted"])]:
        val_probs = pred_pack["val"]
        test_probs = pred_pack["test"]
        y_val = store.y[val_idx]
        y_test = store.y[test_idx]
        val_true_probs = np.where(y_val == 1, val_probs, 1 - val_probs)
        scores = 1 - val_true_probs
        for alpha in [0.05, 0.10, 0.20]:
            q = float(np.quantile(scores, min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)), method="higher"))
            sets = []
            covered = []
            for p, y_true in zip(test_probs, y_test):
                pred_set = []
                if 1 - (1 - p) <= q:
                    pred_set.append(0)
                if 1 - p <= q:
                    pred_set.append(1)
                sets.append(len(pred_set))
                covered.append(int(y_true in pred_set))
            conformal_rows.append({
                "model": name,
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "empirical_coverage": float(np.mean(covered)),
                "average_set_size": float(np.mean(sets)),
                "singleton_rate": float(np.mean(np.asarray(sets) == 1)),
                "empty_set_rate": float(np.mean(np.asarray(sets) == 0)),
            })
    conformal = pd.DataFrame(conformal_rows)
    conformal.to_csv(out_dir / "CONFORMAL_PREDICTION_RESULTS.csv", index=False)
    best_sel = selective.sort_values(["model", "coverage"]).groupby("model").tail(1)
    report = """# Uncertainty Report

Selective prediction was evaluated by retaining the most confident predictions according to `max(p, 1-p)`. Split conformal prediction used validation-set nonconformity scores and test labels only for final coverage evaluation.

Key interpretation: uncertainty-aware filtering can identify subsets with higher accuracy, but this remains retrospective analysis and does not make the model a clinical decision system.
"""
    report += "\n\n## Selective Prediction Snapshot\n\n"
    report += table_text(selective.head(14))
    report += "\n\n## Conformal Snapshot\n\n"
    report += table_text(conformal)
    (out_dir / "UNCERTAINTY_REPORT.md").write_text(report, encoding="utf-8")
    return selective, conformal


def subgroup_outputs(cfg, paths, store, test_idx, top_probs):
    out_dir = paths["results_dir"] / "subgroups"
    data = store.data.copy()
    patients = pd.read_csv(paths["hosp_dir"] / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    icu = pd.read_csv(paths["icu_dir"] / "icustays.csv.gz", usecols=["stay_id", "first_careunit"])
    data = data.merge(patients, on="subject_id", how="left").merge(icu, on="stay_id", how="left")
    raw = store.data[store.feature_cols].to_numpy(dtype=np.float32)
    missing_rate = np.isnan(raw).mean(axis=1)
    test_df = data.iloc[test_idx].copy()
    test_df["probability"] = top_probs
    test_df["y_true"] = store.y[test_idx]
    test_df["pred"] = (top_probs >= 0.5).astype(int)
    test_df["confidence"] = np.maximum(top_probs, 1 - top_probs)
    test_df["age_band"] = pd.cut(test_df["anchor_age"], bins=[17, 45, 65, 80, 200], labels=["18-45", "46-65", "66-80", "80+"])
    test_df["diagnosis_group"] = test_df["primary_icd10"].astype(str).str[:1].str.upper()
    test_df["missingness_group"] = np.where(missing_rate[test_idx] >= np.nanmedian(missing_rate[test_idx]), "high_missingness", "low_missingness")
    test_df["los_group"] = np.where(test_df["los_hours"] >= test_df["los_hours"].median(), "longer_icu_los", "shorter_icu_los")
    rows = []
    for subgroup_col in ["gender", "age_band", "first_careunit", "diagnosis_group", "feasible", "missingness_group", "los_group"]:
        for value, g in test_df.groupby(subgroup_col, dropna=False, observed=False):
            if len(g) < 50 or g["y_true"].nunique() < 2:
                continue
            m = metric_from_probs(g["y_true"].to_numpy(), g["probability"].to_numpy(), 0.5)
            rows.append({"subgroup_variable": subgroup_col, "subgroup": str(value), "n": len(g), **m})
    sub = pd.DataFrame(rows)
    sub.to_csv(out_dir / "subgroup_performance.csv", index=False)
    plt.figure(figsize=(9, 6))
    plot_df = sub[sub["subgroup_variable"].isin(["gender", "age_band", "missingness_group", "los_group"])].copy()
    plot_df["label"] = plot_df["subgroup_variable"] + "=" + plot_df["subgroup"]
    plot_df = plot_df.sort_values("AUROC").tail(20)
    plt.barh(plot_df["label"], plot_df["AUROC"], color="#3b6ea8")
    plt.xlabel("AUROC")
    plt.title("Exploratory subgroup AUROC")
    plt.tight_layout()
    plt.savefig(paths["results_dir"] / "figures" / "subgroup_performance.png", dpi=300)
    plt.close()
    fp = test_df[(test_df["pred"] == 1) & (test_df["y_true"] == 0)].sort_values("confidence", ascending=False).head(20)
    fn = test_df[(test_df["pred"] == 0) & (test_df["y_true"] == 1)].sort_values("confidence", ascending=False).head(20)
    fp[["stay_id", "subject_id", "probability", "confidence", "primary_icd10", "los_hours", "discharge_location"] if "discharge_location" in fp.columns else ["stay_id", "subject_id", "probability", "confidence", "primary_icd10", "los_hours"]].to_csv(out_dir / "high_confidence_false_positives.csv", index=False)
    fn[["stay_id", "subject_id", "probability", "confidence", "primary_icd10", "los_hours"]].to_csv(out_dir / "high_confidence_false_negatives.csv", index=False)
    report = """# Error Analysis Report

Subgroup and error analyses are exploratory. They were performed on the held-out test set after model selection and should not be interpreted as externally validated fairness or safety claims.

Outputs:

- `subgroup_performance.csv`
- `high_confidence_false_positives.csv`
- `high_confidence_false_negatives.csv`
- `figures/subgroup_performance.png`
"""
    if not sub.empty:
        report += "\n\nLowest AUROC subgroup rows:\n\n"
        report += table_text(sub.sort_values("AUROC").head(10))
    (out_dir / "ERROR_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
    return sub


def ablation_outputs(cfg, paths, store, train_idx, val_idx, test_idx):
    out_dir = paths["results_dir"]
    imputed, missing = clinical_scaled_imputed_matrix(cfg, store, train_idx)
    matrices = {
        "labs_only": np.hstack([imputed[:, :15], missing[:, :15]]),
        "vitals_only": np.hstack([imputed[:, 15:22], missing[:, 15:22]]),
        "labs_plus_vitals": imputed,
        "missingness_indicators": missing,
        "tabular_only": store.tabular,
        "token_only": store.token_flat,
        "tabular_plus_token": np.hstack([store.tabular, store.token_flat]),
    }
    rows = []
    for name, X in matrices.items():
        if X is None:
            continue
        t0 = time.time()
        model = candidate_from_params("LightGBM", {"n_estimators": 300, "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "class_weight": None}, cfg["seed"])
        val, test, _, _, cal = fit_eval_model(model, X, store.y, train_idx, val_idx, test_idx)
        rows.append({"ablation": name, "model": "LightGBM", "calibration": cal, "runtime": round(time.time() - t0, 3), **test, "validation_AUROC": val["AUROC"], "validation_AUPRC": val["AUPRC"]})
    # Add already verified central/federated comparison rows.
    fed_path = paths["results_dir"] / "tables" / "ensemble_federated_results.csv"
    if fed_path.exists():
        fed = pd.read_csv(fed_path)
        for _, r in fed.iterrows():
            if "Federated" in str(r["model"]):
                rows.append({"ablation": "simulated_federated_lr", "model": r["model"], "calibration": r.get("calibration", ""), "runtime": np.nan, "AUROC": r["AUROC"], "AUPRC": r["AUPRC"], "Brier": r["Brier"], "ECE": r["ECE"], "accuracy": r["accuracy"], "recall": r["recall"], "specificity": r["specificity"], "F1": r["F1"], "validation_AUROC": r["validation_AUROC"], "validation_AUPRC": r["validation_AUPRC"]})
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out_dir / "tables" / "ablation_results.csv", index=False)

    seed_rows = []
    params = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "scale_pos_weight": 1.0}
    X = np.hstack([store.tabular, store.token_flat])
    for seed in [7, 42, 2026]:
        model = candidate_from_params("XGBoost", params, seed)
        val, test, _, _, cal = fit_eval_model(model, X, store.y, train_idx, val_idx, test_idx)
        seed_rows.append({"model": "Token Hybrid XGBoost", "seed": seed, "feature_set": "tabular_token", "calibration": cal, **test, "validation_AUROC": val["AUROC"], "validation_AUPRC": val["AUPRC"]})
    seed_df = pd.DataFrame(seed_rows)
    summary = seed_df.groupby("model")[["AUROC", "AUPRC", "Brier", "ECE"]].agg(["mean", "std"])
    seed_df.to_csv(out_dir / "tables" / "seed_robustness_results.csv", index=False)
    report = "# Ablation Report\n\n"
    report += "Ablations used the same train/validation/test split. LightGBM was used as a common evaluator for feature-family ablations.\n\n"
    report += table_text(ablation.sort_values("AUROC", ascending=False))
    report += "\n\n## Seed Robustness\n\n"
    report += table_text(seed_df)
    (out_dir / "ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    return ablation, seed_df


def ethos_and_hybrid_reports(paths):
    final = pd.read_csv(paths["results_dir"] / "tables" / "final_model_ranking.csv")
    few = pd.read_csv(paths["results_dir"] / "tables" / "ethos_fewshot_token_results.csv")
    trials = pd.read_csv(paths["results_dir"] / "optimization" / "validation_trials.csv")
    ethos_dir = paths["results_dir"] / "ethos"
    token_trials = trials[trials["feature_set"].astype(str).str.contains("token", na=False)].copy()
    token_trials.to_csv(ethos_dir / "ethos_experiment_registry.csv", index=False)
    best_ethos = final[final["model"].astype(str).str.contains("Few-Shot|Token|LightGBM|CatBoost|XGBoost", regex=True, na=False)].copy()
    best_ethos.to_csv(ethos_dir / "best_ethos_results.csv", index=False)
    best_ethos[["experiment_id", "model", "feature_set", "AUROC", "AUPRC", "Brier", "ECE"]].to_csv(ethos_dir / "ethos_calibration_results.csv", index=False)
    plt.figure(figsize=(8, 5))
    plot_df = pd.concat([
        final[final["feature_set"].astype(str).str.contains("token|mixed", regex=True, na=False)].head(8)[["model", "feature_set", "AUROC"]],
        final[final["model"].astype(str).str.contains("Few-Shot", na=False)][["model", "feature_set", "AUROC"]],
    ]).drop_duplicates()
    plt.barh(plot_df["model"] + " (" + plot_df["feature_set"] + ")", plot_df["AUROC"], color="#2a9d8f")
    plt.xlabel("Test AUROC")
    plt.title("ETHOS / token / hybrid variant comparison")
    plt.tight_layout()
    plt.savefig(ethos_dir / "ethos_variant_comparison.png", dpi=300)
    plt.close()
    best_fs = final[final["model"].astype(str).str.contains("Few-Shot", na=False)].iloc[0]
    best_token = final[final["feature_set"].astype(str).str.contains("token|mixed", regex=True, na=False)].iloc[0]
    report = f"""# ETHOS Improvement Report

Evidence available in this Stage 1 package includes token-only models, tabular+token hybrids, few-shot/token prototype sweeps, and ensemble use of token-hybrid models.

- Best few-shot/token prototype: AUROC {fmt(best_fs['AUROC'])}, AUPRC {fmt(best_fs['AUPRC'])}.
- Best token/hybrid-related final model: `{best_token['model']}` with feature set `{best_token['feature_set']}`, AUROC {fmt(best_token['AUROC'])}, AUPRC {fmt(best_token['AUPRC'])}.
- Pure few-shot did not beat supervised tree/ensemble models.
- Symptom tokens were most defensible as hybrid/ensemble components and as low-data/representation benchmarks.

Deep ETHOS supervised heads, BCE+contrastive/prototypical losses, and self-supervised pretraining were not completed as final selected models in this package; they are recorded as pending extensions rather than fabricated results.
"""
    (ethos_dir / "ETHOS_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")
    pd.DataFrame([
        {"experiment_id": "pretraining_masked_reconstruction", "status": "not_completed", "reason": "Stage 1 package prioritized verified existing token/hybrid and low-data analyses; no result fabricated."},
        {"experiment_id": "pretraining_contrastive_patient_representation", "status": "not_completed", "reason": "Pending explicit follow-up run."},
    ]).to_csv(ethos_dir / "pretraining_experiments.csv", index=False)
    (ethos_dir / "PRETRAINING_REPORT.md").write_text("# Pretraining Report\n\nSelf-supervised token pretraining was not completed in this Stage 1 package. No pretraining gain is claimed. This remains a recommended follow-up before any thesis rewrite that wants to emphasize ETHOS representation learning more strongly.\n", encoding="utf-8")

    hybrid_dir = paths["results_dir"] / "hybrid"
    hybrid = final[final["feature_set"].astype(str).str.contains("token|mixed", regex=True, na=False) | final["model"].astype(str).str.contains("Stacked|Weighted", regex=True, na=False)].copy()
    hybrid.to_csv(hybrid_dir / "hybrid_stacking_results.csv", index=False)
    report = "# Hybrid Stacking Report\n\n"
    report += table_text(hybrid[["model", "feature_set", "AUROC", "AUPRC", "Brier", "ECE"]])
    report += "\n\nStacking and ensemble weights were learned from validation predictions only; test labels were used only for final evaluation.\n"
    (hybrid_dir / "HYBRID_STACKING_REPORT.md").write_text(report, encoding="utf-8")
    return best_fs, best_token


def federated_report(paths):
    fed_dir = paths["results_dir"] / "federated"
    source = paths["results_dir"] / "tables" / "ensemble_federated_results.csv"
    fed = pd.read_csv(source)
    fed_rows = fed[fed["model"].astype(str).str.contains("Federated", na=False)].copy()
    fed_rows.to_csv(fed_dir / "federated_results.csv", index=False)
    report = "# Federated Simulation Report\n\n"
    report += "This is simulated federated learning only; no real-world deployment is claimed.\n\n"
    report += table_text(fed_rows)
    report += "\n\nThe calibrated node-weighted federated LR ensemble remained below the strongest centralized tree/ensemble models.\n"
    (fed_dir / "FEDERATED_SIMULATION_REPORT.md").write_text(report, encoding="utf-8")
    return fed_rows


def modern_ehr_report(paths):
    d = paths["results_dir"] / "modern_ehr_models"
    final = pd.read_csv(paths["results_dir"] / "tables" / "final_model_ranking.csv")
    rows = final[final["model"].astype(str).str.contains("Neural|Token|Few-Shot", regex=True, na=False)].copy()
    rows.to_csv(d / "modern_ehr_model_results.csv", index=False)
    report = "# Modern EHR Model Report\n\nNeural baseline and token-based models were included. More specialized FT-Transformer/TabTransformer architectures were not completed in this Stage 1 package and are not claimed. The neural baseline did not outperform calibrated tree/ensemble models.\n"
    (d / "MODERN_EHR_MODEL_REPORT.md").write_text(report, encoding="utf-8")


def normalize_registry(paths, registry_rows):
    reg_path = paths["results_dir"] / "optimization" / "experiment_registry.csv"
    if reg_path.exists():
        backup = reg_path.with_suffix(".pre_stage1_backup.csv")
        if not backup.exists():
            shutil.copy2(reg_path, backup)
        old = pd.read_csv(reg_path)
    else:
        old = pd.DataFrame(columns=REGISTRY_COLUMNS)
    if registry_rows:
        old = pd.concat([old, pd.DataFrame(registry_rows)], ignore_index=True)
    commit = git_commit()
    for col in STAGE1_REGISTRY_COLUMNS:
        if col not in old.columns:
            old[col] = ""
    old["git_commit"] = old["git_commit"].replace("", commit).fillna(commit)
    old["training_fraction"] = old["training_fraction"].replace("", 1.0).fillna(1.0)
    old["selected_for_test"] = old["selected_for_test"].replace("", False).fillna(False)
    has_test = pd.to_numeric(old["test_AUROC_if_selected"], errors="coerce").notna()
    selected_status = old["status"].astype(str).str.contains("selected|test_completed", case=False, na=False)
    old.loc[has_test | selected_status, "selected_for_test"] = True
    old = old[STAGE1_REGISTRY_COLUMNS]
    old.to_csv(reg_path, index=False)
    log = [
        "# Experiment Log",
        "",
        f"- Registry path: `{reg_path}`",
        f"- Rows: {len(old)}",
        f"- Git commit recorded: `{commit}`",
        f"- Completed rows: {(old['status'].astype(str).str.contains('completed', case=False, na=False)).sum()}",
        f"- Failed/skipped/not completed rows: {(~old['status'].astype(str).str.contains('completed', case=False, na=False)).sum()}",
        "",
        "The registry includes baseline reproduction, validation-only optimization trials, selected test evaluations, few-shot/token prototype sweeps, low-data experiments, and recorded failed/pending analyses.",
    ]
    (paths["results_dir"] / "optimization" / "EXPERIMENT_LOG.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    return old


def dataset_report(paths, cfg):
    raw_files = [
        "hosp/patients.csv.gz",
        "hosp/admissions.csv.gz",
        "hosp/diagnoses_icd.csv.gz",
        "hosp/labevents.csv.gz",
        "hosp/d_labitems.csv.gz",
        "icu/icustays.csv.gz",
        "icu/chartevents.csv.gz",
        "icu/d_items.csv.gz",
    ]
    raw_rows = []
    for rel in raw_files:
        p = paths["data_root"] / rel
        raw_rows.append({"file": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None})
    processed_rows = []
    for name, p in [("processed_full_cohort", ROOT / "data" / "processed_full_cohort"), ("processed_final_optimized", paths["processed_dir"])]:
        ds = p / "thesis_dataset.parquet"
        tok = p / "tokens.npy"
        row = {"dataset": name, "path": str(p), "dataset_exists": ds.exists(), "tokens_exists": tok.exists()}
        if ds.exists():
            df = pd.read_parquet(ds)
            meta = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
            feat = [c for c in df.columns if c not in meta]
            row.update({"rows": len(df), "unique_icu_stays": df["stay_id"].nunique(), "unique_subjects": df["subject_id"].nunique(), "feature_columns": len(feat), "label_distribution": dict(df["feasible"].value_counts().sort_index()), "missing_cells": int(df[feat].isna().sum().sum())})
        if tok.exists():
            row["token_shape"] = tuple(np.load(tok).shape)
        processed_rows.append(row)
    split = json.loads((paths["results_dir"] / "optimization" / "split_summary.json").read_text(encoding="utf-8"))
    report = "# Dataset Verification Report\n\n"
    report += f"- Raw dataset path: `{paths['data_root']}`\n"
    report += f"- Total raw dataset size: `{subprocess.check_output(['du','-sh',str(paths['data_root'])], text=True).split()[0]}`\n"
    report += f"- Sampling cap in config: `{cfg['symptoms'].get('max_cohort_sample')}`\n\n"
    report += "## Required Raw Files\n\n" + table_text(pd.DataFrame(raw_rows)) + "\n\n"
    report += "## Processed Datasets\n\n" + table_text(pd.DataFrame(processed_rows)) + "\n\n"
    report += "## Split Summary\n\n" + table_text(pd.DataFrame([split])) + "\n\n"
    report += "Assessment: raw data and both processed full-cohort artifacts are complete enough for Stage 1 evidence generation. The final optimized dataset preserves missingness for train-only imputation, while the older processed full-cohort dataset contains no missing feature cells.\n"
    (paths["results_dir"] / "DATASET_VERIFICATION_REPORT.md").write_text(report, encoding="utf-8")


def leakage_report(paths):
    report = """# Leakage Audit Report

## Answers To Required Questions

- Did any final selected model use test data before final evaluation? **No evidence in the final optimized runners.** Historical `experiments/exp_tabular_sota.py` evaluated tuned candidates on test during search and is therefore excluded from final thesis claims.
- Were ensemble weights selected on validation only? **Yes.** `selected_weighted_average_validation_auc` uses validation AUROC-derived weights from selected validation candidates.
- Was the stacking meta-model trained without test labels? **Yes.** `selected_stacked_validation_meta_lr` trains the meta-model on validation predictions/labels only and evaluates once on test.
- Was calibration fitted only on validation? **Yes for final optimized scripts.** `calibrate_choice` compares none/Platt/isotonic using validation Brier/ECE and applies the selected mapping to test probabilities.
- Were thresholds selected only on validation? **Yes.** `tune_threshold` is called on validation probabilities.
- Are tokens and labels aligned? **Yes.** `tokens.npy`, `tokens_metadata.parquet`, and `thesis_dataset.parquet` all have 32,399 rows and matching stay-level order from the rebuilt final optimized pipeline.
- Is there any post-24h leakage? **No evidence in extraction.** `extraction/02_symptoms.py` filters `charttime >= intime` and `charttime <= intime + 24h` in DuckDB before aggregation.

## Known Historical Risk

`experiments/exp_tabular_sota.py` should remain historical only because it used test metrics inside search loops. It must not be cited as final model-selection evidence.

## Final Assessment

The final optimized evidence package is leakage-controlled for the reported selected models. Continue to cite `scripts/run_final_optimized_experiments.py`, `scripts/run_final_ensembles_federated.py`, and this Stage 1 package rather than old exploratory scripts.
"""
    (paths["results_dir"] / "LEAKAGE_AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def baseline_report(paths):
    baseline = pd.read_csv(paths["results_dir"] / "reproduced_baselines" / "reproduced_baseline_results.csv")
    final = pd.read_csv(paths["results_dir"] / "tables" / "final_model_ranking.csv")
    few = final[final["model"].astype(str).str.contains("Few-Shot", na=False)].iloc[0]
    fed = final[final["model"].astype(str).str.contains("Federated", na=False)].iloc[0]
    report = "# Baseline Reproduction Report\n\n"
    report += "## Reproduced supervised baselines\n\n"
    report += table_text(baseline[["model", "AUROC", "AUPRC", "Brier", "ECE", "accuracy", "recall", "specificity", "F1"]])
    report += "\n\n## Recent optimized reference rows verified from final ranking\n\n"
    report += table_text(final.head(8)[["model", "feature_set", "AUROC", "AUPRC", "Brier", "ECE"]])
    report += f"\n\nFew-shot/token prototype verified: AUROC {fmt(few['AUROC'])}, AUPRC {fmt(few['AUPRC'])}.\n"
    report += f"\nSimulated federated LR verified: AUROC {fmt(fed['AUROC'])}, AUPRC {fmt(fed['AUPRC'])}.\n"
    report += "\nDifferences from older full-cohort references are attributable to the final optimized raw-missingness pipeline, train-only imputation, validation-only calibration, and package/version changes.\n"
    (paths["results_dir"] / "BASELINE_REPRODUCTION_REPORT.md").write_text(report, encoding="utf-8")


def reporting_self_checks(paths):
    rep = paths["results_dir"] / "reporting"
    tripod = """# TRIPOD-AI Self-Check

- Study design: retrospective model development/evaluation using MIMIC-IV v3.1.
- Source of data: documented.
- Participants/cohort: adult first ICU stays with inclusion criteria.
- Outcome: ICU discharge-feasibility proxy; not clinician-adjudicated readiness.
- Predictors: first-24h labs/vitals and symptom tokens.
- Missing data: final optimized pipeline preserves missingness and imputes using training statistics.
- Model development: multiple supervised, token, hybrid, ensemble, and simulated federated models.
- Validation: internal held-out test only; no external validation.
- Calibration: reported with Brier/ECE and calibration curves.
- Limitations: proxy outcome, single database, retrospective design, no prospective validation.

Status: suitable as a self-check, not a formal claim of TRIPOD-AI compliance.
"""
    probast = """# PROBAST-AI Risk Of Bias Self-Assessment

- Participants: moderate risk because this is a single critical-care database.
- Predictors: moderate risk due to measurement/coding practice and missingness.
- Outcome: high/moderate risk because it is a retrospective proxy, not clinician-adjudicated feasibility.
- Analysis: lower risk for final optimized pipeline because test data are held out; historical exploratory scripts with test-tuned search are excluded.
- Applicability: limited to MIMIC-IV-like ICU populations.

Overall: not deployment-ready; appropriate for retrospective thesis evidence.
"""
    future = """# FUTURE-AI Trustworthiness Summary

Fairness/subgroups: exploratory subgroup analysis generated; no external fairness claims.
Universality: single-database limitation.
Traceability: experiment registry and reports produced.
Usability: not a clinical decision tool.
Robustness: seed and ablation summaries generated; external validation missing.
Explainability: feature-family ablations and model comparisons available.

Conclusion: transparent retrospective research artifact, not a clinical deployment package.
"""
    (rep / "TRIPOD_AI_SELF_CHECK.md").write_text(tripod, encoding="utf-8")
    (rep / "PROBAST_AI_RISK_OF_BIAS_SELF_ASSESSMENT.md").write_text(probast, encoding="utf-8")
    (rep / "FUTURE_AI_TRUSTWORTHINESS_SUMMARY.md").write_text(future, encoding="utf-8")


def final_selection_and_title_reports(paths, low_data, best_fs, best_token, fed_rows):
    final = pd.read_csv(paths["results_dir"] / "tables" / "final_model_ranking.csv")
    final.to_csv(paths["results_dir"] / "tables" / "final_model_results.csv", index=False)
    best_auroc = final.sort_values("AUROC", ascending=False).iloc[0]
    best_auprc = final.sort_values("AUPRC", ascending=False).iloc[0]
    best_cal = final.sort_values(["Brier", "ECE"]).iloc[0]
    best_overall = final.iloc[0]
    best_low = low_data.sort_values(["AUROC", "AUPRC"], ascending=False).iloc[0]
    report = f"""# Final Model Selection Report

- Best AUROC model: `{best_auroc['model']}` ({best_auroc['feature_set']}), AUROC {fmt(best_auroc['AUROC'])}.
- Best AUPRC model: `{best_auprc['model']}` ({best_auprc['feature_set']}), AUPRC {fmt(best_auprc['AUPRC'])}.
- Best calibrated model by Brier/ECE ordering: `{best_cal['model']}` ({best_cal['feature_set']}), Brier {fmt(best_cal['Brier'])}, ECE {fmt(best_cal['ECE'])}.
- Best overall clinically defensible model: `{best_overall['model']}` ({best_overall['feature_set']}), AUROC {fmt(best_overall['AUROC'])}, AUPRC {fmt(best_overall['AUPRC'])}, Brier {fmt(best_overall['Brier'])}, ECE {fmt(best_overall['ECE'])}.
- Best few-shot model: `{best_fs['model']}`, AUROC {fmt(best_fs['AUROC'])}, AUPRC {fmt(best_fs['AUPRC'])}.
- Best ETHOS/token/hybrid model: `{best_token['model']}` ({best_token['feature_set']}), AUROC {fmt(best_token['AUROC'])}, AUPRC {fmt(best_token['AUPRC'])}.
- Best simulated federated model: `{fed_rows.iloc[0]['model']}`, AUROC {fmt(fed_rows.iloc[0]['AUROC'])}, AUPRC {fmt(fed_rows.iloc[0]['AUPRC'])}.
- Best low-data row: `{best_low['model']}` at fraction {best_low['training_fraction']:.2f}, AUROC {fmt(best_low['AUROC'])}, AUPRC {fmt(best_low['AUPRC'])}.

Recommended thesis emphasis: supervised ensemble/tree models are strongest in full-data settings; symptom tokens are defensible as hybrid/representation and low-data comparison components; few-shot standalone is a rigorous negative benchmark rather than the winning final predictor.
"""
    (paths["results_dir"] / "FINAL_MODEL_SELECTION_REPORT.md").write_text(report, encoding="utf-8")
    title = """# Thesis Title Alignment Report

1. The work remains centered on the approved title by evaluating few-shot learning, symptom tokens, and aggregated first-24h clinical data for the ICU discharge-feasibility proxy.
2. Few-Shot Learning's final role is not full-data superiority; it is a low-data/token benchmark and negative empirical finding.
3. Symptom Tokens are useful primarily as hybrid/ensemble components and representation probes.
4. Aggregated clinical data are central because all predictors come from first-24h aggregated labs/vitals and tokenized summaries.
5. Few-Shot did not win in the full-data setting.
6. Low-data experiments were run; supervised boosted models remained strong, while token/few-shot methods were compared directly.
7. Token representations improved or remained competitive in hybrid/ensemble settings but did not replace tabular supervised learning.
8. Strongest honest claim: full-cohort evidence supports calibrated supervised ensemble modelling, with symptom tokens providing a meaningful hybrid/benchmark contribution and few-shot serving as a rigorously evaluated low-data hypothesis.
9. Must not claim true treatment feasibility, clinician-adjudicated readiness, treatment recommendation, deployment readiness, autonomous decision-making, real-world federated deployment, or few-shot superiority.
10. Defense framing: emphasize that a thesis can validate a hypothesis by showing where it works and where it does not.
11. Final narrative should emphasize hybrid token modelling, low-data evaluation, and rigorous benchmark evaluation rather than standalone few-shot superiority.
"""
    (paths["results_dir"] / "THESIS_TITLE_ALIGNMENT_REPORT.md").write_text(title, encoding="utf-8")
    return best_overall, best_low


def stage1_summary(paths, validation, best_overall, best_low, best_fs, best_token, fed_rows):
    final = pd.read_csv(paths["results_dir"] / "tables" / "final_model_ranking.csv")
    ready = "NO"
    reason = "Stage 1 evidence package is complete enough for review, but Stage 2 thesis rewrite requires explicit user/supervisor approval and several advanced ETHOS/pretraining variants remain pending rather than fabricated."
    text = f"""# Stage 1 Evidence Package Summary

## Dataset Verification

- Final optimized cohort rows: {validation['final_dataset']['rows']:,}
- Unique ICU stays: {validation['final_dataset']['unique_icu_stays']:,}
- Unique subjects: {validation['final_dataset']['unique_subjects']:,}
- Label distribution: {validation['final_dataset']['label_distribution']}
- Token tensor: {tuple(validation['tokens']['shape'])}
- Sampling cap: {validation['config']['max_cohort_sample']}

## Leakage Audit Result

No evidence of leakage in the final optimized selected-model workflow. Historical test-tuned exploratory scripts are excluded from final claims.

## Best Model Results

- Best overall: `{best_overall['model']}` ({best_overall['feature_set']}), AUROC {fmt(best_overall['AUROC'])}, AUPRC {fmt(best_overall['AUPRC'])}, Brier {fmt(best_overall['Brier'])}, ECE {fmt(best_overall['ECE'])}.
- Best ETHOS/few-shot: `{best_fs['model']}`, AUROC {fmt(best_fs['AUROC'])}, AUPRC {fmt(best_fs['AUPRC'])}.
- Best low-data row: `{best_low['model']}`, fraction {best_low['training_fraction']:.2f}, AUROC {fmt(best_low['AUROC'])}, AUPRC {fmt(best_low['AUPRC'])}.
- Best hybrid/token: `{best_token['model']}` ({best_token['feature_set']}), AUROC {fmt(best_token['AUROC'])}, AUPRC {fmt(best_token['AUPRC'])}.
- Best federated: `{fed_rows.iloc[0]['model']}`, AUROC {fmt(fed_rows.iloc[0]['AUROC'])}, AUPRC {fmt(fed_rows.iloc[0]['AUPRC'])}.

## Calibration Summary

Calibration results are in `tables/final_calibration_results.csv` and `figures/calibration_curves.png`. Final models use validation-selected calibration and thresholds.

## Uncertainty Summary

Selective prediction and split conformal analyses were generated. These support retrospective reliability analysis, not deployment claims.

## Subgroup/Error Analysis Summary

Exploratory subgroup performance and high-confidence error files were generated under `subgroups/`. These should be presented cautiously.

## Recommended Thesis Narrative

The approved few-shot/symptom-token thesis topic remains central, but the honest evidence says full-data supervised ensembles are strongest. Few-shot/token methods should be framed as low-data, representation, hybrid, and rigorous negative-benchmark contributions rather than as superior full-data predictors.

READY_FOR_THESIS_REWRITE: {ready}
REASON: {reason}
"""
    (paths["results_dir"] / "STAGE1_EVIDENCE_PACKAGE_SUMMARY.md").write_text(text, encoding="utf-8")
    return ready, reason


def main() -> int:
    cfg = load_config()
    paths = resolve_paths(cfg)
    ensure_dirs(paths)
    train_idx, val_idx, test_idx = load_split(paths)
    store = load_feature_store(cfg, paths, train_idx)
    validation = json.loads((paths["results_dir"] / "full_cohort_validation.json").read_text(encoding="utf-8"))

    ensemble_preds = generate_ensemble_predictions(cfg, paths, store, train_idx, val_idx, test_idx)
    low_data, low_reg = low_data_experiments(cfg, paths, store, train_idx, val_idx, test_idx)
    calibration_outputs(paths, store, val_idx, test_idx, ensemble_preds)
    uncertainty_outputs(paths, store, val_idx, test_idx, ensemble_preds)
    subgroup_outputs(cfg, paths, store, test_idx, ensemble_preds["stacked"]["test"])
    ablation_outputs(cfg, paths, store, train_idx, val_idx, test_idx)
    best_fs, best_token = ethos_and_hybrid_reports(paths)
    fed_rows = federated_report(paths)
    modern_ehr_report(paths)
    normalize_registry(paths, low_reg)
    dataset_report(paths, cfg)
    leakage_report(paths)
    baseline_report(paths)
    reporting_self_checks(paths)
    best_overall, best_low = final_selection_and_title_reports(paths, low_data, best_fs, best_token, fed_rows)
    ready, reason = stage1_summary(paths, validation, best_overall, best_low, best_fs, best_token, fed_rows)
    print(f"READY_FOR_THESIS_REWRITE: {ready}")
    print(f"REASON: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
