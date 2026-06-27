#!/usr/bin/env python3
"""Run final stacking/weighted-average and simulated federated comparisons."""
from __future__ import annotations

import json
import sys
import time
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_final_optimized_experiments import (  # noqa: E402
    REGISTRY_COLUMNS,
    calibrate_choice,
    candidate_from_params,
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


def load_split(paths: dict[str, Path]):
    split = np.load(paths["results_dir"] / "optimization" / "split_indices_seed42.npz")
    return split["train_idx"], split["val_idx"], split["test_idx"]


def parse_params(value: str | dict) -> dict:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def append_registry(paths: dict[str, Path], rows: list[dict]) -> None:
    reg_path = paths["results_dir"] / "optimization" / "experiment_registry.csv"
    old = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)
    out = pd.concat([old, pd.DataFrame(rows, columns=REGISTRY_COLUMNS)], ignore_index=True)
    out.to_csv(reg_path, index=False)


def append_final_tables(paths: dict[str, Path], rows: list[dict]) -> None:
    comp_path = paths["tables_dir"] / "optimized_model_comparison.csv"
    rank_path = paths["tables_dir"] / "final_model_ranking.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    comp = pd.concat([comp, pd.DataFrame(rows)], ignore_index=True)
    comp = comp.drop_duplicates("experiment_id", keep="last").sort_values("AUROC", ascending=False)
    comp.to_csv(comp_path, index=False)
    rank = comp.copy()
    rank["overall_score"] = rank["AUROC"] + 0.15 * rank["AUPRC"] - 0.2 * rank["ECE"] - 0.1 * rank["Brier"]
    rank = rank.sort_values("overall_score", ascending=False)
    rank.to_csv(rank_path, index=False)


def run_ensembles(cfg: dict, paths: dict[str, Path], store, train_idx, val_idx, test_idx) -> tuple[list[dict], list[dict]]:
    selected = pd.read_csv(paths["results_dir"] / "optimization" / "selected_candidates_validation_only.csv")
    selected = selected.sort_values("selection_score", ascending=False).drop_duplicates(["model", "feature_set"]).head(5)
    val_cols, test_cols, names = [], [], []
    registry = []
    t0 = time.time()
    for _, row in selected.iterrows():
        try:
            model_name = row["model"]
            fset = row["feature_set"]
            params = parse_params(row["params"])
            X = feature_matrix(store, fset)
            model = candidate_from_params(model_name, params, cfg["seed"])
            model.fit(X[train_idx], store.y[train_idx])
            val_cols.append(predict_positive(model, X[val_idx]))
            test_cols.append(predict_positive(model, X[test_idx]))
            names.append(f"{model_name}:{fset}")
        except Exception as exc:
            registry.append(registry_row(
                "ensemble_base_failed",
                paths["processed_dir"] / "thesis_dataset.parquet",
                str(row.get("model", "")),
                str(row.get("feature_set", "")),
                "flat_25x4" if row.get("feature_set") != "tabular" else "none",
                cfg["seed"],
                cfg["seed"],
                {},
                time.time() - t0,
                "failed",
                failure_reason=str(exc),
            ))
    if len(val_cols) < 2:
        return [], registry

    Z_val = np.column_stack(val_cols)
    Z_test = np.column_stack(test_cols)
    rows = []

    meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=cfg["seed"])
    meta.fit(Z_val, store.y[val_idx])
    val_raw = predict_positive(meta, Z_val)
    test_raw = predict_positive(meta, Z_test)
    cal, val_probs, test_probs = calibrate_choice(store.y[val_idx], val_raw, test_raw)
    threshold = tune_threshold(store.y[val_idx], val_probs)
    val = metric_row(store.y[val_idx], val_probs, threshold)
    test = metric_row(store.y[test_idx], test_probs, threshold)
    rows.append({
        "experiment_id": "selected_stacked_validation_meta_lr",
        "model": "Stacked Validation Meta-LR",
        "feature_set": "mixed",
        "token_variant": "mixed",
        "calibration": cal,
        "hyperparameters": jsonable({"base_models": names}),
        **test,
        "validation_AUROC": val["AUROC"],
        "validation_AUPRC": val["AUPRC"],
        "validation_Brier": val["Brier"],
        "validation_ECE": val["ECE"],
    })
    registry.append(registry_row(
        "selected_stacked_validation_meta_lr",
        paths["processed_dir"] / "thesis_dataset.parquet",
        "Stacked Validation Meta-LR",
        "mixed",
        "mixed",
        cfg["seed"],
        cfg["seed"],
        {"base_models": names},
        time.time() - t0,
        "selected_test_completed",
        notes=f"Validation meta-model; calibration={cal}",
        val=val,
        test=test,
    ))

    weights = np.asarray(selected["validation_AUROC"].head(len(test_cols)), dtype=float)
    weights = weights / weights.sum()
    val_raw = np.average(Z_val, axis=1, weights=weights)
    test_raw = np.average(Z_test, axis=1, weights=weights)
    cal, val_probs, test_probs = calibrate_choice(store.y[val_idx], val_raw, test_raw)
    threshold = tune_threshold(store.y[val_idx], val_probs)
    val = metric_row(store.y[val_idx], val_probs, threshold)
    test = metric_row(store.y[test_idx], test_probs, threshold)
    rows.append({
        "experiment_id": "selected_weighted_average_validation_auc",
        "model": "Weighted Average Ensemble",
        "feature_set": "mixed",
        "token_variant": "mixed",
        "calibration": cal,
        "hyperparameters": jsonable({"base_models": names, "weights": weights.tolist()}),
        **test,
        "validation_AUROC": val["AUROC"],
        "validation_AUPRC": val["AUPRC"],
        "validation_Brier": val["Brier"],
        "validation_ECE": val["ECE"],
    })
    registry.append(registry_row(
        "selected_weighted_average_validation_auc",
        paths["processed_dir"] / "thesis_dataset.parquet",
        "Weighted Average Ensemble",
        "mixed",
        "mixed",
        cfg["seed"],
        cfg["seed"],
        {"base_models": names, "weights": weights.tolist()},
        time.time() - t0,
        "selected_test_completed",
        notes=f"Validation-AUROC weighted probabilities; calibration={cal}",
        val=val,
        test=test,
    ))
    return rows, registry


def run_federated(cfg: dict, paths: dict[str, Path], store, train_idx, val_idx, test_idx) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    registry = []
    data = store.data
    train_stays = set(data.iloc[train_idx]["stay_id"].tolist())
    node_files = sorted(paths["processed_dir"].glob("node_*.parquet"))
    X = feature_matrix(store, "tabular")
    val_probs_list, test_probs_list, weights, nodes = [], [], [], []
    for node_file in node_files:
        node = pd.read_parquet(node_file, columns=["stay_id", "feasible"])
        node_train_stays = set(node[node["stay_id"].isin(train_stays)]["stay_id"].tolist())
        idx = np.asarray([i for i in train_idx if data.iloc[i]["stay_id"] in node_train_stays])
        if len(idx) < 50 or len(np.unique(store.y[idx])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=cfg["seed"]),
        )
        model.fit(X[idx], store.y[idx])
        val_probs_list.append(predict_positive(model, X[val_idx]))
        test_probs_list.append(predict_positive(model, X[test_idx]))
        weights.append(len(idx))
        nodes.append(node_file.stem)
    if not val_probs_list:
        registry.append(registry_row(
            "selected_simulated_federated_lr_weighted_nodes",
            paths["processed_dir"] / "thesis_dataset.parquet",
            "Simulated Federated LR Weighted Nodes",
            "tabular",
            "none",
            cfg["seed"],
            cfg["seed"],
            {},
            time.time() - t0,
            "skipped",
            failure_reason="No trainable nodes",
        ))
        return [], registry
    weights_arr = np.asarray(weights, dtype=float)
    val_raw = np.average(np.vstack(val_probs_list), axis=0, weights=weights_arr)
    test_raw = np.average(np.vstack(test_probs_list), axis=0, weights=weights_arr)
    cal, val_probs, test_probs = calibrate_choice(store.y[val_idx], val_raw, test_raw)
    threshold = tune_threshold(store.y[val_idx], val_probs)
    val = metric_row(store.y[val_idx], val_probs, threshold)
    test = metric_row(store.y[test_idx], test_probs, threshold)
    row = {
        "experiment_id": "selected_simulated_federated_lr_weighted_nodes",
        "model": "Simulated Federated LR Weighted Nodes",
        "feature_set": "tabular",
        "token_variant": "none",
        "calibration": cal,
        "hyperparameters": jsonable({"nodes": nodes, "node_train_rows": weights}),
        **test,
        "validation_AUROC": val["AUROC"],
        "validation_AUPRC": val["AUPRC"],
        "validation_Brier": val["Brier"],
        "validation_ECE": val["ECE"],
    }
    registry.append(registry_row(
        "selected_simulated_federated_lr_weighted_nodes",
        paths["processed_dir"] / "thesis_dataset.parquet",
        "Simulated Federated LR Weighted Nodes",
        "tabular",
        "none",
        cfg["seed"],
        cfg["seed"],
        {"nodes": nodes, "node_train_rows": weights},
        time.time() - t0,
        "selected_test_completed",
        notes=f"Node-weighted simulated federated LR ensemble; calibration={cal}",
        val=val,
        test=test,
    ))
    return [row], registry


def main() -> int:
    cfg = load_config()
    paths = resolve_paths(cfg)
    train_idx, val_idx, test_idx = load_split(paths)
    store = load_feature_store(cfg, paths, train_idx)
    rows_ens, reg_ens = run_ensembles(cfg, paths, store, train_idx, val_idx, test_idx)
    rows_fed, reg_fed = run_federated(cfg, paths, store, train_idx, val_idx, test_idx)
    rows = rows_ens + rows_fed
    append_final_tables(paths, rows)
    append_registry(paths, reg_ens + reg_fed)
    pd.DataFrame(rows).to_csv(paths["tables_dir"] / "ensemble_federated_results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
