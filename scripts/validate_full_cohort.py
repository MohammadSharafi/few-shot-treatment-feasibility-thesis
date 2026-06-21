#!/usr/bin/env python3
"""Validate full-cohort extraction artifacts and write machine-readable summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for key, value in cfg["paths"].items():
        path = Path(value)
        out[key] = path if path.is_absolute() else ROOT / path
    return out


def raw_manifest_status(data_root: Path) -> dict:
    manifest = data_root / "SHA256SUMS.txt"
    status = {
        "data_root": str(data_root),
        "manifest_exists": manifest.exists(),
        "manifest_entries": 0,
        "files_present": 0,
        "files_missing": [],
        "required_files_present": {},
    }
    required = [
        "hosp/patients.csv.gz",
        "hosp/admissions.csv.gz",
        "hosp/diagnoses_icd.csv.gz",
        "hosp/labevents.csv.gz",
        "icu/icustays.csv.gz",
        "icu/chartevents.csv.gz",
    ]
    for rel in required:
        status["required_files_present"][rel] = (data_root / rel).exists()
    if not manifest.exists():
        return status
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        entries.append(parts[1])
    status["manifest_entries"] = len(entries)
    missing = [rel for rel in entries if not (data_root / rel).exists()]
    status["files_missing"] = missing
    status["files_present"] = len(entries) - len(missing)
    status["complete_by_manifest"] = len(missing) == 0 and len(entries) > 0
    return status


def summarize() -> dict:
    cfg = load_config()
    paths = resolve_paths(cfg)
    processed = paths["processed_dir"]
    results = paths["results_dir"]
    tables = paths["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(processed / "cohort.parquet")
    dataset = pd.read_parquet(processed / "thesis_dataset.parquet")
    symptoms = pd.read_parquet(processed / "symptoms.parquet")
    tokens = np.load(processed / "tokens.npy")
    metadata = pd.read_parquet(processed / "tokens_metadata.parquet")

    label_counts = dataset["feasible"].value_counts().sort_index()
    label_dist = pd.DataFrame(
        {
            "feasible": label_counts.index.astype(int),
            "count": label_counts.values.astype(int),
            "percent": (label_counts.values / len(dataset) * 100).round(3),
        }
    )
    label_dist.to_csv(tables / "full_cohort_label_distribution.csv", index=False)

    node_rows = []
    for node_file in sorted(processed.glob("node_*.parquet")):
        node = pd.read_parquet(node_file)
        counts = node["feasible"].value_counts().sort_index()
        node_rows.append(
            {
                "node": node_file.stem,
                "rows": int(len(node)),
                "unique_stays": int(node["stay_id"].nunique()),
                "feasible_0": int(counts.get(0, 0)),
                "feasible_1": int(counts.get(1, 0)),
            }
        )
    pd.DataFrame(node_rows).to_csv(tables / "full_cohort_node_distribution.csv", index=False)

    missing_path = processed / "missingness_summary.csv"
    if missing_path.exists():
        missingness = pd.read_csv(missing_path)
    else:
        meta_cols = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
        feature_cols = [c for c in dataset.columns if c not in meta_cols]
        missingness = pd.DataFrame(
            {
                "feature_index": range(len(feature_cols)),
                "missing_count": dataset[feature_cols].isna().sum().to_numpy(dtype=int),
                "missing_pct": dataset[feature_cols].isna().mean().to_numpy(dtype=float),
            }
        )
    missingness.to_csv(tables / "full_cohort_missingness.csv", index=False)

    feature_cols = [
        c
        for c in dataset.columns
        if c not in ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    ]
    summary = {
        "raw_data": raw_manifest_status(paths["data_root"]),
        "config": {
            "processed_dir": str(processed),
            "results_dir": str(results),
            "max_cohort_sample": cfg["symptoms"].get("max_cohort_sample"),
        },
        "cohort": {
            "rows": int(len(cohort)),
            "unique_icu_stays": int(cohort["stay_id"].nunique()),
            "unique_subjects": int(cohort["subject_id"].nunique()),
        },
        "symptoms": {
            "rows": int(len(symptoms)),
            "unique_icu_stays": int(symptoms["stay_id"].nunique()),
        },
        "final_dataset": {
            "rows": int(len(dataset)),
            "columns": int(dataset.shape[1]),
            "feature_columns": int(len(feature_cols)),
            "unique_icu_stays": int(dataset["stay_id"].nunique()),
            "unique_subjects": int(dataset["subject_id"].nunique()),
            "label_distribution": {str(int(k)): int(v) for k, v in label_counts.items()},
            "post_imputation_missing_cells": int(dataset[feature_cols].isna().sum().sum()),
        },
        "missingness": {
            "mean_pre_imputation_missing_pct": float(missingness["missing_pct"].mean()),
            "max_pre_imputation_missing_pct": float(missingness["missing_pct"].max()),
        },
        "tokens": {
            "shape": [int(x) for x in tokens.shape],
            "metadata_rows": int(len(metadata)),
            "metadata_unique_icu_stays": int(metadata["stay_id"].nunique()),
        },
    }
    summary["all_eligible_data_used"] = (
        summary["config"]["max_cohort_sample"] is None
        and summary["cohort"]["rows"] == summary["final_dataset"]["rows"]
        and summary["cohort"]["unique_icu_stays"] == summary["final_dataset"]["unique_icu_stays"]
        and summary["tokens"]["shape"][0] == summary["final_dataset"]["rows"]
    )

    with open(results / "full_cohort_validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Full Cohort Validation",
        "",
        f"- Raw manifest complete by presence: {summary['raw_data'].get('complete_by_manifest', False)}",
        f"- Sampling cap: {summary['config']['max_cohort_sample']}",
        f"- Cohort rows: {summary['cohort']['rows']:,}",
        f"- Final dataset rows: {summary['final_dataset']['rows']:,}",
        f"- Unique ICU stays: {summary['final_dataset']['unique_icu_stays']:,}",
        f"- Unique subjects: {summary['final_dataset']['unique_subjects']:,}",
        f"- Label distribution: {summary['final_dataset']['label_distribution']}",
        f"- Mean pre-imputation missingness: {summary['missingness']['mean_pre_imputation_missing_pct']:.3%}",
        f"- Token tensor shape: {tuple(summary['tokens']['shape'])}",
        f"- All eligible data used: {summary['all_eligible_data_used']}",
    ]
    (results / "full_cohort_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summarize()
