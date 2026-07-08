#!/usr/bin/env python3
"""
Generate the cohort characteristics table ("Table 1") and an OASIS-style
severity-score baseline for the full-cohort thesis/paper.

These two items were flagged in docs/EXPERT_GAP_REVIEW_2026-06.md as P1-1 and P1-3.
They could not be auto-generated in the review sandbox (no parquet engine / no data),
so run this locally where MIMIC-IV-derived parquet files exist.

Usage:
    python scripts/generate_table1_and_severity.py

Outputs:
    results/full_cohort/tables/table1_characteristics.csv
    results/full_cohort/tables/severity_baseline.csv  (if score columns are available)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PARQUET = "data/processed_full_cohort/thesis_dataset.parquet"
OUT_DIR = "results/full_cohort/tables"
LABEL_CANDIDATES = ["feasible", "label", "y", "target"]


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(PARQUET)  # needs pyarrow or fastparquet
    label = find_col(df, LABEL_CANDIDATES)
    if label is None:
        raise SystemExit(f"No label column among {LABEL_CANDIDATES}; columns={list(df.columns)}")

    # ---- Table 1: cohort characteristics, overall and by label ----
    rows = []
    age = find_col(df, ["age", "anchor_age"])
    sex = find_col(df, ["gender", "sex"])

    def summarize(sub, name):
        r = {"group": name, "n": len(sub), "positive_rate": round(sub[label].mean(), 3)}
        if age:
            r["age_median"] = round(sub[age].median(), 1)
            r["age_iqr"] = f"{sub[age].quantile(0.25):.0f}-{sub[age].quantile(0.75):.0f}"
        if sex:
            vc = sub[sex].astype(str).value_counts(normalize=True)
            for k, v in vc.items():
                r[f"sex_{k}_pct"] = round(100 * v, 1)
        return r

    rows.append(summarize(df, "Overall"))
    rows.append(summarize(df[df[label] == 1], "Feasible (label=1)"))
    rows.append(summarize(df[df[label] == 0], "Infeasible (label=0)"))
    t1 = pd.DataFrame(rows)
    t1.to_csv(f"{OUT_DIR}/table1_characteristics.csv", index=False)
    print("Wrote", f"{OUT_DIR}/table1_characteristics.csv")
    print(t1.to_string(index=False))

    # ---- Severity-score baseline (if a precomputed score column exists) ----
    score = find_col(df, ["oasis", "sofa", "saps", "saps_ii", "apsiii", "sapsii"])
    if score is not None:
        auc = roc_auc_score(df[label], df[score])
        # Higher severity should predict INFEASIBLE; if AUC<0.5 the sign is flipped.
        auc = max(auc, 1 - auc)
        sev = pd.DataFrame([{ "baseline": score, "auroc_vs_label": round(auc, 3), "n": len(df)}])
        sev.to_csv(f"{OUT_DIR}/severity_baseline.csv", index=False)
        print("Wrote", f"{OUT_DIR}/severity_baseline.csv")
        print(sev.to_string(index=False))
    else:
        print("\n[severity] No precomputed severity score column found.")
        print("To add OASIS/SAPS-II as a baseline, compute the score from the first-24h")
        print("variables (HR, MAP, temp, RR, GCS, urine output, ventilation, age, etc.)")
        print("using a standard implementation, store it as a column (e.g. 'oasis'),")
        print("then re-run this script.")


if __name__ == "__main__":
    main()
