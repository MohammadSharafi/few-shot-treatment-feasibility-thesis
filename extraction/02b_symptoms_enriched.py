#!/usr/bin/env python3
"""
Extraction Step 2b (enriched): rich per-stay features from the first 24h of ICU.

Motivation
----------
The original `02_symptoms.py` collapses every lab/vital into a single median over
the first 24h. That discards the *trajectory* and *acuity* signal that clinicians
actually use. This script keeps the same cohort, window, itemids and clinical
ranges, but for each signal computes a panel of statistics in a single DuckDB
pass, then adds demographics and comorbidity burden.

For each of the 22 signals (15 labs + 7 vitals) we compute:
    n, mean, min, max, std, slope (per-hour trend), first, last, range.
Plus a per-signal missingness indicator (1 if the signal was never measured).
Plus demographics (age, sex), admission context, and comorbidity counts.

Missing physiological values are left as NaN (imputed inside the model pipeline
to avoid train/test leakage); missingness indicators are 0/1.

Output: <processed_dir>/enriched_dataset.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
STATS = ["n", "mean", "min", "max", "std", "slope", "first", "last"]


def load_config() -> dict:
    with open(ROOT / os.environ.get("THESIS_CONFIG", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def build_cohort(cfg: dict) -> pd.DataFrame:
    processed = resolve(cfg, "processed_dir")
    cohort = pd.read_parquet(processed / "cohort.parquet")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=24)
    return cohort


def agg_sql(source_csv: str, join_key: str, itemids: list[int], ranges: dict) -> str:
    """Per-(stay_id, itemid) statistics within the 24h window, computed in DuckDB.

    Clinical outlier clipping is applied via a CASE ladder mirroring config ranges.
    slope is per-hour (epoch is seconds, so multiply by 3600).
    """
    itemid_str = ",".join(str(i) for i in itemids)
    clip_cases = []
    for itemid, (lo, hi) in ranges.items():
        if itemid in itemids:
            clip_cases.append(
                f"WHEN e.itemid = {itemid} THEN least(greatest(e.valuenum, {lo}), {hi})"
            )
    clip_expr = "CASE " + " ".join(clip_cases) + " ELSE e.valuenum END" if clip_cases else "e.valuenum"
    return f"""
    WITH ev AS (
        SELECT c.stay_id AS stay_id,
               e.itemid AS itemid,
               {clip_expr} AS v,
               epoch(e.charttime) AS t
        FROM read_csv_auto('{source_csv}', header=true) AS e
        INNER JOIN cohort_tbl AS c
            ON e.{join_key} = c.{join_key}
        WHERE e.itemid IN ({itemid_str})
          AND e.valuenum IS NOT NULL
          AND e.charttime >= c.intime
          AND e.charttime <= c.window_end
    )
    SELECT stay_id, itemid,
           count(*)                 AS n,
           avg(v)                   AS mean,
           min(v)                   AS min,
           max(v)                   AS max,
           coalesce(stddev_samp(v), 0.0) AS std,
           coalesce(regr_slope(v, t) * 3600.0, 0.0) AS slope,
           arg_min(v, t)            AS first,
           arg_max(v, t)            AS last
    FROM ev
    GROUP BY stay_id, itemid
    """


def pivot(agg: pd.DataFrame, itemids: list[int], stays: np.ndarray) -> pd.DataFrame:
    """Long (stay,itemid,stats) -> wide one row per stay with `<itemid>_<stat>` cols."""
    frames = {"stay_id": stays}
    stay_to_row = {s: i for i, s in enumerate(stays)}
    for itemid in itemids:
        for stat in STATS:
            frames[f"{itemid}_{stat}"] = np.full(len(stays), np.nan, dtype=np.float64)
    for _, r in agg.iterrows():
        row = stay_to_row.get(r["stay_id"])
        if row is None:
            continue
        it = int(r["itemid"])
        for stat in STATS:
            frames[f"{it}_{stat}"][row] = r[stat]
    out = pd.DataFrame(frames)
    # Missing-count columns become explicit missingness indicators + 0-fill counts.
    for itemid in itemids:
        ncol = f"{itemid}_n"
        out[f"{itemid}_missing"] = out[ncol].isna().astype(np.int8)
        out[ncol] = out[ncol].fillna(0.0)
    return out


def add_demographics(cfg: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    hosp = resolve(cfg, "hosp_dir")
    patients = pd.read_csv(hosp / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(
        hosp / "admissions.csv.gz",
        usecols=["hadm_id", "admission_type", "admission_location", "insurance", "marital_status", "race"],
    )
    diagnoses = pd.read_csv(hosp / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code"])

    demo = cohort[["stay_id", "subject_id", "hadm_id"]].merge(patients, on="subject_id", how="left")
    demo = demo.merge(admissions, on="hadm_id", how="left")

    demo["age"] = demo["anchor_age"].astype(float)
    demo["sex_male"] = (demo["gender"] == "M").astype(np.int8)

    # Comorbidity burden: number of coded diagnoses for the admission.
    ndx = diagnoses.groupby("hadm_id").size().rename("n_diagnoses")
    demo = demo.merge(ndx, on="hadm_id", how="left")
    demo["n_diagnoses"] = demo["n_diagnoses"].fillna(0).astype(float)

    # Emergency admission is a strong disposition signal; one-hot the common types.
    demo["adm_emergency"] = demo["admission_type"].astype(str).str.contains("EMER|URGENT", case=False, na=False).astype(np.int8)
    demo["adm_elective"] = demo["admission_type"].astype(str).str.contains("ELECTIVE|SURGICAL", case=False, na=False).astype(np.int8)
    demo["ins_medicare"] = (demo["insurance"] == "Medicare").astype(np.int8)
    demo["ins_medicaid"] = (demo["insurance"] == "Medicaid").astype(np.int8)
    demo["married"] = (demo["marital_status"] == "MARRIED").astype(np.int8)

    keep = [
        "stay_id", "age", "sex_male", "n_diagnoses",
        "adm_emergency", "adm_elective", "ins_medicare", "ins_medicaid", "married",
    ]
    return demo[keep]


def make_labels(cfg: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    lab = cfg["labels"]
    base = (
        (cohort["hospital_expire_flag"] == 0)
        & (cohort["discharge_location"].isin(lab["feasible_discharge_locations"]))
        & (cohort["los_hours"] < lab["max_icu_los_hours"])
    )
    out = cohort[["stay_id", "subject_id", "hadm_id", "los_hours", "primary_icd10"]].copy()
    out["feasible"] = base.astype(int)
    return out


def main() -> int:
    cfg = load_config()
    processed = resolve(cfg, "processed_dir")
    processed.mkdir(parents=True, exist_ok=True)

    lab_itemids = cfg["symptoms"]["lab_itemids"]
    vital_itemids = cfg["symptoms"]["vital_itemids"]
    all_itemids = lab_itemids + vital_itemids

    cohort = build_cohort(cfg)
    stays = cohort["stay_id"].drop_duplicates().to_numpy()
    print(f"Cohort stays: {len(stays):,}")

    con = duckdb.connect()
    con.execute("SET threads TO 4")
    cohort_tbl = cohort[["stay_id", "hadm_id", "intime", "window_end"]].drop_duplicates()
    con.register("cohort_tbl", cohort_tbl)

    print("Aggregating labs (single DuckDB pass)...")
    labs = con.execute(
        agg_sql(str(resolve(cfg, "hosp_dir") / "labevents.csv.gz"), "hadm_id", lab_itemids, cfg["symptoms"]["lab_ranges"])
    ).fetchdf()
    print(f"  lab (stay,itemid) groups: {len(labs):,}")

    print("Aggregating vitals (single DuckDB pass)...")
    vitals = con.execute(
        agg_sql(str(resolve(cfg, "icu_dir") / "chartevents.csv.gz"), "stay_id", vital_itemids, cfg["symptoms"]["vital_ranges"])
    ).fetchdf()
    print(f"  vital (stay,itemid) groups: {len(vitals):,}")
    con.close()

    agg = pd.concat([labs, vitals], ignore_index=True)
    wide = pivot(agg, all_itemids, stays)

    demo = add_demographics(cfg, cohort)
    labels = make_labels(cfg, cohort)

    dataset = labels.merge(wide, on="stay_id", how="left").merge(demo, on="stay_id", how="left")

    feat_cols = [c for c in dataset.columns if c not in
                 {"stay_id", "subject_id", "hadm_id", "los_hours", "primary_icd10", "feasible"}]
    out_path = processed / "enriched_dataset.parquet"
    dataset.to_parquet(out_path, index=False)

    print("=" * 60)
    print("ENRICHED EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Rows: {len(dataset):,}  Features: {len(feat_cols)}")
    print(f"Feasible=1: {(dataset['feasible']==1).sum():,} "
          f"({(dataset['feasible']==1).mean()*100:.1f}%)")
    print(f"Median age: {dataset['age'].median():.0f}  Male: {dataset['sex_male'].mean()*100:.0f}%")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
