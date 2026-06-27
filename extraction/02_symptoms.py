#!/usr/bin/env python3
"""
Extraction Step 2: Extract labs and vitals from first 24h of ICU.
Uses DuckDB to filter to eligible stays, restrict to configured itemids and the
first 24 hours, and aggregate before materializing in pandas.
Output: <paths.processed_dir>/symptoms.parquet with raw clipped values.
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def load_cohort(cfg):
    proj_root = Path(__file__).parent.parent
    processed_dir = Path(cfg["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = proj_root / processed_dir
    cohort = pd.read_parquet(processed_dir / "cohort.parquet")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=24)
    n = cfg["symptoms"].get("max_cohort_sample")
    if n is not None:
        raise ValueError(
            "symptoms.max_cohort_sample must be null for the full-cohort revision; "
            "sampling caps are disabled."
        )
    return cohort[["subject_id", "hadm_id", "stay_id", "intime", "window_end"]]

def extract_labs_duckdb(cfg, cohort, hosp_dir):
    """Extract labs via DuckDB for fast filtering."""
    import duckdb
    lab_itemids = cfg["symptoms"]["lab_itemids"]
    lab_ranges = cfg["symptoms"]["lab_ranges"]
    itemid_str = ",".join(str(x) for x in lab_itemids)
    cohort_tbl = cohort[["hadm_id", "stay_id", "intime", "window_end"]].drop_duplicates()
    
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.register("cohort_tbl", cohort_tbl)
    q = f"""
    SELECT
        any_value(l.subject_id) AS subject_id,
        l.hadm_id,
        c.stay_id,
        l.itemid,
        median(CAST(l.valuenum AS DOUBLE)) AS valuenum,
        count(*) AS measurement_count
    FROM read_csv_auto('{hosp_dir}/labevents.csv.gz', header=true) AS l
    INNER JOIN cohort_tbl AS c
        ON l.hadm_id = c.hadm_id
    WHERE l.itemid IN ({itemid_str})
      AND l.valuenum IS NOT NULL
      AND CAST(l.charttime AS TIMESTAMP) >= c.intime
      AND CAST(l.charttime AS TIMESTAMP) <= c.window_end
    GROUP BY l.hadm_id, c.stay_id, l.itemid
    """
    try:
        labs = con.execute(q).fetchdf()
    except Exception as e:
        raise e
    con.close()
    
    if len(labs) == 0:
        return pd.DataFrame()
    
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.dropna(subset=["valuenum"])
    
    for itemid, (lo, hi) in lab_ranges.items():
        if itemid in lab_itemids:
            mask = labs["itemid"] == itemid
            labs.loc[mask, "valuenum"] = labs.loc[mask, "valuenum"].clip(lo, hi)
    
    return labs[["subject_id", "hadm_id", "stay_id", "itemid", "valuenum", "measurement_count"]]

def extract_vitals_duckdb(cfg, cohort, icu_dir):
    """Extract vitals via DuckDB."""
    import duckdb
    vital_itemids = cfg["symptoms"]["vital_itemids"]
    vital_ranges = cfg["symptoms"]["vital_ranges"]
    itemid_str = ",".join(str(x) for x in vital_itemids)
    cohort_tbl = cohort[["stay_id", "intime", "window_end"]].drop_duplicates()
    
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.register("cohort_tbl", cohort_tbl)
    q = f"""
    SELECT
        any_value(v.subject_id) AS subject_id,
        any_value(v.hadm_id) AS hadm_id,
        v.stay_id,
        v.itemid,
        median(CAST(v.valuenum AS DOUBLE)) AS valuenum,
        count(*) AS measurement_count
    FROM read_csv_auto('{icu_dir}/chartevents.csv.gz', header=true) AS v
    INNER JOIN cohort_tbl AS c
        ON v.stay_id = c.stay_id
    WHERE v.itemid IN ({itemid_str})
      AND v.valuenum IS NOT NULL
      AND CAST(v.charttime AS TIMESTAMP) >= c.intime
      AND CAST(v.charttime AS TIMESTAMP) <= c.window_end
    GROUP BY v.stay_id, v.itemid
    """
    try:
        vitals = con.execute(q).fetchdf()
    except Exception as e:
        raise e
    con.close()
    
    if len(vitals) == 0:
        return pd.DataFrame()
    
    vitals["valuenum"] = pd.to_numeric(vitals["valuenum"], errors="coerce")
    vitals = vitals.dropna(subset=["valuenum"])
    
    for itemid, (lo, hi) in vital_ranges.items():
        if itemid in vital_itemids:
            mask = vitals["itemid"] == itemid
            vitals.loc[mask, "valuenum"] = vitals.loc[mask, "valuenum"].clip(lo, hi)
    
    return vitals[["subject_id", "hadm_id", "stay_id", "itemid", "valuenum", "measurement_count"]]

def aggregate_to_patient(labs, vitals, cfg, cohort_stays=None):
    """Pivot one pre-aggregated value per (stay_id, itemid) to one row per stay."""
    lab_itemids = cfg["symptoms"]["lab_itemids"]
    vital_itemids = cfg["symptoms"]["vital_itemids"]
    itemid_to_idx = {i: idx for idx, i in enumerate(lab_itemids + vital_itemids)}
    
    combined = []
    if len(labs) > 0:
        lab_agg = labs.groupby(["stay_id", "itemid"])["valuenum"].median().reset_index()
        lab_agg["source"] = "lab"
        combined.append(lab_agg)
    if len(vitals) > 0:
        vit_agg = vitals.groupby(["stay_id", "itemid"])["valuenum"].median().reset_index()
        vit_agg["source"] = "vital"
        combined.append(vit_agg)
    
    if not combined:
        return pd.DataFrame(), pd.DataFrame()
    
    agg = pd.concat(combined, ignore_index=True)
    if cohort_stays is not None:
        stays = pd.Index(cohort_stays).drop_duplicates().to_numpy()
    else:
        stays = agg["stay_id"].unique()
    n_features = len(lab_itemids) + len(vital_itemids)
    X = np.full((len(stays), n_features), np.nan)
    stay_to_row = {s: i for i, s in enumerate(stays)}
    
    for _, row in agg.iterrows():
        idx = itemid_to_idx.get(row["itemid"])
        if idx is not None:
            if row["stay_id"] not in stay_to_row:
                continue
            r = stay_to_row[row["stay_id"]]
            if np.isnan(X[r, idx]):
                X[r, idx] = row["valuenum"]
            else:
                X[r, idx] = (X[r, idx] + row["valuenum"]) / 2

    itemids = lab_itemids + vital_itemids
    missingness = pd.DataFrame({
        "feature_index": list(range(n_features)),
        "itemid": itemids,
        "missing_count": np.isnan(X).sum(axis=0).astype(int),
        "missing_pct": np.isnan(X).mean(axis=0),
    })
    
    df = pd.DataFrame(X, index=stays)
    df.index.name = "stay_id"
    df = df.reset_index()
    return df, missingness

def main():
    cfg = load_config()
    paths = cfg["paths"]
    proj_root = Path(__file__).parent.parent
    paths = {k: str(proj_root / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in paths.items()}
    hosp_dir = Path(paths["hosp_dir"])
    icu_dir = Path(paths["icu_dir"])
    out_dir = Path(paths["processed_dir"])
    
    cohort = load_cohort(cfg)
    print(f"Cohort: {len(cohort)} stays")
    
    print("Extracting labs (DuckDB)...")
    labs = extract_labs_duckdb(cfg, cohort, hosp_dir)
    print(f"Lab measurements: {len(labs):,}")
    
    print("Extracting vitals (DuckDB)...")
    vitals = extract_vitals_duckdb(cfg, cohort, icu_dir)
    print(f"Vital measurements: {len(vitals):,}")
    
    symptoms, missingness = aggregate_to_patient(labs, vitals, cfg, cohort["stay_id"])
    if len(symptoms) == 0:
        raise RuntimeError("No symptoms extracted. Check cohort and itemids.")
    
    symptoms = symptoms.merge(
        cohort[["stay_id", "subject_id", "hadm_id"]].drop_duplicates(),
        on="stay_id"
    )
    
    out_path = out_dir / "symptoms.parquet"
    symptoms.to_parquet(out_path, index=False)
    missingness_path = out_dir / "missingness_summary.csv"
    missingness.to_csv(missingness_path, index=False)
    
    print("=" * 60)
    print("SYMPTOM EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Missingness: {missingness_path}")
    print(f"Stays with symptoms: {len(symptoms):,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
