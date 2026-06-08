#!/usr/bin/env python3
"""
Extraction Step 2: Extract labs and vitals from first 24h of ICU
Uses DuckDB for fast filtering of large CSVs. Applies clinical ranges, normalizes to [0,1].
Output: data/processed/symptoms.parquet
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def load_cohort(cfg):
    cohort = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "cohort.parquet")
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=24)
    n = cfg["symptoms"].get("max_cohort_sample")
    if n is not None:
        cohort = cohort.sample(n=min(n, len(cohort)), random_state=cfg["seed"])
    return cohort[["subject_id", "hadm_id", "stay_id", "intime", "window_end"]]

def extract_labs_duckdb(cfg, cohort, hosp_dir):
    """Extract labs via DuckDB for fast filtering."""
    import duckdb
    lab_itemids = cfg["symptoms"]["lab_itemids"]
    lab_ranges = cfg["symptoms"]["lab_ranges"]
    hadm_list = cohort["hadm_id"].unique().tolist()
    hadm_str = ",".join(str(x) for x in hadm_list[:50000])
    itemid_str = ",".join(str(x) for x in lab_itemids)
    
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    q = f"""
    SELECT subject_id, hadm_id, charttime, itemid, valuenum
    FROM read_csv_auto('{hosp_dir}/labevents.csv.gz', header=true)
    WHERE hadm_id IN ({hadm_str})
    AND itemid IN ({itemid_str})
    AND valuenum IS NOT NULL
    """
    try:
        labs = con.execute(q).fetchdf()
    except Exception as e:
        if "50000" in str(e) or "limit" in str(e).lower():
            labs = pd.DataFrame()
            for i in range(0, len(hadm_list), 5000):
                batch = hadm_list[i:i+5000]
                hadm_str_b = ",".join(str(x) for x in batch)
                qb = f"""
                SELECT subject_id, hadm_id, charttime, itemid, valuenum
                FROM read_csv_auto('{hosp_dir}/labevents.csv.gz', header=true)
                WHERE hadm_id IN ({hadm_str_b})
                AND itemid IN ({itemid_str})
                AND valuenum IS NOT NULL
                """
                lb = con.execute(qb).fetchdf()
                labs = pd.concat([labs, lb], ignore_index=True) if len(labs) > 0 else lb
        else:
            raise e
    con.close()
    
    if len(labs) == 0:
        return pd.DataFrame()
    
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.dropna(subset=["valuenum"])
    
    cohort_lookup = cohort.set_index("hadm_id")
    labs["intime"] = labs["hadm_id"].map(lambda h: cohort_lookup.loc[h, "intime"] if h in cohort_lookup.index else pd.NaT)
    labs["stay_id"] = labs["hadm_id"].map(lambda h: cohort_lookup.loc[h, "stay_id"] if h in cohort_lookup.index else None)
    labs = labs.dropna(subset=["intime", "stay_id"])
    labs["window_end"] = labs["intime"] + pd.Timedelta(hours=24)
    labs = labs[(labs["charttime"] >= labs["intime"]) & (labs["charttime"] <= labs["window_end"])]
    
    for itemid, (lo, hi) in lab_ranges.items():
        if itemid in lab_itemids:
            mask = labs["itemid"] == itemid
            labs.loc[mask, "valuenum"] = labs.loc[mask, "valuenum"].clip(lo, hi)
    
    return labs[["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"]]

def extract_vitals_duckdb(cfg, cohort, icu_dir):
    """Extract vitals via DuckDB."""
    import duckdb
    vital_itemids = cfg["symptoms"]["vital_itemids"]
    vital_ranges = cfg["symptoms"]["vital_ranges"]
    stay_list = cohort["stay_id"].unique().tolist()
    stay_str = ",".join(str(x) for x in stay_list[:50000])
    itemid_str = ",".join(str(x) for x in vital_itemids)
    
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    q = f"""
    SELECT subject_id, hadm_id, stay_id, charttime, itemid, valuenum
    FROM read_csv_auto('{icu_dir}/chartevents.csv.gz', header=true)
    WHERE stay_id IN ({stay_str})
    AND itemid IN ({itemid_str})
    AND valuenum IS NOT NULL
    """
    try:
        vitals = con.execute(q).fetchdf()
    except Exception:
        vitals = pd.DataFrame()
        for i in range(0, len(stay_list), 5000):
            batch = stay_list[i:i+5000]
            stay_str_b = ",".join(str(x) for x in batch)
            qb = f"""
            SELECT subject_id, hadm_id, stay_id, charttime, itemid, valuenum
            FROM read_csv_auto('{icu_dir}/chartevents.csv.gz', header=true)
            WHERE stay_id IN ({stay_str_b})
            AND itemid IN ({itemid_str})
            AND valuenum IS NOT NULL
            """
            vb = con.execute(qb).fetchdf()
            vitals = pd.concat([vitals, vb], ignore_index=True) if len(vitals) > 0 else vb
    con.close()
    
    if len(vitals) == 0:
        return pd.DataFrame()
    
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals["valuenum"] = pd.to_numeric(vitals["valuenum"], errors="coerce")
    vitals = vitals.dropna(subset=["valuenum"])
    
    cohort_lookup = cohort.set_index("stay_id")
    vitals["intime"] = vitals["stay_id"].map(lambda s: cohort_lookup.loc[s, "intime"] if s in cohort_lookup.index else pd.NaT)
    vitals = vitals.dropna(subset=["intime"])
    vitals["window_end"] = vitals["intime"] + pd.Timedelta(hours=24)
    vitals = vitals[(vitals["charttime"] >= vitals["intime"]) & (vitals["charttime"] <= vitals["window_end"])]
    
    for itemid, (lo, hi) in vital_ranges.items():
        if itemid in vital_itemids:
            mask = vitals["itemid"] == itemid
            vitals.loc[mask, "valuenum"] = vitals.loc[mask, "valuenum"].clip(lo, hi)
    
    return vitals[["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"]]

def aggregate_to_patient(labs, vitals, cfg):
    """Aggregate: median per (stay_id, itemid), then pivot to one row per stay."""
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
        return pd.DataFrame()
    
    agg = pd.concat(combined, ignore_index=True)
    stays = agg["stay_id"].unique()
    n_features = len(lab_itemids) + len(vital_itemids)
    X = np.full((len(stays), n_features), np.nan)
    stay_to_row = {s: i for i, s in enumerate(stays)}
    
    for _, row in agg.iterrows():
        idx = itemid_to_idx.get(row["itemid"])
        if idx is not None:
            r = stay_to_row[row["stay_id"]]
            if np.isnan(X[r, idx]):
                X[r, idx] = row["valuenum"]
            else:
                X[r, idx] = (X[r, idx] + row["valuenum"]) / 2
    
    p2 = np.nanpercentile(X, 2, axis=0)
    p98 = np.nanpercentile(X, 98, axis=0)
    p98 = np.where(p98 <= p2, p2 + 1, p98)
    X_norm = (X - p2) / (p98 - p2)
    X_norm = np.clip(X_norm, 0, 1)
    col_med = np.nanmedian(X_norm, axis=0)
    for j in range(X_norm.shape[1]):
        mask = np.isnan(X_norm[:, j])
        X_norm[mask, j] = col_med[j]
    X_norm = np.nan_to_num(X_norm, nan=0.5)
    
    df = pd.DataFrame(X_norm, index=stays)
    df.index.name = "stay_id"
    df = df.reset_index()
    return df

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
    
    symptoms = aggregate_to_patient(labs, vitals, cfg)
    if len(symptoms) == 0:
        raise RuntimeError("No symptoms extracted. Check cohort and itemids.")
    
    symptoms = symptoms.merge(
        cohort[["stay_id", "subject_id", "hadm_id"]].drop_duplicates(),
        on="stay_id"
    )
    
    out_path = out_dir / "symptoms.parquet"
    symptoms.to_parquet(out_path, index=False)
    
    print("=" * 60)
    print("SYMPTOM EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Stays with symptoms: {len(symptoms):,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
