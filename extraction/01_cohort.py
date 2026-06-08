#!/usr/bin/env python3
"""
Extraction Step 1: Build cohort from MIMIC-IV v3.1
Criteria: age>=18, first ICU stay only, ICU LOS>=4h, has primary ICD-10, not missing discharge_location
Output: data/processed/cohort.parquet
"""
import os
import yaml
import pandas as pd
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    seed = cfg["seed"]
    paths = cfg["paths"]
    proj_root = Path(__file__).parent.parent
    paths = {k: str(proj_root / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in paths.items()}
    cohort_cfg = cfg["cohort"]
    
    hosp_dir = Path(paths["hosp_dir"])
    icu_dir = Path(paths["icu_dir"])
    out_dir = Path(paths["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tables
    patients = pd.read_csv(hosp_dir / "patients.csv.gz")
    admissions = pd.read_csv(hosp_dir / "admissions.csv.gz")
    icustays = pd.read_csv(icu_dir / "icustays.csv.gz")
    diagnoses = pd.read_csv(hosp_dir / "diagnoses_icd.csv.gz")
    
    # Parse datetimes
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
    
    # Merge admissions for discharge_location, hospital_expire_flag
    adm = admissions[["hadm_id", "subject_id", "discharge_location", "hospital_expire_flag"]].copy()
    adm = adm.dropna(subset=["discharge_location"])
    
    # Age >= 18 (anchor_age in MIMIC-IV)
    patients_adult = patients[patients["anchor_age"] >= cohort_cfg["min_age"]][["subject_id"]]
    
    # First ICU stay per patient
    icustays_sorted = icustays.sort_values(["subject_id", "intime"])
    first_stays = icustays_sorted.groupby("subject_id").first().reset_index()
    
    # ICU LOS >= 4 hours
    first_stays = first_stays[first_stays["los_hours"] >= cohort_cfg["min_icu_los_hours"]]
    
    # Merge with admissions
    cohort = first_stays.merge(adm, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(patients_adult, on="subject_id", how="inner")
    
    # Has primary ICD-10 diagnosis (seq_num==1, icd_version==10)
    diag_primary = diagnoses[(diagnoses["seq_num"] == 1) & (diagnoses["icd_version"] == 10)]
    diag_primary = diag_primary[["hadm_id", "icd_code"]].drop_duplicates()
    cohort = cohort.merge(diag_primary, on="hadm_id", how="inner")
    
    # Select columns for output
    cohort_out = cohort[[
        "subject_id", "hadm_id", "stay_id", "intime", "outtime", "los_hours",
        "discharge_location", "hospital_expire_flag", "icd_code"
    ]].copy()
    cohort_out = cohort_out.rename(columns={"icd_code": "primary_icd10"})
    
    out_path = out_dir / "cohort.parquet"
    cohort_out.to_parquet(out_path, index=False)
    
    print("=" * 60)
    print("COHORT EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Total patients: {len(cohort_out):,}")
    print(f"Unique subjects: {cohort_out['subject_id'].nunique():,}")
    print(f"Unique stays: {cohort_out['stay_id'].nunique():,}")
    print(f"LOS (hours) — median: {cohort_out['los_hours'].median():.1f}, mean: {cohort_out['los_hours'].mean():.1f}")
    print(f"Discharge locations: {cohort_out['discharge_location'].value_counts().to_dict()}")
    print(f"hospital_expire_flag=1 (died): {(cohort_out['hospital_expire_flag']==1).sum():,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
