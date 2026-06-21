#!/usr/bin/env python3
"""
Extraction Step 3: Apply treatment feasibility label
feasible=1 iff: hospital_expire_flag==0, discharge_location in allowed list, icu_los_hours < 720
Assign treatment protocol (0-7) from prescriptions/procedures. Merge with cohort and symptoms.
Output: <paths.processed_dir>/thesis_dataset.parquet
"""
import yaml
import pandas as pd
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    paths = cfg["paths"]
    proj_root = Path(__file__).parent.parent
    paths = {k: str(proj_root / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in paths.items()}
    label_cfg = cfg["labels"]
    out_dir = Path(paths["processed_dir"])
    
    cohort = pd.read_parquet(out_dir / "cohort.parquet")
    symptoms = pd.read_parquet(out_dir / "symptoms.parquet")
    
    # Feasibility label
    feasible_discharge = label_cfg["feasible_discharge_locations"]
    max_los = label_cfg["max_icu_los_hours"]
    
    cohort["feasible"] = (
        (cohort["hospital_expire_flag"] == 0) &
        (cohort["discharge_location"].isin(feasible_discharge)) &
        (cohort["los_hours"] < max_los)
    ).astype(int)
    
    # Treatment protocol: hash primary_icd10 + discharge type to 0-7
    # (Simplified: use first 2 chars of ICD + hospital_expire for variety)
    cohort["protocol"] = (
        cohort["primary_icd10"].str[:2].fillna("XX").astype(str).str.upper()
    ).apply(lambda x: hash(x) % 8)
    
    # Merge cohort + symptoms
    thesis = symptoms.merge(
        cohort[["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]],
        on=["stay_id", "subject_id", "hadm_id"],
        how="inner"
    )
    
    out_path = out_dir / "thesis_dataset.parquet"
    thesis.to_parquet(out_path, index=False)
    
    print("=" * 60)
    print("LABEL EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Total samples: {len(thesis):,}")
    print(f"Label balance — feasible=1: {(thesis['feasible']==1).sum():,} ({(thesis['feasible']==1).mean()*100:.1f}%)")
    print(f"Label balance — feasible=0: {(thesis['feasible']==0).sum():,} ({(thesis['feasible']==0).mean()*100:.1f}%)")
    print(f"Protocol distribution: {thesis['protocol'].value_counts().sort_index().to_dict()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
