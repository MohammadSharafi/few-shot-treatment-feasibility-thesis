#!/usr/bin/env python3
"""
Cohort-expansion analysis (thesis pillar #4: "use all the data").

Quantifies how much the eligible cohort grows when ICD-9-coded admissions are
included alongside ICD-10 (the default thesis cohort is ICD-10 only, because the
disease-node grouping needs consistent ICD-10 codes). This documents, with real
numbers, exactly how much data the prespecified criteria exclude and why.

Reads the RAW MIMIC-IV files directly (no parquet engine needed). Reports the
funnel and label distribution for ICD-10-only vs ICD-9+ICD-10.

Outputs: results/canonical/cohort_expansion.csv
To actually MODEL the expanded cohort, run:
    THESIS_CONFIG=config_expanded.yaml python extraction/01_cohort.py   # ...02..05
    THESIS_CONFIG=config_expanded.yaml python scripts/run_full_cohort_models.py
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "mimic-iv-3.1/hosp"; I = ROOT / "mimic-iv-3.1/icu"
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
FAV = ["HOME", "HOME HEALTH CARE", "HOME WITH HOME IV PROVIDR", "REHAB/DISTINCT PART HOSP", "REHAB"]


def build(icd_versions):
    pat = pd.read_csv(H / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    adm = pd.read_csv(H / "admissions.csv.gz",
                      usecols=["hadm_id", "subject_id", "discharge_location", "hospital_expire_flag"])
    icu = pd.read_csv(I / "icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])
    icu["los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600
    first = icu.sort_values(["subject_id", "intime"]).groupby("subject_id").first().reset_index()
    adults = pat[pat.anchor_age >= 18][["subject_id"]]
    f = first.merge(adults, on="subject_id").query("los_hours >= 4")
    f = f.merge(adm.dropna(subset=["discharge_location"]), on=["subject_id", "hadm_id"])
    dx = pd.read_csv(H / "diagnoses_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_version", "icd_code"])
    prim = dx[(dx.seq_num == 1) & (dx.icd_version.isin(icd_versions))][["hadm_id"]].drop_duplicates()
    f = f.merge(prim, on="hadm_id")
    surv = f.hospital_expire_flag == 0
    feas = (surv & f.discharge_location.isin(FAV) & (f.los_hours < 720)).astype(int)
    return {"icd_versions": "+".join(map(str, icd_versions)), "n_stays": len(f),
            "n_feasible": int(feas.sum()), "n_infeasible": int((feas == 0).sum()),
            "prevalence": round(feas.mean(), 4)}


def main():
    rows = [build([10]), build([9, 10])]
    df = pd.DataFrame(rows)
    df["extra_vs_icd10"] = df.n_stays - df.loc[0, "n_stays"]
    df.to_csv(OUT / "cohort_expansion.csv", index=False)
    print(df.to_string(index=False))
    gain = df.loc[1, "n_stays"] - df.loc[0, "n_stays"]
    print(f"\nIncluding ICD-9 adds {gain:,} eligible stays "
          f"({100*gain/df.loc[0,'n_stays']:.0f}% more) -> total {df.loc[1,'n_stays']:,}.")
    print("Wrote", OUT / "cohort_expansion.csv")


if __name__ == "__main__":
    main()
