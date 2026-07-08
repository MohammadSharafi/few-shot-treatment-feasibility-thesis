#!/usr/bin/env python3
"""
Readmission-aware labelling analysis (ETHOS-inspired; thesis labelling focus).

The ETHOS paper (Renc et al., npj Digital Medicine 2024) treats hospital
readmission as a core outcome. A discharge that results in a 30-day readmission is
arguably not a "feasible" discharge. The thesis config already declares
max_readmit_days=30 but the current label does not use it. This script quantifies,
from the raw data, how a readmission-aware label differs from the primary label:

  L_primary      survive AND favourable discharge AND ICU LOS < 720h
  L_readmit      L_primary AND NOT readmitted to hospital within 30 days

Reads raw MIMIC-IV files (no parquet engine needed). Reports the cohort funnel,
how many "feasible" discharges are followed by a 30-day readmission, and the new
label prevalence.

Output: results/canonical/readmission_label.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "mimic-iv-3.1/hosp"; I = ROOT / "mimic-iv-3.1/icu"
OUT = ROOT / "results/canonical"; OUT.mkdir(parents=True, exist_ok=True)
FAV = ["HOME", "HOME HEALTH CARE", "HOME WITH HOME IV PROVIDR", "REHAB/DISTINCT PART HOSP", "REHAB"]
READMIT_DAYS = 30


def main():
    pat = pd.read_csv(H / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    adm = pd.read_csv(H / "admissions.csv.gz",
                      usecols=["hadm_id", "subject_id", "admittime", "dischtime",
                               "discharge_location", "hospital_expire_flag"])
    adm["admittime"] = pd.to_datetime(adm["admittime"]); adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    icu = pd.read_csv(I / "icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])
    icu["los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600

    # cohort: first adult ICU stay, LOS>=4h, known discharge, primary ICD-10
    first = icu.sort_values(["subject_id", "intime"]).groupby("subject_id").first().reset_index()
    adults = pat[pat.anchor_age >= 18][["subject_id"]]
    f = first.merge(adults, on="subject_id").query("los_hours >= 4")
    f = f.merge(adm.dropna(subset=["discharge_location"]), on=["subject_id", "hadm_id"])
    dx = pd.read_csv(H / "diagnoses_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_version"])
    prim = dx[(dx.seq_num == 1) & (dx.icd_version == 10)][["hadm_id"]].drop_duplicates()
    f = f.merge(prim, on="hadm_id")

    # ---- 30-day readmission: next admission within 30 days of index discharge ----
    adm_sorted = adm.sort_values(["subject_id", "admittime"])
    adm_sorted["next_admit"] = adm_sorted.groupby("subject_id")["admittime"].shift(-1)
    nextmap = adm_sorted.set_index("hadm_id")["next_admit"]
    f["next_admit"] = f["hadm_id"].map(nextmap)
    f["gap_days"] = (f["next_admit"] - f["dischtime"]).dt.total_seconds() / 86400
    f["readmit_30d"] = (f["gap_days"] >= 0) & (f["gap_days"] <= READMIT_DAYS)

    surv = f.hospital_expire_flag == 0
    primary = surv & f.discharge_location.isin(FAV) & (f.los_hours < 720)
    readmit_aware = primary & (~f.readmit_30d)

    n = len(f)
    feasible_readmitted = int((primary & f.readmit_30d).sum())
    rows = [
        {"label": "L_primary", "n_pos": int(primary.sum()), "prevalence": round(primary.mean(), 4)},
        {"label": "L_readmit_aware", "n_pos": int(readmit_aware.sum()), "prevalence": round(readmit_aware.mean(), 4)},
    ]
    df = pd.DataFrame(rows)
    df["n_total"] = n
    df.to_csv(OUT / "readmission_label.csv", index=False)

    print(f"Cohort size: {n:,}")
    print(f"30-day readmission rate (whole cohort): {f.readmit_30d.mean()*100:.1f}%")
    print(f"Feasible discharges that were readmitted within 30 days: {feasible_readmitted:,} "
          f"({100*feasible_readmitted/int(primary.sum()):.1f}% of feasible)")
    print(df.to_string(index=False))
    print("Wrote", OUT / "readmission_label.csv")


if __name__ == "__main__":
    main()
