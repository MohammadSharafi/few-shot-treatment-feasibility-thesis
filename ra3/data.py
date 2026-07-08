"""RA3 data layer: load the broker-style tabular medical dataset and define the
threat surface (quasi-identifiers vs. clinical payload).

The enriched MIMIC-IV cohort is treated as a stand-in for the record collection a
medical data broker would hold: per-stay demographic quasi-identifiers plus a large
block of lab/vital summary statistics (the clinically useful "payload").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

# Demographic / administrative attributes an adversary can plausibly obtain from an
# external reference table (voter rolls, insurance, hospital directories). These are
# the re-identification vectors.
QUASI_IDENTIFIERS = [
    "age",
    "sex_male",
    "married",
    "ins_medicare",
    "ins_medicaid",
    "adm_emergency",
    "adm_elective",
    "n_diagnoses",
]

# Direct identifiers – never released.
DIRECT_IDENTIFIERS = ["stay_id", "subject_id", "hadm_id", "primary_icd10"]

TARGET = "feasible"


@dataclass
class BrokerData:
    """A packaged view of the broker dataset ready for mechanisms + attacks."""

    X: pd.DataFrame               # all releasable features (QI + clinical), no target
    y: np.ndarray                 # downstream label (treatment feasibility)
    qi_cols: list[str]            # quasi-identifier columns within X
    clinical_cols: list[str]      # clinical payload columns within X
    member_mask: np.ndarray       # True = record was in the broker's release set
    y_secondary: np.ndarray = None  # a second, unrelated task (prolonged stay)
    feature_names: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.X)


def load_enriched(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def build_broker_data(
    path: str = "data/processed_full_cohort/enriched_dataset.parquet",
    n_sample: int | None = None,
    member_frac: float = 0.5,
    seed: int = 42,
) -> BrokerData:
    """Assemble the release matrix and a member/non-member partition.

    ``member_frac`` of records are the population the broker actually aggregates and
    releases; the remainder are held-out "non-members" drawn from the same population,
    used as the negative class for membership-inference evaluation.
    """
    rng = np.random.default_rng(seed)
    df = load_enriched(path)

    if n_sample is not None and n_sample < len(df):
        idx = rng.choice(len(df), size=n_sample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    y = df[TARGET].to_numpy().astype(int)

    # Secondary, unrelated downstream task: prolonged ICU stay (> cohort median).
    # Used to probe *general* fidelity of a release, not just the task it was tuned on.
    los = df["los_hours"].to_numpy()
    y_secondary = (los > np.nanmedian(los)).astype(int)

    # Feature block = everything releasable.
    drop = set(DIRECT_IDENTIFIERS + [TARGET, "los_hours"])
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols].copy()

    # Median-impute any residual NaNs so mechanisms/attacks are well-defined.
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)

    qi_cols = [c for c in QUASI_IDENTIFIERS if c in X.columns]
    clinical_cols = [c for c in feature_cols if c not in qi_cols]

    member_mask = rng.random(len(X)) < member_frac

    return BrokerData(
        X=X.reset_index(drop=True),
        y=y,
        qi_cols=qi_cols,
        clinical_cols=clinical_cols,
        member_mask=member_mask,
        y_secondary=y_secondary,
        feature_names=feature_cols,
    )


def quasi_identifier_uniqueness(X: pd.DataFrame, qi_cols: list[str], bins: int = 0) -> dict:
    """Report equivalence-class structure over the quasi-identifiers.

    ``bins`` > 0 coarsens continuous QIs first, mirroring what an attacker's reference
    table granularity would be.
    """
    Q = X[qi_cols].copy()
    if bins > 0:
        for c in qi_cols:
            if Q[c].nunique() > bins:
                Q[c] = pd.qcut(Q[c], q=bins, duplicates="drop").cat.codes
    g = Q.groupby(list(qi_cols)).size()
    singletons = int((g == 1).sum())
    return {
        "n_records": int(len(X)),
        "n_classes": int(len(g)),
        "n_singletons": singletons,
        "frac_singleton_records": float(g[g == 1].sum() / len(X)),
        "min_class_size": int(g.min()),
    }


def train_holdout_split(bd: BrokerData, test_size: float = 0.3, seed: int = 42):
    """Split members into a train pool (to be aggregated) and a real test set used
    only to measure downstream utility on genuine, unprotected records."""
    members_idx = np.where(bd.member_mask)[0]
    y_mem = bd.y[members_idx]
    tr, te = train_test_split(
        members_idx, test_size=test_size, random_state=seed, stratify=y_mem
    )
    return tr, te
