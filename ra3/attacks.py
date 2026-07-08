"""Adversarial privacy evaluation for released tabular medical data.

Three attack families, each returning a scalar risk in a comparable range:

- ``membership_inference_attack`` : can an adversary tell whether a target record was
  in the broker's release set? Distance-to-closest-record test calibrated on a
  reference population. Reported as attack AUC (0.5 = no leakage).

- ``linkage_reidentification``    : given an external reference table of quasi-identifiers
  for known individuals, what fraction can be correctly matched back to their released
  record by nearest neighbour? Reported as top-1 re-identification rate.

- ``attribute_inference_attack``  : can a sensitive clinical attribute be predicted from
  the quasi-identifiers of a released record? Reported as balanced accuracy over a
  trivial-baseline.

All attacks are mechanism-agnostic: they see only the released matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def _fit_scaler(ref: pd.DataFrame):
    sc = StandardScaler().fit(ref.values.astype(float))
    return sc


def membership_inference_attack(
    released: pd.DataFrame,
    members_orig: pd.DataFrame,
    nonmembers_orig: pd.DataFrame,
    cols: list[str],
    seed: int = 42,
    max_probe: int = 4000,
) -> dict:
    """Distance-based MIA.

    Intuition: if a record was used to build the release, its *closest* released
    record should be unusually near (the mechanism partially memorised it). We score
    each probe by the negative distance to its nearest released neighbour, then measure
    how well that score separates true members from non-members.

    ``max_probe`` caps the number of member/non-member probes for tractability; the AUC
    estimate is stable well below the full set.
    """
    rng = np.random.default_rng(seed)
    sc = _fit_scaler(released[cols])
    R = sc.transform(released[cols].values.astype(float))
    nn = NearestNeighbors(n_neighbors=1).fit(R)

    def subsample(df):
        if len(df) > max_probe:
            idx = rng.choice(len(df), size=max_probe, replace=False)
            return df.iloc[idx]
        return df

    def closeness(df):
        Q = sc.transform(df[cols].values.astype(float))
        d, _ = nn.kneighbors(Q)
        return -d[:, 0]                       # higher = closer = looks like a member

    s_mem = closeness(subsample(members_orig))
    s_non = closeness(subsample(nonmembers_orig))
    y = np.r_[np.ones(len(s_mem)), np.zeros(len(s_non))]
    s = np.r_[s_mem, s_non]
    auc = roc_auc_score(y, s)
    # Report advantage so direction is unambiguous (>=0.5 attacker-favourable).
    return {"mia_auc": float(max(auc, 1 - auc)),
            "mia_advantage": float(2 * max(auc, 1 - auc) - 1)}


def linkage_reidentification(
    released: pd.DataFrame,
    members_orig: pd.DataFrame,
    qi_cols: list[str],
    n_probe: int = 2000,
    seed: int = 42,
) -> dict:
    """Nearest-neighbour linkage on quasi-identifiers.

    The adversary holds the *true* quasi-identifiers of known members (the reference
    table) and links each to the closest released record. A hit = the closest released
    record is the individual's own released row (index-aligned).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(released))
    if n_probe < len(idx):
        probe = rng.choice(idx, size=n_probe, replace=False)
    else:
        probe = idx
    sc = _fit_scaler(released[qi_cols])
    R = sc.transform(released[qi_cols].values.astype(float))
    nn = NearestNeighbors(n_neighbors=1).fit(R)
    Q = sc.transform(members_orig.iloc[probe][qi_cols].values.astype(float))
    _, nbr = nn.kneighbors(Q)
    hits = (nbr[:, 0] == probe).mean()
    # Baseline random-guess re-identification rate.
    return {"reid_rate": float(hits), "reid_baseline": float(1.0 / len(released))}


def attribute_inference_attack(
    released: pd.DataFrame,
    sensitive: np.ndarray,
    attacker_cols: list[str],
    seed: int = 42,
) -> dict:
    """Predict a sensitive binary attribute from the rest of the released record.

    The adversary sees a target's released feature vector (quasi-identifiers plus the
    clinical payload a lab-data broker distributes) and tries to infer a secret clinical
    attribute that the mechanism should have obscured. Leakage = balanced accuracy above
    chance. A mechanism that preserves the joint distribution too faithfully leaks here.
    """
    rng = np.random.default_rng(seed)
    n = len(released)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    X = released[attacker_cols].values.astype(float)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1,
                                 random_state=seed)
    clf.fit(X[tr], sensitive[tr])
    pred = clf.predict(X[te])
    bacc = balanced_accuracy_score(sensitive[te], pred)
    base = max(np.mean(sensitive[te]), 1 - np.mean(sensitive[te]))
    return {"attr_bacc": float(bacc), "attr_baseline": float(base),
            "attr_leakage": float(max(0.0, bacc - 0.5) / 0.5)}


def run_all_attacks(released, members_orig, nonmembers_orig, qi_cols, all_cols,
                    sensitive, sensitive_source, seed=42) -> dict:
    out = {}
    out.update(membership_inference_attack(released, members_orig, nonmembers_orig,
                                           all_cols, seed))
    out.update(linkage_reidentification(released, members_orig, qi_cols, seed=seed))
    # Attacker uses every released feature except the one the secret is derived from.
    attacker_cols = [c for c in all_cols if c != sensitive_source]
    out.update(attribute_inference_attack(released, sensitive, attacker_cols, seed))
    return out
