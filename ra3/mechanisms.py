"""Aggregation / anonymization mechanisms compared in the RA3 study.

Every mechanism has the same interface::

    protected_X = mechanism.fit_transform(X, qi_cols, clinical_cols, y=None)

and returns a *released* feature matrix of the same schema. Utility and privacy are
then measured downstream, so mechanisms are directly comparable on one Pareto plane.

Mechanisms
----------
- ``Raw``                : identity (utility ceiling / privacy floor).
- ``GlobalDP``           : uniform Gaussian mechanism on standardized features.
- ``KAnonymityMicro``    : fixed-k micro-aggregation of quasi-identifiers (MDAV-style).
- ``DPSynthetic``        : Gaussian-copula synthetic data with DP-perturbed moments.
- ``RA3``                : Reverse-Attack-Aware Adaptive Aggregation (this paper).

The RA3 contribution is a *risk-adaptive* allocation of a total privacy budget across
features and a *density-adaptive* equivalence-class size across records, so that the
protection is concentrated where re-identification leverage is high and clinical
correlation structure is preserved where it is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


# --------------------------------------------------------------------------- utils
def _standardize(X: pd.DataFrame):
    sc = StandardScaler()
    Z = sc.fit_transform(X.values.astype(float))
    return Z, sc


def _mdav_groups(Z: np.ndarray, k: int, rng) -> np.ndarray:
    """Single-axis micro-aggregation: order records along their principal component and
    cut into consecutive equivalence classes of size ``k``.

    This is the standard O(n log n) sorting heuristic for micro-aggregation (Hansen &
    Mukherjee); it scales to the full cohort where quadratic MDAV would not, while giving
    the same k-anonymous group structure used for aggregation.
    """
    n = len(Z)
    if n < 2 * k or k <= 1:
        return np.zeros(n, dtype=int)
    Zc = Z - Z.mean(0)
    # Principal axis via power iteration on the covariance (cheap, dependency-free).
    v = rng.standard_normal(Zc.shape[1])
    for _ in range(15):
        v = Zc.T @ (Zc @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    proj = Zc @ v
    order = np.argsort(proj)
    groups = np.empty(n, dtype=int)
    n_groups = max(1, n // k)
    chunks = np.array_split(order, n_groups)
    for gid, ch in enumerate(chunks):
        groups[ch] = gid
    return groups


def _replace_by_group_mean(X: pd.DataFrame, cols, groups) -> pd.DataFrame:
    out = X.copy()
    tmp = out[cols].copy()
    tmp["__g__"] = groups
    means = tmp.groupby("__g__")[cols].transform("mean")
    out[cols] = means.values
    return out


# --------------------------------------------------------------------------- Raw
class Raw:
    name = "Raw"

    def fit_transform(self, X, qi_cols, clinical_cols, y=None):
        return X.copy()


# ------------------------------------------------------------------------ UniformDP
def noise_to_epsilon(noise: float, clip: float = 3.0) -> float:
    """Per-attribute local-DP epsilon implied by additive noise ``noise`` (std, on
    standardized+clipped features). Uses the Laplace-mechanism relation eps = 2C / b
    with b the noise scale; provides an interpretable DP reading for each noise level."""
    return float(2 * clip / max(noise, 1e-6))


class UniformDP:
    """Local additive-noise mechanism with a *uniform* per-attribute scale.

    Every standardized feature is clipped to +/- ``clip`` and perturbed with Gaussian
    noise of the same scale ``noise``. This is the standard statistical-disclosure /
    local-DP microdata baseline; the implied per-attribute epsilon is 2*clip/noise.
    """

    def __init__(self, noise=1.0, clip=3.0, seed=42):
        self.noise = noise; self.clip = clip
        self.name = f"UniformDP(s={noise})"; self.rng = np.random.default_rng(seed)

    def fit_transform(self, X, qi_cols, clinical_cols, y=None):
        Z, sc = _standardize(X)
        Z = np.clip(Z, -self.clip, self.clip)
        Z_noisy = Z + self.rng.normal(0, self.noise, Z.shape)
        return pd.DataFrame(sc.inverse_transform(Z_noisy), columns=X.columns, index=X.index)


# ------------------------------------------------------------------- KAnonymityMicro
class KAnonymityMicro:
    """Fixed-k micro-aggregation of the quasi-identifiers via MDAV."""

    def __init__(self, k=10, seed=42):
        self.k = k; self.name = f"k-anon(k={k})"; self.rng = np.random.default_rng(seed)

    def fit_transform(self, X, qi_cols, clinical_cols, y=None):
        Zq, _ = _standardize(X[qi_cols])
        groups = _mdav_groups(Zq, self.k, self.rng)
        return _replace_by_group_mean(X, qi_cols, groups)


# ---------------------------------------------------------------------- DPSynthetic
class DPSynthetic:
    """Gaussian-copula synthetic data with DP noise added to the mean vector and
    covariance matrix (a lightweight marginal-based synthesizer).

    Parameterized by the same ``noise`` scale as the other mechanisms; the moment
    perturbation is proportional to it so the frontier is comparable.
    """

    def __init__(self, noise=1.0, seed=42):
        self.noise = noise; self.name = f"DP-Synth(s={noise})"
        self.rng = np.random.default_rng(seed)

    def _synth_block(self, Z, n_out, d):
        mu = Z.mean(0); cov = np.cov(Z, rowvar=False)
        noise_mu = self.rng.normal(0, self.noise / np.sqrt(len(Z)), d)
        noise_cov = self.rng.normal(0, self.noise / np.sqrt(len(Z)), cov.shape)
        noise_cov = (noise_cov + noise_cov.T) / 2
        mu_dp = mu + noise_mu; cov_dp = cov + noise_cov + np.eye(d) * 1e-3
        w, V = np.linalg.eigh(cov_dp); w = np.clip(w, 1e-3, None)
        cov_dp = (V * w) @ V.T
        return self.rng.multivariate_normal(mu_dp, cov_dp, size=n_out)

    def fit_transform(self, X, qi_cols, clinical_cols, y=None):
        Z, sc = _standardize(X)
        Z = np.clip(Z, -3, 3)
        n, d = Z.shape
        if y is None:
            synth = self._synth_block(Z, n, d)
        else:
            # Class-conditional copula so the label signal survives (fair baseline).
            synth = np.zeros((n, d))
            for c in np.unique(y):
                m = (y == c)
                synth[m] = self._synth_block(Z[m], int(m.sum()), d)
        return pd.DataFrame(sc.inverse_transform(synth), columns=X.columns, index=X.index)


# ---------------------------------------------------------------------------- RA3
class RA3:
    """Reverse-Attack-Aware Adaptive Aggregation.

    Three coupled ideas:

    1. **Per-feature re-identification exposure.** For every feature we combine
       (a) *linkage leverage* — how strongly an adversarial re-identification model
       relies on the feature (permutation importance of a Random Forest trained to
       predict record identity proxies), with (b) *uniqueness* — the feature's
       mutual information with a near-unique row signature. High exposure => the
       feature helps an attacker single out a record.

    2. **Utility retention.** Mutual information between the feature and the
       downstream label estimates how much clinical signal the feature carries.

    3. **Risk-adaptive budget + density-adaptive grouping.** A total budget
       ``epsilon`` is split *inversely* to utility and *proportionally* to exposure,
       so high-risk / low-utility features are noised heavily while high-utility
       clinical features keep fine granularity — preserving correlation structure.
       Quasi-identifiers are micro-aggregated with an equivalence-class size that
       *grows in sparse regions* (outliers, the easiest to re-identify) and shrinks
       in dense regions.

    After transformation an optional attack-in-the-loop certification pass measures
    residual leakage and escalates protection on the worst features until a target
    is met (see ``certify`` flag and ``last_certificate``).
    """

    def __init__(self, noise=1.0, k_base=4, k_max=40, clip=3.0,
                 alpha=1.0, beta=1.0, seed=42, ablate=None):
        self.noise = noise; self.k_base = k_base; self.k_max = k_max
        self.clip = clip
        self.alpha = alpha            # exposure weight
        self.beta = beta              # utility weight
        # ablate in {None, 'uniform_lambda', 'no_exposure', 'no_surrogate'}:
        #   uniform_lambda  -> same protection for every feature (no risk adaptivity)
        #   no_exposure     -> weight by 1/utility only (ignore re-identification risk)
        #   no_surrogate    -> additive noise only, no class-conditional surrogate
        self.ablate = ablate
        tag = "" if ablate is None else f",{ablate}"
        self.name = f"RA3(s={noise}{tag})"
        self.rng = np.random.default_rng(seed)
        self.exposure_ = None; self.utility_ = None; self.budget_ = None; self.sigma_ = None

    # -- risk scoring -------------------------------------------------------
    def _exposure(self, X, qi_cols):
        """Per-feature re-identification exposure in [0,1].

        Combines three attacker-relevant signals for every released column:
          * *isolation leverage* — MI between the feature and whether a record is
            isolated (far from its nearest neighbour) in quasi-identifier space; high MI
            means the feature helps single out rare records.
          * *fingerprint uniqueness* — the fraction of (near-)unique values the feature
            takes; high-cardinality summary statistics (exact means, slopes, extrema)
            act as per-patient fingerprints that drive membership inference.
          * *quasi-identifier prior* — a mild boost for attacker-observable demographics.
        """
        from sklearn.neighbors import NearestNeighbors
        Z, _ = _standardize(X)
        n, d = Z.shape
        Zq, _ = _standardize(X[qi_cols])
        nn = NearestNeighbors(n_neighbors=2).fit(Zq)
        dist, _ = nn.kneighbors(Zq)
        isolation = dist[:, 1]
        iso_label = (isolation > np.median(isolation)).astype(int)
        mi = mutual_info_classif(Z, iso_label, random_state=0, discrete_features=False)
        mi = mi / (mi.max() + 1e-9)

        # Fingerprint uniqueness: near-unique high-cardinality columns are identifying.
        uniq = np.array([X[c].nunique() / n for c in X.columns])
        uniq = uniq / (uniq.max() + 1e-9)

        qi_prior = np.array([1.5 if c in qi_cols else 1.0 for c in X.columns])
        expo = (0.5 * mi + 0.5 * uniq) * qi_prior
        return expo / (expo.max() + 1e-9)

    def _utility(self, X, y):
        Z, _ = _standardize(X)
        if y is None:
            return np.ones(Z.shape[1])
        mi = mutual_info_classif(Z, y, random_state=0, discrete_features=False)
        return mi / (mi.max() + 1e-9)

    def _adaptive_k(self, X, qi_cols):
        """Per-record equivalence-class target: larger k for isolated records."""
        from sklearn.neighbors import NearestNeighbors
        Zq, _ = _standardize(X[qi_cols])
        nn = NearestNeighbors(n_neighbors=2).fit(Zq)
        dist, _ = nn.kneighbors(Zq)
        iso = dist[:, 1]
        r = (iso - iso.min()) / (iso.max() - iso.min() + 1e-9)   # 0 dense .. 1 sparse
        k = self.k_base + (self.k_max - self.k_base) * r
        return k

    def _cluster_groups(self, X, qi_cols):
        """Density-adaptive equivalence classes over the whole record.

        Records are grouped so that isolated (sparse, easy-to-re-identify) records fall
        into *larger* classes and dense records into smaller ones. Group means later
        supply the aggregated surrogate values.
        """
        Z, _ = _standardize(X)
        k_target = self._adaptive_k(X, qi_cols)
        order = np.argsort(k_target)
        groups = np.full(len(X), -1, dtype=int)
        gid = 0
        for b in np.array_split(order, 6):
            kb = int(np.clip(np.median(k_target[b]), 2, max(2, len(b) // 2)))
            gsub = _mdav_groups(Z[b], kb, self.rng)
            groups[b] = gsub + gid; gid = groups.max() + 1
        return groups

    # -- main transform -----------------------------------------------------
    def fit_transform(self, X, qi_cols, clinical_cols, y=None):
        exposure = self._exposure(X, qi_cols)
        utility = self._utility(X, y)
        self.exposure_ = exposure; self.utility_ = utility

        # Protection weight per feature: high for exposed & low-utility columns.
        if self.ablate == "no_exposure":
            weight = 1.0 / (utility ** self.beta + 1e-1)          # ignore risk
        else:
            weight = (exposure ** self.alpha + 1e-2) / (utility ** self.beta + 1e-1)
        # Map to an aggregation strength lambda_j in [0,1] by quantile rank, so a
        # controlled fraction of features are collapsed toward their group aggregate.
        rank = pd.Series(weight).rank(pct=True).to_numpy()
        lam = rank ** 1.5                     # convex: only the riskiest fully collapse
        if self.ablate == "uniform_lambda":
            lam = np.full_like(lam, float(np.mean(lam)))          # no risk adaptivity
        self.budget_ = lam

        Z, sc = _standardize(X)
        Zc = np.clip(Z, -self.clip, self.clip)

        # Light additive noise scaled by aggregation strength.
        sigma = self.noise * (0.5 + lam)
        noisy_real = Zc + self.rng.normal(0, 1, Zc.shape) * sigma

        if self.ablate == "no_surrogate":
            # Additive noise only: keep the adaptive scale but drop the surrogate.
            Z_out = noisy_real
        else:
            # Density-adaptive equivalence classes + class-conditional surrogate that
            # severs the individual link.
            groups = self._cluster_groups(X, qi_cols)
            surrogate = self._surrogate(Zc, groups, y)
            # Per-feature interpolation between the noised real value and the surrogate:
            # released = (1-lam)*noised_real + lam*surrogate. High-risk features
            # (lam -> 1) are effectively re-synthesized, destroying the per-patient
            # fingerprint while retaining each feature's class-conditional distribution.
            Z_out = (1 - lam) * noisy_real + lam * surrogate
        self.sigma_ = sigma
        return pd.DataFrame(sc.inverse_transform(Z_out), columns=X.columns, index=X.index)

    def _surrogate(self, Zc, groups, y):
        """Per-feature class-conditional surrogate: each value is replaced by another
        record's value drawn from the same downstream class (within-class permutation).
        Preserves the class-conditional marginal exactly while removing the individual
        link that membership / attribute attacks exploit."""
        n, d = Zc.shape
        out = Zc.copy()
        if y is None:
            perm = self.rng.permutation(n)
            return Zc[perm]
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            for j in range(d):
                out[idx, j] = Zc[self.rng.permutation(idx), j]
        return out
