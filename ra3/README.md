# RA³ — Reverse-Attack-Aware Adaptive Aggregation

Code for the paper *"Reverse-Attack-Aware Adaptive Aggregation: A Certified,
Fidelity-Preserving Framework for Re-identification-Resistant Release of Brokered
Medical Data."*

## What this is

A study of how to release **real, auditable** medical microdata that resists
*reverse-engineering* (re-identification) attacks a data broker faces, and a new
mechanism, **RA³**, that allocates protection per feature by re-identification exposure
and downstream utility.

The dataset is the enriched MIMIC-IV ICU cohort (`data/processed_full_cohort/
enriched_dataset.parquet`): 32,399 stays, 8 demographic quasi-identifiers, 198
lab/vital summary features, a primary label (treatment feasibility) and a secondary,
untuned label (prolonged stay).

## Modules

| File | Role |
|------|------|
| `data.py` | Load the broker table; define quasi-identifiers vs. clinical payload; member/non-member split; QI-uniqueness diagnostics. |
| `mechanisms.py` | `Raw`, `UniformDP`, `KAnonymityMicro` (single-axis micro-aggregation), `DPSynthetic` (class-conditional Gaussian copula), and `RA3`. |
| `attacks.py` | Membership inference (distance test), linkage re-identification (NN on QIs), attribute inference (predict a secret clinical attribute). |
| `utility.py` | Train-on-released / test-on-real downstream AUROC/AUPRC/F1. |
| `run_experiments.py` | Orchestrates the full privacy–utility–fidelity sweep. |
| `make_macros.py` | Splices result numbers into the LaTeX paper. |
| `figures.py` | Generates the four paper figures. |

## Reproduce

```bash
# full cohort (single seed)
python3 ra3/run_experiments.py

# quick sample
python3 ra3/run_experiments.py 3000

# 3-seed variance
python3 ra3/run_experiments.py --multiseed

# inject numbers + build figures + compile paper
python3 ra3/make_macros.py
python3 ra3/figures.py
cd paper_ra3 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## The RA³ mechanism (one paragraph)

For each feature *j* compute an **exposure** score *eⱼ* (isolation mutual-information +
value uniqueness + a quasi-identifier prior) and a **utility** score *uⱼ* (mutual
information with the label). Turn the ratio into an aggregation strength
*λⱼ = rank(eⱼ/uⱼ)^1.5 ∈ [0,1]*. Group records into density-adaptive equivalence classes
(larger classes for isolated, easy-to-re-identify records). Release

```
x̃ᵢⱼ = (1−λⱼ)·(xᵢⱼ + σⱼ·noise) + λⱼ·sᵢⱼ
```

where *sᵢⱼ* is a within-class permuted surrogate. High-exposure/low-utility features
(*λ→1*) are re-synthesized (fingerprint destroyed); high-utility features (*λ→0*) stay
near-real. An attack-in-the-loop pass certifies residual risk.

## Key findings

- **96.3%** of patients are unique on quasi-identifiers alone → trivially linkable.
- **k-anonymity fixes the wrong surface**: membership- and attribute-inference stay at
  raw levels because the clinical payload is a fingerprint.
- **DP synthesis is a fidelity mirage**: great on the tuned task, collapses to ~chance on
  an untuned task, and releases no real records.
- **RA³** holds the best real-record operating point: both tasks preserved, linkage and
  membership risk driven down, with a per-feature certificate.
