# Thesis Revision Report

## What Changed

- Reframed the thesis as a retrospective full-cohort MIMIC-IV clinical machine-learning study.
- Replaced the old 2,000-stay main-cohort framing with the verified full eligible cohort.
- Defined "treatment feasibility" conservatively as an ICU discharge-feasibility proxy.
- Rewrote the Persian thesis source into six coherent academic chapters plus reproducibility appendix.
- Added honest interpretation: pure Few-Shot ETHOS did not outperform classical tabular models.
- Treated XGBoost as the most defensible primary model because it balances discrimination and calibration.

## Full-Cohort Data Used

- Dataset: `data/processed_full_cohort/thesis_dataset.parquet`
- Shape: 32,399 rows x 29 columns
- Feature columns: 22
- Unique ICU stays: 32,399
- Unique subjects: 32,399
- Label distribution: 0=12,102, 1=20,297
- Token tensor: `data/processed_full_cohort/tokens.npy`, shape `(32399, 25, 4)`
- Sampling cap: `null`
- All eligible data used: `true`

## Final Model Results

| Model | AUROC | AUPRC | Accuracy | Recall | Specificity | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stacked Classical | 0.755 | 0.825 | 0.699 | 0.769 | 0.583 | 0.677 | 0.202 | 0.109 |
| XGBoost | 0.753 | 0.825 | 0.697 | 0.761 | 0.589 | 0.675 | 0.189 | 0.020 |
| CatBoost | 0.753 | 0.823 | 0.703 | 0.790 | 0.558 | 0.677 | 0.200 | 0.100 |
| LightGBM | 0.751 | 0.823 | 0.690 | 0.730 | 0.623 | 0.674 | 0.199 | 0.090 |
| Token Hybrid RF | 0.749 | 0.820 | 0.698 | 0.772 | 0.573 | 0.674 | 0.192 | 0.031 |
| Random Forest | 0.748 | 0.820 | 0.689 | 0.742 | 0.601 | 0.670 | 0.192 | 0.028 |
| Neural Baseline | 0.739 | 0.814 | 0.675 | 0.682 | 0.663 | 0.664 | 0.194 | 0.015 |
| Federated LR Node Ensemble | 0.709 | 0.778 | 0.678 | 0.746 | 0.565 | 0.656 | 0.215 | 0.109 |
| Logistic Regression | 0.709 | 0.778 | 0.680 | 0.754 | 0.556 | 0.656 | 0.215 | 0.109 |
| Few-Shot ETHOS | 0.590 | 0.668 | 0.590 | 0.642 | 0.504 | 0.570 | 0.250 | 0.127 |

Best AUROC: Stacked Classical (0.755).
Primary balanced interpretation model: XGBoost (AUROC 0.753, Brier 0.189, ECE 0.020).

## Sections Rewritten

- Persian abstract and English abstract
- Introduction
- Background and literature review
- Methodology
- Experiments and results
- Discussion
- Conclusion
- Reproducibility appendix

## Tables And Figures Regenerated

- Full-cohort dataset summary
- Main model performance table
- Calibration table
- Full-cohort vs historical benchmark comparison
- Label distribution figure
- AUROC figure
- AUPRC figure
- Brier/ECE figure
- XGBoost confusion matrix
- Workflow diagram

## Remaining Limitations

- Retrospective single-database MIMIC-IV study.
- Outcome is a proxy, not clinician-adjudicated treatment feasibility or discharge readiness.
- No prospective or external validation.
- Simulated rather than real multi-site federated learning.
- Few-Shot ETHOS underperformed and requires methodological refinement.
- Raw MIMIC-IV data cannot be redistributed.

## Remaining Author Confirmations

- Confirm exact university administrative metadata, supervisor spelling, date, and defense details.
- Confirm whether local IRB/exemption wording is required by the university.
- Confirm whether the title should keep the conceptual "Therapeutic Few-Shot" wording or add the ICU proxy subtitle on formal submission pages.

## Readiness

- Thesis status: ready for supervisor review after author confirmation of administrative metadata.
- Manuscript package status: full-cohort results are available and the BMC package exists; journal submission should wait for author review of framing and cover-letter details.
