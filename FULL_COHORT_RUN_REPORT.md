# Full Cohort Run Report

## Data Use

- All eligible data used: True
- Sampling cap: None
- Raw MIMIC-IV v3.1 manifest complete by file presence: True
- Raw SHA-256 manifest check passed: True
- Final dataset shape: 32,399 rows x 29 columns
- Unique ICU stays: 32,399
- Unique subjects: 32,399
- Label distribution: 0: 12102, 1: 20297
- Token tensor shape: (32399, 25, 4)
- Mean pre-imputation missingness: 25.239%
- Post-imputation missing cells: 0

## Models Rerun On Full Cohort

- Logistic Regression
- Random Forest
- Neural Baseline
- XGBoost
- LightGBM
- CatBoost
- Stacked Classical
- Token Hybrid RF
- Federated LR Node Ensemble
- Few-Shot ETHOS

## Failed Or Skipped Models

- None

## Updated Results

Best full-cohort model: Stacked Classical with AUROC 0.755, F1 0.677, AUPRC 0.825, accuracy 0.699, MCC 0.354.

Full table: `results/full_cohort/tables/full_cohort_model_results.csv`.

Benchmark comparison table: `results/full_cohort/tables/full_vs_2k_benchmark_comparison.csv`. Historical benchmark rows are context only; non-identical recipes are labelled as benchmark references.

## Remaining Limitations

- The treatment-feasibility label is a proxy based on discharge disposition, survival flag, and ICU length of stay.
- Feature extraction uses selected first-24-hour laboratory and vital-sign variables; missingness is measured before imputation.
- Results are local full-cohort reruns on the available MIMIC-IV v3.1 copy and should be externally validated before clinical use.
- Advanced methods are interpreted only when their full-cohort run completed successfully.

## Thesis And Paper Basis

The revised thesis and manuscript PDFs generated in `papers/full_cohort_revision/` are based on full-cohort outputs. The old 2,000-stay benchmark is historical context only and is not reused as a full-cohort result.
