# TRIPOD+AI reporting checklist

Mapping of TRIPOD+AI (2024) items to the full-cohort thesis / journal manuscript.
Outcome = ICU discharge-feasibility surrogate; model = XGBoost (primary) among a
benchmarked panel; data = MIMIC-IV v3.1, retrospective, single-center.

| # | TRIPOD+AI item | Addressed? | Where |
|---|---|---|---|
| 1 | Title identifies study as developing/validating a prediction model | Yes | Title; abstract |
| 2 | Structured abstract (objective, data, methods, results, conclusion) | Yes | Abstract |
| 3a | Background and rationale | Yes | Introduction / Background |
| 3b | Objectives / research questions | Yes | Introduction (5 RQs) |
| 4a | Data source(s) | Yes | Methods — MIMIC-IV v3.1 |
| 4b | Study dates / data versions | Yes | Methods (v3.1); single extraction |
| 5a | Eligibility criteria (inclusion/exclusion) | Yes | Methods — cohort |
| 5b | Setting / care unit | Yes | Methods — adult ICU, BIDMC |
| 5c | Sample size and rationale (EPV) | Yes | Methods; ~20,297 events / 22 features |
| 6a | Outcome definition (surrogate, 3 conditions) | Yes | Methods — label; emphasized as surrogate |
| 6b | Outcome assessment blinded to predictors | N/A | Outcome is structured/automatic, not adjudicated (stated as limitation) |
| 7a | Predictors (first-24h labs & vitals; tokens) | Yes | Methods; feature dictionary (appendix) |
| 7b | Predictor assessment timing (≤24h) and leakage control | Yes | Methods + leakage audit |
| 8 | Missing data handling (median imputation, missingness reported) | Yes | Methods; high-missingness flagged |
| 9 | Modeling: algorithms, hyperparameters, class imbalance | Yes | Methods + hyperparameter table |
| 10 | Train/validation/test split; threshold & calibration on validation | Yes | Methods (20,735/5,184/6,480) |
| 11 | Performance measures (AUROC, AUPRC, Brier, ECE, calibration, MCC) | Yes | Methods / Results |
| 12 | Model selection rule (prespecified, calibration-based) | Yes | Methods + Results |
| 13a | Statistical comparison (bootstrap CIs, DeLong) | Yes | Results — DeLong section |
| 13b | Clinical utility (decision-curve / net benefit) | Yes | Results — DCA section |
| 14 | Participants flow / cohort counts | Yes | Methods; cohort table |
| 15 | Cohort characteristics (Table 1: age, sex, LOS) | Yes | Methods — Table 1 |
| 16 | Model performance with uncertainty | Yes | Results — main table with 95% CIs |
| 17 | Model presentation / reproducibility | Yes | Appendix; code + commands |
| 18 | Limitations (single-center, surrogate, no external validation) | Yes | Discussion — limitations |
| 19 | Interpretation vs prior evidence | Yes | Discussion |
| 20 | Implications for practice (not deployment-ready) | Yes | Discussion / Conclusion |
| 21 | Funding | Yes | Declarations (none) |
| 22 | Data/code availability | Yes | Declarations; PhysioNet + code |
| AI-1 | Fairness/subgroup analysis | Partial | Flagged as future work (age/sex/diagnosis available) |
| AI-2 | External validation | No | Explicit limitation; future work |
| AI-3 | Risk-of-bias self-assessment (PROBAST-AI) | Yes | `results/final_optimized/reporting/PROBAST_AI_RISK_OF_BIAS_SELF_ASSESSMENT.md` |

Honest gaps remaining: external/prospective validation (AI-2) and full subgroup
fairness analysis (AI-1). Both are stated as limitations and future work — the
correct, defensible position for a single-center retrospective benchmark.
