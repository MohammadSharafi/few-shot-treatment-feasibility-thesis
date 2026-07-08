# Expert Review — Full-Cohort Thesis & Gap Analysis

**Reviewer pass:** independent, reviewer-2 / MIT-committee standard
**Artifact reviewed:** `papers/full_cohort_revision/thesis_FULL_COHORT_REVISED.tex` + `thesis/chapters_full_cohort/*.tex` (canonical full-cohort Persian thesis, 48 pp.)
**Title (unchanged, per author):** *فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده* — "Therapeutic Few-Shot: Intelligent Prediction of Treatment Feasibility with Symptom Tokens from Aggregated Data"
**Date:** 2026-06-28

---

## Overall assessment

The thesis is **scientifically honest and well-positioned**. Its core virtues — a full eligible cohort (32,399 first adult ICU stays), an explicitly *surrogate* label, transparent reporting of a **negative** result for the headline method (Few-Shot ETHOS underperforms), and calibration reported alongside discrimination — are exactly what strong reviewers reward. The reframing from "treatment feasibility" to "ICU discharge-feasibility surrogate" is handled with integrity.

The weaknesses are **not** about honesty; they are about **rigor of evidence presentation and methodological completeness**. In its current state the thesis reads like a competent applied-ML study but stops short of the depth an MIT committee or a Q1 medical-informatics journal expects. The single highest-impact fact is already in your own results files but **missing from the thesis**: every model has a **95% bootstrap CI**, and the top six tabular models' CIs **overlap heavily** — meaning the "best model" ranking on which the narrative leans is *statistically a tie*. Surfacing this makes the thesis both more correct and more sophisticated.

Below, issues are prioritized **P0** (correctness / would draw a major revision or viva objection), **P1** (rigor & completeness expected at this level), **P2** (polish).

---

## P0 — Must fix (correctness & defensibility)

### P0-1. Report confidence intervals and state that the top models are statistically indistinguishable
The main results table (`experiments_results.tex`, Table `tab:model_results`) ranks Stacked Classical (0.755) > CatBoost/XGBoost (0.753) > LightGBM (0.751) > Token Hybrid RF (0.749) > RF (0.748), and the text leans on this ordering. But your canonical file `results/full_cohort/tables/full_cohort_model_results.csv` already contains bootstrap CIs:

| Model | AUROC | 95% CI |
|---|---|---|
| Stacked Classical | 0.755 | 0.743–0.766 |
| CatBoost | 0.753 | 0.741–0.764 |
| XGBoost | 0.753 | 0.740–0.764 |
| LightGBM | 0.751 | 0.739–0.762 |
| Token Hybrid RF | 0.750 | 0.738–0.762 |
| Random Forest | 0.748 | 0.735–0.760 |
| Neural Baseline | 0.735 | 0.722–0.747 |
| Federated LR Ensemble | 0.709 | 0.696–0.722 |
| Logistic Regression | 0.709 | (≈0.696–0.722) |
| Few-Shot ETHOS | 0.590 | — |

The top six CIs overlap almost completely. **Fix:** add a `95% CI` column to the main table, and add one sentence: *"Differences among the top six tabular models lie within overlapping bootstrap confidence intervals and should be treated as a statistical tie; model choice is therefore made on calibration, not on AUROC rank."* This single change converts a soft claim into a rigorous one and pre-empts the most obvious reviewer objection. Ideally add a paired DeLong / bootstrap-difference test for Stacked vs XGBoost vs RF (Δ not significant).

### P0-2. Make the primary-model selection rule prespecified, not post-hoc on test
The thesis selects XGBoost as the "more defensible primary model" because of its ECE (0.020) after presenting test results. Read uncharitably, this is *selection on the test set*. Your `LEAKAGE_AUDIT_REPORT.md` confirms calibration/threshold were chosen on **validation**, which is the defense — but the thesis never says so at the selection moment. **Fix:** state explicitly and early (Methods) that the model-selection criterion (best validation calibration among models within the top AUROC CI band) was **prespecified**, and that test-set numbers are reported once, read-only. Move the "why XGBoost" justification to reference validation metrics.

### P0-3. Quantify the surrogate label's components (circularity / triviality check)
The positive label requires three conditions: survival to discharge, favorable discharge destination, **and** ICU LOS < 720 h (30 days). Reviewers will immediately ask: *how much of the label is just "survived"?* and *is LOS < 720 h ever false?* (30 days is extreme; it may be near-constant, making it a vacuous condition). **Fix:** add a short table giving the marginal frequency of each condition and how many negatives are driven by each (mortality vs. unfavorable destination vs. LOS≥720 h). If LOS<720 h is true for ~99% of stays, say so and note it contributes negligibly. This closes the "is the label trivial / circular?" line of attack.

### P0-4. The 99.8%-missing feature must be addressed
`full_cohort_missingness.csv` shows feature index 9 (itemid 51300, WBC) at **99.81% missing**, itemid 50811 (hemoglobin, blood-gas) at 77.8%, and five features (lactate, ALT, AST, arterial BP ×3) at 51–62% missing — all median-imputed. A feature that is 99.8% missing is **effectively a constant** after imputation and is indefensible to a methods reviewer. **Fix:** either (a) drop features above a prespecified missingness threshold (e.g., >70%) and re-state the feature count, or (b) explicitly justify retention and show via the existing `missingness_indicators` ablation that they carry signal (your ablation actually shows missingness indicators *alone* reach AUROC 0.669 — informative missingness exists, which is worth saying). Report the full per-feature missingness table in an appendix.

### P0-5. Methods chapter omits reproducibility-critical detail
The methodology chapter does not currently state: the **imputation method** (it is per-column median, computed in `extraction/02_symptoms.py`), **feature scaling** (StandardScaler for LR/NN), **class-imbalance handling** (`class_weight="balanced"` / `balanced_subsample` / CatBoost `auto_class_weights`), **hyperparameters** (RF/GBMs n_estimators≈300, XGBoost max_depth 4 / lr 0.05, etc. — fixed, *not* grid-searched), the **internal validation split** used for threshold tuning, **software versions**, and **random seeds**. At MIT/Q1 level these are mandatory. **Fix:** add a "Modeling protocol" subsection + a hyperparameter table in the appendix. State plainly that hyperparameters were fixed at sensible defaults rather than tuned, and that the seed-robustness analysis (below) shows results are stable — this turns a potential weakness into a transparency strength.

---

## P1 — Important (rigor & completeness expected at this level)

### P1-1. No "Table 1" patient characteristics (demographics)
A clinical-prediction study without a cohort-characteristics table (age, sex, admission type, primary diagnosis groups, label prevalence by subgroup) is incomplete and is a near-automatic reviewer request. **Fix:** add Table 1 generated from `data/processed_full_cohort/thesis_dataset.parquet`. (A ready-to-run snippet is provided in the companion fix; it could not be auto-generated here because the analysis sandbox lacks a parquet engine.)

### P1-2. The ablation study exists but is not in the thesis
`results/final_optimized/ABLATION_REPORT.md` contains a **feature-family ablation** and a **seed-robustness** check that are currently absent from the thesis. They are high-value:
- *tabular_only* AUROC 0.7535 vs *tabular+token* 0.7531 → tokens add ≈0 on top of tabular features (honest, supports your hybrid-vs-standalone narrative quantitatively).
- *token_only* 0.739, *labs_only* 0.737, *vitals_only* 0.663, *missingness_indicators_only* 0.669.
- Seed robustness (Token Hybrid XGBoost, seeds 7/42/2026): AUROC 0.7536 / 0.7531 / 0.7532 → essentially invariant.
**Fix:** add an "Ablation and robustness" section with these two tables. This directly answers RQ3 (do tokens add signal?) with evidence rather than assertion, and pre-empts "your results may be seed-luck."

### P1-3. No clinical severity-score baseline (SOFA / SAPS-II / OASIS)
The strongest reviewer question for any ICU-outcome model is: *does it beat the bedside score clinicians already use?* The thesis compares ML families to each other but never to a standard severity score. **Fix (one of):** (a) compute OASIS/SAPS-II from the first-24h data and add it as a baseline row; or, if out of scope, (b) explicitly acknowledge the absence as a limitation and frame AUROC 0.75 relative to typical severity-score AUROCs reported in the literature for related endpoints. Option (a) is strongly preferred and substantially raises the contribution.

### P1-4. Federated experiment is under-specified
One sentence currently. Your code (`run_federated_lr`) reveals: **5 disease-partitioned nodes** (`node_1..5.parquet`), each trains a balanced LR, and node probabilities are **size-weighted averaged** — i.e., a *simulated weighted node ensemble*, not gradient-level FedAvg. **Fix:** describe the partition (by disease group → strongly non-IID), the aggregation rule, nodes_used = 5, and state honestly that this is prediction-level aggregation, not parameter-level federated optimization. Report per-node prevalence/size. Note the non-IID partition is a *harder* setting, which strengthens the "comparable to centralized LR" finding.

### P1-5. Few-Shot ETHOS degeneracy should be named
Its calibration slope is **0.000** (Table `tab:calibration`) — the probability output is essentially non-informative, not merely "weaker." **Fix:** state that the few-shot model's probabilities are near-degenerate (slope≈0), so its AUROC 0.590 reflects weak ranking and effectively no usable calibration; this sharpens the negative result and motivates the hybrid direction.

### P1-6. Related work / background is thin for the level (~8 KB)
Missing strands: MIMIC benchmarking literature (Harutyunyan et al.; Purushotham et al. — both already in your `references.bib`), EHR foundation/representation models (BEHRT, G-BERT — in your bib), the "deep learning rarely beats GBMs on tabular EHR" finding, and prior few-shot-on-tabular results. **Fix:** expand background by ~1.5–2 pp. weaving in these citations (all present in `references.bib`, currently uncited). This grounds the contribution in the field.

### P1-7. Reconcile cross-runner metric inconsistencies (which runner is canonical?)
The thesis main table mixes provenance: AUROC/CIs come from `results/full_cohort/` (`run_full_cohort_models.py`), but Brier/ECE/slope values (e.g., XGBoost Brier 0.189 / ECE 0.020) match the `final_optimized` runner, whose `CALIBRATION_REPORT.md` instead reports best Brier = Stacked 0.187 and best ECE = LightGBM 0.014. Two runners are being blended in one table without a stated mapping. **Fix:** declare one runner canonical for every reported number, regenerate the table from that single source, or add a footnote precisely stating which columns come from which runner and why they are comparable.

### P1-8. Calibration shown only as a bar chart; add reliability diagrams
ECE/Brier bars don't show *where* miscalibration occurs. `results/final_optimized/figures/calibration_curves.png` already exists. **Fix:** include per-model reliability curves for at least XGBoost vs Stacked, and state whether reported probabilities are pre- or post-calibration (your ablation uses isotonic — clarify the main table's calibration state).

### P1-9. TRIPOD / PROBAST cited but not completed
You cite TRIPOD and PROBAST but include no checklist. **Fix:** add a completed TRIPOD (or TRIPOD-AI) checklist as supplementary material; map each item to a thesis section. Low effort, high credibility, and increasingly expected by medical-informatics venues.

---

## P2 — Polish

- **P2-1.** Abstract is number-dense; trim ~20% and lead with the contribution and the "tie + calibration-based selection" insight.
- **P2-2.** Episode-count inconsistency: appendix command uses `--fewshot-episodes 40`; code default is 80. Align the documented value with what produced the reported 0.590.
- **P2-3.** Add an event-per-variable / sample-size justification (≈20,297 positive events / 22 features → EPV well above thresholds; state it).
- **P2-4.** Consider a brief decision-curve / net-benefit analysis (clinical utility) — optional but elevates a clinical-prediction paper.
- **P2-5.** Ensure every figure caption is self-contained (defines the metric and that lower is better for Brier/ECE) — mostly done; verify the AUROC bar chart states the test split and n.
- **P2-6.** Confusion-matrix counts (3,091 TP / 1,425 TN) should be checked against the stated recall 0.761 / specificity 0.589 on 4,060 pos / 2,420 neg (they reconcile to ≈3,090 / ≈1,425 — fine; keep the exact numbers from the run JSON).

---

## What I will change directly (source-level, no re-run needed)
Implemented in the thesis source from existing evidence files: P0-1 (CIs + tie statement), P0-2 (prespecified selection wording), P0-3 (label-component note), P0-4 (missingness handling + appendix table), P0-5 (modeling-protocol + hyperparameter appendix), P1-2 (ablation + seed-robustness section/tables), P1-4 (federated detail), P1-5 (few-shot degeneracy), P1-6 (expanded background), P1-7 (canonical-runner footnote), P2-2/P2-3 (consistency + EPV).

## What needs your machine / data (flagged, with ready snippets)
P1-1 (Table 1 demographics) and P1-3 (severity-score baseline) require reading the raw/processed data with a parquet engine and, for P1-3, computing OASIS/SAPS — both blocked in the review sandbox (no parquet engine, no network). Snippets and exact insertion points are provided so you can run them locally and paste the outputs.

## Build note
The Persian thesis requires XeLaTeX + `xepersian` + `bidi`, which are not installable in this review sandbox (no CTAN access). All source edits are made cleanly; rebuild locally with:
`cd thesis && latexmk -xelatex thesis_FULL_COHORT_REVISED.tex` (run twice for refs).
