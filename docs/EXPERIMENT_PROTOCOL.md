# Experiment Protocol

**Project:** Few-Shot Treatment Feasibility Prediction  
**Author:** Mohammad Sharafi  
**Last updated:** 2026-05-25

This document defines the experimental protocol for all thesis experiments. It is intended as the reference for reviewers and committee members to understand how experiments were designed, what was measured, and what the results mean.

---

## 1. Research Questions

| RQ | Question | Addressed by |
|----|---------|-------------|
| RQ1 | Can few-shot learning predict treatment feasibility with competitive AUROC compared to classical tabular models? | Exp1 |
| RQ2 | How sensitive is few-shot performance to the number of support examples (K)? | Exp2 |
| RQ3 | What is the performance cost of federated learning compared to centralized training? | Exp3 |
| RQ4 | Which components of the ETHOS-FewShot architecture contribute most to performance? | Exp4 |
| RQ5 | How well does the proposed model generalize across disease categories? | Exp5 |
| RQ6 | Can a hybrid/stacked pipeline approach the tabular SOTA ceiling? | exp_fewshot_best, exp_ethos_rf_hybrid |

---

## 2. Dataset

| Property | Value |
|---------|-------|
| Source | MIMIC-IV v3.1 (Johnson et al., 2023) |
| Population | Adult ICU patients, age ≥ 18, first ICU stay only, LOS ≥ 4 hours |
| Sample size | 2,000 patients (stratified random from 32,399 eligible) |
| Class balance | ~58% treatment-feasible, ~42% not feasible |
| ICU types | Mixed (MICU, SICU, CCU, TSICU, CSRU, NICU, NEURO) |
| Temporal scope | First 24 hours of ICU admission |

---

## 3. Features

### Laboratory values (15 items, first 24h mean)

| Item ID | Feature | Clinical range used |
|---------|---------|-------------------|
| 50912 | Creatinine | [0.1, 30] mg/dL |
| 50971 | Potassium | [1.5, 10] mEq/L |
| 50983 | Sodium | [100, 180] mEq/L |
| 50902 | Chloride | [70, 130] mEq/L |
| 50931 | Glucose | [20, 600] mg/dL |
| 50893 | Calcium | [4, 16] mg/dL |
| 50960 | Magnesium | [0.5, 6] mg/dL |
| 50811 | Hemoglobin | [3, 25] g/dL |
| 51265 | Platelets | [5, 1000] K/uL |
| 51300 | WBC | [0.5, 50] K/uL |
| 50813 | Lactate | [0.5, 25] mmol/L |
| 50861 | ALT | [5, 2000] U/L |
| 50878 | AST | [5, 2000] U/L |
| 51006 | BUN | [5, 300] mg/dL |
| 50882 | Bicarbonate | [5, 50] mEq/L |

### Vital signs (7 items, first 24h mean)

| Item ID | Feature | Clinical range |
|---------|---------|---------------|
| 220045 | Heart Rate | [20, 250] bpm |
| 220050 | Systolic BP | [40, 250] mmHg |
| 220051 | Diastolic BP | [20, 180] mmHg |
| 220052 | Mean BP | [30, 200] mmHg |
| 223761 | Temperature | [90, 110] °F |
| 220277 | SpO2 | [50, 100] % |
| 220210 | Respiratory Rate | [1, 60] /min |

Values outside clinical ranges are treated as outliers and set to NaN. Missing values are mean-imputed.

### Symptom tokens
Each feature becomes a token: `(symptom_id, intensity, modality_flag, position)` with sequence length 25.

---

## 4. Label Definition

Binary label: **treatment_feasible** (1 = feasible, 0 = not feasible)

**Feasible** conditions (ALL must be true):
1. Discharge location ∈ {HOME, HOME HEALTH CARE, HOME WITH HOME IV PROVIDR, REHAB/DISTINCT PART HOSP, REHAB}
2. ICU LOS ≤ 720 hours (30 days)
3. No 30-day readmission

**Not feasible**: any other discharge (AMA, SNF, hospice, death, etc.)

---

## 5. Data Splits

| Split | Proportion | Notes |
|-------|-----------|-------|
| Train | 64% | used for model training |
| Validation | 16% | early stopping, threshold tuning |
| Test | 20% | held-out, evaluated once |

Stratified by label. Random seed: 42. Split is the same across all experiments.

---

## 6. Evaluation Metrics

| Metric | Why used | Range |
|--------|---------|-------|
| AUROC | Primary; threshold-independent discrimination | [0, 1] higher better |
| AUPRC | Precision-recall; important for imbalanced data | [0, 1] higher better |
| F1-macro | Balanced F1 across both classes | [0, 1] higher better |
| MCC | Matthews Correlation Coefficient; robust to imbalance | [-1, 1] higher better |
| Accuracy | Standard; note class imbalance | [0, 1] |
| Brier Score | Calibration quality | [0, 1] lower better |
| Calibration (ECE) | Expected calibration error | [0, 1] lower better |

**Primary metric for comparison:** AUROC  
**Confidence intervals:** Bootstrap (1,000 samples) at 95% for exp_fewshot_best; not computed for Exp1 episodic protocol.

---

## 7. Baselines

### Classical tabular baselines (Exp1, exp_tabular_sota)

| Model | Hyperparameter tuning |
|-------|----------------------|
| Logistic Regression | L2 regularization, C ∈ {0.01, 0.1, 1, 10} |
| Random Forest | n_estimators, max_depth, min_samples_leaf, max_features |
| XGBoost | n_estimators, max_depth, learning_rate, subsample |
| LightGBM | num_leaves, learning_rate, n_estimators |
| CatBoost | depth, learning_rate, l2_leaf_reg |

All: 5-fold cross-validation on train+val; hyperparameters selected by val AUROC.

### Neural baselines

| Model | Description |
|-------|-------------|
| StandardNN | 2-layer MLP on tabular features; Adam optimizer |
| FederatedNN_NoFewShot | FedAvg-trained MLP, no episodic head |

---

## 8. Few-Shot Protocol

### Architecture: ETHOS + Prototypical Head

```
Input: symptom token sequence (25 × 4)
     ↓
ETHOS Transformer Encoder
  - d_model=128, nhead=8, num_layers=2
  - Positional encoding, dropout=0.2
     ↓
Mean pooling → embedding (128-dim)
     ↓
Prototypical Head
  - L2-normalize embeddings
  - Compute class prototypes from support set
  - Cosine similarity scoring, temperature=10
  - Focal loss (γ=2.0) + pos_weight=1.38
```

### Episodic training
- N-way: 2 (binary: feasible vs not)
- K-shot: 5 (default; sweep in Exp2)
- Query size: 15 per class
- Training episodes: 1,500
- Validation episodes: 100
- Test episodes: 500
- Early stopping patience: 10

### Best few-shot (exp_fewshot_best)
Combines:
1. GFA (Gradient-based Feature Aggregation) attention
2. Cross-attention between support and query
3. Proto-MAML-style adaptation
4. Stacked meta-learner: few-shot probs + RF probs + embeddings → LogReg

---

## 9. Federated Protocol

### Simulation setup
```
5 federated nodes = 5 disease categories:
  node_1: Sepsis (ICD-10: A40, A41, R652)
  node_2: Cardiac (I21, I22, I25, I50, I47, I48, I49)
  node_3: Respiratory (J18, J44, J96, J80, J15)
  node_4: Renal (N17, N18, N19)
  node_5: Diabetes (E10, E11, E13)
```

### Algorithm: FedAvg (McMahan et al., 2017)
- Communication rounds: 10 (planned: 20; limited by compute)
- Local epochs: 3
- Local batch size: 32
- Aggregation: weighted average by node dataset size

### Evaluation
- Centralized: train on full dataset, test on held-out test set
- Federated: FedAvg-trained global model, test on same held-out test set
- Gap = centralized AUROC - federated AUROC

---

## 10. Hybrid Pipeline Protocol

### ETHOS-RF Hybrid (exp_ethos_rf_hybrid)
1. Train ETHOS encoder (supervised on train split)
2. Extract embeddings for all samples
3. Concatenate embeddings with original tabular features
4. Train RF on concatenated feature matrix
5. Evaluate on test split

### Stacked/Fusion Best (exp_fewshot_best, stacked_best)
1. Train ETHOS few-shot model episodically
2. Extract few-shot probability for each test sample
3. Train RF on tabular features → get RF probability
4. Stack: [few_shot_prob, rf_prob, ethos_embedding] → Logistic Regression meta-learner
5. Threshold-tune on validation set
6. Evaluate on test split

---

## 11. Ablation Study (Exp4)

| Variant | What is removed |
|---------|----------------|
| full_model | Baseline (all components) |
| no_symptom_tokens | Raw tabular features instead of token sequences |
| no_clinical_bias | No class-imbalance correction |
| no_intensity_weighting | No intensity weighting in tokenization |
| no_temporal_decay | No temporal decay in tokenization |
| no_few_shot | Supervised head instead of episodic training |

**Important note:** `no_symptom_tokens` achieves higher AUROC than `full_model`. This is expected: removing the token encoding forces the model to rely on raw tabular features, which are naturally well-suited to tree-based models. The ablation confirms that the bottleneck is the episodic few-shot objective over tabular data, not the tokenization step itself.

---

## 12. Cross-Disease Generalization (Exp5)

Protocol:
1. Train proposed model on nodes {1, 2, 3} (train nodes)
2. Test on node_4 (renal) and node_5 (diabetes)
3. Compare against LR and XGBoost trained on same train nodes

**Limitation:** Due to computational constraints, only nodes 4 and 5 were evaluated as test nodes. Full 5-fold leave-one-out generalization is planned for future work.

---

## 13. Statistical Tests

- Confidence intervals: Bootstrap (1,000 resamples, 95% CI) — applied to exp_fewshot_best results
- Significance test: Wilcoxon signed-rank test at α=0.05 (in diagnostics/statistical_validation.py)
- Note: Exp1 CIs are collapsed (no spread in bootstrap for the episodic evaluation protocol); this is a known limitation

---

## 14. Limitations

1. **Single-center data:** MIMIC-IV is from Beth Israel Deaconess Medical Center; results may not generalize to other institutions.
2. **2,000-patient sample:** Full-cohort experiments would provide tighter CIs and possibly higher AUROC.
3. **Tabular data disadvantage for few-shot:** Clinical tabular data strongly favors classical ML (tree-based models). Few-shot learning is most valuable under very limited data, not on 2,000 samples.
4. **Simulated federation:** Federated learning is simulated on a single machine by splitting by disease. Real federation involves communication overhead, non-IID drift, and privacy amplification.
5. **No external validation:** Results should be validated on an external dataset (e.g., eICU, HIRID) before clinical use.
6. **10 federated rounds:** Lower than planned 20; represents lower bound on federated performance.
7. **Incomplete Exp5:** Only 2/5 disease nodes tested.
8. **Label definition:** Treatment feasibility is a proxy label derived from discharge destination, not a direct clinical assessment.

---

*For full results, see `results/KEY_FINDINGS.md` and `docs/FINAL_VALIDATION_REPORT.md`.*  
*For defense preparation, see `docs/DEFENSE_PLAYBOOK_FA.md`.*
