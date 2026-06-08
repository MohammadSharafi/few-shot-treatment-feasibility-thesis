# Thesis & experiments — review, gaps, and what to add

*(Assumes “tizzes” → thesis / course project work.)*

## 1. Internal consistency to fix

| Issue | Why it matters |
|--------|----------------|
| **`results/improvement_log.md` vs thesis / `KEY_FINDINGS.md`** | **Done:** `improvement_log.md` now has a banner + pointers to `KEY_FINDINGS.md`, `exp_fewshot_best.json`, and Exp1 vs few-shot-best protocol distinction. Keep regenning `KEY_FINDINGS.md` after experiments. |
| **Episodic K-shot metrics (~0.51–0.55) vs full-model AUROC (~0.71+)** | These are **different evaluation protocols** (episodes vs full test set). State this clearly in the thesis so “low K-shot numbers” are not read as the main model quality. |
| **Ablation row “full_model 0.591”** | That is an ablation configuration, not your best pipeline. Always pair it with **best tuned** few-shot / stacked numbers in the same table or paragraph. |

## 2. What to improve (method + writing)

1. **One clear “reporting hierarchy”:** (a) classical **ceiling** (RF / XGB tuned), (b) **hybrid** (ETHOS + tabular), (c) **few-shot best** (V5 / `exp_fewshot_best`), (d) **federated** cost. Same splits, same test set everywhere.
2. **Calibration + fairness:** report **ECE/Brier** and **AUROC by age group and sex** (you already discuss this; add a small table if you have the numbers).
3. **Uncertainty:** if you use conformal sets (V6), one paragraph on **when the model abstains** is high value for clinical readers.
4. **Limitations paragraph:** single-center MIMIC, 2k sample, label definition; **external validation** as future work (eicU, etc.).

## 3. What *additional data* (not “shadow”) helps most

“Shadow data” is often read as **extra signals** or **unlabeled** data — below is what actually raises **accuracy** and **thesis value**:

| Addition | Expected effect | Effort |
|----------|------------------|--------|
| **More patients** (increase `max_cohort_sample` or full MIMIC-IV slice) | Better generalization, more stable AUROC/CIs | High (compute + extraction time) |
| **Temporal features** (e.g. slope of creatinine / lactate over 0–24h, not only median) | Often +0.01–0.03 AUROC on trajectory tasks | Medium |
| **Extra structured fields** (e.g. vasopressor use, ventilation flags, GCS if available) | Stronger clinical signal for discharge | Medium (SQL + cleaning) |
| **ICD comorbidity count / Charlson** | Cheap tabular boost | Low–medium |
| **External cohort** (eICU, etc.) | Thesis *value* (generalization), not only higher AUROC on MIMIC | High |

**Privacy note:** do not add identifiable data; stay within PhysioNet agreements.

## 4. Which models to run (priority order)

**For maximum accuracy on *your current label and features*:**

1. **`experiments/exp_tabular_sota.py`** — tuned RF / XGB / LightGBM / CatBoost + threshold tuning → **empirical ceiling** on this cohort.
2. **`experiments/exp_ethos_rf_hybrid.py`** — strong **hybrid** story (learned embeddings + tabular).
3. **`experiments/exp_fewshot_best.py`** — pushes few-shot stack toward the ceiling (`--quick` for smoke test, full run for thesis numbers).

**Supporting (smaller lift on AUROC, high value for the write-up):**

4. `experiments/exp1_main.py`, `exp2_shots.py`, `exp3_federated.py`, `exp4_ablation.py`, `exp5_generalization.py` — completeness.
5. **`diagnostics/statistical_validation.py`** — bootstrap CIs / comparison sanity checks.
6. **`diagnostics/performance_breakdown.py`** — per-node / calibration view.

**Optional (only if you need a specific chapter claim):** `exp_fewshot_v5.py` / `v6.py` — large runtime; prefer **`exp_fewshot_best.py`** as the unified “best few-shot” script unless you must reproduce V5/V6 exactly.

## 5. What we cannot promise

- AUROC **above ~0.76–0.78** on this task often hits **label noise + missing social/context factors** (not in labs/vitals alone).
- Running **all** `exp_fewshot_v*.py` variants is usually **redundant**; pick **tabular SOTA + one best few-shot pipeline**.

---

Use **`python scripts/run_value_experiments.py`** (see repo root) for an ordered, configurable run.
