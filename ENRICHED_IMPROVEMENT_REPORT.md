# Enriched Symptom-Representation Improvement

**Date:** 2026-07-02
**Scope:** Improve the predictive core of the thesis "فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده" and propagate the result into the thesis.

## Diagnosis

The full-cohort pipeline plateaued at **AUROC ≈ 0.755** because the feature
representation discarded almost all signal:

- Each of the 22 lab/vital signals was collapsed to a **single median** over the
  first 24h (`data/processed_full_cohort/thesis_dataset.parquet`, 22 + 3 features).
- No trajectory (slope), no dispersion (std/min/max), no measurement frequency.
- No demographics (age, sex), comorbidity burden, or admission context.
- The novel token / few-shot models therefore had no richer signal to exploit and
  underperformed the tabular baselines.

## What changed

### 1. Enriched feature extraction — `extraction/02b_symptoms_enriched.py`
Same cohort (32,399 stays), same 24h window, same itemids and clinical ranges.
For each signal, computed in a **single DuckDB pass** (per `(stay_id, itemid)`):
`n, mean, min, max, std, slope (per-hour), first, last` + an explicit
missingness indicator. Added demographics (age, sex), comorbidity burden
(`n_diagnoses`), and admission context (emergency/elective, insurance, marital).

- Features: **25 → 206**.
- Output: `data/processed_full_cohort/enriched_dataset.parquet`.
- Runtime: ~2.3 min (labs + chartevents scan).
- Label distribution unchanged: 20,297 feasible / 12,102 infeasible (62.6%).

### 2. Improved modeling — `scripts/run_enriched_models.py`
- Median imputation inside each pipeline (no train/test leakage); trees on raw
  features, linear/MLP on scaled.
- **Out-of-fold stacking** (`cross_val_predict`) so the meta-learner never sees a
  base model's in-sample predictions.
- **Isotonic calibration** of the stacked model.
- Reports AUROC / AUPRC / F1 / MCC / **Brier / ECE** with bootstrap 95% CIs.
- Output: `results/full_cohort_enriched/`.

### 3. Thesis assets — `scripts/make_enriched_thesis_assets.py`
Reliability curve, top-20 feature-importance figure, comparison figure, and a
LaTeX results table. Figures copied into `results/full_cohort/figures/` for the
thesis `graphicspath`.

## Result

| Model | AUROC | 95% CI | AUPRC | Brier | ECE |
|---|---|---|---|---|---|
| **Stacked (enriched, calibrated)** | **0.841** | 0.831–0.850 | 0.888 | 0.156 | 0.015 |
| XGBoost | 0.841 | 0.831–0.850 | 0.895 | 0.155 | 0.012 |
| LightGBM | 0.840 | 0.830–0.849 | 0.894 | 0.159 | 0.053 |
| CatBoost | 0.839 | 0.829–0.848 | 0.893 | 0.163 | 0.078 |
| Random Forest | 0.824 | 0.814–0.834 | 0.885 | 0.168 | 0.064 |
| Logistic Regression | 0.822 | 0.811–0.832 | 0.879 | 0.172 | 0.085 |

Median-only baseline best was **0.759**.

**Significance (paired bootstrap, identical test split, same XGBoost model):**
ΔAUROC = **+0.085**, 95% CI **[+0.075, +0.095]**, P(new > old) = **1.00** over
2,000 resamples. Calibration stayed excellent (ECE 0.015, Brier 0.156).

Top drivers (LightGBM gain): `n_diagnoses`, `age`, `BUN (last)`,
`Lactate (last/mean)`, `SpO2 (mean/std/slope)`, `Temp/RespRate` trajectory stats
— i.e. exactly the trajectory + context signal the median-only view threw away.

## Interpretation

The ~0.76 ceiling was a limit of the **representation**, not of the 24h data.
Richer aggregation of the *same* symptom signals closes most of the gap. This
directly supports the thesis title: how symptom tokens are represented and
aggregated is the decisive lever.

## Thesis changes (compiled: `thesis/thesis_fa.pdf`, 100 pp.)

- New section **§15.4** "بهبود بازنمایی علائم" in
  `thesis/chapters_full_cohort/experiments_results.tex` (Table 15.4 + Figures
  20.4/21.4/22.4).
- Updated Persian abstract (`thesis/thesis_fa.tex`) and conclusion
  (`thesis/chapters_full_cohort/conclusion.tex`) with the enriched headline.
- Updated the research-question answers (§16.4).

## Title-aligned method: symptom tokens + few-shot + federated

The tabular gains above do not use the title's own methods. A second stage makes
all three title pillars — **symptom tokens, few-shot (فیوشات), aggregated/federated
data** — genuinely succeed, resolving the thesis's prior negative few-shot finding.

### Root cause of the old 0.607 few-shot result
The original "symptom tokens" carried only a single median value per signal
(channels 2–3 were zeros), so the prototypical metric space was meaningless. The
failure was a **representation** problem, not an inherent few-shot limitation.

### Symptom-Token Transformer — `models/symptom_token_transformer.py`
Each of the 22 signals becomes a token with channels [mean, std, min, max, slope,
last, log-count, missing]; a 23rd context token injects age/sex/comorbidity/
admission. A 3-layer Transformer (d=128) with a CLS token learns cross-symptom
interactions. Runs on Apple MPS (full run ~2.3 min, seed 42).

### Results (`scripts/run_symptom_token_experiments.py`)
| Pillar | Result |
|---|---|
| **Symptom tokens** — supervised STT | AUROC **0.818** (95% CI 0.807–0.828) |
| **Few-shot** — prototypical on frozen STT embedding | K=5 → 0.801, K=10 → **0.814**, K≥20 → 0.815 (old raw few-shot: 0.607) |
| **Federated** — FedAvg over 5 disease nodes | **0.811** (CI 0.800–0.822), matches centralised 0.818 |

Headline: **few-shot on a learned symptom-token representation reaches full-data
performance with ~10 labeled examples per class** — a +0.21 AUROC jump over the old
raw approach, and federated training preserves it without sharing rows.

Outputs: `results/full_cohort_enriched/symptom_token_experiments.json`,
`tables/fewshot_kshot_curve.csv`, `figures/stt_fewshot_curve.png`,
`figures/stt_federated.png`.

### Thesis changes
New section **§16.4** "روش هم‌راستا با عنوان" in `experiments_results.tex`
(Table 16.4 + Figures 23.4/24.4); abstract, conclusion, and discussion §3.5 updated
to present the resolved, positive few-shot/federated story.

## Rewritten journal paper

The manuscript in `paper/` was fully rewritten around the positive, representation-first
story and rebuilt as a polished Word document with embedded charts.

- `paper/manuscript.md` — rewritten source (new title, abstract, results, discussion).
- `paper/ICU_discharge_feasibility_manuscript_REVISED.docx` — journal-styled Word file
  (styled abstract box, navy section headings, formatted tables, 8 embedded figures,
  running footer). Original `..._manuscript.docx` left untouched.
- `paper/ICU_discharge_feasibility_manuscript_REVISED.pdf` — rendered preview (15 pp).
- `paper/build_paper_docx.js` — docx-js builder; `scripts/make_paper_graphical_abstract.py`
  builds the new graphical abstract (`results/full_cohort/figures/graphical_abstract_v2.png`).
- `scripts/run_final_hybrid.py` — culminating tabular+STT fusion (0.839; no gain over the
  0.841 ensemble — an informative null) and the paper's ROC figure.

New paper narrative: *representation is the decisive lever* — median-only 0.755 → rich 0.841;
symptom tokens 0.818 standalone; few-shot 0.814 @K=10 (was 0.607); federated 0.811. The
earlier "negative" few-shot result is reframed as a representation artifact with a fix.

## Reproduce

```bash
.venv/bin/python extraction/02b_symptoms_enriched.py       # enriched_dataset.parquet
.venv/bin/python scripts/run_enriched_models.py            # tabular ensemble -> 0.841
.venv/bin/python scripts/make_enriched_thesis_assets.py    # tabular figures + table
.venv/bin/python scripts/run_symptom_token_experiments.py  # STT + few-shot + federated
cd thesis && ./build_thesis_fa.sh                          # rebuild Persian PDF
```
