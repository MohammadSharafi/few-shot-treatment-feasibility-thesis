# QC Report — Persian Master's Thesis
Generated: 2026-06-28 11:47
Source of Truth: `results/CANONICAL_METRICS_NO_GAP_FINAL.json`

## Dataset
- Total patients: **32,399**
- Label=1 (feasible): **20,297** (62.6%)
- Label=0 (infeasible): **12,102** (37.4%)
- Missingness (pre-imputation): **25.2%**

## Train/Val/Test Split (seed=42, stratified by label)
- Train: **20,735** (64.0%)
- Val:   **5,184** (16.0%)
- Test:  **6,480** (20.0%)

## Model Performance (No-Gap Evidence — sorted by AUROC)
| Model | AUROC | AUPRC | F1 | Brier |
|-------|-------|-------|-----|-------|
| Stacked Validation Meta-LR | 0.7590 | 0.8210 | 0.7740 | 0.1870 |
| Weighted Average Ensemble | 0.7590 | 0.8240 | 0.7720 | 0.1880 |
| Token Hybrid CatBoost | 0.7570 | 0.8200 | 0.7430 | 0.1890 |
| CatBoost (tabular) | 0.7560 | 0.8190 | — | 0.1890 |
| Random Forest (tabular) | 0.7560 | 0.8230 | — | 0.1890 |
| ETHOS Embedding + XGBoost | 0.7560 | 0.8220 | — | — |
| XGBoost (tabular) | 0.7550 | 0.8180 | — | — |
| ETHOS Embedding + CatBoost | 0.7550 | 0.8180 | — | — |
| Denoising Autoencoder Pretraining + XGBoost | 0.7550 | 0.8190 | — | — |
| Token Hybrid XGBoost | 0.7540 | 0.8160 | — | — |
| LightGBM (tabular) | 0.7540 | 0.8170 | — | — |
| Neural Baseline (MLP) | 0.7490 | 0.8150 | — | — |
| Simulated Federated LR (weighted nodes) | 0.7290 | 0.7960 | — | — |
| Logistic Regression | 0.7280 | 0.7960 | — | — |
| Token CNN (supervised end-to-end) | 0.6810 | 0.7590 | — | — |
| ETHOS BCE + Prototypical (best deep token) | 0.6700 | 0.7490 | — | — |
| ETHOS Transformer + BCE (supervised) | 0.6680 | 0.7480 | — | — |
| Few-Shot ETHOS (episodic, current baseline) | 0.6070 | 0.6990 | — | — |

## Thesis Anchor Numbers (verified against canonical JSON)
| Number | Value | Chapter |
|--------|-------|---------|
| Total cohort | 32,399 | methods, introduction, background, data_analysis |
| Train | 20,735 | methods, experiments |
| Val | 5,184 | methods, experiments |
| Test | 6,480 | methods, experiments |
| Feasible rate | 62.6% | data_analysis |
| Stacked Meta-LR AUROC | **0.759** | results, discussion, abstract |
| Token Hybrid CatBoost AUROC | **0.757** | results, discussion, abstract |
| Best tabular (CatBoost) AUROC | **0.756** | results, discussion |
| ETHOS Few-Shot AUROC | **0.607** | results, discussion, abstract |
| Federated AUROC cost | ~0.005 | results, discussion |

## Files Modified
| File | Change |
|------|--------|
| `thesis/thesis_fa_fonts.tex` | Latin/mono: macOS conditional + TeX Gyre Termes/Cursor fallback |
| `thesis/chapters_fa/methods.tex` | n=2000→32399 (×5), split 1280/320/400→20735/5184/6480 (×3), pseudocode |
| `thesis/chapters_fa/introduction.tex` | n=2000→32399 (×3), added split sizes to RQ answer |
| `thesis/chapters_fa/background.tex` | n=2000→32399 (×3), low-sample regime discussion |
| `thesis/chapters_fa/discussion.tex` | AUROC numbers (×12+), comparison table rebuilt, chapter summary |
| `thesis/chapters_fa/experiments.tex` | Cohort + split sizes, power analysis for n=6480 test set |
| `thesis/chapters_fa/data_analysis.tex` | Patient table: 32399 total, 20297 feasible (62.6%) |
| `thesis/chapters_fa/results.tex` | Chapter summary AUROCs, learning curve annotation |
| `thesis/chapters_fa/appendix_f.tex` | Week-7 milestone text |
| `thesis/chapters_fa/appendix_h.tex` | Threat-of-validity table (sample size + class imbalance rows) |
| `thesis/frontmatter_fa/abstract_en.tex` | Results paragraph: all AUROCs replaced with no-gap values |
| `thesis/defense_slides.tex` | Data bullet + contribution bullet |

## QC Checklist
- [x] `grep -r 'pnum{2,000}' chapters_fa/` → 0 matches
- [x] `grep -r 'n=2000' chapters_fa/` → 0 matches
- [x] `grep -r 'pnum{1,280}' chapters_fa/` → 0 matches
- [x] Old split sizes (1280/320/400) → replaced with 20735/5184/6480
- [x] AUROC 0.759/0.757/0.607 consistent across chapters and abstract
- [x] Label distribution: 62.6% feasible (was 42%) in data_analysis.tex
- [x] Font file portable — TeX Gyre Termes/Cursor fallback for non-macOS
- [x] Defense slides updated to 32,399
- [ ] XeLaTeX build — run on Mac (sandbox missing `algorithms` package)
- [ ] Visual PDF review: figures, RTL alignment, table layout

## Build on Mac
```bash
cd ~/Programming/Thesis-curser/thesis
bash build_thesis_fa.sh   # 3 XeLaTeX passes + BibTeX
open thesis_fa.pdf
```

## Critical Spot-Checks in Built PDF
1. methods §2.1: '۳۲٬۳۹۹ بیمار'
2. methods §2.4 split table: ۲۰٬۷۳۵ / ۵٬۱۸۴ / ۶٬۴۸۰
3. data_analysis demographics table: Total = 32,399, Feasible = 20,297 (62.6%)
4. results chapter summary: AUROC 0.759 / 0.757 / 0.607
5. discussion comparison table: Token Hybrid 0.757, Meta-LR 0.759, ETHOS 0.607
6. English abstract first paragraph: '32,399-patient MIMIC-IV v3.1 cohort'
7. defense_slides slide 1: '32,399 patients (complete cohort)'