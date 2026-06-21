# Reproducibility Guide

**Project:** Few-Shot Treatment Feasibility Prediction  
**Author:** Mohammad Sharafi  
**Last updated:** 2026-05-25

This guide documents every step required to fully reproduce the thesis results from scratch.

---

## 1. Hardware and Runtime Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| CPU | 4-core | 8-core |
| RAM | 8 GB | 16 GB |
| Disk | 15 GB | 20 GB |
| GPU | Not required | Optional (speeds up ETHOS training) |
| MIMIC-IV raw data | ~9.9 GB | same |
| Runtime (full pipeline) | ~4–8 hours CPU | ~1–2 hours with GPU |

---

## 2. Python Environment Setup

### Python version
```bash
python3 --version   # Should be 3.10, 3.11, or 3.12
```
The project was developed and tested with **Python 3.11**. Earlier 3.10 should work; 3.12+ not tested.

### Create and activate virtual environment
```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or:
.venv\Scripts\activate.bat     # Windows
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify installation
```bash
make check
# or manually:
python -c "import sklearn, torch, pandas, numpy, lightgbm, catboost; print('All OK')"
```

### Freeze environment for submission
```bash
pip freeze > requirements-frozen.txt
```
The committed `requirements-frozen.txt` records the exact environment used to produce the thesis results.

---

## 3. MIMIC-IV Data Access and Placement

### Access requirements
MIMIC-IV requires credentialed access through PhysioNet. Apply at: https://physionet.org/content/mimiciv/

### Download
Download **MIMIC-IV v3.1** from PhysioNet. You need these modules:
- `hosp/` (admissions, patients, labevents, d_labitems)
- `icu/` (icustays, chartevents, d_items)

### Placement
Place the downloaded data under:
```
Thesis-curser/
└── data/
    └── 3.1/
        ├── hosp/
        │   ├── admissions.csv.gz
        │   ├── patients.csv.gz
        │   ├── labevents.csv.gz
        │   └── d_labitems.csv.gz
        └── icu/
            ├── icustays.csv.gz
            ├── chartevents.csv.gz
            └── d_items.csv.gz
```

### Config verification
Check `config.yaml` matches your placement:
```yaml
paths:
  data_root: "data/3.1"
  hosp_dir: "data/3.1/hosp"
  icu_dir: "data/3.1/icu"
  processed_dir: "data/processed"
```

---

## 4. Data Extraction

Run extraction scripts in order:

```bash
# Option 1: Use Make
make extract

# Option 2: Manual step-by-step
python extraction/01_cohort.py          # ~2-5 min
python extraction/02_symptoms.py        # ~5-15 min (depends on chunk_size)
python extraction/03_labels.py          # ~1 min
python extraction/04_tokenize.py        # ~1 min
python extraction/05_federated_split.py # ~1 min
```

### Cohort size control
The default `config.yaml` uses the full eligible cohort:
```yaml
symptoms:
  max_cohort_sample: null
```

The current thesis and manuscript results are based on the full eligible cohort after the study inclusion/exclusion criteria. Historical 2k benchmark outputs are retained only for comparison context.

### Verify extraction
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed_full_cohort/thesis_dataset.parquet')
print(f'Cohort size: {len(df)}')
print(df['feasible'].value_counts())
"
```

Expected output:
```
Cohort size: ~32399
feasible
1    ~20297
0    ~12102
```

---

## 5. Running Experiments

### Option A: Orchestrated pipeline (recommended)

```bash
# Core experiments (assumes processed data exists):
python experiments/run_all.py --skip-extraction

# High-value experiments with quick mode (~20-40 min on CPU):
python -u scripts/run_value_experiments.py --quick --figures

# Full high-value run (hours on CPU):
python -u scripts/run_value_experiments.py --figures
```

### Option B: Individual experiments

```bash
# Exp1: Main comparison (RF, XGBoost, LR, NN, FedNN, Proposed)
python -u experiments/exp1_main.py

# Exp2: K-shot sensitivity
python experiments/exp2_shots.py

# Exp3: Federated vs centralized
python experiments/exp3_federated.py

# Exp4: Ablation study
python experiments/exp4_ablation.py

# Exp5: Cross-disease generalization
python experiments/exp5_generalization.py

# Tabular SOTA (may take 30-60 min)
python -u experiments/exp_tabular_sota.py

# Best few-shot + stacked fusion
python experiments/exp_fewshot_best.py

# ETHOS-RF hybrid
python experiments/exp_ethos_rf_hybrid.py
```

### Using Make
```bash
make experiments          # run all core experiments
make value-experiments    # run high-value experiments (--quick mode)
```

---

## 6. Generating Thesis Assets

```bash
# Option 1: Use Make
make assets

# Option 2: Direct
python scripts/build_thesis_assets.py
```

This regenerates:
- `results/figures/*.png` — all 40 thesis figures
- `results/tables/*.tex` — all LaTeX tables

**Safe to rerun** at any time; does not modify result JSON/CSV files.

---

## 7. Validating Results

```bash
# Option 1: Use Make
make validate

# Option 2: Direct (prints report + writes FINAL_VALIDATION_REPORT.md)
python scripts/validate_results.py --write-report

# Option 3: Basic readiness check (older script)
python results/validate_thesis_readiness.py
```

### What validation checks
1. Canonical AUROC values match JSON/CSV result files
2. README display numbers match canonical values
3. KEY_FINDINGS.md contains expected numbers
4. All 40 figures exist
5. All table CSVs exist
6. Thesis PDF compiled
7. Processed data files present
8. Federated gap < 2%
9. K-shot spread documented

---

## 8. Compiling the Persian Thesis

### Prerequisites
Install XeLaTeX (required; **pdfLaTeX will not work**):
```bash
# macOS
brew install --cask mactex-no-gui   # or full MacTeX
# Ubuntu/Debian
sudo apt-get install texlive-xetex texlive-lang-arabic latexmk
```

### Build
```bash
# Option 1: Use Make
make thesis-fa

# Option 2: Direct
cd thesis && latexmk -xelatex -interaction=nonstopmode thesis_fa.tex

# Option 3: Manual (4 passes for bibliography + cross-refs)
cd thesis
xelatex thesis_fa
bibtex thesis_fa
xelatex thesis_fa
xelatex thesis_fa
```

### Fonts
Persian fonts are in `thesis/fonts/`:
- `BNazanin.ttf` — B Nazanin regular
- `B Nazanin Bold.ttf` — B Nazanin bold

These are loaded by `thesis/thesis_fa_fonts.tex`. No additional font installation needed.

### Expected output
```
thesis/thesis_fa.pdf   # Persian thesis PDF
```

---

## 9. Defense Demo

```bash
# Option 1: Use Make
make defense

# Option 2: Direct
python scripts/defense_demo.py
```

Output: `results/DEFENSE_RUN_SUMMARY.md` — canonical numbers summary for defense day.

---

## 10. Full Reproducibility Pipeline

```bash
# Full sequence from scratch (hours):
make check
make extract          # needs MIMIC-IV credentials
make experiments
make value-experiments
make assets
make validate
make thesis-fa

# Quick verification of existing results:
make check
make validate
make defense
```

---

## 11. Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` from venv |
| `FileNotFoundError: data/3.1/...` | Place MIMIC-IV v3.1 data under `data/3.1/` |
| `FileNotFoundError: data/processed/...` | Run `make extract` |
| XeLaTeX: font not found | Check `thesis/fonts/` contains `BNazanin.ttf` |
| XeLaTeX: `! LaTeX Error: File not found` | Run `make assets` first (figures/tables needed) |
| `catboost_info/` at root | Normal CatBoost artifact; `make clean-temp` removes it |
| Long runtime for exp_tabular_sota.py | Expected (4-phase grid search); use `--quick` flag |
| Few-shot AUROC ~0.55–0.57 | Expected for episodic protocol on tabular data |
| Validation score < 100 | Check specific failures; most are minor rounding differences |

---

## 12. Seed Control

All random seeds are fixed in `config.yaml`:
```yaml
seed: 42
```

Individual scripts read this via `yaml.safe_load(open('config.yaml'))`. If a script does not load config, check that it passes `random_state=42` / `torch.manual_seed(42)` explicitly.

---

## 13. Known Limitations

1. **Sample size:** 2,000-patient sample (out of 32,399 available). Full-cohort run needs `max_cohort_sample: null` in config.
2. **Federated rounds:** 10 communication rounds used (20 planned). Lower-bound performance.
3. **Exp5 completeness:** Cross-disease generalization tested on 2/5 nodes (node_4 renal, node_5 diabetes).
4. **CI in Exp1:** Bootstrap CIs in `exp1_results.csv` are collapsed. Proper CIs are in `exp_fewshot_best.json`.
5. **MIMIC-IV single-center:** Results from one hospital system; external validation needed.
6. **No GPU required:** All experiments can run on CPU, but ETHOS pre-training is significantly faster with GPU.

---

*For further assistance, see `docs/PROJECT_AUDIT_REPORT.md` or `results/experiment_notes.md`.*
