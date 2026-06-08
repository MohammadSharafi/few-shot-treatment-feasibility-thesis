# Few-Shot Treatment Feasibility Prediction

Intelligent prediction of treatment feasibility with Symptom Tokens from aggregated clinical data (MIMIC-IV).

**Official Persian title:** فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده

In this project, **treatment feasibility** is an operational research label derived from early ICU signals and discharge/readmission outcomes. It is not autonomous treatment recommendation, automated prescription, or a replacement for physician judgment.

**Author:** Mohammad Sharafi  
**Supervisor:** Dr. Alimohammadzadeh  
**Institution:** Islamic Azad University, Tehran North Branch  

## Thesis: review & high-value experiments

- **What to improve, what data to add, which models to run:** see [`docs/THESIS_REVIEW_AND_NEXT_STEPS.md`](docs/THESIS_REVIEW_AND_NEXT_STEPS.md).
- **Conflicting numbers?** Treat [`results/KEY_FINDINGS.md`](results/KEY_FINDINGS.md) + [`results/tables/`](results/tables/) as canonical; [`results/improvement_log.md`](results/improvement_log.md) is historical context only (see banner there).
- **Before submission:** [`docs/THESIS_SUBMISSION_CHECKLIST.md`](docs/THESIS_SUBMISSION_CHECKLIST.md) + frozen env [`requirements-frozen.txt`](requirements-frozen.txt).
- **Ordered run (ceiling → hybrid → core → diagnostics → best few-shot):**
  ```bash
  python scripts/run_value_experiments.py --figures
  python scripts/run_value_experiments.py --quick --figures   # faster few-shot; also skips slow ETHOS block in diagnostics
  # CPU-friendly full pass (skip long exp1_main + SOTA tuning):
  python scripts/run_value_experiments.py --quick --figures --skip-sota --skip-exp1
  # Full Exp1 (hours on CPU; log: results/exp1_full_run.log):
  PYTHONUNBUFFERED=1 python -u experiments/exp1_main.py
  ```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract data (requires MIMIC-IV credentials; place v3.1 under data/3.1/)
python extraction/01_cohort.py
python extraction/02_symptoms.py
python extraction/03_labels.py
python extraction/04_tokenize.py
python extraction/05_federated_split.py

# 3. Run core experiments (or use the orchestrator)
python experiments/run_all.py --skip-extraction   # if processed data exists

# 4. Generate all thesis figures + tables (always safe to re-run)
python scripts/build_thesis_assets.py

# 5. Validate thesis readiness (figures, paths)
python results/validate_thesis_readiness.py
```

### Orchestrator options

| Command | Purpose |
|--------|---------|
| `python experiments/run_all.py` | Extraction + core experiments + figure build |
| `python experiments/run_all.py --skip-extraction` | Skip 01–05; run experiments + figures |
| `python experiments/run_all.py --figures-only` | Only `build_thesis_assets.py` |
| `python experiments/run_all.py --with-fewshot` | Also run `exp_fewshot*.py` (long) |

## Reproducibility

- **Random seed:** 42 (fixed in `config.yaml`)
- **Data splits:** 64% train, 16% validation, 20% test (stratified)
- **Environment:** Python 3.10+, PyTorch 2.x, scikit-learn 1.3+

See `thesis/chapters/appendix_i.tex` for the full reproducibility checklist.

## Thesis

- **English (pdfLaTeX):** `thesis/thesis.tex` — compile from `thesis/` with `pdflatex` + `bibtex`; ensure `results/figures/` exists (`scripts/build_thesis_assets.py`).
- **Persian (XeLaTeX):** `thesis/thesis_fa.tex` — use **XeLaTeX only**; see `thesis/README_COMPILE_FA.md` or `./thesis/build_thesis_fa.sh`.
- Defense slides: `thesis/defense_slides.tex`
- Q&A: `thesis/DEFENSE_QA.md`
- Persian defense playbook: `docs/DEFENSE_PLAYBOOK_FA.md`
- Persian code-running guide: `docs/CODE_RUNNING_GUIDE_FA.md`
- Final Persian quality audit: `docs/FINAL_QUALITY_AUDIT_FA.md`
- Defense walkthrough notebook: `notebooks/thesis_defense_walkthrough.ipynb`
- One-command defense demo: `python scripts/defense_demo.py`
- Final 10/10 thesis review: `docs/FINAL_10_OUT_OF_10_THESIS_REVIEW.md`

Final acceptance sequence from project root:

```bash
make clean-temp
make thesis-fa
make validate
make defense
```

## Key Results (frozen submission metrics, 2k cohort)

> **Canonical source:** [`results/CANONICAL_METRICS.json`](results/CANONICAL_METRICS.json) is the single source of truth for every headline number below. The defense demo (`scripts/defense_demo.py`) and the validator (`scripts/validate_results.py`) both read this file.


| Model | AUROC | Notes |
|-------|-------|--------|
| **Tabular SOTA** (dedicated pipeline) | **0.748** | Ceiling for frozen thesis tables |
| **Stacked BEST** (`exp_fewshot_best`) | **0.746** [0.699, 0.797] | Few-shot + classical fusion |
| **Few-shot BEST** (GFA + cross-attention) | **0.670** [0.612, 0.726] | Without stacking |
| **Exp1** `Proposed_FewShot` vs RF (same split) | **0.611** vs **0.762** | Script comparison |
| V-campaign (e.g. RF tuned, Stacked V4, V5 best seed) | 0.760 / 0.732 / 0.737 | Historical table in thesis |
| Federated vs centralized (reported experiment) | **~0.005** AUROC gap (centralized 0.552 → federated 0.547) | Small cost in simulation |
| Ablation paradox (`no_symptom_tokens`) | 0.724 vs full_model 0.583 | Documented in `docs/PROJECT_AUDIT_REPORT.md` §13 — tabular features dominate over episodic embeddings; motivates the hybrid pipeline |

Protocol note: Exp1 RandomForest `0.762` and Tabular SOTA `0.748` are protocol-specific results from different scripts. They should not be read as a contradiction.

Full progression (V1→V6) and appendices are in the thesis PDF; cite primary tables first.

## Project layout

- `extraction/` — MIMIC-IV cohort and features  
- `experiments/` — baselines, federated, few-shot variants  
- `models/` — ETHOS encoder, federated, few-shot heads  
- `results/figures/` — thesis figures (generated, not committed required)  
- `scripts/build_thesis_assets.py` — one-shot figure + table regeneration  
