# Paper C: Symptom Tokens + hybrid neural--classical fusion

**Draft title:** *Symptom Tokens, Transformer Encoders, and Calibrated Stacking for ICU Treatment Feasibility on MIMIC-IV*

## Contents
- `sections/*.tex`: token formalism (definition), ETHOS + training phases, stacking algorithm, full results tables (tabular SOTA, aug best, few-shot vs stacked, Exp1 excerpt), interpretability, discussion.
- `references.bib` — transformers for EHR, stacking, distillation, TabLLM, etc.

## Build
```bash
cd papers/paper_c_symptom_tokens_hybrid
latexmk -pdf -interaction=nonstopmode main.tex
```
**~7+ pages** with pipeline figure; add SHAP/attention figures from `results/figures/` when available.

**Numbers:** `exp_tabular_sota.csv`, `exp_final_best_model.csv`, `exp_fewshot_best.json`, `exp1_results.csv`.
