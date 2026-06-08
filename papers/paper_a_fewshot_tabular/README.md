# Paper A: Few-shot tabular clinical prediction (MIMIC-IV feasibility)

**Draft title:** *Bridging Few-Shot Meta-Learning and Strong Tabular Baselines for ICU Treatment Feasibility*

## Contents (full manuscript)
- `main.tex` — preamble, title, `\input{sections/...}`, bibliography
- `sections/abstract.tex` … `appendix_algorithms.tex` — IMRaD + algorithms + backmatter
- `references.bib` — expanded citations (FL surveys, distillation, stacking, etc.)
- Tables are **synced to repository CSVs** (`exp1`–`exp5`, `exp_tabular_sota`, `exp_final_best_model`, `exp_fewshot_best` JSON summarized in text)

## Build
```bash
cd papers/paper_a_fewshot_tabular
latexmk -pdf -interaction=nonstopmode main.tex
```
Schema figure is **TikZ** via `\input{../../thesis/latex/mimic_iv_schema.tex}` (no `fig33` PNG needed).

**Before journal submission:** swap in venue `\documentclass` (e.g. `elsarticle`, `IEEEtran`), ORCIDs, CRediT, clinical trial registration N/A, and journal word limits.
