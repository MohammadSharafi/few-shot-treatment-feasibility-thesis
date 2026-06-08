# Paper B: Federated few-shot learning for treatment feasibility (MIMIC-IV)

**Draft title:** *Federated Few-Shot Learning for ICU Treatment Feasibility: A Disease-Stratified Simulation on MIMIC-IV*

## Contents
- Modular `sections/*.tex`: abstract, intro, related work, FedAvg preliminaries, methods, experiments, results (incl.\ context table vs.\ tabular ceiling), discussion, conclusion, appendix.
- `references.bib` — FL + medical FL + few-shot basics.

## Build
```bash
cd papers/paper_b_federated_feasibility
latexmk -pdf -interaction=nonstopmode main.tex
```
**~6–7 pages** with schema figure; extend with real multi-site data for venue fit.

**Primary numbers:** `results/tables/exp3_results.csv`.
