# Q1-style journal papers (LaTeX) — full drafts from the thesis

Three **expanded, standalone** article manuscripts. Each folder is self-contained: `main.tex`, `sections/` (modular body), `references.bib`, `README.md`, and compiles to `main.pdf`.

| Folder | Topic | Approx.\ length |
|--------|--------|-----------------|
| `paper_a_fewshot_tabular` | Few-shot vs tabular baselines; Exp1–5; reporting hierarchy; methods + results tables from CSVs | **~11 pages** |
| `paper_b_federated_feasibility` | FedAvg simulation; disease nodes; Exp3; context vs tabular/stacked ceilings | **~6 pages** |
| `paper_c_symptom_tokens_hybrid` | Symptom Tokens definition; ETHOS; stacking; tabular + hybrid results | **~7 pages** |

## Build all
```bash
for d in paper_a_fewshot_tabular paper_b_federated_feasibility paper_c_symptom_tokens_hybrid; do
  (cd papers/$d && latexmk -pdf -interaction=nonstopmode main.tex)
done
```

## Figures
Schema figure: **TikZ** source `thesis/latex/mimic_iv_schema.tex` (monochrome, journal-style). Papers `\input` it; no matplotlib `fig33` PNG required for that diagram.

## What “complete” means here
- Full **IMRaD** structure, equations, algorithms, **numeric tables tied to `results/tables/*.csv`**, bibliography, data/code statements, author contributions.
- **Not** venue-specific: you still swap in the journal LaTeX class, final author list, declarations, and any required **Supplementary Information** PDF (e.g. full hyperparameter grids copied from thesis appendices).

## Suggested next steps for real submission
1. Pick target journal → apply their `\documentclass` and citation style.  
2. Add **Supplementary**: extended tables, full training curves, SQL extraction, compute details.  
3. Run **paired significance** tests for any non-inferiority claim.  
4. Add **PRISMA-style** reporting if systematic; register **prospective** studies separately.
