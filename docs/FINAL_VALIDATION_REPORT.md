# Final Validation Report

**Generated:** 2026-05-25 21:05:55
**Script:** `python scripts/validate_results.py --write-report`
**Canonical source:** `results/CANONICAL_METRICS.json`

---

## Canonical Metrics

| Metric | Display | Raw | Source |
|--------|---------|----:|--------|
| tabular_sota_test_auroc | **0.748**  | 0.747453140458301 | `results/tables/exp_final_best_model.csv` |
| stacked_best_auroc | **0.746** [0.699, 0.797] | 0.7461696836813819 | `results/exp_fewshot_best.json` |
| fewshot_best_auroc | **0.670** [0.612, 0.726] | 0.6698307441375437 | `results/exp_fewshot_best.json` |
| exp1_random_forest_auroc | **0.762**  | 0.7623465868071339 | `results/exp1_main.json` |
| exp1_proposed_fewshot_auroc | **0.611**  | 0.6114735688120003 | `results/exp1_main.json` |
| federated_centralized_auroc | **0.552**  | 0.5517179617302008 | `results/exp3_federated.json` |
| federated_federated_auroc | **0.547**  | 0.5468969222961879 | `results/exp3_federated.json` |
| federated_auroc_gap | **0.005**  | 0.004821039434012908 | `results/exp3_federated.json` |
| ethos_rf_hybrid_auroc | **0.717**  | 0.7172651675178481 | `results/tables/exp_ethos_rf_hybrid.csv` |

---

## Passed Checks

- CANONICAL_METRICS.json contains 'metrics' object ✓
- canonical metric 'tabular_sota_test_auroc' = 0.747453 matches recorded 0.747453 ✓ (source: results/tables/exp_final_best_model.csv)
- canonical metric 'stacked_best_auroc' = 0.746170 matches recorded 0.746170 ✓ (source: results/exp_fewshot_best.json)
- canonical metric 'fewshot_best_auroc' = 0.669831 matches recorded 0.669831 ✓ (source: results/exp_fewshot_best.json)
- canonical metric 'exp1_random_forest_auroc' = 0.762347 matches recorded 0.762347 ✓ (source: results/exp1_main.json)
- canonical metric 'exp1_logreg_auroc' = 0.723736 matches recorded 0.723736 ✓ (source: results/exp1_main.json)
- canonical metric 'exp1_xgboost_auroc' = 0.694003 matches recorded 0.694003 ✓ (source: results/exp1_main.json)
- canonical metric 'exp1_proposed_fewshot_auroc' = 0.611474 matches recorded 0.611474 ✓ (source: results/exp1_main.json)
- canonical metric 'exp1_standardnn_auroc' = 0.651595 matches recorded 0.651595 ✓ (source: results/exp1_main.json)
- canonical metric 'exp1_federatednn_no_fewshot_auroc' = 0.662210 matches recorded 0.662210 ✓ (source: results/exp1_main.json)
- canonical metric 'federated_centralized_auroc' = 0.551718 matches recorded 0.551718 ✓ (source: results/exp3_federated.json)
- canonical metric 'federated_federated_auroc' = 0.546897 matches recorded 0.546897 ✓ (source: results/exp3_federated.json)
- canonical metric 'federated_auroc_gap' = 0.004821 matches recorded 0.004821 ✓ (source: results/exp3_federated.json)
- canonical metric 'ethos_rf_hybrid_auroc' = 0.717265 matches recorded 0.717265 ✓ (source: results/tables/exp_ethos_rf_hybrid.csv)
- canonical metric 'exp4_full_model_auroc' = 0.583495 matches recorded 0.583495 ✓ (source: results/exp4_ablation.json)
- canonical metric 'exp4_no_symptom_tokens_auroc' = 0.723816 matches recorded 0.723816 ✓ (source: results/exp4_ablation.json)
- README contains defense metric 'tabular_sota_test_auroc' display '0.748' ✓
- README contains defense metric 'stacked_best_auroc' display '0.746' ✓
- README contains defense metric 'fewshot_best_auroc' display '0.670' ✓
- README contains defense metric 'exp1_random_forest_auroc' display '0.762' ✓
- README contains defense metric 'exp1_proposed_fewshot_auroc' display '0.611' ✓
- README contains defense metric 'federated_centralized_auroc' display '0.552' ✓
- README contains defense metric 'federated_federated_auroc' display '0.547' ✓
- README contains defense metric 'federated_auroc_gap' display '0.005' ✓
- README contains defense metric 'exp4_no_symptom_tokens_auroc' display '0.724' ✓
- KEY_FINDINGS contains '0.762' (exp1_random_forest_auroc) ✓
- KEY_FINDINGS contains '0.611' (exp1_proposed_fewshot_auroc) ✓
- proposal alignment: proposal-alignment subsection present ✓
- proposal alignment: proposal-alignment table label present ✓
- proposal alignment: coverage level column present ✓
- proposal alignment: defense note column present ✓
- proposal alignment: honest scoping explanation present ✓
- thesis/frontmatter_fa/cover.tex: contains canonical title key 'فیوشات درمانی' ✓
- thesis/frontmatter_fa/commitment.tex: contains canonical title key 'فیوشات درمانی' ✓
- thesis_fa.tex \title{} uses canonical title ✓
- thesis_fa.tex \hypersetup{pdftitle=...} uses canonical title ✓
- All LaTeX \includegraphics targets resolved ✓
- All LaTeX \input/\include targets resolved ✓
- experiments/ contains only the 9 canonical scripts ✓
- thesis/results/ is not a duplicate result folder ✓
- CANONICAL_METRICS.json exists only at results/ (single source of truth) ✓
- thesis_fa.log contains no LaTeX errors ✓
- thesis_fa.log: no missing-character warnings ✓
- thesis_fa.log: no undefined citations ✓
- thesis_fa.log: no undefined references ✓
- thesis_fa.log: 37 Overfull hbox warnings (within cosmetic tolerance)
- PDF /Title exactly matches the official Persian title ✓
- artifact exists: thesis/thesis_fa.pdf ✓
- artifact exists: thesis/defense_slides.pdf ✓
- artifact exists: data/processed/thesis_dataset.parquet ✓
- artifact exists: data/processed/tokens.npy ✓
- artifact exists: results/figures/fig1_main_comparison.png ✓
- artifact exists: results/figures/fig4_federated.png ✓
- artifact exists: results/tables/exp1_results.csv ✓
- artifact exists: results/tables/ALL_TABLES.tex ✓

## Warnings

- **WARN:** experiments/archive/ exists but contains no .py files

## Failed Checks

- None

## LaTeX Build Log Analysis

Source: `thesis/thesis_fa.log` (last `make thesis-fa` run).

| Category | Count |
|----------|------:|
| LaTeX `! ` errors | 0 |
| Missing character warnings | 0 |
| Undefined citations | 0 |
| Undefined references | 0 |
| LaTeX Warning lines | 7 |
| Package warnings | 4 |
| Overfull hbox/vbox | 37 |
| Underfull hbox/vbox | 41 |

All zero errors / missing characters means the build is clean. Overfull/Underfull hbox messages with magnitudes ≤ 5pt are cosmetic in XePersian RTL typesetting.

---

---

## Final Readiness Score: **99/100**

- Passed: 55
- Warnings: 1
- Failed: 0

## Final Recommendation

- Submit after minor fixes: review warnings and confirm each is acceptable.
