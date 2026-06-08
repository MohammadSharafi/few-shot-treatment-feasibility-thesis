# Final Full Acceptance Audit

**Project:** Few-Shot Treatment Feasibility Prediction
**Thesis title (FA):** فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده
**Pass:** Final full acceptance audit (Pass 4 — strict clean rebuild)
**Date:** 2026-05-25
**Engineer:** AI auditor (final acceptance pass)

This audit was conducted under the explicit rule that the project must be
genuinely reproducible from source. The pass *forced* a clean rebuild
(`make clean-temp && make thesis-fa`) and surfaced a latent submission
blocker that was not detectable from the cached build of earlier passes.

---

## 1. Project folder structure (verified 2026-05-25)

```
Thesis-curser/
├── Makefile
├── README.md
├── config.yaml
├── requirements.txt / requirements-frozen.txt
├── data/processed/                 ← parquet + tokens (raw MIMIC under data/3.1/ not tracked)
├── extraction/                     ← 01_cohort → 05_federated_split (5 canonical files)
├── models/                          ← encoder, baselines, federated, few_shot, joblib + pt artefacts
├── utils/                           ← feature_engineering
├── experiments/                     ← 9 canonical scripts (see §3); archive/ holds 12 historical
├── scripts/                         ← validate_results.py, defense_demo.py, build_thesis_assets.py,
│                                      generate_thesis_figures.py, run_value_experiments.py
├── diagnostics/                     ← performance_breakdown.py, statistical_validation.py
├── results/                         ← CANONICAL source of truth (see §6)
│   ├── CANONICAL_METRICS.json
│   ├── KEY_FINDINGS.md
│   ├── DEFENSE_RUN_SUMMARY.md
│   ├── *.json / *.log              (experiment outputs)
│   ├── figures/  (32 png)
│   ├── tables/   (21 csv/tex)
│   └── archive/thesis_results_duplicate_backup/
├── thesis/
│   ├── thesis_fa.tex               ← Persian main file (canonical)
│   ├── thesis_fa.pdf               (rebuilt clean 2026-05-25 04:43, 256 pages, 1.9 MB)
│   ├── references.bib              (143 keys)
│   ├── chapters_fa/  (27 files: 8 chapters + 19 appendices)
│   ├── frontmatter_fa/ (6 files)
│   ├── tables_demographics.tex, tables_extended.tex, tables_ranges.tex
│   ├── fonts/ (BNazanin)
│   ├── latexmkrc
│   └── defense_slides.tex / .pdf
└── docs/                           ← audits, reports, build notes, playbooks
```

## 2. Thesis source files — sanity check

| File | LoC | Sanity |
|------|----:|--------|
| `thesis_fa.tex` | 379 | engine guard, all packages load in correct order (hyperref → xepersian → listings) |
| `chapters_fa/introduction.tex` | 261 | clean |
| `chapters_fa/background.tex`   | 550 | clean |
| `chapters_fa/methods.tex`      | 704 | clean |
| `chapters_fa/experiments.tex`  | 282 | clean (after §5 fix) |
| `chapters_fa/results.tex`      | 681 | clean (after §5 fix) |
| `chapters_fa/discussion.tex`   | 528 | clean (after §5 fix) |
| `chapters_fa/conclusion.tex`   | 103 | clean (after §5 fix) |
| `chapters_fa/appendix_*.tex`   | 19 files, 45–175 each | clean |
| `references.bib`               | 143 entries | resolves 100% of citations |

## 3. Experiment scripts — canonical set (9)

Top-level `experiments/`:
- `run_all.py`
- `exp1_main.py`
- `exp2_shots.py`
- `exp3_federated.py`
- `exp4_ablation.py`
- `exp5_generalization.py`
- `exp_tabular_sota.py`
- `exp_fewshot_best.py`
- `exp_ethos_rf_hybrid.py`

`experiments/archive/` (12 scripts): `exp1_fewshot_final`, `exp1_fewshot_sweep`,
`exp1_improved`, `exp_baseline_strong`, `exp_ethos_supervised`, `exp_fewshot_v{2..6}`,
`exp_fewshot_v4_stacked`, `exp_improved_fewshot`.

## 4. Result files — verified

All 15 headline-metric values match their source files within `|Δ| < 1e-9`:

| Metric | Verified source | Match |
|--------|----------------|-------|
| `tabular_sota_test_auroc` (0.7475) | `results/tables/exp_final_best_model.csv` | ✓ Δ=0 |
| `stacked_best_auroc` (0.7462) | `results/exp_fewshot_best.json` | ✓ Δ=0 |
| `fewshot_best_auroc` (0.6698) | `results/exp_fewshot_best.json` | ✓ Δ=0 |
| Exp1 RF / LR / XGB / SNN / FedNN / Proposed (6 values) | `results/exp1_main.json` | ✓ all Δ=0 |
| Federated centralized/federated/gap (3 values) | `results/exp3_federated.json` | ✓ all Δ=0 |
| Exp4 full_model / no_symptom_tokens (2 values) | `results/exp4_ablation.json` | ✓ Δ=0 |
| ETHOS-RF hybrid AUROC (0.7173) | `results/tables/exp_ethos_rf_hybrid.csv` | ✓ Δ=0 |

## 5. Canonical metrics file

`results/CANONICAL_METRICS.json` is the **single source of truth**.
- Verified to be the only `CANONICAL_METRICS.json` anywhere outside `.venv`.
- Contains 14 metrics with raw_value, display, source_file, source_key_or_table,
  used_in_thesis, used_in_defense, interpretation.
- Defines rounding convention (3-decimal display) and the *explicit* prohibition of slash-joined double-rounding.

## 6. Figure / table assets

- 32 PNG figures in `results/figures/`
- 21 CSV / TeX tables in `results/tables/`
- Every `\includegraphics` and `\input` target referenced from `thesis/chapters_fa/`
  and `thesis/frontmatter_fa/` resolves (verified by `check_latex_refs`).

## 7. Documentation files

`docs/`: CODE_RUNNING_GUIDE_FA, DEFENSE_PLAYBOOK_FA, EXPERIMENT_PROTOCOL,
FINAL_AGENT_SUMMARY, FINAL_CLEANUP_AUDIT, FINAL_FULL_ACCEPTANCE_AUDIT (this file),
FINAL_FULL_ACCEPTANCE_REPORT, FINAL_MICRO_CLEANUP_REPORT, FINAL_QUALITY_AUDIT_FA,
FINAL_SUBMISSION_CLEANUP_REPORT, FINAL_VALIDATION_REPORT, LATEX_BUILD_NOTES,
PROJECT_AUDIT_REPORT, REPRODUCIBILITY_GUIDE, THESIS_REVIEW_AND_NEXT_STEPS,
THESIS_SUBMISSION_CHECKLIST.

## 8. Defense files

- `docs/DEFENSE_PLAYBOOK_FA.md` — narrative + Q&A safety phrases (honest, canonical)
- `thesis/DEFENSE_QA_FA.md` — Persian Q&A index
- `results/DEFENSE_RUN_SUMMARY.md` — auto-regenerated from `CANONICAL_METRICS.json`
- `scripts/defense_demo.py` — reads canonical JSON, prints + writes summary
- `thesis/defense_slides.tex` / `defense_slides.pdf` — 11-page slide deck

## 9. LaTeX build files

- `thesis/thesis_fa.tex` (main, engine-guard for XeLaTeX)
- `thesis/latexmkrc` — corrected to set `$bibtex_use=2`, `$max_repeat=7`, explicit `$bibtex` command, clean extension list
- Engine: `xelatex` + `bibtex` (via `latexmk -xelatex`)

## 10. Validation scripts

- `scripts/validate_results.py` — 51 strict checks (see §13)

## 11. Duplicated / stale files

| Concern | Status |
|---------|--------|
| `thesis/results/` duplicate folder | RESOLVED (Pass 3) — consolidated into root `results/`; traceability marker present |
| Historical `exp_fewshot_v*.py` at top level | RESOLVED (Pass 2, 3) — 12 historical scripts archived |
| Duplicate `CANONICAL_METRICS.json` | NONE — only at root `results/` |
| Slash-joined double-rounding form (e.g. the deprecated `0.747` vs `0.748` slash notation) | NONE in markdown — validator enforces |
| Old `چندسطحی درمانی` title fragments | NONE in source files |

## 12. Title inconsistencies

ZERO. The canonical Persian title appears identically in:

| Location | Status |
|----------|--------|
| `thesis/frontmatter_fa/cover.tex` | ✓ |
| `thesis/frontmatter_fa/commitment.tex` | ✓ |
| `thesis/thesis_fa.tex` `\title{}` | ✓ |
| `thesis/thesis_fa.tex` `\hypersetup{pdftitle=...}` | ✓ |
| `thesis/thesis_fa.pdf` `/Title` metadata (after clean rebuild) | ✓ |
| `README.md`, defense docs, validation reports | ✓ |

## 13. Metric inconsistencies

ZERO. The validator scans:

- README.md
- every `*.md` under `docs/`
- every `*.md` under `results/`
- every `*.md` under `thesis/`

for the forbidden slash-joined double-rounding form, and verifies that every
defense-relevant display value is present in README.md. All scans pass.

## 14. LaTeX / XePersian issues found and fixed

### 14.1 Critical: clean-rebuild segfault at page 75 (Pass 4 discovery)

When invoking `make clean-temp && make thesis-fa` for the first time, **xetex
segfaulted deterministically at page ≈ 75** of the thesis. The previous
passes were not aware of this because they ran with cached `.aux/.toc/.bbl`
files inherited from an earlier successful manual build. A truly clean build
exposed the bug.

**Root cause:** an `\texttt{path/with/slashes_and_underscores}` construct
inside RTL Persian text — specifically `\texttt{results/tables/exp1\_results.csv}`
on `chapters_fa/experiments.tex` line 142 — corrupts the XePersian +
bidi direction stack and triggers a segmentation fault in xetex. The user's
own LaTeX safety guidance lists exactly this pattern as unsafe:

> For filenames: `\texttt{\lr{exp\_fewshot\_best.py}}`

**Bisect chain:**
- Skipping `experiments..appendix` → builds OK (88 pages).
- Adding `experiments.tex` back → segfault.
- Truncating `experiments.tex` at line 141 → builds OK (94 pages).
- Adding line 142 (the dense mixed-script result-summary line with `\texttt{path/to/file.csv}`) → segfault.
- Replacing only the `\texttt{path}` with `\texttt{\lr{path}}` → builds OK.

**Fix applied:** all `\texttt{X}` whose contents contain a slash or backslash-underscore (and that are not already wrapped by `\lr{}`) are now wrapped: `\texttt{X}` → `\texttt{\lr{X}}`. A total of **176** occurrences were rewritten across 15 files.

### 14.2 latexmkrc was missing critical bibtex settings

Old:
```
$pdf_mode = 5;
$postscript_mode = 0;
$dvi_mode = 0;
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
```

This is incomplete: `latexmk` does not always invoke `bibtex` reliably without `$bibtex_use=2`, and the default `$max_repeat=5` does not always converge for a thesis with this many forward references.

**Fix applied:** updated `thesis/latexmkrc` to set:
- `$bibtex_use = 2;` (force bibtex)
- `$bibtex = 'bibtex %O %S';`
- `$max_repeat = 7;`
- `$warnings_as_errors = 0;`
- `$clean_ext = 'synctex.gz acn acr alg glg glo gls ist nlo nls run.xml';`

### 14.3 Missing-character / U+0654 / kashida warnings

ZERO missing-character warnings in the clean rebuild log. U+0654 (Hamza
above) is already handled by `\newunicodechar{ٔ}{{\ThesisLatinUpright\symbol{"654}}}`
in `thesis_fa.tex` (line 244).

### 14.4 Remaining warnings (cosmetic, documented)

- 7 `LaTeX Warning: 'h' float specifier changed to 'ht'.` (benign LaTeX auto-promotion)
- 4 `Package bidi Warning: Oops! patching \f@nch@hfbox@... failed.` (benign; headers render correctly)
- 25 Overfull `\hbox` (magnitudes 2.93–3.66pt; cosmetic only)
- 22 Underfull `\hbox` (RTL/LTR line justification; cosmetic)

## 15. Reproducibility issues

NONE. The full chain reproduces from source:

```
make clean-temp        # removes aux + PDF
make thesis-fa         # xelatex → bibtex → xelatex (×N until converged) → PDF
make validate          # 51 strict checks
make defense           # CANONICAL_METRICS.json → DEFENSE_RUN_SUMMARY.md
```

All four targets exit 0 with the fixes applied.

## 16. Defense risks

LOW — same five documented limitations from prior passes:

- Ablation paradox (`no_symptom_tokens` > `full_model`) — documented as a scientific finding, not a defect
- Exp5 covers 2/5 disease nodes (computational constraint, declared limitation)
- Exp1 bootstrap CI collapse — canonical CIs come from `exp_fewshot_best.json`
- 2,000-patient subsample (of 32,399 available) — declared limitation
- Single-center data (MIMIC-IV only) — declared limitation

All risks have prepared defense talking points in `docs/DEFENSE_PLAYBOOK_FA.md`.

## 17. Submission blockers

NONE. The single blocker discovered (§14.1) was fixed within this audit.

## 18. Cleanup plan (executed in this pass)

| # | Action | Outcome |
|---|--------|---------|
| 1 | Force-clean rebuild to expose latent bugs | Detected xetex segfault |
| 2 | Bisect xetex crash | Identified `\texttt{path}` pattern as trigger |
| 3 | Apply user-prescribed `\texttt{\lr{...}}` fix globally | 176 wraps across 15 files |
| 4 | Update `latexmkrc` for reliable bibtex + convergence | bibtex now drives bibliography |
| 5 | Verify clean rebuild produces 256-page PDF with canonical metadata | ✓ |
| 6 | Extend `validate_results.py` to check PDF /Title + undefined citations + undefined references | 3 new strict checks |
| 7 | Re-run full sequence (`check / thesis-fa / validate / defense`) | All exit 0 |
| 8 | Document the bug + fix in this audit and in `LATEX_BUILD_NOTES.md` | Done |

## 19. Cleanup conclusion

The project survived a strict clean rebuild. The bug that would have blocked
submission has been fixed in the source files. The validator is now strict
enough to catch the same class of bug if it recurs (undefined refs/citations
become hard failures with non-zero counts).

Single source of truth: `results/CANONICAL_METRICS.json`.
Single canonical thesis title everywhere: `فیوشات درمانی: ...`.
Clean rebuild: 256 pages, 0 errors, 0 missing characters, 0 undefined refs, 0 undefined citations.
Validation score: 100/100 (51 passed, 0 warnings, 0 failed).

The project is genuinely submission-ready.
