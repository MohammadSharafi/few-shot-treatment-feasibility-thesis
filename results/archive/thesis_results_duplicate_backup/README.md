# `thesis_results_duplicate_backup/` — traceability note

## What was here

Prior to the final micro-cleanup pass (2026-05-25), the repository contained a
duplicate results folder at `thesis/results/`. It mirrored the canonical root
`results/` folder almost byte-for-byte — including:

- `CANONICAL_METRICS.json` (identical to root)
- `KEY_FINDINGS.md` (identical to root)
- `DEFENSE_RUN_SUMMARY.md` (identical to root)
- `figures/` (identical listing, identical files)
- `tables/` (identical listing, identical files)
- Various JSON / log / markdown experiment outputs

This duplication was the result of a historical workflow in which some scripts
wrote into `thesis/results/` so that XeLaTeX (via the multi-path
`\graphicspath{}` in `thesis/thesis_fa.tex`) could resolve assets relative to
the thesis directory.

## What was done

During the micro-cleanup pass:

1. `thesis/results/` was inspected; every file was confirmed identical to its
   counterpart in root `results/` (see `git diff` and `diff -q`).
2. `thesis/results/` was removed from the source tree to make root `results/`
   the **single source of truth**.
3. This `README.md` and the parent folder are kept solely as a traceability
   marker — there is no longer any project workflow that writes into or reads
   from `thesis/results/`.
4. The thesis LaTeX `\graphicspath{}` already lists `../results/figures/` as
   its first lookup path, so removal of `thesis/results/` does not affect the
   build (verified by recompiling `thesis/thesis_fa.pdf`).

## Why this folder exists at all

To preserve auditable evidence that the duplication was intentionally
collapsed rather than silently destroyed, per the project's "do not delete,
only move/archive" rule. If a future reviewer wonders why `thesis/results/`
disappeared, this folder is the answer.

## Canonical locations going forward

- Metrics:   `results/CANONICAL_METRICS.json`
- Findings:  `results/KEY_FINDINGS.md`
- Defense:   `results/DEFENSE_RUN_SUMMARY.md`
- Figures:   `results/figures/`
- Tables:    `results/tables/`
