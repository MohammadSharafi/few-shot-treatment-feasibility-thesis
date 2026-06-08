# Final Submission Cleanup Report

**Project:** Few-Shot Treatment Feasibility Prediction  
**Official Persian title:** فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده  
**Updated:** 2026-05-25

## Current Cleanup State

The project now has a reproducible submission path:

- `experiments/` contains the canonical scripts.
- `results/CANONICAL_METRICS.json` is the single metric source of truth.
- Duplicate `thesis/results/` is removed.
- `make clean-temp` preserves final PDFs and removes only auxiliary/build files.
- `make thesis-fa` rebuilds the Persian thesis and defense slides.
- `make validate` fails honestly on build, citation, reference, metadata, artifact, and metric problems.
- `make defense` requires both final PDFs and reads canonical metrics only.

## Title Status

The official Persian title remains:

```text
فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده
```

Validation checks the title in the LaTeX sources and the final PDF metadata.

## Validation Snapshot

Latest successful validator result:

- Passed: 51
- Warnings: 0
- Failed: 0
- Score: 100/100
- Missing-character warnings: 0
- Undefined citations: 0
- Undefined references: 0

## Remaining Risks

- Scientific limitations remain as documented: 2,000-patient MIMIC-IV subset, no external validation, simulated rather than deployed federation.
- Cosmetic overfull/underfull XePersian warnings remain, but they are not submission blockers.

## Final Recommendation

Submit only if the exact final sequence in
`docs/FINAL_REPRODUCIBILITY_FIX_REPORT.md` passes:

```bash
make clean-temp
make thesis-fa
make validate
make defense
```
