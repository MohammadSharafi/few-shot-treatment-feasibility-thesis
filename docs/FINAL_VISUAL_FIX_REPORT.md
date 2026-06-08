# Final Visual Fix Report

Date: 2026-05-25

## Main issue reviewed

The suspected remaining issue was the appearance of `1F` in automated parsed text. The generated PDF was visually checked on the relevant table-of-contents pages. The metric is rendered visually as `F1`, not `1F`. The `1F` string was caused by PDF text extraction order in right-to-left context, not by the visible PDF layout.

## Verified fixes / checks

- No unacceptable source occurrences of `1F`, `Forest Random`, `Shot-Few`, `shot-few`, `Attention-Self`, `Loss Focal`, `SOTA Tabular`, or raw `ETHOS_RF_Hybrid` headings were found in the Persian LaTeX source.
- `make validate` passed with score 100/100.
- `make defense` passed.
- `thesis/thesis_fa.pdf` exists and has the correct Persian title metadata.
- `thesis/defense_slides.pdf` exists.

## Build script improvement

`thesis/build_thesis_fa.sh` was updated to run an additional XeLaTeX pass after BibTeX. This makes citation/reference resolution more reliable on a clean local rebuild.

## Remaining note

The project package intentionally does not include raw commercial/local font files. The existing PDF is ready to use. For local rebuild, keep your local Persian thesis fonts in `thesis/fonts/` as described in `thesis/fonts/README.md`.
