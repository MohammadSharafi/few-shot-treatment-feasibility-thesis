# Font and Table Size Audit

Date: 2026-05-25  
Scope: Persian thesis PDF typography, table font sizes, appendix table density, code-block readability, and mixed Persian/English inline terms.

## Font Strategy

- Main Persian body font was left unchanged to preserve the university template and the existing XePersian setup.
- English inline terms remain in `\lr{...}`.
- Code identifiers, filenames, commands, and metric keys use `\texttt{\lr{...}}` where exact reproducibility labels are needed.
- Tables use readable sizing first; shrinking is used only where the table is intrinsically wide.
- Appendix generated tables are allowed to be more compact, but they must remain legible and must be clearly identified as reproducibility output.

## Tables Resized or Wrapped

| Area | Action |
|---|---|
| Chapter 5 comparison tables | Labels were shortened/humanized rather than shrinking the table further. |
| Chapter 5 CI and K-shot tables | Headers were changed to Persian-readable forms (`بازۀ اطمینان 95 درصد`, `انحراف معیار`). |
| Chapter 5 ablation table | Raw row labels were replaced with readable Persian labels, improving column fit. |
| Chapter 5 strategy table | Converted to consistent compact `tabularx` sizing; metric columns widened enough to prevent `AUPRC`/`AUROC` header collision. |
| Chapter 7 cross-domain comparison table | Wrapped Latin benchmark/model cells and narrowed spacing consistently to remove overlap between accuracy text and benchmark names. |
| Appendix V5/V6 detailed tables | Converted to compact `tabularx` sizing and widened metric columns to prevent header collision. |
| Appendix glossary/abbreviation tables | Terminology standardized without increasing width. |
| API and hyperparameter appendix tables | Technical names retained; table density accepted because these are reference appendices. |

## Visual Findings

- Main tables are readable at normal thesis zoom.
- Chapter 5 tables no longer look like raw CSV/script dumps.
- Table captions are visually distinct and academic.
- English technical terms no longer appear visually reversed or in the wrong word order.
- The generated appendix tables are compact but still readable.

## Build Warning Context

The current XeLaTeX log reports 37 overfull hbox warnings, all within the validation tolerance. Visual inspection did not identify a defense-blocking overflow in the reviewed table pages. These warnings are mostly cosmetic and relate to dense technical content, formulas, or compact appendix material.

## Remaining Visual Risks

- Generated appendix tables are intentionally dense; they are acceptable for reproducibility but less elegant than the main Chapter 5 narrative tables.
- Some English parameter names in appendix tables are code-level identifiers and should remain English.
- Raster plot legends may have English model labels that are part of images, not LaTeX typography.

## Status

Pass. Table/font sizing is suitable for a final defense PDF, with only minor non-blocking density in reproducibility appendices. The specific overlap reported in Table 7.3 was fixed and visually verified after a clean rebuild.
