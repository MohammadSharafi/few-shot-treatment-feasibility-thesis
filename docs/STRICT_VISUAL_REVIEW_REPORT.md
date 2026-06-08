# Strict Visual Review Report

**Project:** Few-Shot Treatment Feasibility Prediction  
**Official Persian title:** فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده  
**Review date:** 2026-05-25  
**Scope:** Persian thesis PDF visual/typographic/table/caption review only. No metrics, result values, title text, model rankings, or scientific conclusions were changed.

## Review Coverage

- Final PDF reviewed: `thesis/thesis_fa.pdf`
- Defense PDF rebuilt: `thesis/defense_slides.pdf`
- Front matter and TOC pages visually checked: physical pages 1-34.
- Table-bearing printed pages visually checked from the compiled list of tables: 11, 33, 35, 41, 46, 48, 49, 50, 51, 58, 75, 86, 88, 89, 90, 91, 92, 93, 95, 96, 99, 102, 109, 113, 114, 115, 120, 129, 134, 139, 144, 147, 152, 155, 156, 157, 165, 166, 167, 173, 174, 178, 180, 181, 184, 185, 186, 189, 191, 192, 193, 194, 195, 200, 201, 204, 206, 207, 208.
- Figure-heavy printed pages checked from the compiled list of figures: 47, 87, 88, 90, 91, 92, 94, 97, 98, 112, 116, 117, 118, 119, 120, 173.
- Total distinct printed pages visually reviewed in this pass: 76.
- Total tables reviewed from `thesis/thesis_fa.lot`: 72.

## Issue Log

| # | Page | Section/table/figure | Problem type | Current problematic text or visual issue | Exact fix applied | File changed | Status |
|---:|---:|---|---|---|---|---|---|
| 1 | 86 | Section 5.2 | RTL/LTR direction | The heading rendered visually as `FewShot+ETHOS` instead of `ETHOS+FewShot`. | Replaced split `\lr{ETHOS}+\lr{FewShot}` with one LTR span `\lr{ETHOS+FewShot}`. | `thesis/chapters_fa/results.tex` | Fixed |
| 2 | 88 | Table 5.2 caption | XePersian source hygiene | Caption had nested `\lr{\texttt{\lr{...}}}` around the script name. | Simplified to `\texttt{\lr{exp\_fewshot\_best.py}}`. | `thesis/chapters_fa/results.tex` | Fixed |
| 3 | 89-91 | Tables 5.3, 5.4, 5.5, 5.7 captions | XePersian source hygiene | Several captions used nested LTR/code wrappers around script names. | Simplified each script reference to direct `\texttt{\lr{...}}` form. | `thesis/chapters_fa/results.tex` | Fixed |
| 4 | 89, 93, 94, 98, 108 | Tables 5.3, 5.9, 5.11, 5.13, 5.15 | Code/model identifier formatting | Code-like model identifiers such as `Proposed_FewShot`, `FederatedNN_NoFewShot`, `ETHOS_RF_Hybrid`, and `RF_threshold_tuned` were visually LTR but not consistently styled. | Replaced narrative-table identifiers with human-readable display names where possible, and kept code formatting only where the identifier itself is the subject. | `thesis/chapters_fa/results.tex` | Fixed |
| 5 | 93 | Table 5.10 caption | Code/file formatting | `exp_fewshot_best.json` was wrapped through a nested `\lr{\texttt{...}}` construct. | Simplified to `\texttt{\lr{exp\_fewshot\_best.json}}`. | `thesis/chapters_fa/results.tex` | Fixed |
| 6 | 98 | Table 5.13 | Code/model identifier formatting | `Proposed_FewShot` appeared in the caption/header as a raw model ID. | Replaced the visible caption/header text with readable Persian display labels. | `thesis/chapters_fa/results.tex` | Fixed |
| 7 | 102 | Table 5.14 caption and nearby text | RTL/LTR direction | Historical architecture table still used split `\lr{ETHOS}+\lr{FewShot}` in restored source. | Normalized all restored split spans on that page to `\lr{ETHOS+FewShot}`. | `thesis/chapters_fa/results.tex` | Fixed |
| 8 | 108 | Table 5.15 | Table readability | The strategy table was compact and visually dense under `\scriptsize`; after standardization, the metric headers also needed more room. | Changed table size to `\footnotesize`, widened metric columns, and kept the table inside the text block. | `thesis/chapters_fa/results.tex` | Fixed |
| 9 | 108 | Table 5.15 | Code/model identifier formatting | `Proposed_FewShot` and `RF_threshold_tuned` appeared as raw-looking model identifiers. | Replaced them with readable Persian/English display names and separated `AUROC`/`AUPRC` headers clearly. | `thesis/chapters_fa/results.tex` | Fixed |
| 13 | 129, 134 | Tables 7.1, 7.2 | Raw identifier in narrative tables | Discussion tables used raw script/model keys in recommendation and comparison rows. | Replaced with readable labels such as `Stacked BEST`, `RF` تنظیم آستانه, and مدل پیشنهادی کم‌نمونه. | `thesis/chapters_fa/discussion.tex` | Fixed |
| 14 | 184-185 | Appendix D generated tables | Raw generated labels | Generated pipeline appendix rows used raw model, variant, and node keys. | Updated `results/generate_all_outputs.py` display labels and regenerated `ALL_TABLES_APPENDIX.tex`. | `results/generate_all_outputs.py`, `results/tables/ALL_TABLES_APPENDIX.tex` | Fixed |
| 10 | 157 | Appendix A heading | Mixed text in heading/bookmark | The section heading for `ETHOS_RF_Hybrid` used a raw code-like identifier in a heading context. | Rewrote with `\texorpdfstring{...}{...}` and `\texttt{\lr{ETHOS\_RF\_Hybrid}}`. | `thesis/chapters_fa/methods.tex` | Fixed |
| 11 | 194 | Appendix E section | Metric terminology consistency | Appendix section title used `F1-Macro` while the thesis standard elsewhere is `F1-macro`. | Normalized heading to `\lr{F1-macro}`. | `thesis/chapters_fa/appendix_e.tex` | Fixed |
| 12 | 7, 14 | Introduction narrative | Code identifier formatting | `Proposed_FewShot` and `exp_fewshot_best` appeared through nested wrappers. | Simplified to direct `\texttt{\lr{...}}` form. | `thesis/chapters_fa/introduction.tex` | Fixed |

## Final Visual Recheck

After the fixes, the required sequence was run again:

```bash
make clean-temp
make thesis-fa
make validate
make defense
```

The regenerated PDF was rechecked visually on the affected pages:

- Page 86: `ETHOS+FewShot` heading now renders in the correct order.
- Page 99: per-disease model identifier header/caption remains inside margins.
- Page 108: Table 5.15 remains inside margins and has cleaner metric headers/model labels.
- Front matter and TOC: no visible broken RTL/LTR entries were observed.

## Remaining Visual Risks

- Some appendix tables are intentionally dense because they preserve full generated outputs and hyperparameter grids. They remain inside the text block but are compact.
- `thesis_fa.log` still reports 25 overfull hbox warnings, which the validator treats as cosmetic and within tolerance. No reviewed page showed a table spilling outside the printable area.
- The generated appendix tables in Appendix D remain English/CSV-like by design because they mirror pipeline exports for reproducibility; they are labeled as exported tables and are not used as the primary narrative result tables.

## Recommendation

**Submit after archiving the final project zip.** The strict visual pass found and fixed all concrete issues observed in captions, Chapter 5 result tables, mixed English/Persian spans, metric notation, and code/model identifier formatting without changing any scientific value or conclusion.
