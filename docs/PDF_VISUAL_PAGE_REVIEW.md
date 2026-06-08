# PDF Visual Page Review

Date: 2026-05-25  
Scope: generated Persian thesis PDF `thesis/thesis_fa.pdf`.

## Rendering Method

Pages were rendered with the local PDF page-rendering utility to PNG under:

- `/private/tmp/thesis_visual_review/`
- `/private/tmp/thesis_visual_review2/`
- `/private/tmp/thesis_table_pages_physical/`
- `/private/tmp/thesis_table_sheets_physical/`

The list-of-tables page numbers are logical thesis pages, so the table-page render set used the PDF physical-page offset. This correction was important: rendering the raw LoT page numbers alone showed earlier pages, not the actual table pages.

## Pages Reviewed

Unique physical PDF pages reviewed: 100

Reviewed pages:

`1, 2, 3, 4, 5, 10, 14, 20, 22, 24, 27, 30, 32, 35, 45, 65, 67, 69, 75, 79, 80, 82, 83, 84, 85, 92, 105, 109, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 154, 160, 163, 169, 174, 178, 181, 183, 186, 187, 189, 190, 191, 197, 199, 200, 201, 204, 207, 208, 212, 214, 215, 218, 219, 220, 223, 225, 226, 227, 228, 229, 234, 235, 236, 238, 241, 242, 243, 248, 258`

## Area-by-Area Findings

| Area | Pages / Range | Visible Issues Found | Fixes Applied | Remaining Risk |
|---|---|---|---|---|
| Cover and early front matter | 1--5 | Title and logo visible; no title drift | No title change needed | None |
| TOC / list pages | 10, 14, 20, 22, 24, 27, 30, 32, 35 | Earlier raw headings and awkward English/Persian terms were visible | Headings corrected with Persian-first labels and safe English wrapping | None observed after rebuild |
| Chapter openings | 45, 80, 118, 157, 184, 190, 199, 215, 220, 226, 235, 238 | Chapter starts are clean; appendix headings mostly professional | API/ROI and appendix heading terminology polished | None blocking |
| Literature tables | 65, 67, 69, 75 | Literature tables dense but readable; English names safe | No numeric changes; formatting checked | Minor density inherent to comparison tables |
| Methodology tables | 80, 82, 83, 84, 85, 92, 109 | Tables readable; no margin overflow | No change beyond terminology already applied | None |
| Chapter 5 result pages | 120--145 | Raw experiment labels, old model strings, and percent-direction issues were visible before fixes | Chapter 5 captions, headings, table headers, and labels polished | Raster plot legends still English |
| Data-analysis chapter figures/tables | 147--160 | Captions mostly good; one ETHOS/FewShot caption needed consistency | Caption rewritten | Some plot legends remain English |
| Discussion/conclusion pages | 163--187 | A remaining `Symptom Tokens` phrase appeared in conclusion context | Replaced with `توکن‌های علائم` | None observed |
| Appendices and generated tables | 189--243 | Generated appendix tables are compact; API/hyperparameter tables technical | Terminology and headings polished; generated tables retained as reproducibility material | Compact appendix tables are readable but less elegant |
| Bibliography / final pages | 248, 258 | No visible thesis-format blocker observed | No change needed | None |

## Chapter 5 Deep Visual Review

Chapter 5 was reviewed page by page. The following high-risk issues were specifically checked:

- `1F`: not visible.
- `Forest Random`: not visible.
- `shot-few` / `Shot-Few`: not visible.
- raw `ETHOS_RF_Hybrid` in headings/prose: not visible.
- raw script names in captions: removed.
- table overflow: not observed.
- wrong percent direction around confidence intervals: fixed to `95 درصد` where needed.
- raw ablation labels: replaced with Persian readable labels.

## Status

Pass. The PDF is visually professional enough for defense, with only minor residual English labels inside raster plot legends and dense reproducibility appendix tables.
