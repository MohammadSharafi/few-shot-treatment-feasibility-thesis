# Strict Final Review Issue Log

Date: 2026-05-25  
Scope: active Persian thesis build rooted at `thesis/thesis_fa.tex`, generated Persian PDF, Chapter 5, appendices, captions, TOC/LOT/LOF, and validation output.  
Guardrails: official title, canonical metrics, model rankings, numerical values, scientific conclusions, and proposal-alignment meaning were not changed.

## 1. Source Scan Findings

The source scan covered `thesis/thesis_fa.tex`, `thesis/chapters_fa/*.tex`, `thesis/frontmatter_fa/*.tex`, active generated table snippets, `thesis/thesis_fa.toc`, `thesis/thesis_fa.lot`, and `thesis/thesis_fa.lof`.

Unacceptable occurrences of the critical strings below were removed or verified absent in active Persian thesis output:

- `1F`
- `Forest Random`
- `shot-few` / `Shot-Few`
- `Tokens Symptom`
- raw `ETHOS_RF_Hybrid` in prose/headings/TOC
- raw unescaped underscores in normal Persian prose

Remaining matches are acceptable technical uses: `Focal Loss` appears only as `زیان کانونی (\lr{Focal Loss})`, `ROI` and `API` are expanded in Persian headings, and `FewShot V2--V6` appears as controlled model-version labels in the historical strategy section.

## 2. PDF Visual Findings

The visual review covered front matter, TOC, list pages, first pages of all chapters, all Chapter 5 pages, selected appendix pages, bibliography-adjacent pages, and post-fix re-rendered Chapter 5 pages. The most visible issues found were raw identifiers in headings/captions, old experiment labels in Chapter 5, percent-direction issues, and table rows that read like script output.

## 3. Table-by-Table Findings

The active Persian PDF lists 72 tables. All were audited. The highest-risk fixes were in Chapter 5 tables 5.1, 5.2, 5.4, 5.7, 5.9, 5.10, 5.11, 5.14, and 5.15, plus appendix tables with generated or technical labels. No numeric result was changed.

## 4. Figure/Caption Findings

The active Persian PDF lists 20 figures. All captions were audited. Captions with awkward `ETHOS+FewShot` wording and percent-direction issues were rewritten into Persian-first academic captions. Raster figure legends were not regenerated; a few still contain English model labels as plot content, which is acceptable and recorded as a minor residual risk.

## 5. TOC and Heading Findings

TOC-visible headings were checked after rebuild. Raw headings such as `ETHOS_RF_Hybrid :معماری ترکیبی` and awkward mixed headings around `ETHOS+FewShot`, `F1-macro`, `API`, `ROI`, and strategy labels were replaced or contextualized with Persian-first titles and safe `\texorpdfstring`.

## 6. Appendix Findings

Appendices E, F, G, H, I, K, L, M, N, Q, and R were reviewed. The most important fixes were terminology normalization for low-shot learning, focal loss, tokenization, API naming, F1-macro labels, and extended V5/V6 result language. Code and command references remain in appendices only where needed for reproducibility and are formatted with `\texttt{\lr{...}}`.

## 7. Human-Writing Findings

The thesis did not need a broad scientific rewrite. Local human-writing polish was applied to the Persian abstract, Chapter 5 result interpretation, experiment protocol prose, discussion terminology, conclusion-adjacent phrasing, and appendix labels. The conservative scientific story was preserved: raw Few-Shot remains weaker than strong classical tabular models, while hybrid/stacked modeling closes most of the gap.

## 8. Prioritized Fix Plan and Result

1. Fix critical TOC/heading issues: completed.
2. Replace visible raw script/model names in captions and prose where not required: completed.
3. Standardize metric/model terminology in Chapter 5 and appendices: completed.
4. Polish figure captions and appendix headings: completed.
5. Audit all 72 tables and 20 figures: completed.
6. Rebuild with `make clean-temp`, `make thesis-fa`, `make validate`, and `make defense`: passed on 2026-05-25.
7. Re-inspect regenerated PDF pages: completed for the pre-final build and Chapter 5 post-fix pages; final sequence reuses the same source corrections.

## Issue Register

| Issue ID | Page / Area | Source File | Line | Current Problematic Text | Category | Severity | Fix Applied | Status |
|---|---:|---|---:|---|---|---|---|---|
| SF-001 | TOC, Chapter 3 | `thesis/chapters_fa/methods.tex` | heading | `ETHOS_RF_Hybrid` in heading/TOC | Raw identifier in heading | Critical | Replaced with `معماری ترکیبی \lr{ETHOS} و \lr{Random Forest}` using safe heading form | Fixed / Verified |
| SF-002 | TOC, Chapter 4/5 | `experiments.tex`, `results.tex` | multiple | `ETHOS+FewShot` in headings | Awkward mixed heading | Major | Rewritten as `ETHOS` plus `یادگیری کم‌نمونه` | Fixed / Verified |
| SF-003 | TOC, appendices | `appendix_e.tex`, `appendix_n.tex`, `appendix_m.tex` | multiple | `F1-macro` as heading/label | Metric terminology | Major | Standardized to `امتیاز F1 کلان`, `F1 کلان`, or safe `Macro-F1` where formula-specific | Fixed / Verified |
| SF-004 | Chapter 5 table caption | `results.tex` | table 5.2 area | `exp_fewshot_best.py` in caption/prose | Raw filename in caption | Major | Replaced with human-readable best low-shot/stacked run description | Fixed / Verified |
| SF-005 | Chapter 5 prose/caption | `results.tex` | Exp1 area | `exp1_main.py` visible in result narrative | Raw filename in prose | Minor | Replaced with `آزمایش ۱` and table references | Fixed / Verified |
| SF-006 | Chapter 5 table caption | `results.tex` | table 5.14 | `ETHOS+FewShot` | Awkward mixed caption | Major | Rewritten as `ETHOS` and low-shot learning in Persian | Fixed / Verified |
| SF-007 | Chapter 5 strategy headings | `results.tex` | strategy section | Compact V2/V3/V4/V5/V6 headings | Heading polish | Major | Rewritten as readable Persian strategy labels with safe English terms | Fixed / Verified |
| SF-008 | Chapter 5 strategy text | `results.tex` | strategy section | `Cross-Attention`, `Ensemble` in dense headings | RTL/LTR heading polish | Major | Wrapped with `\lr{}` and contextualized in Persian | Fixed / Verified |
| SF-009 | Appendix E | `appendix_e.tex` | statistical section | Strong DeLong/p-value interpretation risk | Scientific safety wording | Major | Clarified paired tests require aligned prediction vectors; cross-protocol comparisons are descriptive | Fixed / Verified |
| SF-010 | Appendix F | `appendix_f.tex` | multiple | `ETHOS+FewShot` | Terminology consistency | Major | Rewritten as `ETHOS` and low-shot learning | Fixed / Verified |
| SF-011 | Appendix H | `appendix_h.tex` | risk table | `F1-macro` | Metric terminology | Minor | Standardized to Persian F1 macro wording | Fixed / Verified |
| SF-012 | Appendix L | `appendix_l.tex` | chapter/section | `API` heading without Persian context | Heading polish | Minor | Expanded to `رابط برنامه‌نویسی کاربردی (\lr{API})` | Fixed / Verified |
| SF-013 | Persian abstract | `thesis_fa.tex` | abstract | Raw script/model identifiers | Abstract polish | Major | Replaced with human-readable pipeline/model descriptions | Fixed / Verified |
| SF-014 | English abstract | `frontmatter_fa/abstract_en.tex` | abstract | Raw script identifiers | Abstract polish | Minor | Replaced with readable experiment labels | Fixed / Verified |
| SF-015 | Figure captions | `results.tex`, `data_analysis.tex` | multiple | `ETHOS+FewShot` in captions | Caption polish | Major | Rewritten as `ETHOS` baseline with low-shot learning | Fixed / Verified |
| SF-016 | Appendix generated tables | active appendix tables | multiple | Raw result-dump style labels | Table polish | Major | Main narrative tables humanized; generated reproducibility appendix retained but contextualized | Fixed / Verified |
| SF-017 | Table of contents | generated from headings | N/A | Dense raw English technical terms | TOC polish | Major | Persian-first headings and safe PDF strings applied | Fixed / Verified |
| SF-018 | Full source/PDF | multiple | N/A | Need verify no unacceptable `1F` remains | Metric notation | Critical | Source scan found zero unacceptable instances | Fixed / Verified |
| SF-019 | Full source/PDF | multiple | N/A | Need verify no unacceptable `Forest Random` remains | Model name order | Critical | Source scan found zero unacceptable instances | Fixed / Verified |
| SF-020 | Code/file identifiers | multiple | N/A | Raw underscores in prose/headings | Identifier formatting | Major | Humanized where possible; required reproducibility identifiers formatted as `\texttt{\lr{...}}` | Fixed / Verified |

## Remaining Risks

- Some raster figure legends still contain English model labels because they are embedded inside PNG figures. Captions and surrounding thesis prose are corrected; regenerating figures was intentionally avoided to prevent unintended scientific or visual changes.
- The PDF still has a small number of cosmetic overfull hbox warnings within the validation tolerance. They do not correspond to visible table overflow in the reviewed pages.
