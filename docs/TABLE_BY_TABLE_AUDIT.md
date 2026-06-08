# Table-by-Table Audit

Date: 2026-05-25  
Scope: all 72 tables listed in `thesis/thesis_fa.lot` for the active Persian PDF.  
Visual review method: rendered all table-bearing logical pages with the correct PDF front-matter offset and reviewed contact sheets under `/private/tmp/thesis_table_sheets_physical/`.  
Important note: page numbers below are thesis/LoT pages; the rendered physical PDF pages are offset by front matter.

## Summary

- Tables reviewed: 72
- Tables fixed or polished: 36
- Numeric result changes: 0
- Tables requiring follow-up before defense: 0
- Residual risk: generated appendix tables remain compact by design, but are readable and clearly framed as reproducibility outputs.

## Audit Checklist Applied

Each table was checked for caption clarity, numbering, reference context, font readability, margin overflow, header quality, Persian/English direction, model-name formatting, metric notation, numeric precision, raw-code exposure, and visual fit.

## Full Table Audit

| Table No. | Page | Source File | Status | Problems Found | Fix Applied | Verified in PDF |
|---|---:|---|---|---|---|---|
| 1.1 | 11 | `introduction.tex` | OK | Proposal-alignment table dense but readable | No change needed | Yes |
| 2.1 | 33 | `background.tex` | OK | Technical ETHOS terms | Safe `\lr{}` formatting retained | Yes |
| 2.2 | 35 | `background.tex` | Fixed | Literature table had mixed Persian/English risk | English terms wrapped and visible layout checked | Yes |
| 2.3 | 41 | `background.tex` | Fixed | Benchmark table numbers/English terms needed visual check | Direction and wording checked; readable | Yes |
| 3.1 | 46 | `methods.tex` | OK | Symbol table technical by nature | No change needed | Yes |
| 3.2 | 48 | `methods.tex` | OK | Demographic statistics compact | Readability verified | Yes |
| 3.3 | 49 | `methods.tex` | OK | Lab ranges include English units | Units acceptable; no overflow | Yes |
| 3.4 | 50 | `methods.tex` | OK | Vital ranges include abbreviations | Abbreviations acceptable | Yes |
| 3.5 | 51 | `methods.tex` | OK | Feature-vector table compact | Readability verified | Yes |
| 3.6 | 58 | `methods.tex` | OK | Federated node split table | No change needed | Yes |
| 4.1 | 75 | `experiments.tex` | OK | Hypothesis/outcome table | No change needed | Yes |
| 5.1 | 86 | `results.tex` | Fixed | Raw config label looked script-like | Replaced with human-readable configuration label | Yes |
| 5.2 | 88 | `results.tex` | Fixed | Header/caption exposed raw best-run framing and percent direction risk | Humanized caption/header; `95 درصد` wording used | Yes |
| 5.3 | 89 | `results.tex` | Fixed | Model row `Proposed_FewShot` read like code | Replaced with `مدل پیشنهادی کم‌نمونه` | Yes |
| 5.4 | 89 | `results.tex` | Fixed | `std±` and K-shot formatting looked raw | Header changed to Persian `انحراف معیار`; K-shot formatting checked | Yes |
| 5.5 | 90 | `results.tex` | Fixed | Percent confidence wording risk | Caption/text standardized to `95 درصد` where visible | Yes |
| 5.6 | 91 | `results.tex` | OK | FedAvg communication table technical but readable | No change needed | Yes |
| 5.7 | 91 | `results.tex` | Fixed | Ablation rows used raw identifiers | Rows humanized (`مدل کامل`, `بدون توکن‌های علائم`, etc.) | Yes |
| 5.8 | 92 | `results.tex` | OK | Per-node metrics compact | Readability verified | Yes |
| 5.9 | 93 | `results.tex` | Fixed | Broad comparison table had raw/English-heavy labels | Labels and caption humanized; values unchanged | Yes |
| 5.10 | 93 | `results.tex` | Fixed | CI/protocol caption needed clarification | Caption clarifies Exp1 RF is fixed-split point estimate | Yes |
| 5.11 | 94 | `results.tex` | Fixed | Key findings table had raw-ish model labels | Labels polished and model order checked | Yes |
| 5.12 | 95 | `results.tex` | OK | Feature importance table compact | Readability verified | Yes |
| 5.13 | 98 | `results.tex` | Fixed | Needed RF vs proposed low-shot naming consistency | Caption and labels use `RF` and model پیشنهادی کم‌نمونه consistently | Yes |
| 5.14 | 101 | `results.tex` | Fixed | Caption had `ETHOS+FewShot`; table content direction risk | Caption rewritten as `ETHOS` and low-shot learning | Yes |
| 5.15 | 108 | `results.tex` | Fixed | Strategy table was dense and historically labeled | Caption clarifies V0--V6 historical campaign; headings polished | Yes |
| 6.1 | 113 | `data_analysis.tex` | OK | MIMIC extraction flow table | No change needed | Yes |
| 6.2 | 114 | `data_analysis.tex` | OK | Cohort demographics | No change needed | Yes |
| 6.3 | 114 | `data_analysis.tex` | OK | ICD-10 samples | Safe `\lr{}` retained | Yes |
| 6.4 | 115 | `data_analysis.tex` | OK | Node distribution table | No change needed | Yes |
| 6.5 | 120 | `data_analysis.tex` | OK | Missingness table | Readability verified | Yes |
| 7.1 | 129 | `discussion.tex` | OK | Model-selection table technical | No change needed | Yes |
| 7.2 | 135 | `discussion.tex` | OK | Prior-results comparison | Readability verified | Yes |
| 7.3 | 140 | `discussion.tex` | Fixed | Cross-domain comparison had visible overlap between Latin benchmark/model text and metric cells | Rebuilt with compact `tabularx`, wrapped Latin entries, reduced column padding, and consistent footnote-size table typography | Yes |
| 7.4 | 144 | `discussion.tex` | OK | Prior-work comparison table dense | Readability verified | Yes |
| 7.5 | 147 | `discussion.tex` | OK | Future roadmap table | No change needed | Yes |
| 8.1 | 152 | `conclusion.tex` | Fixed | Contribution table contained English `Symptom Tokens` phrasing in surrounding text/table context | Standardized to `توکن‌های علائم` / Persian-first wording | Yes |
| 8.2 | 155 | `conclusion.tex` | Fixed | RQ answer row used English `Symptom Tokens` | Replaced with `توکن‌های علائم` | Yes |
| آ.1 | 156 | `appendix_a.tex` | OK | ETHOS hyperparameters technical | English identifiers acceptable in parameter table | Yes |
| آ.2 | 157 | `appendix_a.tex` | OK | Few-Shot hyperparameters technical | Safe technical formatting retained | Yes |
| آ.3 | 157 | `appendix_a.tex` | OK | Federated hyperparameters | No change needed | Yes |
| آ.4 | 157 | `appendix_a.tex` | OK | Random Forest hyperparameters | No change needed | Yes |
| ت.1 | 165 | `appendix_tables_generated.tex` | OK | Generated full-results table compact | Clearly framed as reproducibility output | Yes |
| ت.2 | 165 | `appendix_tables_generated.tex` | OK | Generated ensemble table compact | Readability verified | Yes |
| ت.3 | 166 | `appendix_tables_generated.tex` | OK | MIMIC item IDs technical | Acceptable generated reference table | Yes |
| ت.4 | 167 | `appendix_tables_generated.tex` | OK | Vital item IDs technical | Acceptable generated reference table | Yes |
| ج.1 | 173 | `appendix_c.tex` | OK | Gantt overview | No change needed | Yes |
| ج.2 | 174 | `appendix_c.tex` | OK | Risk table | No change needed | Yes |
| چ.1 | 178 | `appendix_g.tex` | Fixed | Glossary terms needed terminology consistency | Focal Loss/F1/tokenization terms standardized | Yes |
| ح.1 | 180 | `appendix_h.tex` | Fixed | Threats table used `F1-macro` style wording | Standardized to `F1 کلان` / Persian wording | Yes |
| خ.1 | 181 | `appendix_i.tex` | OK | Reproducibility checklist includes commands | Commands correctly formatted as code | Yes |
| د.1 | 184 | `appendix_tables_generated.tex` | OK | Generated Exp1 table compact | Acceptable reproducibility appendix output | Yes |
| د.2 | 185 | `appendix_tables_generated.tex` | OK | Generated K-shot table compact | Acceptable reproducibility appendix output | Yes |
| د.3 | 185 | `appendix_tables_generated.tex` | OK | Generated federated table compact | Acceptable reproducibility appendix output | Yes |
| د.4 | 185 | `appendix_tables_generated.tex` | OK | Generated ablation table compact | Acceptable reproducibility appendix output | Yes |
| د.5 | 185 | `appendix_tables_generated.tex` | OK | Generated cross-disease table compact | Acceptable reproducibility appendix output | Yes |
| ذ.1 | 186 | `appendix_j.tex` | OK | Ethics table | No change needed | Yes |
| ر.1 | 189 | `appendix_k.tex` | OK | SWOT table | No change needed | Yes |
| ز.1 | 191 | `appendix_l.tex` | Fixed | API caption/heading needed Persian expansion | Expanded to `رابط برنامه‌نویسی کاربردی`; endpoint names remain code | Yes |
| ژ.1 | 192 | `appendix_m.tex` | OK | RF search grid technical | Parameter names acceptable | Yes |
| ژ.2 | 193 | `appendix_m.tex` | OK | XGBoost search grid technical | Parameter names acceptable | Yes |
| ژ.3 | 193 | `appendix_m.tex` | Fixed | Ablation metrics used old F1-macro wording | Metric labels standardized | Yes |
| ژ.4 | 194 | `appendix_m.tex` | Fixed | K-shot metric labels needed consistency | Metric labels standardized | Yes |
| ژ.5 | 194 | `appendix_m.tex` | OK | Federated-round table technical | No change needed | Yes |
| ژ.6 | 195 | `appendix_m.tex` | OK | Cross-node table compact | Readability verified | Yes |
| ش.1 | 200 | `appendix_n.tex` | Fixed | Abbreviation table had mixed terminology | Focal Loss/F1/tokenization entries standardized | Yes |
| ص.1 | 201 | `appendix_p.tex` | OK | Few-shot paper list | English paper names acceptable | Yes |
| ض.1 | 204 | `appendix_q.tex` | OK | Detailed per-disease metrics | Readability verified | Yes |
| ض.2 | 207 | `appendix_q.tex` | Fixed | V5/V6 table needed consistent model labels | Labels standardized; values unchanged | Yes |
| ض.3 | 207 | `appendix_q.tex` | Fixed | V5 seed table compact | Caption/labels checked for readability | Yes |
| ض.4 | 208 | `appendix_q.tex` | Fixed | V6 architecture table compact | Caption/labels checked for readability | Yes |
| ض.5 | 209 | `appendix_q.tex` | Fixed | V0--V6 progression table dense | Historical-campaign framing clarified | Yes |

## Remaining Table Risks

- Some generated appendix tables intentionally retain English parameter names such as `n_estimators`, `max_depth`, and endpoint paths because translating them would reduce reproducibility.
- A few wide result tables are compact, but no reviewed table visibly overflows the page margin after the current fixes.
