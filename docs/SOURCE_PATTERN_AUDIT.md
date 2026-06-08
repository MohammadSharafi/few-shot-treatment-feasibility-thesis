# Source Pattern Audit

Date: 2026-05-25  
Scope: active Persian thesis sources and generated navigation files used by `thesis/thesis_fa.pdf`.

## Scan Scope

Reviewed:

- `thesis/thesis_fa.tex`
- `thesis/frontmatter_fa/*.tex`
- `thesis/chapters_fa/*.tex`
- active generated table/caption snippets included from the Persian thesis
- `thesis/thesis_fa.toc`
- `thesis/thesis_fa.lot`
- `thesis/thesis_fa.lof`

Not treated as active Persian PDF prose:

- inactive English thesis drafts under `thesis/chapters/`
- unused `thesis/tables_extended.tex` tables not included by `thesis/thesis_fa.tex`
- code blocks and formulas where underscores and English identifiers are semantically required

## Mandatory Pattern Results

| Pattern | Result | Acceptability | Fix Applied / Note |
|---|---:|---|---|
| `1F` | 0 unacceptable active occurrences | OK | Metric notation standardized to `F1`. |
| `Score1-F` | 0 | OK | Not present in active Persian build. |
| `Forest Random` | 0 unacceptable active occurrences | OK | All visible prose/headings use `Random Forest` or Persian description. |
| `shot-few` / `Shot-Few` | 0 unacceptable active occurrences | OK | Standardized to `یادگیری کم‌نمونه` or safe `Few-Shot`. |
| `Learning Shot-Few` | 0 | OK | Not present. |
| `Tokens Symptom` | 0 | OK | Standardized to `توکن‌های علائم`. |
| `Tokenization Symptom` | 0 | OK | Standardized to `توکن‌سازی علائم`. |
| `Attention-Self` | 0 | OK | Standardized to `خودتوجهی` / `Self-Attention`. |
| raw `ETHOS_RF_Hybrid` | 0 unacceptable active occurrences | OK | Main headings/prose use `معماری ترکیبی ETHOS و Random Forest`; any required identifier is formatted. |
| raw `RandomForest_threshold_tuned` | 0 unacceptable active occurrences | OK | Not exposed as raw prose/headings. |
| raw `CANONICAL_METRICS.json` | acceptable only where formatted | OK | Reproducibility references use `\texttt{\lr{...}}` where needed. |
| raw `exp_fewshot_best.py` | 0 unacceptable active occurrences | OK | Removed from captions/prose; reproducibility appendix uses safe command formatting only when needed. |
| raw `tabular_sota_auroc_raw` | 0 unacceptable active occurrences | OK | Not exposed as prose. |
| `ETHOS+FewShot` | 0 unacceptable active occurrences | OK | Rewritten in captions/headings; figure legends inside raster images not counted as source prose. |
| `F1-macro` | 0 unacceptable prose/heading occurrences | OK | Replaced with `امتیاز F1 کلان`, `F1 کلان`, or formula-specific `Macro-F1`. |
| `SOTA Tabular` | 0 unacceptable occurrences | OK | Standardized to `Tabular SOTA` or `سقف مدل‌های جدولی`. |

## Remaining Matches Classified as Acceptable

| Matched Text | Location Type | Why Acceptable |
|---|---|---|
| `زیان کانونی (\lr{Focal Loss})` | headings and formulas | Persian-first heading with the English technical term safely wrapped. |
| `FewShot V2--V6` | Chapter 5 historical strategy section and appendix Q | These are controlled model-version labels requested for the V0--V6 development campaign; wrapped with `\lr{}` and not raw reversed English. |
| `SupCon`, `Proto-MAML`, `Cross-Attention`, `Ensemble`, `XGBoost` | Chapter 5 strategy details | Technical model components are wrapped and contextualized in Persian prose. |
| `ROI` | Discussion heading/prose | Expanded as `بازگشت سرمایه‌ی توضیحی (\lr{ROI})`. |
| `API` | Discussion/appendix heading/prose | Expanded as `رابط برنامه‌نویسی کاربردی (\lr{API})`. |
| `Proposed\_FewShot`, `ETHOS\_Supervised`, command/file names | Reproducibility and protocol context | Only retained where exact experiment identifiers are needed; formatted as `\texttt{\lr{...}}`. |

## Raw Underscore Audit

Raw underscores in normal Persian prose/headings are not present. Remaining underscores occur in one of these acceptable contexts:

- LaTeX labels such as `\label{tab:...}`
- math variables such as `\theta_t`
- code blocks/listings
- reproducibility commands and filenames formatted with `\texttt{\lr{...}}`
- exact model identifiers in protocol discussion, also formatted with `\texttt{\lr{...}}`

## Fixes Applied

- Replaced raw or semi-raw model identifiers in Chapter 5 captions and prose with human-readable Persian labels.
- Converted `F1-macro` labels to Persian-first F1 macro terminology.
- Replaced `ETHOS+FewShot` headings/captions with `ETHOS` plus `یادگیری کم‌نمونه`.
- Converted `API` and `ROI` headings to expanded Persian forms.
- Fixed percent-direction wording from symbolic percent in Persian prose where visually risky.
- Kept exact script/model identifiers only where reproducibility requires them and applied safe XePersian formatting.

## Acceptance Status

Pass. No unacceptable active occurrence remains for the critical user-reported patterns. Remaining English technical terms are intentional, wrapped, and explained.
