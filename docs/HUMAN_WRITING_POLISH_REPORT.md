# Human Writing Polish Report

Date: 2026-05-25  
Scope: local Persian academic polish only. No broad scientific rewrite, no metric change, and no conclusion strengthening beyond the evidence.

## Sections Reviewed

- Persian abstract
- English abstract
- Introduction and proposal-alignment language
- Experiment protocol wording
- Chapter 5 result interpretation
- Discussion terminology
- Conclusion and research-question answer table
- Appendices E, F, G, H, I, K, L, M, N, Q, and R

## Main Writing Problems Found

| Area | Problem | Fix |
|---|---|---|
| Abstract | Raw script/model identifiers made the abstract sound like an execution log | Replaced with human-readable model/pipeline descriptions |
| Chapter 5 | Some captions and rows read like generated CSV output | Captions and model labels rewritten in Persian-first academic form |
| Method/experiment prose | Percent and metric terminology could display awkwardly in Persian text | Replaced risky percent notation with `95 درصد`; standardized F1 wording |
| Discussion | API/ROI headings looked abrupt in English | Expanded to Persian descriptive headings with English abbreviations in parentheses |
| Conclusion | `Symptom Tokens` appeared in Persian narrative and RQ table | Replaced with `توکن‌های علائم` |
| Appendices | Several technical terms looked machine-translated or raw | Standardized focal loss, tokenization, F1 macro, API, and low-shot terminology |

## Scientific Meaning Preserved

The final thesis still states:

- Raw low-shot/ETHOS models do not beat the strongest classical tabular baselines.
- `FewShot BEST` improves over the raw proposed model but remains below strong classical models.
- `Stacked BEST` closes most of the gap to the tabular ceiling.
- Federated learning has a small AUROC cost but offers privacy-aware deployment value.
- Classical tabular models remain strong for structured clinical data.
- Hybrid/stacked modeling is the practical direction.

## Human-Style Improvements

- Replaced script-like labels with thesis-style labels.
- Reduced awkward English/Persian collisions in headings and captions.
- Standardized terminology so the reader does not have to decode multiple names for the same concept.
- Preserved formal Persian tone without making unsupported promotional claims.

## Remaining Risks

- Some historical V0--V6 strategy sections remain technically dense because they document the development path. They are now readable, but still intentionally technical.
- Reproducibility appendices include exact code names and commands; these are formatted safely and should remain exact.

## Status

Pass. The writing is substantially more natural and defense-ready while preserving the scientific caution and the original results.
