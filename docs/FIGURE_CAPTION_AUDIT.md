# Figure and Caption Audit

Date: 2026-05-25  
Scope: all 20 figures listed in `thesis/thesis_fa.lof`.

## Summary

- Figures/captions reviewed: 20
- Captions fixed or polished: 4
- Figures regenerated: 0
- Numeric/plot data changed: 0

## Full Caption Audit

| Figure No. | Page | Source File | Caption Status | Problems Found | Fix Applied |
|---|---:|---|---|---|---|
| 3.1 | 47 | `methods.tex` | OK | Pipeline figure caption technical but clear | No change needed |
| 5.1 | 87 | `results.tex` | OK | ROC caption includes model terms | Terms safely wrapped; no change needed |
| 5.2 | 87 | `results.tex` | Fixed | Caption previously had awkward English punctuation/term flow | Caption polished for Random Forest and ETHOS PR curves |
| 5.3 | 88 | `results.tex` | Fixed | Caption needed clearer distinction of V0 vs later improvements | Caption clarified without changing plotted data |
| 5.4 | 90 | `results.tex` | Fixed | Percent direction risk in confidence-band phrase | Caption uses `95 درصد` form |
| 5.5 | 90 | `results.tex` | OK | Federated convergence caption clear | No change needed |
| 5.6 | 92 | `results.tex` | OK | Radar ablation caption clear | No change needed |
| 5.7 | 94 | `results.tex` | OK | Main model-comparison caption clear | No change needed |
| 5.8 | 96 | `results.tex` | OK | Feature importance caption clear | No change needed |
| 5.9 | 96 | `results.tex` | OK | SHAP waterfall caption clear | No change needed |
| 5.10 | 97 | `results.tex` | OK | Calibration caption clear and honest | No change needed |
| 6.1 | 112 | `data_analysis.tex` | OK | MIMIC extraction schematic caption clear | No change needed |
| 6.2 | 116 | `data_analysis.tex` | OK | Pearson correlation caption clear | No change needed |
| 6.3 | 116 | `data_analysis.tex` | OK | SOFA proxy caption clear | No change needed |
| 6.4 | 117 | `data_analysis.tex` | Fixed | Caption used older combined ETHOS+FewShot style | Rewritten as RF, V4 stacking, and ETHOS low-shot baseline |
| 6.5 | 117 | `data_analysis.tex` | OK | Creatinine error caption clear | No change needed |
| 6.6 | 118 | `data_analysis.tex` | OK | t-SNE caption clear | No change needed |
| 6.7 | 119 | `data_analysis.tex` | OK | Prototype illustration caption clear | No change needed |
| 6.8 | 120 | `data_analysis.tex` | OK | Patient timeline caption clear | No change needed |
| ج.1 | 173 | `appendix_c.tex` | OK | Future-work phase caption clear | No change needed |

## Raster Figure Note

Some PNG plot legends still contain English model labels such as `ETHOS+FewShot` because those strings are embedded inside already-generated figures. The LaTeX captions and surrounding prose were corrected. The figures were not regenerated because doing so could unintentionally alter visual assets or invite metric/result drift. This is acceptable for defense because the captions and tables provide the formal terminology.

## Status

Pass. Captions are academic, readable, and directionally safe. Remaining English inside plot legends is a minor visual risk, not a LaTeX caption error.
