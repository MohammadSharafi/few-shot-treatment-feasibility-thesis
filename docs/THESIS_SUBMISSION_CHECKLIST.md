# Thesis submission checklist (Thesis-curser)

Use this list before you upload / print. **Automated** items can be run from the repo; **manual** items depend on your university.

## Automated (run locally)

1. **Regenerate tables for LaTeX**
   ```bash
   python results/generate_all_outputs.py
   ```
2. **Figures / path sanity**
   ```bash
   python results/validate_thesis_readiness.py
   ```
3. **Build the final Persian PDF** (XeLaTeX; from repo root)
   ```bash
   cd thesis && ./build_thesis_fa.sh
   ```
4. **Defense smoke test**
   ```bash
   python scripts/defense_demo.py
   ```
5. **Optional:** full experiment pass (long) — see `README.md` and `scripts/run_value_experiments.py`.

## Numbers and narrative

- **Canonical metrics:** `results/KEY_FINDINGS.md`, `results/exp_fewshot_best.json`, `results/tables/`, and generated `ALL_TABLES*.tex`.
- **Frozen headline:** tabular SOTA test AUROC **0.748**; `exp_fewshot_best` stacked **0.746** [0.699, 0.797]; few-shot-only BEST **0.670**; Exp1 `Proposed_FewShot` **0.611** vs RF **0.762** (same split).
- **V1–V6 tables** in the thesis are **development / historical** unless the same value appears in the primary tables (see reader note in Chapter Results).
- **Scientific boundary:** do not claim that raw few-shot beats RF. The defensible claim is that the **stacked frozen pipeline** reaches 0.746, very close to the tabular ceiling 0.748, while raw/proposed few-shot is lower.
- **Clinical boundary:** claims are predictive, not causal. Economic impact, ROI, temporal models, and multitask learning are framed as future validation paths unless backed by a frozen result table.

## Manual (institution / you)

- [ ] Supervisor final read and sign-off  
- [ ] University logo, commitment page, title page, abstract, and front matter match the faculty template  
- [ ] Abstract word limit, keywords, and Persian/English title pages checked  
- [ ] Plagiarism / similarity tool if required  
- [ ] Final PDF opened manually and checked around wide tables, mixed Persian/English lines, and page breaks  
- [ ] Library deposit format (PDF/A, metadata)  
- [ ] Print/bind specifications  
- [ ] Submission portal deadlines and file naming  

## Snapshot (optional)

After a successful build and sign-off:

```bash
git tag -a thesis-submission-YYYY-MM-DD -m "Thesis submission snapshot"
```

## Environment record

- `requirements.txt` — intended dependencies  
- `requirements-frozen.txt` — `pip freeze` from the environment used for final numbers (regenerate before submission if you upgrade packages)
