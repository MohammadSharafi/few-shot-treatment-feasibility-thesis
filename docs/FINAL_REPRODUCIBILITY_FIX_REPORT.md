# Final Reproducibility Fix Report

Date: 2026-05-25

Official Persian title preserved exactly:

> فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده

## 1. Initial Failure Reproduced

The original clean rebuild path was not reproducible. The observed blocker was reproduced from the project root:

```bash
make clean-temp
make thesis-fa
```

The first failing cause was that `xelatex` was not found on the default shell `PATH`, even though TeX Live was installed under `/usr/local/texlive/2025/bin/universal-darwin`. The previous cleanup target was also unsafe because it used a full LaTeX cleanup pattern capable of removing final PDF artifacts.

## 2. Root Causes

- The LaTeX toolchain path was not exported by the build system, so XeLaTeX/BibTeX were unavailable in a fresh shell.
- `clean-temp` behaved like a destructive cleanup target and could remove final PDFs.
- The Persian thesis build did not consistently enforce the full XeLaTeX/BibTeX/XeLaTeX/XeLaTeX sequence.
- Several XePersian-sensitive Unicode forms existed in source text, especially Arabic hamza-above ezafe forms.
- PDF string handling for mixed Persian/English commands needed safer Hyperref definitions.
- The validation script treated some critical reproducibility checks too softly.
- Defense slides were not guaranteed to be regenerated as part of the build path.
- Some final reports contained stale success language before the clean rebuild was actually passing.

## 3. Files Changed

- `Makefile`
- `scripts/validate_results.py`
- `thesis/build_thesis_fa.sh`
- `thesis/thesis_fa.tex`
- `thesis/defense_slides.tex`
- `thesis/chapters_fa/experiments.tex`
- Persian thesis source files affected by safe ezafe normalization
- `docs/FINAL_VALIDATION_REPORT.md`
- `docs/LATEX_BUILD_NOTES.md`
- `docs/FINAL_FULL_ACCEPTANCE_REPORT.md`
- `docs/FINAL_AGENT_SUMMARY.md`
- `docs/FINAL_SUBMISSION_CLEANUP_REPORT.md`
- `docs/FINAL_REPRODUCIBILITY_FIX_REPORT.md`

## 4. clean-temp Behavior Fixed

`make clean-temp` now removes only auxiliary/build files such as `.aux`, `.log`, `.toc`, `.lof`, `.lot`, `.out`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, `.xdv`, `.nav`, `.snm`, and `.vrb`.

It no longer removes final PDFs:

- `thesis/thesis_fa.pdf`
- `thesis/defense_slides.pdf`

## 5. thesis-fa Build Fixed

`make thesis-fa` now:

- Adds the detected TeX Live binary directory to `PATH`.
- Requires `xelatex` and `bibtex`.
- Builds the Persian thesis with XeLaTeX, not pdfLaTeX.
- Runs the required citation/reference sequence through `thesis/build_thesis_fa.sh`.
- Verifies `thesis/thesis_fa.pdf`.
- Builds and verifies `thesis/defense_slides.pdf`.

## 6. Missing-Character Warnings Fixed

Problematic Unicode source causes were corrected:

- Arabic hamza-above ezafe forms were replaced with XePersian-safe wording such as `داده‌ی`.
- Hidden left-to-right/right-to-left marker characters were checked and are not present in thesis `.tex` sources.
- Hyperref PDF-string handling was hardened for mixed Persian/English commands.

Final thesis log result:

- Missing-character warnings: `0`

## 7. Citations and References Fixed

The thesis build now runs enough passes after BibTeX to resolve citations and references.

Final thesis log result:

- Undefined citations: `0`
- Undefined references: `0`

## 8. Defense PDF Issue Fixed

`make thesis-fa` now regenerates `thesis/defense_slides.pdf`, and `make defense` checks that both final PDFs exist before reporting success.

Final defense artifact:

- `thesis/defense_slides.pdf`

## 9. Validation Fixed

`scripts/validate_results.py` now fails honestly for critical conditions, including:

- Missing `thesis/thesis_fa.pdf`
- Missing required defense slides PDF
- Missing or stale thesis build log
- Missing-character warnings
- Undefined citations
- Undefined references
- Incorrect PDF metadata title
- Inconsistent official title usage
- Duplicated/conflicting canonical metrics

The validation report now reflects the fresh build state and does not report success unless critical checks pass.

## 10. Final Commands Run

The required sequence was run exactly from the project root:

```bash
make clean-temp
make thesis-fa
make validate
make defense
```

## 11. Final Pass/Fail Status

Final status: `PASS`

- `make clean-temp`: `PASS`
- `make thesis-fa`: `PASS`
- `make validate`: `PASS`
- `make defense`: `PASS`

Validation score after the clean rebuild:

- `100/100`
- Passed checks: `51`
- Warnings: `0`
- Failed checks: `0`

## 12. Final Warning Counts

From `thesis/thesis_fa.log` after the clean rebuild:

- LaTeX errors: `0`
- Missing-character warnings: `0`
- Undefined citations: `0`
- Undefined references: `0`
- LaTeX warnings: `7`
- Package warnings: `4`
- Overfull hbox warnings: `25`
- Underfull hbox warnings: `23`

The remaining hbox warnings are cosmetic layout warnings and are within the current validation tolerance. They are not citation, reference, missing-character, or artifact blockers.

## 13. Final PDF Paths

- `thesis/thesis_fa.pdf`
- `thesis/defense_slides.pdf`

Final artifact sizes:

- `thesis/thesis_fa.pdf`: approximately `1.9M`
- `thesis/defense_slides.pdf`: approximately `50K`

## 14. Remaining Risks

- The thesis log still contains cosmetic overfull/underfull hbox warnings. They do not block reproducibility, but a final visual pass before institutional submission is still recommended.
- The defense slides compile successfully, but Beamer reports cosmetic overfull boxes from theme/navigation layout. This does not block `make defense`.
- The project directory is inside a broader Git repository rooted at `/Users/moe`, so Git status is noisy and not useful for this project unless the thesis directory is isolated as its own repository.

## 15. Final Recommendation

Submit.

The project now rebuilds cleanly from the required command sequence, the Persian thesis PDF is regenerated from source, the defense PDF exists, critical XePersian missing-character warnings are eliminated, citations and references resolve to zero undefined entries, validation is strict and honest, and the official thesis title remains unchanged.
