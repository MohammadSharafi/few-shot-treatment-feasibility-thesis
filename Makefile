# =============================================================================
# Makefile — Few-Shot Treatment Feasibility Prediction
# =============================================================================
# Usage: make <target>
# Run from the project root (directory containing this Makefile).
#
# Requires: Python 3.10+  |  pip install -r requirements.txt
# For thesis compilation: XeLaTeX (xelatex) and BibTeX.
#
# All Python commands use the project venv if .venv/ exists,
# otherwise falls back to the system python3.
# =============================================================================

PYTHON := $(shell [ -f .venv/bin/python ] && echo .venv/bin/python || echo python3)
ROOT    := $(shell pwd)
THESIS  := $(ROOT)/thesis

# macOS TeX Live does not always add binaries to non-interactive shells.
# Prefer the newest installed TeX Live bin directory when it exists.
TEXLIVE_BINS := $(sort $(wildcard /usr/local/texlive/*/bin/universal-darwin))
TEXLIVE_BIN  := $(lastword $(TEXLIVE_BINS))
ifneq ($(TEXLIVE_BIN),)
export PATH := $(TEXLIVE_BIN):$(PATH)
endif

.PHONY: all setup check extract experiments value-experiments assets validate \
        defense thesis-fa clean-temp help

# Default target
all: help

# -----------------------------------------------------------------------------
# setup — install dependencies (print instructions if venv not active)
# -----------------------------------------------------------------------------
setup:
	@echo "=== Setup ==="
	@echo "Recommended: create a virtual environment first:"
	@echo "  python3 -m venv .venv && source .venv/bin/activate"
	@echo ""
	@echo "Installing requirements..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Done. Frozen env: requirements-frozen.txt"
	@echo "To freeze current env: pip freeze > requirements-frozen.txt"

# -----------------------------------------------------------------------------
# check — verify environment, folders, data, and dependencies
# -----------------------------------------------------------------------------
check:
	@echo "=== Environment Check ==="
	$(PYTHON) -c "import sys; print(f'Python: {sys.version}')"
	$(PYTHON) -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
	$(PYTHON) -c "import torch; print(f'torch: {torch.__version__}')"
	$(PYTHON) -c "import pandas; print(f'pandas: {pandas.__version__}')"
	$(PYTHON) -c "import numpy; print(f'numpy: {numpy.__version__}')"
	@echo ""
	@echo "=== Required folders ==="
	@[ -d data/processed ]  && echo "  ✓ data/processed"   || echo "  ✗ data/processed MISSING"
	@[ -d results ]          && echo "  ✓ results"          || echo "  ✗ results MISSING"
	@[ -d results/figures ]  && echo "  ✓ results/figures"  || echo "  ✗ results/figures MISSING"
	@[ -d results/tables ]   && echo "  ✓ results/tables"   || echo "  ✗ results/tables MISSING"
	@[ -d thesis ]           && echo "  ✓ thesis"           || echo "  ✗ thesis MISSING"
	@echo ""
	@echo "=== Key data files ==="
	@[ -f data/processed/thesis_dataset.parquet ] && echo "  ✓ thesis_dataset.parquet" || echo "  ✗ thesis_dataset.parquet MISSING — run: make extract"
	@[ -f data/processed/tokens.npy ]             && echo "  ✓ tokens.npy"             || echo "  ✗ tokens.npy MISSING — run: make extract"
	@echo ""
	@echo "=== XeLaTeX (for thesis compilation) ==="
	@command -v xelatex >/dev/null 2>&1 && echo "  ✓ xelatex found" || echo "  ✗ xelatex NOT FOUND — install MacTeX or TeX Live"
	@(command -v bibtex >/dev/null 2>&1 || command -v bibtex.original >/dev/null 2>&1) && echo "  ✓ bibtex found" || echo "  ✗ bibtex NOT FOUND"

# -----------------------------------------------------------------------------
# extract — run extraction pipeline (01 → 05)
# Requires: MIMIC-IV v3.1 data under data/3.1/ (see docs/REPRODUCIBILITY_GUIDE.md)
# -----------------------------------------------------------------------------
extract:
	@echo "=== Extraction Pipeline ==="
	@echo "NOTE: Requires MIMIC-IV v3.1 credentials and data under data/3.1/"
	@echo "See docs/REPRODUCIBILITY_GUIDE.md for setup instructions."
	@echo ""
	$(PYTHON) extraction/01_cohort.py
	$(PYTHON) extraction/02_symptoms.py
	$(PYTHON) extraction/03_labels.py
	$(PYTHON) extraction/04_tokenize.py
	$(PYTHON) extraction/05_federated_split.py
	@echo ""
	@echo "Extraction complete. Data written to data/processed/"

# -----------------------------------------------------------------------------
# experiments — run all core experiments (uses existing processed data)
# -----------------------------------------------------------------------------
experiments:
	@echo "=== Core Experiments ==="
	$(PYTHON) -u experiments/run_all.py --skip-extraction
	@echo ""
	@echo "Core experiments complete. Results in results/tables/ and results/figures/"

# -----------------------------------------------------------------------------
# value-experiments — run high-value experiments for thesis credibility
# Uses --quick to skip long ETHOS blocks; remove for full run
# -----------------------------------------------------------------------------
value-experiments:
	@echo "=== High-Value Experiments ==="
	@echo "Running orchestrated pipeline (--quick mode; ~20-40 min on CPU)..."
	@echo "For full run (hours): $(PYTHON) scripts/run_value_experiments.py --figures"
	$(PYTHON) -u scripts/run_value_experiments.py --quick --figures
	@echo ""
	@echo "High-value experiments complete."

# -----------------------------------------------------------------------------
# assets — regenerate all thesis figures and LaTeX tables
# Safe to re-run at any time; does NOT change result files.
# -----------------------------------------------------------------------------
assets:
	@echo "=== Build Thesis Assets ==="
	$(PYTHON) scripts/build_thesis_assets.py
	@echo ""
	@echo "Assets regenerated:"
	@echo "  Figures → results/figures/"
	@echo "  Tables  → results/tables/"

# -----------------------------------------------------------------------------
# validate — validate metrics, figures, tables, and thesis consistency
# Writes: docs/FINAL_VALIDATION_REPORT.md
# -----------------------------------------------------------------------------
validate:
	@echo "=== Validate Results ==="
	$(PYTHON) scripts/validate_results.py --write-report
	@echo ""
	@echo "Validation report written to docs/FINAL_VALIDATION_REPORT.md"

# -----------------------------------------------------------------------------
# defense — run the one-command defense demo
# Writes: results/DEFENSE_RUN_SUMMARY.md
# -----------------------------------------------------------------------------
defense:
	@echo "=== Defense Demo ==="
	@[ -f $(THESIS)/thesis_fa.pdf ] || (echo "ERROR: thesis/thesis_fa.pdf missing. Run: make thesis-fa" && exit 1)
	@[ -f $(THESIS)/defense_slides.pdf ] || (echo "ERROR: thesis/defense_slides.pdf missing. Run: make thesis-fa" && exit 1)
	$(PYTHON) scripts/defense_demo.py
	@echo ""
	@echo "Defense summary written to results/DEFENSE_RUN_SUMMARY.md"

# -----------------------------------------------------------------------------
# thesis-fa — compile the Persian thesis PDF (and defense slides) with XeLaTeX
# Requires: XeLaTeX + fonts in thesis/fonts/
# -----------------------------------------------------------------------------
thesis-fa:
	@echo "=== Compile Persian Thesis (XeLaTeX) ==="
	@command -v xelatex >/dev/null 2>&1 || (echo "ERROR: xelatex not found. Install MacTeX or TeX Live." && exit 1)
	@(command -v bibtex >/dev/null 2>&1 || command -v bibtex.original >/dev/null 2>&1) || (echo "ERROR: bibtex not found. Install MacTeX or TeX Live." && exit 1)
	cd $(THESIS) && ./build_thesis_fa.sh
	@echo ""
	@echo "Compiled: thesis/thesis_fa.pdf"
	@test -f $(THESIS)/thesis_fa.pdf || (echo "ERROR: thesis/thesis_fa.pdf was not produced" && exit 1)
	@ls -lh $(THESIS)/thesis_fa.pdf
	@echo ""
	@echo "=== Compile Defense Slides (XeLaTeX) ==="
	cd $(THESIS) && xelatex -file-line-error -interaction=nonstopmode defense_slides.tex
	cd $(THESIS) && xelatex -file-line-error -interaction=nonstopmode defense_slides.tex
	@test -f $(THESIS)/defense_slides.pdf || (echo "ERROR: thesis/defense_slides.pdf was not produced" && exit 1)
	@ls -lh $(THESIS)/defense_slides.pdf

# -----------------------------------------------------------------------------
# clean-temp — remove LaTeX auxiliary files and Python caches
# Does NOT delete: results/, data/, models/*.pt, models/*.joblib
# -----------------------------------------------------------------------------
clean-temp:
	@echo "=== Clean Temporary Files ==="
	@echo "Removing LaTeX aux/build files; final PDFs are preserved..."
	find $(THESIS) \( -name "*.aux" -o -name "*.log" -o -name "*.toc" -o -name "*.lot" \
		-o -name "*.lof" -o -name "*.out" -o -name "*.nav" -o -name "*.snm" \
		-o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.bbl" -o -name "*.blg" \
		-o -name "*.synctex.gz" -o -name "*.xdv" -o -name "*.vrb" \) \
		-not -path "*/.git/*" -print0 | xargs -0 rm -f 2>/dev/null || true
	@echo "Removing Python caches..."
	find $(ROOT) -type d -name "__pycache__" -not -path "*/.venv/*" | xargs rm -rf 2>/dev/null || true
	find $(ROOT) -name "*.pyc" -not -path "*/.venv/*" | xargs rm -f 2>/dev/null || true
	@echo "Removing CatBoost artifacts..."
	rm -rf $(ROOT)/catboost_info 2>/dev/null || true
	@echo ""
	@echo "Clean complete. Results and data preserved."

# -----------------------------------------------------------------------------
# help — show available targets
# -----------------------------------------------------------------------------
help:
	@echo ""
	@echo "Few-Shot Treatment Feasibility — Project Makefile"
	@echo "=================================================="
	@echo ""
	@echo "  make setup            Install Python dependencies"
	@echo "  make check            Verify environment, folders, data, and XeLaTeX"
	@echo "  make extract          Run MIMIC-IV extraction pipeline (01→05)"
	@echo "  make experiments      Run all core experiments (needs processed data)"
	@echo "  make value-experiments Run high-value experiments (--quick mode)"
	@echo "  make assets           Regenerate thesis figures and LaTeX tables"
	@echo "  make validate         Validate metrics + consistency → FINAL_VALIDATION_REPORT.md"
	@echo "  make defense          Run one-command defense demo"
	@echo "  make thesis-fa        Compile Persian thesis PDF with XeLaTeX"
	@echo "  make clean-temp       Remove LaTeX aux files and Python caches"
	@echo ""
	@echo "Recommended defense-day sequence:"
	@echo "  make check && make validate && make defense"
	@echo ""
	@echo "Full reproducibility pipeline (hours):"
	@echo "  make check && make extract && make experiments && make assets && make validate"
	@echo ""
	@echo "See docs/REPRODUCIBILITY_GUIDE.md for detailed instructions."
	@echo ""
