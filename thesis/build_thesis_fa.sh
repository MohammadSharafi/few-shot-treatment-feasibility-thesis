#!/usr/bin/env bash
# ساخت کامل thesis_fa.pdf — فقط XeLaTeX
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v xelatex >/dev/null 2>&1; then
  for texbin in /usr/local/texlive/*/bin/universal-darwin; do
    if [ -x "$texbin/xelatex" ]; then
      export PATH="$texbin:$PATH"
    fi
  done
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "ERROR: xelatex not found. Install MacTeX/TeX Live or add TeX Live to PATH." >&2
  exit 1
fi

BIBTEX_BIN=""
if command -v bibtex >/dev/null 2>&1; then
  BIBTEX_BIN="bibtex"
elif command -v bibtex.original >/dev/null 2>&1; then
  BIBTEX_BIN="bibtex.original"
else
  echo "ERROR: bibtex not found. Install MacTeX/TeX Live or add TeX Live to PATH." >&2
  exit 1
fi

echo ">>> Engine: $(command -v xelatex)"
echo ">>> BibTeX: $(command -v "$BIBTEX_BIN")"

# First XeLaTeX pass may return non-zero before BibTeX/cross-references exist.
# The final pass reads the generated .bbl and refreshes cross-references.
xelatex -file-line-error -interaction=batchmode thesis_fa.tex || true
"$BIBTEX_BIN" thesis_fa || true
xelatex -file-line-error -interaction=batchmode thesis_fa.tex
xelatex -file-line-error -interaction=batchmode thesis_fa.tex
test -f thesis_fa.pdf
echo ">>> Done: $(pwd)/thesis_fa.pdf"
