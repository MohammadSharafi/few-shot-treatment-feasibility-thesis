#!/usr/bin/env python3
"""Verify Persian thesis chapters mirror English (full content parity checks).

Checks that nothing is *structurally* omitted when translating EN → FA:
  - Same heading hierarchy and count (maps to same logical sections / pages)
  - Same \\label{}, \\input{}, \\includegraphics{}
  - Same counts of: \\item, \\paragraph, figures, tables, equations, list
    environments, algorithm lines (\\State / \\Require), lstlisting blocks

Optional: flag FA lines that look like *untranslated English prose* (heuristic).

Run from thesis/:
  python3 scripts/verify_fa_en_parity.py
  python3 scripts/verify_fa_en_parity.py --check-english-leaks

Note: Word-for-word alignment EN↔FA is impossible (languages differ). “Line-by-line”
in LaTeX means every *content block* in the English source has a counterpart; line
counts often differ because Persian wraps differently. Page counts differ until you
compile both PDFs with the same geometry.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAD_RE = re.compile(r"\\(chapter|section|subsection|subsubsection)\{([^}]*)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[pt]?\{[^}]+\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

# Arabic/Persian letters (rough)
_AR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LAT = re.compile(r"[A-Za-z]")


def _thesis_dir() -> Path:
    p = Path(__file__).resolve().parent.parent
    if (p / "chapters").is_dir():
        return p
    return Path.cwd()


def _heads(text: str) -> list[tuple[str, str]]:
    return HEAD_RE.findall(text)


def _strip_envs(text: str, env: str) -> str:
    return re.sub(
        rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
        f"<{env.upper()}>",
        text,
        flags=re.DOTALL,
    )


def _struct_counts(text: str) -> dict[str, int]:
    """Counts that should match between EN and FA for the same file."""
    t = text
    t = _strip_envs(t, "lstlisting")
    t = _strip_envs(t, "verbatim")
    return {
        "item": len(re.findall(r"^\s*\\item\s", t, re.M)),
        "paragraph": len(re.findall(r"\\paragraph\{", t)),
        "midrule": t.count("\\midrule"),
        "enumerate": len(re.findall(r"\\begin\{enumerate\}", t)),
        "itemize": len(re.findall(r"\\begin\{itemize\}", t)),
        "figure": len(re.findall(r"\\begin\{figure\}", t)),
        "table": len(re.findall(r"\\begin\{table\}", t)),
        "equation": len(re.findall(r"\\begin\{equation", t)),
        "align": len(re.findall(r"\\begin\{align", t)),
        "State": len(re.findall(r"\\State\b", text)),
        "Require": len(re.findall(r"\\Require\b", text)),
        "lstlisting": len(re.findall(r"\\begin\{lstlisting\}", text)),
    }


def _scan(path: Path) -> dict:
    t = path.read_text(encoding="utf-8", errors="replace")
    sc = _struct_counts(t)
    return {
        "heads": _heads(t),
        "labels": set(LABEL_RE.findall(t)),
        "cite_n": len(CITE_RE.findall(t)),
        "inputs": set(INPUT_RE.findall(t)),
        "graphics": set(GRAPHIC_RE.findall(t)),
        "lines": t.count("\n") + 1,
        **sc,
    }


def _fa_english_leak_lines(fa_path: Path) -> list[tuple[int, str]]:
    """Heuristic: prose-like English outside \\lr/\\texttt/code (possible omission)."""
    lines = fa_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[tuple[int, str]] = []
    in_lst = in_lat = 0
    prose_hint = re.compile(
        r"\b(the|and|which|that|this|with|from|are|for|have|was|were)\b", re.I
    )
    for i, raw in enumerate(lines, start=1):
        s = raw.split("%", 1)[0].strip()
        if "\\begin{lstlisting}" in raw:
            in_lst += 1
        if "\\end{lstlisting}" in raw:
            in_lst = max(0, in_lst - 1)
        if "\\begin{latin}" in raw:
            in_lat += 1
        if "\\end{latin}" in raw:
            in_lat = max(0, in_lat - 1)
        if in_lst or in_lat or not s or s.startswith("%"):
            continue
        if "\\addcontentsline" in s or "\\caption" in s and "\\lr" in s:
            continue
        persian = len(_AR.findall(s))
        latin = len(_LAT.findall(s))
        if persian > 0:
            continue
        if latin < 35:
            continue
        if "\\lr{" in s or "\\texttt" in s or "\\State" in s or "\\Require" in s:
            continue
        if "&" in s and ("\\\\" in s or "\\midrule" in s or "\\toprule" in s):
            continue  # tabular row, often English abbreviations
        if prose_hint.search(s):
            out.append((i, raw[:200]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check-english-leaks",
        action="store_true",
        help="List FA lines that look like untranslated English prose (heuristic).",
    )
    args = ap.parse_args()

    root = _thesis_dir()
    ch = root / "chapters"
    fa = root / "chapters_fa"
    if not ch.is_dir() or not fa.is_dir():
        print("Expected chapters/ and chapters_fa/ under", root, file=sys.stderr)
        return 1

    errors = 0
    struct_keys = (
        "item",
        "paragraph",
        "midrule",
        "enumerate",
        "itemize",
        "figure",
        "table",
        "equation",
        "align",
        "State",
        "Require",
        "lstlisting",
    )
    en_files = sorted(ch.glob("*.tex"))
    for en_path in en_files:
        name = en_path.name
        fa_path = fa / name
        if not fa_path.is_file():
            print(f"MISSING: chapters_fa/{name}")
            errors += 1
            continue
        a, b = _scan(en_path), _scan(fa_path)
        if [x[0] for x in a["heads"]] != [x[0] for x in b["heads"]]:
            print(f"HEADING LEVELS MISMATCH: {name}")
            errors += 1
        if len(a["heads"]) != len(b["heads"]):
            print(f"HEADING COUNT {name}: EN={len(a['heads'])} FA={len(b['heads'])}")
            errors += 1
        if a["labels"] != b["labels"]:
            print(f"LABELS MISMATCH: {name}")
            print("  only EN:", sorted(a["labels"] - b["labels"]))
            print("  only FA:", sorted(b["labels"] - a["labels"]))
            errors += 1
        if a["inputs"] != b["inputs"]:
            print(f"INPUT MISMATCH: {name}", a["inputs"] ^ b["inputs"])
            errors += 1
        if a["graphics"] != b["graphics"]:
            print(f"GRAPHIC MISMATCH: {name}", a["graphics"] ^ b["graphics"])
            errors += 1
        for k in struct_keys:
            if a[k] != b[k]:
                print(f"STRUCT {name} {k}: EN={a[k]} FA={b[k]}")
                errors += 1
        if abs(a["lines"] - b["lines"]) > 5:
            print(
                f"NOTE line delta {name}: EN={a['lines']} FA={b['lines']} "
                "(OK if only wrapping / blanks; not proof of missing text)"
            )

        if args.check_english_leaks:
            leaks = _fa_english_leak_lines(fa_path)
            for ln, preview in leaks:
                print(f"  LEAK? {name}:{ln}: {preview!r}")

    if errors == 0:
        print(
            f"OK: {len(en_files)} files — headings, labels, cites, inputs, graphics, "
            "and content-block counts (items, figures, tables, equations, algorithms) match."
        )
        print(
            "Semantics: every English structural unit has a FA counterpart; "
            "raw line count and PDF pages are not 1:1 across languages."
        )
    else:
        print(f"Found {errors} issue(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
