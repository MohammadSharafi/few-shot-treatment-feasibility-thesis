#!/usr/bin/env python3
"""
Generate all assets needed for LaTeX thesis compilation.

Run from repository root:
    python scripts/build_thesis_assets.py

Order matters:
  1. scripts/generate_thesis_figures.py — figures cited in results.tex (fig16–fig25) + placeholders
  2. generate_figures.py (repo root) — overwrites fig17, fig26–fig38 with richer plots
  3. results/generate_all_outputs.py — ALL_TABLES.tex and KEY_FINDINGS.md from CSVs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(desc: str, argv: list[str], cwd: Path) -> int:
    print(f"\n=== {desc} ===")
    print(" ", " ".join(argv))
    r = subprocess.run(argv, cwd=cwd)
    if r.returncode != 0:
        print(f"Warning: step exited with code {r.returncode}", file=sys.stderr)
    return r.returncode


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "ERROR: matplotlib is required. Create a venv and run:\n"
            "  python -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python scripts/build_thesis_assets.py",
            file=sys.stderr,
        )
        return 1
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)

    py = sys.executable
    codes = []
    codes.append(
        run_step(
            "Thesis figures (fig16–25 + placeholders)",
            [py, str(root / "scripts" / "generate_thesis_figures.py")],
            root,
        )
    )
    codes.append(
        run_step(
            "Extended figures (fig17, 26–38)",
            [py, str(root / "generate_figures.py")],
            root,
        )
    )
    codes.append(
        run_step(
            "Tables / KEY_FINDINGS from experiment CSVs",
            [py, str(root / "results" / "generate_all_outputs.py")],
            root,
        )
    )

    print("\nDone. Figures: results/figures/")
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
