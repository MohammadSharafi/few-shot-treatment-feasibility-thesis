#!/usr/bin/env python3
"""
Run the core experiment pipeline in order.

Usage (from repository root):
    python experiments/run_all.py              # full pipeline
    python experiments/run_all.py --figures-only   # only regenerate thesis figures/tables

Extraction scripts require MIMIC-IV v3.1 under data/3.1/ (see config.yaml).
Few-shot / V4–V6 experiments can take hours; they are skipped unless --with-fewshot is passed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_script(rel_path: str) -> int:
    path = ROOT / rel_path
    if not path.exists():
        print(f"Skip (missing): {rel_path}", file=sys.stderr)
        return 0
    print(f"\n>>> {rel_path}")
    return subprocess.run([sys.executable, str(path)], cwd=ROOT).returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Run thesis experiment pipeline")
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Only run scripts/build_thesis_assets.py",
    )
    p.add_argument(
        "--with-fewshot",
        action="store_true",
        help="Also run long few-shot experiments (exp_fewshot_*.py)",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction/01–05 (use existing data/processed)",
    )
    args = p.parse_args()

    if args.figures_only:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_thesis_assets.py")],
            cwd=ROOT,
        ).returncode

    steps: list[str] = []
    if not args.skip_extraction:
        steps.extend(
            [
                "extraction/01_cohort.py",
                "extraction/02_symptoms.py",
                "extraction/03_labels.py",
                "extraction/04_tokenize.py",
                "extraction/05_federated_split.py",
            ]
        )

    steps.extend(
        [
            "experiments/exp1_main.py",
            "experiments/exp2_shots.py",
            "experiments/exp3_federated.py",
            "experiments/exp4_ablation.py",
            "experiments/exp5_generalization.py",
            "experiments/exp_tabular_sota.py",
            "experiments/exp_ethos_rf_hybrid.py",
        ]
    )

    if args.with_fewshot:
        for name in sorted((ROOT / "experiments").glob("exp_fewshot*.py")):
            rel = str(name.relative_to(ROOT))
            if "sweep" not in rel:  # optional: skip very long sweeps
                steps.append(rel)

    rc = 0
    for s in steps:
        c = run_script(s)
        if c != 0:
            rc = c
            print(f"Failed: {s} (exit {c})", file=sys.stderr)

    # Always refresh figures from available CSVs
    b = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_thesis_assets.py")],
        cwd=ROOT,
    )
    if b.returncode != 0:
        rc = b.returncode

    return rc


if __name__ == "__main__":
    sys.exit(main())
