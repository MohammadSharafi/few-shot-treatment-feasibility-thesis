#!/usr/bin/env python3
"""
High-value experiment pipeline for thesis credibility and accuracy.

Run from repository root with a venv and MIMIC-IV processed data:
    pip install -r requirements.txt
    python -u scripts/run_value_experiments.py --quick --figures

With --quick, diagnostics/performance_breakdown.py is also run with --quick
(skips slow ETHOS+Few-Shot there; still covered by exp_fewshot_best).

Use -u (unbuffered) so progress prints immediately.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(rel: str, extra_args: list[str] | None, step: int, total: int) -> int:
    path = ROOT / rel
    if not path.exists():
        print(f"[{_ts()}] [{step}/{total}] SKIP (missing): {rel}", flush=True)
        return 0
    # -u = unbuffered Python; child scripts print line-by-line
    cmd = [sys.executable, "-u", str(path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n[{_ts()}] >>> [{step}/{total}] START", flush=True)
    print("    " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        print(f"[{_ts()}] [{step}/{total}] FAIL exit {r.returncode}: {rel}", file=sys.stderr, flush=True)
    else:
        print(f"[{_ts()}] [{step}/{total}] OK: {rel}", flush=True)
    return r.returncode


def _check_deps() -> bool:
    try:
        import pandas  # noqa: F401
    except ImportError:
        print(
            "ERROR: Dependencies missing. Create a venv and install:\n"
            "  python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate\n"
            "  pip install -r requirements.txt\n"
            "Then run with unbuffered output:\n"
            "  PYTHONUNBUFFERED=1 python -u scripts/run_value_experiments.py --quick --figures",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def main() -> int:
    if not _check_deps():
        return 1

    ap = argparse.ArgumentParser(description="Run high-value thesis experiments")
    ap.add_argument("--quick", action="store_true", help="exp_fewshot_best.py --quick")
    ap.add_argument("--with-extraction", action="store_true", help="Run extraction/01–05 first")
    ap.add_argument("--skip-sota", action="store_true", help="Skip exp_tabular_sota (use exp_baseline_strong)")
    ap.add_argument("--skip-core", action="store_true", help="Skip exp1–5")
    ap.add_argument(
        "--skip-exp1",
        action="store_true",
        help="Skip exp1_main.py (slow ETHOS training on CPU; run separately overnight if needed)",
    )
    ap.add_argument(
        "--with-fewshot-full",
        action="store_true",
        help="Full exp_fewshot_best (no --quick)",
    )
    ap.add_argument("--figures", action="store_true", help="Run build_thesis_assets.py at end")
    ap.add_argument(
        "--skip-fewshot",
        action="store_true",
        help="Skip exp_fewshot_best.py (needs PyTorch; use if GPU/torch not installed)",
    )
    args = ap.parse_args()

    # Build flat list: (relpath, extra_args or None)
    planned: list[tuple[str, list[str] | None]] = []

    if args.with_extraction:
        for s in [
            "extraction/01_cohort.py",
            "extraction/02_symptoms.py",
            "extraction/03_labels.py",
            "extraction/04_tokenize.py",
            "extraction/05_federated_split.py",
        ]:
            planned.append((s, None))

    if not args.skip_sota:
        planned.append(("experiments/exp_tabular_sota.py", None))
    else:
        planned.append(("experiments/archive/exp_baseline_strong.py", None))

    planned.append(("experiments/exp_ethos_rf_hybrid.py", None))

    if not args.skip_core:
        core = [
            "experiments/exp1_main.py",
            "experiments/exp2_shots.py",
            "experiments/exp3_federated.py",
            "experiments/exp4_ablation.py",
            "experiments/exp5_generalization.py",
        ]
        if args.skip_exp1:
            core = [s for s in core if "exp1_main" not in s]
            print(f"[{_ts()}] Note: skipping exp1_main.py (--skip-exp1)", flush=True)
        for s in core:
            xtra = ["--quick"] if args.quick and "exp1_main" in s else None
            planned.append((s, xtra))

    diag_pb_args = ["--quick"] if args.quick else None
    planned.append(("diagnostics/performance_breakdown.py", diag_pb_args))
    planned.append(("diagnostics/statistical_validation.py", None))

    if not args.skip_fewshot:
        fs_args: list[str] = [] if args.with_fewshot_full else ["--quick"]
        planned.append(("experiments/exp_fewshot_best.py", fs_args if fs_args else None))

    if args.figures:
        planned.append(("scripts/build_thesis_assets.py", None))

    total = len(planned)
    print(
        f"[{_ts()}] Pipeline: {total} step(s). Root={ROOT}",
        flush=True,
    )

    rc = 0
    for i, (rel, xargs) in enumerate(planned, start=1):
        c = run(rel, xargs, i, total)
        if c != 0:
            rc = c

    print(f"\n[{_ts()}] Finished. Last exit code (0=all ok): cumulative failures tracked as rc={rc}", flush=True)
    print("Outputs: results/tables/, results/*.json, diagnostics/, results/figures/", flush=True)
    if rc != 0:
        print("Some steps failed; scroll up for [FAIL].", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
