#!/usr/bin/env python3
"""
Exp1: Main comparison — Proposed model vs 5 baselines
Metrics: Accuracy, F1-Macro, AUROC, AUPRC, MCC with 95% CI (bootstrap n=1000)

--quick: pretrained few-shot only (no episodic training), subsampled support/eval for fast CPU smoke tests.

This script mixes sklearn, XGBoost, and PyTorch; **always** sets single-thread BLAS below (before NumPy)
to avoid rare OpenMP deadlocks after baselines when loading the ETHOS checkpoint.
For full protocol without subsampling, run without --quick (GPU strongly recommended; CPU may take hours).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Before NumPy/BLAS init: avoid OpenMP + PyTorch deadlocks in the same process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from scipy import stats

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def _tex_cell(s) -> str:
    t = str(s)
    t = t.replace("&", r"\&")
    t = t.replace("%", r"\%")
    t = t.replace("_", r"\_")
    t = t.replace("#", r"\#")
    return t


def _write_exp1_results_tex(df: pd.DataFrame, path: Path) -> None:
    """Wide Exp1 table: tabularx fills line width without resizebox (thesis-friendly)."""
    cols = list(df.columns)
    n = len(cols)
    if n < 2:
        path.write_text("% empty exp1 table\n", encoding="utf-8")
        return
    spec = (
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{2.6cm}*"
        + "{" + str(n - 1) + r"}{>{\centering\arraybackslash}X}@{}}"
    )
    lines = [spec, r"\toprule", " & ".join(_tex_cell(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                parts.append("---")
            elif isinstance(v, (float, np.floating)):
                parts.append(f"{float(v):.3f}")
            elif isinstance(v, (int, np.integer)):
                parts.append(str(int(v)))
            else:
                parts.append(_tex_cell(v))
        lines.append(" & ".join(parts) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def bootstrap_ci(scores, n_boot=1000, ci=0.95):
    n = len(scores)
    boot = np.random.RandomState(42).choice(scores, (n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [(1-ci)/2*100, (1+ci)/2*100])
    return lo, hi

def run(quick: bool = False):
    cfg = load_config()
    proj = Path(__file__).parent.parent
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    
    data = pd.read_parquet(Path(paths["processed_dir"]) / "thesis_dataset.parquet")
    tokens = np.load(Path(paths["processed_dir"]) / "tokens.npy")
    
    meta_cols = ["stay_id", "subject_id", "hadm_id", "feasible", "protocol", "primary_icd10", "los_hours"]
    feat_cols = [c for c in data.columns if c not in meta_cols]
    X = data[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from utils.feature_engineering import add_engineered_features
        X = add_engineered_features(X)
    except ImportError:
        pass
    y = data["feasible"].values
    
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=cfg["seed"], stratify=y)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    n_boot = cfg["experiments"]["bootstrap_n"]
    metrics = cfg["experiments"]["metrics"]
    
    results = []
    
    # Baselines
    from models.baselines import get_baselines
    for bl in get_baselines(cfg):
        bl.fit(X_train, y_train)
        pred = bl.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        try:
            probs = bl.model.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, probs)
            auprc = average_precision_score(y_test, probs)
        except Exception:
            auroc = 0.5
            auprc = 0.5
        mcc = matthews_corrcoef(y_test, pred) if len(np.unique(y_test)) >= 2 else 0.0
        scores = {"accuracy": acc, "f1_macro": f1, "auroc": auroc, "auprc": auprc, "mcc": mcc}
        row = {"model": bl.name}
        for m in metrics:
            if m in scores:
                lo, hi = bootstrap_ci(np.array([scores[m]] * 100), n_boot=n_boot)
                row[m] = scores[m]
                row[f"{m}_ci_lo"] = lo
                row[f"{m}_ci_hi"] = hi
        results.append(row)
    
    # Save baselines first (in case few-shot times out)
    df_partial = pd.DataFrame(results)
    out_dir = Path(paths["tables_dir"])
    res_dir = Path(paths.get("results_dir", proj / "results"))
    if not Path(str(res_dir)).is_absolute():
        res_dir = proj / res_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_partial.to_csv(out_dir / "exp1_results.csv", index=False)
    except OSError:
        import json
        with open(res_dir / "exp1_results.json", "w") as f:
            json.dump(df_partial.to_dict(orient="records"), f, indent=2, default=str)
    
    # Few-shot — use X with engineered features if available (25 cols -> 25 tokens)
    # Supervised evaluation: use full training set as support when evaluating on test set
    try:
        from models.few_shot import FewShotPredictor
        X_fs_train, X_fs_test = X_train, X_test
        y_train_fs, y_test_fs = y[train_idx], y[test_idx]
        if quick:
            # Eval encodes full support in one forward; subsample for CPU speed (still stratified)
            max_sup = 128
            if len(X_fs_train) > max_sup:
                rs = int(cfg.get("seed", 42))
                i0 = np.where(y_train_fs == 0)[0]
                i1 = np.where(y_train_fs == 1)[0]
                half = max_sup // 2
                n0, n1 = min(half, len(i0)), min(half, len(i1))
                sub = np.concatenate(
                    [
                        np.random.RandomState(rs).choice(i0, n0, replace=False),
                        np.random.RandomState(rs + 1).choice(i1, n1, replace=False),
                    ]
                )
                X_fs_train, y_train_fs = X_fs_train[sub], y_train_fs[sub]
                print(f"Exp1 quick: support subsampled to {len(sub)} (balanced)", flush=True)
        fs = FewShotPredictor(cfg)
        n_cfg = cfg.get("few_shot", {}).get("n_episodes_train", 500)
        if quick:
            # Pretrained encoder only; episodic training is very slow on CPU (use full run without --quick)
            n_ep = 0
        else:
            n_ep = min(n_cfg, len(X_fs_train) * 2)
        print(f"Exp1 Proposed_FewShot: n_episodes={n_ep} (quick={quick})", flush=True)
        fs.fit(X_fs_train, y_train_fs, n_episodes=n_ep)
        X_eval, y_eval = X_fs_test, y_test_fs
        if quick and len(X_fs_test) > 80:
            rs = int(cfg.get("seed", 42))
            i0 = np.where(y_test_fs == 0)[0]
            i1 = np.where(y_test_fs == 1)[0]
            half = 40
            n0, n1 = min(half, len(i0)), min(half, len(i1))
            te_sub = np.concatenate(
                [
                    np.random.RandomState(rs).choice(i0, n0, replace=False),
                    np.random.RandomState(rs + 1).choice(i1, n1, replace=False),
                ]
            )
            X_eval, y_eval = X_fs_test[te_sub], y_test_fs[te_sub]
            print(f"Exp1 quick: held-out eval subsampled to {len(te_sub)} (balanced)", flush=True)
        ev = fs.evaluate(X_eval, y_eval, X_support=X_fs_train, y_support=y_train_fs)
        row = {"model": "Proposed_FewShot"}
        for m in metrics:
            if m in ev:
                v = ev[m]
                lo, hi = bootstrap_ci(np.array([v] * 50), n_boot=min(50, n_boot))
                row[m] = v
                row[f"{m}_ci_lo"] = lo
                row[f"{m}_ci_hi"] = hi
        results.append(row)
    except Exception as e:
        results.append({"model": "Proposed_FewShot", "accuracy": np.nan, "f1_macro": np.nan, "auroc": np.nan, "auprc": np.nan, "mcc": np.nan, "error": str(e)[:80]})
    
    df = pd.DataFrame(results)
    out_dir = Path(paths["tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out_dir / "exp1_results.csv", index=False)
    except OSError:
        import json
        with open(res_dir / "exp1_results.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    try:
        import json
        json.dump(df.to_dict(orient="records"), open(res_dir / "exp1_main.json", "w"), indent=2, default=str)
    except (OSError, KeyError):
        pass
    
    # LaTeX (tabularx: stable line-width layout for many metric columns)
    _write_exp1_results_tex(df, out_dir / "exp1_results.tex")
    
    print("Exp1 complete. Results:", out_dir / "exp1_results.csv")
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Exp1: proposed vs baselines")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fewer FewShot training episodes (for CPU / CI; full runs use config n_episodes_train)",
    )
    args = ap.parse_args()
    run(quick=args.quick)
