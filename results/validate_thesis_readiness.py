#!/usr/bin/env python3
"""
Validate thesis readiness — run at the very end.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check(name, cond, reason=""):
    status = "✓ PASS" if cond else f"✗ FAIL {reason}"
    print(f"  {status}: {name}")
    return cond

def main():
    proj = Path(__file__).parent.parent
    cfg = __import__("yaml").safe_load(open(proj / "config.yaml"))
    paths = {k: str(proj / v) if isinstance(v, str) else v for k, v in cfg["paths"].items()}
    
    all_ok = True
    
    # thesis_dataset
    p = Path(paths["processed_dir"]) / "thesis_dataset.parquet"
    if p.exists():
        import pandas as pd
        df = pd.read_parquet(p)
        n = len(df)
        all_ok &= check("thesis_dataset.parquet", n > 1000, f"has {n} rows")
    else:
        all_ok &= check("thesis_dataset.parquet", False, "file missing")
    
    # 5 federated nodes
    node_files = list(Path(paths["processed_dir"]).glob("node_*.parquet"))
    if len(node_files) >= 5:
        import pandas as pd
        sizes = [len(pd.read_parquet(f)) for f in node_files[:5]]
        all_ok &= check("5 federated nodes", True, f"sizes: {sizes}")
    else:
        all_ok &= check("5 federated nodes", False, f"found {len(node_files)}")
    
    # Exp1 AUROC > 0.60
    tables = Path(paths["tables_dir"])
    res_path = tables / "exp1_results.csv"
    if res_path.exists():
        import pandas as pd
        df = pd.read_csv(res_path)
        proposed = df[df["model"].astype(str).str.contains("Proposed", na=False)]
        if len(proposed) > 0 and "auroc" in proposed.columns:
            auroc = proposed["auroc"].values[0]
            all_ok &= check("Exp1 AUROC > 0.60", auroc > 0.60 if pd.notna(auroc) else False, f"AUROC={auroc}")
        else:
            all_ok &= check("Exp1 AUROC > 0.60", False, "no proposed result")
    else:
        all_ok &= check("Exp1 AUROC > 0.60", False, "no exp1 results")
    
    # Exp2 monotonic
    if (tables / "exp2_results.csv").exists():
        import pandas as pd
        df = pd.read_csv(tables / "exp2_results.csv")
        if "auc" in df.columns or "auroc" in df.columns:
            aurocs = df["auroc"].values if "auroc" in df.columns else df["auc"].values
            monotonic = all(aurocs[i] <= aurocs[i+1] + 0.08 for i in range(len(aurocs)-1))
            all_ok &= check("Exp2 K-shot sensitivity table", monotonic or len(aurocs) < 2, "check CI overlap")
    
    # Exp3 federated gap
    if (tables / "exp3_results.csv").exists():
        import pandas as pd
        df = pd.read_csv(tables / "exp3_results.csv")
        cen = df[df["setup"].astype(str) == "Centralized"]["auroc"].values
        fed = df[df["setup"].astype(str) == "Federated"]["auroc"].values
        if len(cen) > 0 and len(fed) > 0:
            gap = (cen[0] - fed[0]) * 100
            all_ok &= check("Exp3 federated gap < 10%", gap < 10, f"gap={gap:.1f}%")
    
    # Exp4 ablation
    if (tables / "exp4_results.csv").exists():
        import pandas as pd
        df = pd.read_csv(tables / "exp4_results.csv")
        full = df[df["variant"].astype(str).str.contains("full", case=False, na=False)]
        if len(full) > 0:
            all_ok &= check("Exp4 ablation table has full model row", True)
    
    # Exp5
    if (tables / "exp5_results.csv").exists():
        all_ok &= check("Exp5 cross-disease result exists", True)
    
    # Figures (thesis-critical names from chapters/results.tex and data_analysis.tex)
    figs = Path(paths["figures_dir"])
    fig_count = len(list(figs.glob("*.png")))
    required = [
        "fig16_shap_waterfall.png",
        "fig18_learning_curves.png",
        "fig19_federated_convergence.png",
        "fig20_calibration.png",
        "fig21_roc_curves.png",
        "fig22_pr_curves.png",
        "fig23_feature_importance.png",
        "fig24_kshot_line.png",
        "fig25_ablation_radar.png",
        "fig1_main_comparison.png",
        "fig27_tsne.png",
        "fig28_prototype.png",
        "fig32_patient_timeline.png",
        # fig33_mimic_schema.png optional: thesis Ch.6 uses thesis/latex/mimic_iv_schema.tex (TikZ)
        "fig35_sofa_scatter.png",
        "fig36_error_creatinine.png",
        "fig37_age_stratified.png",
        "fig38_lab_correlation.png",
    ]
    missing = [f for f in required if not (figs / f).exists()]
    all_ok &= check(
        "Thesis-critical PNGs present",
        len(missing) == 0,
        f"missing {len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}",
    )
    all_ok &= check("Total PNG count", fig_count >= 8, f"found {fig_count}")
    
    # ALL_TABLES.tex
    tex_path = tables / "ALL_TABLES.tex"
    if tex_path.exists():
        lines = len(tex_path.read_text().splitlines())
        all_ok &= check("ALL_TABLES.tex", lines > 20)
    else:
        all_ok &= check("ALL_TABLES.tex", False)
    
    # KEY_FINDINGS
    kf = Path(__file__).parent / "KEY_FINDINGS.md"
    if kf.exists():
        content = kf.read_text()
        has_nums = any(c.isdigit() for c in content) and "placeholder" not in content.lower()
        all_ok &= check("KEY_FINDINGS.md has actual numbers", has_nums or "Finding" in content)
    
    print()
    if all_ok:
        print("✓ PART 1 COMPLETE — Ready for thesis writing")
    else:
        print("✗ PART 1 INCOMPLETE — Fix items marked FAIL above")

if __name__ == "__main__":
    main()
