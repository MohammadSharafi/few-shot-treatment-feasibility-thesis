#!/usr/bin/env python3
"""
Generate all figures, ALL_TABLES.tex, and KEY_FINDINGS.md from experiment results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
import json

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None

def tex_cell(s) -> str:
    """Escape text for LaTeX table cells (underscores in model names, etc.)."""
    t = str(s)
    t = t.replace("&", r"\&")
    t = t.replace("%", r"\%")
    t = t.replace("_", r"\_")
    t = t.replace("#", r"\#")
    return t


DISPLAY_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "StandardNN": "Standard NN",
    "FederatedNN_NoFewShot": "Federated NN (No Few-Shot)",
    "Proposed_FewShot": "Proposed Few-Shot",
    "full_model": "Full model",
    "no_symptom_tokens": "No symptom tokens",
    "no_clinical_bias": "No clinical bias",
    "no_intensity_weighting": "No intensity weighting",
    "no_temporal_decay": "No temporal decay",
    "no_few_shot": "No few-shot head",
    "node_4": "Node 4 (renal)",
    "node_5": "Node 5 (diabetes)",
    "Centralized": "Centralized",
    "Federated": "Federated",
    "Gap": "Gap",
}


def display_label(value) -> str:
    """Use readable display labels in generated appendix tables."""
    return DISPLAY_LABELS.get(str(value), str(value))


def interpret_exp1(proposed_auroc: float, baseline_auroc: float) -> str:
    """Keep the generated findings aligned with the actual numbers."""
    gap = proposed_auroc - baseline_auroc
    if gap > 0.01:
        return "Proposed outperforms the strongest baseline in this protocol"
    if abs(gap) <= 0.02:
        return "Proposed is close to the strongest baseline in this protocol"
    return (
        "Few-shot alone underperforms the strongest classical baseline in this strict protocol; "
        "the final thesis claim should therefore rely on the stacked/fusion pipeline, not on raw few-shot alone"
    )


def interpret_kshot(aurocs: np.ndarray) -> str:
    """Summarize K-shot sensitivity without overstating a weak trend."""
    if len(aurocs) < 2:
        return "Single K value reported"
    span = float(np.nanmax(aurocs) - np.nanmin(aurocs))
    if span < 0.02:
        return "Performance is nearly flat across K; simply increasing support shots did not materially improve AUROC"
    if aurocs[-1] > aurocs[0]:
        return "Higher K improves AUROC in this experiment"
    return "Higher K does not consistently improve AUROC; representation quality appears more important than K alone"


def interpret_exp5(mean_improvement: float) -> str:
    if mean_improvement > 0.02:
        return "The proposed model improves over LR on average across held-out disease nodes"
    if mean_improvement >= -0.02:
        return "Cross-disease transfer is roughly comparable to LR on average, with node-level variance"
    return (
        "Cross-disease transfer remains a limitation: the proposed model trails LR on average, "
        "so external/generalization claims should be stated cautiously"
    )


def load_result(tables_dir, results_dir, name, csv_name=None, json_names=None):
    """Load from CSV first, else JSON fallbacks."""
    csv_name = csv_name or f"{name}_results.csv"
    json_names = json_names or [f"{name}_main.json", f"{name}_shots.json", f"{name}_federated.json", f"{name}_ablation.json", f"{name}_generalization.json", f"{name}_results.json"]
    data = load_csv(tables_dir / csv_name)
    if data is not None:
        return data
    for jn in json_names:
        data = load_json(results_dir / jn)
        if data is not None:
            return pd.DataFrame(data) if isinstance(data, list) else data
    return None

def main():
    cfg = load_config()
    proj = Path(__file__).parent.parent
    paths = {k: str(proj / v) if isinstance(v, str) and not Path(v).is_absolute() else v for k, v in cfg["paths"].items()}
    tables_dir = Path(paths["tables_dir"])
    figures_dir = Path(paths["figures_dir"])
    results_dir = Path(paths.get("results_dir", proj / "results"))
    if not Path(str(results_dir)).is_absolute():
        results_dir = proj / results_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    exp1 = load_result(tables_dir, results_dir, "exp1", json_names=["exp1_main.json", "exp1_results.json"])
    exp2 = load_result(tables_dir, results_dir, "exp2", json_names=["exp2_shots.json", "exp2_results.json"])
    exp3 = load_result(tables_dir, results_dir, "exp3", json_names=["exp3_federated.json", "exp3_results.json"])
    exp4 = load_result(tables_dir, results_dir, "exp4", json_names=["exp4_ablation.json", "exp4_results.json"])
    exp5 = load_result(tables_dir, results_dir, "exp5", json_names=["exp5_generalization.json", "exp5_results.json"])
    
    if exp2 is not None and not isinstance(exp2, pd.DataFrame):
        exp2 = pd.DataFrame(exp2)
    if exp3 is not None and not isinstance(exp3, pd.DataFrame):
        exp3 = pd.DataFrame(exp3)
    if exp4 is not None and not isinstance(exp4, pd.DataFrame):
        exp4 = pd.DataFrame(exp4)
    if exp5 is not None and not isinstance(exp5, pd.DataFrame):
        exp5 = pd.DataFrame(exp5)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    colors = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#FBAFE4"]
    
    # Fig1: Main comparison
    try:
        if exp1 is not None and hasattr(exp1, "columns") and "model" in exp1.columns:
            df = pd.DataFrame(exp1) if not isinstance(exp1, pd.DataFrame) else exp1
            metrics = [c for c in ["accuracy", "f1_macro", "auroc"] if c in df.columns]
            if metrics:
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(df))
                w = 0.25
                for i, m in enumerate(metrics):
                    vals = df[m].fillna(0).values
                    ax.bar(x + i * w, vals, w, label=m.replace("_", " ").title())
                ax.set_xticks(x + w)
                ax.set_xticklabels(df["model"].values, rotation=45, ha="right")
                ax.set_ylabel("Score")
                ax.legend()
                ax.set_title("Experiment 1: Model Comparison")
                plt.tight_layout()
                plt.savefig(figures_dir / "fig1_main_comparison.png")
                plt.close()
    except Exception as e:
        print(f"Warning: fig1 skipped: {e}")
    
    # Fig2: ROC curves placeholder
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curves")
        plt.savefig(figures_dir / "fig2_roc_curves.png")
        plt.close()
    except Exception as e:
        print(f"Warning: fig2 skipped: {e}")
    
    # Fig3: K-shot curve with error bars
    try:
        if exp2 is not None and hasattr(exp2, "columns") and "k" in exp2.columns:
            df = pd.DataFrame(exp2)
            fig, ax = plt.subplots(figsize=(6, 4))
            k = df["k"].values
            auroc = df["auroc"].fillna(0.5).values
            std = df["auroc_std"].values if "auroc_std" in df.columns else np.zeros_like(auroc)
            ax.errorbar(k, auroc, yerr=std, fmt="o-", color=colors[0], capsize=3)
            ax.set_xlabel("K (shots)")
            ax.set_ylabel("AUROC")
            ax.set_title("Experiment 2: K-Shot Sensitivity")
            plt.savefig(figures_dir / "fig3_kshot_curve.png")
            plt.close()
    except Exception as e:
        print(f"Warning: fig3 skipped: {e}")
    
    # Fig4: Federated
    try:
        if exp3 is not None and hasattr(exp3, "columns") and "setup" in exp3.columns:
            df = pd.DataFrame(exp3)
            df = df[df["setup"] != "Gap"]
            if len(df) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(df["setup"], df["auroc"].fillna(0.5), color=colors[:len(df)])
                ax.set_ylabel("AUROC")
                ax.set_title("Experiment 3: Federated vs Centralized")
                plt.savefig(figures_dir / "fig4_federated.png")
                plt.close()
    except Exception as e:
        print(f"Warning: fig4 skipped: {e}")
    
    # Fig5: Ablation
    try:
        if exp4 is not None and hasattr(exp4, "columns") and "variant" in exp4.columns:
            df = pd.DataFrame(exp4)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(df["variant"], df["auroc"].fillna(0.5), color=colors[:len(df)])
            ax.set_xlabel("AUROC")
            ax.set_title("Experiment 4: Ablation Study")
            plt.tight_layout()
            plt.savefig(figures_dir / "fig5_ablation.png")
            plt.close()
    except Exception as e:
        print(f"Warning: fig5 skipped: {e}")
    
    # Fig6: Generalization
    try:
        if exp5 is not None and hasattr(exp5, "columns") and "test_node" in exp5.columns:
            df = pd.DataFrame(exp5)
            if "proposed_auroc" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(len(df))
                w = 0.2
                ax.bar(x - 1.5*w, df["proposed_auroc"].fillna(0.5), w, label="Proposed")
                if "lr_auroc" in df.columns:
                    ax.bar(x - 0.5*w, df["lr_auroc"], w, label="LR")
                if "xgboost_auroc" in df.columns:
                    ax.bar(x + 0.5*w, df["xgboost_auroc"], w, label="XGBoost")
                if "nn_auroc" in df.columns:
                    ax.bar(x + 1.5*w, df["nn_auroc"], w, label="NN")
                ax.set_xticks(x)
                ax.set_xticklabels(df["test_node"])
                ax.legend()
                ax.set_title("Experiment 5: Cross-Disease Generalization")
                plt.savefig(figures_dir / "fig6_generalization.png")
                plt.close()
    except Exception as e:
        print(f"Warning: fig6 skipped: {e}")
    
    # ALL_TABLES.tex
    tex_parts = []
    if exp1 is not None:
        df = pd.DataFrame(exp1)
        tex_parts.append(r"""
\begin{table}[htbp]
\centering
\caption{\maybeLr{Experiment 1: Main Model Comparison}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabularx}{\linewidth}{@{}L{4.4cm}C{1.45cm}C{1.35cm}C{1.55cm}C{1.35cm}@{}}
\toprule
Model & AUROC & F1 & Accuracy & MCC \\
\midrule
""")
        for _, r in df.iterrows():
            tex_parts.append(
                f"{tex_cell(display_label(r.get('model', '')))} & {r.get('auroc', 0):.3f} & {r.get('f1_macro', 0):.3f} & {r.get('accuracy', 0):.3f} & {r.get('mcc', 0):.3f} \\\\\n"
            )
        tex_parts.append(r"""\bottomrule
\end{tabularx}
\label{tab:exp1}
\end{table}
""")
    if exp2 is not None:
        df = pd.DataFrame(exp2)
        tex_parts.append(r"""
\begin{table}[htbp]
\centering
\caption{\maybeLr{Experiment 2: K-Shot Sensitivity}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabularx}{\linewidth}{@{}C{1.4cm}C{1.8cm}C{3.2cm}C{1.8cm}@{}}
\toprule
K & AUROC & AUROC std & F1 \\
\midrule
""")
        for _, r in df.iterrows():
            tex_parts.append(
                f"{tex_cell(r.get('k', ''))} & {r.get('auroc', 0):.3f} & {r.get('auroc_std', 0):.3f} & {r.get('f1_macro', 0):.3f} \\\\\n"
            )
        tex_parts.append(r"""\bottomrule
\end{tabularx}
\label{tab:exp2}
\end{table}
""")
    if exp3 is not None:
        df = pd.DataFrame(exp3)
        tex_parts.append(r"""
\begin{table}[htbp]
\centering
\caption{\maybeLr{Experiment 3: Federated vs Centralized}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabularx}{\linewidth}{@{}L{4.2cm}C{1.8cm}C{1.8cm}@{}}
\toprule
Setup & AUROC & F1 \\
\midrule
""")
        for _, r in df.iterrows():
            tex_parts.append(
                f"{tex_cell(display_label(r.get('setup', '')))} & {r.get('auroc', 0):.3f} & {r.get('f1_macro', 0):.3f} \\\\\n"
            )
        tex_parts.append(r"""\bottomrule
\end{tabularx}
\label{tab:exp3}
\end{table}
""")
    if exp4 is not None:
        df = pd.DataFrame(exp4)
        tex_parts.append(r"""
\begin{table}[htbp]
\centering
\caption{\maybeLr{Experiment 4: Ablation Study}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabularx}{\linewidth}{@{}L{4.2cm}C{1.8cm}C{1.8cm}@{}}
\toprule
Variant & AUROC & F1 \\
\midrule
""")
        for _, r in df.iterrows():
            tex_parts.append(
                f"{tex_cell(display_label(r.get('variant', '')))} & {r.get('auroc', 0):.3f} & {r.get('f1_macro', 0):.3f} \\\\\n"
            )
        tex_parts.append(r"""\bottomrule
\end{tabularx}
\label{tab:exp4}
\end{table}
""")
    if exp5 is not None:
        df = pd.DataFrame(exp5)
        tex_parts.append(r"""
\begin{table}[htbp]
\centering
\caption{\maybeLr{Experiment 5: Cross-Disease Generalization}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.14}
\begin{tabularx}{\linewidth}{@{}L{3.2cm}C{1.8cm}C{1.45cm}C{1.7cm}C{1.45cm}@{}}
\toprule
Test Node & Proposed & LR & XGBoost & NN \\
\midrule
""")
        for _, r in df.iterrows():
            tex_parts.append(
                f"{tex_cell(display_label(r.get('test_node', '')))} & {r.get('proposed_auroc', 0):.3f} & {r.get('lr_auroc', 0):.3f} & {r.get('xgboost_auroc', 0):.3f} & {r.get('nn_auroc', 0):.3f} \\\\\n"
            )
        tex_parts.append(r"""\bottomrule
\end{tabularx}
\label{tab:exp5}
\end{table}
""")
    
    body = "\n".join(tex_parts) if tex_parts else "% No results"
    with open(tables_dir / "ALL_TABLES.tex", "w") as f:
        f.write(body)
    # Thesis appendix: identical content but labels tab:pipeline_exp* (avoids duplicate \label vs chapters/results.tex)
    apx_path = tables_dir / "ALL_TABLES_APPENDIX.tex"
    if tex_parts:
        apx = body.replace(r"\label{tab:exp", r"\label{tab:pipeline_exp")
        hdr = (
            "% Auto-generated for thesis appendix — do not edit by hand.\n"
            "% Regenerate: python results/generate_all_outputs.py\n\n"
        )
        apx_path.write_text(hdr + apx, encoding="utf-8")
    else:
        apx_path.write_text(
            "% No results — run experiments then generate_all_outputs.py\n", encoding="utf-8"
        )
    
    # KEY_FINDINGS.md
    findings = []
    if exp1 is not None:
        df = pd.DataFrame(exp1)
        proposed = df[df["model"].astype(str).str.contains("Proposed", na=False)]
        best_bl = df[~df["model"].astype(str).str.contains("Proposed", na=False)]
        if len(proposed) > 0 and len(best_bl) > 0:
            p_auroc = proposed["auroc"].values[0]
            b_auroc = best_bl["auroc"].max()
            findings.append(f"""## Finding 1: Core Performance
**Result:** Proposed model achieves AUROC={p_auroc:.3f} vs best baseline {b_auroc:.3f}.
**Evidence:** Exp1 results.
**Interpretation:** {interpret_exp1(p_auroc, b_auroc)}.""")
    
    if exp2 is not None:
        df = pd.DataFrame(exp2)
        if "auroc" in df.columns:
            aurocs = df["auroc"].values
            k_auroc = ", ".join(f"K={k}: {v:.3f}" for k, v in zip(df['k'].values, df['auroc'].values))
            findings.append(f"""## Finding 2: Minimum Shots Needed
**Result:** AUROC by K: {k_auroc}
**Evidence:** Exp2 K-shot sensitivity.
**Interpretation:** {interpret_kshot(aurocs)}.""")
    
    if exp3 is not None:
        df = pd.DataFrame(exp3)
        if {"setup", "auroc"}.issubset(df.columns):
            cen = df[df["setup"].astype(str) == "Centralized"]["auroc"].values
            fed = df[df["setup"].astype(str) == "Federated"]["auroc"].values
            if len(cen) > 0 and len(fed) > 0:
                gap = (cen[0] - fed[0]) * 100
                findings.append(f"""## Finding 3: Federated Privacy Cost
**Result:** Centralized AUROC={cen[0]:.3f}, Federated={fed[0]:.3f}, Gap={gap:.1f}%.
**Evidence:** Exp3 federated experiment.
**Interpretation:** Federated learning preserves privacy with {"minimal" if gap < 5 else "moderate"} performance cost.""")
    
    if exp4 is not None:
        df = pd.DataFrame(exp4)
        full = df[df["variant"].astype(str).str.contains("full", case=False, na=False)]
        if len(full) > 0:
            findings.append(f"""## Finding 4: Ablation Sensitivity
**Result:** Full model AUROC={full['auroc'].values[0]:.3f}.
**Evidence:** Exp4 ablation study.
**Interpretation:** The ablation table documents component sensitivity; do not claim that every added component improves performance unless it is supported by the row-level numbers.""")
    
    if exp5 is not None:
        df = pd.DataFrame(exp5)
        if "improvement_vs_lr" in df.columns:
            imp = df["improvement_vs_lr"].mean() * 100
            findings.append(f"""## Finding 5: Cross-Disease Generalization
**Result:** Proposed vs LR improvement: {imp:+.1f}% mean across test nodes.
**Evidence:** Exp5 cross-disease experiment.
**Interpretation:** {interpret_exp5(imp / 100)}.""")
    
    md = "\n\n".join(findings) if findings else "# Key Findings\n*Run experiments to populate.*"
    with open(Path(__file__).parent / "KEY_FINDINGS.md", "w") as f:
        f.write(md)
    
    print("Generated: figures/, ALL_TABLES.tex, ALL_TABLES_APPENDIX.tex, KEY_FINDINGS.md")

if __name__ == "__main__":
    main()
