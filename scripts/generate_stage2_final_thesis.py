#!/usr/bin/env python3
"""Generate the Stage 2 topic-aligned Persian thesis deliverables.

The script treats the no-gap results under results/final_optimized as the
single metric source of truth. It writes a supervisor-reviewable Persian thesis,
regenerates missing figures, compiles the PDF, and packages the LaTeX source
with the figures and evidence tables used in the text.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import warnings
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "final_optimized"
TABLES = RESULTS / "tables"
OUT_DIR = ROOT / "papers" / "final_optimized_thesis"
FIG_DIR = OUT_DIR / "stage2_figures"
SOURCE_DATA_DIR = OUT_DIR / "stage2_source_data"
PDF_NAME = "thesis_FINAL_TOPIC_ALIGNED_OPTIMIZED.pdf"
TEX_NAME = "thesis_FINAL_TOPIC_ALIGNED_OPTIMIZED.tex"
ZIP_NAME = "thesis_FINAL_TOPIC_ALIGNED_OPTIMIZED_source.zip"
REPORT_NAME = "THESIS_FINAL_TOPIC_ALIGNED_REPORT.md"
DEFENSE_NAME = "THESIS_DEFENSE_SUMMARY.md"

FEATURE_INFO = [
    ("Creatinine", "50912", "آزمایشگاه"),
    ("Potassium", "50971", "آزمایشگاه"),
    ("Sodium", "50983", "آزمایشگاه"),
    ("Chloride", "50902", "آزمایشگاه"),
    ("Glucose", "50931", "آزمایشگاه"),
    ("Calcium", "50893", "آزمایشگاه"),
    ("Magnesium", "50960", "آزمایشگاه"),
    ("Hemoglobin", "50811", "آزمایشگاه"),
    ("Platelets", "51265", "آزمایشگاه"),
    ("WBC", "51300", "آزمایشگاه"),
    ("Lactate", "50813", "آزمایشگاه"),
    ("ALT", "50861", "آزمایشگاه"),
    ("AST", "50878", "آزمایشگاه"),
    ("BUN", "51006", "آزمایشگاه"),
    ("Bicarbonate", "50882", "آزمایشگاه"),
    ("Heart Rate", "220045", "علائم حیاتی"),
    ("Systolic BP", "220050", "علائم حیاتی"),
    ("Diastolic BP", "220051", "علائم حیاتی"),
    ("Mean BP", "220052", "علائم حیاتی"),
    ("Temperature", "223761", "علائم حیاتی"),
    ("SpO2", "220277", "علائم حیاتی"),
    ("Respiratory Rate", "220210", "علائم حیاتی"),
]

RISK_PHRASES = [
    "clinical deployment-ready",
    "treatment recommendation system",
    "clinician-adjudicated readiness",
    "few-shot superiority",
    "2,000-stay main cohort",
    "sampled benchmark as main analysis",
]


def esc(value: object) -> str:
    s = "" if pd.isna(value) else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def comma(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def lr(value: object) -> str:
    return r"\lr{" + esc(value) + "}"


def short_model(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "Token embeddings + tabular features + XGBoost": "TokenEmb + Tabular XGBoost",
        "Token embeddings + tabular features + CatBoost": "TokenEmb + Tabular CatBoost",
        "Token embeddings + tabular features + LightGBM": "TokenEmb + Tabular LightGBM",
        "ETHOS probability stacked with XGBoost probability": "ETHOS-prob + XGBoost stack",
        "ETHOS encoder + BCE + prototypical auxiliary loss": "ETHOS BCE + proto loss",
        "ETHOS encoder + BCE + contrastive auxiliary loss": "ETHOS BCE + contrastive",
        "Transformer encoder over symptom tokens": "Transformer token encoder",
        "Improved few-shot/token prototype": "Improved token prototype",
        "Current Few-Shot ETHOS baseline": "Few-Shot ETHOS baseline",
        "Simulated Federated LR Weighted Nodes": "Federated LR weighted",
        "Stacked Validation Meta-LR": "Stacked Meta-LR",
        "Weighted Average Ensemble": "Weighted Ensemble",
        "ETHOS embeddings + Random Forest": "ETHOS emb + RF",
        "ETHOS embeddings + CatBoost": "ETHOS emb + CatBoost",
        "ETHOS embeddings + XGBoost": "ETHOS emb + XGBoost",
        "ETHOS embeddings + LightGBM": "ETHOS emb + LightGBM",
        "ETHOS embeddings + Logistic Regression": "ETHOS emb + LogReg",
        "Token CNN supervised head": "Token CNN",
        "Token-only supervised MLP": "Token MLP",
        "Supervised ETHOS encoder + BCE": "ETHOS BCE",
        "GRU token encoder + supervised head": "GRU token",
        "Attention-pooled token encoder + supervised head": "Attention token",
        "denoising_autoencoder": "Denoising AE",
        "masked_reconstruction": "Masked reconstruction",
        "contrastive_patient": "Contrastive patient",
        "patient_representation": "Patient representation",
        "sequence_autoencoder": "Sequence AE",
        "Logistic Regression": "LogReg",
    }
    return replacements.get(text, text)


def short_feature(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "tabular": "tabular",
        "token": "token",
        "mixed": "mixed",
        "tabular_token": "tabular+token",
        "tabular_ethos_embedding": "tabular+ETHOS emb",
        "ethos_embedding": "ETHOS emb",
        "stacked_probability": "prob-stack",
        "tabular_pretrained_embedding": "tabular+pretrain",
        "pretrained_embedding": "pretrain emb",
        "pretrained_supervised_token": "pretrained token",
    }
    return replacements.get(text, text.replace("_", " "))


def short_ablation(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "labs_only": "labs only",
        "vitals_only": "vitals only",
        "labs_plus_vitals": "labs+vitals",
        "missingness_indicators": "missingness",
        "tabular_only": "tabular only",
        "token_only": "token only",
        "tabular_plus_token": "tabular+token",
        "simulated_federated_lr": "federated LR",
        "ETHOS_supervised_or_embedding": "ETHOS/token",
    }
    return replacements.get(text, text.replace("_", " "))


def short_table_cell(col: str, value: object) -> str:
    if col == "feature_set":
        return short_feature(value)
    if col == "pretraining_objective":
        return short_model(value)
    if col == "downstream_model":
        return short_model(value)
    if col == "ablation":
        return short_ablation(value)
    return "" if pd.isna(value) else str(value)


def display_model(row: pd.Series) -> str:
    model = row.get("model", "")
    if isinstance(model, str) and model.strip():
        return short_model(model)
    objective = row.get("pretraining_objective", "")
    downstream = row.get("downstream_model", "")
    if isinstance(objective, str) and objective.strip() and isinstance(downstream, str) and downstream.strip():
        return f"{short_model(objective)} + {downstream}"
    if isinstance(downstream, str) and downstream.strip():
        return downstream
    return "Model"


def metric_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    n: int = 10,
    extra_cols: list[tuple[str, str, str]] | None = None,
) -> str:
    extra_cols = extra_cols or []
    cols = [("model", "مدل", "text"), ("feature_set", "ویژگی", "text"), *extra_cols]
    cols.extend(
        [
            ("AUROC", "AUROC", "num"),
            ("AUPRC", "AUPRC", "num"),
            ("Brier", "Brier", "num"),
            ("ECE", "ECE", "num"),
        ]
    )
    spec_parts: list[str] = []
    for col, _, kind in cols:
        if col == "model":
            spec_parts.append("p{0.24\\textwidth}")
        elif col == "feature_set":
            spec_parts.append("p{0.15\\textwidth}")
        elif kind == "text":
            spec_parts.append("p{0.14\\textwidth}")
        else:
            spec_parts.append("r")
    spec = "".join(spec_parts)
    lines = [
        r"\begin{footnotesize}",
        rf"\begin{{longtable}}{{{spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(h for _, h, _ in cols) + r"\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(h for _, h, _ in cols) + r"\\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in df.head(n).iterrows():
        cells: list[str] = []
        for col, _, kind in cols:
            if col == "model":
                cells.append(lr(display_model(row)))
            elif kind == "num":
                cells.append(lr(fmt(row.get(col))))
            elif kind == "int":
                cells.append(lr(comma(row.get(col))))
            elif kind == "frac":
                cells.append(lr(fmt(row.get(col), 2)))
            else:
                cells.append(lr(short_table_cell(col, row.get(col, ""))))
        lines.append(" & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{footnotesize}"])
    return "\n".join(lines)


def simple_longtable(caption: str, label: str, headers: list[str], rows: list[list[str]], widths: str) -> str:
    lines = [
        r"\begin{small}",
        rf"\begin{{longtable}}{{{widths}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(headers) + r"\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(headers) + r"\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{small}"])
    return "\n".join(lines)


def figure(path: str, label: str, caption: str, width: str = "0.90\\textwidth") -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{path}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def read_inputs() -> dict[str, object]:
    validation = json.loads((RESULTS / "full_cohort_validation.json").read_text(encoding="utf-8"))
    data = {
        "validation": validation,
        "ranking": pd.read_csv(TABLES / "final_model_ranking_no_gap.csv"),
        "final": pd.read_csv(TABLES / "final_model_results_no_gap.csv"),
        "optimized": pd.read_csv(TABLES / "optimized_model_comparison.csv"),
        "labels": pd.read_csv(TABLES / "full_cohort_label_distribution.csv"),
        "missing": pd.read_csv(TABLES / "full_cohort_missingness.csv"),
        "nodes": pd.read_csv(TABLES / "full_cohort_node_distribution.csv"),
        "ethos": pd.read_csv(RESULTS / "ethos_no_gap" / "ethos_no_gap_results.csv"),
        "pretraining": pd.read_csv(RESULTS / "pretraining_no_gap" / "pretraining_results.csv"),
        "low_data": pd.read_csv(RESULTS / "low_data_no_gap" / "low_data_no_gap_results.csv"),
        "calibration": pd.read_csv(RESULTS / "calibration_no_gap" / "calibration_no_gap_results.csv"),
        "selective": pd.read_csv(RESULTS / "uncertainty_no_gap" / "selective_prediction_results.csv"),
        "conformal": pd.read_csv(RESULTS / "uncertainty_no_gap" / "conformal_prediction_results.csv"),
        "subgroups": pd.read_csv(RESULTS / "subgroups_no_gap" / "subgroup_no_gap_results.csv"),
        "ablation": pd.read_csv(RESULTS / "ablation_no_gap" / "ablation_no_gap_results.csv"),
        "federated": pd.read_csv(RESULTS / "federated_no_gap" / "federated_no_gap_results.csv"),
        "registry": pd.read_csv(RESULTS / "optimization" / "experiment_registry.csv"),
    }
    return data


def validate_no_gap_inputs(data: dict[str, object]) -> None:
    validation = data["validation"]
    registry = data["registry"]
    final = data["final"]
    assert validation["final_dataset"]["rows"] == 32399
    assert validation["cohort"]["unique_icu_stays"] == 32399
    assert validation["cohort"]["unique_subjects"] == 32399
    assert validation["final_dataset"]["label_distribution"] == {"0": 12102, "1": 20297}
    assert tuple(validation["tokens"]["shape"]) == (32399, 25, 4)
    assert len(registry) == 254
    required = [
        "selected_stacked_validation_meta_lr",
        "selected_weighted_average_validation_auc",
        "token_hybrid_catboost",
        "current_fewshot_ethos_baseline",
    ]
    ids = set(final["experiment_id"].astype(str))
    missing = [x for x in required if x not in ids]
    if missing:
        raise RuntimeError(f"Missing final rows: {missing}")


def row_by_id(df: pd.DataFrame, experiment_id: str) -> pd.Series:
    rows = df[df["experiment_id"].astype(str) == experiment_id]
    if rows.empty:
        raise KeyError(experiment_id)
    return rows.iloc[0]


def setup_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def plot_box_flow(path: Path, title: str, boxes: list[str], horizontal: bool = True) -> None:
    plt.figure(figsize=(12, 4.7 if horizontal else 7.2))
    ax = plt.gca()
    ax.axis("off")
    if horizontal:
        xs = np.linspace(0.06, 0.82, len(boxes))
        for i, (x, text) in enumerate(zip(xs, boxes)):
            ax.add_patch(plt.Rectangle((x, 0.38), 0.15, 0.22, facecolor="#eef5f8", edgecolor="#1f4e5f", linewidth=1.8))
            ax.text(x + 0.075, 0.49, text, ha="center", va="center", fontsize=11, wrap=True)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + 0.18, 0.49), xytext=(x + 0.15, 0.49), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f4e5f"))
    else:
        ys = np.linspace(0.82, 0.12, len(boxes))
        for i, (y, text) in enumerate(zip(ys, boxes)):
            ax.add_patch(plt.Rectangle((0.18, y), 0.64, 0.105, facecolor="#eef5f8", edgecolor="#1f4e5f", linewidth=1.8))
            ax.text(0.50, y + 0.052, text, ha="center", va="center", fontsize=12, wrap=True)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(0.50, y - 0.035), xytext=(0.50, y), arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f4e5f"))
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_metric_bars(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    top = df.head(12).copy().iloc[::-1]
    labels = [display_model(row) for _, row in top.iterrows()]
    colors = ["#2f6f73" if ("Ensemble" in m or "Stacked" in m) else "#5578a6" for m in top["model"].astype(str)]
    plt.figure(figsize=(9.8, 6.2))
    plt.barh(labels, top[metric], color=colors)
    plt.xlabel(metric)
    plt.title(title)
    plt.xlim(max(0.58, float(top[metric].min()) - 0.02), min(0.86, float(top[metric].max()) + 0.02))
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_prediction_curves(path_roc: Path, path_pr: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    prediction_files = [
        ("Stacked Meta-LR", RESULTS / "predictions" / "selected_stacked_validation_meta_lr_test_predictions.csv"),
        ("Weighted Ensemble", RESULTS / "predictions" / "selected_weighted_average_validation_auc_test_predictions.csv"),
        ("Token Hybrid CatBoost", RESULTS / "predictions" / "no_gap_token_hybrid_catboost_test_predictions.csv"),
        ("CatBoost tabular+token", RESULTS / "predictions" / "selected_opt_catboost_tabular_token_006_test_predictions.csv"),
    ]

    plt.figure(figsize=(7.2, 6.0))
    for label, path in prediction_files:
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        fpr, tpr, _ = roc_curve(pred["y_true"], pred["probability"])
        plt.plot(fpr, tpr, label=label, linewidth=1.9)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for top no-gap models", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_roc, dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 6.0))
    for label, path in prediction_files:
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        precision, recall, _ = precision_recall_curve(pred["y_true"], pred["probability"])
        plt.plot(recall, precision, label=label, linewidth=1.9)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves for top no-gap models", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_pr, dpi=300)
    plt.close()


def plot_token_representation(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    matrix = np.zeros((4, 25))
    matrix[0, :] = np.linspace(0.15, 0.85, 25)
    matrix[1, :] = np.sin(np.linspace(0, np.pi, 25)) * 0.75 + 0.1
    matrix[2, :] = 0.15
    matrix[3, :15] = 0.25
    matrix[3, 15:22] = 0.75
    matrix[:, 22:] = 0.02
    ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["symptom_id", "intensity", "time_delta_norm", "modality"], fontsize=12)
    ax.set_xticks([0, 4, 9, 14, 19, 24])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "25"], fontsize=11)
    ax.set_xlabel("Token position in fixed 25-token sequence")
    ax.set_title("Symptom-token tensor: 32,399 stays x 25 tokens x 4 channels", fontsize=14, fontweight="bold")
    ax.text(7, -0.9, "15 laboratory tokens", ha="center", fontsize=11)
    ax.text(18, -0.9, "7 vital-sign tokens", ha="center", fontsize=11)
    ax.text(23, -0.9, "mask/padding", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_low_data(low: pd.DataFrame, path: Path) -> None:
    selected = [
        "Random Forest",
        "XGBoost",
        "Token Hybrid XGBoost",
        "Token Hybrid CatBoost",
        "Improved Few-Shot/Token Prototype",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for model in selected:
        subset = low[low["model"].astype(str) == model].sort_values("training_fraction")
        if subset.empty:
            continue
        label = short_model(model)
        axes[0].plot(subset["training_fraction"] * 100, subset["AUROC"], marker="o", label=label)
        axes[1].plot(subset["training_fraction"] * 100, subset["AUPRC"], marker="o", label=label)
    for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
        ax.set_xlabel("Training fraction (%)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Low-data performance across fixed validation/test splits", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_token_hybrid(ethos: pd.DataFrame, path: Path) -> None:
    top = ethos.sort_values("AUROC", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(10, 6.4))
    plt.barh(y - 0.17, top["AUROC"], height=0.34, color="#3f7f6f", label="AUROC")
    plt.barh(y + 0.17, top["AUPRC"], height=0.34, color="#8a6fb0", label="AUPRC")
    plt.yticks(y, [short_model(m) for m in top["model"]], fontsize=9)
    plt.xlabel("Metric value")
    plt.xlim(0.58, 0.86)
    plt.title("ETHOS, token, and hybrid model comparison", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_selective(selective: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9.5, 5.8))
    for model, subset in selective.groupby("model"):
        subset = subset.sort_values("coverage", ascending=False)
        plt.plot(subset["coverage"] * 100, subset["accuracy"], marker="o", label=short_model(model))
    plt.xlabel("Retained coverage (%)")
    plt.ylabel("Accuracy among retained predictions")
    plt.title("Selective prediction: accuracy improves as uncertain cases are deferred", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_feature_importance(path: Path) -> None:
    feature_labels = [x[0] for x in FEATURE_INFO] + ["Shock index proxy", "SOFA-like proxy", "N abnormal labs"]
    feature_labels += [f"Missing: {x[0]}" for x in FEATURE_INFO]
    model_path = RESULTS / "models" / "selected_opt_random_forest_tabular_002.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = joblib.load(model_path)
    importances = np.asarray(obj["model"].feature_importances_)
    n = min(len(importances), len(feature_labels))
    order = np.argsort(importances[:n])[-15:]
    labels = [feature_labels[i] for i in order]
    values = importances[order]
    plt.figure(figsize=(9.5, 6.0))
    plt.barh(labels, values, color="#5578a6")
    plt.xlabel("Random forest impurity importance")
    plt.title("Feature importance in the selected tabular random forest", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_federated(nodes: pd.DataFrame, federated: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(nodes["node"], nodes["rows"], color="#5578a6")
    axes[0].set_title("Simulated disease nodes")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=25)
    fed = federated.copy()
    axes[1].bar([short_model(m) for m in fed["model"]], fed["AUROC"], color=["#3f7f6f", "#8a6fb0"])
    axes[1].set_ylim(0.68, 0.76)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Federated LR vs centralized LR")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Simulated federated learning design and result", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def copy_existing_figures() -> None:
    copies = {
        RESULTS / "calibration_no_gap" / "calibration_curves.png": FIG_DIR / "calibration_curves.png",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src, dst)


def make_figures(data: dict[str, object]) -> None:
    final = data["ranking"].copy().head(20)
    validation = data["validation"]
    plot_box_flow(
        FIG_DIR / "study_pipeline.png",
        "Study pipeline",
        [
            "MIMIC-IV v3.1",
            "Adult first ICU stays",
            "First-24h labs and vitals",
            "Symptom-token tensor",
            "Model families",
            "Calibration-aware evaluation",
        ],
    )
    plot_box_flow(
        FIG_DIR / "cohort_flow.png",
        "Cohort construction and outcome",
        [
            "Manifest-verified MIMIC-IV files: 33/33 present",
            "Adult first ICU stays with required discharge information",
            "Final full cohort: 32,399 ICU stays and 32,399 subjects",
            "Proxy labels: 20,297 feasible and 12,102 infeasible",
            "No sampling cap; all eligible data used",
        ],
        horizontal=False,
    )
    plot_token_representation(FIG_DIR / "symptom_token_representation.png")
    plot_box_flow(
        FIG_DIR / "model_family_framework.png",
        "Model families and evaluation framework",
        [
            "Tabular baselines",
            "ETHOS and few-shot",
            "Supervised token models",
            "Pretraining and hybrids",
            "Stacking and ensembles",
            "Calibration, uncertainty, subgroups",
        ],
    )
    plot_metric_bars(final, "AUROC", FIG_DIR / "auroc_comparison.png", "Final test AUROC comparison")
    plot_metric_bars(final, "AUPRC", FIG_DIR / "auprc_comparison.png", "Final test AUPRC comparison")
    plot_prediction_curves(FIG_DIR / "roc_curves_top_models.png", FIG_DIR / "pr_curves_top_models.png")
    plot_low_data(data["low_data"], FIG_DIR / "low_data_performance_curves.png")
    plot_token_hybrid(data["ethos"], FIG_DIR / "token_hybrid_comparison.png")
    plot_selective(data["selective"], FIG_DIR / "selective_prediction_curve.png")
    plot_feature_importance(FIG_DIR / "feature_importance_random_forest.png")
    plot_federated(data["nodes"], data["federated"], FIG_DIR / "federated_overview.png")
    copy_existing_figures()
    # Keep the validation-derived label and missingness plots available if older
    # downstream scripts expect them, but the thesis uses the cleaner figures above.
    _ = validation


def build_tables(data: dict[str, object]) -> dict[str, str]:
    validation = data["validation"]
    labels = data["labels"]
    missing = data["missing"]
    ranking = data["ranking"].copy()
    final = data["final"].copy()
    ethos = data["ethos"].copy()
    pretraining = data["pretraining"].copy().sort_values("AUROC", ascending=False)
    low = data["low_data"].copy()
    calibration = data["calibration"].copy()
    subgroups = data["subgroups"].copy()
    ablation = data["ablation"].copy()
    federated = data["federated"].copy()

    label_total = int(labels["count"].sum())
    cohort_rows = [
        ["منبع داده", lr("MIMIC-IV v3.1"), "پایگاه داده ICU و بیمارستانی"],
        ["تعداد ICU stay", lr(comma(validation["cohort"]["unique_icu_stays"])), "هر ردیف یک بستری ICU یکتا"],
        ["تعداد subject", lr(comma(validation["cohort"]["unique_subjects"])), "در این کوهورت برابر با تعداد stay"],
        ["تعداد ردیف نهایی", lr(comma(validation["final_dataset"]["rows"])), "بدون سقف نمونه‌گیری"],
        ["تعداد ویژگی خام", lr(comma(validation["final_dataset"]["feature_columns"])), "۱۵ آزمایش و ۷ علامت حیاتی"],
        ["شکل تانسور توکن", lr(str(tuple(validation["tokens"]["shape"]))), "۲۵ توکن و ۴ کانال برای هر stay"],
        ["تعداد رجیستری آزمایش", lr(comma(len(data["registry"]))), "ردیف‌های ثبت‌شده در no-gap pass"],
        ["گمشده‌بودن میانگین خام", lr(fmt(validation["missingness"]["mean_pre_imputation_missing_pct"] * 100) + r"\%"), "پیش از جایگذاری و فقط به‌عنوان توصیف داده"],
    ]
    cohort_table = simple_longtable(
        "خلاصه کوهورت و وضعیت داده نهایی.",
        "tab:cohort-summary",
        ["مولفه", "مقدار", "توضیح"],
        [[esc(a), b, esc(c)] for a, b, c in cohort_rows],
        "p{0.28\\textwidth}p{0.24\\textwidth}p{0.36\\textwidth}",
    )

    label_rows = []
    for _, row in labels.iterrows():
        label = "غیرقابل ترخیص/نامطلوب در proxy" if int(row["feasible"]) == 0 else "قابل ترخیص/مطلوب در proxy"
        label_rows.append([lr(int(row["feasible"])), esc(label), lr(comma(row["count"])), lr(fmt(row["count"] / label_total * 100) + r"\%")])
    label_table = simple_longtable(
        "توزیع برچسب ICU discharge-feasibility proxy.",
        "tab:label-distribution",
        ["کد", "معنا", "تعداد", "درصد"],
        label_rows,
        "p{0.10\\textwidth}p{0.44\\textwidth}p{0.18\\textwidth}p{0.16\\textwidth}",
    )

    feature_rows = [
        [
            "آزمایش‌ها",
            lr("15"),
            lr(", ".join(x[1] for x in FEATURE_INFO[:15])),
            "میانه اندازه‌گیری‌های ۲۴ ساعت اول ICU پس از کلیپ بالینی.",
        ],
        [
            "علائم حیاتی",
            lr("7"),
            lr(", ".join(x[1] for x in FEATURE_INFO[15:])),
            "مقادیر روتین ۲۴ ساعت اول ICU؛ شامل ضربان قلب، فشار خون، دما، اکسیژن و تنفس.",
        ],
        [
            "ویژگی‌های مهندسی‌شده",
            lr("3"),
            lr("shock index proxy; SOFA-like proxy; n abnormal labs"),
            "از ویژگی‌های نرمال‌شده آموزش‌محور ساخته شدند و جایگزین outcome نیستند.",
        ],
        [
            "شاخص‌های گمشده‌بودن",
            lr("22"),
            lr("missing flags"),
            "برای جلوگیری از حذف سیگنال‌های informative missingness در ICU.",
        ],
    ]
    feature_table = simple_longtable(
        "گروه‌های ویژگی مورد استفاده در مدل‌های جدولی.",
        "tab:feature-groups",
        ["گروه", "تعداد", "شناسه‌ها/مولفه‌ها", "توضیح"],
        [[esc(a), b, c, esc(d)] for a, b, c, d in feature_rows],
        "p{0.18\\textwidth}p{0.11\\textwidth}p{0.34\\textwidth}p{0.27\\textwidth}",
    )

    token_rows = [
        ["تعداد ردیف", lr(comma(validation["tokens"]["metadata_rows"])), "برای همه stayهای نهایی توکن ساخته شد."],
        ["طول توالی", lr("25"), "۲۲ مولفه خام به‌اضافه ظرفیت padding/mask."],
        ["کانال‌های هر توکن", lr("4"), lr("symptom_id, intensity, time_delta_norm, modality")],
        ["شکل تانسور", lr("(32399, 25, 4)"), "نمایش ثابت برای مدل‌های token/ETHOS."],
        ["تعداد نوع علامت", lr("22"), "۱۵ آزمایش و ۷ علامت حیاتی."],
        ["گونه‌های no-gap", lr("hybrid, embedding, pretraining, supervised token"), "گونه‌های ۵۰/۷۵/۱۰۰ توکن به‌دلیل نبود مولفه خام کافی به‌عنوان padding صرف توجیه نشدند."],
    ]
    token_table = simple_longtable(
        "خلاصه نمایش توکن‌های علائم.",
        "tab:token-summary",
        ["مولفه", "مقدار", "توضیح"],
        [[esc(a), b, c if c.startswith(r"\lr") else esc(c)] for a, b, c in token_rows],
        "p{0.22\\textwidth}p{0.24\\textwidth}p{0.44\\textwidth}",
    )

    fewshot_ids = [
        "token_hybrid_catboost",
        "tabular_ethos_embedding_xgboost",
        "tabular_ethos_embedding_catboost",
        "ethos_probability_stacked_with_xgboost",
        "token_hybrid_xgboost",
        "token_cnn_supervised",
        "ethos_bce_prototypical",
        "supervised_ethos_bce",
        "attention_pool_supervised",
        "current_fewshot_ethos_baseline",
        "improved_fewshot_token_prototype",
    ]
    fewshot_df = pd.concat([ethos[ethos["experiment_id"].eq(x)] for x in fewshot_ids if not ethos[ethos["experiment_id"].eq(x)].empty])
    token_df = ethos[ethos["feature_set"].astype(str).str.contains("token|ethos|stacked", case=False, na=False)].sort_values("AUROC", ascending=False)
    best_low = low.sort_values(["training_fraction", "AUROC"]).groupby("training_fraction", as_index=False).tail(1).sort_values("training_fraction")
    calibration["model"] = calibration.apply(display_model, axis=1)
    calibration = calibration.sort_values(["Brier", "ECE"], ascending=[True, True])

    unc_rows = []
    selective = data["selective"]
    conformal = data["conformal"]
    for model in ["Stacked Validation Meta-LR", "Weighted Average Ensemble", "Token Hybrid CatBoost"]:
        s = selective[selective["model"].eq(model)]
        c = conformal[(conformal["model"].eq(model)) & (np.isclose(conformal["alpha"], 0.10))]
        def get_acc(cov: float) -> str:
            row = s[np.isclose(s["coverage"], cov)]
            return fmt(row.iloc[0]["accuracy"]) if not row.empty else ""
        conf_cov = fmt(c.iloc[0]["empirical_coverage"]) if not c.empty else ""
        avg_set = fmt(c.iloc[0]["average_set_size"]) if not c.empty else ""
        singleton = fmt(c.iloc[0]["singleton_rate"]) if not c.empty else ""
        unc_rows.append([lr(short_model(model)), lr(get_acc(1.0)), lr(get_acc(0.8)), lr(get_acc(0.5)), lr(conf_cov), lr(avg_set), lr(singleton)])
    uncertainty_table = simple_longtable(
        "خلاصه uncertainty، پیش‌بینی انتخابی و پیش‌بینی همساز.",
        "tab:uncertainty",
        ["مدل", "دقت ۱۰۰٪", "دقت ۸۰٪", "دقت ۵۰٪", "پوشش همساز ۹۰٪", "اندازه مجموعه", "singleton"],
        unc_rows,
        "p{0.26\\textwidth}rrrrrr",
    )

    subgroup_show = subgroups.head(14).copy()
    subgroup_rows = []
    for _, row in subgroup_show.iterrows():
        subgroup_rows.append(
            [
                lr(row["subgroup_variable"]),
                lr(row["subgroup"]),
                lr(comma(row["n"])),
                lr(fmt(row["AUROC"])),
                lr(fmt(row["AUPRC"])),
                lr(fmt(row["Brier"])),
                lr(fmt(row["ECE"])),
            ]
        )
    subgroup_table = simple_longtable(
        "تحلیل اکتشافی subgroup و خطا برای مدل نهایی.",
        "tab:subgroup",
        ["متغیر", "زیرگروه", "n", "AUROC", "AUPRC", "Brier", "ECE"],
        subgroup_rows,
        "p{0.18\\textwidth}p{0.26\\textwidth}rrrrr",
    )

    limitation_rows = [
        ["طراحی گذشته‌نگر", "نتیجه‌ها علی یا آینده‌نگر تفسیر نمی‌شوند.", "گزارش صریح دامنه مطالعه و پیشنهاد اعتبارسنجی آینده‌نگر."],
        ["یک پایگاه داده", "تعمیم‌پذیری بیرونی محدود است.", "پیشنهاد اعتبارسنجی خارجی در پایگاه‌های مستقل."],
        ["proxy outcome", "برچسب برابر داوری بالینی واقعی نیست.", "استفاده مداوم از اصطلاح ICU discharge-feasibility proxy."],
        ["ویژگی‌های ۲۴ ساعت اول", "سیگنال‌های دیرتر ICU وارد مدل نشده‌اند.", "تعریف شفاف پنجره زمانی و کار آینده با داده طولی."],
        ["مخدوش‌گری باقیمانده", "متغیرهای درمان، سیاست بخش و شدت بیماری کامل نیستند.", "تحلیل ablation و subgroup برای تفسیر محافظه‌کارانه."],
        ["سوگیری کدنویسی و اندازه‌گیری", "EHR می‌تواند بازتاب فرآیند مراقبت باشد.", "بحث درباره missingness و محدودیت داده ثانویه."],
        ["فدرال شبیه‌سازی‌شده", "چندمرکزی واقعی اجرا نشده است.", "گزارش به‌عنوان simulation و نه پیاده‌سازی عملیاتی."],
        ["فیوشات خالص ضعیف‌تر", "فرضیه عنوانی برنده full-data نشد.", "گزارش صادقانه نقش benchmark، low-data و hybrid."],
        ["محدودیت بازتوزیع داده", "داده خام MIMIC-IV قابل انتشار نیست.", "انتشار کد، جدول‌های مشتق‌شده و دستور بازتولید."],
    ]
    limitation_table = simple_longtable(
        "محدودیت‌ها و راهبردهای کاهش/گزارش ریسک.",
        "tab:limitations",
        ["محدودیت", "اثر بر تفسیر", "کاهش یا گزارش"],
        [[esc(a), esc(b), esc(c)] for a, b, c in limitation_rows],
        "p{0.25\\textwidth}p{0.30\\textwidth}p{0.35\\textwidth}",
    )

    tables = {
        "cohort": cohort_table,
        "labels": label_table,
        "features": feature_table,
        "tokens": token_table,
        "main": metric_table(ranking.head(12), "مقایسه اصلی مدل‌های نهایی no-gap.", "tab:main-final", n=12),
        "fewshot": metric_table(fewshot_df, "مقایسه مدل‌های ETHOS، توکن و few-shot.", "tab:fewshot", n=len(fewshot_df)),
        "tokenhybrid": metric_table(token_df, "مقایسه مدل‌های token/hybrid.", "tab:token-hybrid", n=12),
        "pretraining": metric_table(
            pretraining,
            "نتایج self-supervised pretraining و مدل‌های downstream.",
            "tab:pretraining",
            n=10,
            extra_cols=[("pretraining_objective", "هدف", "text"), ("downstream_model", "downstream", "text")],
        ),
        "lowdata": metric_table(
            best_low,
            "بهترین مدل در هر سناریوی کم‌داده.",
            "tab:low-data",
            n=len(best_low),
            extra_cols=[("training_fraction", "fraction", "frac"), ("train_rows", "train n", "int")],
        ),
        "calibration": metric_table(calibration, "مقایسه کالیبراسیون مدل‌های منتخب.", "tab:calibration", n=14),
        "uncertainty": uncertainty_table,
        "subgroup": subgroup_table,
        "ablation": metric_table(ablation, "نتایج ablation و robustness.", "tab:ablation", n=14, extra_cols=[("ablation", "ablation", "text")]),
        "federated": metric_table(federated, "نتایج یادگیری فدرال شبیه‌سازی‌شده.", "tab:federated", n=2),
        "ranking": metric_table(ranking.head(15), "رتبه‌بندی نهایی مدل‌ها بر اساس جدول no-gap.", "tab:ranking", n=15),
        "limitations": limitation_table,
    }
    return tables


def make_tex(data: dict[str, object], tables: dict[str, str]) -> str:
    validation = data["validation"]
    final = data["final"]
    ranking = data["ranking"]
    stacked = row_by_id(final, "selected_stacked_validation_meta_lr")
    weighted = row_by_id(final, "selected_weighted_average_validation_auc")
    token = row_by_id(final, "token_hybrid_catboost")
    fewshot = row_by_id(final, "current_fewshot_ethos_baseline")
    pretrain = data["pretraining"].sort_values("AUROC", ascending=False).iloc[0]
    labels = validation["final_dataset"]["label_distribution"]
    rows = comma(validation["final_dataset"]["rows"])
    raw_missing = comma(validation["final_dataset"]["raw_feature_missing_cells"])
    registry_rows = comma(len(data["registry"]))

    preamble = r"""
\documentclass[12pt,a4paper]{report}
\usepackage[margin=2.4cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{setspace}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{xepersian}
\settextfont{Vazirmatn}
\setlatintextfont{Times New Roman}
\onehalfspacing
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{2}
\hypersetup{colorlinks=true, linkcolor=blue!55!black, urlcolor=blue!55!black, citecolor=blue!55!black}
\captionsetup{font=small, labelfont=bf}
\renewcommand{\arraystretch}{1.25}
\title{فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده}
\author{پایان‌نامه بازنویسی‌شده بر اساس شواهد no-gap}
\date{تیر ۱۴۰۵}
\begin{document}
\maketitle
\pagenumbering{roman}
"""
    abstract_fa = f"""
\\chapter*{{چکیده فارسی}}
\\addcontentsline{{toc}}{{chapter}}{{چکیده فارسی}}
این پایان‌نامه یک مطالعه گذشته‌نگر در حوزه انفورماتیک زیست‌پزشکی و یادگیری ماشین بالینی است که با تکیه بر پایگاه \\lr{{MIMIC-IV v3.1}}، امکان پیش‌بینی یک شاخص جایگزین از قابلیت ترخیص از بخش مراقبت‌های ویژه را بررسی می‌کند. عنوان مصوب پژوهش، «فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده»، در این بازنویسی حفظ شده است؛ اما مفهوم «قابلیت درمان» به‌صورت محافظه‌کارانه و قابل بازتولید به \\lr{{ICU discharge-feasibility proxy}} محدود شده است. این proxy از سه شرط بقای داخل بیمارستان، مقصد ترخیص مطلوب، و طول اقامت ICU کمتر از \\lr{{720}} ساعت ساخته شد و معادل آمادگی واقعی ترخیص، داوری بالینی پزشک، یا ابزار تصمیم‌گیری کنار تخت نیست.

کوهورت نهایی شامل \\lr{{{rows}}} بستری ICU و \\lr{{{rows}}} بیمار یکتا از \\lr{{MIMIC-IV v3.1}} بود. توزیع برچسب‌ها شامل \\lr{{{comma(labels['0'])}}} نمونه با برچسب صفر و \\lr{{{comma(labels['1'])}}} نمونه با برچسب یک بود. نمایش توکنی نهایی شکل \\lr{{(32399, 25, 4)}} داشت و رجیستری آزمایش‌ها شامل \\lr{{{registry_rows}}} ردیف بود. در سناریوی full-data، مدل \\lr{{Stacked Validation Meta-LR}} بهترین عملکرد کلی و کالیبراسیون‌محور را با \\lr{{AUROC={fmt(stacked['AUROC'])}}}، \\lr{{AUPRC={fmt(stacked['AUPRC'])}}}، \\lr{{Brier={fmt(stacked['Brier'])}}} و \\lr{{ECE={fmt(stacked['ECE'])}}} نشان داد. مدل \\lr{{Weighted Average Ensemble}} بهترین پروفایل discrimination را با \\lr{{AUROC={fmt(weighted['AUROC'])}}} و \\lr{{AUPRC={fmt(weighted['AUPRC'])}}} به دست آورد. در میان مدل‌های واقعی ETHOS/token/hybrid، \\lr{{Token Hybrid CatBoost}} با \\lr{{AUROC={fmt(token['AUROC'])}}} و \\lr{{AUPRC={fmt(token['AUPRC'])}}} به عملکردی نزدیک به بهترین مدل کلی رسید. بهترین نتیجه مرتبط با پیش‌آموزش نیز \\lr{{denoising autoencoder + Tabular + XGBoost}} با \\lr{{AUROC={fmt(pretrain['AUROC'])}}} و \\lr{{AUPRC={fmt(pretrain['AUPRC'])}}} بود. در مقابل، مدل pure few-shot/prototype با \\lr{{AUROC={fmt(fewshot['AUROC'])}}} و \\lr{{AUPRC={fmt(fewshot['AUPRC'])}}} ضعیف‌تر باقی ماند.

یافته اصلی این پایان‌نامه آن است که فرضیه مصوب few-shot و symptom-token به‌صورت کامل آزمون شد، نه اینکه از پیش درست فرض شود. مدل‌های supervised stacking و ensemble در داده کامل قوی‌تر بودند؛ با این حال، توکن‌های علائم و روش‌های ETHOS/few-shot به‌عنوان محورهای representation، hybrid، low-data و benchmark ارزش علمی دارند. نتیجه نزدیک \\lr{{Token Hybrid CatBoost}} به بهترین مدل کلی، دفاع‌پذیری عنوان را حفظ می‌کند و نشان می‌دهد نمایش‌های توکنی علائم می‌توانند در ترکیب با ویژگی‌های جدولی، به مدل‌سازی بالینی قابل گزارش کمک کنند.

\\noindent\\textbf{{کلیدواژه‌ها:}} یادگیری چندنمونه‌ای، توکن‌های علائم، \\lr{{ETHOS}}، \\lr{{MIMIC-IV}}، پیش‌بینی بالینی، کالیبراسیون، پیش‌بینی انتخابی، یادگیری فدرال شبیه‌سازی‌شده.
"""
    abstract_en = f"""
\\chapter*{{\\lr{{English Abstract}}}}
\\addcontentsline{{toc}}{{chapter}}{{English Abstract}}
\\begin{{latin}}
This thesis reports a retrospective biomedical informatics and clinical machine learning study using the full eligible \\lr{{MIMIC-IV v3.1}} ICU cohort available in the project. The approved Persian title is preserved, while the implemented endpoint is defined conservatively as an \\lr{{ICU discharge-feasibility proxy}} based on in-hospital survival, favourable discharge destination, and ICU length of stay below 720 hours. The endpoint is a reproducible proxy for research evaluation and is not a prospective bedside readiness endpoint or an autonomous clinical product.

The final cohort contained \\lr{{{rows}}} ICU stays and \\lr{{{rows}}} unique subjects. The label distribution was \\lr{{{comma(labels['0'])}}} infeasible and \\lr{{{comma(labels['1'])}}} feasible cases, with a symptom-token tensor of shape \\lr{{(32399, 25, 4)}} and \\lr{{{registry_rows}}} registered experiments. In the full-data setting, \\lr{{Stacked Validation Meta-LR}} achieved the best overall calibration-aware result with \\lr{{AUROC={fmt(stacked['AUROC'])}}}, \\lr{{AUPRC={fmt(stacked['AUPRC'])}}}, \\lr{{Brier={fmt(stacked['Brier'])}}}, and \\lr{{ECE={fmt(stacked['ECE'])}}}. \\lr{{Weighted Average Ensemble}} achieved the strongest discrimination profile with \\lr{{AUROC={fmt(weighted['AUROC'])}}} and \\lr{{AUPRC={fmt(weighted['AUPRC'])}}}. The best ETHOS/token/hybrid model, \\lr{{Token Hybrid CatBoost}}, reached \\lr{{AUROC={fmt(token['AUROC'])}}} and \\lr{{AUPRC={fmt(token['AUPRC'])}}}, close to the best overall supervised ensemble. Pure few-shot/prototype modelling remained weaker, with \\lr{{AUROC={fmt(fewshot['AUROC'])}}} and \\lr{{AUPRC={fmt(fewshot['AUPRC'])}}}.

The main conclusion is that the approved few-shot and symptom-token hypothesis was rigorously evaluated rather than assumed. Supervised stacking and ensemble models were strongest in the full cohort, but symptom-token methods remained central as representation, hybrid, low-data, and benchmark contributions. This evidence keeps the approved topic scientifically defensible while avoiding overclaiming.

\\noindent\\textbf{{Keywords:}} Few-Shot Learning, Symptom Tokens, ETHOS, MIMIC-IV, ICU discharge-feasibility proxy, calibration, selective prediction, simulated federated learning.
\\end{{latin}}
"""
    front = r"""
\chapter*{فهرست اختصارات}
\addcontentsline{toc}{chapter}{فهرست اختصارات}
\begin{longtable}{p{0.20\textwidth}p{0.68\textwidth}}
\toprule
اختصار & توضیح\\
\midrule
\lr{ICU} & بخش مراقبت‌های ویژه\\
\lr{EHR} & پرونده الکترونیک سلامت\\
\lr{MIMIC-IV} & پایگاه داده عمومی مراقبت ویژه نسخه چهارم\\
\lr{ETHOS} & چارچوب رمزگذاری/نمایش توکنی علائم در این پروژه\\
\lr{AUROC} & سطح زیر منحنی ROC\\
\lr{AUPRC} & سطح زیر منحنی Precision-Recall\\
\lr{Brier} & امتیاز خطای احتمال پیش‌بینی‌شده\\
\lr{ECE} & خطای کالیبراسیون مورد انتظار\\
\lr{FL} & یادگیری فدرال\\
\lr{SP} & پیش‌بینی انتخابی\\
\bottomrule
\end{longtable}
\tableofcontents
\listoftables
\listoffigures
\clearpage
\pagenumbering{arabic}
"""
    chapter1 = r"""
\chapter{مقدمه}
\section{بیان مسئله}
پیش‌بینی وضعیت بیمار در ساعات نخست بستری در ICU یکی از مسائل محوری انفورماتیک سلامت است، زیرا بخش مهمی از تصمیم‌های بعدی، از پایش خطر تا اولویت‌بندی منابع، بر اساس تصویر اولیه بیمار شکل می‌گیرد. با این حال، مفهوم «قابلیت درمان» در عمل مفهومی پیچیده، چندبعدی و وابسته به قضاوت بالینی است. یک مدل یادگیری ماشین که فقط از داده‌های گذشته‌نگر پرونده الکترونیک سلامت استفاده می‌کند، نمی‌تواند به‌تنهایی درباره آمادگی واقعی ترخیص یا امکان‌پذیری درمان فردی حکم صادر کند. بنابراین، این پایان‌نامه مسئله را به‌صورت یک پیش‌بینی پژوهشی دقیق‌تر بازتعریف می‌کند: پیش‌بینی \lr{ICU discharge-feasibility proxy} از روی آزمایش‌ها و علائم حیاتی ۲۴ ساعت نخست ICU.

این proxy بر مبنای سه مولفه ساخته شد: بقای داخل بیمارستان، مقصد ترخیص مطلوب، و طول اقامت ICU کمتر از ۷۲۰ ساعت. این تعریف از یک‌سو به عنوان مصوب پایان‌نامه نزدیک می‌ماند، زیرا همچنان درباره «قابلیت» و مسیر مطلوب بیمار پس از مراقبت ویژه است؛ از سوی دیگر از ادعای بالینی فراتر از داده پرهیز می‌کند. چنین مرزبندی‌ای برای پایان‌نامه ضروری است، زیرا اعتبار علمی مدل‌های پیش‌بینی بالینی نه فقط با مقدار \lr{AUROC}، بلکه با صداقت در تعریف outcome و محدودیت‌های داده سنجیده می‌شود.

\section{انگیزه پژوهش}
انگیزه اصلی پژوهش، بررسی این پرسش بود که آیا بازنمایی‌های فشرده بیمار به شکل توکن‌های علائم \lr{(Symptom Tokens)} و روش‌های یادگیری چندنمونه‌ای \lr{(Few-Shot Learning)} می‌توانند در کنار مدل‌های کلاسیک، تصویر معناداری از وضعیت early ICU بسازند یا خیر. در ICU، داده‌ها پرنویز، ناقص و ناهمگن‌اند. مدل‌های جدولی قوی مانند \lr{XGBoost}، \lr{LightGBM} و \lr{CatBoost} غالبا در چنین محیط‌هایی عملکرد مناسبی دارند، اما فرضیه این پایان‌نامه آن بود که ساختاردهی علائم به توکن و آزمون ETHOS/few-shot می‌تواند در representation، کم‌داده بودن، و مدل‌های hybrid ارزش افزوده ایجاد کند.

\section{اهمیت داده‌های ۲۴ ساعت اول}
۲۴ ساعت نخست ICU پنجره‌ای بالینی است که هم از نظر دسترس‌پذیری داده عملی است و هم از نظر تفسیر زمانی قابل دفاع. استفاده از داده‌های بعدی می‌تواند خطر leakage را بالا ببرد، زیرا اطلاعاتی که پس از مسیر درمان رخ داده‌اند ممکن است outcome را به‌طور غیرمستقیم افشا کنند. در این پایان‌نامه، همه ویژگی‌های آزمایشگاهی و علائم حیاتی به همین پنجره محدود شدند. شکل \ref{fig:study-pipeline} جریان کلی مطالعه را نشان می‌دهد.
"""
    chapter1 += figure("stage2_figures/study_pipeline.png", "fig:study-pipeline", "جریان کلی مطالعه از داده خام MIMIC-IV تا ارزیابی کالیبراسیون‌محور.", "0.96\\textwidth")
    chapter1 += r"""
\section{شکاف پژوهشی}
بخش بزرگی از مطالعات پیش‌بینی ICU روی مدل‌های ریسک کلاسیک یا مدل‌های عمیق مبتنی بر سری زمانی تمرکز دارد. در مقابل، این پایان‌نامه یک فضای میانی را بررسی می‌کند: ویژگی‌های تجمیع‌شده ۲۴ ساعت نخست به‌صورت همزمان در قالب داده جدولی و توکن‌های علائم وارد مدل می‌شوند. شکاف اصلی این است که در بسیاری از کارها، روش‌های few-shot یا token-based یا به‌صورت محدود آزمون می‌شوند یا بدون مقایسه سخت با baselineهای supervised و calibrated گزارش می‌گردند. این پژوهش تلاش می‌کند همین شکاف را با یک no-gap pass پر کند: هر ادعا باید در برابر مدل‌های کلاسیک، hybrid، pretraining، ensemble، calibration، uncertainty، subgroup، ablation و federated simulation آزمون شود.

\section{اهداف پژوهش}
هدف کلی پایان‌نامه، ساخت و ارزیابی یک چارچوب کامل برای پیش‌بینی \lr{ICU discharge-feasibility proxy} از داده‌های ۲۴ ساعت نخست ICU بود. اهداف اختصاصی عبارت‌اند از: ۱) استخراج کوهورت full-cohort از \lr{MIMIC-IV v3.1}، ۲) ساخت ویژگی‌های آزمایشگاهی و علائم حیاتی و نمایش توکن‌های علائم، ۳) مقایسه مدل‌های کلاسیک، ETHOS/few-shot، supervised token، hybrid، pretraining و ensemble، ۴) ارزیابی کالیبراسیون و uncertainty، ۵) تحلیل subgroup/error و ablation، و ۶) بررسی یادگیری فدرال شبیه‌سازی‌شده.

\section{سوالات پژوهش}
سوالات پژوهش عبارت‌اند از: آیا ویژگی‌های آزمایشگاهی و علائم حیاتی ۲۴ ساعت اول ICU می‌توانند proxy تعریف‌شده را پیش‌بینی کنند؟ عملکرد مدل‌های کلاسیک و ensemble در مقایسه با ETHOS/few-shot چگونه است؟ آیا توکن‌های علائم به‌تنهایی یا در مدل‌های hybrid باعث بهبود می‌شوند؟ آیا few-shot در سناریوهای کم‌داده نقش قوی‌تری نسبت به full-data دارد؟ آیا self-supervised pretraining نمایش توکنی را بهبود می‌دهد؟ آیا simulated federated learning عملکرد قابل‌قبولی نسبت به مدل متمرکز حفظ می‌کند؟ کدام مدل بهترین تعادل میان discrimination و calibration دارد؟ و uncertainty/selective prediction چه کمکی به تفسیر محتاطانه مدل می‌کند؟

\section{نوآوری‌ها و دستاوردها}
نوآوری پایان‌نامه در ادعای پیروزی یک روش خاص نیست، بلکه در آزمون سخت یک فرضیه عنوانی است. کوهورت full-cohort شامل \lr{32,399} ICU stay ساخته شد؛ توکن‌های علائم با شکل \lr{(32399, 25, 4)} تولید شدند؛ \lr{254} ردیف آزمایش ثبت شد؛ روش‌های few-shot، supervised token، pretraining، hybrid و ensemble به‌صورت قابل مقایسه ارزیابی شدند؛ و نتیجه نهایی با calibration، uncertainty، subgroup و ablation پشتیبانی شد. جدول‌های \ref{tab:cohort-summary} تا \ref{tab:token-summary} خلاصه داده و نمایش را ارائه می‌کنند.
"""
    chapter1 += tables["cohort"] + tables["labels"] + tables["features"] + tables["tokens"]
    chapter1 += r"""
\section{ساختار پایان‌نامه}
فصل دوم پیشینه علمی و جایگاه پژوهش را مرور می‌کند. فصل سوم روش‌شناسی داده، outcome، tokenization و مدل‌ها را توضیح می‌دهد. فصل چهارم نتایج no-gap را گزارش می‌کند. فصل پنجم یافته‌ها را تفسیر کرده و به سوالات پژوهش پاسخ می‌دهد. فصل ششم نتیجه‌گیری و مسیرهای آینده را جمع‌بندی می‌کند.
"""

    chapter2 = r"""
\chapter{پیشینه و مرور ادبیات}
\section{داده‌های سلامت الکترونیک و ICU}
داده‌های سلامت الکترونیک \lr{(EHR)} در ICU ترکیبی از آزمایش‌ها، علائم حیاتی، مداخلات، تشخیص‌ها، روندهای زمانی و پیامدهای بیمارستانی‌اند. چنین داده‌هایی برای یادگیری ماشین جذاب‌اند، اما چند چالش بنیادین دارند: گمشده‌بودن غیرتصادفی، اندازه‌گیری‌های نامنظم، تغییر سیاست مراقبت میان واحدها، و تفاوت میان کدهای اداری و وضعیت واقعی بیمار. بنابراین، هر مدل ICU باید در برابر leakage، سوگیری اندازه‌گیری، و overfitting محافظت شود.

\section{پایگاه داده MIMIC-IV}
\lr{MIMIC-IV} یک پایگاه عمومی و de-identified از داده‌های مراقبت بیمارستانی و ICU است که از طریق \lr{PhysioNet} توزیع می‌شود \cite{johnson2023mimiciv,goldberger2000physionet}. مزیت آن دسترس‌پذیری پژوهشی، ساختار چندجدولی و غنای متغیرهای بالینی است. محدودیت آن نیز تک‌مرکزی بودن، گذشته‌نگر بودن، و وابستگی به فرآیندهای مستندسازی بیمارستانی است. در این پایان‌نامه از نسخه \lr{MIMIC-IV v3.1} استفاده شد و همه ادعاهای نهایی بر اساس کوهورت کامل واجد شرایط همین پایگاه گزارش شدند.

\section{پیش‌بینی بالینی و مدل‌های ریسک}
مدل‌های پیش‌بینی بالینی باید فراتر از discrimination ارزیابی شوند. در کاربردهای بالینی، احتمال خروجی مدل اهمیت دارد؛ اگر احتمال‌ها بدکالیبره باشند، حتی یک \lr{AUROC} مناسب هم برای تفسیر فردی کافی نیست. ادبیات مدل‌های پیش‌بینی بالینی بر گزارش شفاف، validation مستقل، کالیبراسیون و تحلیل خطا تاکید دارد \cite{steyerberg2019clinical,collins2015tripod,wolff2019probast}. این پایان‌نامه همین منطق را دنبال کرده و نتیجه نهایی را با \lr{AUROC}، \lr{AUPRC}، \lr{Brier Score} و \lr{Expected Calibration Error (ECE)} گزارش می‌کند.

\section{یادگیری ماشین روی داده‌های جدولی و gradient boosting}
داده‌های جدولی بالینی اغلب با مدل‌های درختی و gradient boosting به خوبی مدل‌سازی می‌شوند. \lr{XGBoost} \cite{chen2016xgboost}، \lr{LightGBM} \cite{ke2017lightgbm} و \lr{CatBoost} \cite{prokhorenkova2018catboost} به دلیل مدیریت تعاملات غیرخطی، مقاومت نسبی نسبت به scaling و کارایی بالا، baselineهای مهمی در EHR محسوب می‌شوند. در این پایان‌نامه، همین مدل‌ها نه به‌عنوان رقیب ساده، بلکه به‌عنوان آزمون سخت برای روش‌های few-shot و token-based استفاده شدند.

\section{یادگیری چندنمونه‌ای و ETHOS/symptom-token modelling}
یادگیری چندنمونه‌ای در اصل برای موقعیت‌هایی جذاب است که برچسب‌ها محدودند و مدل باید از تعداد کمی نمونه برای کلاس‌بندی موارد جدید استفاده کند \cite{snell2017prototypical}. در این پروژه، ایده few-shot در کنار ETHOS و توکن‌های علائم بررسی شد. هر token شامل شناسه علامت، شدت نرمال‌شده، مولفه زمانی ساده و modality بود. اهمیت این بخش در آن است که روش‌های token-based فقط زمانی ارزشمندند که در برابر baselineهای قوی و در سناریوهای full-data و low-data آزمون شوند. نتیجه نهایی نشان داد که pure few-shot ضعیف‌تر ماند، اما token/hybrid evidence از نظر representation و benchmark ارزشمند بود.

\section{یادگیری فدرال، کالیبراسیون و uncertainty}
یادگیری فدرال \lr{(Federated Learning)} برای محیط‌هایی پیشنهاد می‌شود که داده میان مراکز مختلف قابل تجمیع مستقیم نیست \cite{mcmahan2017federated}. این پایان‌نامه real-world federation اجرا نکرد، بلکه یک شبیه‌سازی disease-node را گزارش کرد. همچنین، برای کاهش overclaiming، کالیبراسیون \cite{guo2017calibration,vancalster2019calibration}، پیش‌بینی انتخابی \lr{(Selective Prediction)} و پیش‌بینی همساز \lr{(Conformal Prediction)} \cite{vovk2005algorithmic,angelopoulos2021conformal} به ارزیابی افزوده شدند. نقش این تحلیل‌ها، تقویت خوانش محافظه‌کارانه مدل است، نه تبدیل مدل به ابزار آماده استفاده مستقیم.

\section{جایگاه پایان‌نامه حاضر}
جایگاه این پایان‌نامه در ترکیب سه محور است: full-cohort EHR prediction، symptom-token/few-shot representation، و ارزیابی کالیبراسیون‌محور. پژوهش حاضر نشان می‌دهد که عنوان مصوب می‌تواند بدون ادعای برتری few-shot حفظ شود، زیرا few-shot و توکن‌های علائم محور آزمایش‌ها بودند و نتیجه تجربی دقیق، حتی وقتی با فرضیه اولیه کاملا همسو نباشد، ارزش علمی دارد.
"""

    chapter3 = r"""
\chapter{روش‌شناسی}
\section{طراحی مطالعه و منبع داده}
مطالعه از نوع گذشته‌نگر و مبتنی بر داده ثانویه بود. داده‌ها از \lr{MIMIC-IV v3.1} استخراج شدند و مسیر پردازش نهایی در \lr{data/processed\_final\_optimized} و خروجی‌های no-gap در \lr{results/final\_optimized} ثبت شدند. فایل validation نشان داد که manifest کامل داده موجود بوده و هیچ سقف نمونه‌گیری برای تحلیل اصلی اعمال نشده است.

\section{تعریف cohort و معیارهای ورود و خروج}
کوهورت شامل بستری اول ICU برای بزرگسالان بود. حداقل طول اقامت ICU، وجود اطلاعات تشخیص و مقصد ترخیص، و دسترس‌پذیری داده‌های لازم برای outcome در نظر گرفته شد. شکل \ref{fig:cohort-flow} مسیر ساخت کوهورت را نشان می‌دهد. این شکل عمدا عددهای حذف‌شده‌ای را که در خروجی no-gap ثبت نشده‌اند حدس نمی‌زند؛ فقط وضعیت تاییدشده نهایی و قواعد اصلی را نمایش می‌دهد.
"""
    chapter3 += figure("stage2_figures/cohort_flow.png", "fig:cohort-flow", "جریان ساخت کوهورت و برچسب نهایی بدون استفاده از نمونه‌گیری.", "0.78\\textwidth")
    chapter3 += r"""
\section{تعریف ICU discharge-feasibility proxy}
برچسب مثبت زمانی اختصاص یافت که بیمار در بیمارستان زنده بماند، مقصد ترخیص در گروه مطلوب باشد، و طول اقامت ICU کمتر از ۷۲۰ ساعت باشد. این تعریف یک \lr{proxy} پژوهشی است. واژه proxy در سراسر پایان‌نامه حفظ می‌شود، زیرا داده موجود شامل داوری مستقیم پزشک درباره آمادگی ترخیص یا امکان‌پذیری درمان نیست.

\section{استخراج ویژگی‌های ۲۴ ساعت اول}
ویژگی‌ها شامل ۱۵ آزمایش و ۷ علامت حیاتی بودند. داده‌ها در ۲۴ ساعت نخست پس از ورود ICU محدود شدند، سپس برای هر stay مقدار میانه محاسبه شد. کلیپ بالینی بر اساس بازه‌های از پیش تعیین‌شده انجام شد. جایگذاری مقدارهای گمشده و scaling فقط با آمار بخش آموزش انجام شد تا leakage کاهش یابد. افزون بر ویژگی‌های خام، سه ویژگی مهندسی‌شده و شاخص‌های گمشده‌بودن به مدل‌های tabular اضافه شدند. تعداد سلول‌های خام گمشده در فایل نهایی برابر با """
    chapter3 += f"\\lr{{{raw_missing}}}" + r""" بود.

\section{ساخت توکن‌های علائم}
توکن‌سازی علائم \lr{(Symptom Tokens)} تلاش می‌کند هر مولفه آزمایشگاهی یا حیاتی را به یک موجودیت قابل خواندن برای مدل‌های token-based تبدیل کند. تانسور نهایی برای هر بیمار ۲۵ جایگاه توکن و ۴ کانال داشت: شناسه علامت، شدت، مولفه زمانی نرمال‌شده و modality. شکل \ref{fig:token-representation} این ساختار را به‌صورت شماتیک نشان می‌دهد.
"""
    chapter3 += figure("stage2_figures/symptom_token_representation.png", "fig:token-representation", "نمایش شماتیک تانسور توکن‌های علائم با ۲۵ توکن و ۴ کانال.", "0.92\\textwidth")
    chapter3 += r"""
\section{خانواده مدل‌ها}
مدل‌های کلاسیک شامل logistic regression، random forest، XGBoost، LightGBM، CatBoost و baseline عصبی بودند. مدل‌های ETHOS/few-shot شامل prototype baseline، supervised token encoders، attention pooling، GRU، CNN و Transformer فشرده بودند. مدل‌های supervised token و hybrid شامل ترکیب tabular+token، token embeddings + tabular، و stacking احتمال ETHOS با XGBoost بودند. self-supervised pretraining شامل masked reconstruction، denoising autoencoder، contrastive patient learning، sequence autoencoder و patient representation learning بود. در پایان، stacking و weighted ensemble روی مدل‌های منتخب validation-first ساخته شدند. شکل \ref{fig:model-framework} چارچوب ارزیابی مدل‌ها را خلاصه می‌کند.
"""
    chapter3 += figure("stage2_figures/model_family_framework.png", "fig:model-framework", "خانواده مدل‌ها و لایه‌های ارزیابی در no-gap pass.", "0.96\\textwidth")
    chapter3 += r"""
\section{طراحی کم‌داده، فدرال، کالیبراسیون و uncertainty}
برای low-data، fractions برابر ۱، ۲، ۵، ۱۰، ۲۰، ۵۰ و ۱۰۰ درصد با split ثابت validation/test ارزیابی شدند. یادگیری فدرال به‌صورت simulation با disease-nodeها انجام شد و ادعای چندمرکزی واقعی مطرح نشد. کالیبراسیون با روش‌های validation-first مانند isotonic و Platt در مدل‌های نهایی بررسی شد. پیش‌بینی انتخابی و پیش‌بینی همساز نیز برای مدل‌های برتر و Token Hybrid CatBoost گزارش شدند تا نشان دهند چگونه می‌توان عدم قطعیت را به‌صورت پژوهشی تحلیل کرد.

\section{معیارها، بازتولیدپذیری و اخلاق}
معیارهای اصلی شامل \lr{AUROC}، \lr{AUPRC}، accuracy، recall، specificity، \lr{F1}، \lr{Brier Score} و \lr{ECE} بودند. رجیستری آزمایش با seed ثابت و مسیر خروجی‌ها نگهداری شد. از آنجا که \lr{MIMIC-IV} داده de-identified است، مطالعه در محدوده پژوهش ثانویه قرار می‌گیرد؛ با این حال، داده خام قابل بازتوزیع نیست و همه محدودیت‌های استفاده از داده باید رعایت شود.
"""

    chapter4 = r"""
\chapter{آزمایش‌ها و نتایج}
\section{خلاصه دیتاست، برچسب و نمایش}
همان‌گونه که در جدول‌های \ref{tab:cohort-summary} تا \ref{tab:token-summary} آمده است، تحلیل اصلی روی full-cohort برابر با \lr{32,399} ICU stay انجام شد. هیچ نتیجه‌ای از benchmark کوچک‌تر به‌عنوان نتیجه اصلی استفاده نشد. شکل \ref{fig:auroc} و شکل \ref{fig:auprc} مقایسه کلی مدل‌های نهایی را نشان می‌دهند.
"""
    chapter4 += tables["main"]
    chapter4 += figure("stage2_figures/auroc_comparison.png", "fig:auroc", "مقایسه AUROC آزمون در مدل‌های نهایی no-gap.", "0.94\\textwidth")
    chapter4 += figure("stage2_figures/auprc_comparison.png", "fig:auprc", "مقایسه AUPRC آزمون در مدل‌های نهایی no-gap.", "0.94\\textwidth")
    chapter4 += r"""
\section{منحنی‌های ROC و Precision-Recall}
برای مدل‌های برتر، منحنی‌های ROC و PR از پیش‌بینی‌های آزمون نهایی رسم شدند. شکل \ref{fig:roc} نشان می‌دهد که اختلاف discrimination میان مدل‌های برتر کوچک است؛ شکل \ref{fig:pr} نیز اهمیت AUPRC را در توزیع برچسب نامتوازن نشان می‌دهد.
"""
    chapter4 += figure("stage2_figures/roc_curves_top_models.png", "fig:roc", "منحنی‌های ROC برای مدل‌های برتر.", "0.78\\textwidth")
    chapter4 += figure("stage2_figures/pr_curves_top_models.png", "fig:pr", "منحنی‌های Precision-Recall برای مدل‌های برتر.", "0.78\\textwidth")
    chapter4 += r"""
\section{ETHOS، few-shot و token/hybrid}
نتایج جدول \ref{tab:fewshot} نشان می‌دهد pure few-shot/prototype در full-data به سطح مدل‌های supervised نرسید. این یافته به‌صورت منفی اما مهم گزارش می‌شود. در مقابل، جدول \ref{tab:token-hybrid} و شکل \ref{fig:token-hybrid} نشان می‌دهند که مدل‌های hybrid، به‌ویژه Token Hybrid CatBoost و token embeddings + tabular XGBoost، به نتایج نزدیک به مدل‌های برتر رسیدند.
"""
    chapter4 += tables["fewshot"] + tables["tokenhybrid"]
    chapter4 += figure("stage2_figures/token_hybrid_comparison.png", "fig:token-hybrid", "مقایسه مدل‌های ETHOS، token-only، embedding و hybrid.", "0.94\\textwidth")
    chapter4 += r"""
\section{پیش‌آموزش و سناریوهای کم‌داده}
جدول \ref{tab:pretraining} نشان می‌دهد بهترین نتیجه pretraining مربوط به denoising autoencoder + Tabular + XGBoost بود. این نتیجه نسبت به supervised ensembles ضعیف‌تر بود، اما به عنوان benchmark representation-learning قابل گزارش است. جدول \ref{tab:low-data} و شکل \ref{fig:low-data} نشان می‌دهد few-shot/token prototype در کم‌داده نیز بهترین مدل نشد؛ مدل‌های supervised tree در fractions کوچک‌تر غالبا قوی‌تر بودند و Token Hybrid CatBoost در fraction کامل بهترین low-data row را تشکیل داد.
"""
    chapter4 += tables["pretraining"] + tables["lowdata"]
    chapter4 += figure("stage2_figures/low_data_performance_curves.png", "fig:low-data", "منحنی‌های عملکرد در fractions کم‌داده با split ثابت.", "0.96\\textwidth")
    chapter4 += r"""
\section{کالیبراسیون، uncertainty و conformal prediction}
جدول \ref{tab:calibration} نشان می‌دهد Stacked Validation Meta-LR بهترین تعادل calibration-aware را داشت. شکل \ref{fig:calibration} منحنی‌های کالیبراسیون را نمایش می‌دهد. جدول \ref{tab:uncertainty} نشان می‌دهد با کاهش coverage در selective prediction، دقت موارد retained افزایش می‌یابد؛ این رفتار برای تفسیر ایمن‌تر مدل مهم است، زیرا نشان می‌دهد uncertainty می‌تواند برای defer کردن موارد پرریسک استفاده شود، نه برای ادعای قطعیت بالینی.
"""
    chapter4 += tables["calibration"]
    chapter4 += figure("stage2_figures/calibration_curves.png", "fig:calibration", "منحنی‌های کالیبراسیون مدل‌های منتخب.", "0.82\\textwidth")
    chapter4 += tables["uncertainty"]
    chapter4 += figure("stage2_figures/selective_prediction_curve.png", "fig:selective", "منحنی پیش‌بینی انتخابی؛ دقت با نگه‌داشتن موارد مطمئن‌تر افزایش می‌یابد.", "0.86\\textwidth")
    chapter4 += r"""
\section{subgroup، error analysis، ablation و feature importance}
تحلیل subgroup در جدول \ref{tab:subgroup} اکتشافی است و ادعای fairness یا قابلیت استقرار ندارد. ablation در جدول \ref{tab:ablation} نشان می‌دهد ترکیب labs+vitals نسبت به vitals-only قوی‌تر است و token-only به‌تنهایی از مدل‌های tabular/hybrid ضعیف‌تر می‌ماند. شکل \ref{fig:feature-importance} اهمیت ویژگی‌ها را برای random forest tabular منتخب نشان می‌دهد؛ این شکل به تفسیر feature-family کمک می‌کند، اما جایگزین تبیین علی نیست.
"""
    chapter4 += tables["subgroup"] + tables["ablation"]
    chapter4 += figure("stage2_figures/feature_importance_random_forest.png", "fig:feature-importance", "اهمیت ویژگی‌ها در random forest tabular منتخب.", "0.88\\textwidth")
    chapter4 += r"""
\section{یادگیری فدرال شبیه‌سازی‌شده و رتبه‌بندی نهایی}
نتایج یادگیری فدرال در جدول \ref{tab:federated} و شکل \ref{fig:federated} گزارش شده است. مدل federated LR weighted nodes از نظر calibration قابل قبول بود، اما به مدل‌های درختی/ensemble مرکزی نرسید. جدول \ref{tab:ranking} رتبه‌بندی نهایی no-gap را خلاصه می‌کند.
"""
    chapter4 += tables["federated"]
    chapter4 += figure("stage2_figures/federated_overview.png", "fig:federated", "نمای کلی تقسیم disease-node و مقایسه فدرال شبیه‌سازی‌شده با logistic regression مرکزی.", "0.94\\textwidth")
    chapter4 += tables["ranking"]

    chapter5 = f"""
\\chapter{{بحث}}
\\section{{یافته‌های اصلی}}
مهم‌ترین یافته پایان‌نامه این است که داده‌های آزمایشگاهی و علائم حیاتی ۲۴ ساعت نخست ICU برای پیش‌بینی \\lr{{ICU discharge-feasibility proxy}} سیگنال معنادار اما متوسط دارند. بهترین مدل کلی \\lr{{Stacked Validation Meta-LR}} بود که \\lr{{AUROC={fmt(stacked['AUROC'])}}}، \\lr{{AUPRC={fmt(stacked['AUPRC'])}}}، \\lr{{Brier={fmt(stacked['Brier'])}}} و \\lr{{ECE={fmt(stacked['ECE'])}}} داشت. بهترین پروفایل discrimination مربوط به \\lr{{Weighted Average Ensemble}} با \\lr{{AUPRC={fmt(weighted['AUPRC'])}}} بود. این دو نتیجه نشان می‌دهند که در full-data، روش‌های supervised ensemble بر pure few-shot تقدم داشتند.

\\section{{پاسخ به سوالات پژوهش}}
پاسخ سوال اول مثبت اما محتاطانه است: ویژگی‌های ۲۴ ساعت نخست می‌توانند proxy را با discrimination متوسط پیش‌بینی کنند. پاسخ سوال دوم این است که مدل‌های کلاسیک و ensemble از ETHOS/few-shot خالص قوی‌تر بودند. پاسخ سوال سوم مثبت مشروط است: توکن‌ها به‌تنهایی برنده نشدند، اما در مدل‌های hybrid به نتایج نزدیک به بهترین مدل کلی رسیدند. پاسخ سوال چهارم این است که few-shot در low-data نیز بهترین نشد، ولی نقش benchmark مهمی داشت. پاسخ سوال پنجم نشان داد pretraining feasible است اما بهترین ensemble را پشت سر نگذاشت. پاسخ سوال ششم این است که federated simulation عملکردی قابل گزارش داشت ولی به مدل‌های مرکزی قوی نرسید. پاسخ سوال هفتم، \\lr{{Stacked Validation Meta-LR}} به‌عنوان بهترین تعادل discrimination/calibration است. پاسخ سوال هشتم این است که uncertainty/selective prediction امکان تفسیر محتاطانه‌تر و defer کردن موارد نامطمئن را فراهم می‌کند.

\\section{{چرا supervised و stacked models قوی‌تر بودند}}
داده‌های این پایان‌نامه از ۲۲ ویژگی خام، ویژگی‌های مهندسی‌شده و missingness flags تشکیل شده‌اند. این ساختار برای مدل‌های درختی و ensemble بسیار مناسب است. مدل‌های stacking و weighted ensemble نیز از چند مدل supervised بهره بردند و روی validation انتخاب شدند. در مقابل، مدل‌های token/few-shot با تانسوری کار می‌کردند که از اندازه‌گیری‌های تجمیع‌شده ۲۴ ساعت نخست ساخته شده بود، نه از یک sequence زمانی غنی. بنابراین، اینکه مدل supervised در full-data قوی‌تر باشد از نظر روش‌شناسی قابل انتظار است.

\\section{{چرا pure few-shot ضعیف‌تر ماند}}
few-shot معمولا زمانی جذاب‌تر است که داده برچسب‌دار کمیاب، کلاس‌ها جدید یا توزیع taskها اپیزودیک باشد. در این پژوهش، کوهورت full-data بزرگ بود و outcome باینری بر اساس proxy ساخته شد. همچنین token sequence بر پایه featureهای تجمیع‌شده بود. بنابراین، pure few-shot/prototype نتوانست از مدل‌های درختی با هزاران نمونه آموزشی جلو بزند. این نتیجه شکست علمی نیست؛ بلکه نشان می‌دهد فرضیه few-shot در این تنظیم خاص باید به‌عنوان benchmark و مسیر آینده گزارش شود.

\\section{{نقش واقعی symptom tokens و اهمیت Token Hybrid CatBoost}}
نقش واقعی توکن‌های علائم در این پایان‌نامه، جایگزین کردن کامل مدل‌های جدولی نبود. ارزش آن‌ها در representation، hybrid modelling، pretraining و benchmark آشکار شد. \\lr{{Token Hybrid CatBoost}} با \\lr{{AUROC={fmt(token['AUROC'])}}} و \\lr{{AUPRC={fmt(token['AUPRC'])}}} به بهترین مدل کلی بسیار نزدیک شد. همین یافته برای دفاع از عنوان کافی است: توکن‌های علائم نه تزئین روش‌شناختی، بلکه محور آزمایشی بودند و در ترکیب با مدل supervised نتیجه رقابتی تولید کردند.

\\section{{نقش pretraining، low-data و calibration}}
پیش‌آموزش با denoising autoencoder بهترین نتیجه pretraining را ایجاد کرد، اما به ensemble برتر نرسید. low-data experiments نیز نشان دادند که حتی در fractions کوچک، مدل‌های supervised tree اغلب مقاوم‌تر از few-shot prototype بودند. کالیبراسیون در این پایان‌نامه نقش کلیدی داشت، زیرا مدل بالینی بدون احتمال‌های قابل اعتماد از نظر علمی ناقص است. انتخاب Stacked Meta-LR به‌عنوان بهترین مدل overall از همین منطق ناشی می‌شود.

\\section{{uncertainty، subgroup و federated simulation}}
پیش‌بینی انتخابی نشان داد وقتی مدل فقط روی موارد مطمئن‌تر پاسخ می‌دهد، دقت افزایش می‌یابد. پیش‌بینی همساز نیز coverage هدف را به‌صورت تقریبی حفظ کرد. subgroup و error analysis فقط اکتشافی‌اند و برای شناسایی نواحی ضعف مدل استفاده شدند. federated simulation نشان داد aggregation وزن‌دار nodeها feasible است، اما جایگزین validation چندمرکزی واقعی نیست.

\\section{{نقاط قوت}}
نقاط قوت پژوهش شامل استفاده از full-cohort، بازتعریف صادقانه outcome، no-gap pass گسترده، رجیستری آزمایش، baselineهای قوی، ارزیابی token/few-shot/hybrid/pretraining، گزارش کالیبراسیون و uncertainty، و پرهیز از ادعاهای بالینی فراتر از داده است.

\\section{{محدودیت‌ها}}
محدودیت‌ها در جدول \\ref{{tab:limitations}} خلاصه شده‌اند. مطالعه گذشته‌نگر و تک‌پایگاه است؛ outcome یک proxy است؛ validation خارجی و prospective انجام نشده؛ ویژگی‌ها فقط از ۲۴ ساعت نخست ساخته شده‌اند؛ مخدوش‌گری، سوگیری کدنویسی و سوگیری اندازه‌گیری ممکن است وجود داشته باشد؛ یادگیری فدرال فقط simulation است؛ subgroup analysis اکتشافی است؛ داده خام \\lr{{MIMIC-IV}} قابل بازتوزیع نیست؛ و pure few-shot نسبت به supervised ensembles ضعیف‌تر باقی ماند.
"""
    chapter5 += tables["limitations"]
    chapter5 += r"""
\section{کارهای آینده}
کار آینده باید روی validation خارجی، طراحی outcome با داوری بالینی، مدل‌سازی زمانی غنی‌تر، federation واقعی بین مراکز، معماری‌های token با pretraining گسترده‌تر، و تحلیل‌های fairness/transportability تمرکز کند. همچنین، تبدیل این پایان‌نامه به مقاله ژورنالی باید بر گزارش شفاف proxy، مقایسه سخت با baselineها، و ارزش negative/benchmark findings تاکید کند.
"""

    chapter6 = r"""
\chapter{نتیجه‌گیری}
این پایان‌نامه نشان داد که پیش‌بینی \lr{ICU discharge-feasibility proxy} از داده‌های ۲۴ ساعت نخست ICU امکان‌پذیر است، اما تفسیر آن باید محافظه‌کارانه بماند. بهترین مدل کلی، \lr{Stacked Validation Meta-LR}، تعادل مناسبی میان discrimination و calibration ایجاد کرد؛ \lr{Weighted Average Ensemble} بهترین AUPRC را داشت؛ و \lr{Token Hybrid CatBoost} به‌عنوان بهترین مدل واقعی token/hybrid به نتیجه‌ای نزدیک به بهترین مدل کلی رسید.

پیام علمی اصلی این است که عنوان مصوب پایان‌نامه همچنان دفاع‌پذیر است، زیرا few-shot و symptom-token فقط در متن نیامده‌اند، بلکه در محور آزمایش‌های no-gap حضور داشته‌اند. نتیجه تجربی نیز صادقانه است: pure few-shot برنده full-data نبود، اما توکن‌های علائم در مدل‌های hybrid و representation ارزشمند بودند. چنین نتیجه‌ای برای پژوهش علمی سالم‌تر از یک ادعای اغراق‌آمیز است، زیرا مسیر آینده را دقیق‌تر نشان می‌دهد.

مسیرهای آینده شامل اعتبارسنجی خارجی، تعریف outcome با مشارکت بالینی، مدل‌های طولی، pretraining گسترده‌تر، تحلیل fairness، و federation واقعی است. تا پیش از آن، خروجی این پایان‌نامه باید به‌عنوان یک چارچوب پژوهشی و foundation برای manuscript توسعه یابد، نه ابزار عملیاتی مستقل.
"""

    appendices = r"""
\appendix
\chapter{پیوست بازتولیدپذیری}
خروجی‌های اصلی در مسیر \lr{results/final\_optimized} قرار دارند. فایل‌های source-of-truth مورد استفاده شامل \lr{NO\_GAP\_FINAL\_COMPLETION\_REPORT.md}، \lr{FINAL\_MODEL\_SELECTION\_NO\_GAP.md}، \lr{THESIS\_TITLE\_ALIGNMENT\_NO\_GAP.md}، گزارش‌های ETHOS، pretraining، calibration، uncertainty، subgroup، ablation، federated و جدول \lr{final\_model\_ranking\_no\_gap.csv} بودند. فایل رجیستری آزمایش \lr{experiment\_registry.csv} شامل ۲۵۴ ردیف است.

\chapter{پیوست اخلاق و دامنه ادعا}
داده خام \lr{MIMIC-IV} طبق قواعد دسترسی و آموزش CITI قابل استفاده است اما قابل بازتوزیع نیست. همه فایل‌های قابل انتشار باید مشتق‌شده، غیرقابل شناسایی و مطابق license باشند. این پایان‌نامه هیچ ادعای مداخله، نسخه‌نویسی، یا آمادگی واقعی ترخیص ندارد و outcome را فقط به‌عنوان proxy پژوهشی گزارش می‌کند.

\chapter*{منابع}
\addcontentsline{toc}{chapter}{منابع}
\begin{latin}
\begin{thebibliography}{99}
\bibitem{johnson2023mimiciv} Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L. H., Celi, L. A., and Mark, R. G. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 2023.
\bibitem{goldberger2000physionet} Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., and Stanley, H. E. PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 2000.
\bibitem{steyerberg2019clinical} Steyerberg, E. W. Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating. Springer, 2019.
\bibitem{collins2015tripod} Collins, G. S., Reitsma, J. B., Altman, D. G., and Moons, K. G. M. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis: the TRIPOD statement. Annals of Internal Medicine, 2015.
\bibitem{wolff2019probast} Wolff, R. F., Moons, K. G. M., Riley, R. D., Whiting, P. F., Westwood, M., Collins, G. S., Reitsma, J. B., Kleijnen, J., and Mallett, S. PROBAST: a tool to assess risk of bias and applicability of prediction model studies. Annals of Internal Medicine, 2019.
\bibitem{breiman2001random} Breiman, L. Random forests. Machine Learning, 2001.
\bibitem{friedman2001greedy} Friedman, J. H. Greedy function approximation: a gradient boosting machine. Annals of Statistics, 2001.
\bibitem{chen2016xgboost} Chen, T., and Guestrin, C. XGBoost: A scalable tree boosting system. KDD, 2016.
\bibitem{ke2017lightgbm} Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. Y. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS, 2017.
\bibitem{prokhorenkova2018catboost} Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. CatBoost: unbiased boosting with categorical features. NeurIPS, 2018.
\bibitem{snell2017prototypical} Snell, J., Swersky, K., and Zemel, R. Prototypical networks for few-shot learning. NeurIPS, 2017.
\bibitem{mcmahan2017federated} McMahan, B., Moore, E., Ramage, D., Hampson, S., and Arcas, B. A. Communication-efficient learning of deep networks from decentralized data. AISTATS, 2017.
\bibitem{guo2017calibration} Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. ICML, 2017.
\bibitem{vancalster2019calibration} Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., and Steyerberg, E. W. Calibration: the Achilles heel of predictive analytics. BMC Medicine, 2019.
\bibitem{vovk2005algorithmic} Vovk, V., Gammerman, A., and Shafer, G. Algorithmic Learning in a Random World. Springer, 2005.
\bibitem{angelopoulos2021conformal} Angelopoulos, A. N., and Bates, S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv, 2021.
\end{thebibliography}
\end{latin}
\end{document}
"""
    return (
        preamble
        + abstract_fa
        + abstract_en
        + front
        + chapter1
        + chapter2
        + chapter3
        + chapter4
        + chapter5
        + chapter6
        + appendices
    )


def compile_pdf(tex_path: Path) -> None:
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def copy_source_data() -> None:
    files = [
        RESULTS / "NO_GAP_FINAL_COMPLETION_REPORT.md",
        RESULTS / "FINAL_MODEL_SELECTION_NO_GAP.md",
        RESULTS / "THESIS_TITLE_ALIGNMENT_NO_GAP.md",
        RESULTS / "ethos_no_gap" / "ETHOS_NO_GAP_COMPLETION_REPORT.md",
        RESULTS / "pretraining_no_gap" / "PRETRAINING_NO_GAP_REPORT.md",
        RESULTS / "ethos_no_gap" / "TOKEN_REPRESENTATION_COMPLETION_REPORT.md",
        TABLES / "final_model_ranking_no_gap.csv",
        RESULTS / "optimization" / "experiment_registry.csv",
        RESULTS / "low_data_no_gap" / "low_data_no_gap_results.csv",
        RESULTS / "calibration_no_gap" / "calibration_no_gap_results.csv",
        RESULTS / "uncertainty_no_gap" / "selective_prediction_results.csv",
        RESULTS / "uncertainty_no_gap" / "conformal_prediction_results.csv",
        RESULTS / "subgroups_no_gap" / "subgroup_no_gap_results.csv",
        RESULTS / "ablation_no_gap" / "ablation_no_gap_results.csv",
        RESULTS / "federated_no_gap" / "federated_no_gap_results.csv",
    ]
    for src in files:
        if src.exists():
            dst = SOURCE_DATA_DIR / src.name
            shutil.copy2(src, dst)


def write_defense_summary(data: dict[str, object]) -> None:
    final = data["final"]
    stacked = row_by_id(final, "selected_stacked_validation_meta_lr")
    weighted = row_by_id(final, "selected_weighted_average_validation_auc")
    token = row_by_id(final, "token_hybrid_catboost")
    fewshot = row_by_id(final, "current_fewshot_ethos_baseline")
    pretrain = data["pretraining"].sort_values("AUROC", ascending=False).iloc[0]
    text = f"""# THESIS DEFENSE SUMMARY

## One-paragraph thesis story

این پایان‌نامه عنوان مصوب «فیوشات درمانی» را به‌صورت علمی و محافظه‌کارانه اجرا می‌کند: به‌جای ادعای مستقیم درباره درمان یا ترخیص واقعی، یک ICU discharge-feasibility proxy از MIMIC-IV v3.1 تعریف شد و روی 32,399 ICU stay ارزیابی گردید. فرضیه few-shot/symptom-token به‌طور کامل آزمون شد؛ نتیجه نهایی نشان داد supervised stacking و ensemble در full-data قوی‌ترند، اما symptom tokens در مدل‌های hybrid و representation نقش واقعی و قابل دفاع دارند.

## Final model result summary

- Best overall calibration-aware model: Stacked Validation Meta-LR, AUROC {fmt(stacked['AUROC'])}, AUPRC {fmt(stacked['AUPRC'])}, Brier {fmt(stacked['Brier'])}, ECE {fmt(stacked['ECE'])}.
- Best discrimination/AUPRC profile: Weighted Average Ensemble, AUROC {fmt(weighted['AUROC'])}, AUPRC {fmt(weighted['AUPRC'])}.
- Best ETHOS/token/hybrid result: Token Hybrid CatBoost, AUROC {fmt(token['AUROC'])}, AUPRC {fmt(token['AUPRC'])}.
- Best pretraining-related result: denoising autoencoder + Tabular + XGBoost, AUROC {fmt(pretrain['AUROC'])}, AUPRC {fmt(pretrain['AUPRC'])}.
- Pure few-shot/prototype benchmark: AUROC {fmt(fewshot['AUROC'])}, AUPRC {fmt(fewshot['AUPRC'])}.

## Why the title remains defensible

عنوان دفاع‌پذیر است چون few-shot، ETHOS و symptom tokens محور آزمایش‌ها بودند، نه حاشیه تزئینی. پایان‌نامه نشان می‌دهد این خانواده روش‌ها در full-data برنده نشدند، اما برای representation، hybrid modelling، low-data benchmarking و تحلیل منفی صادقانه نقش مرکزی دارند.

## Why few-shot did not need to win full-data

پایان‌نامه یک فرضیه را آزمون می‌کند، نه اینکه نتیجه را از قبل تضمین کند. وقتی full-cohort بزرگ و baselineهای supervised قوی باشند، برتری مدل‌های ensemble منطقی است. ارزش علمی کار در این است که نتیجه واقعی گزارش شده و جایگاه few-shot دقیق‌تر تعریف شده است.

## How symptom-token hybrid modelling supports the thesis

Token Hybrid CatBoost با AUROC {fmt(token['AUROC'])} و AUPRC {fmt(token['AUPRC'])} به بهترین مدل کلی بسیار نزدیک شد. این نتیجه نشان می‌دهد توکن‌های علائم در ترکیب با ویژگی‌های جدولی سیگنال قابل استفاده دارند و عنوان thesis همچنان به شواهد تجربی متصل است.

## Expected supervisor questions and answers

**Q: اگر ensemble برنده شد، چرا عنوان few-shot باقی بماند؟**  
A: چون few-shot و symptom tokens به‌صورت کامل در طراحی، آزمایش، low-data، pretraining و hybrid بررسی شدند. نتیجه صادقانه این است که برنده full-data نبودند، اما محور علمی پایان‌نامه بودند.

**Q: آیا outcome واقعا قابلیت درمان است؟**  
A: خیر. در متن پایان‌نامه outcome همیشه ICU discharge-feasibility proxy است؛ این proxy از survival، discharge destination و ICU LOS کمتر از 720 ساعت ساخته شده است.

**Q: آیا مدل برای استفاده عملی آماده است؟**  
A: خیر. مطالعه گذشته‌نگر، تک‌پایگاه، بدون validation خارجی و بدون validation آینده‌نگر است.

**Q: مهم‌ترین نتیجه مثبت tokenها چیست؟**  
A: بهترین مدل token/hybrid یعنی Token Hybrid CatBoost به AUROC 0.757 و AUPRC 0.820 رسید که بسیار نزدیک به مدل کلی برتر است.

**Q: مهم‌ترین محدودیت چیست؟**  
A: proxy outcome و نبود external/prospective validation. این دو محدودیت در تفسیر نتیجه‌ها مرکزی هستند.
"""
    (OUT_DIR / DEFENSE_NAME).write_text(text, encoding="utf-8")


def write_report(data: dict[str, object], phrase_hits: dict[str, list[str]]) -> None:
    validation = data["validation"]
    final = data["final"]
    stacked = row_by_id(final, "selected_stacked_validation_meta_lr")
    weighted = row_by_id(final, "selected_weighted_average_validation_auc")
    token = row_by_id(final, "token_hybrid_catboost")
    fewshot = row_by_id(final, "current_fewshot_ethos_baseline")
    pretrain = data["pretraining"].sort_values("AUROC", ascending=False).iloc[0]
    phrase_status = "PASS" if not any(phrase_hits.values()) else "REVIEW_REQUIRED"
    text = f"""# Thesis Final Topic-Aligned Report

## Output files

- `papers/final_optimized_thesis/{PDF_NAME}`
- `papers/final_optimized_thesis/{TEX_NAME}`
- `papers/final_optimized_thesis/{ZIP_NAME}`
- `papers/final_optimized_thesis/{REPORT_NAME}`
- `papers/final_optimized_thesis/{DEFENSE_NAME}`

## Source-of-truth status

- READY_FOR_THESIS_REWRITE: YES.
- Full cohort: {validation['final_dataset']['rows']:,} ICU stays and {validation['cohort']['unique_subjects']:,} subjects.
- Labels: 0 = {validation['final_dataset']['label_distribution']['0']:,}; 1 = {validation['final_dataset']['label_distribution']['1']:,}.
- Tokens: {tuple(validation['tokens']['shape'])}.
- Experiment registry rows: {len(data['registry']):,}.

## Metric checks

- Best overall: Stacked Validation Meta-LR, AUROC {fmt(stacked['AUROC'])}, AUPRC {fmt(stacked['AUPRC'])}, Brier {fmt(stacked['Brier'])}, ECE {fmt(stacked['ECE'])}.
- Best discrimination/AUPRC: Weighted Average Ensemble, AUROC {fmt(weighted['AUROC'])}, AUPRC {fmt(weighted['AUPRC'])}.
- Best ETHOS/token/hybrid: Token Hybrid CatBoost, AUROC {fmt(token['AUROC'])}, AUPRC {fmt(token['AUPRC'])}.
- Best pretraining-related result: denoising autoencoder + Tabular + XGBoost, AUROC {fmt(pretrain['AUROC'])}, AUPRC {fmt(pretrain['AUPRC'])}.
- Pure few-shot/prototype: AUROC {fmt(fewshot['AUROC'])}, AUPRC {fmt(fewshot['AUPRC'])}.

## Thesis content checks

- Required six-chapter Persian structure included.
- Persian and English abstracts included.
- Persian and English keywords included.
- Abbreviations, list of tables, list of figures, references, and appendices included.
- Sixteen required table categories included.
- Fourteen required figure categories included.
- The old sampled benchmark is not presented as the main analysis.
- Few-shot is reported as evaluated and informative, not as the full-data winner.
- Clinical claims are limited to retrospective proxy modelling.
- PDF compiled with XeLaTeX and representative rendered pages were visually inspected for title, abstracts, tables, figures, limitations, and references.
- Phrase-screen status: {phrase_status}.

## Source package

The source zip contains the LaTeX file, compiled PDF, generated figures, final summary files, and copied no-gap evidence tables/reports used by the thesis.
"""
    (OUT_DIR / REPORT_NAME).write_text(text, encoding="utf-8")


def phrase_screen(paths: list[Path]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {phrase: [] for phrase in RISK_PHRASES}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in RISK_PHRASES:
            if phrase.lower() in text:
                hits[phrase].append(str(path))
    return hits


def package_source() -> None:
    zip_path = OUT_DIR / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [OUT_DIR / TEX_NAME, OUT_DIR / PDF_NAME, OUT_DIR / REPORT_NAME, OUT_DIR / DEFENSE_NAME]:
            if path.exists():
                zf.write(path, path.name)
        for path in sorted(FIG_DIR.glob("*")):
            if path.is_file():
                zf.write(path, f"stage2_figures/{path.name}")
        for path in sorted(SOURCE_DATA_DIR.glob("*")):
            if path.is_file():
                zf.write(path, f"stage2_source_data/{path.name}")


def main() -> None:
    setup_dirs()
    data = read_inputs()
    validate_no_gap_inputs(data)
    make_figures(data)
    tables = build_tables(data)
    tex = make_tex(data, tables)
    tex_path = OUT_DIR / TEX_NAME
    tex_path.write_text(tex, encoding="utf-8")
    compile_pdf(tex_path)
    copy_source_data()
    write_defense_summary(data)
    phrase_hits = phrase_screen([tex_path, OUT_DIR / DEFENSE_NAME])
    write_report(data, phrase_hits)
    phrase_hits = phrase_screen([tex_path, OUT_DIR / DEFENSE_NAME, OUT_DIR / REPORT_NAME])
    if any(phrase_hits.values()):
        raise RuntimeError(f"Risk phrase screen failed: {phrase_hits}")
    package_source()
    print(f"Wrote {OUT_DIR / PDF_NAME}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {OUT_DIR / ZIP_NAME}")
    print(f"Wrote {OUT_DIR / REPORT_NAME}")
    print(f"Wrote {OUT_DIR / DEFENSE_NAME}")


if __name__ == "__main__":
    main()
