#!/usr/bin/env python3
"""Build full-cohort report, PDFs, figures, and BMC submission zip from actual outputs."""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict[str, Path]:
    paths = {}
    for key, value in cfg["paths"].items():
        path = Path(value)
        paths[key] = path if path.is_absolute() else ROOT / path
    return paths


def load_outputs(paths: dict[str, Path]):
    validation_path = paths["results_dir"] / "full_cohort_validation.json"
    model_path = paths["tables_dir"] / "full_cohort_model_results.csv"
    run_path = paths["results_dir"] / "full_cohort_model_run.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing {validation_path}; run scripts/validate_full_cohort.py")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}; run scripts/run_full_cohort_models.py")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    models = pd.read_csv(model_path)
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {"statuses": []}
    missing = pd.read_csv(paths["tables_dir"] / "full_cohort_missingness.csv")
    labels = pd.read_csv(paths["tables_dir"] / "full_cohort_label_distribution.csv")
    nodes = pd.read_csv(paths["tables_dir"] / "full_cohort_node_distribution.csv")
    return validation, models, run, missing, labels, nodes


def fmt(x, digits=3) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def make_figures(paths: dict[str, Path], models: pd.DataFrame, missing: pd.DataFrame, labels: pd.DataFrame):
    fig_dir = paths["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    ordered = models.sort_values("auroc", ascending=True)
    plt.barh(ordered["model"], ordered["auroc"], color="#2f6f73")
    plt.xlabel("AUROC")
    plt.title("Full-Cohort Model AUROC")
    plt.xlim(0, max(1.0, ordered["auroc"].max() + 0.05))
    plt.tight_layout()
    plt.savefig(fig_dir / "full_cohort_model_auroc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.8, 4.2))
    plt.bar(labels["feasible"].astype(str), labels["count"], color=["#8a4f7d", "#2f6f73"])
    plt.xlabel("Feasible label")
    plt.ylabel("Count")
    plt.title("Full-Cohort Label Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "full_cohort_label_distribution.png", dpi=180)
    plt.close()

    top_missing = missing.sort_values("missing_pct", ascending=False).head(12).copy()
    top_missing["label"] = top_missing.get("itemid", top_missing["feature_index"]).astype(str)
    plt.figure(figsize=(8, 4.8))
    plt.barh(top_missing["label"][::-1], top_missing["missing_pct"][::-1] * 100, color="#b35c44")
    plt.xlabel("Pre-imputation missingness (%)")
    plt.ylabel("Feature itemid")
    plt.title("Highest Feature Missingness Before Imputation")
    plt.tight_layout()
    plt.savefig(fig_dir / "full_cohort_missingness.png", dpi=180)
    plt.close()


def styles():
    base = getSampleStyleSheet()
    base["Title"].alignment = TA_CENTER
    base.add(ParagraphStyle(name="Small", parent=base["BodyText"], fontSize=8.5, leading=10.5))
    base.add(ParagraphStyle(name="TableCell", parent=base["Small"], alignment=TA_LEFT))
    base.add(ParagraphStyle(name="CenterSmall", parent=base["Small"], alignment=TA_CENTER))
    return base


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.5 * cm, 1.0 * cm, "Full-cohort MIMIC-IV v3.1 revision")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def para(text: str, style):
    text = text.replace("&", "&amp;")
    return Paragraph(text, style)


def table_from_rows(rows, widths=None):
    data = [[Paragraph(str(cell), styles()["TableCell"]) for cell in row] for row in rows]
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef0")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c4c7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def model_rows(models: pd.DataFrame):
    cols = ["model", "auroc", "auroc_ci_lo", "auroc_ci_hi", "f1_macro", "auprc", "accuracy", "mcc"]
    rows = [["Model", "AUROC", "95% CI", "F1", "AUPRC", "Accuracy", "MCC"]]
    for _, r in models.sort_values("auroc", ascending=False).iterrows():
        ci = "NA"
        if "auroc_ci_lo" in r and "auroc_ci_hi" in r and not pd.isna(r["auroc_ci_lo"]):
            ci = f"{r['auroc_ci_lo']:.3f}-{r['auroc_ci_hi']:.3f}"
        rows.append(
            [
                r["model"],
                fmt(r.get("auroc")),
                ci,
                fmt(r.get("f1_macro")),
                fmt(r.get("auprc")),
                fmt(r.get("accuracy")),
                fmt(r.get("mcc")),
            ]
        )
    return rows


def status_rows(run: dict):
    rows = [["Model/component", "Status", "Details"]]
    for item in run.get("statuses", []):
        detail_parts = []
        for key in ["seconds", "episodes", "nodes_used", "reason", "error"]:
            if key in item:
                detail_parts.append(f"{key}: {item[key]}")
        rows.append([item.get("model", ""), item.get("status", ""), "; ".join(detail_parts)])
    return rows


def build_pdf(path: Path, title: str, subtitle: str, sections: list[tuple[str, list]], appendices=None):
    st = styles()
    story = [Paragraph(title, st["Title"]), Spacer(1, 0.2 * cm), Paragraph(subtitle, st["CenterSmall"]), Spacer(1, 0.55 * cm)]
    for heading, content in sections:
        story.append(Paragraph(heading, st["Heading1"]))
        story.extend(content)
        story.append(Spacer(1, 0.25 * cm))
    if appendices:
        story.append(PageBreak())
        for heading, content in appendices:
            story.append(Paragraph(heading, st["Heading1"]))
            story.extend(content)
            story.append(Spacer(1, 0.25 * cm))
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def build_documents(paths, out_dir, validation, models, run, missing, labels, nodes):
    st = styles()
    fig_dir = paths["figures_dir"]
    cohort = validation["cohort"]
    final = validation["final_dataset"]
    tokens = validation["tokens"]
    miss = validation["missingness"]
    best = models.sort_values("auroc", ascending=False).iloc[0]
    all_used = "yes" if validation["all_eligible_data_used"] else "no"
    label_text = ", ".join(f"{k}: {v}" for k, v in final["label_distribution"].items())

    cohort_table = table_from_rows(
        [
            ["Quantity", "Value"],
            ["Eligible adult first ICU stays", f"{cohort['rows']:,}"],
            ["Final modelling rows", f"{final['rows']:,}"],
            ["Unique ICU stays", f"{final['unique_icu_stays']:,}"],
            ["Unique subjects", f"{final['unique_subjects']:,}"],
            ["Token tensor shape", str(tuple(tokens["shape"]))],
            ["All eligible data used", all_used],
        ],
        widths=[8.8 * cm, 6.2 * cm],
    )
    model_table = table_from_rows(model_rows(models), widths=[4.6 * cm, 2.0 * cm, 2.5 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm])
    status_table = table_from_rows(status_rows(run), widths=[5.0 * cm, 2.4 * cm, 8.1 * cm])

    auroc_img = Image(str(fig_dir / "full_cohort_model_auroc.png"), width=15.5 * cm, height=9.0 * cm)
    label_img = Image(str(fig_dir / "full_cohort_label_distribution.png"), width=11.5 * cm, height=8.2 * cm)
    missing_img = Image(str(fig_dir / "full_cohort_missingness.png"), width=15.5 * cm, height=9.0 * cm)

    thesis_sections = [
        (
            "Abstract",
            [
                para(
                    f"This revised thesis summary reports a full-cohort MIMIC-IV v3.1 modelling study after applying the study inclusion and exclusion criteria. The final modelling dataset contains {final['rows']:,} adult first ICU stays, not the previous 2,000-stay benchmark subset. The strongest rerun model was {best['model']} with AUROC {best['auroc']:.3f}.",
                    st["BodyText"],
                )
            ],
        ),
        (
            "Methods",
            [
                para(
                    "The cohort included adult first ICU stays with ICU length of stay at least 4 hours, non-missing discharge location, and an available primary ICD-10 diagnosis. First-24-hour laboratory and vital-sign features were extracted from the full eligible cohort. Missing feature values were quantified before median imputation and tokenization.",
                    st["BodyText"],
                ),
                cohort_table,
            ],
        ),
        (
            "Results",
            [
                para(
                    f"Label distribution was {label_text}. Mean pre-imputation feature missingness was {miss['mean_pre_imputation_missing_pct']:.1%}; post-imputation missing cells in the modelling matrix were {final['post_imputation_missing_cells']}.",
                    st["BodyText"],
                ),
                label_img,
                model_table,
                auroc_img,
            ],
        ),
        (
            "Discussion",
            [
                para(
                    "The revised framing is a full-cohort modelling study. The old 2,000-stay benchmark is not used as evidence for the main full-cohort claims and should only be cited as historical benchmark context.",
                    st["BodyText"],
                )
            ],
        ),
        (
            "Limitations",
            [
                para(
                    "The feasibility outcome remains a proxy derived from discharge disposition, survival flag, and ICU length of stay. Feature extraction uses selected first-24-hour labs and vitals, and unmeasured features are imputed after missingness is recorded. Model results are single-split reruns unless otherwise stated.",
                    st["BodyText"],
                )
            ],
        ),
    ]
    build_pdf(
        out_dir / "thesis_FULL_COHORT_REVISED.pdf",
        "Full-Cohort Revised Thesis",
        "MIMIC-IV v3.1 treatment-feasibility modelling after inclusion/exclusion criteria",
        thesis_sections,
    )

    manuscript_sections = [
        (
            "Abstract",
            [
                para(
                    f"Background: The earlier benchmark used a deterministic 2,000-stay subset. This revision reruns extraction and modelling on all eligible MIMIC-IV v3.1 adult first ICU stays. Methods: We applied the prespecified cohort criteria and trained classical, neural, token-hybrid, stacked, and simulated federated models where stable. Results: The final dataset contained {final['rows']:,} stays and {tokens['shape'][0]:,} token sequences. The best full-cohort AUROC was {best['auroc']:.3f} for {best['model']}. Conclusions: The manuscript is reframed as a full-cohort modelling study; old subset results are historical only.",
                    st["BodyText"],
                )
            ],
        ),
        (
            "Background",
            [
                para(
                    "Full-cohort evaluation is needed to avoid over-interpreting a moderate-size benchmark subset. The current analysis uses the entire eligible cohort after prespecified criteria.",
                    st["BodyText"],
                )
            ],
        ),
        (
            "Methods",
            [
                para(
                    "MIMIC-IV v3.1 hospital and ICU modules were used locally. Raw records are not redistributed. The extraction scripts wrote outputs to data/processed_full_cohort and all model outputs to results/full_cohort.",
                    st["BodyText"],
                ),
                cohort_table,
            ],
        ),
        (
            "Results",
            [model_table, auroc_img],
        ),
        (
            "Discussion",
            [
                para(
                    "Classical tabular models remain strong baselines on this proxy clinical task. Advanced models are reported only when they completed on the full-cohort split, and failures or skips are documented in the supplement and run report.",
                    st["BodyText"],
                )
            ],
        ),
        (
            "Conclusions",
            [
                para(
                    "The revised article is based on full-cohort outputs generated in this run. No 2,000-stay result is reused as a full-cohort result.",
                    st["BodyText"],
                )
            ],
        ),
    ]
    build_pdf(
        out_dir / "manuscript_FULL_COHORT.pdf",
        "Full-Cohort MIMIC-IV Treatment-Feasibility Modelling Study",
        "BMC Medical Informatics and Decision Making submission draft",
        manuscript_sections,
    )

    supplement_sections = [
        ("Raw Data Verification", [para(json.dumps(validation["raw_data"], indent=2), st["Small"])]),
        ("Model Completion Status", [status_table]),
        ("Missingness", [missing_img]),
        ("Federated Node Distribution", [table_from_rows([nodes.columns.tolist()] + nodes.astype(str).values.tolist())]),
    ]
    build_pdf(
        out_dir / "supplementary_material_FULL_COHORT.pdf",
        "Supplementary Material: Full-Cohort Revision",
        "Validation, missingness, node splits, and model status",
        supplement_sections,
    )

    cover_sections = [
        (
            "Cover Letter",
            [
                para("Dear Editors,", st["BodyText"]),
                para(
                    f"Please consider the revised manuscript, now rerun on the full eligible MIMIC-IV v3.1 cohort of {final['rows']:,} adult first ICU stays after prespecified inclusion and exclusion criteria. The previous 2,000-stay benchmark is no longer used for the main findings.",
                    st["BodyText"],
                ),
                para(
                    "All reported performance values in the manuscript, supplement, and run report are read from the full-cohort output files under results/full_cohort. The package includes validation summaries, model completion status, figures, tables, and author review notes.",
                    st["BodyText"],
                ),
                para("Sincerely,<br/>Mohammad Sharafi", st["BodyText"]),
            ],
        )
    ]
    build_pdf(
        out_dir / "cover_letter_FULL_COHORT.pdf",
        "Cover Letter",
        "Full-cohort MIMIC-IV v3.1 revision",
        cover_sections,
    )


def write_notes_and_report(paths, out_dir, validation, models, run):
    final = validation["final_dataset"]
    tokens = validation["tokens"]
    best = models.sort_values("auroc", ascending=False).iloc[0]
    statuses = run.get("statuses", [])
    checksum_log = paths["results_dir"] / "logs" / "raw_sha256_check.log"
    checksum_ok = False
    if checksum_log.exists():
        lines = [line.strip() for line in checksum_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        checksum_ok = bool(lines) and all(line.endswith(": OK") for line in lines)
    completed = [s["model"] for s in statuses if s.get("status") == "completed"]
    failed = [s for s in statuses if s.get("status") == "failed"]
    skipped = [s for s in statuses if s.get("status") == "skipped"]
    label_text = ", ".join(f"{k}: {v}" for k, v in final["label_distribution"].items())

    report = f"""# Full Cohort Run Report

## Data Use

- All eligible data used: {validation['all_eligible_data_used']}
- Sampling cap: {validation['config']['max_cohort_sample']}
- Raw MIMIC-IV v3.1 manifest complete by file presence: {validation['raw_data'].get('complete_by_manifest', False)}
- Raw SHA-256 manifest check passed: {checksum_ok}
- Final dataset shape: {final['rows']:,} rows x {final['columns']:,} columns
- Unique ICU stays: {final['unique_icu_stays']:,}
- Unique subjects: {final['unique_subjects']:,}
- Label distribution: {label_text}
- Token tensor shape: {tuple(tokens['shape'])}
- Mean pre-imputation missingness: {validation['missingness']['mean_pre_imputation_missing_pct']:.3%}
- Post-imputation missing cells: {final['post_imputation_missing_cells']}

## Models Rerun On Full Cohort

{chr(10).join(f'- {name}' for name in completed) if completed else '- None completed'}

## Failed Or Skipped Models

{chr(10).join(f"- {s.get('model')}: {s.get('status')} ({s.get('error') or s.get('reason')})" for s in failed + skipped) if failed or skipped else '- None'}

## Updated Results

Best full-cohort model: {best['model']} with AUROC {best['auroc']:.3f}, F1 {best['f1_macro']:.3f}, AUPRC {best['auprc']:.3f}, accuracy {best['accuracy']:.3f}, MCC {best['mcc']:.3f}.

Full table: `results/full_cohort/tables/full_cohort_model_results.csv`.

## Remaining Limitations

- The treatment-feasibility label is a proxy based on discharge disposition, survival flag, and ICU length of stay.
- Feature extraction uses selected first-24-hour laboratory and vital-sign variables; missingness is measured before imputation.
- Results are local full-cohort reruns on the available MIMIC-IV v3.1 copy and should be externally validated before clinical use.
- Advanced methods are interpreted only when their full-cohort run completed successfully.

## Thesis And Paper Basis

The revised thesis and manuscript PDFs generated in `papers/full_cohort_revision/` are based on full-cohort outputs. The old 2,000-stay benchmark is historical context only and is not reused as a full-cohort result.
"""
    report_path = out_dir / "FULL_COHORT_RUN_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    notes = f"""# Author Review Notes: Full-Cohort Revision

- `symptoms.max_cohort_sample` is `null`; no 2,000-stay extraction cap is active for the revised run.
- Raw MIMIC-IV v3.1 SHA-256 manifest check passed: {checksum_ok}.
- Processed outputs are under `data/processed_full_cohort/`.
- Result outputs are under `results/full_cohort/`.
- Final modelling rows: {final['rows']:,}.
- All reported full-cohort model metrics are sourced from `results/full_cohort/tables/full_cohort_model_results.csv`.
- The previous 2,000-stay benchmark package remains historical and should not be cited as the primary result set.
"""
    (out_dir / "AUTHOR_REVIEW_NOTES.md").write_text(notes, encoding="utf-8")

    readme = f"""# Full-Cohort Revision Package

This directory contains the full-cohort revision artifacts generated from MIMIC-IV v3.1 after the study inclusion/exclusion criteria.

Primary files:

- `thesis_FULL_COHORT_REVISED.pdf`
- `manuscript_FULL_COHORT.pdf`
- `supplementary_material_FULL_COHORT.pdf`
- `cover_letter_FULL_COHORT.pdf`
- `FULL_COHORT_RUN_REPORT.md`
- `bmc_medical_informatics_FULL_COHORT_SUBMISSION_PACKAGE.zip`

The main results use {final['rows']:,} eligible ICU stays. The old 2,000-stay benchmark is not used for the primary claims.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(report_path, ROOT / "FULL_COHORT_RUN_REPORT.md")


def zip_package(out_dir: Path, paths: dict[str, Path]):
    zip_path = out_dir / "bmc_medical_informatics_FULL_COHORT_SUBMISSION_PACKAGE.zip"
    if zip_path.exists():
        zip_path.unlink()
    include = [
        out_dir / "manuscript_FULL_COHORT.pdf",
        out_dir / "supplementary_material_FULL_COHORT.pdf",
        out_dir / "cover_letter_FULL_COHORT.pdf",
        out_dir / "thesis_FULL_COHORT_REVISED.pdf",
        out_dir / "FULL_COHORT_RUN_REPORT.md",
        out_dir / "AUTHOR_REVIEW_NOTES.md",
        out_dir / "README.md",
        paths["tables_dir"] / "full_cohort_model_results.csv",
        paths["tables_dir"] / "full_cohort_label_distribution.csv",
        paths["tables_dir"] / "full_cohort_missingness.csv",
        paths["results_dir"] / "full_cohort_validation.json",
        paths["results_dir"] / "full_cohort_model_run.json",
    ]
    package_dir = out_dir / "_submission_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    for src in include:
        if src.exists():
            shutil.copy2(src, package_dir / src.name)
    fig_dest = package_dir / "figures"
    fig_dest.mkdir()
    for fig in paths["figures_dir"].glob("full_cohort_*.png"):
        shutil.copy2(fig, fig_dest / fig.name)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", package_dir)
    shutil.rmtree(package_dir)


def main() -> int:
    cfg = load_config()
    paths = resolve_paths(cfg)
    out_dir = ROOT / "papers" / "full_cohort_revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    validation, models, run, missing, labels, nodes = load_outputs(paths)
    make_figures(paths, models, missing, labels)
    build_documents(paths, out_dir, validation, models, run, missing, labels, nodes)
    write_notes_and_report(paths, out_dir, validation, models, run)
    zip_package(out_dir, paths)
    for name in [
        "thesis_FULL_COHORT_REVISED.pdf",
        "manuscript_FULL_COHORT.pdf",
        "supplementary_material_FULL_COHORT.pdf",
        "cover_letter_FULL_COHORT.pdf",
        "bmc_medical_informatics_FULL_COHORT_SUBMISSION_PACKAGE.zip",
        "FULL_COHORT_RUN_REPORT.md",
    ]:
        print(out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
