#!/usr/bin/env python3
"""Generate final optimized thesis/manuscript deliverables from verified outputs."""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "final_optimized"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
THESIS_DIR = ROOT / "papers" / "final_optimized_thesis"
MANUSCRIPT_DIR = ROOT / "papers" / "final_optimized_manuscript"


def esc(text) -> str:
    s = str(text)
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


def fmt(x, digits=3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def ensure_dirs() -> None:
    for d in [FIGURES, THESIS_DIR, MANUSCRIPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def plot_bar(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    top = df.head(10).iloc[::-1]
    plt.figure(figsize=(9, 5.2))
    colors = ["#3b6ea8" if "Ensemble" not in m and "Stacked" not in m else "#2a9d8f" for m in top["model"]]
    plt.barh(top["model"] + " (" + top["feature_set"] + ")", top[metric], color=colors)
    plt.xlabel(metric)
    plt.title(title)
    plt.xlim(max(0.55, top[metric].min() - 0.02), min(0.9, top[metric].max() + 0.02))
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_label_distribution(validation: dict) -> None:
    labels = validation["final_dataset"]["label_distribution"]
    plt.figure(figsize=(5.5, 4))
    plt.bar(["Infeasible", "Feasible"], [labels["0"], labels["1"]], color=["#9b5de5", "#00a896"])
    plt.ylabel("Count")
    plt.title("ICU Discharge-Feasibility Proxy Labels")
    plt.tight_layout()
    plt.savefig(FIGURES / "label_distribution.png", dpi=300)
    plt.close()


def plot_missingness() -> None:
    miss = pd.read_csv(TABLES / "full_cohort_missingness.csv")
    plt.figure(figsize=(8, 4.5))
    plt.bar(miss["itemid"].astype(str), miss["missing_pct"], color="#577590")
    plt.ylabel("Missing proportion")
    plt.xlabel("MIMIC itemid")
    plt.title("First-24h Feature Missingness")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "missingness.png", dpi=300)
    plt.close()


def plot_curves(final: pd.DataFrame) -> None:
    top_ids = final.head(4)["experiment_id"].tolist()
    plt.figure(figsize=(6, 5))
    for exp_id in top_ids:
        p = RESULTS / "calibration" / f"{exp_id}_roc_curve.csv"
        if p.exists():
            c = pd.read_csv(p)
            label = final.loc[final["experiment_id"] == exp_id, "model"].iloc[0]
            plt.plot(c["fpr"], c["tpr"], label=label)
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curves For Top Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curves_top_models.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    for exp_id in top_ids:
        p = RESULTS / "calibration" / f"{exp_id}_pr_curve.csv"
        if p.exists():
            c = pd.read_csv(p)
            label = final.loc[final["experiment_id"] == exp_id, "model"].iloc[0]
            plt.plot(c["recall"], c["precision"], label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves For Top Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "pr_curves_top_models.png", dpi=300)
    plt.close()


def plot_framework_diagram() -> None:
    plt.figure(figsize=(10, 3.8))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        ("MIMIC-IV v3.1", 0.05),
        ("Adult first ICU stays", 0.22),
        ("First-24h labs/vitals", 0.41),
        ("Tabular + symptom tokens", 0.61),
        ("Validation-first models", 0.81),
    ]
    for text, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.38), 0.14, 0.24, fill=False, linewidth=1.8, edgecolor="#264653"))
        ax.text(x + 0.07, 0.5, text, ha="center", va="center", fontsize=9)
    for _, x in boxes[:-1]:
        ax.annotate("", xy=(x + 0.17, 0.5), xytext=(x + 0.14, 0.5), arrowprops=dict(arrowstyle="->", color="#264653", lw=1.8))
    plt.tight_layout()
    plt.savefig(FIGURES / "data_pipeline_framework.png", dpi=300)
    plt.close()


def make_figures(final: pd.DataFrame, validation: dict) -> None:
    plot_bar(final, "AUROC", FIGURES / "auroc_comparison.png", "Final Test AUROC Comparison")
    plot_bar(final, "AUPRC", FIGURES / "auprc_comparison.png", "Final Test AUPRC Comparison")
    plot_label_distribution(validation)
    plot_missingness()
    plot_curves(final)
    plot_framework_diagram()


def latex_table(df: pd.DataFrame, cols: list[str], headers: list[str], caption: str, label: str, n: int = 12) -> str:
    colspec = "p{0.28\\textwidth}" + "r" * (len(headers) - 1)
    lines = [
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        " & ".join(headers) + r"\\ \hline",
        r"\endfirsthead",
        " & ".join(headers) + r"\\ \hline",
        r"\endhead",
    ]
    for _, row in df.head(n).iterrows():
        vals = []
        for c in cols:
            if c == "model":
                vals.append(r"\lr{" + esc(row[c]) + "}")
            elif c == "feature_set":
                vals.append(r"\lr{" + esc(row[c]) + "}")
            else:
                vals.append(r"\lr{" + fmt(row[c]) + "}")
        lines.append(" & ".join(vals) + r"\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def thesis_tex(final: pd.DataFrame, baseline: pd.DataFrame, fewshot: pd.DataFrame, validation: dict) -> str:
    best = final.iloc[0]
    best_token = final[final["feature_set"].astype(str).str.contains("token", na=False)].iloc[0]
    best_fed = final[final["model"].astype(str).str.contains("Federated", na=False)].iloc[0]
    best_fs = final[final["model"].astype(str).str.contains("Few-Shot", na=False)].iloc[0]
    label_dist = validation["final_dataset"]["label_distribution"]
    return rf"""
\documentclass[12pt,a4paper]{{report}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{setspace}}
\usepackage{{xepersian}}
\settextfont{{Vazirmatn}}
\setlatintextfont{{Times New Roman}}
\onehalfspacing
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}}
\title{{فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده}}
\author{{گزارش نهایی بازسازی و بهینه‌سازی تمام‌کوهورت}}
\date{{خرداد ۱۴۰۵}}
\begin{{document}}
\maketitle
\pagenumbering{{roman}}
\chapter*{{چکیده فارسی}}
این پایان‌نامه یک مطالعه یادگیری ماشین بالینی گذشته‌نگر بر پایه پایگاه \lr{{MIMIC-IV v3.1}} است. عنوان اولیه پژوهش بر «قابلیت درمان» تاکید داشت، اما در این بازسازی علمی، خروجی به‌صورت دقیق‌تر به‌عنوان «نمایه جانشین قابلیت ترخیص از ICU» تعریف شد. این نمایه از بقای داخل بیمارستان، مقصد ترخیص مطلوب، و طول اقامت ICU کمتر از \lr{{720}} ساعت ساخته شد و به‌هیچ‌وجه معادل آمادگی بالینی ترخیص یا توصیه درمانی خودکار نیست.

کوهورت نهایی شامل \lr{{{validation['final_dataset']['rows']:,}}} بستری اول ICU بزرگسالان بود. برچسب مثبت در \lr{{{label_dist['1']:,}}} نمونه و برچسب منفی در \lr{{{label_dist['0']:,}}} نمونه مشاهده شد. بهترین مدل نهایی از نظر امتیاز کلی، \lr{{{esc(best['model'])}}} با مجموعه ویژگی \lr{{{esc(best['feature_set'])}}} بود و در آزمون نهایی به \lr{{AUROC={fmt(best['AUROC'])}}}، \lr{{AUPRC={fmt(best['AUPRC'])}}}، \lr{{Brier={fmt(best['Brier'])}}} و \lr{{ECE={fmt(best['ECE'])}}} رسید. مدل‌های فیوشات و توکن‌محور خالص از مدل‌های درختی جدولی ضعیف‌تر بودند، اما مدل‌های هیبریدی و تجمیعی نشان دادند که توکن‌های علائم می‌توانند در ترکیب با ویژگی‌های جدولی سهم محدودی اما قابل گزارش داشته باشند.

\textbf{{کلیدواژه‌ها:}} یادگیری چندنمونه‌ای، توکن‌های علائم، \lr{{MIMIC-IV}}، یادگیری ماشین بالینی، کالیبراسیون، یادگیری فدرال شبیه‌سازی‌شده.

\chapter*{{English Abstract}}
This thesis reports a full-cohort retrospective clinical machine learning study using \lr{{MIMIC-IV v3.1}}. The original title is preserved, but ``treatment feasibility'' is operationalized honestly as an ICU discharge-feasibility proxy based on in-hospital survival, favourable discharge destination, and ICU length of stay below 720 hours. The study does not claim readiness adjudicated by clinicians, treatment recommendation, deployment readiness, or autonomous decision-making.

The final cohort contained \lr{{{validation['final_dataset']['rows']:,}}} adult first ICU stays. The best overall model was \lr{{{esc(best['model'])}}} using \lr{{{esc(best['feature_set'])}}} features with test \lr{{AUROC={fmt(best['AUROC'])}}}, \lr{{AUPRC={fmt(best['AUPRC'])}}}, \lr{{Brier={fmt(best['Brier'])}}}, and \lr{{ECE={fmt(best['ECE'])}}}. Pure few-shot/token models remained weaker than calibrated tree-based and ensemble models, while hybrid token-tabular models were competitive.

\textbf{{Keywords:}} Few-Shot Learning, Symptom Tokens, MIMIC-IV, Clinical Machine Learning, Calibration, Simulated Federated Learning.

\tableofcontents
\listoftables
\listoffigures
\clearpage
\pagenumbering{{arabic}}

\chapter{{مقدمه}}
پیش‌بینی زودهنگام وضعیت بیماران ICU از روی اطلاعات ۲۴ ساعت نخست می‌تواند برای پژوهش‌های تصمیم‌یار بالینی ارزشمند باشد. با این حال، هرگونه ادعای مستقیم درباره امکان درمان یا آمادگی ترخیص نیازمند داوری بالینی آینده‌نگر است. بنابراین، این پایان‌نامه مسئله را به‌صورت یک پیش‌بینی گذشته‌نگر و قابل بازتولید تعریف می‌کند: آیا داده‌های آزمایشگاهی و علائم حیاتی ۲۴ ساعت نخست می‌توانند یک نمایه جانشین قابلیت ترخیص از ICU را پیش‌بینی کنند؟

پرسش‌های پژوهش شامل مقایسه مدل‌های کلاسیک جدولی با یادگیری چندنمونه‌ای \lr{{(Few-Shot Learning)}}، ارزیابی توکن‌های علائم \lr{{(Symptom Tokens)}}، بررسی مدل‌های هیبریدی و تجمیعی، شبیه‌سازی یادگیری فدرال \lr{{(Federated Learning)}}، و انتخاب مدل با توجه همزمان به تمایز و کالیبراسیون بود.

\chapter{{پیشینه و چارچوب نظری}}
داده‌های پرونده الکترونیک سلامت، به‌ویژه در ICU، ناهمگن، ناقص و شدیدا وابسته به زمان هستند. مدل‌های تقویت گرادیانی مانند \lr{{XGBoost}}، \lr{{LightGBM}} و \lr{{CatBoost}} معمولا روی داده‌های جدولی بالینی عملکرد نیرومندی دارند. در برابر آن‌ها، روش‌های یادگیری چندنمونه‌ای و نمایش توکنی علائم تلاش می‌کنند الگوهای فشرده‌تری از وضعیت بیمار بسازند. در این پژوهش، ارزش علمی ETHOS و توکن‌های علائم نه به‌عنوان ادعای برتری قطعی، بلکه به‌عنوان یک فرضیه تجربی آزمون‌پذیر بررسی شد.

کالیبراسیون نیز بخش اصلی ارزیابی است. مدلی که \lr{{AUROC}} بالاتری دارد، اگر احتمال‌های بدکالیبره تولید کند، برای تفسیر بالینی کمتر قابل دفاع است. بنابراین \lr{{Brier Score}}، \lr{{Expected Calibration Error (ECE)}}، شیب و عرض از مبدا کالیبراسیون در کنار \lr{{AUROC}} و \lr{{AUPRC}} گزارش شدند.

\chapter{{روش‌شناسی}}
داده خام از مسیر \lr{{/Users/moe/Programming/Thesis-curser/mimic-iv-3.1}} خوانده شد. فایل‌های اصلی بیمارستانی و ICU با manifest کامل بررسی شدند. معیارهای ورود شامل سن حداقل ۱۸ سال، بستری اول ICU، طول اقامت حداقل ۴ ساعت، وجود تشخیص اصلی ICD-10، و مقصد ترخیص غیرگمشده بود.

ویژگی‌ها از آزمایش‌ها و علائم حیاتی ۲۴ ساعت نخست پس از ورود به ICU ساخته شدند. استخراج با \lr{{DuckDB}} انجام شد تا فیلتر شناسه‌های واجد شرایط، محدودیت ۲۴ ساعته، انتخاب \lr{{itemid}}های مورد نیاز، و تجمیع میانه پیش از ورود داده به \lr{{pandas}} انجام شود. مقادیر خام کلیپ‌شده ذخیره شدند و نرمال‌سازی، جایگذاری مقدارهای گمشده، و شاخص‌های گمشده بودن در مرحله آزمایش و فقط با آمار بخش آموزش انجام شد.

خروجی به صورت زیر تعریف شد:
\[
\text{{feasible}} = 1
\]
اگر بیمار در بیمارستان زنده بماند، مقصد ترخیص مطلوب داشته باشد، و طول اقامت ICU کمتر از \lr{{720}} ساعت باشد. این تعریف یک نمایه جانشین گذشته‌نگر است.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../../results/final_optimized/figures/data_pipeline_framework.png}}
\caption{{چارچوب داده، نمایش ویژگی و ارزیابی مدل‌ها.}}
\end{{figure}}

\chapter{{آزمایش‌ها و نتایج}}
کوهورت نهایی \lr{{{validation['final_dataset']['rows']:,}}} ردیف، \lr{{{validation['final_dataset']['unique_icu_stays']:,}}} ICU stay یکتا، و \lr{{{validation['final_dataset']['unique_subjects']:,}}} بیمار یکتا داشت. تعداد ویژگی‌های جدولی خام \lr{{{validation['final_dataset']['feature_columns']}}} بود و تانسور توکن‌ها شکل \lr{{({validation['tokens']['shape'][0]}, {validation['tokens']['shape'][1]}, {validation['tokens']['shape'][2]})}} داشت.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.62\textwidth]{{../../results/final_optimized/figures/label_distribution.png}}
\caption{{توزیع برچسب نمایه جانشین قابلیت ترخیص از ICU.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.86\textwidth]{{../../results/final_optimized/figures/missingness.png}}
\caption{{گمشده‌بودن ویژگی‌های آزمایشگاهی و علائم حیاتی در ۲۴ ساعت نخست.}}
\end{{figure}}

{latex_table(baseline, ['model','AUROC','AUPRC','accuracy','recall','specificity','F1','Brier'], ['مدل','AUROC','AUPRC','Acc','Recall','Spec','F1','Brier'], 'بازاجرای مدل‌های پایه روی کوهورت نهایی.', 'tab:baseline', 6)}

{latex_table(final, ['model','AUROC','AUPRC','accuracy','recall','specificity','F1','Brier'], ['مدل','AUROC','AUPRC','Acc','Recall','Spec','F1','Brier'], 'مقایسه نهایی مدل‌های منتخب آزمون‌شده.', 'tab:final', 12)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{../../results/final_optimized/figures/auroc_comparison.png}}
\caption{{مقایسه \lr{{AUROC}} در مدل‌های نهایی منتخب.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{../../results/final_optimized/figures/auprc_comparison.png}}
\caption{{مقایسه \lr{{AUPRC}} در مدل‌های نهایی منتخب.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{../../results/final_optimized/figures/roc_curves_top_models.png}}
\caption{{منحنی‌های ROC برای مدل‌های برتر.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{../../results/final_optimized/figures/pr_curves_top_models.png}}
\caption{{منحنی‌های Precision-Recall برای مدل‌های برتر.}}
\end{{figure}}

بهترین نتیجه نهایی از نظر امتیاز کلی مربوط به \lr{{{esc(best['model'])}}} بود. مدل \lr{{{esc(best_token['model'])}}} با ویژگی \lr{{{esc(best_token['feature_set'])}}} بهترین مدل حاوی توکن بود. مدل شبیه‌سازی فدرال \lr{{AUROC={fmt(best_fed['AUROC'])}}} و \lr{{AUPRC={fmt(best_fed['AUPRC'])}}} داشت و نسبت به مدل‌های مرکزی درختی عقب‌تر بود. بهترین مدل فیوشات/توکن پروتوتایپی نیز با \lr{{AUROC={fmt(best_fs['AUROC'])}}} ضعیف‌تر از مدل‌های جدولی بود.

{latex_table(fewshot.rename(columns={'test_AUROC':'AUROC','test_AUPRC':'AUPRC','test_accuracy':'accuracy','test_recall':'recall','test_specificity':'specificity','test_F1':'F1','test_Brier':'Brier'}), ['model','AUROC','AUPRC','accuracy','recall','specificity','F1','Brier'], ['مدل','AUROC','AUPRC','Acc','Recall','Spec','F1','Brier'], 'نتایج فیوشات/توکن پروتوتایپی.', 'tab:fewshot', 7)}

\chapter{{بحث}}
نتایج نشان می‌دهد که ویژگی‌های آزمایشگاهی و علائم حیاتی ۲۴ ساعت نخست برای پیش‌بینی نمایه جانشین قابلیت ترخیص از ICU دارای سیگنال معنادار هستند. با این حال، عملکرد نهایی در محدوده متوسط باقی ماند و نباید به‌عنوان آمادگی برای استقرار بالینی تعبیر شود.

مدل‌های درختی و تجمیعی بهترین توازن تمایز و کالیبراسیون را فراهم کردند. مدل تجمیعی وزن‌دار بالاترین \lr{{AUROC}} و \lr{{AUPRC}} آزمون را داشت، در حالی که مدل‌های کالیبره‌شده CatBoost و stacking نیز از نظر \lr{{Brier}} و \lr{{ECE}} قابل دفاع بودند. روش‌های فیوشات خالص ضعیف باقی ماندند؛ این شکست نسبی پنهان نشده و به‌عنوان یافته‌ای مهم گزارش می‌شود: نمایش توکنی ساده از ویژگی‌های تجمیع‌شده، به‌تنهایی جایگزین مدل‌های جدولی نیرومند نیست.

پاسخ به پرسش‌های پژوهش چنین است: ۱) بله، داده‌های ۲۴ ساعت نخست توان پیش‌بینی متوسط دارند. ۲) مدل‌های کلاسیک و تقویت گرادیانی از فیوشات خالص بهتر بودند. ۳) توکن‌ها در حالت خالص برتری نداشتند، اما در مدل‌های هیبریدی/تجمیعی به نتایج رقابتی کمک کردند. ۴) یادگیری فدرال شبیه‌سازی‌شده عملکردی نزدیک به رگرسیون لجستیک مرکزی داشت، اما به مدل‌های درختی نرسید. ۵) بهترین توازن کلی با مدل‌های ensemble و CatBoost کالیبره حاصل شد. ۶) بهبود فیوشات با پروتوتایپ‌های بزرگ‌تر محدود بود و همچنان ضعیف‌تر از مدل‌های جدولی باقی ماند.

\chapter{{محدودیت‌ها}}
این مطالعه گذشته‌نگر و مبتنی بر یک پایگاه داده است. خروجی یک proxy است و نه آمادگی ترخیص داوری‌شده توسط پزشک. اعتبارسنجی خارجی، اعتبارسنجی آینده‌نگر، و ارزیابی اثر بالینی انجام نشده است. ویژگی‌ها فقط از ۲۴ ساعت نخست ساخته شدند. سوگیری اندازه‌گیری، سوگیری کدنویسی، و مخدوش‌گری باقیمانده ممکن است وجود داشته باشد. یادگیری فدرال صرفا شبیه‌سازی‌شده بود. داده خام \lr{{MIMIC-IV}} قابل بازتوزیع نیست.

\chapter{{نتیجه‌گیری}}
این بازسازی نشان داد که یک pipeline تمام‌کوهورت، قابل ردیابی و کالیبراسیون‌محور برای پیش‌بینی نمایه جانشین قابلیت ترخیص از ICU امکان‌پذیر است. قوی‌ترین نتیجه از مدل‌های جدولی/تجمیعی حاصل شد و روش‌های فیوشات خالص هنوز نیازمند طراحی نمایش، اپیزودسازی و آموزش بهتر هستند. پایان‌نامه از نظر علمی باید بر صداقت در تعریف outcome، استفاده از کل کوهورت واجد شرایط، و گزارش همزمان موفقیت‌ها و شکست‌ها تاکید کند.

\appendix
\chapter{{پیوست بازتولیدپذیری}}
خروجی‌های اصلی در \lr{{data/processed\_final\_optimized}} و \lr{{results/final\_optimized}} ذخیره شده‌اند. رجیستری آزمایش در \lr{{results/final\_optimized/optimization/experiment\_registry.csv}} قرار دارد. هیچ نتیجه‌ای از benchmark دو هزار نمونه‌ای به‌عنوان نتیجه اصلی استفاده نشده است.

\begin{{thebibliography}}{{9}}
\bibitem{{mimiciv}} Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset.
\bibitem{{xgboost}} Chen T, Guestrin C. XGBoost: A scalable tree boosting system.
\bibitem{{lightgbm}} Ke G, et al. LightGBM: A highly efficient gradient boosting decision tree.
\bibitem{{catboost}} Prokhorenkova L, et al. CatBoost: unbiased boosting with categorical features.
\bibitem{{calibration}} Guo C, et al. On Calibration of Modern Neural Networks.
\end{{thebibliography}}
\end{{document}}
"""


def manuscript_tex(final: pd.DataFrame, validation: dict) -> str:
    best = final.iloc[0]
    return rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{booktabs,longtable,graphicx,hyperref}}
\newcommand{{\lr}}[1]{{#1}}
\title{{Full-Cohort MIMIC-IV Prediction of an ICU Discharge-Feasibility Proxy}}
\author{{Prepared from the final optimized thesis analysis}}
\date{{2026-06-27}}
\begin{{document}}
\maketitle
\begin{{abstract}}
We evaluated classical, gradient-boosted, neural, symptom-token, hybrid, stacked, and simulated federated models for predicting a retrospective ICU discharge-feasibility proxy in MIMIC-IV v3.1. The full eligible adult first-ICU-stay cohort contained {validation['final_dataset']['rows']:,} stays. The best final model was {esc(best['model'])} with test AUROC {fmt(best['AUROC'])}, AUPRC {fmt(best['AUPRC'])}, Brier {fmt(best['Brier'])}, and ECE {fmt(best['ECE'])}. The outcome is a proxy and should not be interpreted as clinical discharge readiness or treatment recommendation.
\end{{abstract}}
\section*{{Key Results}}
{latex_table(final, ['model','AUROC','AUPRC','accuracy','recall','specificity','F1','Brier'], ['Model','AUROC','AUPRC','Acc','Recall','Spec','F1','Brier'], 'Final selected model comparison.', 'tab:journalfinal', 10)}
\section*{{Limitations}}
Retrospective single-database design, proxy outcome, no clinician adjudication, no external/prospective validation, first-24h features only, and simulated rather than deployed federated learning.
\end{{document}}
"""


def supplementary_tex(validation: dict) -> str:
    return rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{booktabs,longtable}}
\title{{Supplementary Material}}
\date{{2026-06-27}}
\begin{{document}}
\maketitle
\section*{{Dataset Validation}}
Rows: {validation['final_dataset']['rows']:,}. Unique ICU stays: {validation['final_dataset']['unique_icu_stays']:,}. Unique subjects: {validation['final_dataset']['unique_subjects']:,}. Feature columns: {validation['final_dataset']['feature_columns']}. Token tensor shape: {tuple(validation['tokens']['shape'])}. Sampling cap: null.
\section*{{Reproducibility}}
The experiment registry is stored at results/final\_optimized/optimization/experiment\_registry.csv. Raw MIMIC-IV data are not redistributed.
\end{{document}}
"""


def cover_letter_tex() -> str:
    return r"""
\documentclass[11pt,a4paper]{letter}
\usepackage[margin=2.5cm]{geometry}
\signature{Corresponding Author}
\address{Biomedical Informatics Research Group}
\begin{document}
\begin{letter}{Editorial Office}
\opening{Dear Editor,}
Please find enclosed a manuscript package derived from a full-cohort MIMIC-IV v3.1 retrospective machine learning study. The study predicts an ICU discharge-feasibility proxy and explicitly avoids claims of treatment recommendation, discharge readiness adjudication, or clinical deployment readiness.

The package includes the manuscript summary, supplementary material, and a reproducibility report. All reported metrics are linked to generated result files and the experiment registry.
\closing{Sincerely,}
\end{letter}
\end{document}
"""


def compile_tex(tex_path: Path) -> Path:
    cmd = ["xelatex", "-interaction=nonstopmode", tex_path.name]
    for _ in range(2):
        subprocess.run(cmd, cwd=tex_path.parent, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return tex_path.with_suffix(".pdf")


def render_preview(pdf_path: Path) -> None:
    out_prefix = pdf_path.parent / (pdf_path.stem + "_preview")
    subprocess.run(["pdftoppm", "-png", "-f", "1", "-l", "1", str(pdf_path), str(out_prefix)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_report(final: pd.DataFrame, baseline: pd.DataFrame, fewshot: pd.DataFrame, validation: dict) -> None:
    best = final.iloc[0]
    best_fs = final[final["model"].astype(str).str.contains("Few-Shot", na=False)].iloc[0]
    best_token = final[final["feature_set"].astype(str).str.contains("token", na=False)].iloc[0]
    best_fed = final[final["model"].astype(str).str.contains("Federated", na=False)].iloc[0]
    lines = [
        "# THESIS FINAL OPTIMIZATION REPORT",
        "",
        "## Raw Dataset Verification",
        f"- Dataset path: `{validation['raw_data']['data_root']}`",
        f"- Manifest complete: `{validation['raw_data'].get('complete_by_manifest')}`",
        f"- Required files present: `{validation['raw_data']['required_files_present']}`",
        "",
        "## Processed Dataset Verification",
        f"- Final rows: {validation['final_dataset']['rows']:,}",
        f"- Unique ICU stays: {validation['final_dataset']['unique_icu_stays']:,}",
        f"- Unique subjects: {validation['final_dataset']['unique_subjects']:,}",
        f"- Feature columns: {validation['final_dataset']['feature_columns']}",
        f"- Label distribution: {validation['final_dataset']['label_distribution']}",
        f"- Token shape: {tuple(validation['tokens']['shape'])}",
        f"- Sampling cap: {validation['config']['max_cohort_sample']}",
        "",
        "## Experiments Attempted",
        "- Baseline reproduction: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Neural Baseline.",
        "- Validation-first optimization: 140 trials across tabular, token-only, and tabular+token feature sets.",
        "- Few-shot/token prototype sweep: K = 1, 2, 4, 8, 16, 32, 64.",
        "- Hybrid and ensemble modelling: CatBoost/XGBoost/LightGBM/RF/MLP variants, validation meta-stacking, weighted average ensemble.",
        "- Simulated federated learning: node-weighted LR ensemble over five diagnosis/node splits.",
        "",
        "## Final Selected Results",
        f"- Best overall: {best['model']} ({best['feature_set']}), AUROC {fmt(best['AUROC'])}, AUPRC {fmt(best['AUPRC'])}, Brier {fmt(best['Brier'])}, ECE {fmt(best['ECE'])}.",
        f"- Best ETHOS/few-shot/token prototype: {best_fs['model']}, AUROC {fmt(best_fs['AUROC'])}, AUPRC {fmt(best_fs['AUPRC'])}.",
        f"- Best token/hybrid model: {best_token['model']} ({best_token['feature_set']}), AUROC {fmt(best_token['AUROC'])}, AUPRC {fmt(best_token['AUPRC'])}.",
        f"- Best simulated federated model: {best_fed['model']}, AUROC {fmt(best_fed['AUROC'])}, AUPRC {fmt(best_fed['AUPRC'])}.",
        "",
        "## Honest Interpretation",
        "- First-24h ICU features predict the proxy outcome with moderate discrimination.",
        "- Few-shot/token-only methods did not outperform tree-based tabular or ensemble models.",
        "- Symptom tokens were most useful as hybrid/ensemble inputs, not as standalone superior predictors.",
        "- Simulated federated LR preserved logistic-regression-level performance but remained below tree-based centralized models.",
        "- Final claims must remain retrospective and proxy-based; no clinical deployment or treatment recommendation claim is supported.",
        "",
        "## Files Produced",
        "- `papers/final_optimized_thesis/thesis_FINAL_OPTIMIZED_FULL_COHORT.pdf`",
        "- `papers/final_optimized_thesis/thesis_FINAL_OPTIMIZED_FULL_COHORT.tex`",
        "- `papers/final_optimized_thesis/thesis_FINAL_OPTIMIZED_FULL_COHORT_source.zip`",
        "- `papers/final_optimized_thesis/THESIS_FINAL_OPTIMIZATION_REPORT.md`",
        "- `papers/final_optimized_manuscript/manuscript_FINAL_OPTIMIZED_FULL_COHORT.pdf`",
        "- `papers/final_optimized_manuscript/supplementary_FINAL_OPTIMIZED_FULL_COHORT.pdf`",
        "- `papers/final_optimized_manuscript/cover_letter_FINAL_OPTIMIZED_FULL_COHORT.pdf`",
        "- `papers/final_optimized_manuscript/bmc_medical_informatics_FINAL_OPTIMIZED_FULL_COHORT_PACKAGE.zip`",
        "",
        "## Remaining Limitations",
        "- Retrospective design, single database, proxy outcome, no clinician adjudication, no external or prospective validation, first-24h features only, residual confounding, measurement/coding bias, and simulated-only federated learning.",
    ]
    report = "\n".join(lines) + "\n"
    (THESIS_DIR / "THESIS_FINAL_OPTIMIZATION_REPORT.md").write_text(report, encoding="utf-8")
    (RESULTS / "THESIS_FINAL_OPTIMIZATION_REPORT.md").write_text(report, encoding="utf-8")


def make_zips() -> None:
    source_zip = THESIS_DIR / "thesis_FINAL_OPTIMIZED_FULL_COHORT_source.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [THESIS_DIR / "thesis_FINAL_OPTIMIZED_FULL_COHORT.tex", THESIS_DIR / "THESIS_FINAL_OPTIMIZATION_REPORT.md"]:
            z.write(p, arcname=p.name)
        for fig in FIGURES.glob("*.png"):
            z.write(fig, arcname=f"figures/{fig.name}")
        for table in TABLES.glob("*.csv"):
            z.write(table, arcname=f"tables/{table.name}")
        z.write(RESULTS / "optimization" / "experiment_registry.csv", arcname="optimization/experiment_registry.csv")

    package_zip = MANUSCRIPT_DIR / "bmc_medical_informatics_FINAL_OPTIMIZED_FULL_COHORT_PACKAGE.zip"
    with zipfile.ZipFile(package_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [
            MANUSCRIPT_DIR / "manuscript_FINAL_OPTIMIZED_FULL_COHORT.pdf",
            MANUSCRIPT_DIR / "supplementary_FINAL_OPTIMIZED_FULL_COHORT.pdf",
            MANUSCRIPT_DIR / "cover_letter_FINAL_OPTIMIZED_FULL_COHORT.pdf",
            THESIS_DIR / "THESIS_FINAL_OPTIMIZATION_REPORT.md",
        ]:
            z.write(p, arcname=p.name)


def main() -> int:
    ensure_dirs()
    validation = json.loads((RESULTS / "full_cohort_validation.json").read_text(encoding="utf-8"))
    final = pd.read_csv(TABLES / "final_model_ranking.csv")
    baseline = pd.read_csv(RESULTS / "reproduced_baselines" / "reproduced_baseline_results.csv")
    fewshot = pd.read_csv(TABLES / "ethos_fewshot_token_results.csv")
    make_figures(final, validation)

    thesis_path = THESIS_DIR / "thesis_FINAL_OPTIMIZED_FULL_COHORT.tex"
    thesis_path.write_text(thesis_tex(final, baseline, fewshot, validation), encoding="utf-8")
    manuscript_path = MANUSCRIPT_DIR / "manuscript_FINAL_OPTIMIZED_FULL_COHORT.tex"
    supplementary_path = MANUSCRIPT_DIR / "supplementary_FINAL_OPTIMIZED_FULL_COHORT.tex"
    cover_path = MANUSCRIPT_DIR / "cover_letter_FINAL_OPTIMIZED_FULL_COHORT.tex"
    manuscript_path.write_text(manuscript_tex(final, validation), encoding="utf-8")
    supplementary_path.write_text(supplementary_tex(validation), encoding="utf-8")
    cover_path.write_text(cover_letter_tex(), encoding="utf-8")

    for tex in [thesis_path, manuscript_path, supplementary_path, cover_path]:
        pdf = compile_tex(tex)
        render_preview(pdf)
    write_report(final, baseline, fewshot, validation)
    make_zips()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
