#!/usr/bin/env python3
"""Generate the full-cohort Persian thesis source and revision artifacts.

The generated thesis uses the verified full-cohort CSV/JSON outputs as the
single source of truth. It intentionally keeps the old 2,000-stay benchmark as
historical context only.
"""
from __future__ import annotations

import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = ROOT / "thesis"
OUT_DIR = ROOT / "papers" / "full_cohort_revision"
CHAPTER_DIR = THESIS_DIR / "chapters_full_cohort"
FRONT_DIR = THESIS_DIR / "frontmatter_fa_full"
RESULTS_DIR = ROOT / "results" / "full_cohort"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
SOURCE_TEX = THESIS_DIR / "thesis_FULL_COHORT_REVISED.tex"
PAPER_TEX = OUT_DIR / "thesis_FULL_COHORT_REVISED.tex"
PAPER_PDF = OUT_DIR / "thesis_FULL_COHORT_REVISED.pdf"
SOURCE_ZIP = OUT_DIR / "thesis_FULL_COHORT_REVISED_source.zip"
REVISION_REPORT = OUT_DIR / "THESIS_REVISION_REPORT.md"


def fmt(x: float | int | str, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def read_inputs():
    validation = json.loads((RESULTS_DIR / "full_cohort_validation.json").read_text(encoding="utf-8"))
    models = pd.read_csv(TABLES_DIR / "full_cohort_model_results.csv")
    comparison = pd.read_csv(TABLES_DIR / "full_vs_2k_benchmark_comparison.csv")
    labels = pd.read_csv(TABLES_DIR / "full_cohort_label_distribution.csv")
    missingness = pd.read_csv(TABLES_DIR / "full_cohort_missingness.csv")
    return validation, models, comparison, labels, missingness


def ensure_extra_references() -> None:
    bib = THESIS_DIR / "references.bib"
    text = bib.read_text(encoding="utf-8")
    entries = {
        "goldberger2000physionet": r"""
@article{goldberger2000physionet,
  title={PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals},
  author={Goldberger, Ary L and Amaral, Luis AN and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
  journal={Circulation},
  volume={101},
  number={23},
  pages={e215--e220},
  year={2000}
}
""",
        "collins2015tripod": r"""
@article{collins2015tripod,
  title={Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis ({TRIPOD}): the {TRIPOD} statement},
  author={Collins, Gary S and Reitsma, Johannes B and Altman, Douglas G and Moons, Karel G M},
  journal={Annals of Internal Medicine},
  volume={162},
  number={1},
  pages={55--63},
  year={2015}
}
""",
        "vanCalster2019calibration": r"""
@article{vanCalster2019calibration,
  title={Calibration: the {Achilles} heel of predictive analytics},
  author={Van Calster, Ben and McLernon, David J and van Smeden, Maarten and Wynants, Laure and Steyerberg, Ewout W},
  journal={BMC Medicine},
  volume={17},
  number={1},
  pages={230},
  year={2019}
}
""",
        "steyerberg2019clinical": r"""
@book{steyerberg2019clinical,
  title={Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating},
  author={Steyerberg, Ewout W},
  edition={2},
  publisher={Springer},
  year={2019}
}
""",
        "wolff2019probast": r"""
@article{wolff2019probast,
  title={{PROBAST}: A tool to assess the risk of bias and applicability of prediction model studies},
  author={Wolff, Robert F and Moons, Karel G M and Riley, Richard D and Whiting, Penny F and Westwood, Marie and Collins, Gary S and Reitsma, Johannes B and Kleijnen, Jos and Mallett, Susan},
  journal={Annals of Internal Medicine},
  volume={170},
  number={1},
  pages={51--58},
  year={2019}
}
""",
    }
    for key, entry in entries.items():
        if key not in text:
            text += "\n" + textwrap.dedent(entry).strip() + "\n"
    bib.write_text(text, encoding="utf-8")


def make_figures(models: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = models.sort_values("auroc", ascending=True)

    def barh(metric: str, title: str, path: str, color: str) -> None:
        plt.figure(figsize=(8.5, 5.2))
        plt.barh(ordered["model"], ordered[metric], color=color)
        plt.xlim(0, 1)
        plt.xlabel(metric.upper())
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / path, dpi=220)
        plt.close()

    barh("auprc", "Full-Cohort Model AUPRC", "full_cohort_model_auprc.png", "#496f9e")

    cal = models.sort_values("calibration_ece_10bin", ascending=False)
    plt.figure(figsize=(8.6, 5.2))
    y = np.arange(len(cal))
    plt.barh(y - 0.18, cal["calibration_ece_10bin"], height=0.36, label="ECE", color="#b0564b")
    plt.barh(y + 0.18, cal["brier_score"], height=0.36, label="Brier", color="#3f7f6f")
    plt.yticks(y, cal["model"])
    plt.xlabel("Lower is better")
    plt.title("Calibration and Probability Error")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "full_cohort_calibration_brier_ece.png", dpi=220)
    plt.close()

    xgb = models[models["model"] == "XGBoost"].iloc[0]
    cm = np.array([[int(xgb["tn"]), int(xgb["fp"])], [int(xgb["fn"]), int(xgb["tp"])]])
    plt.figure(figsize=(5.2, 4.5))
    plt.imshow(cm, cmap="YlGnBu")
    plt.title("XGBoost Test Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "full_cohort_xgboost_confusion_matrix.png", dpi=220)
    plt.close()


def model_table_rows(models: pd.DataFrame) -> str:
    rows = []
    cols = [
        "model",
        "train_rows",
        "validation_rows",
        "test_rows",
        "auroc",
        "auprc",
        "accuracy",
        "sensitivity_recall",
        "specificity",
        "f1_macro",
        "brier_score",
        "calibration_ece_10bin",
    ]
    for _, r in models.sort_values("auroc", ascending=False)[cols].iterrows():
        split = f"{int(r.train_rows)}/{int(r.validation_rows)}/{int(r.test_rows)}"
        rows.append(
            " & ".join(
                [
                    r"{}".format(r["model"]),
                    split,
                    fmt(r.auroc),
                    fmt(r.auprc),
                    fmt(r.accuracy),
                    fmt(r.sensitivity_recall),
                    fmt(r.specificity),
                    fmt(r.f1_macro),
                    fmt(r.brier_score),
                    fmt(r.calibration_ece_10bin),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def calibration_rows(models: pd.DataFrame) -> str:
    rows = []
    cols = ["model", "threshold", "brier_score", "calibration_ece_10bin", "calibration_intercept", "calibration_slope"]
    for _, r in models.sort_values(["brier_score", "calibration_ece_10bin"])[cols].iterrows():
        rows.append(
            " & ".join(
                [
                    r["model"],
                    fmt(r.threshold),
                    fmt(r.brier_score),
                    fmt(r.calibration_ece_10bin),
                    fmt(r.calibration_intercept),
                    fmt(r.calibration_slope),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def comparison_rows(comparison: pd.DataFrame) -> str:
    rows = []
    comp = comparison.sort_values("full_cohort_auroc", ascending=False)
    for _, r in comp.iterrows():
        rows.append(
            " & ".join(
                [
                    r["model"],
                    fmt(r.full_cohort_auroc),
                    fmt(r.benchmark_auroc),
                    fmt(r.delta_auroc),
                    str(r.benchmark_source).replace("_", r"\_"),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def build_driver(validation: dict, models: pd.DataFrame) -> None:
    preamble = (THESIS_DIR / "thesis_fa.tex").read_text(encoding="utf-8").split(r"\begin{document}")[0]
    preamble = preamble.replace(
        r"\graphicspath{{../results/figures/}{./results/figures/}{../results/}{./results/}{./}{../}}",
        r"\graphicspath{{../results/full_cohort/figures/}{../results/full_cohort/}{../results/figures/}{./results/figures/}{../results/}{./results/}{./}{../}}",
    )
    preamble = preamble.replace(
        "pdfsubject={Few-Shot Treatment Feasibility Prediction Using Symptom Tokens from Aggregated Data}",
        "pdfsubject={Full-cohort retrospective ICU discharge-feasibility proxy modelling using MIMIC-IV v3.1}",
    )
    final = validation["final_dataset"]
    best = models.sort_values("auroc", ascending=False).iloc[0]
    xgb = models[models["model"] == "XGBoost"].iloc[0]
    abstract_fa = f"""
\\chapter*{{چکیده}}
\\addcontentsline{{toc}}{{chapter}}{{چکیده}}
این پایان‌نامه یک مطالعۀ گذشته‌نگر یادگیری ماشین بالینی بر پایۀ پایگاه دادۀ \\lr{{MIMIC\\text{{-}}IV v3.1}} است. عنوان پژوهش از مفهوم «فیوشات درمانی» و پیش‌بینی قابلیت درمان با توکن‌های علائم آغاز می‌شود، اما در اجرای نهایی، این مفهوم به‌صورت دقیق و محدود به یک \\emph{{نشانگر جانشین امکان‌پذیری ترخیص از \\lr{{ICU}}}} عملیاتی شده است. این نشانگر برابر یک پیامد دودویی است که از سه شرط استخراج می‌شود: زنده ماندن تا ترخیص بیمارستانی، ترخیص به مقصد مطلوب، و طول اقامت \\lr{{ICU}} کمتر از \\pnum{{720}} ساعت. بنابراین مدل‌ها آمادگی واقعی ترخیص، توصیه درمان، یا قابلیت استقرار بالینی را پیش‌بینی نمی‌کنند؛ بلکه یک پیامد گذشته‌نگر و قابل‌بازتولید را برای پژوهش انفورماتیک پزشکی مدل می‌کنند.

کوهورت نهایی شامل \\textbf{{\\pnum{{{final['rows']:,}}}}} بستری نخست بزرگسال در \\lr{{ICU}} بود. همۀ نمونه‌های واجد شرایط پس از معیارهای ورود و خروج استفاده شدند و سقف نمونه‌گیری برابر \\lr{{null}} بود. توزیع برچسب شامل \\pnum{{{final['label_distribution']['0']:,}}} نمونۀ ناممکن و \\pnum{{{final['label_distribution']['1']:,}}} نمونۀ ممکن بود. از داده‌های آزمایشگاهی و علائم حیاتی در \\pnum{{24}} ساعت نخست بستری، \\pnum{{{final['feature_columns']}}} ویژگی جدولی و تانسور توکن با شکل \\lr{{(32399, 25, 4)}} ساخته شد.

مدل‌های ارزیابی‌شده شامل رگرسیون لجستیک، جنگل تصادفی، \\lr{{XGBoost}}، \\lr{{LightGBM}}، \\lr{{CatBoost}}، خط پایۀ عصبی، \\lr{{Few-Shot ETHOS}}، مدل هیبریدی توکن-جدولی، مدل انباشتی کلاسیک، و شبیه‌سازی فدرال بودند. بهترین \\lr{{AUROC}} مربوط به مدل \\lr{{{best['model']}}} با مقدار \\cnum{{{best['auroc']:.3f}}} بود؛ اما \\lr{{XGBoost}} با \\lr{{AUROC}} \\cnum{{{xgb['auroc']:.3f}}}، \\lr{{AUPRC}} \\cnum{{{xgb['auprc']:.3f}}}، امتیاز بریر \\cnum{{{xgb['brier_score']:.3f}}} و \\lr{{ECE}} \\cnum{{{xgb['calibration_ece_10bin']:.3f}}} از نظر توازن تفکیک و کالیبراسیون، مدل اصلی قابل‌دفاع‌تر برای تفسیر پژوهشی بود. مدل \\lr{{Few-Shot ETHOS}} با \\lr{{AUROC}} \\cnum{{0.590}} از مدل‌های جدولی ضعیف‌تر عمل کرد؛ این نتیجه پنهان نشده و به‌عنوان یافته‌ای مهم درباره محدودیت یادگیری کم‌نمونه خام در داده‌های ساخت‌یافتۀ \\lr{{EHR}} گزارش می‌شود.

نتایج نشان دادند که در این وظیفۀ ساخت‌یافته، مدل‌های درختی و گرادیان‌بوستینگ همچنان قوی‌ترین خط پایه را تشکیل می‌دهند. بااین‌حال، عملکرد نزدیک مدل \\lr{{Token Hybrid RF}} به مدل‌های جدولی نشان می‌دهد که توکن‌های علائم در ترکیب با مدل‌های کلاسیک می‌توانند سیگنال مکمل فراهم کنند. شبیه‌سازی فدرال نیز نشان داد که تجمیع گره‌ای مبتنی بر رگرسیون لجستیک می‌تواند عملکردی نزدیک به رگرسیون لجستیک متمرکز حفظ کند، هرچند به مدل‌های درختی نزدیک نشد. محدودیت‌های اصلی شامل ماهیت گذشته‌نگر تک‌پایگاهی، جانشین بودن برچسب، نبود اعتبارسنجی خارجی یا آینده‌نگر، شبیه‌سازی بودن یادگیری فدرال، و محدودیت‌های بازتوزیع دادۀ \\lr{{MIMIC\\text{{-}}IV}} است.

\\vspace{{1em}}
\\noindent\\textbf{{کلیدواژه‌ها:}}
یادگیری ماشین بالینی؛
\\lr{{MIMIC\\text{{-}}IV}}؛
نشانگر جانشین امکان‌پذیری ترخیص از \\lr{{ICU}}؛
توکن‌های علائم؛
یادگیری چندنمونه‌ای؛
یادگیری فدرال؛
کالیبراسیون.
"""
    abbreviations = r"""
\chapter*{فهرست اختصارات}
\addcontentsline{toc}{chapter}{فهرست اختصارات}
\begin{longtable}{>{\raggedleft\arraybackslash}p{3cm}>{\raggedleft\arraybackslash}p{10cm}}
\toprule
\textbf{اختصار} & \textbf{شرح} \\
\midrule
\lr{ICU} & بخش مراقبت‌های ویژه \\
\lr{EHR} & پرونده الکترونیک سلامت \\
\lr{MIMIC-IV} & پایگاه دادۀ مراقبت ویژه \lr{Medical Information Mart for Intensive Care IV} \\
\lr{AUROC} & سطح زیر منحنی مشخصۀ عملکرد گیرنده \\
\lr{AUPRC} & سطح زیر منحنی دقت-بازخوانی \\
\lr{ECE} & خطای کالیبراسیون مورد انتظار \\
\lr{Brier} & امتیاز بریر برای خطای احتمال پیش‌بینی‌شده \\
\lr{ETHOS} & رمزگذار توکنی علائم استفاده‌شده در چارچوب پژوهش \\
\lr{RF} & جنگل تصادفی \\
\bottomrule
\end{longtable}
\cleardoublepage
"""
    preface = r"""
\chapter*{مقدمه}
\addcontentsline{toc}{chapter}{مقدمه}
این نسخۀ بازنگری‌شده پایان‌نامه بر پایۀ اجرای کامل کوهورت واجد شرایط در \lr{MIMIC\text{-}IV v3.1} نوشته شده است. در نسخه‌های مقدماتی، بخشی از آزمایش‌ها روی زیرمجموعۀ نمونه‌گیری‌شده انجام شده بود؛ اما شواهد اصلی این متن اکنون به کوهورت کامل پس از معیارهای ورود و خروج، شامل \pnum{32,399} بستری نخست بزرگسال در \lr{ICU}، متکی است.

هدف متن حاضر نه دفاع از برتری یک روش خاص، بلکه گزارش صادقانه یک مسیر پژوهشی است: ایدۀ اولیه بررسی یادگیری چندنمونه‌ای و توکن‌های علائم بود؛ اجرای کامل نشان داد مدل‌های جدولی و درختی در این وظیفۀ ساخت‌یافتۀ بالینی همچنان بسیار قوی‌اند. همین نتیجه، ارزش علمی کار را افزایش می‌دهد، زیرا مرز واقع‌بینانۀ روش‌های نوین را در برابر خط پایه‌های قوی نشان می‌دهد.
\cleardoublepage
"""
    body = rf"""
\begin{{document}}
\pagenumbering{{gobble}}
\input{{frontmatter_fa_full/cover}}
\input{{frontmatter_fa/commitment}}
\input{{frontmatter_fa/dedication}}
\cleardoublepage
\ThesisPrelimNumberingStart

\begin{{acknowledgments}}
این پایان‌نامه نتیجۀ مسیری پژوهشی است که از دغدغۀ دسترسی محدود به دادۀ سلامت، حریم خصوصی بیمار، و نیاز به ارزیابی صادقانۀ مدل‌های یادگیری ماشین بالینی آغاز شد. از استاد راهنما، دکتر علی‌محمدزاده، برای راهنمایی و حمایت علمی، و از خانواده‌ام برای همراهی در مسیر طولانی بازنگری، صمیمانه سپاسگزارم.
\end{{acknowledgments}}

\tableofcontents
\listoftables
\listoffigures

{abstract_fa}
\cleardoublepage
{abbreviations}
{preface}

\ThesisPrelimNumberingEnd

\input{{chapters_full_cohort/introduction}}
\input{{chapters_full_cohort/background}}
\input{{chapters_full_cohort/methodology}}
\input{{chapters_full_cohort/experiments_results}}
\input{{chapters_full_cohort/discussion}}
\input{{chapters_full_cohort/conclusion}}

\begin{{appendices}}
\input{{chapters_full_cohort/appendix_reproducibility}}
\end{{appendices}}

\bibliographystyle{{apalike}}
\bibliography{{references}}

\input{{frontmatter_fa_full/titlepage_en}}
\input{{frontmatter_fa_full/abstract_en}}

\end{{document}}
"""
    SOURCE_TEX.write_text(preamble + body, encoding="utf-8")
    PAPER_TEX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_TEX, PAPER_TEX)


def write_frontmatter(validation: dict, models: pd.DataFrame) -> None:
    final = validation["final_dataset"]
    best = models.sort_values("auroc", ascending=False).iloc[0]
    xgb = models[models["model"] == "XGBoost"].iloc[0]
    write_text(
        FRONT_DIR / "cover.tex",
        r"""
% جلد فارسی بازنگری کامل کوهورت
\thispagestyle{empty}
\begin{center}
\vspace*{0.05cm}
\IfFileExists{assets/islamic_azad_university_logo.png}{%
  \includegraphics[height=2.25cm]{assets/islamic_azad_university_logo.png}\\[0.25cm]
}{}
{\Large\bfseries دانشگاه آزاد اسلامی}\\[0.12cm]
{\large واحد تهران شمال}\\[0.2cm]
{\normalsize دانشکدۀ برق و کامپیوتر، گروه مهندسی کامپیوتر}\\[0.75cm]

{\large\bfseries پایان‌نامۀ کارشناسی ارشد}\\[0.25cm]
{\normalsize رشته: مهندسی کامپیوتر}\\[0.1cm]
{\normalsize گرایش: نرم‌افزار}\\[0.65cm]

\textbf{عنوان:}\\[0.25cm]
\rule{0.82\textwidth}{0.8pt}\\[0.35cm]
{\Large\bfseries
فیوشات درمانی:\\[0.18cm]
پیش‌بینی هوشمند قابلیت درمان\\[0.15cm]
با توکن‌های علائم از داده‌های تجمیع‌شده
}\\[0.35cm]
{\normalsize\bfseries
مطالعۀ گذشته‌نگر تمام‌کوهورت بر نشانگر جانشین امکان‌پذیری ترخیص از \lr{ICU} در \lr{MIMIC\text{-}IV}
}\\[0.35cm]
\rule{0.82\textwidth}{0.8pt}\\[0.75cm]

\begin{tabular}{rl}
\textbf{استاد راهنما:} & جناب آقای دکتر علی‌محمدزاده \\
\textbf{نگارش:} & محمد شرفی \\
\end{tabular}\\[0.8cm]

{\normalsize اسفند \pnum{1404}}
\end{center}
\vfill\null
\cleardoublepage
""",
    )
    write_text(
        FRONT_DIR / "titlepage_en.tex",
        r"""
\thispagestyle{empty}
\begin{latin}
\begin{center}
\vspace*{0.8cm}
\IfFileExists{assets/islamic_azad_university_logo.png}{%
  \includegraphics[height=2.8cm]{assets/islamic_azad_university_logo.png}\\[0.5cm]
}{}
{\large\bfseries Islamic Azad University, Tehran North Branch}\\[0.25cm]
{\normalsize Faculty of Computer Engineering}\\[1.2cm]
{\Large\bfseries Master's Thesis}\\[0.9cm]
{\LARGE\bfseries Therapeutic Few-Shot: Intelligent Prediction of Treatment Feasibility}\\[0.35cm]
{\LARGE\bfseries with Symptom Tokens from Aggregated Data}\\[0.55cm]
{\large Full-Cohort Retrospective ICU Discharge-Feasibility Proxy Modelling in MIMIC-IV}\\[1.6cm]
{\large Mohammad Sharafi}\\[0.5cm]
{\normalsize Computer Engineering --- Software}\\[0.3cm]
{\normalsize Supervisor: Dr.\ Alimohammadzadeh}\\[2cm]
{\normalsize March 2026}
\end{center}
\end{latin}
\cleardoublepage
""",
    )
    write_text(
        FRONT_DIR / "abstract_en.tex",
        f"""
\\chapter*{{چکیده انگلیسی}}
\\addcontentsline{{toc}}{{chapter}}{{چکیده انگلیسی}}
\\begin{{latin}}
\\small
\\noindent
This thesis reports a retrospective full-cohort clinical machine-learning study using MIMIC-IV v3.1. The thesis title refers to therapeutic few-shot learning and symptom tokens; in the implemented study, this concept is operationalized conservatively as prediction of an ICU discharge-feasibility proxy, defined by in-hospital survival, favourable discharge destination, and ICU length of stay below 720 hours. The task is a reproducible retrospective proxy-modelling study, not a bedside readiness tool or clinical product.

The final modelling dataset included {final['rows']:,} eligible adult first ICU stays, with {final['label_distribution']['0']:,} proxy-negative and {final['label_distribution']['1']:,} proxy-positive cases. The token tensor had shape (32399, 25, 4), and all eligible data after the study inclusion and exclusion criteria were used.

Models evaluated on a fixed train/validation/test split included Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, a neural baseline, Few-Shot ETHOS, a token-hybrid Random Forest, stacked classical modelling, and a simulated federated logistic-regression node ensemble. Stacked Classical achieved the highest AUROC ({best['auroc']:.3f}), while XGBoost achieved a nearly identical AUROC ({xgb['auroc']:.3f}) with stronger Brier score ({xgb['brier_score']:.3f}) and ECE ({xgb['calibration_ece_10bin']:.3f}). Therefore, XGBoost is treated as the most defensible primary model when discrimination and calibration are considered together. Pure Few-Shot ETHOS underperformed the classical tabular models (AUROC 0.590), an important empirical finding in this structured EHR task.

The study contributes a full-cohort MIMIC-IV ICU discharge-feasibility proxy dataset, a symptom-token representation, a comparative benchmark of classical, neural, few-shot, hybrid, stacked, and simulated federated models, and a transparent discussion of where few-shot/token methods helped and where classical tabular models remained stronger. Limitations include the retrospective single-database design, the proxy nature of the outcome, lack of external or prospective validation, simulated rather than deployed federation, and MIMIC-IV data-use restrictions.

\\vspace{{1em}}
\\noindent\\textbf{{Keywords:}} clinical machine learning, MIMIC-IV, ICU discharge-feasibility proxy, symptom tokens, few-shot learning, federated learning, calibration.
\\end{{latin}}
\\cleardoublepage
""",
    )


def write_chapters(validation: dict, models: pd.DataFrame, comparison: pd.DataFrame, missingness: pd.DataFrame) -> None:
    final = validation["final_dataset"]
    tokens = validation["tokens"]
    raw = validation["raw_data"]
    miss = validation["missingness"]
    best = models.sort_values("auroc", ascending=False).iloc[0]
    xgb = models[models["model"] == "XGBoost"].iloc[0]
    lr = models[models["model"] == "Logistic Regression"].iloc[0]
    fed = models[models["model"] == "Federated LR Node Ensemble"].iloc[0]
    few = models[models["model"] == "Few-Shot ETHOS"].iloc[0]
    hybrid = models[models["model"] == "Token Hybrid RF"].iloc[0]

    write_text(
        CHAPTER_DIR / "introduction.tex",
        r"""
\chapter{کلیات تحقیق}
\label{ch:introduction}

\section{بیان مسئله}
بخش مراقبت‌های ویژه (\lr{ICU}) یکی از داده‌محورترین و در عین حال پرخطرترین محیط‌های نظام سلامت است. در ساعات نخست بستری، داده‌های آزمایشگاهی، علائم حیاتی، سن، مسیر پذیرش و تشخیص‌های اولیه به‌تدریج تصویری از وضعیت بیمار می‌سازند. پژوهش‌های یادگیری ماشین می‌کوشند از همین داده‌های اولیه برای تخمین خطر، پیامد، و مسیر احتمالی بیمار استفاده کنند. بااین‌حال، هر مدل بالینی باید با دقت تعریف کند که دقیقاً چه چیزی را پیش‌بینی می‌کند و از چه چیزی نمی‌توان نتیجه گرفت.

این پایان‌نامه از ایدۀ «فیوشات درمانی» آغاز شد: آیا می‌توان با استفاده از توکن‌های علائم و یادگیری چندنمونه‌ای (\lr{Few-Shot Learning}) از داده‌های محدود و حساس بالینی مدلی ساخت که به پیش‌بینی امکان‌پذیری درمان کمک کند؟ در اجرای نهایی و بازنگری‌شده، این ایده به یک مسئلۀ قابل‌بازتولید و محدودتر تبدیل شد: پیش‌بینی \emph{نشانگر جانشین امکان‌پذیری ترخیص از \lr{ICU}} در پایگاه دادۀ \lr{MIMIC\text{-}IV v3.1}. این نشانگر جایگزین قضاوت پزشک، آمادگی واقعی ترخیص، یا توصیه درمان نیست. هدف، ارزیابی پژوهشی مدل‌ها روی یک برچسب گذشته‌نگر و شفاف است.

\section{تعریف عملیاتی امکان‌پذیری}
در این پژوهش، برچسب مثبت زمانی به بیمار داده می‌شود که سه شرط برقرار باشد: بیمار تا ترخیص بیمارستانی زنده مانده باشد؛ مقصد ترخیص در گروه مقصدهای مطلوب تعریف‌شده باشد؛ و طول اقامت در \lr{ICU} کمتر از \pnum{720} ساعت باشد. این تعریف از روی داده‌های ساخت‌یافتۀ \lr{MIMIC\text{-}IV} قابل استخراج است و امکان بازتولید دارد، اما یک برچسب جانشین است. بنابراین، در سراسر پایان‌نامه از اصطلاح «نشانگر جانشین امکان‌پذیری ترخیص از \lr{ICU}» استفاده می‌شود.

\section{انگیزه و اهمیت}
تصمیم‌های مرتبط با مسیر مراقبت و ترخیص در \lr{ICU} پیامدهای مستقیم برای بیمار، تخت‌های مراقبت ویژه، هزینه و برنامه‌ریزی بیمارستان دارند. اگر مدلی بتواند از داده‌های \pnum{24} ساعت نخست سیگنالی قابل‌اعتماد درباره پیامد جانشین تولید کند، می‌تواند در آینده به پژوهش‌های پشتیبان تصمیم، اولویت‌بندی پایش، یا طراحی مطالعات آینده‌نگر کمک کند. اما چنین مدلی تنها زمانی ارزشمند است که با خط پایه‌های قوی مقایسه شود، کالیبراسیون آن بررسی گردد، و محدودیت‌های برچسب و داده آشکارا بیان شود.

انگیزۀ دوم، حریم خصوصی و محدودیت دسترسی به داده‌های سلامت است. پایگاه‌هایی مانند \lr{MIMIC\text{-}IV} با وجود ارزش پژوهشی بالا، طبق مقررات سخت‌گیرانه و پس از احراز صلاحیت در دسترس قرار می‌گیرند \cite{johnson2023mimic,goldberger2000physionet}. در دنیای واقعی، داده‌های بیمارستانی معمولاً میان مراکز قابل‌انتقال نیستند. به همین دلیل، این پایان‌نامه علاوه بر مدل‌های متمرکز، یک شبیه‌سازی یادگیری فدرال (\lr{Federated Learning}) را نیز بررسی می‌کند \cite{mcmahan2017communication,li2019federated}.

\section{شکاف پژوهشی}
مطالعات زیادی نشان داده‌اند که مدل‌های یادگیری ماشین می‌توانند از داده‌های \lr{EHR} برای پیش‌بینی پیامدهای بالینی استفاده کنند \cite{rajkomar2018scalable}. بااین‌حال، سه شکاف در این حوزه باقی است. نخست، بسیاری از مطالعات، خط پایه‌های کلاسیک قوی و مدل‌های نوین را در یک چارچوب یکسان و با گزارش کالیبراسیون مقایسه نمی‌کنند. دوم، یادگیری چندنمونه‌ای در داده‌های جدولی ساخت‌یافته، برخلاف تصویر و زبان، همیشه مزیت روشن ندارد و نیازمند ارزیابی صادقانه است \cite{snell2017prototypical,vinyals2016matching}. سوم، کاربرد بازنمایی‌های توکنی علائم در کنار مدل‌های جدولی و شبیه‌سازی فدرال هنوز نیازمند بنچمارک‌های بازتولیدپذیر است.

\section{پرسش‌های پژوهش}
این پایان‌نامه به پنج پرسش پاسخ می‌دهد:
\begin{enumerate}
\item آیا ویژگی‌های آزمایشگاهی و علائم حیاتی \pnum{24} ساعت نخست \lr{ICU} می‌توانند نشانگر جانشین امکان‌پذیری ترخیص را در \lr{MIMIC\text{-}IV} پیش‌بینی کنند؟
\item مدل‌های کلاسیک و درختی جدولی در مقایسه با مدل \lr{Few-Shot ETHOS} چگونه عمل می‌کنند؟
\item آیا بازنمایی توکن‌های علائم، به‌ویژه در مدل‌های هیبریدی، سیگنال مکمل فراهم می‌کند؟
\item آیا شبیه‌سازی فدرال می‌تواند عملکردی نزدیک به خط پایۀ متمرکز هم‌خانواده حفظ کند؟
\item کدام مدل بهترین توازن میان تفکیک (\lr{AUROC/AUPRC}) و کالیبراسیون (\lr{Brier/ECE}) را دارد؟
\end{enumerate}

\section{مشارکت‌ها}
مشارکت‌های اصلی پژوهش عبارت‌اند از:
\begin{enumerate}
\item ساخت و اعتبارسنجی یک کوهورت کامل \lr{MIMIC\text{-}IV} شامل \pnum{32,399} بستری نخست بزرگسال در \lr{ICU}.
\item تعریف شفاف یک نشانگر جانشین امکان‌پذیری ترخیص از \lr{ICU} و تفکیک آن از آمادگی واقعی ترخیص یا توصیه درمان.
\item ساخت بازنمایی توکن‌های علائم برای داده‌های آزمایشگاهی و علائم حیاتی \pnum{24} ساعت نخست.
\item مقایسۀ مدل‌های رگرسیون لجستیک، جنگل تصادفی، گرادیان‌بوستینگ، خط پایۀ عصبی، \lr{Few-Shot ETHOS}، مدل هیبریدی، مدل انباشتی، و شبیه‌سازی فدرال.
\item نشان دادن اینکه در این وظیفه، مدل‌های درختی و جدولی از \lr{Few-Shot ETHOS} خام قوی‌ترند، در حالی که توکن‌های علائم در حالت هیبریدی می‌توانند به عملکرد نزدیک به مدل‌های قوی برسند.
\item گزارش معیارهای کالیبراسیون در کنار معیارهای تفکیک، و انتخاب \lr{XGBoost} به‌عنوان مدل اصلی قابل‌دفاع‌تر از نظر توازن عملکرد.
\end{enumerate}

\section{ساختار پایان‌نامه}
فصل دوم پیشینۀ علمی را مرور می‌کند. فصل سوم روش‌شناسی، داده، برچسب، ویژگی‌ها، توکن‌ها، مدل‌ها، تقسیم داده و معیارهای ارزیابی را توضیح می‌دهد. فصل چهارم نتایج کامل کوهورت، جداول و شکل‌ها را گزارش می‌کند. فصل پنجم یافته‌ها، محدودیت‌ها و پیامدها را بحث می‌کند و فصل ششم جمع‌بندی نهایی و مسیرهای آینده را ارائه می‌دهد.
""",
    )

    write_text(
        CHAPTER_DIR / "background.tex",
        r"""
\chapter{پیشینه و مبانی نظری}
\label{ch:background}

\section{پرونده الکترونیک سلامت و داده‌های مراقبت ویژه}
پرونده الکترونیک سلامت (\lr{EHR}) شامل مجموعه‌ای ناهمگون از داده‌هاست: آزمایش‌ها، علائم حیاتی، تشخیص‌ها، داروها، اقدامات، یادداشت‌ها و اطلاعات جمعیت‌شناختی. در \lr{ICU}، این داده‌ها با چگالی زمانی بالاتری ثبت می‌شوند و برای پژوهش‌های پیش‌بینی پیامد ارزش ویژه دارند. بااین‌حال، ناهمگونی ثبت، خطاهای اندازه‌گیری، مقادیر گمشده، تغییر الگوهای مراقبت و تفاوت مراکز باعث می‌شود مدل‌سازی این داده‌ها دشوار باشد.

پایگاه \lr{MIMIC\text{-}IV} یکی از مهم‌ترین منابع باز اما کنترل‌شدۀ پژوهش مراقبت ویژه است \cite{johnson2023mimic}. این پایگاه شامل داده‌های بیماران بستری در مرکز \lr{Beth Israel Deaconess Medical Center} است و از طریق \lr{PhysioNet} پس از آموزش و تعهدنامۀ استفاده از داده در دسترس قرار می‌گیرد \cite{goldberger2000physionet}. استفاده از این داده‌ها امکان بازتولید پژوهش را فراهم می‌کند، اما به‌دلیل تک‌مرکزی و گذشته‌نگر بودن، نباید با اعتبارسنجی خارجی یا آمادگی استقرار بالینی اشتباه گرفته شود.

\section{مدل‌های پیش‌بینی بالینی}
مدل پیش‌بینی بالینی باید یک جمعیت هدف، زمان پیش‌بینی، پیامد، ویژگی‌ها و معیارهای ارزیابی روشن داشته باشد. راهنماهایی مانند \lr{TRIPOD} و ابزارهایی مانند \lr{PROBAST} بر شفافیت در تعریف کوهورت، برچسب، اعتبارسنجی و خطر سوگیری تأکید می‌کنند \cite{collins2015tripod,wolff2019probast}. در این پایان‌نامه، زمان پیش‌بینی \pnum{24} ساعت نخست \lr{ICU} و پیامد، نشانگر جانشین ترخیص است.

\section{مدل‌های جدولی کلاسیک و درختی}
داده‌های ساخت‌یافته \lr{EHR} اغلب جدولی‌اند. در چنین داده‌هایی، مدل‌های کلاسیک مانند رگرسیون لجستیک و جنگل تصادفی \cite{breiman2001random} همچنان خط پایه‌های مهمی هستند. گرادیان‌بوستینگ، به‌ویژه \lr{XGBoost} \cite{chen2016xgboost}، \lr{LightGBM} \cite{ke2017lightgbm} و \lr{CatBoost} \cite{prokhorenkova2018catboost}، معمولاً روی داده‌های جدولی عملکرد قوی دارند. این مدل‌ها روابط غیرخطی و تعامل ویژگی‌ها را بدون نیاز به معماری‌های عصبی پیچیده یاد می‌گیرند.

از منظر پزشکی، تنها \lr{AUROC} کافی نیست. مدلی که احتمال‌های بدکالیبره تولید کند، حتی با تفکیک مناسب، برای تفسیر خطر قابل‌اعتماد نیست. به همین دلیل، امتیاز بریر و خطای کالیبراسیون مورد انتظار (\lr{ECE}) نیز باید بررسی شود \cite{steyerberg2019clinical,vanCalster2019calibration}.

\section{یادگیری چندنمونه‌ای}
یادگیری چندنمونه‌ای (\lr{Few-Shot Learning}) به دنبال یادگیری از تعداد اندک نمونۀ برچسب‌دار برای هر کلاس است. شبکه‌های تطبیقی و نمونه‌محور مانند \lr{Matching Networks} و \lr{Prototypical Networks} در حوزه‌هایی مانند تصویر و زبان موفق بوده‌اند \cite{vinyals2016matching,snell2017prototypical}. ایدۀ اصلی این است که مدل به‌جای یادگیری یک مرز تصمیم ثابت برای یک مجموعه بزرگ، بازنمایی‌ای بیاموزد که با چند نمونۀ پشتیبان بتواند نمونه‌های جدید را دسته‌بندی کند.

بااین‌حال، انتقال این منطق به داده‌های جدولی بالینی ساده نیست. ویژگی‌های آزمایشگاهی و علائم حیاتی معمولاً تعداد کمی بعد دارند، معنای بالینی مستقیم دارند و مدل‌های درختی روی آن‌ها قوی‌اند. بنابراین، مزیت یادگیری چندنمونه‌ای باید تجربی و در برابر خط پایه‌های قوی نشان داده شود. در این پایان‌نامه، \lr{Few-Shot ETHOS} به‌عنوان یک مسیر پژوهشی بررسی می‌شود، اما اگر ضعیف‌تر از مدل‌های جدولی باشد، این نتیجه صریحاً گزارش می‌شود.

\section{توکن‌های علائم}
توکن‌های علائم (\lr{Symptom Tokens}) تلاشی برای تبدیل داده‌های ناهمگون آزمایشگاهی و علائم حیاتی به قالبی فشرده و قابل‌پردازش هستند. هر توکن می‌تواند شناسه ویژگی، شدت، جهت تغییر یا وضعیت نرمال/غیرنرمال را رمزگذاری کند. چنین بازنمایی‌ای از نظر مفهومی به مدل‌های توکنی در پردازش زبان نزدیک است \cite{vaswani2017attention}، اما در اینجا بر داده‌های ساخت‌یافته بالینی اعمال می‌شود.

ارزش توکن‌ها ممکن است در دو سطح ظاهر شود. نخست، به‌عنوان ورودی مدل چندنمونه‌ای؛ دوم، به‌عنوان ویژگی مکمل در مدل‌های هیبریدی. نتایج این پایان‌نامه نشان می‌دهد سطح دوم در کوهورت کامل امیدوارکننده‌تر بوده است.

\section{یادگیری فدرال در سلامت}
یادگیری فدرال (\lr{Federated Learning}) اجازه می‌دهد مدل‌ها بدون انتقال مستقیم داده‌های خام میان گره‌ها آموزش ببینند \cite{mcmahan2017communication,yang2019federated}. در سلامت، این ایده با محدودیت‌های حریم خصوصی، مالکیت داده و مقررات سازگار است. بااین‌حال، فدرال بودن به‌تنهایی تضمین‌کنندۀ عملکرد یا ایمنی نیست. ناهمگونی گره‌ها، اندازۀ نمونه، ارتباطات، امنیت و اعتبارسنجی محلی باید جداگانه بررسی شوند \cite{li2019federated,bonawitz2019towards}.

در این پژوهش، یادگیری فدرال به‌صورت شبیه‌سازی‌شده انجام شد. هدف، ادعای استقرار چندمرکزی نبود، بلکه سنجش این بود که آیا یک مدل لجستیک گره‌ای می‌تواند نسبت به رگرسیون لجستیک متمرکز عملکرد قابل‌مقایسه حفظ کند یا نه.

\section{جایگاه این پایان‌نامه}
این پایان‌نامه در مرز میان انفورماتیک مراقبت ویژه، مدل‌سازی جدولی، بازنمایی توکنی، یادگیری چندنمونه‌ای و یادگیری فدرال قرار می‌گیرد. دستاورد اصلی آن یک ادعای اغراق‌آمیز درباره برتری روش نوین نیست؛ بلکه یک ارزیابی تمام‌کوهورت و قابل‌بازتولید است که نشان می‌دهد کدام روش‌ها در این وظیفه واقعاً قوی‌اند، کدام روش‌ها نیازمند بهبودند، و کدام معیارها برای تفسیر بالینی مهم‌اند.
""",
    )

    write_text(
        CHAPTER_DIR / "methodology.tex",
        rf"""
\chapter{{روش‌شناسی}}
\label{{ch:methodology}}

\section{{طراحی مطالعه}}
این پژوهش یک مطالعۀ گذشته‌نگر مبتنی بر داده‌های ساخت‌یافتۀ \lr{{MIMIC\text{{-}}IV v3.1}} است. هدف، پیش‌بینی نشانگر جانشین امکان‌پذیری ترخیص از \lr{{ICU}} با استفاده از داده‌های \pnum{{24}} ساعت نخست بستری است. تحلیل روی کوهورت کامل واجد شرایط انجام شد و هیچ سقف نمونه‌گیری فعال نبود.

\section{{منبع داده و کامل بودن داده خام}}
داده‌های خام از مسیر محلی \texttt{{\lr{{mimic-iv-3.1/}}}} خوانده شدند. اعتبارسنجی نهایی نشان داد فایل‌های ضروری بیمارستان و \lr{{ICU}}، از جمله \texttt{{\lr{{patients.csv.gz}}}}، \texttt{{\lr{{admissions.csv.gz}}}}، \texttt{{\lr{{diagnoses\_icd.csv.gz}}}}، \texttt{{\lr{{labevents.csv.gz}}}}، \texttt{{\lr{{icustays.csv.gz}}}} و \texttt{{\lr{{chartevents.csv.gz}}}} موجود بودند. گزارش اعتبارسنجی شامل \pnum{{{raw['manifest_entries']}}} ورودی در مانیفست و \pnum{{{raw['files_present']}}} فایل موجود بود؛ بنابراین مانیفست کامل تأیید شد.

\section{{تعریف کوهورت}}
کوهورت شامل بستری نخست بزرگسال در \lr{{ICU}} پس از معیارهای ورود و خروج مطالعه بود. ردیف‌های خارج از این کوهورت وارد مدل‌سازی نشدند. مجموعۀ نهایی دارای \textbf{{\pnum{{{final['rows']:,}}}}} ردیف، \pnum{{{final['columns']}}} ستون، \pnum{{{final['feature_columns']}}} ستون ویژگی، \pnum{{{final['unique_icu_stays']:,}}} بستری یکتای \lr{{ICU}} و \pnum{{{final['unique_subjects']:,}}} بیمار یکتا بود. همۀ ردیف‌ها پس از معیارهای مطالعه استفاده شدند و مقدار \lr{{max\_cohort\_sample}} برابر \lr{{null}} بود.

\section{{تعریف برچسب}}
برچسب مثبت نشان می‌دهد بیمار طبق تعریف جانشین، پیامد امکان‌پذیر داشته است. سه شرط هم‌زمان لازم بود: زنده ماندن تا ترخیص بیمارستان، مقصد ترخیص مطلوب، و طول اقامت \lr{{ICU}} کمتر از \pnum{{720}} ساعت. این تعریف از داده‌های ساخت‌یافته استخراج می‌شود و به‌معنای آمادگی واقعی ترخیص یا تصمیم درمانی نیست.

توزیع نهایی برچسب‌ها چنین بود: \pnum{{{final['label_distribution']['0']:,}}} نمونه در کلاس \pnum{{0}} و \pnum{{{final['label_distribution']['1']:,}}} نمونه در کلاس \pnum{{1}}. نسبت کلاس مثبت حدود \pnum{{62.6}}\% بود.

\section{{استخراج ویژگی}}
ویژگی‌ها از پنجرۀ \pnum{{24}} ساعت نخست بستری \lr{{ICU}} استخراج شدند. این ویژگی‌ها شامل خلاصه‌های آزمایشگاهی، علائم حیاتی، و متغیرهای مشتق‌شده بودند. پیش از جایگذاری، میانگین درصد گمشده‌ها \cnum{{{miss['mean_pre_imputation_missing_pct']:.3f}}} و بیشترین درصد گمشده‌ها \cnum{{{miss['max_pre_imputation_missing_pct']:.3f}}} بود. پس از جایگذاری، تعداد سلول‌های گمشده در دادۀ مدل‌سازی صفر گزارش شد.

\section{{توکن‌های علائم}}
در کنار ویژگی‌های جدولی، بازنمایی توکنی علائم ساخته شد. خروجی نهایی در فایل \texttt{{\lr{{data/processed\_full\_cohort/tokens.npy}}}} ذخیره شد و شکل آن \lr{{({tokens['shape'][0]}, {tokens['shape'][1]}, {tokens['shape'][2]})}} بود. هر نمونه دارای \pnum{{25}} گام توکنی و \pnum{{4}} بعد رمزگذاری بود. این توکن‌ها برای مدل \lr{{Few-Shot ETHOS}} و مدل هیبریدی استفاده شدند.

\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}[
  node distance=1.2cm,
  box/.style={{draw, rounded corners=2pt, align=center, minimum width=3.2cm, minimum height=0.9cm}},
  arrow/.style={{-Latex, thick}}
]
\node[box] (raw) {{داده خام \\ \lr{{MIMIC-IV v3.1}}}};
\node[box, left=of raw] (cohort) {{معیارهای ورود/خروج \\ بستری نخست بزرگسال}};
\node[box, below=of raw] (features) {{ویژگی‌های \pnum{{24}} ساعت نخست \\ آزمایش و علائم حیاتی}};
\node[box, right=of raw] (label) {{برچسب جانشین \\ ترخیص از \lr{{ICU}}}};
\node[box, below=of features] (tokens) {{توکن‌های علائم \\ \lr{{(32399,25,4)}}}};
\node[box, right=of tokens] (models) {{مدل‌های جدولی، هیبریدی، \\ چندنمونه‌ای و فدرال}};
\node[box, below=of models] (eval) {{ارزیابی: \\ \lr{{AUROC, AUPRC, Brier, ECE}}}};
\draw[arrow] (cohort) -- (raw);
\draw[arrow] (raw) -- (features);
\draw[arrow] (raw) -- (label);
\draw[arrow] (features) -- (tokens);
\draw[arrow] (tokens) -- (models);
\draw[arrow] (label) |- (models);
\draw[arrow] (models) -- (eval);
\end{{tikzpicture}}
\caption{{جریان کلی استخراج داده، ساخت توکن‌ها و ارزیابی مدل‌ها}}
\label{{fig:workflow}}
\end{{figure}}

\section{{مدل‌ها}}
مدل‌های اصلی عبارت بودند از رگرسیون لجستیک، جنگل تصادفی، \lr{{XGBoost}}، \lr{{LightGBM}}، \lr{{CatBoost}}، خط پایۀ عصبی، \lr{{Few-Shot ETHOS}}، \lr{{Token Hybrid RF}}، مدل انباشتی کلاسیک، و \lr{{Federated LR Node Ensemble}}. مدل انباشتی پیش‌بینی‌های مدل‌های کلاسیک تکمیل‌شده را با یک فرا-یادگیر ترکیب کرد. مدل فدرال با تقسیم گره‌ای شبیه‌سازی‌شده آموزش داده شد و خروجی گره‌ها را تجمیع کرد.

\section{{تقسیم داده و معیارهای ارزیابی}}
داده به سه بخش آموزش، اعتبارسنجی و آزمون تقسیم شد: \pnum{{20,735}} ردیف آموزش، \pnum{{5,184}} ردیف اعتبارسنجی و \pnum{{6,480}} ردیف آزمون. آستانه‌ها با استفاده از مجموعۀ اعتبارسنجی تنظیم شدند و گزارش نهایی بر مجموعۀ آزمون نگه‌داشته‌شده انجام شد.

معیارهای گزارش‌شده شامل \lr{{AUROC}}، \lr{{AUPRC}}، دقت، بازخوانی یا حساسیت، ویژگی، \lr{{F1}}، امتیاز بریر، \lr{{ECE}}، عرض از مبدأ و شیب کالیبراسیون، و ماتریس درهم‌ریختگی بودند. استفاده از معیارهای کالیبراسیون برای جلوگیری از تفسیر بیش‌ازحد \lr{{AUROC}} ضروری است \cite{{steyerberg2019clinical,vanCalster2019calibration}}.

\section{{بازتولیدپذیری و اخلاق}}
کد استخراج و مدل‌سازی مسیرهای جداگانۀ \texttt{{\lr{{data/processed\_full\_cohort/}}}} و \texttt{{\lr{{results/full\_cohort/}}}} را استفاده می‌کند. دادۀ خام \lr{{MIMIC\text{{-}}IV}} قابل بازتوزیع نیست و فقط نتایج تجمیعی، کد و پیکربندی قابل اشتراک‌اند. داده‌ها بی‌نام‌سازی شده‌اند، دسترسی نیازمند احراز صلاحیت \lr{{PhysioNet}} و توافق‌نامه استفاده از داده است، تماس مستقیم با بیمار وجود نداشت، و هیچ تلاشی برای بازشناسایی بیماران انجام نشد. وضعیت معافیت یا تأیید نهاد اخلاق دانشگاهی باید توسط نویسنده با مقررات محلی تأیید شود.
""",
    )

    write_text(
        CHAPTER_DIR / "experiments_results.tex",
        rf"""
\chapter{{آزمایش‌ها و نتایج}}
\label{{ch:results}}

\section{{خلاصۀ کوهورت کامل}}
جدول~\ref{{tab:dataset_summary}} خلاصۀ داده را نشان می‌دهد. این جدول مبنای تمام نتایج اصلی پایان‌نامه است و جایگزین زیرمجموعۀ قدیمی \pnum{{2,000}} بستری شده است.

\begin{{table}}[htbp]
\centering
\caption{{خلاصۀ کوهورت کامل مدل‌سازی}}
\label{{tab:dataset_summary}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{مولفه}} & \textbf{{مقدار}} \\
\midrule
ردیف‌های نهایی & \pnum{{{final['rows']:,}}} \\
ستون‌های نهایی & \pnum{{{final['columns']}}} \\
ستون‌های ویژگی & \pnum{{{final['feature_columns']}}} \\
بستری یکتای \lr{{ICU}} & \pnum{{{final['unique_icu_stays']:,}}} \\
بیمار یکتا & \pnum{{{final['unique_subjects']:,}}} \\
کلاس \pnum{{0}}: ناممکن & \pnum{{{final['label_distribution']['0']:,}}} \\
کلاس \pnum{{1}}: ممکن & \pnum{{{final['label_distribution']['1']:,}}} \\
شکل تانسور توکن & \lr{{({tokens['shape'][0]}, {tokens['shape'][1]}, {tokens['shape'][2]})}} \\
سقف نمونه‌گیری & \lr{{null}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.72\textwidth]{{full_cohort_label_distribution.png}}
\caption{{توزیع برچسب نشانگر جانشین امکان‌پذیری ترخیص از \lr{{ICU}} در کوهورت کامل}}
\label{{fig:label_distribution}}
\end{{figure}}

\section{{جدول اصلی عملکرد مدل‌ها}}
جدول~\ref{{tab:model_results}} نتایج کامل مدل‌ها را روی آزمون نگه‌داشته‌شده نشان می‌دهد. همۀ مدل‌های برنامه‌ریزی‌شده اجرا شدند و هیچ مدل اصلی به‌دلیل ناپایداری حذف نشد.

\begin{{table}}[htbp]
\centering
\caption{{عملکرد مدل‌ها روی کوهورت کامل}}
\label{{tab:model_results}}
\begin{{latin}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrr}}
\toprule
Model & Split & AUROC & AUPRC & Acc & Recall & Spec & F1 & Brier & ECE \\
\midrule
{model_table_rows(models)}
\bottomrule
\end{{tabular}}}}
\end{{latin}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{full_cohort_model_auroc.png}}
\caption{{مقایسۀ \lr{{AUROC}} مدل‌ها روی کوهورت کامل}}
\label{{fig:model_auroc}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{full_cohort_model_auprc.png}}
\caption{{مقایسۀ \lr{{AUPRC}} مدل‌ها روی کوهورت کامل}}
\label{{fig:model_auprc}}
\end{{figure}}

\section{{نتایج تفکیک}}
بالاترین \lr{{AUROC}} به مدل \lr{{{best['model']}}} با مقدار \cnum{{{best['auroc']:.3f}}} تعلق داشت. \lr{{XGBoost}} و \lr{{CatBoost}} تقریباً همان سطح عملکرد را با \lr{{AUROC}} \cnum{{{xgb['auroc']:.3f}}} و \cnum{{{models[models['model']=='CatBoost'].iloc[0]['auroc']:.3f}}} نشان دادند. فاصله میان مدل انباشتی و \lr{{XGBoost}} از نظر \lr{{AUROC}} بسیار کوچک بود، بنابراین انتخاب مدل تنها بر اساس رتبه \lr{{AUROC}} کافی نیست.

\section{{کالیبراسیون و خطای احتمال}}
جدول~\ref{{tab:calibration}} معیارهای کالیبراسیون را خلاصه می‌کند. \lr{{XGBoost}} با امتیاز بریر \cnum{{{xgb['brier_score']:.3f}}} و \lr{{ECE}} \cnum{{{xgb['calibration_ece_10bin']:.3f}}} یکی از بهترین توازن‌ها را میان تفکیک و کالیبراسیون داشت. مدل انباشتی اگرچه بالاترین \lr{{AUROC}} را داشت، \lr{{ECE}} بالاتری نشان داد. ازاین‌رو، در این پایان‌نامه \lr{{XGBoost}} مدل اصلی قابل‌دفاع‌تر برای تفسیر پژوهشی معرفی می‌شود.

\begin{{table}}[htbp]
\centering
\caption{{معیارهای کالیبراسیون و آستانه}}
\label{{tab:calibration}}
\begin{{latin}}
\scriptsize
\resizebox{{0.9\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Model & Threshold & Brier & ECE & Intercept & Slope \\
\midrule
{calibration_rows(models)}
\bottomrule
\end{{tabular}}}}
\end{{latin}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{full_cohort_calibration_brier_ece.png}}
\caption{{مقایسۀ امتیاز بریر و \lr{{ECE}}؛ مقدار کمتر بهتر است}}
\label{{fig:calibration}}
\end{{figure}}

\section{{مدل‌های چندنمونه‌ای و توکن‌محور}}
مدل \lr{{Few-Shot ETHOS}} با \lr{{AUROC}} \cnum{{{few['auroc']:.3f}}}، \lr{{AUPRC}} \cnum{{{few['auprc']:.3f}}} و \lr{{F1}} \cnum{{{few['f1_macro']:.3f}}} از مدل‌های جدولی ضعیف‌تر بود. این نتیجه نشان می‌دهد که در وظیفۀ فعلی، توکن‌سازی و یادگیری چندنمونه‌ای خام به‌تنهایی برای رقابت با مدل‌های درختی کافی نیستند. بااین‌حال، مدل \lr{{Token Hybrid RF}} با \lr{{AUROC}} \cnum{{{hybrid['auroc']:.3f}}} به جنگل تصادفی و مدل‌های گرادیان‌بوستینگ نزدیک شد. بنابراین توکن‌ها در نقش مکمل برای مدل هیبریدی امیدوارکننده‌تر از استفاده مستقل بودند.

\section{{شبیه‌سازی فدرال}}
مدل \lr{{Federated LR Node Ensemble}} با \lr{{AUROC}} \cnum{{{fed['auroc']:.3f}}} تقریباً هم‌سطح رگرسیون لجستیک متمرکز با \lr{{AUROC}} \cnum{{{lr['auroc']:.3f}}} بود. این نتیجه نشان می‌دهد که در خانوادۀ مدل‌های لجستیک، شبیه‌سازی فدرال می‌تواند عملکرد نزدیک به مدل متمرکز حفظ کند. بااین‌حال، هر دو مدل لجستیک از مدل‌های درختی ضعیف‌تر بودند. بنابراین یادگیری فدرال در این پژوهش به‌عنوان مسیر حفظ حریم خصوصی بررسی می‌شود، نه راهی برای افزایش عملکرد.

\section{{ماتریس درهم‌ریختگی}}
شکل~\ref{{fig:xgb_cm}} ماتریس درهم‌ریختگی \lr{{XGBoost}} را نشان می‌دهد. این مدل روی آزمون \pnum{{6,480}} نمونه‌ای، \pnum{{3,091}} مثبت درست و \pnum{{1,425}} منفی درست داشت. بازخوانی بالاتر از ویژگی بود؛ یعنی مدل در شناسایی کلاس مثبت حساس‌تر از رد کلاس منفی عمل کرد.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.62\textwidth]{{full_cohort_xgboost_confusion_matrix.png}}
\caption{{ماتریس درهم‌ریختگی مدل \lr{{XGBoost}} روی مجموعۀ آزمون}}
\label{{fig:xgb_cm}}
\end{{figure}}

\section{{مقایسه با بنچمارک تاریخی \pnum{{2,000}} بستری}}
زیرمجموعۀ قدیمی \pnum{{2,000}} بستری فقط به‌عنوان زمینۀ تاریخی استفاده می‌شود و مبنای ادعاهای اصلی نیست. جدول~\ref{{tab:benchmark_comparison}} و شکل~\ref{{fig:benchmark_comparison}} نشان می‌دهند که برخی مدل‌ها در کوهورت کامل بهبود داشته‌اند و برخی اندکی کاهش نشان داده‌اند. به‌دلیل تفاوت پروتکل‌ها، این مقایسه باید توصیفی تفسیر شود.

\begin{{table}}[htbp]
\centering
\caption{{مقایسۀ توصیفی کوهورت کامل با بنچمارک تاریخی}}
\label{{tab:benchmark_comparison}}
\begin{{latin}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrl}}
\toprule
Model & Full AUROC & Old AUROC & Delta & Source \\
\midrule
{comparison_rows(comparison)}
\bottomrule
\end{{tabular}}}}
\end{{latin}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{full_vs_2k_benchmark_auroc.png}}
\caption{{مقایسۀ توصیفی \lr{{AUROC}} کوهورت کامل با بنچمارک تاریخی \pnum{{2,000}} بستری}}
\label{{fig:benchmark_comparison}}
\end{{figure}}

\section{{پاسخ مستقیم به پرسش‌های پژوهش}}
\begin{{enumerate}}
\item ویژگی‌های \pnum{{24}} ساعت نخست توان پیش‌بینی متوسط تا خوب برای برچسب جانشین داشتند؛ بهترین \lr{{AUROC}} برابر \cnum{{{best['auroc']:.3f}}} بود.
\item مدل‌های جدولی و درختی از \lr{{Few-Shot ETHOS}} خام قوی‌تر بودند.
\item توکن‌ها در مدل هیبریدی مفیدتر از حالت مستقل ظاهر شدند.
\item شبیه‌سازی فدرال عملکردی نزدیک به رگرسیون لجستیک متمرکز حفظ کرد، اما به مدل‌های درختی نرسید.
\item \lr{{XGBoost}} بهترین توازن میان تفکیک و کالیبراسیون را داشت و برای گزارش اصلی پژوهش مناسب‌تر بود.
\end{{enumerate}}
""",
    )

    write_text(
        CHAPTER_DIR / "discussion.tex",
        rf"""
\chapter{{بحث}}
\label{{ch:discussion}}

\section{{یافته‌های اصلی}}
این پایان‌نامه نشان داد که در کوهورت کامل \lr{{MIMIC\text{{-}}IV}}، داده‌های \pnum{{24}} ساعت نخست \lr{{ICU}} می‌توانند نشانگر جانشین امکان‌پذیری ترخیص را با عملکرد قابل‌توجه اما نه قطعی پیش‌بینی کنند. بالاترین \lr{{AUROC}} متعلق به مدل انباشتی کلاسیک بود، اما \lr{{XGBoost}} از نظر کالیبراسیون و امتیاز بریر قابل‌دفاع‌تر بود. این تمایز مهم است، زیرا در پژوهش‌های سلامت، احتمال خروجی مدل تنها زمانی مفید است که تا حد امکان با خطر مشاهده‌شده هم‌خوان باشد.

\section{{چرا \lr{{XGBoost}} مدل اصلی قابل‌دفاع‌تر است؟}}
مدل انباشتی با \lr{{AUROC}} \cnum{{{best['auroc']:.3f}}} اندکی بالاتر از \lr{{XGBoost}} بود، اما \lr{{ECE}} آن \cnum{{{best['calibration_ece_10bin']:.3f}}} بود. در مقابل، \lr{{XGBoost}} با \lr{{AUROC}} \cnum{{{xgb['auroc']:.3f}}} و \lr{{ECE}} \cnum{{{xgb['calibration_ece_10bin']:.3f}}} توازن بهتری ارائه کرد. در یک سناریوی پژوهشی نزدیک به پشتیبان تصمیم، چنین توازنی مهم‌تر از بهبود بسیار کوچک \lr{{AUROC}} است. به همین دلیل، این پایان‌نامه \lr{{XGBoost}} را مدل اصلی قابل‌دفاع‌تر می‌داند، نه الزاماً مدل با بیشترین عدد در یک معیار واحد.

\section{{چرا \lr{{Few-Shot ETHOS}} ضعیف‌تر بود؟}}
نتیجۀ ضعیف‌تر \lr{{Few-Shot ETHOS}} یافته‌ای منفی اما ارزشمند است. یادگیری چندنمونه‌ای در حوزه‌هایی موفق است که بازنمایی‌های پرظرفیت و ساختارهای شباهت پیچیده سودمند باشند. در این وظیفه، داده‌ها جدولی، کم‌بعد و دارای معنای مستقیم بالینی بودند. مدل‌های درختی می‌توانستند آستانه‌ها، تعامل‌ها و غیرخطی‌ها را به‌خوبی استخراج کنند. از سوی دیگر، توکن‌سازی فشرده ممکن است بخشی از شدت و پیوستگی مقادیر آزمایشگاهی را کاهش داده باشد. بنابراین، \lr{{ETHOS}} خام نتوانست با مدل‌های جدولی رقابت کند.

این نتیجه به‌معنای بی‌ارزش بودن توکن‌ها نیست. عملکرد \lr{{Token Hybrid RF}} نشان داد وقتی توکن‌ها در کنار ویژگی‌های جدولی استفاده شوند، می‌توانند سیگنال مکمل داشته باشند. مسیر آینده باید بر آموزش نظارت‌شدۀ بهتر توکن‌ها، اپیزودهای بالینی معنادارتر، و هم‌جوشی کالیبراسیون‌محور تمرکز کند.

\section{{پیامدهای مدل هیبریدی و انباشتی}}
مدل هیبریدی و مدل انباشتی نشان دادند ترکیب بازنمایی‌ها می‌تواند مفید باشد. بااین‌حال، بهبود مدل انباشتی باید با احتیاط تفسیر شود؛ زیرا کالیبراسیون آن از \lr{{XGBoost}} ضعیف‌تر بود. در کاربردهای پژوهشی، مدل انباشتی می‌تواند برای رتبه‌بندی خطر مناسب باشد، اما برای گزارش احتمال، نیازمند کالیبراسیون جداگانه است.

\section{{یادگیری فدرال}}
شبیه‌سازی فدرال نشان داد مدل گره‌ای لجستیک می‌تواند به رگرسیون لجستیک متمرکز نزدیک بماند. این یافته برای حریم خصوصی مهم است، زیرا نشان می‌دهد اگر هدف یک مدل ساده و قابل‌توضیح باشد، توزیع گره‌ای الزاماً افت شدید ایجاد نمی‌کند. اما این نتیجه نباید به استقرار واقعی تعمیم داده شود. در بیمارستان‌های واقعی، گره‌ها ناهمگن‌ترند، اندازه نمونه‌ها متفاوت است، ارتباطات محدود است و نیاز به ارزیابی امنیت، انصاف و پایش انحراف وجود دارد.

\section{{رابطه با عنوان پایان‌نامه}}
عنوان پایان‌نامه از «فیوشات درمانی» و «قابلیت درمان» سخن می‌گوید. بازنگری علمی متن لازم داشت که این عنوان با اجرای واقعی همسو شود. بنابراین، «قابلیت درمان» در این پایان‌نامه به‌صورت «نشانگر جانشین امکان‌پذیری ترخیص از \lr{{ICU}}» تعریف شده است. این شفاف‌سازی از بیش‌ادعایی جلوگیری می‌کند و نشان می‌دهد پژوهش یک سامانه درمان‌گر یا توصیه‌گر درمان نیست، بلکه یک مطالعۀ مدل‌سازی گذشته‌نگر است.

\section{{نقاط قوت}}
نقاط قوت پژوهش شامل استفاده از کوهورت کامل واجد شرایط، حذف سقف نمونه‌گیری، استفاده از داده خام کامل و مانیفست‌تأییدشده، مقایسۀ چند خانوادۀ مدل، گزارش کالیبراسیون، و تفکیک روشن میان نتایج اصلی و بنچمارک تاریخی است. همچنین متن حاضر نتایج منفی را پنهان نکرده و نشان داده است که مدل‌های کلاسیک در داده‌های جدولی بالینی همچنان باید جدی گرفته شوند.

\section{{محدودیت‌ها}}
محدودیت‌های اصلی عبارت‌اند از:
\begin{{enumerate}}
\item مطالعه گذشته‌نگر و مبتنی بر یک پایگاه داده است.
\item برچسب، جانشین است و توسط متخصصان به‌عنوان آمادگی ترخیص یا امکان‌پذیری واقعی درمان داوری نشده است.
\item اعتبارسنجی خارجی روی بیمارستان دیگر انجام نشده است.
\item اعتبارسنجی آینده‌نگر یا مداخله‌ای وجود ندارد.
\item ویژگی‌های \pnum{{24}} ساعت نخست مسیرهای دیرتر بیماری را کامل نشان نمی‌دهند.
\item احتمال مخدوش‌سازی باقیمانده وجود دارد.
\item یادگیری فدرال شبیه‌سازی شده و در شبکه واقعی بیمارستانی اجرا نشده است.
\item \lr{{Few-Shot ETHOS}} در این نسخه ضعیف‌تر از مدل‌های جدولی بود و نیاز به بهبود دارد.
\item کالیبراسیون میان مدل‌ها متفاوت است و \lr{{AUROC}} به‌تنهایی کافی نیست.
\item دادۀ خام \lr{{MIMIC\text{{-}}IV}} طبق محدودیت‌های استفاده قابل بازتوزیع نیست.
\end{{enumerate}}

\section{{کارهای آینده}}
مسیرهای آینده شامل اعتبارسنجی خارجی، اعتبارسنجی آینده‌نگر، مدل‌سازی سری زمانی غنی‌تر، آموزش نظارت‌شدۀ توکن‌ها، ساخت اپیزودهای بالینی معنادارتر برای یادگیری چندنمونه‌ای، آموزش کالیبراسیون‌محور، تحلیل انصاف و زیرگروه، فدرال واقعی چندمرکزی، و ارزیابی انسان-در-حلقه با بالین‌گران است.
""",
    )

    write_text(
        CHAPTER_DIR / "conclusion.tex",
        rf"""
\chapter{{نتیجه‌گیری}}
\label{{ch:conclusion}}

این پایان‌نامه یک ارزیابی تمام‌کوهورت و گذشته‌نگر از پیش‌بینی نشانگر جانشین امکان‌پذیری ترخیص از \lr{{ICU}} در \lr{{MIMIC\text{{-}}IV v3.1}} ارائه کرد. نسخۀ نهایی از \pnum{{32,399}} بستری نخست بزرگسال استفاده کرد و دیگر بر زیرمجموعۀ \pnum{{2,000}} بستری به‌عنوان تحلیل اصلی متکی نیست.

پرسش نخست پژوهش پاسخ مثبت مشروط گرفت: داده‌های \pnum{{24}} ساعت نخست \lr{{ICU}} توان پیش‌بینی متوسط تا خوب برای برچسب جانشین داشتند. پرسش دوم نشان داد مدل‌های جدولی و درختی از \lr{{Few-Shot ETHOS}} خام قوی‌تر بودند. پرسش سوم نشان داد توکن‌های علائم در حالت هیبریدی مفیدتر از حالت مستقل‌اند. پرسش چهارم نشان داد فدرال شبیه‌سازی‌شده می‌تواند به خط پایۀ لجستیک متمرکز نزدیک بماند. پرسش پنجم نشان داد \lr{{XGBoost}} بهترین توازن عملی میان تفکیک و کالیبراسیون را فراهم کرد، هرچند مدل انباشتی بالاترین \lr{{AUROC}} را داشت.

جمع‌بندی علمی کار این است که نوآوری روش‌شناختی باید در برابر خط پایه‌های قوی آزموده شود. در این وظیفۀ ساخت‌یافتۀ \lr{{EHR}}، روش چندنمونه‌ای خام برتری نداشت؛ اما همین نتیجه پایان‌نامه را صادقانه‌تر و مفیدتر می‌کند. مدل‌های درختی، به‌ویژه \lr{{XGBoost}}، برای داده‌های جدولی بالینی بسیار رقابتی باقی ماندند. توکن‌های علائم و یادگیری فدرال نیز مسیرهای پژوهشی قابل‌پیگیری هستند، به شرط آنکه در مطالعات آینده با برچسب‌های بالینی دقیق‌تر، داده‌های چندمرکزی و اعتبارسنجی آینده‌نگر ارزیابی شوند.

بنابراین، دستاورد اصلی پایان‌نامه یک سامانه آماده استقرار نیست؛ بلکه یک بنچمارک تمام‌کوهورت، شفاف و بازتولیدپذیر است که نشان می‌دهد در مسئلۀ پیش‌بینی جانشین امکان‌پذیری ترخیص از \lr{{ICU}}، کدام رویکردها اکنون قوی‌ترند و کدام مسیرها برای پژوهش‌های بعدی باید بهبود یابند.
""",
    )

    write_text(
        CHAPTER_DIR / "appendix_reproducibility.tex",
        rf"""
\chapter{{پیوست بازتولیدپذیری و ممیزی نتایج}}
\label{{app:reproducibility}}

\section{{مسیرهای اصلی}}
\begin{{itemize}}
\item دادۀ پردازش‌شده: \texttt{{\lr{{data/processed\_full\_cohort/thesis\_dataset.parquet}}}}
\item توکن‌ها: \texttt{{\lr{{data/processed\_full\_cohort/tokens.npy}}}}
\item نتایج: \texttt{{\lr{{results/full\_cohort/}}}}
\item جدول اصلی مدل‌ها: \texttt{{\lr{{results/full\_cohort/tables/full\_cohort\_model\_results.csv}}}}
\end{{itemize}}

\section{{فرمان‌های استخراج}}
\begin{{latin}}
\begin{{verbatim}}
python extraction/01_cohort.py
python extraction/02_symptoms.py
python extraction/03_labels.py
python extraction/04_tokenize.py
python extraction/05_federated_split.py
\end{{verbatim}}
\end{{latin}}

\section{{فرمان‌های مدل‌سازی}}
\begin{{latin}}
\begin{{verbatim}}
python scripts/validate_full_cohort.py
python scripts/run_full_cohort_models.py --fewshot-episodes 40
\end{{verbatim}}
\end{{latin}}

\section{{کنترل ادعاها}}
در نگارش نهایی، بنچمارک \pnum{{2,000}} بستری فقط به‌عنوان سابقۀ تاریخی گزارش شده است. هیچ نتیجه‌ای از آن به‌عنوان نتیجۀ کوهورت کامل استفاده نشده است. مدل‌های ضعیف‌تر، از جمله \lr{{Few-Shot ETHOS}}، از جدول اصلی حذف نشده‌اند و نتیجه آن‌ها صریحاً گزارش شده است.
""",
    )


def write_report(validation: dict, models: pd.DataFrame) -> None:
    final = validation["final_dataset"]
    best = models.sort_values("auroc", ascending=False).iloc[0]
    xgb = models[models["model"] == "XGBoost"].iloc[0]
    rows = []
    for _, r in models.sort_values("auroc", ascending=False).iterrows():
        rows.append(
            f"| {r['model']} | {r['auroc']:.3f} | {r['auprc']:.3f} | {r['accuracy']:.3f} | {r['sensitivity_recall']:.3f} | {r['specificity']:.3f} | {r['f1_macro']:.3f} | {r['brier_score']:.3f} | {r['calibration_ece_10bin']:.3f} |"
        )
    report = f"""# Thesis Revision Report

## What Changed

- Reframed the thesis as a retrospective full-cohort MIMIC-IV clinical machine-learning study.
- Replaced the old 2,000-stay main-cohort framing with the verified full eligible cohort.
- Defined "treatment feasibility" conservatively as an ICU discharge-feasibility proxy.
- Rewrote the Persian thesis source into six coherent academic chapters plus reproducibility appendix.
- Added honest interpretation: pure Few-Shot ETHOS did not outperform classical tabular models.
- Treated XGBoost as the most defensible primary model because it balances discrimination and calibration.

## Full-Cohort Data Used

- Dataset: `data/processed_full_cohort/thesis_dataset.parquet`
- Shape: {final['rows']:,} rows x {final['columns']} columns
- Feature columns: {final['feature_columns']}
- Unique ICU stays: {final['unique_icu_stays']:,}
- Unique subjects: {final['unique_subjects']:,}
- Label distribution: 0={final['label_distribution']['0']:,}, 1={final['label_distribution']['1']:,}
- Token tensor: `data/processed_full_cohort/tokens.npy`, shape `(32399, 25, 4)`
- Sampling cap: `null`
- All eligible data used: `true`

## Final Model Results

| Model | AUROC | AUPRC | Accuracy | Recall | Specificity | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Best AUROC: {best['model']} ({best['auroc']:.3f}).
Primary balanced interpretation model: XGBoost (AUROC {xgb['auroc']:.3f}, Brier {xgb['brier_score']:.3f}, ECE {xgb['calibration_ece_10bin']:.3f}).

## Sections Rewritten

- Persian abstract and English abstract
- Introduction
- Background and literature review
- Methodology
- Experiments and results
- Discussion
- Conclusion
- Reproducibility appendix

## Tables And Figures Regenerated

- Full-cohort dataset summary
- Main model performance table
- Calibration table
- Full-cohort vs historical benchmark comparison
- Label distribution figure
- AUROC figure
- AUPRC figure
- Brier/ECE figure
- XGBoost confusion matrix
- Workflow diagram

## Remaining Limitations

- Retrospective single-database MIMIC-IV study.
- Outcome is a proxy, not clinician-adjudicated treatment feasibility or discharge readiness.
- No prospective or external validation.
- Simulated rather than real multi-site federated learning.
- Few-Shot ETHOS underperformed and requires methodological refinement.
- Raw MIMIC-IV data cannot be redistributed.

## Remaining Author Confirmations

- Confirm exact university administrative metadata, supervisor spelling, date, and defense details.
- Confirm whether local IRB/exemption wording is required by the university.
- Confirm whether the title should keep the conceptual "Therapeutic Few-Shot" wording or add the ICU proxy subtitle on formal submission pages.

## Readiness

- Thesis status: ready for supervisor review after author confirmation of administrative metadata.
- Manuscript package status: full-cohort results are available and the BMC package exists; journal submission should wait for author review of framing and cover-letter details.
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVISION_REPORT.write_text(report, encoding="utf-8")
    (ROOT / "THESIS_REVISION_REPORT.md").write_text(report, encoding="utf-8")


def package_source() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    include_files: list[Path] = [
        SOURCE_TEX,
        THESIS_DIR / "references.bib",
        THESIS_DIR / "thesis_fa_digits.tex",
        THESIS_DIR / "thesis_fa_fonts.tex",
        THESIS_DIR / "thesis_fa_frontmatter.tex",
    ]
    include_files += sorted(CHAPTER_DIR.glob("*.tex"))
    include_files += sorted(FRONT_DIR.glob("*.tex"))
    include_files += sorted((THESIS_DIR / "frontmatter_fa").glob("commitment.tex"))
    include_files += sorted((THESIS_DIR / "frontmatter_fa").glob("dedication.tex"))
    include_files += sorted((THESIS_DIR / "fonts").glob("*"))
    include_files += sorted((THESIS_DIR / "assets").glob("*"))
    include_files += sorted(FIGURES_DIR.glob("full_cohort_*.png"))
    include_files += sorted(FIGURES_DIR.glob("full_vs_2k_*.png"))
    include_files += sorted(TABLES_DIR.glob("full_cohort_*.csv"))
    include_files += sorted(TABLES_DIR.glob("full_vs_2k_*.csv"))
    if (THESIS_DIR / "thesis_FULL_COHORT_REVISED.pdf").exists():
        include_files.append(THESIS_DIR / "thesis_FULL_COHORT_REVISED.pdf")
    include_files.append(REVISION_REPORT)

    with zipfile.ZipFile(SOURCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include_files:
            if not path.exists() or path.is_dir():
                continue
            if path.is_relative_to(THESIS_DIR):
                arc = Path("thesis_source") / path.relative_to(THESIS_DIR)
            elif path.is_relative_to(ROOT):
                arc = Path("thesis_source") / path.relative_to(ROOT)
            else:
                arc = Path("thesis_source") / path.name
            zf.write(path, arc.as_posix())


def main() -> int:
    validation, models, comparison, labels, missingness = read_inputs()
    ensure_extra_references()
    make_figures(models)
    write_frontmatter(validation, models)
    write_chapters(validation, models, comparison, missingness)
    build_driver(validation, models)
    write_report(validation, models)
    package_source()
    print(SOURCE_TEX)
    print(PAPER_TEX)
    print(REVISION_REPORT)
    print(SOURCE_ZIP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
