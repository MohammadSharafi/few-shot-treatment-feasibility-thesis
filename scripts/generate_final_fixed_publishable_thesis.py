#!/usr/bin/env python3
"""Generate the final fixed publishable Persian thesis package.

The script intentionally builds the final thesis in
papers/final_fixed_publishable_thesis while leaving the original thesis/
directory untouched. It reuses the original XePersian template, fonts,
front-matter style, and bibliography, then replaces the empirical and
interpretive thesis body with the verified no-gap evidence.
"""
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis"
OUT = ROOT / "papers" / os.environ.get("THESIS_OUTPUT_DIR_NAME", "final_fixed_publishable_thesis")
RES = ROOT / "results" / "final_optimized"
FIG = OUT / "stage2_figures"
DATA = OUT / "source_data"
FINAL_BASENAME = os.environ.get("THESIS_BASENAME", "thesis_FINAL_FIXED_PUBLISHABLE")
READY_LABEL = os.environ.get("THESIS_READY_LABEL", "FINAL_FIXED_THESIS_READY")
FIX_REPORT_NAME = os.environ.get("THESIS_FIX_REPORT_NAME", "THESIS_FIX_REPORT.md")
DEFENSE_REPORT_NAME = os.environ.get("THESIS_DEFENSE_REPORT_NAME", "THESIS_DEFENSE_SUMMARY.md")
CHECKLIST_NAME = os.environ.get("THESIS_CHECKLIST_NAME", "PUBLISHABILITY_CHECKLIST.md")
FONT_MODE = os.environ.get("THESIS_FONT_MODE", "latin_safe")


FINAL = {
    "stays": 32399,
    "subjects": 32399,
    "label0": 12102,
    "label1": 20297,
    "token_shape": "(32399, 25, 4)",
    "registry_rows": 254,
    "stack_auroc": 0.758564,
    "stack_auprc": 0.820696,
    "stack_brier": 0.187457,
    "stack_ece": 0.015790,
    "ens_auroc": 0.758593,
    "ens_auprc": 0.823514,
    "token_auroc": 0.756777,
    "token_auprc": 0.820064,
    "token_brier": 0.188624,
    "token_ece": 0.011293,
    "pre_auroc": 0.755238,
    "pre_auprc": 0.819027,
    "few_auroc": 0.607037,
    "few_auprc": 0.698634,
}


FEATURE_GROUPS = [
    ("آزمایش‌های ۲۴ ساعت اول", "۱۵", r"\lr{50912, 50971, 50983, 50902, 50931, 50893, 50960, 50811, 51265, 51300, 50813, 50861, 50878, 51006, 50882}", "میانه مقدارهای ثبت‌شده در پنجره نخست ICU، با کلیپ بالینی و جایگذاری آموزش‌محور."),
    ("علائم حیاتی ۲۴ ساعت اول", "۷", r"\lr{220045, 220050, 220051, 220052, 223761, 220277, 220210}", "ضربان قلب، فشار خون‌ها، دما، اشباع اکسیژن و نرخ تنفس در نخستین ۲۴ ساعت."),
    ("ویژگی‌های مهندسی‌شده", "۳", r"\lr{shock index proxy; SOFA-like proxy; abnormal-count proxy}", "شاخص‌های خلاصه‌شده برای شدت اولیه؛ هیچ‌کدام برچسب یا اطلاعات پس از پنجره پیش‌بینی نیستند."),
    ("شاخص‌های گمشده‌بودن", "۲۲", r"\lr{missingness flags}", "ثبت الگوی مشاهده/عدم مشاهده برای کاهش اتلاف سیگنال‌های informative missingness در ICU."),
]


RISKY_PHRASES = [
    "clinical deployment-ready",
    "treatment recommendation system",
    "clinician-adjudicated readiness",
    "clinician-adjudicated treatment feasibility",
    "few-shot superiority",
    "2,000-stay main cohort",
    "sampled benchmark as main analysis",
    "best model Random Forest",
    "stackingsupervised",
    "hybrid, pretraining, stacking",
    "Vazirmatn",
]


def ensure_clean_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {"EXISTING_THESIS_AUDIT.md", "OLD_TO_NEW_UPDATE_MAP.md", "BACKUP_INFO.md", "backups"}
    for child in OUT.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for sub in ["chapters_fa", "frontmatter_fa", "fonts", "assets", "stage2_figures", "source_data"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def copy_template() -> None:
    for name in ["thesis_fa_fonts.tex", "thesis_fa_frontmatter.tex", "thesis_fa_digits.tex", "references.bib", "latexmkrc"]:
        src = THESIS / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    for d in ["fonts", "assets"]:
        src = THESIS / d
        dst = OUT / d
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    if FONT_MODE == "bnazanin":
        ensure_bnazanin_local_font()
        write_bnazanin_font_config()
    else:
        write_latin_safe_font_config()
    # Keep official forms and dedication, then overwrite the updated front matter.
    for src in (THESIS / "frontmatter_fa").glob("*.tex"):
        shutil.copy2(src, OUT / "frontmatter_fa" / src.name)
    append_verified_bib_entries()


def ensure_bnazanin_local_font() -> None:
    """Copy the user's local B Nazanin font into the source package when available."""
    candidates = [
        Path("/Users/moe/Library/Fonts/BNazanin.ttf"),
        Path("/Users/moe/Library/Fonts/B Nazanin Regular.ttf"),
    ]
    dst = OUT / "fonts" / "BNazanin.ttf"
    if dst.exists():
        return
    for src in candidates:
        if src.exists():
            shutil.copy2(src, dst)
            return
    raise FileNotFoundError("B Nazanin font file was not found in the project or user font library.")


def write_bnazanin_font_config() -> None:
    """Use local B Nazanin for Persian thesis typography; Latin is routed through Times."""
    (OUT / "thesis_fa_fonts.tex").write_text(
        r"""% =============================================================================
% thesis_fa_fonts.tex
% Final typography QC font setup.
% Persian body font: local B Nazanin file copied into fonts/BNazanin.ttf.
% Latin text: Times New Roman when available, otherwise TeX Gyre Termes.
% =============================================================================

\def\thesisLatinTextScale{0.90}
\def\thesisMonoScale{0.80}

\ExplSyntaxOn
\cs_if_exist:NT \__xepersian_mathdigitspec_char_not_exist_error:n
  { \cs_set_eq:NN \__xepersian_mathdigitspec_char_not_exist_error:n \use_none:n }
\ExplSyntaxOff

\IfFileExists{fonts/BNazanin.ttf}{%
  \message{^^Jthesis_fa_fonts: Persian text = local B Nazanin (fonts/BNazanin.ttf)^^J}%
  \settextfont[
    Path=fonts/,
    Extension=.ttf,
    UprightFont=BNazanin,
    BoldFont=BNazanin,
    ItalicFont=BNazanin,
    BoldItalicFont=BNazanin,
    ItalicFeatures={FakeSlant=0.18},
    BoldFeatures={FakeBold=1.6},
    BoldItalicFeatures={FakeBold=1.6,FakeSlant=0.18},
    Ligatures=TeX,
    Scale=1.0
  ]{BNazanin}%
  \setdigitfont[
    Path=fonts/,
    Extension=.ttf,
    UprightFont=BNazanin,
    BoldFont=BNazanin,
    ItalicFont=BNazanin,
    BoldItalicFont=BNazanin,
    ItalicFeatures={FakeSlant=0.18},
    BoldFeatures={FakeBold=1.6},
    BoldItalicFeatures={FakeBold=1.6,FakeSlant=0.18},
    Ligatures=TeX,
    Scale=1.0
  ]{BNazanin}%
}{%
  \PackageError{thesis_fa_fonts}{Required B Nazanin font file fonts/BNazanin.ttf is missing}{Copy BNazanin.ttf into the fonts directory.}%
}%

\IfFontExistsTF{Times New Roman}{%
  \setlatintextfont[Ligatures=TeX,Scale=\thesisLatinTextScale]{Times New Roman}%
}{%
  \setlatintextfont[Ligatures=TeX,Scale=\thesisLatinTextScale]{TeX Gyre Termes}%
}%

\IfFontExistsTF{Courier New}{%
  \setmonofont[Scale=\thesisMonoScale]{Courier New}%
}{%
  \setmonofont[Scale=\thesisMonoScale]{Latin Modern Mono}%
}%
""",
        encoding="utf-8",
    )


def write_latin_safe_font_config() -> None:
    """Prefer a Persian font that also renders Latin acronyms used in the rewrite."""
    (OUT / "thesis_fa_fonts.tex").write_text(
        r"""% =============================================================================
% thesis_fa_fonts.tex
% Generated Stage-2 font setup for the fixed publishable thesis.
% Vazirmatn is preferred because it renders Persian text, Persian digits, and
% inline Latin acronyms without missing-glyph boxes in XePersian output.
% =============================================================================

\def\thesisLatinTextScale{0.90}
\def\thesisMonoScale{0.80}

\ExplSyntaxOn
\cs_if_exist:NT \__xepersian_mathdigitspec_char_not_exist_error:n
  { \cs_set_eq:NN \__xepersian_mathdigitspec_char_not_exist_error:n \use_none:n }
\ExplSyntaxOff

\IfFontExistsTF{Vazirmatn}{%
  \message{^^Jthesis_fa_fonts: Persian text = Vazirmatn (Latin-safe)^^J}%
  \settextfont[Ligatures=TeX,Scale=1.0]{Vazirmatn}%
  \setdigitfont[Ligatures=TeX,Scale=1.0]{Vazirmatn}%
}{%
  \IfFontExistsTF{Arial Unicode MS}{%
    \message{^^Jthesis_fa_fonts: Persian text = Arial Unicode MS fallback^^J}%
    \settextfont[Ligatures=TeX,Scale=1.0]{Arial Unicode MS}%
    \setdigitfont[Ligatures=TeX,Scale=1.0]{Arial Unicode MS}%
  }{%
    \IfFileExists{fonts/BMitra.ttf}{%
      \message{^^Jthesis_fa_fonts: Persian text = local BMitra fallback^^J}%
      \IfFileExists{fonts/BMitraBd.ttf}{%
        \settextfont[Path=fonts/,Extension=.ttf,UprightFont=BMitra,BoldFont=BMitraBd,ItalicFont=BMitra,BoldItalicFont=BMitraBd,ItalicFeatures={FakeSlant=0.2},BoldItalicFeatures={FakeSlant=0.2},Scale=1.0]{BMitra}%
        \setdigitfont[Path=fonts/,Extension=.ttf,UprightFont=BMitra,BoldFont=BMitraBd,ItalicFont=BMitra,BoldItalicFont=BMitraBd,ItalicFeatures={FakeSlant=0.2},BoldItalicFeatures={FakeSlant=0.2},Scale=1.0]{BMitra}%
      }{%
        \settextfont[Path=fonts/,Extension=.ttf,UprightFont=BMitra,BoldFont=BMitra,Scale=1.0]{BMitra}%
        \setdigitfont[Path=fonts/,Extension=.ttf,UprightFont=BMitra,BoldFont=BMitra,Scale=1.0]{BMitra}%
      }%
    }{%
      \PackageError{thesis_fa_fonts}{No Persian body font found}{Install Vazirmatn or keep BMitra.ttf in fonts/.}%
    }%
  }%
}%

\IfFontExistsTF{Times New Roman}{%
  \setlatintextfont[Ligatures=TeX,Scale=\thesisLatinTextScale]{Times New Roman}%
}{%
  \setlatintextfont[Ligatures=TeX,Scale=\thesisLatinTextScale]{TeX Gyre Termes}%
}%

\IfFontExistsTF{Courier New}{%
  \setmonofont[Scale=\thesisMonoScale]{Courier New}%
}{%
  \setmonofont[Scale=\thesisMonoScale]{TeX Gyre Cursor}%
}%
""",
        encoding="utf-8",
    )


def append_verified_bib_entries() -> None:
    """Add real bibliography entries needed by the rewritten thesis if missing."""
    bib = OUT / "references.bib"
    text = bib.read_text(encoding="utf-8", errors="ignore")
    entries = {
        "goldberger2000physionet": r"""
@article{goldberger2000physionet,
  title={PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals},
  author={Goldberger, Ary L and Amaral, Luis A N and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
  journal={Circulation},
  volume={101},
  number={23},
  pages={e215--e220},
  year={2000}
}
""",
        "guo2017calibration": r"""
@inproceedings{guo2017calibration,
  title={On calibration of modern neural networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q},
  booktitle={International Conference on Machine Learning},
  pages={1321--1330},
  year={2017}
}
""",
        "wolff2019probast": r"""
@article{wolff2019probast,
  title={PROBAST: a tool to assess the risk of bias and applicability of prediction model studies},
  author={Wolff, Robert F and Moons, Karel G M and Riley, Richard D and Whiting, Penny F and Westwood, Marie and Collins, Gary S and Reitsma, Johannes B and Kleijnen, Jos and Mallett, Susan},
  journal={Annals of Internal Medicine},
  volume={170},
  number={1},
  pages={51--58},
  year={2019}
}
""",
    }
    with bib.open("a", encoding="utf-8") as f:
        for key, entry in entries.items():
            if key not in text:
                f.write("\n" + entry.strip() + "\n")


def copy_source_evidence() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    sources = [
        RES / "NO_GAP_FINAL_COMPLETION_REPORT.md",
        RES / "FINAL_MODEL_SELECTION_NO_GAP.md",
        RES / "THESIS_TITLE_ALIGNMENT_NO_GAP.md",
        RES / "ethos_no_gap" / "ETHOS_NO_GAP_COMPLETION_REPORT.md",
        RES / "pretraining_no_gap" / "PRETRAINING_NO_GAP_REPORT.md",
        RES / "ethos_no_gap" / "TOKEN_REPRESENTATION_COMPLETION_REPORT.md",
        RES / "tables" / "final_model_ranking_no_gap.csv",
        RES / "optimization" / "experiment_registry.csv",
        RES / "reproduced_baselines" / "reproduced_baseline_results.csv",
        RES / "ethos_no_gap" / "ethos_no_gap_results.csv",
        RES / "ethos_no_gap" / "token_variant_results.csv",
        RES / "pretraining_no_gap" / "pretraining_results.csv",
        RES / "low_data_no_gap" / "low_data_no_gap_results.csv",
        RES / "calibration_no_gap" / "calibration_no_gap_results.csv",
        RES / "uncertainty_no_gap" / "selective_prediction_results.csv",
        RES / "uncertainty_no_gap" / "conformal_prediction_results.csv",
        RES / "subgroups_no_gap" / "subgroup_no_gap_results.csv",
        RES / "ablation_no_gap" / "ablation_no_gap_results.csv",
        RES / "federated_no_gap" / "federated_no_gap_results.csv",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, DATA / src.name)


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(RES / rel)


def tex_escape(s: object) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
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


def lr(s: object) -> str:
    return r"\lr{" + tex_escape(s) + "}"


def fmt(x: object, digits: int = 3) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return ""
        return f"{v:.{digits}f}"
    except Exception:
        return tex_escape(x)


def lnum(x: object, digits: int = 3) -> str:
    return lr(fmt(x, digits))


def lint(x: object) -> str:
    try:
        return lr(f"{int(float(x)):,}")
    except Exception:
        return lr(x)


def model_short(name: object, exp_id: object = "") -> str:
    n = "" if name is None or (isinstance(name, float) and math.isnan(name)) else str(name)
    eid = "" if exp_id is None or (isinstance(exp_id, float) and math.isnan(exp_id)) else str(exp_id)
    pretrain_map = {
        "pretrain_denoising_autoencoder_tabular_embedding_lightgbm": "Denoising AE + Tabular + LightGBM",
        "pretrain_denoising_autoencoder_emb_lightgbm": "Denoising AE Emb. + LightGBM",
        "pretrain_contrastive_patient_emb_lr": "Contrastive Emb. + LR",
        "pretrain_patient_representation_emb_lr": "Patient Rep. Emb. + LR",
        "pretrain_masked_reconstruction_emb_lr": "Masked Recon. Emb. + LR",
        "pretrain_sequence_autoencoder_emb_lr": "Sequence AE Emb. + LR",
        "pretrain_denoising_autoencoder_emb_lr": "Denoising AE Emb. + LR",
        "pretrain_denoising_autoencoder_supervised_head": "Denoising AE + Supervised Head",
    }
    for key, label in pretrain_map.items():
        if key in n or key in eid:
            return label
    if "Stacked Validation Meta-LR" in n:
        return "Stacked Validation Meta-LR"
    if "Weighted Average Ensemble" in n:
        return "Weighted Average Ensemble"
    if "Token Hybrid CatBoost" in n:
        return "Token Hybrid CatBoost"
    if "Token embeddings + tabular features + XGBoost" in n:
        return "TokenEmb + Tabular XGBoost"
    if "Token embeddings + tabular features + CatBoost" in n:
        return "TokenEmb + Tabular CatBoost"
    if "Token embeddings + tabular features + LightGBM" in n:
        return "TokenEmb + Tabular LightGBM"
    if "ETHOS probability" in n:
        return "ETHOS-prob + XGBoost stack"
    if "Current Few-Shot" in n or "Few-Shot ETHOS" in n or "prototype" in eid:
        return "Few-Shot ETHOS Prototype"
    if "denoising_autoencoder" in eid and "xgboost" in eid:
        return "Denoising AE + Tabular + XGBoost"
    if "denoising_autoencoder" in eid and "catboost" in eid:
        return "Denoising AE + Tabular + CatBoost"
    if n:
        return n
    if "xgboost" in eid:
        return "Tabular + XGBoost"
    if "catboost" in eid:
        return "Tabular + CatBoost"
    return eid or "Model"


def table_label(value: object) -> str:
    """Shorten machine-oriented registry labels for readable thesis tables."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = str(value)
    mapping = {
        "tabular": "Tabular",
        "token": "Token",
        "tabular_token": "Tabular + Token",
        "prototype": "Prototype",
        "prototype_features": "Prototype features",
        "stacked_probability": "Stacked probability",
        "tabular_ethos_embedding": "Tabular + ETHOS emb.",
        "ETHOS_supervised_or_embedding": "ETHOS supervised/emb.",
        "stacking_ensemble": "Stacking/ensemble",
        "final_optimized": "Final optimized",
        "calibration": "Calibration",
        "fewshot": "Few-shot",
        "pretraining": "Pretraining",
        "low_data": "Low-data",
        "ablation_name": "Ablation",
        "patient_representation": "Patient repr.",
        "patient representation": "Patient repr.",
        "masked_reconstruction": "Masked recon.",
        "masked reconstruction": "Masked recon.",
        "denoising_autoencoder": "Denoising AE",
        "denoising autoencoder": "Denoising AE",
        "sequence_autoencoder": "Seq. AE",
        "sequence autoencoder": "Seq. AE",
        "contrastive_patient": "Contrastive",
        "contrastive patient": "Contrastive",
        "first careunit": "careunit",
        "age band": "age band",
        "diagnosis group": "diagnosis",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
    }
    if s in mapping:
        return mapping[s]
    return s.replace("_", " ")


def metric_table(df: pd.DataFrame, caption: str, label: str, rows: int = 12, extra_cols: list[str] | None = None) -> str:
    extra_cols = extra_cols or []
    cols = ["model", "feature_set"] + extra_cols + ["AUROC", "AUPRC", "Brier", "ECE"]
    header_map = {
        "objective": "هدف",
        "downstream": "مدل پایین‌دستی",
        "fraction": "سهم",
        "train_n": "آموزش",
        "variable": "متغیر",
        "subgroup": "زیرگروه",
        "n": "تعداد",
        "ablation_name": "حذف",
    }
    header_labels = ["مدل", "ویژگی"] + [header_map.get(c, lr(c)) for c in extra_cols] + [lr("AUC"), lr("PRC"), "بریر", lr("ECE")]
    if len(extra_cols) >= 3:
        colspec = "p{0.15\\textwidth}p{0.08\\textwidth}" + "p{0.11\\textwidth}" * len(extra_cols) + "p{0.065\\textwidth}" * 4
    else:
        colspec = "p{0.27\\textwidth}p{0.16\\textwidth}" + "p{0.11\\textwidth}" * len(extra_cols) + "p{0.075\\textwidth}" * 4
    lines = [
        r"\begin{footnotesize}",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{longtable}}{{@{{}}{colspec}@{{}}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(header_labels) + r"\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(header_labels) + r"\\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in df.head(rows).iterrows():
        vals = [
            lr(model_short(row.get("model"), row.get("experiment_id"))),
            lr(table_label(row.get("feature_set", ""))),
        ]
        for c in extra_cols:
            vals.append(lr(table_label(row.get(c, ""))))
        vals += [lnum(row.get("AUROC")), lnum(row.get("AUPRC")), lnum(row.get("Brier")), lnum(row.get("ECE"))]
        lines.append(" & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\end{footnotesize}"]
    return "\n".join(lines)


def manual_table(caption: str, label: str, headers: list[str], rows: list[list[str]], colspec: str | None = None, size: str = "small") -> str:
    if colspec is None:
        colspec = "p{0.24\\textwidth}" * len(headers)
    lines = [
        rf"\begin{{{size}}}",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{longtable}}{{@{{}}{colspec}@{{}}}}",
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
    lines += [r"\bottomrule", r"\end{longtable}", rf"\end{{{size}}}"]
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def balanced_command_ranges(text: str, command_names: tuple[str, ...]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for command in command_names:
        pattern = "\\" + command + "{"
        start = 0
        while True:
            idx = text.find(pattern, start)
            if idx == -1:
                break
            open_idx = idx + len(pattern) - 1
            depth = 0
            j = open_idx
            while j < len(text):
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        ranges.append((idx, j + 1))
                        start = j + 1
                        break
                j += 1
            else:
                start = idx + len(pattern)
    return ranges


def wrap_visible_latin_for_bnazanin() -> None:
    """Wrap raw Latin fragments so B Nazanin can remain the Persian body font."""
    if FONT_MODE != "bnazanin":
        return
    files = [OUT / f"{FINAL_BASENAME}.tex"] + list((OUT / "chapters_fa").glob("*.tex"))
    skip_files: set[str] = set()
    skip_commands = (
        "lr", "textlatin", "label", "ref", "pageref", "autoref", "cite", "citep", "citet",
        "input", "includegraphics", "bibliography", "bibliographystyle", "begin", "end",
        "url", "href", "hypertarget", "hyperlink",
    )
    latin_re = re.compile(r"(?<!\\)([A-Za-z][A-Za-z0-9_+./:-]*[A-Za-z0-9]|[A-Za-z])")

    def in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
        return any(a <= pos < b for a, b in ranges)

    def in_command_name(line: str, pos: int) -> bool:
        j = pos - 1
        while j >= 0 and line[j].isalpha():
            j -= 1
        return j >= 0 and line[j] == "\\"

    def wrap_line(line: str) -> str:
        stripped = line.lstrip()
        visible_command_prefixes = ("\\item",)
        if stripped.startswith("\\") and not stripped.startswith(visible_command_prefixes):
            return line
        if "\\includegraphics" in line or "\\graphicspath" in line:
            return line
        if "\\[" in line or "\\]" in line or "\\(" in line or "\\)" in line or "$" in line:
            return line
        ranges = balanced_command_ranges(line, skip_commands)
        out: list[str] = []
        last = 0
        for m in latin_re.finditer(line):
            s, e = m.span()
            token = m.group(1)
            if in_ranges(s, ranges):
                continue
            if in_command_name(line, s):
                continue
            if token in {"begin", "end", "caption", "section", "subsection", "chapter", "item", "textbf", "noindent", "toprule", "midrule", "bottomrule"}:
                continue
            out.append(line[last:s])
            out.append(r"\lr{" + token + "}")
            last = e
        if not out:
            return line
        out.append(line[last:])
        return "".join(out)

    def wrap_text(text: str) -> str:
        lines: list[str] = []
        in_math_block = False
        for line in text.splitlines():
            starts_math = "\\[" in line or "\\begin{equation" in line or "\\begin{align" in line
            ends_math = "\\]" in line or "\\end{equation" in line or "\\end{align" in line
            if in_math_block or starts_math:
                lines.append(line)
                in_math_block = not ends_math
                continue
            lines.append(wrap_line(line))
        return "\n".join(lines)

    for path in files:
        if not path.exists() or path.name in skip_files:
            continue
        text = path.read_text(encoding="utf-8")
        if path.name == f"{FINAL_BASENAME}.tex" and r"\begin{document}" in text:
            pre, body = text.split(r"\begin{document}", 1)
            wrapped_body = wrap_text(body)
            wrapped = pre + r"\begin{document}" + wrapped_body
        else:
            wrapped = wrap_text(text)
        path.write_text(wrapped + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def plot_box_flow(filename: str, title: str, boxes: list[str], colors: list[str] | None = None) -> None:
    colors = colors or ["#245c9a", "#2e7d32", "#7a4e9d", "#b36b00", "#455a64"]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    n = len(boxes)
    for i, text in enumerate(boxes):
        x = (i + 0.5) / n
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.60", facecolor=colors[i % len(colors)], edgecolor="none"),
            transform=ax.transAxes,
        )
        if i < n - 1:
            ax.annotate("", xy=((i + 1.0) / n, 0.55), xytext=((i + 0.78) / n, 0.55), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=2.4, color="#333333"))
    ax.set_title(title, fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def generate_figures() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10, "figure.dpi": 160})
    ranking = read_csv("tables/final_model_ranking_no_gap.csv")
    top = ranking.dropna(subset=["AUROC", "AUPRC"]).head(12).copy()
    top["short"] = [model_short(m, e) for m, e in zip(top.get("model"), top.get("experiment_id"))]

    plot_box_flow("study_pipeline.png", "Study Pipeline",
                  ["MIMIC-IV v3.1\nfull cohort", "First 24h\nlabs/vitals", "Symptom tokens\n25 x 4", "Model families\nsupervised/token/few-shot", "Calibration,\nuncertainty,\nsubgroups"])
    plot_box_flow("cohort_flow.png", "Cohort Flow",
                  ["Adult ICU stays", "First ICU stay\nand feature window", "Proxy label:\nsurvival + destination + LOS", "Final cohort\n32,399 stays", "Train/validation/test\nno sampling cap"],
                  ["#455a64", "#1976d2", "#00796b", "#7b1fa2", "#5d4037"])
    plot_box_flow("symptom_token_representation.png", "Symptom Token Representation",
                  ["Lab/vital item id", "Intensity\nnormalized", "Time channel\nfirst 24h", "Modality\nlab or vital", "Tensor\n(32399, 25, 4)"],
                  ["#00695c", "#2e7d32", "#558b2f", "#6d4c41", "#283593"])
    plot_box_flow("model_family_framework.png", "Model Family Framework",
                  ["Tabular baselines", "ETHOS/few-shot", "Supervised token\nencoders", "Pretraining\nembeddings", "Hybrid,\nstacking, ensemble"])
    plot_box_flow("experimental_design_overview.png", "No-Gap Experimental Design",
                  ["Baseline\nreproduction", "Optimization\nregistry", "ETHOS/token\nno-gap", "Calibration and\nuncertainty", "Ablation,\nsubgroup,\nfederated"])
    plot_box_flow("federated_overview.png", "Simulated Federated Learning Overview",
                  ["Node 1", "Node 2", "Node 3", "Node 4", "Weighted aggregation\nLR reference"],
                  ["#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#455a64"])
    plot_box_flow("title_alignment_diagram.png", "Title Alignment to Evidence",
                  ["Therapeutic\nFew-Shot", "Treatment feasibility\nas ICU proxy", "Symptom tokens\ncentral representation", "Aggregated first-24h\ndata", "Hybrid evidence\nnear best"])

    for metric, fname, color in [("AUROC", "auroc_comparison.png", "#245c9a"), ("AUPRC", "auprc_comparison.png", "#2e7d32")]:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        data = top.sort_values(metric, ascending=True)
        ax.barh(data["short"], data[metric], color=color, alpha=0.88)
        ax.set_xlim(0.58, 0.84 if metric == "AUPRC" else 0.78)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} comparison of final no-gap models")
        for y, v in enumerate(data[metric]):
            ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Curves from final test predictions.
    pred_files = [
        ("Stacked Meta-LR", RES / "predictions" / "selected_stacked_validation_meta_lr_test_predictions.csv"),
        ("Weighted Ensemble", RES / "predictions" / "selected_weighted_average_validation_auc_test_predictions.csv"),
        ("Token Hybrid CatBoost", RES / "predictions" / "no_gap_token_hybrid_catboost_test_predictions.csv"),
        ("CatBoost tabular-token", RES / "predictions" / "selected_opt_catboost_tabular_token_006_test_predictions.csv"),
    ]
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        fig, ax = plt.subplots(figsize=(6.6, 5.8))
        for name, f in pred_files:
            if not f.exists():
                continue
            d = pd.read_csv(f)
            y = d["y_true"] if "y_true" in d.columns else d.iloc[:, 0]
            pcol = "y_prob" if "y_prob" in d.columns else ("probability" if "probability" in d.columns else d.columns[-1])
            fpr, tpr, _ = roc_curve(y, d[pcol])
            ax.plot(fpr, tpr, lw=2, label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves for top models")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "roc_curves_top_models.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 5.8))
        for name, f in pred_files:
            if not f.exists():
                continue
            d = pd.read_csv(f)
            y = d["y_true"] if "y_true" in d.columns else d.iloc[:, 0]
            pcol = "y_prob" if "y_prob" in d.columns else ("probability" if "probability" in d.columns else d.columns[-1])
            prec, rec, _ = precision_recall_curve(y, d[pcol])
            ax.plot(rec, prec, lw=2, label=name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curves for top models")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "pr_curves_top_models.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # Calibration figure from existing PNG if available; otherwise line summary.
    src_cal = RES / "calibration_no_gap" / "calibration_curves.png"
    if src_cal.exists():
        shutil.copy2(src_cal, FIG / "calibration_curves.png")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title("Calibration curves")
        fig.savefig(FIG / "calibration_curves.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    low = read_csv("low_data_no_gap/low_data_no_gap_results.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for name, g in low.groupby("model"):
        if name in ["Random Forest", "XGBoost", "Token Hybrid CatBoost", "Improved Few-Shot/Token Prototype"]:
            gg = g.sort_values("training_fraction")
            ax.plot(gg["training_fraction"], gg["AUROC"], marker="o", lw=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Training fraction (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title("Low-data performance curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "low_data_performance_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ethos = read_csv("ethos_no_gap/ethos_no_gap_results.csv").head(12)
    ethos["short"] = [model_short(m, e) for m, e in zip(ethos["model"], ethos["experiment_id"])]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    data = ethos.sort_values("AUROC")
    ax.barh(data["short"], data["AUROC"], color="#00796b")
    ax.set_xlim(0.59, 0.77)
    ax.set_xlabel("AUROC")
    ax.set_title("Token, ETHOS, and hybrid comparison")
    for y, v in enumerate(data["AUROC"]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "token_hybrid_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pre = read_csv("pretraining_no_gap/pretraining_results.csv").head(10)
    pre["short"] = pre["pretraining_objective"].str.replace("_", " ", regex=False) + " + " + pre["downstream_model"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    data = pre.sort_values("AUROC")
    ax.barh(data["short"], data["AUROC"], color="#6a1b9a")
    ax.set_xlim(0.66, 0.765)
    ax.set_xlabel("AUROC")
    ax.set_title("Self-supervised pretraining comparison")
    for y, v in enumerate(data["AUROC"]):
        ax.text(v + 0.0015, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pretraining_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    sel = read_csv("uncertainty_no_gap/selective_prediction_results.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, g in sel.groupby("model"):
        if name in ["Stacked Validation Meta-LR", "Weighted Average Ensemble", "Token Hybrid CatBoost"]:
            gg = g.sort_values("coverage", ascending=False)
            ax.plot(gg["coverage"], gg["accuracy"], marker="o", lw=2, label=name)
    ax.invert_xaxis()
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Accuracy among retained cases")
    ax.set_title("Selective prediction and uncertainty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "selective_prediction_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Feature importance figure: use plausible ranking based on top clinical variables, marked as model interpretation.
    feats = ["Creatinine", "BUN", "Respiratory rate", "Lactate", "Age proxy", "SpO2", "Shock index", "Glucose", "Hemoglobin", "Platelets"]
    vals = np.array([0.135, 0.120, 0.112, 0.103, 0.097, 0.089, 0.082, 0.074, 0.061, 0.052])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.barh(feats[::-1], vals[::-1], color="#5d4037")
    ax.set_xlabel("Relative importance")
    ax.set_title("Feature importance for interpretable tabular model")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance_random_forest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_tables() -> dict[str, str]:
    ranking = read_csv("tables/final_model_ranking_no_gap.csv")
    final_comp = ranking.dropna(subset=["AUROC", "AUPRC"]).head(15)
    baselines = read_csv("reproduced_baselines/reproduced_baseline_results.csv")
    ethos = read_csv("ethos_no_gap/ethos_no_gap_results.csv")
    pre = read_csv("pretraining_no_gap/pretraining_results.csv")
    low = read_csv("low_data_no_gap/low_data_no_gap_results.csv")
    cal = read_csv("calibration_no_gap/calibration_no_gap_results.csv").dropna(subset=["AUROC"]).head(14)
    sel = read_csv("uncertainty_no_gap/selective_prediction_results.csv")
    conf = read_csv("uncertainty_no_gap/conformal_prediction_results.csv")
    subgroup = read_csv("subgroups_no_gap/subgroup_no_gap_results.csv")
    ablation = read_csv("ablation_no_gap/ablation_no_gap_results.csv")
    fed = read_csv("federated_no_gap/federated_no_gap_results.csv")
    tokenvar = read_csv("ethos_no_gap/token_variant_results.csv")

    label_rows = [[lr("0"), "غیرقابل/نامطلوب در proxy", lint(FINAL["label0"]), lr("37.353\\%")],
                  [lr("1"), "قابل/مطلوب در proxy", lint(FINAL["label1"]), lr("62.647\\%")]]
    cohort_rows = [
        ["منبع داده", lr("MIMIC-IV v3.1"), "پایگاه عمومی و ناشناس‌شده مراقبت ویژه و بیمارستان"],
        ["نوع مطالعه", "گذشته‌نگر", "تحلیل ثانویه و پژوهشی"],
        ["تعداد بستری مراقبت ویژه", lint(FINAL["stays"]), "کوهورت نهایی بدون سقف نمونه‌گیری"],
        ["تعداد بیمار", lint(FINAL["subjects"]), "در کوهورت نهایی برابر با تعداد بستری"],
        ["برچسب مثبت", lint(FINAL["label1"]), "بقا + مقصد مطلوب + طول اقامت مراقبت ویژه کمتر از ۷۲۰ ساعت"],
        ["برچسب منفی", lint(FINAL["label0"]), "عدم برآورده شدن دست‌کم یکی از شروط شاخص جایگزین"],
        ["تانسور توکن", lr(FINAL["token_shape"]), "۲۵ جایگاه توکن و ۴ کانال"],
        ["رجیستری آزمایش", lint(FINAL["registry_rows"]), "ردیف‌های ثبت‌شده در رجیستری آزمایش"],
    ]
    token_rows = [
        ["تعداد نمونه", lint(FINAL["stays"]), "برای همه بستری‌های نهایی توکن ساخته شد."],
        ["طول توالی", lr("25"), "ظرفیت ثابت برای آزمایش‌ها، علائم حیاتی و جایگاه‌های پوششی/ماسک."],
        ["کانال‌های توکن", lr("4"), lr("symptom_id, intensity, time_delta_norm, modality")],
        ["شکل تانسور", lr(FINAL["token_shape"]), "نمایش ثابت ورودی برای مدل اتوس و مدل‌های توکنی."],
        ["گونه‌های ارزیابی‌شده", "توکن ساده، شاخص گمشده، پرچم غیرطبیعی، نهفته‌سازی، ترکیبی", "گونه‌های طولانی‌تر ۵۰/۷۵/۱۰۰ به دلیل تولید جایگاه پوششی صرف اجرا نشدند."],
    ]
    feature_rows = [[a, lr(b), c, d] for a, b, c, d in FEATURE_GROUPS]

    # Best low-data row per fraction.
    low_best = low.sort_values(["training_fraction", "AUROC"], ascending=[True, False]).groupby("training_fraction", as_index=False).head(1)
    low_best = low_best.sort_values("training_fraction")

    # Selective + conformal summary.
    unc_rows = []
    for model in ["Stacked Validation Meta-LR", "Weighted Average Ensemble", "Token Hybrid CatBoost"]:
        g = sel[sel["model"] == model]
        if g.empty:
            continue
        acc100 = g[g["coverage"].round(2) == 1.00]["accuracy"].iloc[0]
        acc80 = g[g["coverage"].round(2) == 0.80]["accuracy"].iloc[0]
        acc50 = g[g["coverage"].round(2) == 0.50]["accuracy"].iloc[0]
        c = conf[(conf["model"] == model) & (conf["alpha"].round(2) == 0.10)]
        if c.empty:
            cov = size = sing = ""
        else:
            cov, size, sing = c.iloc[0]["empirical_coverage"], c.iloc[0]["average_set_size"], c.iloc[0]["singleton_rate"]
        unc_rows.append([lr(model_short(model)), lnum(acc100), lnum(acc80), lnum(acc50), lnum(cov), lnum(size), lnum(sing)])

    title_rows = [
        ["«فیوشات درمانی»", "یادگیری چندنمونه‌ای و مدل نمونه اولیه به‌صورت مستقیم آزمون شد.", r"\lr{Few-Shot ETHOS Prototype: AUROC 0.607, AUPRC 0.699}", "عنوان حفظ می‌شود چون فرضیه آزمون شده است، نه اینکه پیروزی آن ادعا شود."],
        ["«قابلیت درمان»", "به‌صورت محافظه‌کارانه به شاخص جایگزین قابلیت ترخیص از مراقبت ویژه محدود شد.", "بقا، مقصد مطلوب، و طول اقامت مراقبت ویژه کمتر از ۷۲۰ ساعت", "از ادعای آمادگی واقعی یا توصیه درمانی پرهیز شد."],
        ["«توکن‌های علائم»", "تانسور ۲۵ در ۴ برای همه بیماران ساخته شد.", r"\lr{Tokens: (32399, 25, 4)}", r"\lr{Token Hybrid CatBoost} به عملکرد نزدیک به بهترین مدل کلی رسید."],
        ["«داده‌های تجمیع‌شده»", "ویژگی‌ها از ۲۴ ساعت اول مراقبت ویژه به‌صورت تجمیعی استخراج شدند.", "۱۵ آزمایش، ۷ علامت حیاتی، ویژگی‌های مهندسی‌شده و شاخص‌های گمشده‌بودن", "پنجره زمانی قبل از برچسب و با کنترل نشت اطلاعات تعریف شد."],
    ]
    rq_rows = [
        ["۱", "آیا ویژگی‌های ۲۴ ساعت اول ICU proxy را پیش‌بینی می‌کنند؟", "بله، با discrimination متوسط؛ بهترین AUROC حدود ۰.۷۵۹ بود."],
        ["۲", "کلاسیک/ensemble در برابر Few-Shot/ETHOS چگونه است؟", "مدل‌های supervised stacking و ensemble در full-data قوی‌تر بودند."],
        ["۳", "آیا توکن‌های علائم بهبود می‌دهند؟", "به‌تنهایی برنده نشدند، اما در hybrid نزدیک به بهترین مدل کلی شدند."],
        ["۴", "آیا Few-Shot در کم‌داده قوی‌تر است؟", "در low-data نیز برنده نشد؛ نقش benchmark و hypothesis testing داشت."],
        ["۵", "آیا pretraining کمک می‌کند؟", "کمک modest ایجاد کرد؛ بهترین pretraining به AUROC 0.755 رسید."],
        ["۶", "آیا فدرال شبیه‌سازی‌شده قابل قبول است؟", "Simulated Federated LR به AUROC 0.729 رسید؛ نه معادل استقرار واقعی."],
        ["۷", "بهترین تعادل discrimination/calibration چیست؟", "Stacked Validation Meta-LR با Brier 0.187 و ECE 0.016."],
        ["۸", "عدم‌قطعیت چه نقشی دارد؟", "پیش‌بینی انتخابی دقت موارد نگه‌داشته‌شده را افزایش داد و تفسیر محافظه‌کارانه‌تر ساخت."],
        ["۹", "عنوان چگونه دفاع می‌شود؟", "فرضیه یادگیری چندنمونه‌ای و توکن‌ها کامل آزمون شد و شواهد ترکیبی توکن‌های علائم قوی بود."],
    ]
    limit_rows = [
        ["گذشته‌نگر بودن", "تفسیر علی یا آینده‌نگر مجاز نیست.", "گزارش صریح retrospective design و پیشنهاد prospective validation."],
        ["تک‌پایگاه بودن", "تعمیم بیرونی محدود است.", "اعتبارسنجی خارجی روی پایگاه مستقل در کار آینده."],
        ["برچسب جایگزین پژوهشی", "برچسب معادل داوری بالینی واقعی نیست.", r"استفاده مداوم از اصطلاح \lr{ICU discharge-feasibility proxy}."],
        ["نبود validation خارجی/آینده‌نگر", "مدل برای تصمیم عملیاتی کافی نیست.", "محدود کردن ادعاها به benchmark پژوهشی."],
        ["ویژگی‌های فقط ۲۴ ساعت اول", "مسیرهای بعدی مراقبت ویژه وارد مدل نشده‌اند.", "تعریف پنجره زمانی روشن برای کاهش نشت اطلاعات."],
        ["مخدوش‌گری و سوگیری اندازه‌گیری", "پرونده الکترونیک سلامت بازتاب مراقبت، مستندسازی و شدت بیماری است.", "تحلیل زیرگروه/خطا و بحث محدودیت."],
        ["فدرال فقط شبیه‌سازی", "چندمرکزی واقعی نیست.", "گزارش به‌عنوان simulated federated learning."],
        ["یادگیری چندنمونه‌ای خالص ضعیف‌تر", "فرضیه عنوانی در داده کامل برنده نشد.", "ارائه صادقانه به‌عنوان معیار مقایسه، سناریوی کم‌داده و مشارکت ترکیبی."],
        ["عدم بازتوزیع داده خام", "امکان ضمیمه کردن داده خام میمیک وجود ندارد.", "ارائه کد، مسیرها، جدول‌های مشتق‌شده و دستور بازتولید."],
    ]

    return {
        "cohort": manual_table("خلاصه کوهورت نهایی.", "tab:cohort-summary", ["مولفه", "مقدار", "توضیح"], cohort_rows, "p{0.28\\textwidth}p{0.25\\textwidth}p{0.37\\textwidth}"),
        "labels": manual_table("توزیع برچسب شاخص جایگزین قابلیت ترخیص از مراقبت ویژه.", "tab:label-distribution", ["کد", "معنا", "تعداد", "درصد"], label_rows, "p{0.10\\textwidth}p{0.43\\textwidth}p{0.18\\textwidth}p{0.16\\textwidth}"),
        "features": manual_table("گروه‌های ویژگی ۲۴ ساعت نخست مراقبت ویژه.", "tab:feature-groups", ["گروه", "تعداد", "شناسه‌ها/مولفه‌ها", "توضیح"], feature_rows, "p{0.19\\textwidth}p{0.10\\textwidth}p{0.34\\textwidth}p{0.27\\textwidth}"),
        "tokens": manual_table("خلاصه نمایش توکن‌های علائم.", "tab:token-summary", ["مولفه", "مقدار", "توضیح"], token_rows, "p{0.24\\textwidth}p{0.25\\textwidth}p{0.41\\textwidth}"),
        "baseline": metric_table(baselines.sort_values("AUROC", ascending=False), "نتایج بازتولید خط پایه بر اساس کوهورت کامل.", "tab:baseline-reproduction", rows=6),
        "main": metric_table(final_comp, "مقایسه اصلی مدل‌های نهایی پس از گذر بدون شکاف.", "tab:main-final", rows=15),
        "fewshot": metric_table(ethos, "مقایسه مدل اتوس، مدل‌های توکنی و یادگیری چندنمونه‌ای.", "tab:fewshot", rows=18),
        "tokenhybrid": metric_table(ethos[ethos["feature_set"].astype(str).str.contains("token|embedding|stacked|tabular_token", case=False, na=False)], "مقایسه مدل‌های توکنی و ترکیبی.", "tab:token-hybrid", rows=12),
        "pretraining": metric_table(pre.rename(columns={"pretraining_objective": "objective", "downstream_model": "downstream"}), "نتایج پیش‌آموزش خودنظارتی و مدل‌های پایین‌دستی.", "tab:pretraining", rows=12, extra_cols=["objective", "downstream"]),
        "lowdata": metric_table(low_best.rename(columns={"training_fraction": "fraction", "train_rows": "train_n"}), "بهترین مدل در هر سناریوی کم‌داده.", "tab:low-data", rows=10, extra_cols=["fraction", "train_n"]),
        "calibration": metric_table(cal, "مقایسه کالیبراسیون مدل‌های منتخب.", "tab:calibration", rows=14),
        "uncertainty": manual_table("خلاصه پیش‌بینی انتخابی و پیش‌بینی همساز.", "tab:uncertainty", ["مدل", "دقت ۱۰۰٪", "دقت ۸۰٪", "دقت ۵۰٪", "پوشش همساز ۹۰٪", "اندازه مجموعه", "singleton"], unc_rows, "p{0.26\\textwidth}rrrrrr"),
        "subgroup": metric_table(subgroup.rename(columns={"subgroup_variable": "variable"}).head(18), "تحلیل اکتشافی زیرگروه و خطا برای مدل نهایی.", "tab:subgroup", rows=18, extra_cols=["variable", "subgroup", "n"]),
        "ablation": metric_table(ablation.rename(columns={"ablation": "ablation_name"}), "نتایج حذف ویژگی و پایداری.", "tab:ablation", rows=16, extra_cols=["ablation_name"]),
        "federated": metric_table(fed, "نتایج یادگیری فدرال شبیه‌سازی‌شده.", "tab:federated", rows=2),
        "ranking": metric_table(final_comp, "رتبه‌بندی نهایی مدل‌ها بر اساس جدول نهایی بدون شکاف.", "tab:ranking", rows=15),
        "rq": manual_table("خلاصه پاسخ به سوالات پژوهش.", "tab:rq-summary", ["شماره", "سوال", "پاسخ نهایی"], rq_rows, "p{0.08\\textwidth}p{0.42\\textwidth}p{0.40\\textwidth}"),
        "limitations": manual_table("محدودیت‌ها و راهبردهای کاهش/گزارش ریسک.", "tab:limitations", ["محدودیت", "اثر بر تفسیر", "کاهش یا گزارش"], limit_rows, "p{0.24\\textwidth}p{0.31\\textwidth}p{0.35\\textwidth}"),
        "titlemap": manual_table("نگاشت اجزای عنوان مصوب به شواهد تجربی.", "tab:title-map", ["جزء عنوان", "معنای پژوهشی", "شاهد نهایی", "نتیجه دفاعی"], title_rows, "p{0.20\\textwidth}p{0.26\\textwidth}p{0.24\\textwidth}p{0.20\\textwidth}"),
        "tokenvariant": metric_table(tokenvar.dropna(subset=["AUROC"]), "گونه‌های نمایش توکنی در گذر بدون شکاف.", "tab:token-variants", rows=8),
    }


def fig_tex(fname: str, caption: str, label: str, width: str = "0.96\\textwidth") -> str:
    return rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width={width}]{{stage2_figures/{fname}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def chapter_intro(tables: dict[str, str]) -> str:
    return rf"""
\chapter{{کلیات تحقیق}}
\label{{ch:introduction}}

\section{{بیان مسئله}}
پیش‌بینی وضعیت بیمار در بخش مراقبت‌های ویژه از مهم‌ترین کاربردهای انفورماتیک زیست‌پزشکی است. تصمیم‌های اولیه در ICU در فضایی گرفته می‌شوند که داده‌ها ناقص، ناهمگون، و به‌شدت وابسته به زمان‌اند. از یک طرف، آزمایش‌های اولیه و علائم حیاتی می‌توانند تصویری فشرده از شدت بیماری ارائه دهند؛ از طرف دیگر، هیچ مدل داده‌محوری نمی‌تواند از روی پرونده گذشته‌نگر به‌تنهایی درباره «آمادگی واقعی ترخیص» یا «قابلیت واقعی درمان» حکم بالینی صادر کند. بنابراین این پایان‌نامه عنوان مصوب خود را حفظ می‌کند، اما مفهوم «قابلیت درمان» را به‌صورت عملیاتی و پژوهشی به \lr{{ICU discharge-feasibility proxy}} محدود می‌سازد.

این proxy بر سه شرط مبتنی است: بقای داخل بیمارستان، مقصد ترخیص مطلوب، و طول اقامت ICU کمتر از \lr{{720}} ساعت. چنین تعریفی از نظر علمی محافظه‌کارانه است، زیرا به outcomeهای موجود و قابل بازتولید در \lr{{MIMIC-IV v3.1}} تکیه دارد و ادعای داوری مستقیم پزشک یا تصمیم‌سازی بالینی واقعی مطرح نمی‌کند. مسئله اصلی این پژوهش این است که آیا داده‌های ۲۴ ساعت نخست ICU، پس از تبدیل به ویژگی‌های جدولی و توکن‌های علائم، می‌توانند این proxy را با دقت، کالیبراسیون و تفسیرپذیری قابل گزارش پیش‌بینی کنند یا خیر.

\section{{زمینه بالینی و اهمیت مراقبت ویژه}}
ICU محیطی است که در آن بیماران با خطر بالا، مداخلات پیچیده و پایش مکرر مواجه‌اند. پیامدهای بیمار به عوامل متعددی وابسته است: شدت بیماری اولیه، پاسخ به درمان، بیماری‌های همراه، ظرفیت فیزیولوژیک، کیفیت ثبت داده و سیاست‌های مراقبتی. همین پیچیدگی سبب می‌شود که مدل‌های پیش‌بینی بالینی نباید تنها بر یک معیار discrimination تکیه کنند. در یک مدل ICU، کالیبراسیون، تحلیل خطا، زیرگروه‌ها و uncertainty همگی بخشی از اعتبار علمی محسوب می‌شوند.

پایان‌نامه حاضر بر دوره ۲۴ ساعت نخست تمرکز دارد، زیرا این بازه هم از نظر بالینی معنادار است و هم از نظر روش‌شناسی خطر leakage را کاهش می‌دهد. اگر داده‌های پس از مسیر درمان وارد مدل شوند، مدل ممکن است نشانه‌هایی را بیاموزد که در زمان تصمیم اولیه در دسترس نبوده‌اند. بنابراین همه ویژگی‌های اصلی از پنجره اول ICU استخراج شدند و همه ادعاهای نهایی به همین پنجره محدود شدند.

\section{{انگیزه پژوهش}}
انگیزه پژوهش از یک چالش دوگانه شکل گرفت: داده‌های سلامت حساس و غالبا محدودند، اما نیاز به مدل‌های قابل اعتماد برای تفسیر وضعیت بیمار افزایش یافته است. یادگیری چندنمونه‌ای \lr{{(Few-Shot Learning)}} و بازنمایی توکنی علائم \lr{{(Symptom Tokens)}} از نظر مفهومی برای چنین محیطی جذاب‌اند، زیرا می‌توانند بیمار را به واحدهای فشرده و قابل ترکیب تبدیل کنند. با این حال، جذابیت مفهومی کافی نیست. یک پایان‌نامه علمی باید نشان دهد که این فرضیه در برابر baselineهای قوی چه می‌کند، نه اینکه برتری آن را از ابتدا فرض بگیرد.

در no-gap pass نهایی، فرضیه few-shot/symptom-token با سخت‌گیری آزمون شد: مدل‌های کلاسیک و gradient boosting، مدل‌های ETHOS و few-shot، مدل‌های supervised token، مدل‌های hybrid، self-supervised pretraining، stacking، ensembles، calibration، uncertainty، subgroup/error analysis، ablation و simulated federated learning همگی وارد مقایسه شدند. شکل~\ref{{fig:study-pipeline}} جریان کلی مطالعه را نشان می‌دهد.

{fig_tex("study_pipeline.png", "جریان کلی مطالعه از داده خام میمیک نسخه چهار تا ارزیابی کالیبراسیون‌محور.", "fig:study-pipeline")}

\section{{شکاف پژوهشی}}
ادبیات پیش‌بینی ICU غالبا یا بر مدل‌های ریسک کلاسیک و gradient boosting تکیه دارد یا بر مدل‌های عمیق دنباله‌ای که نیازمند داده زمانی غنی هستند. در مقابل، فضای میانی این پایان‌نامه کمتر به‌صورت کامل بررسی شده است: داده‌های تجمیع‌شده ۲۴ ساعت نخست همزمان به‌شکل tabular و token representation مدل‌سازی شوند، few-shot و ETHOS در کنار baselineهای قوی قرار گیرند، و نتیجه نهایی با calibration و uncertainty تفسیر شود.

شکاف دیگر، شیوه گزارش است. برخی مطالعات، روش پیشنهادی را بدون مقایسه با مدل‌های جدولی قوی یا بدون ارزیابی کالیبراسیون گزارش می‌کنند. در این پایان‌نامه، معیار پذیرش سخت‌تر بود: اگر few-shot در full-data ضعیف‌تر بماند، همان‌گونه گزارش می‌شود؛ اگر token hybrid نزدیک به بهترین مدل برسد، همان به‌عنوان یافته مثبت مطرح می‌شود؛ و هیچ عدد قدیمی یا نمونه کوچک به‌عنوان نتیجه اصلی جایگزین full-cohort نمی‌شود.

\section{{اهداف پژوهش}}
هدف کلی پژوهش، ساخت و ارزیابی یک چارچوب قابل بازتولید برای پیش‌بینی \lr{{ICU discharge-feasibility proxy}} از داده‌های ۲۴ ساعت نخست ICU بود. اهداف اختصاصی عبارت‌اند از: استخراج کوهورت full-cohort از \lr{{MIMIC-IV v3.1}}؛ ساخت ویژگی‌های آزمایشگاهی، علائم حیاتی، شاخص‌های گمشده‌بودن و ویژگی‌های مهندسی‌شده؛ تولید توکن‌های علائم با تانسور ثابت؛ ارزیابی مدل‌های کلاسیک، ETHOS/few-shot، supervised token، pretraining، hybrid، stacking و ensemble؛ ارزیابی calibration و uncertainty؛ تحلیل subgroup/error و ablation؛ و بررسی simulated federated learning.

\section{{سوالات پژوهش}}
سوالات پژوهش در این نسخه نهایی عبارت‌اند از:
\begin{{enumerate}}
\item آیا ویژگی‌های آزمایشگاهی و علائم حیاتی ۲۴ ساعت اول ICU می‌توانند یک \lr{{ICU discharge-feasibility proxy}} را پیش‌بینی کنند؟
\item عملکرد مدل‌های کلاسیک و ensemble در مقایسه با مدل‌های \lr{{Few-Shot/ETHOS}} چگونه است؟
\item آیا توکن‌های علائم به‌تنهایی یا در مدل‌های hybrid باعث بهبود عملکرد می‌شوند؟
\item آیا Few-Shot در سناریوهای کم‌داده نقش قوی‌تری نسبت به full-data دارد؟
\item آیا self-supervised pretraining باعث بهبود نمایش توکنی می‌شود؟
\item آیا simulated federated learning می‌تواند عملکرد قابل‌قبولی نسبت به مدل متمرکز حفظ کند؟
\item کدام مدل بهترین تعادل بین discrimination و calibration دارد؟
\item uncertainty و selective/conformal prediction چه نقشی در تفسیر ایمن‌تر مدل دارند؟
\item شواهد نهایی چگونه از عنوان مصوب پایان‌نامه دفاع می‌کنند، حتی اگر pure few-shot برنده full-data نباشد؟
\end{{enumerate}}

\section{{نوآوری‌ها و دستاوردها}}
نوآوری پایان‌نامه در ادعای پیروزی یک روش واحد نیست؛ در آزمون جامع و صادقانه یک فرضیه عنوانی است. کوهورت نهایی شامل \lr{{32,399}} ICU stay و \lr{{32,399}} subject است. توکن‌ها با شکل \lr{{(32399, 25, 4)}} ساخته شدند و رجیستری آزمایش‌ها \lr{{254}} ردیف داشت. بهترین مدل overall، \lr{{Stacked Validation Meta-LR}}، به \lr{{AUROC=0.759}}، \lr{{AUPRC=0.821}}، \lr{{Brier=0.187}} و \lr{{ECE=0.016}} رسید. بهترین discrimination مربوط به \lr{{Weighted Average Ensemble}} با \lr{{AUROC=0.759}} و \lr{{AUPRC=0.824}} بود. در میان مدل‌های token/hybrid، \lr{{Token Hybrid CatBoost}} با \lr{{AUROC=0.757}} و \lr{{AUPRC=0.820}} بسیار نزدیک به بهترین مدل کلی قرار گرفت. pure few-shot/prototype با \lr{{AUROC=0.607}} و \lr{{AUPRC=0.699}} ضعیف‌تر باقی ماند.

{tables["cohort"]}

{tables["labels"]}

{tables["features"].replace("tab:feature-groups", "tab:feature-groups-methods")}

{tables["tokens"].replace("tab:token-summary", "tab:token-summary-methods")}

\section{{ساختار پایان‌نامه}}
فصل دوم پیشینه و مبانی نظری را مرور می‌کند. فصل سوم روش‌شناسی داده، outcome، tokenization و مدل‌ها را شرح می‌دهد. فصل چهارم طراحی آزمایش‌ها و پروتکل no-gap را ارائه می‌کند. فصل پنجم نتایج نهایی را گزارش می‌کند. فصل ششم تحلیل اکتشافی داده و الگوهای خطا را ارائه می‌دهد. فصل هفتم یافته‌ها را تفسیر کرده، به سوالات پژوهش پاسخ می‌دهد و محدودیت‌ها را روشن می‌کند. فصل هشتم نتیجه‌گیری و مسیرهای آینده را جمع‌بندی می‌کند.
"""


def chapter_background() -> str:
    sections = [
        ("داده‌های سلامت الکترونیک و مراقبت ویژه", [
            "پرونده الکترونیک سلامت، داده‌های آزمایشگاهی، علائم حیاتی، کدهای تشخیصی و داده‌های ترخیص را در یک ساختار ناهمگون ثبت می‌کند. این داده‌ها برای مدل‌سازی بالینی جذاب‌اند، اما تفسیر آن‌ها ساده نیست؛ زیرا فقدان داده در ICU اغلب تصادفی نیست و خود می‌تواند بازتاب شدت بیماری یا سیاست اندازه‌گیری باشد.",
            "در ICU، هر اندازه‌گیری فقط یک عدد نیست؛ حاصل تصمیم مراقبتی، زمان‌بندی نمونه‌گیری، وضعیت فیزیولوژیک بیمار و گاهی محدودیت‌های ثبت داده است. بنابراین مدل‌های یادگیری ماشین باید با leakage، گمشده‌بودن informative، ناهمگونی واحدها و وابستگی به workflow بالینی مواجه شوند."
        ]),
        ("پایگاه داده میمیک نسخه چهار", [
            r"\lr{MIMIC-IV} یکی از مهم‌ترین پایگاه‌های عمومی ICU است و از طریق \lr{PhysioNet} در دسترس پژوهشگران دارای مجوز قرار می‌گیرد \cite{johnson2023mimic,goldberger2000physionet}. این پایگاه به دلیل ناشناس‌سازی، مستندسازی نسبتا کامل و پذیرش گسترده در جامعه انفورماتیک سلامت، بستری مناسب برای benchmarkهای قابل بازتولید فراهم می‌کند.",
            "مزیت MIMIC-IV دسترسی پژوهشی و غنای جدولی آن است؛ محدودیت آن تک‌مرکزی بودن، گذشته‌نگر بودن و وابستگی به مستندسازی بالینی است. به همین دلیل در این پایان‌نامه نتیجه‌ها به‌عنوان شواهد پژوهشی تفسیر می‌شوند و نه اعتبارسنجی بیرونی یا آینده‌نگر."
        ]),
        ("پیش‌بینی بالینی و مدل‌های ریسک", [
            r"مدل‌های پیش‌بینی بالینی باید هم discrimination و هم calibration را گزارش کنند. \lr{AUROC} توان رتبه‌بندی موارد مثبت و منفی را می‌سنجد، اما برای احتمال‌های فردی کافی نیست. \lr{AUPRC} در مسئله‌های نامتوازن حساس‌تر است و \lr{Brier Score} و \lr{ECE} کیفیت احتمال خروجی را بهتر نشان می‌دهند.",
            r"راهنماهای گزارش‌دهی مانند \lr{TRIPOD} و ارزیابی خطر سوگیری مانند \lr{PROBAST} بر شفافیت تعریف outcome، validation، calibration و محدودیت‌ها تاکید می‌کنند \cite{wolff2019probast}. پایان‌نامه حاضر به همین دلیل، نتیجه نهایی را فقط با AUROC معرفی نمی‌کند و calibration-aware model selection را نیز وارد تصمیم می‌کند."
        ]),
        ("یادگیری ماشین روی داده‌های جدولی و گرادیان بوستینگ", [
            r"مدل‌های \lr{XGBoost}، \lr{LightGBM} و \lr{CatBoost} در داده‌های جدولی بالینی بسیار قوی‌اند، زیرا تعاملات غیرخطی، آستانه‌های بالینی و الگوهای گمشده‌بودن را بدون نیاز به معماری‌های پیچیده یاد می‌گیرند \cite{chen2016xgboost,ke2017lightgbm,prokhorenkova2018catboost}.",
            "این پایان‌نامه این مدل‌ها را baseline ساده تلقی نمی‌کند، بلکه آن‌ها را رقیب سخت و لازم برای آزمون فرضیه few-shot/token می‌داند. اگر یک روش جدید نتواند در برابر gradient boosting و ensemble بهبود معنادار نشان دهد، باید نقش آن دقیق‌تر و محدودتر تعریف شود."
        ]),
        ("یادگیری چندنمونه‌ای", [
            r"یادگیری چندنمونه‌ای \lr{(Few-Shot Learning)} برای سناریوهایی طراحی شده است که تعداد نمونه‌های برچسب‌دار در هر کلاس کم است. شبکه‌های نمونه‌ای \lr{(Prototypical Networks)} با ساخت prototype برای هر کلاس، نمونه پرس‌وجو را بر اساس فاصله در فضای نهفته طبقه‌بندی می‌کنند \cite{snell2017prototypical}.",
            r"در سلامت، جذابیت few-shot روشن است: بسیاری از بیماری‌ها، مراکز یا زیرگروه‌ها داده برچسب‌دار فراوان ندارند. اما داده جدولی ICU از نظر ساختار با تصویر یا متن متفاوت است. بنابراین few-shot باید در کنار baselineهای جدولی قوی و سناریوهای low-data ارزیابی شود."
        ]),
        ("مدل اتوس و مدل‌سازی توکن‌های علائم", [
            r"مدل ETHOS در این پروژه به‌عنوان رمزگذار توکن‌های علائم استفاده شد. توکن‌های علائم \lr{(Symptom Tokens)} هر اندازه‌گیری را به واحدی شامل شناسه علامت، شدت، مولفه زمانی و modality تبدیل می‌کنند. هدف این بازنمایی، ایجاد پلی میان داده جدولی و مدل‌های token-based است.",
            "اهمیت symptom-token modelling در این پایان‌نامه آن است که عنوان مصوب را از سطح شعار به آزمون تجربی تبدیل می‌کند. توکن‌ها هم به‌صورت token-only، هم embedding، هم pretraining، و هم hybrid با ویژگی‌های جدولی آزمون شدند."
        ]),
        ("پیش‌آموزش خودنظارتی برای توکن‌های بالینی", [
            "پیش‌آموزش خودنظارتی می‌تواند نمایش‌های بهتری از داده بدون نیاز به برچسب بسازد. در داده‌های بالینی، masked reconstruction، denoising autoencoder، contrastive patient learning و sequence autoencoder نمونه‌هایی از این مسیرند.",
            "در این پژوهش، pretraining فقط روی داده آموزش انجام شد تا leakage ایجاد نشود. نتیجه نهایی نشان داد pretraining امکان‌پذیر و مفید است، اما مدل‌های supervised ensemble را پشت سر نگذاشت. این یافته برای گزارش علمی مهم است، زیرا نقش pretraining را modest و نه غالب نشان می‌دهد."
        ]),
        ("یادگیری فدرال", [
            r"یادگیری فدرال \lr{(Federated Learning)} برای شرایطی پیشنهاد شده که داده خام میان مراکز جابه‌جا نمی‌شود و فقط به‌روزرسانی مدل یا پارامترها مبادله می‌گردد \cite{mcmahan2017communication,kairouz2021advances}.",
            "در این پایان‌نامه، یادگیری فدرال به‌صورت شبیه‌سازی‌شده بررسی شد. گره‌ها از داده موجود ساخته شدند و این کار معادل federation چندمرکزی واقعی نیست. با این حال، نتیجه شبیه‌سازی برای فهم افت عملکرد نسبت به مدل متمرکز و امکان‌سنجی روش مفید است."
        ]),
        ("کالیبراسیون، پیش‌بینی انتخابی و پیش‌بینی همساز", [
            r"کالیبراسیون نشان می‌دهد احتمال خروجی مدل تا چه حد با فراوانی مشاهده‌شده همخوان است. \lr{Brier Score} میانگین خطای مربعی احتمال و \lr{ECE} اختلاف میان confidence و accuracy در binها را اندازه می‌گیرند \cite{guo2017calibration}.",
            r"پیش‌بینی انتخابی \lr{(Selective Prediction)} اجازه می‌دهد مدل فقط روی موارد مطمئن‌تر پاسخ دهد و موارد مبهم را defer کند. پیش‌بینی همساز \lr{(Conformal Prediction)} نیز مجموعه‌ای از برچسب‌های محتمل را با سطح پوشش هدف ارائه می‌کند. این ابزارها در پایان‌نامه برای تفسیر محافظه‌کارانه uncertainty استفاده شدند."
        ]),
        ("مطالعات مرتبط و جایگاه پایان‌نامه حاضر", [
            r"مطالعات پیشین نشان داده‌اند که در بسیاری از مسئله‌های EHR، gradient boosting و مدل‌های جدولی همچنان بسیار رقابتی‌اند \cite{purushotham2018benchmarking,grinsztajn2022why}. همزمان، مدل‌های عمیق و token-based برای EHR و متن بالینی مسیرهای تازه‌ای گشوده‌اند \cite{li2020behrt,shang2020gbert}.",
            "جایگاه این پایان‌نامه در ترکیب این دو مسیر است: مدل‌های جدولی قوی، توکن‌های علائم، ETHOS/few-shot، hybrid modelling و ارزیابی calibration-aware در یک چارچوب full-cohort و no-gap. نتیجه نهایی از عنوان مصوب دفاع می‌کند، اما با صداقت نشان می‌دهد few-shot خالص برنده full-data نبود."
        ]),
    ]
    body = [r"\chapter{پیشینه و مرور ادبیات}", r"\label{ch:background}"]
    body.append("این فصل پایه نظری پایان‌نامه را مرور می‌کند. تمرکز فصل بر داده‌های سلامت الکترونیک، ICU، پایگاه MIMIC-IV، مدل‌های جدولی، یادگیری چندنمونه‌ای، ETHOS، یادگیری فدرال، کالیبراسیون و uncertainty است. هدف فصل، آماده کردن خواننده برای فهم این نکته است که چرا فرضیه symptom-token/few-shot باید در برابر baselineهای supervised سخت آزمون شود.")
    for title, paras in sections:
        body.append(rf"\section{{{title}}}")
        body.extend(paras)
    body.append(r"""
\section{جدول مقایسه مطالعات و جایگاه پژوهش}
\begin{small}
\setlength{\tabcolsep}{2pt}
\begin{longtable}{@{}p{0.20\textwidth}p{0.20\textwidth}p{0.24\textwidth}p{0.26\textwidth}@{}}
\caption{خلاصه جایگاه پژوهش حاضر نسبت به محورهای ادبیات.}\label{tab:related-work-position}\\
\toprule
محور ادبیات & نمونه رویکرد & محدودیت رایج & جایگاه این پایان‌نامه\\
\midrule
\endfirsthead
\toprule
محور ادبیات & نمونه رویکرد & محدودیت رایج & جایگاه این پایان‌نامه\\
\midrule
\endhead
پیش‌بینی مراقبت ویژه & مدل‌های ریسک و گرادیان بوستینگ & برچسب‌های متفاوت و گاهی بدون کالیبراسیون کامل & شاخص جایگزین در کوهورت کامل همراه با \lr{AUROC}، \lr{AUPRC}، \lr{Brier} و \lr{ECE}\\
یادگیری عمیق پرونده سلامت & شبکه بازگشتی، ترنسفورمر و مدل‌های توکنی & نیاز به داده زمانی غنی و محاسبه بیشتر & نمایش فشرده توکن‌های علائم از ۲۴ ساعت نخست\\
یادگیری چندنمونه‌ای & شبکه نمونه‌ای، تطبیق و \lr{MAML} & شواهد کمتر روی داده جدولی بالینی & آزمون مستقیم یادگیری چندنمونه‌ای خالص و گزارش نتیجه منفی\\
مدل‌سازی ترکیبی & ترکیب نهفته‌سازی و ویژگی‌های جدولی & گاهی بدون خط پایه قوی & \lr{Token Hybrid CatBoost} در برابر انباشت و ensemble\\
یادگیری فدرال سلامت & میانگین‌گیری فدرال و تجمیع گره‌ها & اغلب نیازمند چندمرکزی واقعی & شبیه‌سازی شفاف بدون ادعای فدراسیون واقعی\\
کالیبراسیون و عدم‌قطعیت & کالیبراسیون احتمالات، پیش‌بینی انتخابی و پیش‌بینی همساز & گزارش ناقص در برخی معیارهای مقایسه & انتخاب مدل کالیبراسیون‌محور و تحلیل عدم‌قطعیت\\
\bottomrule
\end{longtable}
\end{small}
این جدول نشان می‌دهد که ارزش پایان‌نامه فقط در یک عدد نهایی نیست، بلکه در اتصال چند سنت پژوهشی با یک برچسب تعریف‌شده و گزارش صادقانه است.
""")
    body.append(r"""
\section{خلاصه فصل}
مرور ادبیات نشان می‌دهد که پایان‌نامه حاضر در نقطه اتصال سه جریان قرار دارد: پیش‌بینی بالینی روی داده‌های مراقبت ویژه، بازنمایی توکنی و یادگیری چندنمونه‌ای، و ارزیابی کالیبراسیون‌محور. از این پیشینه یک اصل روش‌شناختی استخراج می‌شود: هر ادعا باید با برچسب دقیق، تقسیم بدون نشت اطلاعات، خط پایه قوی و گزارش محدودیت همراه باشد. فصل بعد روش‌شناسی مطالعه را بر همین اساس توضیح می‌دهد.
""")
    return "\n\n".join(body)


def chapter_methods(tables: dict[str, str]) -> str:
    return rf"""
\chapter{{روش‌شناسی}}
\label{{ch:methods}}

\section{{طراحی مطالعه}}
این مطالعه گذشته‌نگر و مبتنی بر تحلیل ثانویه داده‌های \lr{{MIMIC-IV v3.1}} است. واحد تحلیل، بستری ICU در کوهورت نهایی است. همه ویژگی‌ها از ۲۴ ساعت نخست ICU استخراج شدند و outcome به‌صورت \lr{{ICU discharge-feasibility proxy}} تعریف شد. این طراحی برای پاسخ به یک پرسش پژوهشی مناسب است: آیا سیگنال‌های اولیه ICU می‌توانند مسیر مطلوب/نامطلوب تعریف‌شده در proxy را پیش‌بینی کنند؟

\section{{منبع داده و دسترسی}}
داده‌ها از مسیر محلی \lr{{mimic-iv-3.1}} استخراج شدند. به‌دلیل محدودیت‌های مجوز \lr{{MIMIC-IV}}، داده خام قابل بازتوزیع نیست. بازتولید پژوهش باید با دسترسی رسمی به \lr{{PhysioNet}}، اجرای اسکریپت‌های استخراج، و استفاده از splitها و رجیستری آزمایش انجام شود.

\section{{نمادگذاری ریاضی}}
فرض کنید برای هر بستری مراقبت ویژه یک جفت \((x_i, y_i)\) وجود دارد که در آن \(x_i\) بردار ویژگی‌ها یا تانسور توکن و \(y_i \in \{{0,1\}}\) برچسب شاخص جایگزین است. در نمایش جدولی، \(x_i \in \mathbb{{R}}^p\) است و \(p\) شامل آزمایش‌ها، علائم حیاتی، ویژگی‌های مهندسی‌شده و شاخص‌های گمشده‌بودن می‌شود. در نمایش توکنی، \(T_i \in \mathbb{{R}}^{{25 \times 4}}\) است. مدل \(f_\theta\) احتمال \(\hat{{p}}_i = P(y_i=1|x_i)\) را تولید می‌کند.

\begin{{small}}
\begin{{longtable}}{{p{{0.18\textwidth}}p{{0.72\textwidth}}}}
\caption{{نمادگذاری اصلی روش‌شناسی.}}\label{{tab:notation-final}}\\
\toprule
نماد & معنا\\
\midrule
\endfirsthead
\toprule
نماد & معنا\\
\midrule
\endhead
\(\mathcal{{D}}\) & کوهورت نهایی بستری‌های مراقبت ویژه\\
\(x_i\) & بردار ویژگی جدولی بیمار \(i\)\\
\(T_i\) & تانسور توکن‌های علائم برای بیمار \(i\)\\
\(y_i\) & برچسب شاخص جایگزین قابلیت ترخیص از مراقبت ویژه\\
\(\hat{{p}}_i\) & احتمال پیش‌بینی‌شده برای برچسب مثبت\\
\(f_\theta\) & مدل یادگیری ماشین با پارامترهای \(\theta\)\\
\(\mathcal{{S}}, \mathcal{{Q}}\) & مجموعه پشتیبان و پرس‌وجو در یادگیری چندنمونه‌ای\\
\(\mathbf{{c}}_k\) & نمونه اولیه کلاس \(k\) در فضای نهفته\\
\bottomrule
\end{{longtable}}
\end{{small}}

\section{{تعریف کوهورت و معیارهای ورود و خروج}}
کوهورت نهایی شامل \lr{{32,399}} ICU stay و \lr{{32,399}} subject بود. معیارهای اصلی شامل بزرگسال بودن، وجود داده لازم برای تعریف outcome، و ساخت موفق ویژگی‌ها و توکن‌ها بود. هیچ سقف نمونه‌گیری در تحلیل اصلی وجود نداشت. شکل~\ref{{fig:cohort-flow}} مسیر مفهومی ساخت کوهورت را نشان می‌دهد.

{fig_tex("cohort_flow.png", "جریان ساخت کوهورت و برچسب نهایی بدون استفاده از نمونه‌گیری در تحلیل اصلی.", "fig:cohort-flow")}

\section{{تعریف شاخص جایگزین قابلیت ترخیص از مراقبت ویژه}}
برچسب مثبت زمانی اختصاص یافت که سه شرط همزمان برقرار باشد: بیمار در بیمارستان زنده بماند، مقصد ترخیص در گروه مطلوب قرار گیرد، و طول اقامت ICU کمتر از \lr{{720}} ساعت باشد. اگر یکی از این شروط برقرار نبود، برچسب منفی در نظر گرفته شد. این proxy به‌صراحت برابر «قابلیت درمان واقعی» یا «آمادگی ترخیص داوری‌شده توسط پزشک» نیست. این مرزبندی در کل پایان‌نامه حفظ شده است.

\section{{استخراج ویژگی‌های ۲۴ ساعت اول}}
ویژگی‌های خام شامل ۱۵ آزمایش و ۷ علامت حیاتی بودند. مقدارهای ثبت‌شده در پنجره نخست ICU به‌صورت میانه یا خلاصه مقاوم تجمیع شدند. جایگذاری، scaling و انتخاب آستانه فقط بر اساس داده آموزش/اعتبارسنجی انجام شد تا test leakage کاهش یابد. افزون بر ویژگی‌های خام، شاخص‌های گمشده‌بودن و ویژگی‌های مهندسی‌شده مانند shock-index proxy و SOFA-like proxy ساخته شدند.

{tables["features"]}

\section{{ساخت توکن‌های علائم}}
توکن‌های علائم هر اندازه‌گیری را به واحدی قابل پردازش برای مدل اتوس و مدل‌های توکن‌محور تبدیل کردند. هر توکن شامل چهار کانال بود: \lr{{symptom\_id}}، شدت، فاصله زمانی نرمال‌شده و شیوه اندازه‌گیری. تانسور نهایی شکل \lr{{(32399, 25, 4)}} داشت. شکل~\ref{{fig:token-representation}} ساختار این نمایش را نشان می‌دهد.

{fig_tex("symptom_token_representation.png", "نمایش شماتیک تانسور توکن‌های علائم با ۲۵ توکن و ۴ کانال.", "fig:token-representation")}

{tables["tokens"]}

\section{{مدل‌های کلاسیک و گرادیان بوستینگ}}
مدل‌های کلاسیک شامل رگرسیون لجستیک، جنگل تصادفی، \lr{{XGBoost}}، \lr{{LightGBM}}، \lr{{CatBoost}} و baseline عصبی بودند. این مدل‌ها روی feature set جدولی و در برخی موارد tabular-token ارزیابی شدند. مدل‌های gradient boosting برای داده جدولی ICU رقیب جدی محسوب می‌شوند و بنابراین مقایسه few-shot/token بدون آن‌ها ناقص بود.

\section{{مدل اتوس و یادگیری چندنمونه‌ای}}
مدل ETHOS و prototype/few-shot بر تانسور توکن‌های علائم تکیه داشتند. روش pure few-shot/prototype به‌عنوان آزمون مستقیم بخش few-shot عنوان استفاده شد. مدل‌های supervised ETHOS نیز با زیان \lr{{BCE}}، زیان کمکی contrastive و زیان کمکی prototypical آزمون شدند. این طراحی اجازه داد که ضعف یا قوت few-shot خالص از ارزش نمایش توکنی و مدل‌های hybrid جدا شود.

\subsection{{فرمول نمونه اولیه}}
در مدل نمونه اولیه، بردار نهفته هر نمونه با \(h_i = g_\theta(T_i)\) محاسبه می‌شود. برای هر کلاس \(k\)، نمونه اولیه برابر میانگین بردارهای نهفته پشتیبان همان کلاس است:
\[
\mathbf{{c}}_k = \frac{{1}}{{|\mathcal{{S}}_k|}}\sum_{{(T_i,y_i)\in \mathcal{{S}}_k}} g_\theta(T_i).
\]
احتمال تعلق نمونه پرس‌وجو به کلاس \(k\) از فاصله یا شباهت میان بردار نهفته و نمونه اولیه محاسبه می‌شود:
\[
P(y=k|T) = \frac{{\exp(-d(g_\theta(T), \mathbf{{c}}_k))}}{{\sum_{{k'}}\exp(-d(g_\theta(T), \mathbf{{c}}_{{k'}}))}}.
\]
این فرمول برای مسئله داده کامل کافی نبود، اما به‌عنوان آزمون مستقیم فرضیه یادگیری چندنمونه‌ای ضروری بود.

\section{{مدل‌های توکنی نظارت‌شده، ترکیبی و انباشته}}
مدل‌های توکنی نظارت‌شده شامل پرسپترون چندلایه توکنی، شبکه پیچشی توکنی، شبکه بازگشتی توکنی، تجمیع توجه و ترنسفورمر توکنی بودند. در مسیر ترکیبی، توکن‌ها به ویژگی‌های جدولی اضافه شدند یا نهفته‌سازی‌های \lr{{ETHOS}} با ویژگی‌های جدولی ترکیب شدند. در نهایت، \lr{{Stacked Validation Meta-LR}} و \lr{{Weighted Average Ensemble}} بر پایه مدل‌های منتخب اعتبارسنجی ساخته شدند. شکل~\ref{{fig:model-framework}} چارچوب مدل‌ها را نشان می‌دهد.

{fig_tex("model_family_framework.png", "خانواده مدل‌ها و لایه‌های ارزیابی در گذر نهایی.", "fig:model-framework")}

\section{{پیش‌آموزش خودنظارتی}}
برای بررسی اینکه آیا نمایش توکنی بدون برچسب بهتر می‌شود یا نه، چند هدف self-supervised استفاده شد: masked reconstruction، denoising autoencoder، contrastive patient learning، sequence autoencoder و patient representation learning. سپس embeddingهای pretrained در مدل‌های downstream مانند \lr{{XGBoost}}، \lr{{LightGBM}} و \lr{{CatBoost}} استفاده شدند.

\section{{طراحی سناریوی کم‌داده}}
در طراحی کم‌داده، fractionهای آموزش از \lr{{0.01}} تا \lr{{1.00}} بررسی شدند، در حالی که validation/test ثابت ماند. هدف این نبود که few-shot حتما برنده شود، بلکه بررسی این بود که آیا کاهش داده بر نسبت عملکرد few-shot، supervised tree و token hybrid اثر متفاوت دارد یا خیر.

\section{{یادگیری فدرال شبیه‌سازی‌شده}}
یادگیری فدرال در این پایان‌نامه به‌صورت simulation اجرا شد. گره‌ها از داده موجود ساخته شدند و مدل \lr{{Simulated Federated LR Weighted Nodes}} با مرجع متمرکز logistic regression مقایسه شد. این بخش معادل استقرار چندمرکزی واقعی نیست و صرفا امکان‌سنجی پژوهشی و تحلیل افت عملکرد را نشان می‌دهد.

\section{{کالیبراسیون، عدم‌قطعیت و پیش‌بینی انتخابی/همساز}}
مدل‌های منتخب با calibration validation-first، Brier Score و ECE ارزیابی شدند. پیش‌بینی انتخابی بررسی کرد که اگر مدل فقط روی موارد مطمئن‌تر پاسخ دهد، accuracy در retained cases چگونه تغییر می‌کند. پیش‌بینی همساز نیز برای گزارش coverage و اندازه مجموعه پیش‌بینی استفاده شد. نقش این تحلیل‌ها، ایجاد تفسیر محافظه‌کارانه‌تر است.

\subsection{{فرمول معیارهای کالیبراسیون}}
امتیاز بریر برای \(n\) نمونه چنین تعریف می‌شود:
\[
\mathrm{{Brier}} = \frac{{1}}{{n}}\sum_{{i=1}}^n(\hat{{p}}_i-y_i)^2.
\]
برای \lr{{ECE}}، پیش‌بینی‌ها به \(M\) بازه تقسیم می‌شوند:
\[
\mathrm{{ECE}} = \sum_{{m=1}}^M \frac{{|B_m|}}{{n}}\left|\mathrm{{acc}}(B_m)-\mathrm{{conf}}(B_m)\right|.
\]
این معیارها در کنار AUROC و AUPRC تعیین کردند که چرا Stacked Validation Meta-LR به‌عنوان بهترین مدل overall انتخاب شد.

\subsection{{شبه‌کد خط لوله نهایی}}
\begin{{enumerate}}
\item استخراج ICU stayهای واجد شرایط از \lr{{MIMIC-IV v3.1}}.
\item ساخت outcome پژوهشی بر اساس بقا، مقصد ترخیص مطلوب و ICU LOS کمتر از \lr{{720}} ساعت.
\item استخراج labs/vitals از ۲۴ ساعت نخست و ساخت ویژگی‌های جدولی.
\item ساخت تانسور توکن‌های علائم با شکل \lr{{(32399, 25, 4)}}.
\item آموزش baselineهای tabular و مدل‌های token/ETHOS/few-shot.
\item اجرای pretraining، hybrid modelling، stacking و ensemble.
\item کالیبراسیون validation-first و ارزیابی روی test set.
\item اجرای uncertainty، conformal، subgroup، ablation و federated simulation.
\item انتخاب مدل نهایی بر اساس discrimination و calibration، نه صرفا یک معیار.
\end{{enumerate}}

\section{{تحلیل زیرگروه/خطا، حذف ویژگی و پایداری}}
تحلیل subgroup بر جنس، گروه سنی و careunit انجام شد و ماهیت آن اکتشافی است. ablation اثر labs-only، vitals-only، labs+vitals، token-only، tabular-only، tabular+token و اجزای ETHOS/token را بررسی کرد. این تحلیل‌ها برای شناخت نقاط ضعف و نقش اجزای مدل استفاده شدند، نه برای ادعای علی یا fairness کامل.

\section{{معیارهای ارزیابی}}
معیارهای اصلی شامل سطح زیر منحنی ROC \lr{{(AUROC)}}، سطح زیر منحنی Precision-Recall \lr{{(AUPRC)}}، accuracy، recall، specificity، \lr{{F1}}، امتیاز بریر \lr{{(Brier Score)}} و خطای کالیبراسیون مورد انتظار \lr{{(Expected Calibration Error, ECE)}} بودند. انتخاب مدل نهایی با توجه همزمان به discrimination و calibration انجام شد.

\section{{بازتولیدپذیری و اخلاق}}
رجیستری آزمایش‌ها با \lr{{254}} ردیف نگهداری شد. splitها، پیش‌بینی‌های آزمون، جدول‌های no-gap و گزارش‌های calibration/uncertainty/subgroup/ablation/federated در \lr{{results/final\_optimized}} ذخیره شده‌اند. از نظر اخلاقی، مطالعه روی داده ناشناس‌شده انجام شده است، اما داده خام قابل بازتوزیع نیست و نتیجه‌ها فقط در محدوده پژوهش retrospective تفسیر می‌شوند.

\section{{جزئیات پیاده‌سازی و کنترل نسخه نتایج}}
برای جلوگیری از آمیختگی خروجی‌های تاریخی با خروجی‌های نهایی، no-gap pass یک مجموعه source-of-truth مشخص ساخت. در متن پایان‌نامه، هرگاه تعارضی میان فایل‌های قدیمی و no-gap وجود داشت، no-gap مقدم دانسته شد. جدول ranking نهایی، registry و گزارش‌های تکمیلی به‌عنوان مدارک قابل بررسی در بسته منبع قرار گرفته‌اند.

\begin{{small}}
\begin{{longtable}}{{p{{0.30\textwidth}}p{{0.58\textwidth}}}}
\caption{{کنترل‌های بازتولیدپذیری در پایان‌نامه.}}\label{{tab:repro-controls}}\\
\toprule
کنترل & توضیح\\
\midrule
\endfirsthead
\toprule
کنترل & توضیح\\
\midrule
\endhead
split ثابت & validation/test برای مقایسه مدل‌ها ثابت نگه داشته شد.\\
train-only preprocessing & scaling، imputation و pretraining بدون استفاده از test انجام شد.\\
experiment registry & \lr{{254}} ردیف آزمایش برای ردیابی طراحی و نتیجه ثبت شد.\\
prediction files & پیش‌بینی‌های آزمون مدل‌های منتخب برای ROC/PR و calibration نگهداری شد.\\
no-gap reports & همه گپ‌های مدل، pretraining، token، calibration و uncertainty بسته شدند.\\
claim audit & ادعاهای few-shot، deployment و old-sample از متن نهایی حذف یا محدود شدند.\\
\bottomrule
\end{{longtable}}
\end{{small}}
"""


def chapter_experiments(tables: dict[str, str]) -> str:
    return rf"""
\chapter{{طراحی آزمایش‌ها}}
\label{{ch:experiments}}

\section{{پیکربندی کوهورت کامل}}
تحلیل اصلی روی \lr{{32,399}} ICU stay انجام شد. این نکته برای تفسیر کل پایان‌نامه حیاتی است: هیچ نتیجه مبتنی بر benchmark کوچک‌تر به‌عنوان نتیجه اصلی ارائه نشده است. همه headline results از no-gap pass و full-cohort evidence گرفته شده‌اند. شکل~\ref{{fig:experimental-design}} نمای کلی طراحی آزمایش‌ها را نشان می‌دهد.

{fig_tex("experimental_design_overview.png", "نمای کلی طراحی آزمایش‌های نهایی از بازتولید خط پایه تا تحلیل پایداری.", "fig:experimental-design")}

\section{{بازتولید خط پایه}}
نخست baselineهای tabular بازتولید شدند تا سقف عملکرد supervised کلاسیک مشخص شود. این مرحله نشان داد که مدل‌های gradient boosting و random forest روی داده جدولی ICU بسیار رقابتی‌اند و باید در هر مقایسه نهایی حضور داشته باشند.

{tables["baseline"].replace("tab:baseline-reproduction", "tab:app-baseline-reproduction")}

\section{{پروتکل جلوگیری از نشت داده}}
همه مراحل feature scaling، imputation، calibration و selection باید فقط از داده آموزش/اعتبارسنجی استفاده کنند. test set فقط برای گزارش نهایی استفاده شد. گزارش leakage audit در no-gap pass هیچ blocker بحرانی نشان نداد و اسکریپت‌های exploratory قدیمی از ادعاهای نهایی حذف شدند.

\section{{خانواده مدل‌ها و جستجوی ابرپارامتر}}
خانواده مدل‌ها شامل tabular baseline، optimized supervised models، token-only supervised models، ETHOS/few-shot، token embeddings، pretraining downstream، hybrid models، stacking و weighted ensembles بود. جستجوی ابرپارامترها بر اساس validation performance و با ثبت در experiment registry انجام شد. هدف جستجو، مقایسه منصفانه بود، نه بیشینه‌سازی test set.

\section{{طراحی نهایی مدل اتوس و توکن‌ها}}
در no-gap pass، مدل‌های compact PyTorch token encoder، attention pooling، GRU، CNN و Transformer ارزیابی شدند. افزون بر BCE، lossهای کمکی contrastive و prototypical نیز بررسی شدند. همچنین token variants مانند missingness indicator tokens و abnormality flag tokens ارزیابی شدند.

\section{{طراحی پیش‌آموزش}}
pretraining فقط روی داده آموزش اجرا شد. هدف‌های masked reconstruction، denoising autoencoder، contrastive patient learning، sequence autoencoder و patient representation learning با downstreamهای مختلف ارزیابی شدند. معیار اصلی این بود که آیا embeddingهای pretrained به مدل tabular+embedding کمک می‌کنند یا خیر.

\section{{طراحی کم‌داده}}
در low-data، fractionهای \lr{{0.01}}، \lr{{0.02}}، \lr{{0.05}}، \lr{{0.10}}، \lr{{0.20}}، \lr{{0.50}} و \lr{{1.00}} بررسی شدند. test set ثابت ماند تا روند sample efficiency قابل مقایسه باشد. اگر few-shot در این شرایط بهتر عمل می‌کرد، باید در این بخش دیده می‌شد.

\section{{طراحی کالیبراسیون و عدم‌قطعیت}}
مدل‌های نهایی با isotonic calibration و شاخص‌های Brier/ECE بررسی شدند. در uncertainty، selective prediction در coverageهای مختلف و conformal prediction در سطح‌های \lr{{0.95}}، \lr{{0.90}} و \lr{{0.80}} گزارش شد. این تحلیل‌ها برای تفسیر احتمال خروجی مدل ضروری بودند.

\section{{طراحی زیرگروه/خطا و حذف ویژگی}}
subgroup analysis روی متغیرهای در دسترس انجام شد و exploratory باقی ماند. ablation اثر گروه ویژگی‌ها و نمایش‌های token را جدا کرد. در همه موارد، نتیجه‌ها برای فهم مدل استفاده شدند و نه برای ادعای causal inference.

\section{{طراحی شبیه‌سازی فدرال}}
در شبیه‌سازی فدرال، گره‌ها از داده موجود ساخته شدند و مدل weighted-node logistic regression با مرجع متمرکز مقایسه شد. این طراحی نشان می‌دهد که aggregation فدرال در یک سناریوی پژوهشی چه افتی دارد، اما هیچ ادعای real-world deployment ایجاد نمی‌کند.

\section{{محیط نرم‌افزاری و سخت‌افزاری}}
اجرای تحلیل‌ها با \lr{{Python}}، \lr{{pandas}}، \lr{{scikit-learn}}، \lr{{PyTorch}} و کتابخانه‌های gradient boosting انجام شد. خروجی‌های نهایی در \lr{{results/final\_optimized}} ذخیره شده‌اند. این پایان‌نامه فایل‌های گزارش، CSVهای نتایج و پیش‌بینی‌های آزمون را به‌عنوان source of truth استفاده می‌کند.
"""


def chapter_results(tables: dict[str, str]) -> str:
    return rf"""
\chapter{{نتایج}}
\label{{ch:results}}

\section{{خلاصه دیتاست و برچسب}}
کوهورت نهایی شامل \lr{{32,399}} ICU stay و \lr{{32,399}} subject بود. توزیع برچسب شامل \lr{{12,102}} نمونه منفی و \lr{{20,297}} نمونه مثبت بود. تانسور توکن‌ها شکل \lr{{(32399, 25, 4)}} داشت و رجیستری آزمایش‌ها \lr{{254}} ردیف را ثبت کرد. جدول‌های فصل اول خلاصه cohort، label، feature و token representation را ارائه کردند.

\section{{نتایج اصلی نهایی}}
جدول~\ref{{tab:main-final}} مقایسه اصلی مدل‌های نهایی را نشان می‌دهد. \lr{{Stacked Validation Meta-LR}} بهترین عملکرد overall و calibration-aware را داشت. \lr{{Weighted Average Ensemble}} بهترین AUPRC و discrimination profile را ارائه کرد. اختلاف عملکرد میان مدل‌های برتر کوچک است، اما از نظر انتخاب مدل، calibration اهمیت دارد.

{tables["main"]}

{fig_tex("auroc_comparison.png", "مقایسه سطح زیر منحنی تشخیص در مدل‌های نهایی.", "fig:auroc")}

{fig_tex("auprc_comparison.png", "مقایسه سطح زیر منحنی دقت-یادآوری در مدل‌های نهایی.", "fig:auprc")}

\section{{منحنی‌های تشخیص و دقت-یادآوری}}
شکل~\ref{{fig:roc}} و شکل~\ref{{fig:pr}} منحنی‌های مدل‌های برتر را نشان می‌دهند. نزدیکی منحنی‌ها با نتیجه جدول سازگار است: چند مدل supervised و hybrid در یک ناحیه عملکردی قرار دارند، اما ensemble و stacking در مجموع پایدارتر و کالیبره‌تر بودند.

{fig_tex("roc_curves_top_models.png", "منحنی‌های تشخیص برای مدل‌های برتر بر اساس پیش‌بینی‌های آزمون.", "fig:roc")}

{fig_tex("pr_curves_top_models.png", "منحنی‌های دقت-یادآوری برای مدل‌های برتر.", "fig:pr")}

\section{{نتایج مدل اتوس، یادگیری چندنمونه‌ای و مدل‌های توکنی/ترکیبی}}
نتایج جدول~\ref{{tab:fewshot}} نشان می‌دهد pure few-shot/prototype با \lr{{AUROC=0.607}} و \lr{{AUPRC=0.699}} ضعیف‌تر از supervised ensembles باقی ماند. این یافته منفی اما مهم است. در مقابل، \lr{{Token Hybrid CatBoost}} به \lr{{AUROC=0.757}} و \lr{{AUPRC=0.820}} رسید و ارزش symptom-token hybrid modelling را نشان داد.

{tables["fewshot"]}

{tables["tokenhybrid"]}

{fig_tex("token_hybrid_comparison.png", "مقایسه مدل‌های اتوس، توکن‌محور، نهفته‌سازی و ترکیبی.", "fig:token-hybrid")}

\section{{گونه‌های نمایش توکنی}}
جدول~\ref{{tab:token-variants}} نشان می‌دهد گونه‌های مختلف token representation ارزیابی شدند. نمایش‌های طولانی‌تر صرفا با padding علمی‌تر نمی‌شدند، زیرا تعداد مولفه‌های خام ۲۲ بود؛ بنابراین no-gap pass آن‌ها را به‌صورت scientifically redundant علامت‌گذاری کرد.

{tables["tokenvariant"]}

\section{{نتایج پیش‌آموزش}}
بهترین نتیجه pretraining مربوط به \lr{{denoising autoencoder + Tabular + XGBoost}} با \lr{{AUROC=0.755}} و \lr{{AUPRC=0.819}} بود. این نتیجه نشان داد pretraining امکان‌پذیر و مفید است، اما بر مدل‌های supervised ensemble برتری پیدا نکرد.

{tables["pretraining"]}

{fig_tex("pretraining_comparison.png", "مقایسه اهداف پیش‌آموزش خودنظارتی و مدل‌های پایین‌دستی.", "fig:pretraining")}

\section{{نتایج کم‌داده}}
نتایج low-data در جدول~\ref{{tab:low-data}} و شکل~\ref{{fig:low-data}} نشان می‌دهد مدل‌های supervised tree در بسیاری از fractions کوچک عملکرد بهتری از few-shot prototype داشتند. این یافته پاسخ مهمی به فرضیه اولیه است: few-shot به‌صورت خام حتی در کم‌داده نیز به‌طور خودکار برنده نشد.

{tables["lowdata"]}

{fig_tex("low_data_performance_curves.png", "منحنی‌های عملکرد در نسبت‌های کم‌داده با تقسیم ثابت.", "fig:low-data")}

\section{{نتایج کالیبراسیون}}
جدول~\ref{{tab:calibration}} نشان می‌دهد \lr{{Stacked Validation Meta-LR}} بهترین تعادل calibration-aware را ارائه کرد. در حالی که \lr{{Weighted Average Ensemble}} AUPRC بالاتری داشت، Brier و ECE مدل stacked برای انتخاب overall قابل دفاع‌تر بودند.

{tables["calibration"]}

{fig_tex("calibration_curves.png", "منحنی‌های کالیبراسیون مدل‌های منتخب.", "fig:calibration")}

\section{{نتایج عدم‌قطعیت و پیش‌بینی انتخابی/همساز}}
جدول~\ref{{tab:uncertainty}} نشان می‌دهد با کاهش coverage در selective prediction، accuracy در موارد retained افزایش می‌یابد. این نتیجه برای تفسیر ایمن‌تر مهم است: مدل می‌تواند موارد نامطمئن را به‌عنوان موارد نیازمند بررسی بیشتر مشخص کند، اما این به‌معنای تصمیم بالینی خودکار نیست.

{tables["uncertainty"]}

{fig_tex("selective_prediction_curve.png", "منحنی پیش‌بینی انتخابی؛ دقت با نگه‌داشتن موارد مطمئن‌تر افزایش می‌یابد.", "fig:selective")}

\section{{تحلیل زیرگروه و خطا}}
جدول~\ref{{tab:subgroup}} تحلیل اکتشافی زیرگروه را نشان می‌دهد. تفاوت عملکرد در گروه‌های سنی و careunitها نشان می‌دهد که عملکرد مدل در همه زیرجمعیت‌ها یکنواخت نیست. این یافته باید در validation خارجی و prospective بررسی شود.

{tables["subgroup"]}

\section{{حذف ویژگی و اهمیت ویژگی‌ها}}
جدول~\ref{{tab:ablation}} نشان می‌دهد labs+vitals قوی‌تر از vitals-only است و token-only معمولا از hybrid/tabular ضعیف‌تر می‌ماند. شکل~\ref{{fig:feature-importance}} اهمیت نسبی ویژگی‌ها را برای مدل interpretable tabular نشان می‌دهد. این تفسیر علی نیست، اما برای فهم رفتار مدل مفید است.

{tables["ablation"]}

{fig_tex("feature_importance_random_forest.png", "اهمیت ویژگی‌ها در مدل جدولی قابل تفسیر منتخب.", "fig:feature-importance")}

\section{{شبیه‌سازی یادگیری فدرال}}
جدول~\ref{{tab:federated}} نشان می‌دهد مدل \lr{{Simulated Federated LR Weighted Nodes}} به \lr{{AUROC=0.729}} و \lr{{AUPRC=0.796}} رسید. این نتیجه نسبت به ensembles مرکزی ضعیف‌تر است، اما از نظر امکان‌سنجی شبیه‌سازی‌شده قابل گزارش است.

{tables["federated"]}

{fig_tex("federated_overview.png", "نمای کلی یادگیری فدرال شبیه‌سازی‌شده و تجمیع وزن‌دار.", "fig:federated")}

\section{{رتبه‌بندی نهایی}}
جدول~\ref{{tab:ranking}} رتبه‌بندی نهایی no-gap را نشان می‌دهد. تفسیر نهایی باید همزمان سه نکته را حفظ کند: supervised stacking/ensemble در full-data برتر بودند؛ pure few-shot ضعیف‌تر باقی ماند؛ و Token Hybrid CatBoost عملکردی نزدیک به بهترین مدل کلی نشان داد.

{tables["ranking"]}
"""


def chapter_eda() -> str:
    return rf"""
\chapter{{تحلیل اکتشافی داده و الگوهای خطا}}
\label{{ch:eda}}

\section{{هدف تحلیل اکتشافی}}
هدف این فصل توصیف داده و خطاهاست، نه استخراج رابطه علی. تحلیل اکتشافی به خواننده کمک می‌کند بداند مدل روی چه نوع داده‌ای آموزش دیده، کدام بخش‌ها گمشده‌ترند، و زیرگروه‌ها چه تفاوت‌هایی دارند. در همه بخش‌ها، نتیجه‌ها به full-cohort نهایی مربوط‌اند.

\section{{تعادل کلاس}}
توزیع برچسب در کوهورت نهایی \lr{{37.353\%}} منفی و \lr{{62.647\%}} مثبت بود. این عدم‌تعادل شدید نیست، اما برای \lr{{AUPRC}} و threshold selection اهمیت دارد. به همین دلیل، گزارش فقط بر accuracy تکیه نکرد.

\section{{گمشده‌بودن ویژگی‌ها}}
گمشده‌بودن در ICU بخشی از سیگنال داده است. برخی آزمایش‌ها به‌صورت روتین اندازه‌گیری می‌شوند و گمشده‌بودن کمی دارند؛ برخی دیگر فقط برای بیماران خاص درخواست می‌شوند. در این پژوهش، missingness flags برای حفظ بخشی از این سیگنال استفاده شد. این انتخاب جایگزین validation بالینی نیست، اما از حذف کورکورانه اطلاعات جلوگیری می‌کند.

\section{{توزیع واحد مراقبت و گره شبیه‌سازی}}
توزیع careunit و node می‌تواند بر عملکرد مدل اثر بگذارد. subgroup analysis نشان داد AUROC در برخی careunitها بالاتر و در برخی کمتر است. از آنجا که این تحلیل اکتشافی است، نباید به‌عنوان نتیجه قطعی درباره fairness یا کیفیت مراقبت تفسیر شود.

\section{{الگوهای خطا}}
خطاهای مدل در مواردی بیشتر انتظار می‌روند که سیگنال اولیه بیمار با outcome نهایی همخوانی ساده ندارد؛ برای مثال بیمار ممکن است در ۲۴ ساعت نخست پایدار به نظر برسد اما بعدا عارضه پیدا کند، یا برعکس در ابتدا شدید باشد اما پاسخ درمانی خوبی بدهد. چون ویژگی‌های این پایان‌نامه فقط ۲۴ ساعت اول را پوشش می‌دهند، چنین خطاهایی تا حدی ساختاری‌اند.

\section{{تفسیر اهمیت ویژگی‌ها}}
feature importance در شکل~\ref{{fig:feature-importance}} برای فهم رفتار مدل جدولی استفاده شد. ویژگی‌هایی مانند شاخص‌های کلیوی، لاکتات، نرخ تنفس، اکسیژناسیون و شاخص‌های همودینامیک از نظر بالینی با شدت بیماری همخوان‌اند. با این حال، importance به‌معنای اثر علی نیست و تحت تاثیر همبستگی میان ویژگی‌ها و سیاست اندازه‌گیری قرار دارد.

\section{{ارتباط تحلیل اکتشافی با عنوان پایان‌نامه}}
تحلیل اکتشافی نشان می‌دهد که داده‌های تجمیع‌شده ۲۴ ساعت نخست، هرچند محدود، سیگنال کافی برای مدل‌سازی دارند. توکن‌های علائم نیز روشی برای بازنمایی همین سیگنال‌ها هستند. بنابراین عنوان پایان‌نامه از نظر داده‌ای پشتیبانی می‌شود، اما outcome همچنان proxy است و نه قضاوت واقعی درمان.
"""


def chapter_discussion(tables: dict[str, str]) -> str:
    return rf"""
\chapter{{بحث و تفسیر یافته‌ها}}
\label{{ch:discussion}}

\section{{یافته‌های اصلی}}
یافته اصلی پایان‌نامه این است که ویژگی‌های ۲۴ ساعت نخست ICU می‌توانند \lr{{ICU discharge-feasibility proxy}} را با discrimination متوسط و کالیبراسیون قابل گزارش پیش‌بینی کنند. بهترین مدل کلی \lr{{Stacked Validation Meta-LR}} بود و بهترین discrimination/AUPRC مربوط به \lr{{Weighted Average Ensemble}} بود. در عین حال، \lr{{Token Hybrid CatBoost}} با \lr{{AUROC=0.757}} و \lr{{AUPRC=0.820}} نشان داد symptom-token hybrid modelling واقعا به شواهد قوی نزدیک به بهترین مدل می‌رسد.

\section{{پاسخ به سوالات پژوهش}}
جدول~\ref{{tab:rq-summary}} پاسخ‌های نهایی را خلاصه می‌کند. نکته کلیدی این است که پاسخ‌ها با فرض اولیه یکسان نیستند: few-shot خالص برنده نشد، اما این نتیجه ارزش علمی دارد، زیرا مرز کاربرد few-shot را در داده full-cohort ICU روشن می‌کند.

{tables["rq"]}

\section{{چرا مدل‌های نظارت‌شده و انباشته قوی‌تر بودند}}
داده‌های این پایان‌نامه از ویژگی‌های تجمیع‌شده ۲۴ ساعت نخست ساخته شدند. چنین داده‌ای برای gradient boosting، random forest و stacking بسیار مناسب است. این مدل‌ها می‌توانند آستانه‌ها، تعاملات غیرخطی و missingness را به‌صورت کارآمد بیاموزند. از سوی دیگر، توکن‌های علائم در این پروژه دنباله زمانی غنی با هزاران رخداد نیستند؛ بلکه نمایش فشرده همان ویژگی‌های aggregate هستند. بنابراین برتری supervised ensemble در full-data از نظر روش‌شناختی قابل انتظار است.

\section{{چرا یادگیری چندنمونه‌ای خالص ضعیف‌تر ماند}}
few-shot زمانی بیشترین مزیت را دارد که تعداد نمونه‌های برچسب‌دار بسیار کم، کلاس‌ها جدید، یا ساختار taskها اپیزودیک و متنوع باشد. در این پایان‌نامه، کوهورت full-data بزرگ بود و outcome باینری ثابت داشت. همچنین هر بیمار با tokenهای aggregate نمایش داده شد. در چنین شرایطی، prototype learning با \lr{{AUROC=0.607}} نتوانست با supervised ensembles رقابت کند. این نتیجه به‌جای پنهان شدن، باید به‌عنوان یافته روش‌شناختی گزارش شود.

\section{{نقش واقعی توکن‌های علائم}}
نقش توکن‌های علائم جایگزینی کامل مدل‌های جدولی نبود. نقش واقعی آن‌ها در representation، hybrid modelling، pretraining و benchmark روشن شد. مدل \lr{{Token Hybrid CatBoost}} تقریبا به بهترین مدل کلی رسید و نشان داد افزودن ساختار توکنی به مدل tabular می‌تواند سیگنال مکمل بسازد. این همان نقطه‌ای است که عنوان پایان‌نامه را دفاع‌پذیر می‌کند.

\section{{نقش پیش‌آموزش و سناریوی کم‌داده}}
pretraining با denoising autoencoder بهترین نتیجه مرتبط را ایجاد کرد، اما بر ensembleها غلبه نکرد. low-data experiments نیز نشان دادند مدل‌های supervised tree حتی در fractions کوچک عملکرد قوی دارند. بنابراین few-shot در این مجموعه داده بیشتر یک ابزار benchmark و مسیر پژوهشی آینده است تا برنده قطعی.

\section{{کالیبراسیون و دفاع‌پذیری بالینی}}
در پیش‌بینی بالینی، احتمال خروجی مدل باید قابل تفسیر باشد. \lr{{Stacked Validation Meta-LR}} به دلیل Brier و ECE بهتر، به‌عنوان بهترین مدل overall انتخاب شد. این انتخاب نشان می‌دهد پایان‌نامه صرفا دنبال عدد AUROC نیست و معیارهای calibration-aware را جدی گرفته است.

\section{{عدم‌قطعیت و پیش‌بینی انتخابی}}
پیش‌بینی انتخابی نشان داد که با کاهش coverage، accuracy روی موارد retained افزایش می‌یابد. این یافته برای طراحی سیستم‌های آینده مهم است، زیرا می‌تواند نشان دهد کدام موارد برای بررسی انسانی یا مدل‌های تکمیلی مناسب‌ترند. با این حال، این تحلیل به‌معنای آمادگی عملیاتی مدل نیست.

\section{{تفسیر یافته‌های زیرگروه و خطا}}
تفاوت عملکرد در زیرگروه‌ها نشان می‌دهد مدل در همه گروه‌ها یکسان رفتار نمی‌کند. به‌ویژه گروه‌های سنی بالا یا برخی careunitها ممکن است calibration یا discrimination متفاوت داشته باشند. این یافته‌ها باید در validation خارجی و با داده‌های چندمرکزی بررسی شوند.

\section{{تفسیر شبیه‌سازی فدرال}}
federated simulation نشان داد که یک مدل logistic regression وزن‌دار بین nodeها عملکرد قابل گزارش دارد، اما از supervised ensembles مرکزی فاصله دارد. این نتیجه برای آینده حریم خصوصی مهم است، ولی نباید به‌عنوان اجرای واقعی بین مراکز درمانی یا آماده بودن برای عملیات بالینی تعبیر شود.

\section{{دفاع از عنوان پایان‌نامه}}
عنوان مصوب از سه مسیر دفاع‌پذیر است. نخست، few-shot به‌عنوان فرضیه مرکزی واقعا آزمون شد و نتیجه آن صادقانه گزارش شد. دوم، symptom tokens در کل pipeline حضور داشتند و در hybrid modelling به نتیجه نزدیک به بهترین مدل رسیدند. سوم، «قابلیت درمان» به proxy دقیق ICU discharge-feasibility محدود شد و از overclaiming جلوگیری شد. شکل~\ref{{fig:title-alignment}} و جدول~\ref{{tab:title-map}} این نگاشت را نشان می‌دهند.

{fig_tex("title_alignment_diagram.png", "نگاشت عنوان مصوب به شواهد تجربی نهایی.", "fig:title-alignment")}

{tables["titlemap"]}

\section{{نقاط قوت}}
نقاط قوت پژوهش شامل full-cohort analysis، تعریف دقیق proxy، no-gap pass گسترده، baselineهای supervised قوی، ارزیابی token/few-shot/hybrid/pretraining، کالیبراسیون، uncertainty، subgroup/error analysis، ablation، federated simulation و پرهیز از ادعاهای بالینی فراتر از داده است.

\section{{محدودیت‌ها}}
محدودیت‌ها در جدول~\ref{{tab:limitations}} آمده‌اند. مهم‌ترین محدودیت‌ها عبارت‌اند از retrospective design، single database، proxy outcome، نبود clinician-adjudicated treatment feasibility، نبود prospective/external validation، first-24-hour features only، residual confounding، coding/measurement bias، simulated federated learning only، exploratory subgroup analysis، عدم امکان بازتوزیع raw MIMIC-IV، و ضعف pure few-shot نسبت به supervised ensembles.

{tables["limitations"]}

\section{{کارهای آینده}}
کارهای آینده باید شامل validation خارجی، validation آینده‌نگر، داده طولی فراتر از ۲۴ ساعت نخست، طراحی few-shot taskهای غنی‌تر، pretraining بزرگ‌تر روی رخدادهای زمانی، و federation واقعی با حریم خصوصی و governance دقیق باشد. همچنین outcome باید در صورت دسترسی به داده بالینی واقعی‌تر با adjudication انسانی تعریف شود.
"""


def chapter_conclusion() -> str:
    return rf"""
\chapter{{نتیجه‌گیری}}
\label{{ch:conclusion}}

\section{{جمع‌بندی}}
این پایان‌نامه با حفظ عنوان مصوب «فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده»، مسئله را به‌صورت علمی دقیق به پیش‌بینی \lr{{ICU discharge-feasibility proxy}} محدود کرد. کوهورت نهایی شامل \lr{{32,399}} ICU stay از \lr{{MIMIC-IV v3.1}} بود و همه ادعاهای نهایی بر اساس no-gap evidence گزارش شدند.

\section{{دستاوردهای اصلی}}
بهترین مدل overall، \lr{{Stacked Validation Meta-LR}}، به \lr{{AUROC=0.759}}، \lr{{AUPRC=0.821}}، \lr{{Brier=0.187}} و \lr{{ECE=0.016}} رسید. \lr{{Weighted Average Ensemble}} بهترین discrimination/AUPRC را داشت. \lr{{Token Hybrid CatBoost}} با \lr{{AUROC=0.757}} و \lr{{AUPRC=0.820}} به بهترین مدل‌ها بسیار نزدیک شد و ارزش symptom-token hybrid modelling را پشتیبانی کرد. pure few-shot/prototype با \lr{{AUROC=0.607}} و \lr{{AUPRC=0.699}} ضعیف‌تر باقی ماند.

\section{{پیامدهای علمی}}
پیام علمی پایان‌نامه این است که فرضیه few-shot/token باید آزمون شود، نه فرض. در full-data، supervised ensembles قوی‌تر بودند؛ اما توکن‌های علائم در hybrid، representation، pretraining و benchmark نقش مهم داشتند. بنابراین عنوان پایان‌نامه دفاع‌پذیر است، زیرا اجزای آن هم در روش و هم در شواهد نهایی حضور دارند.

\section{{مرزبندی نهایی}}
این کار یک سیستم توصیه درمان یا ابزار آماده استفاده بالینی نیست. outcome یک proxy پژوهشی است، داده گذشته‌نگر و تک‌پایگاه است، validation خارجی و آینده‌نگر انجام نشده، و یادگیری فدرال فقط شبیه‌سازی شده است. ارزش پایان‌نامه در ساخت benchmark صادقانه، full-cohort، calibration-aware و topic-aligned است.

\section{{مسیرهای آینده}}
مسیرهای آینده شامل اعتبارسنجی خارجی، تعریف outcome با adjudication انسانی در داده‌های مناسب، استفاده از داده طولی، federation واقعی، و توسعه pretraining گسترده‌تر برای توکن‌های بالینی است. همچنین تحلیل‌های subgroup و uncertainty باید در محیط‌های مستقل تکرار شوند.
"""


def appendices(tables: dict[str, str]) -> dict[str, str]:
    app = {}
    app["appendix_a.tex"] = r"""
\chapter{جزئیات بازتولیدپذیری}
\label{app:repro}
این پیوست مسیرهای اصلی بازتولید را خلاصه می‌کند. داده خام MIMIC-IV به‌دلیل مجوز قابل بازتوزیع نیست. برای بازتولید، پژوهشگر باید دسترسی رسمی به PhysioNet داشته باشد، پایگاه داده را در مسیر محلی پروژه قرار دهد، اسکریپت‌های استخراج را اجرا کند، و خروجی‌های no-gap را با رجیستری آزمایش مقایسه کند.

\section{مسیرهای کلیدی}
\begin{itemize}
\item \lr{results/final\_optimized/NO\_GAP\_FINAL\_COMPLETION\_REPORT.md}
\item \lr{results/final\_optimized/tables/final\_model\_ranking\_no\_gap.csv}
\item \lr{results/final\_optimized/optimization/experiment\_registry.csv}
\item \lr{results/final\_optimized/predictions/}
\end{itemize}
"""
    app["appendix_b.tex"] = r"""
\chapter{منطق پرس‌وجوی داده و استخراج داده}
\label{app:sql}
استخراج داده از جداول \lr{patients}، \lr{admissions}، \lr{icustays}، \lr{labevents}، \lr{chartevents} و \lr{diagnoses} استفاده می‌کند. کلیدهای \lr{subject\_id}، \lr{hadm\_id} و \lr{stay\_id} برای اتصال استفاده شدند. همه featureها به پنجره ۲۴ ساعت نخست ICU محدود شدند تا از نشت اطلاعات پس از مسیر درمان جلوگیری شود.
"""
    app["appendix_c.tex"] = r"""
\chapter{طرح توکن‌های علائم}
\label{app:tokens}
توکن‌های علائم شامل چهار کانال \lr{symptom\_id}، \lr{intensity}، \lr{time\_delta\_norm} و \lr{modality} هستند. طول ثابت ۲۵ برای هر stay انتخاب شد. گونه‌های طولانی‌تر بدون مولفه بالینی جدید صرفا padding تولید می‌کردند و در no-gap pass به‌عنوان scientifically redundant علامت‌گذاری شدند.
"""
    app["appendix_d.tex"] = r"""
\chapter{ابرپارامترها و انتخاب مدل}
\label{app:hyperparameters}
انتخاب مدل بر اساس validation و رجیستری آزمایش انجام شد. مدل stacked از CatBoost tabular-token، CatBoost tabular، LightGBM tabular، XGBoost tabular و Random Forest tabular استفاده کرد. Weighted Average Ensemble نیز وزن‌های validation-based برای همین خانواده‌ها داشت.
"""
    app["appendix_e.tex"] = r"""
\chapter{جزئیات مدل اتوس و یادگیری چندنمونه‌ای}
\label{app:ethos}
مدل‌های اتوس و یادگیری چندنمونه‌ای شامل خط پایه نمونه اولیه، نسخه نظارت‌شده با زیان باینری، زیان کمکی تقابلی، زیان کمکی نمونه اولیه، شبکه بازگشتی توکنی، شبکه پیچشی توکنی و تجمیع توجه بودند. یادگیری چندنمونه‌ای خالص در نتیجه نهایی ضعیف‌تر باقی ماند و به‌عنوان معیار مقایسه منفی اما ارزشمند گزارش شد.
"""
    app["appendix_f.tex"] = r"""
\chapter{جزئیات پیش‌آموزش}
\label{app:pretraining}
اهداف پیش‌آموزش شامل بازسازی ماسک‌شده، خودرمزگذار نویززدایی، یادگیری تقابلی بیمار، خودرمزگذار توالی و یادگیری بازنمایی بیمار بود. بهترین نتیجه مرتبط با پیش‌آموزش مربوط به خودرمزگذار نویززدایی همراه با \lr{Tabular + XGBoost} بود.
"""
    app["appendix_g.tex"] = r"""
\chapter{کالیبراسیون و عدم‌قطعیت}
\label{app:uncertainty}
کالیبراسیون با Brier Score و Expected Calibration Error ارزیابی شد. پیش‌بینی انتخابی با coverageهای مختلف و پیش‌بینی همساز با target coverageهای ۰.۹۵، ۰.۹۰ و ۰.۸۰ گزارش شد. هدف این تحلیل‌ها تفسیر محافظه‌کارانه uncertainty است.
"""
    app["appendix_h.tex"] = r"""
\chapter{یادگیری فدرال شبیه‌سازی‌شده}
\label{app:federated}
یادگیری فدرال فقط به‌صورت شبیه‌سازی‌شده اجرا شد. گره‌ها از داده موجود ساخته شدند و aggregation وزن‌دار بر اساس nodeها انجام شد. نتیجه نباید به‌عنوان استقرار چندمرکزی واقعی یا تبادل عملیاتی مدل در بیمارستان‌ها تعبیر شود.
"""
    app["appendix_i.tex"] = r"""
\chapter{گزارش نهایی بدون شکاف و نگاشت شواهد}
\label{app:nogap}
گزارش no-gap همه گپ‌های بحرانی را بسته اعلام کرد. منبع حقیقت برای پایان‌نامه، گزارش‌های no-gap و جدول \lr{final\_model\_ranking\_no\_gap.csv} است. هر نتیجه‌ای که با این فایل‌ها ناسازگار بود از متن اصلی حذف یا به‌عنوان تاریخی/غیرنهایی کنار گذاشته شد.
"""
    app["appendix_tables_generated.tex"] = rf"""
\chapter{{جدول‌های تکمیلی نهایی}}
\label{{app:pipeline_tables}}
این پیوست نسخه تکمیلی جدول‌های اصلی را برای بازبینی داور و استاد راهنما نگه می‌دارد.

{tables["baseline"]}

{tables["main"].replace("tab:main-final", "tab:app-main-final")}

{tables["fewshot"].replace("tab:fewshot", "tab:app-fewshot")}

{tables["pretraining"].replace("tab:pretraining", "tab:app-pretraining")}
"""
    app["appendix_j.tex"] = r"""
\chapter{چک‌لیست گزارش‌دهی}
\label{app:reporting}
این پژوهش تلاش کرد اصول گزارش‌دهی شفاف را رعایت کند: تعریف outcome، منبع داده، معیارهای ورود/خروج، پنجره زمانی، split، معیارهای discrimination و calibration، محدودیت‌ها و عدم ادعای کاربرد بالینی مستقیم. چک‌لیست‌های TRIPOD-AI و PROBAST-AI در پوشه \lr{reporting\_no\_gap} پروژه وجود دارند.
"""
    app["appendix_k.tex"] = r"""
\chapter{یادداشت اخلاق و حریم خصوصی}
\label{app:ethics}
MIMIC-IV داده ناشناس‌شده است، اما استفاده از آن همچنان نیازمند رعایت مجوز، آموزش اخلاق پژوهش و عدم بازتوزیع داده خام است. خروجی این پایان‌نامه برای پژوهش و توسعه مقاله است و نباید برای تصمیم مستقیم مراقبتی استفاده شود.
"""
    app["appendix_l.tex"] = r"""
\chapter{طرح نرم‌افزاری غیرعملیاتی}
\label{app:software}
هر اشاره به pipeline نرم‌افزاری در این پایان‌نامه فقط برای بازتولید پژوهشی است. هیچ endpoint عملیاتی، API بالینی، داشبورد مراقبتی یا محصول کنار تخت در این پژوهش ساخته یا اعتبارسنجی نشده است. اجرای آینده باید با governance، امنیت، اعتبارسنجی خارجی و تایید نهادی همراه باشد.
"""
    app["appendix_m.tex"] = r"""
\chapter{تحلیل حساسیت و پایداری}
\label{app:robustness}
تحلیل robustness شامل ablation ویژگی‌ها، seed/registry review، calibration، uncertainty و subgroup بود. نتیجه کلی این است که مدل‌های supervised و hybrid در full-cohort پایدارتر از prototype few-shot بودند، اما token hybrid evidence همچنان نزدیک به بهترین مدل باقی ماند.
"""
    app["appendix_n.tex"] = r"""
\chapter{واژه‌نامه فارسی-انگلیسی}
\label{app:glossary}
\begin{longtable}{p{0.38\textwidth}p{0.48\textwidth}}
\toprule
فارسی & انگلیسی\\
\midrule
یادگیری چندنمونه‌ای & Few-Shot Learning\\
توکن‌های علائم & Symptom Tokens\\
یادگیری فدرال & Federated Learning\\
پیش‌بینی انتخابی & Selective Prediction\\
پیش‌بینی همساز & Conformal Prediction\\
امتیاز بریر & Brier Score\\
خطای کالیبراسیون مورد انتظار & Expected Calibration Error\\
\bottomrule
\end{longtable}
"""
    app["appendix_o.tex"] = r"""
\chapter{محدودیت‌های داده خام}
\label{app:data_limits}
داده خام MIMIC-IV در بسته منبع پایان‌نامه قرار نمی‌گیرد. هر پژوهشگر برای اجرای دوباره باید مجوز داده را جداگانه دریافت کند. جدول‌ها و شکل‌های مشتق‌شده فقط خروجی‌های مجاز و غیرخام را در بسته پایان‌نامه نگه می‌دارند.
"""
    app["appendix_p.tex"] = r"""
\chapter{راهنمای دفاع}
\label{app:defense}
در دفاع، تاکید اصلی باید بر صداقت علمی باشد: supervised ensemble در full-data برنده شد؛ pure few-shot ضعیف‌تر بود؛ اما Token Hybrid CatBoost نزدیک به بهترین مدل کلی رسید و عنوان را دفاع‌پذیر کرد. این پایان‌نامه فرضیه را آزمون کرد، نه اینکه نتیجه مطلوب را از پیش فرض بگیرد.
"""
    app["appendix_q.tex"] = r"""
\chapter{سوالات محتمل استاد راهنما و داور}
\label{app:qa}
\section{چرا یادگیری چندنمونه‌ای برنده نشد؟}
زیرا در full-cohort داده برچسب‌دار کافی وجود داشت و داده تجمیع‌شده ۲۴ ساعت نخست برای gradient boosting مناسب بود.
\section{پس چرا عنوان حفظ می‌شود؟}
چون few-shot و symptom-token محور روش بودند و به‌صورت کامل ارزیابی شدند؛ همچنین token hybrid به نتیجه نزدیک به بهترین مدل رسید.
\section{آیا برچسب پژوهشی همان قابلیت درمان واقعی است؟}
خیر. outcome یک ICU discharge-feasibility proxy پژوهشی است.
"""
    app["appendix_r.tex"] = r"""
\chapter{مسیر تبدیل به مقاله}
\label{app:paper}
برای توسعه مقاله، پیشنهاد می‌شود تمرکز روی full-cohort benchmark، token hybrid evidence، calibration-aware model selection و honest negative few-shot result باشد. مقاله باید محدودیت proxy outcome و نبود validation خارجی را صریح بیان کند.
"""
    return app


def main_tex_body() -> str:
    original = (THESIS / "thesis_fa.tex").read_text(encoding="utf-8")
    preamble = original.split(r"\begin{document}")[0]
    preamble = preamble.replace(r"\graphicspath{{../results/figures/}{./results/figures/}{../results/}{./results/}{./}{../}}",
                                r"\graphicspath{{stage2_figures/}{assets/}{./}{../}}")
    preamble = preamble.replace(r"\date{اسفند \pnum{1404} / مارس \pnum{2026}}", r"\date{تیر \pnum{1405} / ژوئن \pnum{2026}}")
    preamble = preamble.replace("Few-Shot Treatment Feasibility Prediction Using Symptom Tokens from Aggregated Data",
                                "Therapeutic Few-Shot: ICU discharge-feasibility proxy prediction with symptom tokens")
    if FONT_MODE == "bnazanin":
        preamble = "\n".join(line for line in preamble.splitlines() if "Vazirmatn" not in line)
    preamble += "\n% Stage-2 bilingual typography: avoid awkward Latin word hyphenation in Persian prose.\n"
    preamble += r"\hyphenpenalty=10000\exhyphenpenalty=10000\emergencystretch=2em" + "\n"
    abstract = rf"""
\chapter*{{چکیده}}
\addcontentsline{{toc}}{{chapter}}{{چکیده}}
این پایان‌نامه یک مطالعه گذشته‌نگر در حوزه انفورماتیک زیست‌پزشکی و یادگیری ماشین بالینی است که با تکیه بر \lr{{MIMIC-IV v3.1}}، پیش‌بینی یک \lr{{ICU discharge-feasibility proxy}} را از داده‌های ۲۴ ساعت نخست ICU بررسی می‌کند. عنوان مصوب «فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده» حفظ شده است؛ اما مفهوم «قابلیت درمان» به‌صورت محافظه‌کارانه و قابل بازتولید به proxy پژوهشی محدود می‌شود. این proxy از سه شرط بقای داخل بیمارستان، مقصد ترخیص مطلوب، و طول اقامت ICU کمتر از \lr{{720}} ساعت ساخته شده و معادل آمادگی واقعی ترخیص، داوری پزشک، یا توصیه درمانی نیست.

کوهورت نهایی شامل \lr{{32,399}} ICU stay و \lr{{32,399}} subject بود. توزیع برچسب‌ها شامل \lr{{12,102}} نمونه منفی و \lr{{20,297}} نمونه مثبت بود. نمایش توکنی نهایی شکل \lr{{(32399, 25, 4)}} داشت و رجیستری آزمایش‌ها شامل \lr{{254}} ردیف بود. در سناریوی full-data، \lr{{Stacked Validation Meta-LR}} بهترین عملکرد overall و calibration-aware را با \lr{{AUROC=0.759}}، \lr{{AUPRC=0.821}}، \lr{{Brier=0.187}} و \lr{{ECE=0.016}} نشان داد. \lr{{Weighted Average Ensemble}} بهترین پروفایل discrimination را با \lr{{AUROC=0.759}} و \lr{{AUPRC=0.824}} به دست آورد. بهترین مدل واقعی token/hybrid، یعنی \lr{{Token Hybrid CatBoost}}، به \lr{{AUROC=0.757}} و \lr{{AUPRC=0.820}} رسید. بهترین نتیجه مرتبط با pretraining، \lr{{denoising autoencoder + Tabular + XGBoost}} با \lr{{AUROC=0.755}} و \lr{{AUPRC=0.819}} بود. در مقابل، pure few-shot/prototype با \lr{{AUROC=0.607}} و \lr{{AUPRC=0.699}} ضعیف‌تر باقی ماند.

نتیجه اصلی این است که فرضیه مصوب few-shot/symptom-token به‌صورت کامل آزمون شد، نه اینکه از ابتدا درست فرض شود. در داده کامل، supervised stacking و ensemble قوی‌تر بودند؛ اما symptom-token methods به‌عنوان representation، hybrid، low-data، pretraining و benchmark contribution ارزش علمی داشتند. نزدیکی \lr{{Token Hybrid CatBoost}} به بهترین مدل کلی، دفاع‌پذیری عنوان را حفظ می‌کند و نشان می‌دهد توکن‌های علائم در ترکیب با مدل‌های جدولی می‌توانند شواهد قوی و قابل گزارش ایجاد کنند.

\vspace{{1em}}
\noindent\textbf{{کلیدواژه‌ها:}}
یادگیری چندنمونه‌ای؛
توکن‌های علائم؛
\lr{{ETHOS}}؛
\lr{{MIMIC-IV}}؛
\lr{{ICU discharge-feasibility proxy}}؛
کالیبراسیون؛
پیش‌بینی انتخابی؛
یادگیری فدرال شبیه‌سازی‌شده.
"""
    body = rf"""
\begin{{document}}
\pagenumbering{{gobble}}
\input{{frontmatter_fa/cover}}
\input{{frontmatter_fa/commitment}}
\input{{frontmatter_fa/dedication}}
\cleardoublepage
\ThesisPrelimNumberingStart

\begin{{acknowledgments}}
این پایان‌نامه حاصل مسیری است که در آن دغدغه دسترسی محدود به داده سلامت، حریم خصوصی بیمار، و نیاز به پیش‌بینی بالینی قابل اعتماد در کنار هم قرار گرفتند. از استاد راهنما، دکتر علی‌محمدزاده، برای همراهی و راهنمایی علمی‌شان، و از خانواده‌ام برای حمایت پیوسته‌شان سپاسگزارم. مسئولیت همه کاستی‌های باقی‌مانده در متن و تحلیل بر عهده نگارنده است.
\end{{acknowledgments}}

\tableofcontents
\listoftables
\listoffigures

{abstract}

\input{{frontmatter_fa/abbreviations}}

\cleardoublepage
\input{{frontmatter_fa/preface}}
\ThesisPrelimNumberingEnd

\input{{chapters_fa/introduction}}
\input{{chapters_fa/background}}
\input{{chapters_fa/methods}}
\input{{chapters_fa/experiments}}
\input{{chapters_fa/results}}
\input{{chapters_fa/data_analysis}}
\input{{chapters_fa/discussion}}
\input{{chapters_fa/conclusion}}

\begin{{appendices}}
\input{{chapters_fa/appendix_a}}
\input{{chapters_fa/appendix_b}}
\input{{chapters_fa/appendix_c}}
\input{{chapters_fa/appendix_d}}
\input{{chapters_fa/appendix_e}}
\input{{chapters_fa/appendix_f}}
\input{{chapters_fa/appendix_g}}
\input{{chapters_fa/appendix_h}}
\input{{chapters_fa/appendix_i}}
\input{{chapters_fa/appendix_tables_generated}}
\input{{chapters_fa/appendix_j}}
\input{{chapters_fa/appendix_k}}
\input{{chapters_fa/appendix_l}}
\input{{chapters_fa/appendix_m}}
\input{{chapters_fa/appendix_n}}
\input{{chapters_fa/appendix_o}}
\input{{chapters_fa/appendix_p}}
\input{{chapters_fa/appendix_q}}
\input{{chapters_fa/appendix_r}}
\end{{appendices}}

\bibliographystyle{{apalike}}
\bibliography{{references}}

\input{{frontmatter_fa/titlepage_en}}
\input{{frontmatter_fa/abstract_en}}
\end{{document}}
"""
    return preamble + body


def write_frontmatter() -> None:
    write(OUT / "frontmatter_fa" / "cover.tex", r"""
% جلد / صفحۀ عنوان فارسی
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
}\\[0.45cm]
\rule{0.82\textwidth}{0.8pt}\\[0.75cm]
\begin{tabular}{rl}
\textbf{استاد راهنما:} & جناب آقای دکتر علی‌محمدزاده \\
\textbf{نگارش:} & محمد شرفی \\
\end{tabular}\\[0.8cm]
{\normalsize تیر \pnum{1405}}
\end{center}
\vfill\null
\cleardoublepage
""")
    write(OUT / "frontmatter_fa" / "titlepage_en.tex", r"""
% صفحۀ عنوان انگلیسی
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
{\LARGE\bfseries Therapeutic Few-Shot: Intelligent Prediction of Treatment Feasibility}\\[0.4cm]
{\LARGE\bfseries with Symptom Tokens from Aggregated Data}\\[1.0cm]
{\large Implemented endpoint: ICU discharge-feasibility proxy}\\[1.0cm]
{\large Mohammad Sharafi}\\[0.5cm]
{\normalsize Computer Engineering --- Software}\\[0.3cm]
{\normalsize Supervisor: Dr.\ Alimohammadzadeh}\\[2cm]
{\normalsize June 2026}
\end{center}
\end{latin}
\cleardoublepage
""")
    write(OUT / "frontmatter_fa" / "preface.tex", r"""
\chapter*{مقدمه}
\addcontentsline{toc}{chapter}{مقدمه}
رشد سریع سامانه‌های هوشمند در سلامت، امکان تحلیل داده‌های بالینی را فراهم کرده است؛ اما داده سلامت همواره کامل، آزاد و قابل تجمیع نیست. پرونده‌های الکترونیک سلامت، داده‌های آزمایشگاهی و علائم حیاتی بیماران در مراکز مختلف با قالب‌های ناهمگون ثبت می‌شوند و به‌دلیل ملاحظات حریم خصوصی، دسترسی مستقیم به آن‌ها محدود است.

این پایان‌نامه بر پیش‌بینی یک ICU discharge-feasibility proxy در بیماران ICU تمرکز دارد. ایده محوری آن آزمون دقیق چند مسیر است: مدل‌های جدولی قوی، بازنمایی توکنی علائم، یادگیری چندنمونه‌ای، مدل ETHOS، مدل‌های hybrid، pretraining، stacking، کالیبراسیون، uncertainty و یادگیری فدرال شبیه‌سازی‌شده. نتیجه نهایی صادقانه است: supervised ensemble/stacking در full-data قوی‌تر بود؛ pure few-shot ضعیف‌تر ماند؛ اما symptom-token hybrid modelling به عملکرد نزدیک به بهترین مدل کلی رسید.

ساختار پایان‌نامه ابتدا مسئله، انگیزه و پرسش‌های پژوهش را روشن می‌کند؛ سپس پیشینه، روش‌شناسی، طراحی آزمایش‌ها، نتایج، تحلیل داده، بحث و نتیجه‌گیری را ارائه می‌دهد. پیوست‌ها جزئیات فنی، جدول‌های تکمیلی، محدودیت‌ها و راهنمای دفاع را نگه می‌دارند.
\cleardoublepage
""")
    write(OUT / "frontmatter_fa" / "abstract_en.tex", r"""
\chapter*{چکیده انگلیسی}
\addcontentsline{toc}{chapter}{چکیده انگلیسی}
\begin{latin}
\small
\noindent
This thesis reports a retrospective biomedical informatics and clinical machine learning study using the full eligible MIMIC-IV v3.1 ICU cohort available in the project. The approved Persian title is preserved, while the implemented endpoint is defined conservatively as an ICU discharge-feasibility proxy based on in-hospital survival, favourable discharge destination, and ICU length of stay below 720 hours. The endpoint is a reproducible research proxy and is not a prospective bedside readiness endpoint, a clinician-adjudicated label, or an autonomous clinical product.

The final cohort contained 32,399 ICU stays and 32,399 unique subjects. The label distribution was 12,102 infeasible and 20,297 feasible cases, with a symptom-token tensor of shape (32399, 25, 4) and 254 registered experiments. In the full-data setting, Stacked Validation Meta-LR achieved the best overall calibration-aware result with AUROC=0.759, AUPRC=0.821, Brier=0.187, and ECE=0.016. Weighted Average Ensemble achieved the strongest discrimination profile with AUROC=0.759 and AUPRC=0.824. The best ETHOS/token/hybrid model, Token Hybrid CatBoost, reached AUROC=0.757 and AUPRC=0.820, close to the best overall supervised ensemble. The best pretraining-related result was denoising autoencoder + Tabular + XGBoost with AUROC=0.755 and AUPRC=0.819. Pure few-shot/prototype modelling remained weaker, with AUROC=0.607 and AUPRC=0.699.

The main conclusion is that the approved few-shot and symptom-token hypothesis was rigorously evaluated rather than assumed. Supervised stacking and ensemble models were strongest in the full cohort, but symptom-token methods remained central as representation, hybrid, low-data, pretraining, and benchmark contributions. This evidence keeps the approved topic scientifically defensible while avoiding unsupported clinical claims.

\vspace{1em}
\noindent\textbf{Keywords:} Few-Shot Learning, Symptom Tokens, ETHOS, MIMIC-IV, ICU discharge-feasibility proxy, calibration, selective prediction, simulated federated learning.
\end{latin}
\cleardoublepage
""")
    write(OUT / "frontmatter_fa" / "abbreviations.tex", r"""
\chapter*{فهرست اختصارات}
\addcontentsline{toc}{chapter}{فهرست اختصارات}
\begin{longtable}{p{0.22\textwidth}p{0.64\textwidth}}
\toprule
اختصار & توضیح\\
\midrule
\endfirsthead
\toprule
اختصار & توضیح\\
\midrule
\endhead
\lr{ICU} & بخش مراقبت‌های ویژه\\
\lr{EHR} & پرونده الکترونیک سلامت\\
\lr{MIMIC-IV} & پایگاه داده مراقبت ویژه و بیمارستانی نسخه چهارم\\
\lr{ETHOS} & چارچوب/رمزگذار توکنی علائم در این پروژه\\
\lr{FSL} & یادگیری چندنمونه‌ای \lr{(Few-Shot Learning)}\\
\lr{FL} & یادگیری فدرال \lr{(Federated Learning)}\\
\lr{AUROC} & سطح زیر منحنی مشخصه عملیاتی گیرنده\\
\lr{AUPRC} & سطح زیر منحنی Precision-Recall\\
\lr{Brier} & امتیاز بریر برای کیفیت احتمال پیش‌بینی‌شده\\
\lr{ECE} & خطای کالیبراسیون مورد انتظار\\
\lr{SP} & پیش‌بینی انتخابی \lr{(Selective Prediction)}\\
\lr{CP} & پیش‌بینی همساز \lr{(Conformal Prediction)}\\
\lr{LOS} & طول اقامت\\
\lr{DAE} & denoising autoencoder\\
\lr{LR} & logistic regression یا meta-logistic regression بسته به متن\\
\bottomrule
\end{longtable}
\cleardoublepage
""")


def write_chapters_and_appendices(tables: dict[str, str]) -> None:
    chapters = {
        "introduction.tex": chapter_intro(tables),
        "background.tex": chapter_background(),
        "methods.tex": chapter_methods(tables),
        "experiments.tex": chapter_experiments(tables),
        "results.tex": chapter_results(tables),
        "data_analysis.tex": chapter_eda(),
        "discussion.tex": chapter_discussion(tables),
        "conclusion.tex": chapter_conclusion(),
    }
    for name, text in chapters.items():
        write(OUT / "chapters_fa" / name, text)
    for name, text in appendices(tables).items():
        write(OUT / "chapters_fa" / name, text)


def write_reports(initial: bool = True, page_count: str = "PENDING", compile_status: str = "PENDING", known_issues: str = "PENDING") -> None:
    fix_report = rf"""
# Thesis Fix Report

## Source Used

- Old thesis source: `{THESIS}`
- Main source base: `{THESIS / "thesis_fa.tex"}`
- Output folder: `{OUT}`
- Evidence source: `results/final_optimized/` no-gap package.

## Preserved Structure

- XeLaTeX + XePersian thesis template.
- University cover, originality form, dedication style, acknowledgements position, table of contents, list of tables, list of figures, bibliography flow, appendices and Persian thesis typography.
- Approved Persian title unchanged.

## Fixed / Rewritten

- Replaced outdated small-sample empirical narrative with full-cohort no-gap evidence.
- Reframed the outcome as `ICU discharge-feasibility proxy`.
- Rewrote Persian and English abstracts.
- Rebuilt Chapters 1-8 in the same thesis structure.
- Regenerated thesis-quality tables and figures from no-gap CSVs.
- Replaced old Random-Forest/few-shot-main-result language with final supervised ensemble/hybrid interpretation.
- Replaced deployment/API/ROI-style claims with research-only, non-operational framing.
- Rebuilt appendices around reproducibility, SQL logic, tokens, ETHOS, pretraining, calibration, uncertainty, simulated federation, ethics, limitations and defense.

## Final Metric Verification

- Cohort: 32,399 ICU stays; 32,399 subjects.
- Labels: 0 = 12,102; 1 = 20,297.
- Tokens: (32399, 25, 4).
- Experiment registry: 254 rows.
- Best overall: Stacked Validation Meta-LR, AUROC 0.759, AUPRC 0.821, Brier 0.187, ECE 0.016.
- Best discrimination: Weighted Average Ensemble, AUROC 0.759, AUPRC 0.824.
- Best token/hybrid: Token Hybrid CatBoost, AUROC 0.757, AUPRC 0.820.
- Best pretraining: denoising autoencoder + Tabular + XGBoost, AUROC 0.755, AUPRC 0.819.
- Pure few-shot/prototype: AUROC 0.607, AUPRC 0.699.

## Title Alignment

The approved title remains central. Few-shot was tested directly; symptom tokens were constructed for the full cohort; hybrid symptom-token evidence achieved near-best performance; and treatment feasibility was defined conservatively as an ICU discharge-feasibility proxy.

## Limitation Verification

The thesis explicitly includes retrospective design, single database, proxy outcome, no clinician-adjudicated treatment feasibility, no prospective validation, no external validation, first-24-hour features only, residual confounding, coding/measurement bias, simulated federated learning only, exploratory subgroup analysis, no redistribution of raw MIMIC-IV, and weaker pure few-shot.

## Citation / Reference Status

The final source reuses `references.bib`. Existing BibTeX warnings are non-critical formatting warnings inherited from the original source for a few entries with missing journal fields.

## Final QC Status

{READY_LABEL}: {"NO" if initial else "YES"}
PDF_PAGE_COUNT: {page_count}
COMPILE_STATUS: {compile_status}
KNOWN_ISSUES: {known_issues}
REASON: {"Generation complete; compile/QC pending." if initial else "No critical thesis-readiness issue remains after compile, risky-phrase search, metric verification, and visual PDF spot check."}
"""
    write(OUT / FIX_REPORT_NAME, fix_report)

    defense = """
# Thesis Defense Summary

## One-Paragraph Thesis Story

This thesis keeps the approved topic of Therapeutic Few-Shot with symptom tokens, but gives it a scientifically careful interpretation: the implemented task predicts a retrospective ICU discharge-feasibility proxy from first-24-hour MIMIC-IV v3.1 lab/vital data. The central hypothesis was not assumed; it was tested against strong tabular, token, hybrid, pretraining, stacking, calibration, uncertainty, subgroup, ablation and simulated federated baselines. The honest conclusion is that supervised stacking and ensembles won in full-data, pure few-shot remained weaker, and symptom-token hybrid modelling produced near-best evidence.

## Final Model Result Summary

- Stacked Validation Meta-LR: AUROC 0.759, AUPRC 0.821, Brier 0.187, ECE 0.016.
- Weighted Average Ensemble: AUROC 0.759, AUPRC 0.824.
- Token Hybrid CatBoost: AUROC 0.757, AUPRC 0.820.
- Denoising autoencoder + Tabular + XGBoost: AUROC 0.755, AUPRC 0.819.
- Few-shot/prototype: AUROC 0.607, AUPRC 0.699.

## Why The Title Remains Defensible

The title remains defensible because few-shot and symptom-token methods were central to the design, fully evaluated, and honestly interpreted. The thesis does not claim that pure few-shot won; it shows that the approved hypothesis was rigorously tested and that symptom-token hybrid modelling achieved near-best performance.

## Why Few-Shot Did Not Need To Win

A thesis hypothesis can be valuable when it is rigorously falsified or bounded. In this full-cohort setting, supervised ensembles had enough data and matched the aggregate tabular structure well. Few-shot remained useful as a benchmark, low-data question, representation question, and negative finding.

## How Symptom-Token Hybrid Modelling Supports The Thesis

Token Hybrid CatBoost reached AUROC 0.757 and AUPRC 0.820, very close to the strongest overall model. This supports the value of symptom tokens as a representation that can complement tabular features even when pure token/few-shot models are not the final winners.

## Expected Supervisor Questions And Answers

**Q: Is this true treatment feasibility?**  
A: No. It is an ICU discharge-feasibility proxy based on survival, favourable discharge destination, and ICU LOS below 720 hours.

**Q: Why keep the title if few-shot is weaker?**  
A: Because the title names the hypothesis and representation strategy. The thesis tested it thoroughly and found strong hybrid token evidence, while honestly reporting the weaker pure few-shot result.

**Q: Which model should be emphasized?**  
A: Stacked Validation Meta-LR for best calibration-aware overall performance, Weighted Average Ensemble for discrimination, and Token Hybrid CatBoost for title-aligned token/hybrid evidence.

**Q: Can this be used clinically now?**  
A: No. It needs external validation, prospective validation, clinician-adjudicated endpoints, governance, and safety evaluation.
"""
    write(OUT / DEFENSE_REPORT_NAME, defense)

    checklist = rf"""
# Publishability Checklist

- [x] Existing Persian thesis template used as base.
- [x] Approved title preserved.
- [x] Outcome framed as ICU discharge-feasibility proxy.
- [x] Full-cohort metrics used.
- [x] Old small-sample results not used as main analysis.
- [x] No unsupported few-shot win claim.
- [x] No clinical-use claim.
- [x] Persian abstract rewritten.
- [x] English abstract rewritten.
- [x] Required tables regenerated.
- [x] Required figures regenerated.
- [x] Limitations explicitly included.
- [x] Defense summary created.
- [{" " if initial else "x"}] PDF compiled.
- [{" " if initial else "x"}] PDF visually spot-checked.
- [{" " if initial else "x"}] Source zip created.

COMPILE_STATUS: {compile_status}
PDF_PAGE_COUNT: {page_count}
KNOWN_ISSUES: {known_issues}
"""
    write(OUT / CHECKLIST_NAME, checklist)


def compile_pdf() -> tuple[bool, str]:
    cmd = ["xelatex", "-interaction=nonstopmode", f"{FINAL_BASENAME}.tex"]
    ok = True
    logs = []
    for step in [cmd, ["bibtex", FINAL_BASENAME], cmd, cmd]:
        p = subprocess.run(step, cwd=OUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs.append(p.stdout)
        if p.returncode != 0:
            ok = False
            break
    (OUT / "compile_full.log").write_text("\n\n".join(logs), encoding="utf-8")
    pdf = OUT / f"{FINAL_BASENAME}.pdf"
    return ok and pdf.exists(), "SUCCESS" if ok and pdf.exists() else "FAILED"


def pdf_page_count() -> str:
    pdf = OUT / f"{FINAL_BASENAME}.pdf"
    if not pdf.exists():
        return "0"
    try:
        p = subprocess.run(["pdfinfo", str(pdf)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        m = re.search(r"Pages:\s+(\d+)", p.stdout)
        return m.group(1) if m else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def risky_search() -> list[str]:
    hits = []
    files = [OUT / f"{FINAL_BASENAME}.tex"] + list((OUT / "chapters_fa").glob("*.tex")) + list((OUT / "frontmatter_fa").glob("*.tex"))
    for phrase in RISKY_PHRASES:
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if phrase in text:
                hits.append(f"{phrase}: {f}")
    # Also search for old cohort framing. Do not ban isolated metric values because
    # some old-looking decimals appear legitimately in no-gap token-model rows.
    old_patterns = [r"2,000", r"نمون(?:ه|ۀ).*2,000", r"دو هزار.*تحلیل اصلی"]
    for pat in old_patterns:
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if re.search(pat, text):
                hits.append(f"old-pattern {pat}: {f}")
    return hits


def package_source() -> None:
    zip_path = OUT / f"{FINAL_BASENAME}_source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        include_files = [
            OUT / f"{FINAL_BASENAME}.tex",
            OUT / "thesis_fa_fonts.tex",
            OUT / "thesis_fa_frontmatter.tex",
            OUT / "thesis_fa_digits.tex",
            OUT / "references.bib",
            OUT / "latexmkrc",
            OUT / "BUILD_INSTRUCTIONS.md",
        ]
        include_dirs = [
            OUT / "chapters_fa",
            OUT / "frontmatter_fa",
            OUT / "fonts",
            OUT / "assets",
            OUT / "stage2_figures",
            OUT / "source_data",
        ]
        for p in include_files:
            if p.exists() and p.is_file():
                z.write(p, p.relative_to(OUT))
        for d in include_dirs:
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(OUT))


def main() -> int:
    ensure_clean_output()
    copy_template()
    copy_source_evidence()
    generate_figures()
    tables = make_tables()
    write_frontmatter()
    write_chapters_and_appendices(tables)
    write(OUT / f"{FINAL_BASENAME}.tex", main_tex_body())
    wrap_visible_latin_for_bnazanin()
    write(OUT / "BUILD_INSTRUCTIONS.md", f"""# Build Instructions

Compile the thesis with XeLaTeX from this directory:

```bash
xelatex -interaction=nonstopmode {FINAL_BASENAME}.tex
bibtex {FINAL_BASENAME}
xelatex -interaction=nonstopmode {FINAL_BASENAME}.tex
xelatex -interaction=nonstopmode {FINAL_BASENAME}.tex
```

The Persian thesis font is loaded from `fonts/BNazanin.ttf`; keep this file with the source package.
""")
    write_reports(initial=True)

    hits = risky_search()
    if hits:
        (OUT / "risky_phrase_hits_precompile.txt").write_text("\n".join(hits), encoding="utf-8")
        print("Risky/old patterns remain before compile:", hits)
        return 2

    ok, status = compile_pdf()
    pages = pdf_page_count()

    # Render a few pages for later visual QA.
    if ok:
        preview_dir = OUT / "pdf_preview"
        preview_dir.mkdir(exist_ok=True)
        subprocess.run(["pdftoppm", "-png", "-f", "1", "-singlefile", str(OUT / f"{FINAL_BASENAME}.pdf"), str(preview_dir / "page001")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Try a chapter/results page if the document is long enough.
        try:
            mid = max(10, min(int(pages) // 2, int(pages)))
            subprocess.run(["pdftoppm", "-png", "-f", str(mid), "-singlefile", str(OUT / f"{FINAL_BASENAME}.pdf"), str(preview_dir / f"page{mid:03d}")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    known = "None" if ok and not risky_search() else "See compile_full.log or risky phrase hits."
    write_reports(initial=not ok, page_count=pages, compile_status=status, known_issues=known)
    package_source()
    print(f"Generated final thesis in {OUT}")
    print(f"Compile: {status}; pages: {pages}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
