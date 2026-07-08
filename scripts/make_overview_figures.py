#!/usr/bin/env python3
"""
Generate original conceptual/overview figures for the thesis.
English labels (consistent with existing figures); Persian captions live in the LaTeX.
Outputs into results/full_cohort/figures/ (already on the LaTeX graphicspath).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "full_cohort", "figures")
os.makedirs(OUT, exist_ok=True)

# professional muted palette
C = dict(data="#4C6E91", token="#C77B3B", model="#5B8C5A", eval="#7A5C8E",
         win="#2E7D32", lose="#B23A3A", neutral="#5f6b76", bg="#F4F6F8",
         ink="#22303C", line="#9aa7b2")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": C["ink"]})


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, tc="white", bold=True, r=0.02):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.4, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)


def arrow(ax, p1, p2, color=None, lw=2.0, style="-|>"):
    color = color or C["line"]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, shrinkA=2, shrinkB=2, zorder=1))


# ---------------------------------------------------------------- 1. GRAPHICAL ABSTRACT
def graphical_abstract():
    fig, ax = plt.subplots(figsize=(11, 6.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 96, "Therapeutic Few-Shot — Study at a Glance", ha="center",
            fontsize=15, fontweight="bold", color=C["ink"])

    box(ax, 3, 74, 22, 13, "MIMIC-IV v3.1\n94,458 ICU stays", C["data"], fs=10)
    box(ax, 3, 56, 22, 13, "Eligibility filters\n-> 32,399 stays\n(full cohort)", C["data"], fs=9.5)
    arrow(ax, (14, 74), (14, 69))
    ax.text(14, 51, "first 24 h of each stay", ha="center", fontsize=9, style="italic", color=C["neutral"])

    # two representations
    box(ax, 33, 70, 27, 9, "Tabular features\n(labs + vitals, 24 h)", C["data"], fs=9.5)
    box(ax, 33, 57, 27, 9, "Symptom tokens\n(quantile + time, ETHOS-style)", C["token"], fs=9)
    arrow(ax, (25, 62.5), (33, 74.5)); arrow(ax, (25, 62.5), (33, 61.5))

    # three regimes
    box(ax, 67, 78, 30, 8, "Raw few-shot on tokens", C["lose"], fs=9.5)
    box(ax, 67, 66, 30, 8, "Token-Hybrid model", C["win"], fs=9.5)
    box(ax, 67, 54, 30, 8, "Full-data tabular models", C["model"], fs=9.5)
    arrow(ax, (60, 61.5), (67, 82), C["lose"]); arrow(ax, (60, 61.5), (67, 70), C["win"])
    arrow(ax, (60, 74.5), (67, 58), C["model"])

    # results band
    ax.add_patch(FancyBboxPatch((3, 22), 94, 24, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=C["bg"], ec=C["line"], lw=1.2))
    ax.text(50, 42.5, "What we found", ha="center", fontsize=12, fontweight="bold", color=C["ink"])
    ax.text(8, 36, "•  Raw few-shot on tokens stays low (AUROC 0.61–0.67) — even with GPT-2 pretraining and clinical LLMs.",
            fontsize=10, color=C["ink"])
    ax.text(8, 31, "•  Token-Hybrid CatBoost reaches the top group (AUROC 0.756) with the best calibration (ECE 0.015).",
            fontsize=10, color=C["ink"])
    ax.text(8, 26, "•  Conclusion holds across 5 labels, 7 tokenizations, 5 seeds and a 2x cohort (64,444 stays).",
            fontsize=10, color=C["ink"])

    # take-home
    box(ax, 14, 5, 72, 12,
        "Take-home: symptom tokens pay off in COMBINATION (hybrid),\nnot as a stand-alone raw few-shot learner.",
        C["ink"], fs=11.5)
    fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. PIPELINE
def pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "End-to-End Study Pipeline", ha="center", fontsize=14, fontweight="bold", color=C["ink"])
    stages = [
        ("Raw MIMIC-IV\ntables", C["data"]),
        ("Cohort\nextraction\n& filters", C["data"]),
        ("Feature +\nsymptom-token\nconstruction", C["token"]),
        ("Train / Val / Test\n(leakage-audited)", C["neutral"]),
        ("11 models\n+ ensembles\n+ federated", C["model"]),
        ("Evaluation:\nAUROC, calib.,\nDCA, DeLong", C["eval"]),
    ]
    x = 2.5; w = 14.2; gap = 1.8; y = 52; h = 20
    centers = []
    for i, (t, c) in enumerate(stages):
        box(ax, x, y, w, h, t, c, fs=9)
        centers.append((x + w, y + h / 2));
        if i < len(stages) - 1:
            arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2))
        x += w + gap
    # robustness layer beneath
    ax.add_patch(FancyBboxPatch((2.5, 16), 95.2, 22, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=C["bg"], ec=C["line"], lw=1.2))
    ax.text(50, 33.5, "Robustness layer (same single run)", ha="center", fontsize=11,
            fontweight="bold", color=C["ink"])
    for i, t in enumerate(["5 label definitions", "7 tokenization schemes",
                            "5 seeds + 5-fold CV", "ICD-9+10 cohort (64,444)"]):
        box(ax, 5 + i * 23.3, 20, 21, 8.5, t, C["neutral"], fs=8.6)
    arrow(ax, (50, 52), (50, 38), C["line"])
    fig.savefig(os.path.join(OUT, "study_pipeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. MODEL TAXONOMY
def taxonomy():
    fig, ax = plt.subplots(figsize=(11, 6.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "Models Compared in One Common Framework", ha="center", fontsize=14,
            fontweight="bold", color=C["ink"])
    box(ax, 38, 82, 24, 9, "Symptom-token data\n+ tabular features", C["ink"], fs=9.5)
    groups = [
        ("Linear", ["Logistic Regression", "Federated LR (sim.)"], C["data"], 3),
        ("Tree ensembles", ["Random Forest", "XGBoost", "LightGBM", "CatBoost"], C["model"], 27.5),
        ("Neural / few-shot", ["Neural baseline", "Few-Shot ETHOS", "GPT-2 pretrained"], C["eval"], 52),
        ("Hybrid / stacked", ["Token-Hybrid CatBoost*", "Stacked ensemble", "Weighted ensemble"], C["token"], 76.5),
    ]
    for name, items, c, x in groups:
        box(ax, x, 60, 21, 8, name, c, fs=10)
        arrow(ax, (50, 82), (x + 10.5, 68), C["line"], lw=1.4)
        for j, it in enumerate(items):
            y = 50 - j * 9
            star = it.endswith("*")
            label = it[:-1] + "\n(primary model)" if star else it
            box(ax, x, y, 21, 7, label,
                C["win"] if star else "white", ec=c, fs=7.0 if star else 8.2,
                tc="white" if star else C["ink"], bold=star)
    fig.savefig(os.path.join(OUT, "model_taxonomy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. TOKENIZATION EXAMPLE
def token_example():
    fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "From Raw Events to Symptom Tokens (one patient, first 24 h)",
            ha="center", fontsize=13, fontweight="bold", color=C["ink"])
    # raw events row
    ax.text(3, 80, "Raw timed events:", fontsize=10, fontweight="bold", color=C["ink"])
    ev = [("0h", "BUN 42"), ("2h", "Lactate 3.1"), ("3h", "HR 96"), ("11h", "BUN 55"), ("24h", "SpO2 92")]
    x = 3
    for t, v in ev:
        box(ax, x, 66, 16, 9, f"{t}\n{v}", C["data"], fs=8.6); x += 18
    # tokenization
    arrow(ax, (50, 66), (50, 56), C["line"])
    ax.text(3, 50, "Encoded tokens:", fontsize=10, fontweight="bold", color=C["ink"])
    toks = [("BUN", "Q8", C["token"]), ("dt", "1-3h", C["neutral"]), ("Lact", "Q9", C["token"]),
            ("dt", "<1h", C["neutral"]), ("HR", "Q5", C["token"]), ("dt", ">6h", C["neutral"]),
            ("BUN", "Q9", C["token"]), ("SpO2", "Q2", C["token"])]
    x = 3
    for a, b, c in toks:
        box(ax, x, 36, 11, 9, f"{a}\n{b}", c, fs=8.4); x += 12
    # two outputs
    box(ax, 12, 12, 32, 11, "Ordered sequence\n(for GPT-2 pretraining)", C["eval"], fs=9)
    box(ax, 56, 12, 32, 11, "Bag of (category x quantile)\n(for hybrid model)", C["win"], fs=9)
    arrow(ax, (35, 36), (28, 23), C["line"]); arrow(ax, (65, 36), (72, 23), C["line"])
    fig.savefig(os.path.join(OUT, "tokenization_example.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. EVIDENCE CONVERGENCE
def convergence():
    fig, ax = plt.subplots(figsize=(10, 5.8)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 95, "Convergence of Independent Evidence", ha="center", fontsize=14,
            fontweight="bold", color=C["ink"])
    cx, cy, hw, hh = 50, 48, 17, 9
    box(ax, cx - hw, cy - hh, 2 * hw, 2 * hh, "Same conclusion:\nhybrid wins,\nraw few-shot does not",
        C["ink"], fs=9.5)

    def edge_point(px, py):
        dx, dy = cx - px, cy - py
        best = 1.0
        for t in ([((cx - hw - 1) - px) / dx] if dx > 0 else []) + \
                 ([((cx + hw + 1) - px) / dx] if dx < 0 else []) + \
                 ([((cy - hh - 1) - py) / dy] if dy > 0 else []) + \
                 ([((cy + hh + 1) - py) / dy] if dy < 0 else []):
            if 0 < t < 1:
                x, y = px + t * dx, py + t * dy
                if cx - hw - 1.5 <= x <= cx + hw + 1.5 and cy - hh - 1.5 <= y <= cy + hh + 1.5:
                    best = min(best, t)
        return px + best * dx, py + best * dy

    # spoke box CENTERS, kept fully inside the 0..100 canvas (half-width 12, half-height 6.5)
    spokes = [("Main leaderboard\n(DeLong-tested)", 16, 80), ("Label sensitivity\n(5 definitions)", 84, 80),
              ("Tokenization\n(7 schemes)", 15, 48), ("Seeds + CV\n(5 seeds, 5-fold)", 85, 48),
              ("Cohort 2x\n(64,444 stays)", 27, 12), ("Clinical utility\n(decision curve)", 73, 12)]
    for t, x, y in spokes:
        box(ax, x - 12, y - 6.5, 24, 13, t, C["model"], fs=8.4)
        arrow(ax, (x, y), edge_point(x, y), C["line"], lw=1.7, style="-|>")
    fig.savefig(os.path.join(OUT, "evidence_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6. THREE-REGIME COMPARISON
def three_regime():
    Ks = [1, 5, 10, 20, 50, 100]
    fs = [0.550, 0.583, 0.603, 0.622, 0.661, 0.673]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(Ks, fs, "o-", color=C["lose"], lw=2.4, ms=7, label="Raw few-shot on tokens")
    ax.axhline(0.756, color=C["win"], lw=2.4, ls="--", label="Token-Hybrid CatBoost (full data)")
    ax.axhline(0.728, color=C["data"], lw=1.8, ls=":", label="Logistic Regression (full data)")
    ax.fill_between([1, 100], 0.50, 0.50, color="none")
    ax.annotate("persistent gap", xy=(100, 0.71), xytext=(58, 0.70),
                arrowprops=dict(arrowstyle="<->", color=C["neutral"]), color=C["neutral"], fontsize=10)
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
    ax.set_xlabel("Support examples per class (K)"); ax.set_ylabel("AUROC")
    ax.set_title("Few-shot never closes the gap to full-data models", color=C["ink"], fontweight="bold")
    ax.set_ylim(0.52, 0.79); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "three_regime_comparison.png"), dpi=300,
                                    bbox_inches="tight"); plt.close(fig)


for fn in (graphical_abstract, pipeline, taxonomy, token_example, convergence, three_regime):
    fn(); print("ok:", fn.__name__)
print("Saved overview figures to", OUT)
