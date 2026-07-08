"""Publication-quality figures for the RA3 study (Q1 revision).

Aesthetic: muted palette, single crimson accent for RA3, no on-plot titles, thin
left/bottom spines, direct labels, high resolution. Each figure carries ONE message and
the interpretation lives in the caption.

Figures
  1  fig_pareto     — privacy-utility Pareto curves over the full noise sweep
  2  fig_fidelity   — primary vs untuned-task utility (fidelity mirage)
  3  fig_attacks    — three attacks per mechanism
  4  fig_allocation — where RA3 spends protection (by feature group)
  5  fig_ablation   — contribution of each RA3 component (needs ablation.csv)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

RES = Path("results/ra3")
FIG = Path("paper_ra3/figures"); FIG.mkdir(parents=True, exist_ok=True)
DPI = 340

_pref = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _pref if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": FONT, "font.size": 12,
    "axes.labelsize": 12.5, "axes.labelcolor": "#222", "axes.labelpad": 7,
    "axes.edgecolor": "#4d4d4d", "axes.linewidth": 0.9,
    "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "legend.frameon": False, "legend.fontsize": 11,
})

INK = "#2b2b2b"; SUB = "#6b6b6b"; FAINT = "#c9c9c9"
COL = {"Raw": "#a6a6a6", "kAnon": "#6f9c93", "UniformDP": "#6b8cc4",
       "DPSynth": "#d8a24a", "RA3": "#c02b3a"}
NAME = {"Raw": "Raw", "kAnon": "k-anonymity", "UniformDP": "Uniform DP",
        "DPSynth": "DP-Synthetic", "RA3": "RA³"}
MARK = {"Raw": "*", "kAnon": "s", "UniformDP": "o", "DPSynth": "^", "RA3": "D"}
REP = {"Raw": "Raw", "kAnon": "kAnon_k10", "UniformDP": "UniformDP_s1.0",
       "DPSynth": "DPSynth_s0.25", "RA3": "RA3_s1.0"}
ORDER = ["Raw", "kAnon", "UniformDP", "DPSynth", "RA3"]


def family_of(m):
    for k in ORDER:
        if m.startswith(k):
            return k
    return None


def load():
    single = json.load(open(RES / "results.json"))
    ms = pd.read_csv(RES / "results_multiseed.csv") if (RES / "results_multiseed.csv").exists() else None
    dfj = pd.DataFrame(single["results"]).set_index("mechanism")
    abl = pd.read_csv(RES / "ablation.csv") if (RES / "ablation.csv").exists() else None
    return single, dfj, ms, abl


def val(dfj, ms, mech, col):
    if ms is not None:
        r = ms[ms["mechanism"] == mech]
        if len(r) and f"{col}_mean" in ms.columns:
            return float(r[f"{col}_mean"].iloc[0])
    return float(dfj.loc[mech, col])


def _despine(ax):
    ax.tick_params(length=3.5, colors="#4d4d4d")


# --------------------------------------------------------------------- Fig 1
def fig_pareto(dfj, ms):
    """Utility vs membership-inference AUC over each mechanism's full noise sweep."""
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    _despine(ax)
    ax.axvline(0.5, color=FAINT, lw=1.0, zorder=0)
    src = ms if ms is not None else dfj.reset_index()
    mcol = "mechanism"
    for k in ORDER:
        sub = src[src[mcol].apply(lambda s: family_of(s) == k)].copy()
        if sub.empty:
            continue
        xcol = "mia_auc_mean" if ms is not None else "mia_auc"
        ycol = "auroc_mean" if ms is not None else "auroc"
        sub = sub.sort_values(xcol)
        xs, ys = sub[xcol].to_numpy(), sub[ycol].to_numpy()
        accent = k == "RA3"
        if len(xs) > 1:
            ax.plot(xs, ys, color=COL[k], lw=2.6 if accent else 1.6,
                    alpha=0.95 if accent else 0.7, zorder=4 if accent else 2)
        ax.scatter(xs, ys, s=95 if accent else 46, marker=MARK[k], facecolor=COL[k],
                   edgecolor="white", linewidth=1.0, zorder=5 if accent else 3)
    # direct labels at each curve's leftmost (best) point
    for k in ORDER:
        sub = src[src[mcol].apply(lambda s: family_of(s) == k)]
        if sub.empty:
            continue
        xcol = "mia_auc_mean" if ms is not None else "mia_auc"
        ycol = "auroc_mean" if ms is not None else "auroc"
        i = sub[xcol].idxmin()
        x, y = sub.loc[i, xcol], sub.loc[i, ycol]
        dx = 0.012
        ax.annotate(NAME[k], (x, y), xytext=(x - dx, y), textcoords="data",
                    ha="right", va="center", fontsize=11.5,
                    color=COL["RA3"] if k == "RA3" else INK,
                    fontweight="bold" if k == "RA3" else "normal")
    ax.set_xlim(0.44, 1.04); ax.set_ylim(0.755, 0.83)
    ax.set_xlabel("Membership-inference AUC  (privacy risk — lower is safer)")
    ax.set_ylabel("Downstream utility (AUROC)")
    ax.text(0.5, 0.7565, "no leakage", color=SUB, fontsize=9.5, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "frontier.pdf"); fig.savefig(FIG / "frontier.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------- Fig 2
def fig_fidelity(dfj, ms):
    rows = ORDER[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _despine(ax)
    for i, k in enumerate(rows):
        p = val(dfj, ms, REP[k], "auroc"); s = val(dfj, ms, REP[k], "auroc_secondary")
        ax.plot([s, p], [i, i], color=FAINT, lw=2.4, zorder=1, solid_capstyle="round")
        ax.scatter([p], [i], s=95, color=COL[k], zorder=3, edgecolor="white", linewidth=1.0)
        ax.scatter([s], [i], s=95, facecolor="white", edgecolor=COL[k], linewidth=1.8, zorder=3)
        if p - s > 0.1:
            ax.annotate(f"−{p - s:.2f}", ((p + s) / 2, i + 0.22), ha="center",
                        va="bottom", fontsize=11, color=COL[k])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAME[k] for k in rows], fontsize=12)
    ax.set_ylim(-0.5, len(rows) - 0.35); ax.set_xlim(0.50, 0.84)
    ax.set_xticks([0.55, 0.65, 0.75])
    ax.set_xlabel("Downstream utility (AUROC)")
    ax.scatter([], [], s=90, color=SUB, label="tuned task")
    ax.scatter([], [], s=90, facecolor="white", edgecolor=SUB, linewidth=1.8, label="untuned task")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=10.5,
              handletextpad=0.4, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fidelity.pdf"); fig.savefig(FIG / "fidelity.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------- Fig 3
def fig_attacks(dfj, ms):
    keys = ORDER
    mia = [val(dfj, ms, REP[k], "mia_auc") for k in keys]
    attr = [val(dfj, ms, REP[k], "attr_bacc") for k in keys]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    _despine(ax)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0); ax.set_axisbelow(True)
    b1 = ax.bar(x - w/2, mia, w, color="#3f5566", zorder=3, label="Membership inference")
    b2 = ax.bar(x + w/2, attr, w, color="#d8a24a", zorder=3, label="Attribute inference")
    ax.axhline(0.5, color=FAINT, lw=1.0, zorder=2)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}",
                        (r.get_x() + r.get_width()/2, r.get_height() + 0.007),
                        ha="center", va="bottom", fontsize=9.5, color=SUB)
    ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k in keys], fontsize=11.5)
    ax.set_ylim(0.5, 1.06); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_ylabel("Attack success  (0.5 = no leakage)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
              handletextpad=0.5, columnspacing=1.6, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "kanon_failure.pdf"); fig.savefig(FIG / "kanon_failure.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------- Fig 4
def fig_allocation(single):
    b = single["ra3_budget"][sorted(single["ra3_budget"])[-1]]
    util = np.array(b["utility"]); lam = np.array(b["budget"]); feats = b["features"]
    qi = {"age", "sex_male", "married", "ins_medicare", "ins_medicaid",
          "adm_emergency", "adm_elective", "n_diagnoses"}
    is_qi = np.array([f in qi for f in feats]); med = np.median(util)
    groups = [("Quasi-identifiers", is_qi),
              ("Low-utility\nclinical", (~is_qi) & (util < med)),
              ("High-utility\nclinical", (~is_qi) & (util >= med))]
    means = [lam[g].mean() for _, g in groups]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    _despine(ax)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.bar(range(3), means, width=0.56, color="#6b8cc4", zorder=3)
    for i, m in enumerate(means):
        ax.annotate(f"{m:.2f}", (i, m + 0.02), ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _ in groups], fontsize=11.5)
    ax.set_ylim(0, 0.62); ax.set_ylabel("Mean protection  λ")
    fig.tight_layout()
    fig.savefig(FIG / "budget_map.pdf"); fig.savefig(FIG / "budget_map.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------- Fig 5
def fig_ablation(abl):
    if abl is None:
        return
    order = ["RA3-full", "RA3-uniform_lambda", "RA3-no_exposure", "RA3-no_surrogate"]
    nice = {"RA3-full": "Full RA³", "RA3-uniform_lambda": "− adaptive\nallocation",
            "RA3-no_exposure": "− exposure\nsignal", "RA3-no_surrogate": "− surrogate\n(noise only)"}
    abl = abl.set_index("variant").reindex(order)
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _despine(ax)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0); ax.set_axisbelow(True)
    util = abl["auroc_mean"].to_numpy(); util_e = abl["auroc_std"].to_numpy()
    mia = abl["mia_auc_mean"].to_numpy(); mia_e = abl["mia_auc_std"].to_numpy()
    b1 = ax.bar(x - w/2, util, w, yerr=util_e, capsize=3, color="#6f9c93", zorder=3,
                label="Utility (higher is better)", error_kw=dict(ecolor="#888", elinewidth=1))
    b2 = ax.bar(x + w/2, mia, w, yerr=mia_e, capsize=3, color="#c02b3a", zorder=3,
                label="Membership AUC (lower is safer)", error_kw=dict(ecolor="#888", elinewidth=1))
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}",
                        (r.get_x() + r.get_width()/2, r.get_height() + 0.012),
                        ha="center", va="bottom", fontsize=9.5, color=SUB)
    ax.axhline(0.5, color=FAINT, lw=1.0, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([nice[o] for o in order], fontsize=11)
    ax.set_ylim(0.45, 1.02); ax.set_ylabel("Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
              handletextpad=0.5, columnspacing=1.6, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.pdf"); fig.savefig(FIG / "ablation.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------- Fig: schematic
def fig_schematic():
    """How RA3 works: a 4-stage pipeline, drawn as clean boxes and arrows."""
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tcol="#1a1a1a", fs=11.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6,rounding_size=2.2",
                     linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                color=tcol, zorder=3, fontweight="bold" if bold else "normal")

    def arrow(x0, x1, y=20):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=16, lw=1.6, color="#8a8a8a", zorder=1))

    # 1 input
    box(1, 13, 17, 15, "Patient record\n\nquasi-identifiers\n+ clinical payload",
        "#eef1f4", "#9aa7b0")
    arrow(18.2, 24)
    # 2 scoring (two stacked)
    box(24.5, 22, 20, 8, "Exposure  $e_j$\n(re-identification risk)", "#fbeceb", "#c02b3a",
        tcol="#8a1f2a", fs=10.5)
    box(24.5, 11, 20, 8, "Utility  $u_j$\n(predictive value)", "#e9f0ea", "#4f8a63",
        tcol="#2f5f42", fs=10.5)
    arrow(44.9, 51)
    # 3 lambda
    box(51.5, 13, 18, 15, "Protection  $\\lambda_j$\n\n$\\lambda_j = $rank$(e_j / u_j)$\n\nhigh risk, low use:\nprotect most",
        "#f4f0fa", "#7a5aa8", tcol="#4a3170", fs=10.5)
    arrow(69.7, 76)
    # 4 interpolation -> output
    box(76.5, 13, 22.5, 15,
        "Released feature\n\n$(1{-}\\lambda)\\cdot$(real + noise)\n$+\\ \\lambda\\cdot$(class surrogate)\n\nkeep real  ·  re-synthesize",
        "#eef4fb", "#3f6fb0", tcol="#274a78", fs=10.5)
    # certificate strip
    ax.add_patch(FancyBboxPatch((24.5, 1.5), 74.5, 6.2,
                 boxstyle="round,pad=0.5,rounding_size=1.6", linewidth=1.2,
                 facecolor="#fbfbf0", edgecolor="#c9b458", zorder=2))
    ax.text(61.7, 4.6, "Certificate: measured membership, linkage and attribute risk; re-escalate $\\lambda$ until the target is met",
            ha="center", va="center", fontsize=10, color="#6b5a15", zorder=3, style="italic")
    ax.add_patch(FancyArrowPatch((87.7, 13), (87.7, 8), arrowstyle="-|>",
                 mutation_scale=13, lw=1.4, color="#c9b458", zorder=1))
    fig.tight_layout()
    fig.savefig(FIG / "schematic.pdf"); fig.savefig(FIG / "schematic.png", dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------- Fig: heatmap
def fig_heatmap():
    """At-a-glance comparison: mechanisms x criteria, coloured by goodness."""
    rows = [
        ("Raw",              0.815, 0.800, 1.000, 0.794, 0.987, 1),
        ("k-anonymity",      0.802, 0.798, 0.968, 0.003, 0.986, 1),
        ("Uniform DP",       0.803, 0.769, 0.698, 0.005, 0.754, 1),
        ("DP-Synthetic",     0.769, 0.551, 0.500, 0.000, 0.500, 0),
        ("RA³ (balanced)",   0.797, 0.780, 0.573, 0.002, 0.803, 1),
        ("RA³ (high-privacy)",0.782, 0.749, 0.515, 0.001, 0.690, 1),
    ]
    cols = ["Utility\n(primary)", "Utility\n(untuned)", "Membership\nattack AUC",
            "Linkage\nrate", "Attribute\nattack acc.", "Real\nrecords"]
    def good(kind, v):
        if kind == "util":   return np.clip((v - 0.5) / (0.83 - 0.5), 0, 1)
        if kind == "mia":    return np.clip(1 - 2 * (v - 0.5), 0, 1)
        if kind == "reid":   return np.clip(1 - v, 0, 1)
        if kind == "attr":   return np.clip(1 - 2 * (v - 0.5), 0, 1)
        if kind == "real":   return float(v)
    G, T = [], []
    for name, p, s, mia, reid, attr, real in rows:
        G.append([good("util", p), good("util", s), good("mia", mia),
                  good("reid", reid), good("attr", attr), good("real", real)])
        T.append([f"{p:.2f}", f"{s:.2f}", f"{mia:.2f}", f"{reid:.3f}", f"{attr:.2f}",
                  "Yes" if real else "No"])
    G = np.array(G)
    cmap = LinearSegmentedColormap.from_list("rag", ["#c94b4b", "#e9d16a", "#4f9d69"])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.imshow(G, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=10.5)
    ax.set_yticks(range(len(rows)))
    ylabels = [r[0] for r in rows]
    ax.set_yticklabels(ylabels, fontsize=11.5)
    for i, lab in enumerate(ax.get_yticklabels()):
        if ylabels[i].startswith("RA³"):
            lab.set_fontweight("bold"); lab.set_color("#c02b3a")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, T[i][j], ha="center", va="center", fontsize=10.5,
                    color="#1a1a1a")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.5, -1.05, "Cell colour: green = better, red = worse   ·   utility higher is better, attacks lower is better",
            fontsize=9.5, color=SUB, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "heatmap.pdf"); fig.savefig(FIG / "heatmap.png", dpi=DPI)
    plt.close(fig)


def main():
    single, dfj, ms, abl = load()
    fig_schematic()
    fig_heatmap()
    fig_pareto(dfj, ms)
    fig_fidelity(dfj, ms)
    fig_attacks(dfj, ms)
    fig_allocation(single)
    fig_ablation(abl)
    print(f"Figures written to {FIG} (font: {FONT}, dpi: {DPI}, ablation: {abl is not None})")


if __name__ == "__main__":
    main()
