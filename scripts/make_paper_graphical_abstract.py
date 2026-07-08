#!/usr/bin/env python3
"""Publication-grade graphical abstract for the ICU discharge-feasibility paper.

Narrative, left to right:
  (1) cohort funnel  ->  (2) the representation contrast, drawn as a token metaphor
  (single flat cell per signal vs a multi-channel token)  ->  (3) a real results bar
  chart grounded in the verified AUROC numbers.  A takeaway banner closes it.
All numbers are the verified values from results/full_cohort_enriched/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parent.parent

# ---- palette ----
NAVY = "#12314F"; BLUE = "#2C6FBB"; TEAL = "#12868B"; AMBER = "#E08A1E"
GREEN = "#2E7D32"; GREY = "#AAB2BA"; INK = "#243746"; SUB = "#5C6B78"; RED = "#B0432F"
PAPER = "#FFFFFF"

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(12.8, 6.7), dpi=200)
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.8); ax.set_ylim(0, 6.7); ax.axis("off")

TOP = 5.02          # common top edge for the three panels' content
BODY_BOT = 1.98     # everything above the banner

def rbox(x, y, w, h, fc, ec="none", lw=0, rad=0.10, alpha=1.0, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                 fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, mutation_aspect=1))

def arrow(x1, y1, x2, y2, color=GREY, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                 lw=lw, color=color, zorder=4))

def chip(x, y, n, label):
    ax.add_patch(Circle((x, y), 0.155, color=NAVY, zorder=6))
    ax.text(x, y, n, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=7)
    ax.text(x + 0.29, y, label, ha="left", va="center", color=NAVY, fontsize=10.5, fontweight="bold")

# ================= header =================
ax.text(6.4, 6.35, "Symptom Tokens for ICU Discharge-Feasibility Prediction",
        ha="center", va="center", fontsize=18, fontweight="bold", color=NAVY)
ax.text(6.4, 5.96, "The representation of the signals — not the choice of model — is the bottleneck",
        ha="center", va="center", fontsize=12.5, style="italic", color=SUB)
ax.plot([0.45, 12.35], [5.66, 5.66], color="#E4E9EE", lw=1.3)

# panel chips on one baseline
chip(0.55, 5.28, "1", "Cohort")
chip(3.25, 5.28, "2", "Same signals, two ways")
chip(6.75, 5.28, "3", "Held-out AUROC")

# ================= (1) cohort funnel =================
rbox(0.45, 4.00, 2.35, 0.92, NAVY)
ax.text(1.625, 4.60, "MIMIC-IV v3.1", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
ax.text(1.625, 4.28, "94,458 ICU stays", ha="center", va="center", color="#C6D5E5", fontsize=9.8)
arrow(1.625, 3.96, 1.625, 3.62, GREY, lw=2.0)
rbox(0.45, 2.70, 2.35, 0.92, BLUE)
ax.text(1.625, 3.30, "Eligibility filters", ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
ax.text(1.625, 2.99, "32,399 first adult stays", ha="center", va="center", color="#DCEAF7", fontsize=9.4)
ax.text(1.625, 2.40, "first 24 h  ·  62.6% feasible", ha="center", va="center", color=SUB, fontsize=9, style="italic")

# ================= (2) representation contrast (token metaphor) =================
CX0 = 3.15                       # card left
CW = 3.05                        # card width
def token_row(x0, y0, kind, n=6):
    cols = [BLUE, TEAL, AMBER, "#C0553B", GREEN, "#7B5EA7"]
    for i in range(n):
        cx = x0 + i * 0.40
        if kind == "flat":
            ax.add_patch(Rectangle((cx, y0), 0.29, 0.34, fc=GREY, ec="white", lw=1.3, zorder=3))
        else:
            for c in range(6):
                ax.add_patch(Rectangle((cx, y0 + c * 0.075), 0.29, 0.075,
                             fc=cols[c], ec="white", lw=0.5, alpha=0.95, zorder=3))

# median-only card (top)
rbox(CX0, 4.02, CW, 0.98, "#F2F4F6", ec="#D9DEE3", lw=1.2)
ax.text(CX0 + CW / 2, 4.80, "Median-only", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
token_row(CX0 + 0.35, 4.28, "flat")
ax.text(CX0 + CW / 2, 4.14, "1 value per signal", ha="center", va="center", fontsize=8.4, color=SUB, style="italic")

# rich card (bottom)
rbox(CX0, 2.56, CW, 1.14, "#FCF3E6", ec=AMBER, lw=1.4)
ax.text(CX0 + CW / 2, 3.52, "Rich symptom token", ha="center", va="center", fontsize=11, fontweight="bold", color="#9A5B12")
token_row(CX0 + 0.35, 2.86, "rich")
ax.text(CX0 + CW / 2, 2.70, "level · trend · spread · extremes · density",
        ha="center", va="center", fontsize=7.9, color="#8A6A3A", style="italic")

# enrich arrow between the two cards
arrow(CX0 + CW / 2, 3.98, CX0 + CW / 2, 3.74, AMBER, lw=2.4)
ax.text(CX0 + CW / 2 + 0.33, 3.86, "enrich", ha="left", va="center", fontsize=8.8, color="#9A5B12", fontweight="bold")

# ================= (3) results bar chart =================
axb = fig.add_axes([0.585, 0.335, 0.375, 0.375]); axb.set_facecolor("none")
labels = ["Federated\n(5 nodes)", "Few-shot 10/class\n(was 0.607 raw)",
          "Symptom-Token\nTransformer", "Enriched tabular\nensemble"]
vals = [0.811, 0.814, 0.818, 0.841]
cols = [GREEN, GREEN, TEAL, BLUE]
yy = list(range(len(vals)))
axb.barh(yy, [v - 0.70 for v in vals], left=0.70, color=cols, height=0.56, zorder=3)
# grey segment on the tabular row shows the median-only starting point
axb.barh([3], [0.757 - 0.70], left=0.70, color=GREY, height=0.56, alpha=0.6, zorder=4)
# reference line capped below its label so the two never overlap
axb.plot([0.757, 0.757], [-0.7, 3.86], color=RED, ls=(0, (4, 2)), lw=1.4, zorder=5)
for i, v in zip(yy, vals):
    axb.text(v + 0.0025, i, f"{v:.3f}", va="center", ha="left", fontsize=10.5, fontweight="bold", color=INK)
# gain bracket + baseline label, in the headroom above the top bar
axb.annotate("", xy=(0.841, 3.50), xytext=(0.757, 3.50),
             arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
axb.text(0.799, 3.60, "+0.085", va="bottom", ha="center", fontsize=9, color=BLUE, fontweight="bold")
axb.text(0.757, 4.14, "median-only 0.757", color=RED, fontsize=8.6, ha="center", va="bottom",
         fontweight="bold", zorder=6)
axb.set_yticks(yy); axb.set_yticklabels(labels, fontsize=9.0, color=INK)
axb.set_ylim(-0.7, 4.6)
axb.set_xlim(0.70, 0.875); axb.set_xticks([0.70, 0.80, 0.85])
axb.set_xlabel("AUROC (held-out test)", fontsize=9.2, color=SUB, labelpad=3)
axb.tick_params(axis="x", labelsize=8.3, colors=SUB, length=3)
axb.tick_params(axis="y", length=0, pad=6)      # no stray y tick marks; breathing room
for s in ["top", "right", "left"]:
    axb.spines[s].set_visible(False)
axb.spines["bottom"].set_color("#C9D0D6")
axb.grid(axis="x", color="#EDF0F3", lw=0.8, zorder=0)

# ================= takeaway banner =================
rbox(0.45, 0.28, 11.9, 1.5, NAVY, rad=0.11)
ax.text(6.4, 1.36, "With a rich symptom-token representation, all three title methods succeed",
        ha="center", va="center", fontsize=12.4, fontweight="bold", color="white")
ax.text(6.4, 0.93, "learned tokens rival strong tabular baselines   ·   few-shot reaches full-data accuracy from ~10 labels/class",
        ha="center", va="center", fontsize=10.4, color="#D7E3F0")
ax.text(6.4, 0.58, "federated training preserves performance without sharing any patient records",
        ha="center", va="center", fontsize=10.4, color="#D7E3F0")

out = ROOT / "results/full_cohort/figures/graphical_abstract_final.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=PAPER, pad_inches=0.12)
print("saved", out)
