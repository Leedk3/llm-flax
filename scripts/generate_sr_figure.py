"""
Generate SR vs. difficulty figure for the LLM-Flax paper.

Shows that Gemma3-12B (LLM-Flax) matches or outperforms Manual on every benchmark.

Output: paper/figures/sr_scale.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── Data from Table 1 (Gemma3-12B as main LLM-Flax model) ────────────────────
DATA = {
    "10×10": {
        "difficulties": ["Easy", "Medium", "Hard"],
        "manual":  [1.000, 0.960, 0.960],  # Flax (Manual)
        "llmflax": [1.000, 1.000, 0.960],
    },
    "12×12": {
        "difficulties": ["Medium", "Hard", "Expert"],
        "manual":  [0.960, 0.880, 0.000],
        "llmflax": [0.980, 0.920, 0.733],
    },
    "15×15": {
        "difficulties": ["Medium", "Hard"],
        "manual":  [0.967, 0.900],
        "llmflax": [0.967, 1.000],
    },
}

# ── Colors ────────────────────────────────────────────────────────────────────
C_MANUAL  = "#4472C4"   # blue
C_LLMFLAX = "#ED7D31"   # orange
C_SHADE   = "#FFF3E0"   # light orange for crossover annotation

BAR_W = 0.32
GROUP_GAP = 0.12

def make_axis(ax, grid_label, difficulties, manual_sr, llm_sr):
    n = len(difficulties)
    x = np.arange(n)

    bars_m = ax.bar(x - BAR_W/2 - GROUP_GAP/2, manual_sr,
                    width=BAR_W, color=C_MANUAL,  label="Flax (Manual)",
                    edgecolor="white", linewidth=0.5, zorder=3)
    bars_l = ax.bar(x + BAR_W/2 + GROUP_GAP/2, llm_sr,
                    width=BAR_W, color=C_LLMFLAX, label="LLM-Flax",
                    edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on top of each bar
    for bar in list(bars_m) + list(bars_l):
        h = bar.get_height()
        if h == 0:
            # Show "ALL\nFAIL" marker for zero-SR bars
            ax.text(bar.get_x() + bar.get_width() / 2, 0.04,
                    "ALL\nFAIL",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#CC0000", fontweight="bold",
                    linespacing=1.1)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#333333", fontweight="normal")

    # Highlight crossover: shade pairs where LLM-Flax > Manual
    for i, (m, l) in enumerate(zip(manual_sr, llm_sr)):
        if l > m:
            ax.axvspan(i - 0.5, i + 0.5, color=C_SHADE, alpha=0.55, zorder=1)
            # Add crossover annotation arrow
            ax.annotate("",
                xy=(i + BAR_W/2 + GROUP_GAP/2, l + 0.04),
                xytext=(i - BAR_W/2 - GROUP_GAP/2, m + 0.04),
                arrowprops=dict(arrowstyle="<->", color="#CC4400",
                                lw=1.2, mutation_scale=8),
                zorder=5)

    # Grid and spine styling
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="y", which="major", color="#CCCCCC", linewidth=0.6,
            linestyle="--", zorder=0)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.4,
            linestyle=":", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")

    # Panel title
    ax.set_title(grid_label, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Difficulty", fontsize=9, labelpad=4)


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0),
                         gridspec_kw={"width_ratios": [3, 3, 2]})
plt.subplots_adjust(wspace=0.32, left=0.07, right=0.99,
                    top=0.88, bottom=0.24)

for ax, (grid_label, d) in zip(axes, DATA.items()):
    make_axis(ax, grid_label,
              d["difficulties"], d["manual"], d["llmflax"])

# Y-axis label only on leftmost panel
axes[0].set_ylabel("Success Rate (SR)", fontsize=9, labelpad=4)

# Shared legend
legend_handles = [
    mpatches.Patch(facecolor=C_MANUAL,  edgecolor="white", label="Flax (Manual)"),
    mpatches.Patch(facecolor=C_LLMFLAX, edgecolor="white", label="LLM-Flax (Gemma3-12B)"),
    mpatches.Patch(facecolor=C_SHADE,   edgecolor="#CC4400",
                   linewidth=0.8, label="LLM-Flax > Flax"),
]
fig.legend(handles=legend_handles,
           loc="lower center", ncol=3,
           fontsize=9, frameon=False,
           bbox_to_anchor=(0.53, 0.01))

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("paper/figures", exist_ok=True)
for ext in ("pdf", "png"):
    path = f"paper/figures/sr_scale.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
