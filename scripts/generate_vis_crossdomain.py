"""
Combine the SokomindPlus and DifficultLogistics qualitative visualizations
(vis_sokomindplus_{a,b,c}.png, vis_difficultlogistics_{a,b,c}.png) into one
figure, in the same row-per-benchmark / 3-column style as
paper/figures/vis/vis_all.pdf (the MazeNamo qualitative-trace figure).

Run generate_sokomindplus_vis.py and generate_difficultlogistics_vis.py first
to produce the individual panels.

Usage: python scripts/generate_vis_crossdomain.py
Output: paper/figures/vis/vis_crossdomain.{pdf,png}
"""

import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures", "vis")

ROWS = [
    ("SokomindPlus",       "vis_sokomindplus"),
    ("DifficultLogistics", "vis_difficultlogistics"),
]
PANEL_SUFFIXES = ["_a", "_b", "_c"]
PANEL_HEADERS = ["(a) Full Problem", "(b) After Pruning", "(c) Final Plan"]


def main():
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 3.3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for row, (label, stem) in enumerate(ROWS):
        for col, suf in enumerate(PANEL_SUFFIXES):
            ax = axes[row, col]
            path = os.path.join(VIS_DIR, f"{stem}{suf}.png")
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis("off")
        axes[row, 0].text(-0.06, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight="bold", rotation=90,
                          va="center", ha="center")

    plt.subplots_adjust(wspace=0.03, hspace=0.02, left=0.04, right=0.99,
                        top=0.99, bottom=0.02)

    os.makedirs(VIS_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(VIS_DIR, f"vis_crossdomain.{ext}")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
