"""
Combine all 8 per-benchmark visualizations into one landscape figure.

Layout: two "3×4" blocks side by side
  Left block  (4 rows): 10×10 Easy / Medium / Hard  + 12×12 Medium
  Right block (4 rows): 12×12 Hard / Expert         + 15×15 Medium / Hard
  Each row has 3 columns: (a) Full Problem | (b) LLM Scoring | (c) After Pruning & Final Plan

Individual panel PNGs (*_a.png, *_b.png, *_c.png) must already exist in VIS_DIR.

Output: paper/figures/vis/vis_all.{pdf,png}

Usage (from /home/leedk/flax/):
    conda run -n flax python scripts/generate_vis_all.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures", "vis")

# Two blocks, each 4 rows.  Each entry: (block, row, label, stem)
BLOCKS = [
    # Left block
    [
        (r"$10{\times}10$ Easy",   "vis_10x10_easy"),
        (r"$10{\times}10$ Medium", "vis_10x10_medium"),
        (r"$10{\times}10$ Hard",   "vis_10x10_hard"),
        (r"$12{\times}12$ Medium", "vis_12x12_medium"),
    ],
    # Right block
    [
        (r"$12{\times}12$ Hard",   "vis_12x12_hard"),
        (r"$12{\times}12$ Expert", "vis_12x12_expert"),
        (r"$15{\times}15$ Medium", "vis_15x15_medium"),
        (r"$15{\times}15$ Hard",   "vis_15x15_hard"),
    ],
]

PANEL_SUFFIXES = ["_a", "_b", "_c"]
PANEL_HEADERS  = ["(a) Full Problem",
                  "(b) LLM Object Scoring",
                  "(c) After Pruning & Final Plan"]


def load_panels():
    """Load all individual panel images.  Return dict stem+suffix → ndarray."""
    imgs = {}
    for block in BLOCKS:
        for label, stem in block:
            for suf in PANEL_SUFFIXES:
                fname = f"{stem}{suf}.png"
                path  = os.path.join(VIS_DIR, fname)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing: {path}")
                imgs[(stem, suf)] = mpimg.imread(path)
    return imgs


def main():
    imgs = load_panels()

    # Measure panel dimensions from first image
    sample    = imgs[(BLOCKS[0][0][1], "_a")]
    img_h, img_w = sample.shape[:2]
    panel_aspect = img_h / img_w   # height / width of one panel

    # Layout parameters (inches)
    panel_w    = 1.95        # width of one panel cell
    panel_h    = panel_w * panel_aspect
    label_w    = 1.05        # row-label column width
    hdr_h      = 0.28        # column-header row height
    row_gap    = 0.18        # vertical gap between rows
    col_gap    = 0.08        # horizontal gap between panel columns
    block_gap  = 0.55        # gap between left and right 3×4 blocks

    n_rows   = 4
    n_panels = 3  # (a)(b)(c)

    block_w = label_w + n_panels * panel_w + (n_panels - 1) * col_gap
    fig_w   = 2 * block_w + block_gap
    fig_h   = hdr_h + n_rows * panel_h + (n_rows - 1) * row_gap + 0.10  # small top/bottom margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    def to_fig(x, y, w, h):
        """Convert absolute-inch coordinates to figure-fraction [0,1]."""
        return [x / fig_w, (fig_h - y - h) / fig_h, w / fig_w, h / fig_h]

    top_margin = 0.05   # inches from top of figure to first content

    for bi, block in enumerate(BLOCKS):
        block_x0 = bi * (block_w + block_gap)  # left edge of this block (inches from fig left)

        # Column headers for this block
        for pi, hdr in enumerate(PANEL_HEADERS):
            col_x = block_x0 + label_w + pi * (panel_w + col_gap)
            ax_hdr = fig.add_axes(to_fig(col_x, top_margin, panel_w, hdr_h))
            ax_hdr.axis("off")
            ax_hdr.text(0.5, 0.5, hdr,
                        ha="center", va="center",
                        fontsize=7.5, fontstyle="italic", color="#333",
                        transform=ax_hdr.transAxes)

        for ri, (label, stem) in enumerate(block):
            row_y = top_margin + hdr_h + ri * (panel_h + row_gap)

            # Row label
            ax_lbl = fig.add_axes(to_fig(block_x0, row_y, label_w - 0.06, panel_h))
            ax_lbl.axis("off")
            ax_lbl.text(1.0, 0.5, label,
                        ha="right", va="center",
                        fontsize=8.5, fontweight="bold",
                        transform=ax_lbl.transAxes)

            # Three panels
            for pi, suf in enumerate(PANEL_SUFFIXES):
                px = block_x0 + label_w + pi * (panel_w + col_gap)
                ax = fig.add_axes(to_fig(px, row_y, panel_w, panel_h))
                ax.imshow(imgs[(stem, suf)], aspect="auto", interpolation="lanczos")
                ax.axis("off")

        # Thin vertical divider between blocks
        if bi == 0:
            div_x = (block_w + block_gap / 2) / fig_w
            fig.add_artist(plt.Line2D(
                [div_x, div_x], [0.01, 0.99],
                transform=fig.transFigure,
                color="#BBBBBB", linewidth=0.8))

    out = os.path.join(VIS_DIR, "vis_all")
    for ext in ("pdf", "png"):
        path = f"{out}.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
