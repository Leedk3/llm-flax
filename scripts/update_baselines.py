"""
After run_baselines.py finishes, run this script to:
  1. Fill in the --- placeholders in paper/main.tex (Appendix C table)
  2. Regenerate paper/figures/sr_scale.{pdf,png} with Pure FD and PLOI included

Usage (from /home/leedk/flax/):
    conda run -n flax python scripts/update_baselines.py
    conda run -n flax python scripts/update_baselines.py --check   # just print loaded results
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PAPER_TEX  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "paper", "main.tex")
FIG_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "paper", "figures")

# Benchmarks in the same order as TABLE rows
BENCHMARKS = [
    dict(size=10, difficulty="easy",   n=50),
    dict(size=10, difficulty="medium", n=50),
    dict(size=10, difficulty="hard",   n=50),
    dict(size=12, difficulty="medium", n=50),
    dict(size=12, difficulty="hard",   n=50),
    dict(size=12, difficulty="expert", n=30),
    dict(size=15, difficulty="medium", n=30),
    dict(size=15, difficulty="hard",   n=30),
]


# ── Load results ──────────────────────────────────────────────────────────────

def load_results():
    """Return dict: {(config, size, difficulty): {sr, avg_time, avg_plan_length}}"""
    results = {}
    for bm in BENCHMARKS:
        for cfg in ("pure", "ploi"):
            fname = (f"baseline_{bm['size']}x{bm['size']}_{bm['difficulty']}"
                     f"_n{bm['n']}_{cfg}.json")
            path = os.path.join(RESULT_DIR, fname)
            if not os.path.exists(path):
                print(f"  [MISSING] {fname}")
                continue
            with open(path) as f:
                d = json.load(f)
            key = (cfg, bm["size"], bm["difficulty"])
            results[key] = d
            print(f"  [OK] {fname}  SR={d['success_rate']:.3f}")
    return results


# ── Update LaTeX table ────────────────────────────────────────────────────────

def _fmt(val, is_sr=False):
    if val is None:
        return "---"
    if is_sr:
        return f"{val:.3f}"
    return f"{val:.3f}"


def update_latex(results):
    """Replace --- placeholders in the tab:baselines rows with real values."""
    with open(PAPER_TEX) as f:
        tex = f.read()

    # We'll rebuild the table section between \label{tab:baselines} and \end{table}
    # by regenerating the Pure FD and PLOI rows for each benchmark.
    # Approach: find each "Pure FD" row block and replace its cells.

    BM_LABELS = {
        (10, "easy"):   r"10$\times$10 Easy",
        (10, "medium"): r"10$\times$10 Medium",
        (10, "hard"):   r"10$\times$10 Hard",
        (12, "medium"): r"12$\times$12 Medium",
        (12, "hard"):   r"12$\times$12 Hard",
        (12, "expert"): r"12$\times$12 Expert",
        (15, "medium"): r"15$\times$15 Medium",
        (15, "hard"):   r"15$\times$15 Hard",
    }

    for bm in BENCHMARKS:
        s, d = bm["size"], bm["difficulty"]
        for cfg, cfg_label in [("pure", "Pure FD"), ("ploi", "PLOI")]:
            key = (cfg, s, d)
            if key not in results:
                continue
            r = results[key]
            sr  = r["success_rate"]
            t   = r["avg_time"] if sr > 0 else None
            l   = r["avg_plan_length"] if sr > 0 else None

            sr_str = _fmt(sr, is_sr=True)
            t_str  = _fmt(t)
            l_str  = _fmt(l)

            # Bold best SR in the group? (skip for now — just fill values)
            new_row = f"      & {cfg_label:<13} & {sr_str} & {t_str} & {l_str} \\\\"

            # Match the existing row for this config in this benchmark block.
            # Pattern: "      & Pure FD      & --- & --- & --- \\"
            pattern = (rf"(      & {re.escape(cfg_label)}\s+& )---( & )---( & )---( \\\\)")
            replacement = rf"\g<1>{sr_str}\g<2>{t_str}\g<3>{l_str}\g<4>"
            tex_new = re.sub(pattern, replacement, tex)
            if tex_new == tex:
                print(f"  [WARN] Could not find placeholder row for {cfg_label} {s}x{s} {d}")
            else:
                tex = tex_new
                print(f"  Updated {cfg_label} {s}x{s} {d}: SR={sr_str} T={t_str} L={l_str}")

    with open(PAPER_TEX, "w") as f:
        f.write(tex)
    print(f"Saved: {PAPER_TEX}")


# ── Regenerate sr_scale figure with 4 configs ─────────────────────────────────

# Manual and LLM-Flax data from Table 1
MANUAL = {
    (10,"easy"):1.000,(10,"medium"):0.960,(10,"hard"):0.960,
    (12,"medium"):0.960,(12,"hard"):0.880,(12,"expert"):0.000,
    (15,"medium"):0.967,(15,"hard"):0.900,
}
LLMFLAX = {
    (10,"easy"):0.940,(10,"medium"):0.900,(10,"hard"):0.900,
    (12,"medium"):0.920,(12,"hard"):0.800,(12,"expert"):1.000,
    (15,"medium"):0.833,(15,"hard"):1.000,
}

C_PURE    = "#888888"   # grey
C_PLOI    = "#9B59B6"   # purple
C_MANUAL  = "#4472C4"   # blue
C_LLMFLAX = "#ED7D31"   # orange
C_SHADE   = "#FFF3E0"

BAR_W = 0.18
GAPS  = [-0.27, -0.09, 0.09, 0.27]   # offsets for 4 bars

GRID_DATA = {
    "10×10": dict(difficulties=["Easy","Medium","Hard"],
                  keys=[(10,"easy"),(10,"medium"),(10,"hard")]),
    "12×12": dict(difficulties=["Medium","Hard","Expert"],
                  keys=[(12,"medium"),(12,"hard"),(12,"expert")]),
    "15×15": dict(difficulties=["Medium","Hard"],
                  keys=[(15,"medium"),(15,"hard")]),
}


def make_axis_4(ax, grid_label, difficulties, keys,
                pure_sr, ploi_sr, ax_idx):
    n = len(difficulties)
    x = np.arange(n)
    configs = [
        (pure_sr,   C_PURE,    "Pure FD"),
        (ploi_sr,   C_PLOI,    "PLOI"),
        ([MANUAL[k]  for k in keys], C_MANUAL,  "Manual"),
        ([LLMFLAX[k] for k in keys], C_LLMFLAX, "LLM-Flax"),
    ]
    bars_all = []
    for i, (sr_list, color, label) in enumerate(configs):
        offset = GAPS[i]
        if sr_list is None:
            bars_all.append(None)
            continue
        bars = ax.bar(x + offset, sr_list,
                      width=BAR_W, color=color, label=label,
                      edgecolor="white", linewidth=0.4, zorder=3)
        bars_all.append(bars)
        for bar in bars:
            h = bar.get_height()
            if h == 0:
                ax.text(bar.get_x() + bar.get_width()/2, 0.03,
                        "FAIL", ha="center", va="bottom",
                        fontsize=5.5, color="#CC0000", fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=5.5, color="#333333")

    # Shade crossover positions (LLM-Flax > Manual)
    for j, k in enumerate(keys):
        if LLMFLAX[k] > MANUAL[k]:
            ax.axvspan(j - 0.5, j + 0.5, color=C_SHADE, alpha=0.5, zorder=1)

    ax.set_ylim(0, 1.20)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", which="major", color="#CCCCCC", linewidth=0.6,
            linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.set_title(grid_label, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Difficulty", fontsize=8.5, labelpad=4)
    if ax_idx == 0:
        ax.set_ylabel("Success Rate (SR)", fontsize=8.5, labelpad=4)


def regenerate_figure(results):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2),
                             gridspec_kw={"width_ratios": [3, 3, 2]})
    plt.subplots_adjust(wspace=0.30, left=0.07, right=0.99,
                        top=0.87, bottom=0.24)

    for ax_idx, (grid_label, gd) in enumerate(GRID_DATA.items()):
        keys = gd["keys"]
        pure_sr = [results.get(("pure", k[0], k[1]), {}).get("success_rate")
                   for k in keys]
        ploi_sr = [results.get(("ploi", k[0], k[1]), {}).get("success_rate")
                   for k in keys]
        # Replace None with 0.0 if missing (will show as FAIL)
        pure_sr = [v if v is not None else 0.0 for v in pure_sr]
        ploi_sr = [v if v is not None else 0.0 for v in ploi_sr]

        make_axis_4(axes[ax_idx], grid_label,
                    gd["difficulties"], keys,
                    pure_sr, ploi_sr, ax_idx)

    legend_handles = [
        mpatches.Patch(fc=C_PURE,    ec="white", label="Pure FD"),
        mpatches.Patch(fc=C_PLOI,    ec="white", label="PLOI"),
        mpatches.Patch(fc=C_MANUAL,  ec="white", label="Manual (Flax)"),
        mpatches.Patch(fc=C_LLMFLAX, ec="white", label="LLM-Flax"),
        mpatches.Patch(fc=C_SHADE,   ec="#CC4400", lw=0.8,
                       label="LLM-Flax > Manual"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=5,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.53, 0.01))

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"sr_scale.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",     action="store_true",
                        help="Print loaded results only, no file changes")
    parser.add_argument("--fig_only",  action="store_true",
                        help="Only regenerate the figure, skip LaTeX update")
    parser.add_argument("--tex_only",  action="store_true",
                        help="Only update LaTeX, skip figure")
    args = parser.parse_args()

    print("Loading baseline results...")
    results = load_results()

    if args.check:
        return

    if not args.fig_only:
        print("\nUpdating LaTeX table...")
        update_latex(results)

    if not args.tex_only:
        print("\nRegenerating sr_scale figure (4 configs)...")
        regenerate_figure(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
