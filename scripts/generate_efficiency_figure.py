"""
Generate two supplementary figures for the paper:

Fig 1 (efficiency_scatter): SR vs. Avg Planning Time scatter for all 4 configs × 8 benchmarks.
  - Shows that Pure FD can match SR but at much higher time cost.
  - Shows that LLM-Flax trades slight SR for massive speed gains on hard problems.

Fig 2 (sr_heatmap): SR heatmap — configs (rows) × benchmarks (cols), colored by SR value.
  - Compact overview of the full result matrix.

Output: paper/figures/efficiency_scatter.{pdf,png}
         paper/figures/sr_heatmap.{pdf,png}
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Full result data (SR, avg_time) for each config × benchmark ───────────────
# (sr, avg_time)  avg_time = timeout if sr==0
BENCHMARKS = ["10E","10M","10H","12M","12H","12X","15M","15H"]
FULL_LABELS = [
    "10×10\nEasy","10×10\nMedium","10×10\nHard",
    "12×12\nMedium","12×12\nHard","12×12\nExpert",
    "15×15\nMedium","15×15\nHard",
]
TIMEOUTS = [10, 10, 10, 30, 30, 30, 40, 40]

DATA = {
    "Pure FD": dict(
        sr  =[1.000, 1.000, 1.000, 1.000, 1.000, 0.333, 1.000, 0.333],
        time=[ 0.67,  1.40,  3.03,  6.19, 17.59, 26.19, 27.33, 34.37],
    ),
    "PLOI": dict(
        sr  =[0.900, 0.920, 0.820, 0.900, 0.580, 0.433, 0.767, 0.700],
        time=[ 0.76,  0.84,  1.31,  1.32,  1.27,  1.09,  5.09,  3.75],
    ),
    "Flax (Manual)": dict(
        sr  =[1.000, 0.960, 0.960, 0.960, 0.880, 0.000, 0.967, 0.900],
        time=[ 0.93,  2.00,  2.20,  1.91,  3.97, 30.00,  8.74, 11.90],
    ),
    # LLM model comparison (Stage 1 rules only, same GNN scorer)
    "Gemma3-12B": dict(
        sr  =[1.000, 1.000, 0.960, 0.980, 0.920, 0.733, 0.967, 1.000],
        time=[ 0.90,  0.95,  1.60,  2.15,  4.53,  5.24,  9.01, 10.98],
    ),
    "Qwen2.5-14B": dict(
        sr  =[0.940, 0.900, 0.900, 0.920, 0.800, 1.000, 0.833, 1.000],
        time=[ 0.85,  1.93,  2.02,  1.77,  3.22,  1.57,  8.00,  1.93],
    ),
    "Llama3.1-8B": dict(
        sr  =[None, None, None, None, None, None, None, None],
        time=[None, None, None, None, None, None, None, None],
    ),
    "Mistral-7B": dict(
        sr  =[None, None, None, None, None, None, None, None],
        time=[None, None, None, None, None, None, None, None],
    ),
}

# ── Auto-load llama/mistral results from JSON if available ────────────────────
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results")
_MODEL_SUFFIX = {
    "Llama3.1-8B": ("llm_rules_llama3.1-8b", "llama3.1-8b"),
    "Mistral-7B":  ("llm_rules_mistral-7b",  "mistral-7b"),
}
_BM_FILES = [
    ("10E", "ablation_10x10_easy_n50"),
    ("10M", "ablation_10x10_medium_n50"),
    ("10H", "ablation_10x10_hard_n50"),
    ("12M", "ablation_12x12_medium_n50"),
    ("12H", "ablation_12x12_hard_n50"),
    ("12X", "ablation_12x12_expert_n30"),
    ("15M", "ablation_15x15_medium_n30"),
    ("15H", "ablation_15x15_hard_n30"),
]

for display_name, (config_key, suffix) in _MODEL_SUFFIX.items():
    sr_list, time_list = [], []
    for bm_key, fname in _BM_FILES:
        path = os.path.join(_RESULTS_DIR, f"{fname}_{suffix}.json")
        val_sr, val_t = None, None
        if os.path.exists(path):
            try:
                d = json.load(open(path))
                for r in d.get("results", []):
                    if r.get("config") == config_key:
                        val_sr = r.get("success_rate")
                        val_t  = r.get("avg_time")
                        break
            except Exception:
                pass
        sr_list.append(val_sr)
        time_list.append(val_t)
    DATA[display_name]["sr"]   = sr_list
    DATA[display_name]["time"] = time_list

COLORS = {
    "Pure FD":    "#888888",
    "PLOI":       "#9B59B6",
    "Flax (Manual)": "#4472C4",
    "Gemma3-12B": "#ED7D31",
    "Qwen2.5-14B":"#FFC000",
    "Llama3.1-8B":"#70AD47",
    "Mistral-7B": "#FF6B6B",
}
MARKERS = {
    "Pure FD":    "s",
    "PLOI":       "^",
    "Flax (Manual)": "D",
    "Gemma3-12B": "o",
    "Qwen2.5-14B":"o",
    "Llama3.1-8B":"o",
    "Mistral-7B": "o",
}
# Marker size encodes grid size group
BM_SIZE = [50, 50, 50, 80, 80, 80, 110, 110]  # 10×10, 12×12, 15×15


# ── Figure 1: Bubble chart ────────────────────────────────────────────────────
# X = benchmark (8 groups), Y = SR, bubble size = planning time (bigger = slower)
# Color = config. Immediately shows "high SR + small bubble = ideal".

def make_bubble():
    configs  = list(DATA.keys())
    n_cfg    = len(configs)
    n_bm     = len(BENCHMARKS)

    # Offsets within each benchmark group so bubbles don't overlap
    x_offsets = np.linspace(-0.30, 0.30, n_cfg)

    # Map planning time to bubble area: normalize so max time → area 900
    all_times = [t for d in DATA.values() for t in d["time"] if t is not None]
    t_max = max(all_times)
    def time_to_area(t):
        return 120 + (t / t_max) * 780   # range 120–900

    fig, ax = plt.subplots(figsize=(10, 4.0))

    for ci, cfg in enumerate(configs):
        for bi in range(n_bm):
            sr = DATA[cfg]["sr"][bi]
            t  = DATA[cfg]["time"][bi]
            if sr is None or t is None:
                continue
            x  = bi + x_offsets[ci]
            area = time_to_area(t)

            # Draw bubble
            ax.scatter(x, sr,
                       s=area, c=COLORS[cfg], alpha=0.82,
                       edgecolors="white", linewidth=0.8, zorder=4)

            # SR value inside bubble (only if bubble large enough)
            if area > 250:
                txt = f"{sr:.2f}" if sr > 0 else "F"
                fontsize = 6.5 if area > 450 else 5.5
                ax.text(x, sr, txt,
                        ha="center", va="center",
                        fontsize=fontsize, color="white",
                        fontweight="bold", zorder=5)

    # Benchmark labels and group separators
    ax.set_xticks(range(n_bm))
    ax.set_xticklabels(FULL_LABELS, fontsize=8.5)
    for sep in [2.5, 5.5]:
        ax.axvline(sep, color="#CCCCCC", lw=1.2, ls="--", zorder=1)

    # Grid-size group labels on top
    for xc, lbl in [(1.0, "10×10"), (4.0, "12×12"), (6.5, "15×15")]:
        ax.text(xc, 1.115, lbl, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#444")

    ax.set_ylim(-0.06, 1.18)
    ax.set_xlim(-0.6, n_bm - 0.4)
    ax.axhline(1.0, color="#DDDDDD", lw=0.8, ls=":", zorder=1)
    ax.set_ylabel("Success Rate (SR)", fontsize=10)
    ax.set_xlabel("Benchmark", fontsize=10)
    ax.grid(axis="y", color="#F0F0F0", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # ── Legends ──
    # Config color legend
    cfg_handles = [
        mpatches.Patch(fc=COLORS[c], ec="white", label=c)
        for c in configs
    ]
    l_cfg = ax.legend(handles=cfg_handles,
                      loc="lower left", fontsize=8.5,
                      title="Config", title_fontsize=8.5,
                      framealpha=0.92, edgecolor="#CCCCCC",
                      bbox_to_anchor=(0.01, 0.01))

    # Bubble size legend (time reference)
    ref_times = [1, 10, 30]
    size_handles = [
        ax.scatter([], [], s=time_to_area(t), c="#AAAAAA",
                   edgecolors="white", lw=0.8,
                   label=f"{t} s")
        for t in ref_times
    ]
    ax.legend(handles=size_handles,
              loc="lower right", fontsize=8.5,
              title="Avg plan time", title_fontsize=8.5,
              framealpha=0.92, edgecolor="#CCCCCC",
              bbox_to_anchor=(0.99, 0.01))
    ax.add_artist(l_cfg)

    ax.set_title(
        "Success Rate (Y) and Planning Time (bubble size) across all benchmarks\n"
        "Smaller bubble = faster planning   ·   Higher position = better SR",
        fontsize=9.5, color="#333", pad=6)

    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"efficiency_scatter.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ── Figure 2: SR heatmap ───────────────────────────────────────────────────────

def make_heatmap():
    configs = list(DATA.keys())
    n_cfg   = len(configs)
    n_bm    = len(BENCHMARKS)

    # Build numeric matrix; None → NaN for heatmap coloring
    raw = [[DATA[c]["sr"][i] for i in range(n_bm)] for c in configs]
    matrix = np.array([[v if v is not None else np.nan for v in row]
                        for row in raw], dtype=float)

    # Row labels: show avg SR for non-None entries
    def row_label(cfg):
        vals = [v for v in DATA[cfg]["sr"] if v is not None]
        if not vals:
            return cfg
        return f"{cfg}  (avg {sum(vals)/len(vals):.3f})"

    row_labels = [row_label(c) for c in configs]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Cell text
    for i in range(n_cfg):
        for j in range(n_bm):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "TBD", ha="center", va="center",
                        fontsize=7.5, color="#888888", fontstyle="italic")
            else:
                color = "white" if v < 0.25 or v > 0.85 else "#222"
                txt = f"{v:.2f}" if v > 0 else "FAIL"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(range(n_bm))
    ax.set_xticklabels(FULL_LABELS, fontsize=8)
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_title("Success Rate across all configurations and benchmarks\n"
                 "(LLM models: Stage 1 rules only, same GNN scorer)",
                 fontsize=10, fontweight="bold", pad=8)

    # Horizontal separator between baselines and LLM models
    ax.axhline(2.5, color="white", lw=2.5, zorder=5)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_bm, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cfg, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # Vertical separators for grid size groups
    for x in [2.5, 5.5]:
        ax.axvline(x, color="white", lw=2.5, zorder=5)

    # Group labels above
    for x_center, label in [(1.0, "10×10"), (4.0, "12×12"), (6.5, "15×15")]:
        ax.text(x_center, -0.85, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#444",
                transform=ax.get_xaxis_transform())

    cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02)
    cbar.set_label("SR", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(FIG_DIR, f"sr_heatmap.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    make_bubble()
    make_heatmap()
    print("Done.")
