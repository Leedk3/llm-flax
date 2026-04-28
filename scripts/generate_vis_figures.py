"""
Generate appendix visualization figures for all 8 MazeNamo benchmarks.

For each benchmark, produces 3 separate PDF/PNG files:
  vis_<size>x<size>_<difficulty>_a.pdf  — (a) Full Problem
  vis_<size>x<size>_<difficulty>_b.pdf  — (b) LLM Object Scoring
  vis_<size>x<size>_<difficulty>_c.pdf  — (c) After Pruning & Final Plan

Usage (from /home/leedk/flax/):
    conda run -n flax python scripts/generate_vis_figures.py
    conda run -n flax python scripts/generate_vis_figures.py --benchmarks 10x10_hard 12x12_expert
    conda run -n flax python scripts/generate_vis_figures.py --problem_idx 3
"""

import os
import sys
import shutil
import pickle
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pddlgym
from pddlgym.structs import LiteralConjunction
from planning import (PlanningTimeout, PlanningFailure, validate_strips_plan,
                      FlaxPlanner, LLMFlaxPlanner)
from guidance import LLMObjectGuidance, GNNSearchGuidance
from my_utils.pddl_utils import _create_planner

# ── Benchmark definitions ──────────────────────────────────────────────────────
BENCHMARKS = [
    dict(size=10, difficulty="easy",   timeout=10.0),
    dict(size=10, difficulty="medium", timeout=10.0),
    dict(size=10, difficulty="hard",   timeout=10.0),
    dict(size=12, difficulty="medium", timeout=30.0),
    dict(size=12, difficulty="hard",   timeout=30.0),
    dict(size=12, difficulty="expert", timeout=30.0),
    dict(size=15, difficulty="medium", timeout=40.0),
    dict(size=15, difficulty="hard",   timeout=40.0),
]

CMPL_RULES = "config/mazenamo_complementary_rules_qwen2.5-14b.json"
RELX_RULES = "config/mazenamo_relaxation_rules_qwen2.5-14b.json"
MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "paper", "figures", "vis")

# ── Visual style (shared with generate_paper_figure.py) ───────────────────────
EMPTY        = 0
WALL         = 1
HEAVY_OBJECT = 2
LIGHT_OBJECT = 3
ROBOT        = 4
GOAL         = 5

CELL_COLOR = {
    EMPTY:        "#FAFAFA",
    WALL:         "#2C2C2C",
    HEAVY_OBJECT: "#3A7EC5",
    LIGHT_OBJECT: "#E8A020",
    ROBOT:        "#27AE60",
    GOAL:         "#E74C3C",
}
CELL_LABEL = {
    HEAVY_OBJECT: "H",
    LIGHT_OBJECT: "L",
    ROBOT:        "R",
    GOAL:         "G",
}
PRUNED_COLOR = "#D0D0D0"
OBJECT_CELLS = {HEAVY_OBJECT, LIGHT_OBJECT, ROBOT, GOAL}
SCORE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "llm_scores",
    [(0.85, 0.15, 0.10, 0.75),
     (0.10, 0.70, 0.20, 0.75)])


# ── Helpers ────────────────────────────────────────────────────────────────────

def obj_to_xy(obj_name, size):
    if obj_name.startswith("o"):
        k = int(obj_name[1:])
        return (k - 1) % size, (k - 1) // size
    if obj_name.startswith("p"):
        k = int(obj_name[1:])
        return k % size, k // size
    return None


def trace_plan_path(robot_pos, plan):
    DIR_DELTA = {
        "whenright": ( 1,  0),
        "whenleft":  (-1,  0),
        "whenup":    ( 0, -1),
        "whendown":  ( 0,  1),
    }
    MOVE_PREFIXES = ("moveforward", "pushobstacle")
    x, y = int(robot_pos[0]), int(robot_pos[1])
    path = [(x, y)]
    for action in plan:
        aname = str(action).split("(")[0].lower()
        if any(aname.startswith(p) for p in MOVE_PREFIXES):
            for suffix, (dx, dy) in DIR_DELTA.items():
                if aname.endswith(suffix):
                    x, y = x + dx, y + dy
                    path.append((x, y))
                    break
    return path


def _set_symlink(size, difficulty):
    link   = os.path.abspath("pddlgym/pddl/mazenamo_test")
    target = os.path.abspath(
        f"pddl_files/problems/mazenamo_problems/pddl_{size}x{size}_{difficulty}")
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        shutil.rmtree(link)
    os.symlink(target, link)


# ── Single-panel drawing ───────────────────────────────────────────────────────

def draw_panel(ax, grid, size, title,
               score_map=None, ignored_xy=None, plan_path=None,
               scores_only=False, label_size=7):
    ignored = set(ignored_xy) if ignored_xy else set()
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)

    for x in range(size):
        for y in range(size):
            val = int(grid[x, y])
            if (x, y) in ignored and val not in (WALL, GOAL, ROBOT):
                fc, alpha = PRUNED_COLOR, 0.6
            else:
                fc, alpha = CELL_COLOR.get(val, CELL_COLOR[EMPTY]), 1.0
            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor="#888888",
                linewidth=0.3, alpha=alpha, zorder=1)
            ax.add_patch(rect)
            if val in CELL_LABEL and (x, y) not in ignored and not scores_only:
                ax.text(x + 0.5, y + 0.5, CELL_LABEL[val],
                        ha="center", va="center",
                        fontsize=label_size, fontweight="bold",
                        color="white", zorder=3)

    if score_map:
        norm = Normalize(vmin=0.0, vmax=1.0)
        for (x, y), score in score_map.items():
            val = int(grid[x, y])
            if val not in OBJECT_CELLS:
                continue
            rgba = SCORE_CMAP(norm(float(score)))
            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                facecolor=rgba, edgecolor="none", zorder=2)
            ax.add_patch(rect)
            if val in CELL_LABEL:
                ax.text(x + 0.18, y + 0.22, CELL_LABEL[val],
                        ha="center", va="center",
                        fontsize=max(4, label_size - 2),
                        fontweight="bold", color="white", alpha=0.9, zorder=4)
            ax.text(x + 0.5, y + 0.6, f"{score:.2f}",
                    ha="center", va="center",
                    fontsize=max(4, label_size - 1),
                    color="white", zorder=4)

    if plan_path and len(plan_path) > 1:
        n = len(plan_path)
        for i in range(n - 1):
            x0, y0 = plan_path[i][0] + 0.5,     plan_path[i][1] + 0.5
            x1, y1 = plan_path[i + 1][0] + 0.5, plan_path[i + 1][1] + 0.5
            frac  = i / max(1, n - 2)
            color = plt.cm.plasma(0.15 + 0.7 * frac)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.4, mutation_scale=9), zorder=5)
        sx, sy = plan_path[0][0] + 0.5,  plan_path[0][1] + 0.5
        ex, ey = plan_path[-1][0] + 0.5, plan_path[-1][1] + 0.5
        ax.plot(sx, sy, "o", color=plt.cm.plasma(0.15), ms=5, zorder=6)
        ax.plot(ex, ey, "*", color=plt.cm.plasma(0.85), ms=7, zorder=6)


# ── Save combined 3-panel figure ─────────────────────────────────────────────
# Fixed total width = 7.5 inches so all benchmarks produce identical-width PDFs.
# Layout: [panel_a | panel_b + colorbar | panel_c], each panel square (size×size).

_FIG_W   = 7.5   # total figure width in inches (fits 0.85\textwidth in IEEE)
_CBAR_W  = 0.18  # colorbar fraction of total width
_GAP     = 0.06  # gap between panels as fraction of total width
_TOP     = 0.91
_BOT     = 0.04


def save_combined(grid, size, stem,
                  score_map=None, ignored_xy=None, plan_path=None):
    """Save (a)+(b)+(c) as one combined PDF/PNG."""
    label_sz = max(5, 9 - size // 3)

    # Panel height = square panels → height equals panel width
    panel_w_frac = (1.0 - _CBAR_W - 4 * _GAP) / 3   # fraction of fig width
    panel_w_inch = _FIG_W * panel_w_frac
    fig_h = panel_w_inch + 0.55   # panel height + title/margin

    fig = plt.figure(figsize=(_FIG_W, fig_h))

    # Compute axes positions in figure-fraction coordinates
    g = _GAP
    cw = _CBAR_W
    pw = panel_w_frac
    # ax_a: left edge at g
    ax_a = fig.add_axes([g,          _BOT, pw, _TOP - _BOT])
    # ax_b: after ax_a + gap
    ax_b = fig.add_axes([g*2 + pw,   _BOT, pw, _TOP - _BOT])
    # colorbar: after ax_b + small gap
    ax_cb = fig.add_axes([g*2 + pw*2 + g*0.4, _BOT + (_TOP-_BOT)*0.12,
                           cw * 0.55, (_TOP - _BOT) * 0.76])
    # ax_c: after colorbar + gap
    ax_c = fig.add_axes([g*3 + pw*2 + cw, _BOT, pw, _TOP - _BOT])

    draw_panel(ax_a, grid, size, "(a) Full Problem",
               label_size=label_sz)
    draw_panel(ax_b, grid, size, "(b) LLM Object Scoring",
               score_map=score_map, scores_only=True, label_size=label_sz)
    draw_panel(ax_c, grid, size, "(c) After Pruning \u0026 Final Plan",
               ignored_xy=ignored_xy, plan_path=plan_path, label_size=label_sz)

    # Colorbar
    sm = ScalarMappable(cmap=SCORE_CMAP, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_ylabel("LLM score", fontsize=7, labelpad=3,
                       rotation=270, va="bottom")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Run one benchmark ──────────────────────────────────────────────────────────

def run_one(bm, problem_idx):
    size, diff, timeout = bm["size"], bm["difficulty"], bm["timeout"]
    tag  = f"{size}x{size}_{diff}"
    stem = f"vis_{tag}"

    print(f"\n{'='*60}")
    print(f"  {size}x{size} {diff}  (timeout={timeout}s, problem_idx={problem_idx})")
    print(f"{'='*60}")

    # ── Set symlink ──
    _set_symlink(size, diff)

    # ── Load grid map ──
    map_dir  = (f"pddl_files/problems/mazenamo_problems/"
                f"map_{size}x{size}_{diff}")
    map_file = f"{map_dir}/mazenamo_map_{problem_idx}.pkl"
    if not os.path.exists(map_file):
        print(f"  [SKIP] Map file not found: {map_file}")
        return

    with open(map_file, "rb") as f:
        prob_dict = pickle.load(f)
    grid      = prob_dict["grid"]
    robot_pos = prob_dict["robot_pos"]

    # ── Load env & state ──
    env = pddlgym.make("PDDLEnvMazenamoTest-v0")
    idx = None
    for i, prob in enumerate(env.problems):
        if f"mazenamo_problem_{problem_idx}.pddl" in prob.problem_fname:
            idx = i
            break
    if idx is None:
        print(f"  [SKIP] Problem {problem_idx} not found")
        return

    env.fix_problem_index(idx)
    state, _ = env.reset()
    if type(state.goal).__name__ == "Literal":
        state = state.with_goal(LiteralConjunction([state.goal]))

    # ── LLM object scoring — for panel (b) display only ──
    print("  Scoring objects with LLM (display only)...")
    t_llm = time.time()
    llm_guider = LLMObjectGuidance(debug=False, use_cot="none")
    llm_guider.seed(0)
    llm_guider.train("Mazenamo", timeout=120)
    llm_scores = {obj.name: llm_guider.score_object(obj, state)
                  for obj in state.objects}
    print(f"  LLM scoring: {time.time()-t_llm:.1f}s  ({len(llm_scores)} objects)")

    score_map = {}
    for obj_name, score in llm_scores.items():
        xy = obj_to_xy(obj_name, size)
        if xy is not None:
            score_map[xy] = float(score)

    # ── GNN guidance — for actual planning (matches main experiment config) ──
    print("  Loading GNN guidance...")
    gnn_guider = GNNSearchGuidance(
        training_planner=_create_planner("fd-opt-lmcut"),
        num_train_problems=1,
        num_epochs=301,
        criterion_name="bce",
        bce_pos_weight=10,
        load_from_file=True,
        load_dataset_from_file=True,
        dataset_file_prefix=os.path.join(MODEL_DIR, "training_data"),
        save_model_prefix=os.path.join(MODEL_DIR, "bce10_model_last_seed0"),
        is_strips_domain=True,
    )
    gnn_guider.seed(0)
    gnn_guider.train("Mazenamo", timeout=120)

    # ── Run planner with GNN scorer + LLM rules ──
    base_planner = _create_planner("fd-lama-first")
    planner = FlaxPlanner(
        is_strips_domain=True,
        base_planner=base_planner,
        search_guider=gnn_guider,
        seed=0,
        complementary_rules=CMPL_RULES,
        relaxation_rules=RELX_RULES)

    print("  Running planner...")
    try:
        t0 = time.time()
        plan, vis_info = planner(env.domain, state, timeout=timeout)
        print(f"  Plan: len={len(plan)}, t={time.time()-t0:.1f}s")
    except (PlanningTimeout, PlanningFailure) as e:
        print(f"  [FAIL] {e} — saving combined figure without (c)")
        save_combined(grid, size, stem, score_map=score_map)
        return

    # ── Derive vis sets ──
    def to_xy_set(obj_set):
        result = set()
        for obj in (obj_set or set()):
            xy = obj_to_xy(obj.name, size)
            if xy is not None:
                result.add(xy)
        return result

    cmpl_ignored = vis_info.get("cmpl_ignored_objects")
    relx_ignored = vis_info.get("relx_ignored_objects")
    step1_ignored = to_xy_set(vis_info.get("gnn_ignored_objects"))
    if cmpl_ignored is not None:
        final_ignored = to_xy_set(cmpl_ignored)
    elif relx_ignored is not None:
        final_ignored = to_xy_set(relx_ignored)
    else:
        final_ignored = step1_ignored

    plan_path = trace_plan_path(robot_pos, plan)

    # ── Save combined 3-panel figure ──
    save_combined(grid, size, stem,
                  score_map=score_map,
                  ignored_xy=final_ignored,
                  plan_path=plan_path)

    print(f"  Done: {tag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Subset e.g. '10x10_hard 12x12_expert'")
    parser.add_argument("--problem_idx", type=int, default=0,
                        help="Which problem index to visualize (default: 0)")
    args = parser.parse_args()

    bm_filter = set(args.benchmarks) if args.benchmarks else None

    for bm in BENCHMARKS:
        key = f"{bm['size']}x{bm['size']}_{bm['difficulty']}"
        if bm_filter and key not in bm_filter:
            continue
        run_one(bm, args.problem_idx)

    print("\nAll done.")


if __name__ == "__main__":
    main()
