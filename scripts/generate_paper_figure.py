"""
Generate a 4-panel paper figure for the LLM-Flax pipeline.

Panels:
  (a) Full Problem      — all objects, robot, goal
  (b) LLM Object Scores — score heatmap over the maze (Step 1 input)
  (c) After Pruning     — objects below threshold removed (Step 1 output)
  (d) Final Plan        — planning set after Step 2+3 with plan path overlay

Usage:
    cd /home/leedk/flax
    python scripts/generate_paper_figure.py --size 10 --difficulty hard --problem_idx 0
    python scripts/generate_paper_figure.py --size 12 --difficulty hard --problem_idx 3

Output:
    paper/figures/pipeline_NxN_DIFF_IDX.{pdf,png}
"""

import os
import sys
import argparse
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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
from guidance import LLMObjectGuidance
from guidance.llm_recovery_guidance import LLMRecoveryGuidance
from my_utils.pddl_utils import _create_planner, DIRECTIONS

# ── Cell type constants (match minigrid_mazenamo_visualization.py) ────────────
EMPTY        = 0
WALL         = 1
HEAVY_OBJECT = 2
LIGHT_OBJECT = 3
ROBOT        = 4
GOAL         = 5

# ── Visual style ──────────────────────────────────────────────────────────────
CELL_COLOR = {
    EMPTY:        "#FAFAFA",
    WALL:         "#2C2C2C",
    HEAVY_OBJECT: "#3A7EC5",   # steel blue
    LIGHT_OBJECT: "#E8A020",   # amber
    ROBOT:        "#27AE60",   # green
    GOAL:         "#E74C3C",   # red
}
CELL_LABEL = {
    HEAVY_OBJECT: "H",
    LIGHT_OBJECT: "L",
    ROBOT:        "R",
    GOAL:         "G",
}
PRUNED_COLOR   = "#D0D0D0"
SCORE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "llm_scores",
    [(0.85, 0.15, 0.10, 0.75),   # low  = red-ish
     (0.10, 0.70, 0.20, 0.75)])  # high = green-ish


# ── Coordinate helpers ────────────────────────────────────────────────────────

def obj_to_xy(obj_name: str, size: int):
    """PDDL object name → (col, row) = (x, y) in grid[x, y] coords."""
    if obj_name.startswith("o"):
        k = int(obj_name[1:])
        return (k - 1) % size, (k - 1) // size
    if obj_name.startswith("p"):
        k = int(obj_name[1:])
        return k % size, k // size
    return None


def trace_plan_path(robot_pos, plan):
    """Return list of (x, y) grid positions visited by the robot.

    Direction is extracted directly from the action name suffix
    (e.g. 'moveforwardwhenright' → dx=+1, dy=0).
    """
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


# ── Grid drawing ──────────────────────────────────────────────────────────────

OBJECT_CELLS = {HEAVY_OBJECT, LIGHT_OBJECT, ROBOT, GOAL}


def draw_grid(ax, grid, size, title,
              score_map=None,
              ignored_xy=None,
              plan_path=None,
              label_size=7,
              scores_only=False):
    """Draw one maze panel.

    grid        : numpy array [col, row] with EMPTY/WALL/HEAVY.../GOAL values
    score_map   : dict {(x,y): float 0-1}  — overlay heatmap on object cells
    ignored_xy  : set of (x,y) to render as pruned (greyed)
    plan_path   : list of (x,y) tuples for robot trajectory
    scores_only : if True, suppress cell labels and only show score numbers
    """
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)        # y=0 at top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)

    ignored = set(ignored_xy) if ignored_xy else set()

    # ── Background cells ──
    for x in range(size):
        for y in range(size):
            val = int(grid[x, y])
            if (x, y) in ignored and val not in (WALL, GOAL, ROBOT):
                fc = PRUNED_COLOR
                alpha = 0.6
            else:
                fc = CELL_COLOR.get(val, CELL_COLOR[EMPTY])
                alpha = 1.0

            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor="#888888",
                linewidth=0.3, alpha=alpha, zorder=1)
            ax.add_patch(rect)

            # Cell label (suppressed in score-only panel)
            if val in CELL_LABEL and (x, y) not in ignored and not scores_only:
                ax.text(x + 0.5, y + 0.5, CELL_LABEL[val],
                        ha="center", va="center",
                        fontsize=label_size, fontweight="bold",
                        color="white", zorder=3)

    # ── Score heatmap overlay (object cells only) ──
    if score_map:
        norm = Normalize(vmin=0.0, vmax=1.0)
        for (x, y), score in score_map.items():
            val = int(grid[x, y])
            if val not in OBJECT_CELLS:      # skip walls and empty
                continue
            rgba = SCORE_CMAP(norm(float(score)))
            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                facecolor=rgba, edgecolor="none", zorder=2)
            ax.add_patch(rect)
            # Type label (small, top-left corner)
            if val in CELL_LABEL:
                ax.text(x + 0.18, y + 0.22, CELL_LABEL[val],
                        ha="center", va="center",
                        fontsize=max(4, label_size - 2),
                        fontweight="bold", color="white", alpha=0.9, zorder=4)
            # Score value (center)
            ax.text(x + 0.5, y + 0.6, f"{score:.2f}",
                    ha="center", va="center",
                    fontsize=max(4, label_size - 1),
                    color="white", zorder=4)

    # ── Plan path ──
    if plan_path and len(plan_path) > 1:
        n = len(plan_path)
        for i in range(n - 1):
            x0, y0 = plan_path[i][0] + 0.5, plan_path[i][1] + 0.5
            x1, y1 = plan_path[i + 1][0] + 0.5, plan_path[i + 1][1] + 0.5
            frac = i / max(1, n - 2)
            color = plt.cm.plasma(0.15 + 0.7 * frac)
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=1.4,
                    mutation_scale=9),
                zorder=5)
        # Start dot
        sx, sy = plan_path[0][0] + 0.5, plan_path[0][1] + 0.5
        ax.plot(sx, sy, "o", color=plt.cm.plasma(0.15),
                ms=5, zorder=6)
        # End dot
        ex, ey = plan_path[-1][0] + 0.5, plan_path[-1][1] + 0.5
        ax.plot(ex, ey, "*", color=plt.cm.plasma(0.85),
                ms=7, zorder=6)


# ── Pre-computed guidance (avoids LLM overhead inside planner timing) ────────

class PrecomputedGuidance:
    """Wraps a pre-scored dict so score_object() has zero overhead at plan time.

    Usage:
      1. Call LLMObjectGuidance once to score all objects → llm_scores dict
      2. Wrap in PrecomputedGuidance → pass to FlaxPlanner / LLMFlaxPlanner
      3. Step 1 gets the full timeout/6 budget without LLM latency eating into it
    """
    def __init__(self, scores: dict):
        self._scores = scores  # {obj_name: float}

    def score_object(self, obj, state) -> float:
        return self._scores.get(obj.name, 0.5)

    def seed(self, s):
        pass

    def train(self, *args, **kwargs):
        pass


# ── Planner runner ────────────────────────────────────────────────────────────

def run_planner(args):
    """Set up env/planner, pre-score objects with LLM, then run planner.

    Key design: LLM scoring happens BEFORE planner start_time, so Step 1 gets
    its full timeout/6 budget without LLM latency eating into it.

    Returns (plan, vis_info, state, llm_scores).
    """
    # Point pddlgym symlink at the right problem set
    link   = os.path.abspath("pddlgym/pddl/mazenamo_test")
    target = os.path.abspath(
        f"pddl_files/problems/mazenamo_problems/pddl_{args.size}x{args.size}_{args.difficulty}")
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        import shutil; shutil.rmtree(link)
    os.symlink(target, link)
    print(f"Symlink: {link} → {target}")

    base_planner = _create_planner("fd-lama-first")

    # Load environment and state first
    env = pddlgym.make("PDDLEnvMazenamoTest-v0")
    idx = None
    for i, prob in enumerate(env.problems):
        if f"mazenamo_problem_{args.problem_idx}.pddl" in prob.problem_fname:
            idx = i
            break
    assert idx is not None, \
        f"Problem {args.problem_idx} not found in {len(env.problems)} problems"

    env.fix_problem_index(idx)
    state, _ = env.reset()
    if type(state.goal).__name__ == "Literal":
        state = state.with_goal(LiteralConjunction([state.goal]))

    # ── Pre-score all objects with LLM (outside planner timing) ──────────────
    print("\nPre-scoring objects with LLM (outside planner clock)...")
    t_llm = time.time()
    llm_guider = LLMObjectGuidance(debug=False, use_cot="none")
    llm_guider.seed(0)
    llm_guider.train("Mazenamo", timeout=120)
    llm_scores = {
        obj.name: llm_guider.score_object(obj, state)
        for obj in state.objects
    }
    print(f"LLM scoring done in {time.time()-t_llm:.1f}s  "
          f"({len(llm_scores)} objects scored)")

    # ── Run planner with pre-computed scores (no LLM overhead at plan time) ──
    cached_guider = PrecomputedGuidance(llm_scores)

    if args.planner_type == "flax":
        planner = FlaxPlanner(
            is_strips_domain=True,
            base_planner=base_planner, search_guider=cached_guider, seed=0,
            complementary_rules=args.cmpl_rules,
            relaxation_rules=args.relx_rules)
    else:  # llmflax
        llm_recovery = LLMRecoveryGuidance(debug=False)
        planner = LLMFlaxPlanner(
            is_strips_domain=True,
            base_planner=base_planner, search_guider=cached_guider, seed=0,
            complementary_rules=args.cmpl_rules,
            relaxation_rules=args.relx_rules,
            llm_recovery=llm_recovery)

    print(f"\nRunning {args.planner_type} on "
          f"{args.size}x{args.size} {args.difficulty} problem {args.problem_idx}...")
    t0 = time.time()
    plan, vis_info = planner(env.domain, state, timeout=args.timeout)
    print(f"Plan found: length={len(plan)}, time={time.time()-t0:.1f}s")
    return plan, vis_info, state, llm_scores


# ── Main figure generator ─────────────────────────────────────────────────────

def generate_figure(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Load grid map
    map_dir  = (f"pddl_files/problems/mazenamo_problems/"
                f"map_{args.size}x{args.size}_{args.difficulty}")
    map_file = f"{map_dir}/mazenamo_map_{args.problem_idx}.pkl"
    with open(map_file, "rb") as f:
        prob_dict = pickle.load(f)
    grid      = prob_dict["grid"]           # shape (size, size), grid[col, row]
    robot_pos = prob_dict["robot_pos"]      # (col, row)
    size      = args.size

    # Run planner (LLM scoring happens before planner clock starts)
    plan, vis_info, state, llm_scores = run_planner(args)

    # ── Derived data ──────────────────────────────────────────────────────────

    def to_xy_set(obj_set):
        result = set()
        for obj in (obj_set or set()):
            xy = obj_to_xy(obj.name, size)
            if xy is not None:
                result.add(xy)
        return result

    # (b) LLM scores
    score_map = {}
    for obj_name, score in llm_scores.items():
        xy = obj_to_xy(obj_name, size)
        if xy is not None:
            score_map[xy] = float(score)

    # (c) Step 1: objects pruned by threshold decay
    step1_ignored_xy = to_xy_set(vis_info.get("gnn_ignored_objects"))

    # (d) Step 2: objects removed by relaxation rules
    relx_ignored = vis_info.get("relx_ignored_objects")
    step2_ignored_xy = to_xy_set(relx_ignored)
    relaxed_plan     = vis_info.get("relaxed_plan") or []
    relaxed_path     = trace_plan_path(robot_pos, relaxed_plan) if relaxed_plan else None

    # (e) Step 3: objects after complementary expansion
    cmpl_ignored = vis_info.get("cmpl_ignored_objects")
    if cmpl_ignored is not None:
        step3_ignored_xy = to_xy_set(cmpl_ignored)
    elif relx_ignored is not None:
        step3_ignored_xy = step2_ignored_xy   # no complementary step ran
    else:
        step3_ignored_xy = step1_ignored_xy

    # (f) Final plan path
    plan_path = trace_plan_path(robot_pos, plan)

    # ── Layout: 2 rows × 3 cols ───────────────────────────────────────────────
    cell_inch = 0.30
    fig_w = 3 * size * cell_inch + 1.0
    fig_h = 2 * size * cell_inch + 0.9

    fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.08, hspace=0.12,
                        left=0.01, right=0.92,
                        top=0.93, bottom=0.12)

    label_sz = max(5, 9 - size // 3)
    ax = axes  # axes[row, col]

    # Row 0
    draw_grid(ax[0, 0], grid, size,
              "(a) Full Problem",
              label_size=label_sz)

    draw_grid(ax[0, 1], grid, size,
              "(b) LLM Object Scoring",
              score_map=score_map,
              label_size=label_sz, scores_only=True)

    draw_grid(ax[0, 2], grid, size,
              "(c) Step\u00a01: After Pruning",
              ignored_xy=step1_ignored_xy,
              label_size=label_sz)

    # Row 1
    draw_grid(ax[1, 0], grid, size,
              "(d) Step\u00a02: Relaxed Problem",
              ignored_xy=step2_ignored_xy,
              plan_path=relaxed_path,
              label_size=label_sz)

    draw_grid(ax[1, 1], grid, size,
              "(e) Step\u00a03: Complementary Set",
              ignored_xy=step3_ignored_xy,
              label_size=label_sz)

    draw_grid(ax[1, 2], grid, size,
              "(f) Final Plan",
              ignored_xy=step3_ignored_xy,
              plan_path=plan_path,
              label_size=label_sz)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(fc=CELL_COLOR[HEAVY_OBJECT], ec="#555", lw=0.5,
                       label="Heavy box (H)"),
        mpatches.Patch(fc=CELL_COLOR[LIGHT_OBJECT], ec="#555", lw=0.5,
                       label="Light box (L)"),
        mpatches.Patch(fc=CELL_COLOR[ROBOT],        ec="#555", lw=0.5,
                       label="Robot (R)"),
        mpatches.Patch(fc=CELL_COLOR[GOAL],         ec="#555", lw=0.5,
                       label="Goal (G)"),
        mpatches.Patch(fc=PRUNED_COLOR,             ec="#555", lw=0.5,
                       label="Excluded object"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=5,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.46, 0.01))

    # ── Score colorbar — vertical strip at far right ──────────────────────────
    sm = ScalarMappable(cmap=SCORE_CMAP, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    ax_ref = ax[0, 2]
    pos    = ax_ref.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.012,
                            pos.y0 + pos.height * 0.12,
                            0.014,
                            pos.height * 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"])
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_ylabel("LLM score (b)", fontsize=6, labelpad=3,
                       rotation=270, va="bottom")

    # ── Save ──────────────────────────────────────────────────────────────────
    stem = f"pipeline_{size}x{size}_{args.difficulty}_{args.problem_idx}"
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 4-panel LLM-Flax pipeline figure for the paper.")
    parser.add_argument("--size",         type=int,   default=10,
                        help="Grid size (default: 10)")
    parser.add_argument("--difficulty",   type=str,   default="hard",
                        choices=["easy", "medium", "hard", "expert"],
                        help="Problem difficulty (default: hard)")
    parser.add_argument("--problem_idx",  type=str,   default="0",
                        help="Problem index (default: 0)")
    parser.add_argument("--planner_type", type=str,   default="llmflax",
                        choices=["flax", "llmflax"],
                        help="Planner type (default: llmflax)")
    parser.add_argument("--timeout",      type=float, default=30.0,
                        help="Per-problem timeout in seconds (default: 30)")
    parser.add_argument("--cmpl_rules",   type=str,
                        default="config/mazenamo_complementary_rules_qwen2.5-14b.json",
                        help="Complementary rules JSON")
    parser.add_argument("--relx_rules",   type=str,
                        default="config/mazenamo_relaxation_rules_qwen2.5-14b.json",
                        help="Relaxation rules JSON")
    parser.add_argument("--out_dir",      type=str,   default="paper/figures",
                        help="Output directory (default: paper/figures)")
    args = parser.parse_args()
    generate_figure(args)
