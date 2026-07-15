"""
Qualitative visualization of a solved SokomindPlus problem (LLM-Flax,
Gemma3-12B rules), in the same grid style as paper/figures/vis/ (MazeNamo).
SokomindPlus shares MazeNamo's grid mechanic (rAt/oAt/posEmpty, push-box
movement) but has a single box type (no heavy/light distinction) and
0-indexed "p0..pN" position names / "b0..bN" box names instead of MazeNamo's
"o"/"p" 1-indexed convention.

Produces 3 panels: (a) full problem, (b) after Step-1 pruning (grey =
excluded), (c) final plan with robot path traced.

Usage: python scripts/generate_sokomindplus_vis.py --problem_idx 0
Output: paper/figures/vis/vis_sokomindplus_{a,b,c}.{pdf,png}
"""

import os
import sys
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pddlgym
from pddlgym.structs import LiteralConjunction
from planning import FlaxPlanner, PlanningFailure, PlanningTimeout
from guidance import NoSearchGuidance
from my_utils.pddl_utils import _create_planner

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures", "vis")
CMPL_RULES = "config/sokomindplus_complementary_rules_gemma3-12b.json"
RELX_RULES = "config/sokomindplus_relaxation_rules_gemma3-12b.json"

C_EMPTY, C_BOX, C_ROBOT, C_GOAL, C_EXCLUDE = "#FAFAFA", "#3A7EC5", "#27AE60", "#E74C3C", "#D0D0D0"


def _n(v):
    return str(v).split(":")[0]


def pos_xy(name, size):
    """'p123' -> (123 % size, 123 // size)."""
    k = int(name[1:])
    return k % size, k // size


def build_env():
    link = os.path.abspath("pddlgym/pddl/sokomindplus_test")
    target = os.path.abspath("pddl_files/problems/sokomindplus_problems/pddl_test_15x15")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(target, link)
    domain_link = os.path.abspath("pddlgym/pddl/sokomindplus.pddl")
    if os.path.islink(domain_link) or os.path.exists(domain_link):
        os.unlink(domain_link)
    os.symlink(os.path.abspath("pddl_files/domains/sokomindplus.pddl"), domain_link)
    return pddlgym.make("PDDLEnvSokomindplusTest-v0")


def extract(state, size=15):
    by_pred = {}
    for lit in state.literals:
        by_pred.setdefault(lit.predicate.name, []).append(lit.variables)

    o_at = {_n(v[0]): _n(v[1]) for v in by_pred.get("oat", [])}
    r_at = {_n(v[0]): _n(v[1]) for v in by_pred.get("rat", [])}
    goal_boxes = {}
    for lit in state.goal.literals:
        if lit.predicate.name == "oat":
            goal_boxes[_n(lit.variables[0])] = _n(lit.variables[1])
    return dict(o_at=o_at, r_at=r_at, goal_boxes=goal_boxes)


def replay_robot_path(plan, world, size):
    r_name = list(world["r_at"].keys())[0]
    cur = world["r_at"][r_name]
    path = [pos_xy(cur, size)]
    for act in plan:
        name = act.predicate.name.lower()
        args = [_n(v) for v in act.variables]
        # MoveForwardWhen*(r, p1, p2): robot ends at p2 = args[2].
        # PushBoxWhen*(r, o, p1, p2, p3): robot ends at p2 = args[3] (pushes box p2->p3).
        if name.startswith("moveforward"):
            path.append(pos_xy(args[2], size))
        elif name.startswith("pushbox"):
            path.append(pos_xy(args[3], size))
    return path


def draw_panel(ax, world, size, excluded=None, title="", plan_path=None):
    excluded = excluded or set()
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for x in range(size):
        for y in range(size):
            rect = mpatches.FancyBboxPatch((x + 0.04, y + 0.04), 0.92, 0.92,
                                           boxstyle="round,pad=0.04",
                                           facecolor=C_EMPTY, edgecolor="#DDDDDD",
                                           linewidth=0.3, zorder=1)
            ax.add_patch(rect)

    for b, p in world["o_at"].items():
        x, y = pos_xy(p, size)
        grey = b in excluded
        is_goal = b in world["goal_boxes"]
        color = C_EXCLUDE if grey else (C_GOAL if is_goal else C_BOX)
        rect = mpatches.FancyBboxPatch((x + 0.08, y + 0.08), 0.84, 0.84,
                                       boxstyle="round,pad=0.03",
                                       facecolor=color, edgecolor="white",
                                       linewidth=0.4, zorder=2)
        ax.add_patch(rect)

    for r, p in world["r_at"].items():
        x, y = pos_xy(p, size)
        grey = r in excluded
        ax.plot(x + 0.5, y + 0.5, marker="o",
               color=C_EXCLUDE if grey else C_ROBOT, markersize=10,
               markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    if plan_path and len(plan_path) > 1:
        n = len(plan_path)
        for i in range(n - 1):
            x0, y0 = plan_path[i][0] + 0.5, plan_path[i][1] + 0.5
            x1, y1 = plan_path[i + 1][0] + 0.5, plan_path[i + 1][1] + 0.5
            frac = i / max(1, n - 2)
            color = plt.cm.plasma(0.15 + 0.7 * frac)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                       mutation_scale=8), zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_idx", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=15)
    args = parser.parse_args()

    env = build_env()
    env.fix_problem_index(args.problem_idx)
    state, _ = env.reset()
    if type(state.goal).__name__ == "Literal":
        state = state.with_goal(LiteralConjunction([state.goal]))

    world = extract(state, size=args.size)

    guidance = NoSearchGuidance(); guidance.seed(0)
    planner = FlaxPlanner(is_strips_domain=True, base_planner=_create_planner("fd-lama-first"),
                          search_guider=guidance, seed=0,
                          complementary_rules=CMPL_RULES, relaxation_rules=RELX_RULES)
    plan, vis_info = planner(env.domain, state, timeout=args.timeout)
    print(f"Solved problem {args.problem_idx}: plan length {len(plan)}")

    excluded = {o.name for o in (vis_info.get("cmpl_ignored_objects") or set())}
    plan_path = replay_robot_path(plan, world, args.size)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_panel(ax, world, args.size, title=f"(a) Full Problem\n{len(state.objects)} objects")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_sokomindplus_a.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_panel(ax, world, args.size, excluded=excluded,
              title=f"(b) After Pruning\n{len(state.objects) - len(excluded)} retained")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_sokomindplus_b.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_panel(ax, world, args.size, excluded=excluded, plan_path=plan_path,
              title=f"(c) Final Plan\nlength {len(plan)}")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_sokomindplus_c.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 3 panels -> {OUT_DIR}/vis_sokomindplus_{{a,b,c}}.{{pdf,png}}")


if __name__ == "__main__":
    main()
