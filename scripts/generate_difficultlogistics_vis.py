"""
Qualitative visualization of a solved DifficultLogistics problem (LLM-Flax,
Gemma3-12B rules), mirroring the style of paper/figures/vis/ (MazeNamo
grid visualizations) but adapted to DifficultLogistics's graph structure
(cities of locations connected by roads, airports connected by air-links).

Produces 3 panels, saved as separate files (matching the MazeNamo vis_*_a/b/c
convention):
  (a) Full problem: all locations/roads/trucks/airplanes/packages, goal
      packages and key-packages/locked hubs marked.
  (b) After Step 1 pruning + Step 2/3 rule expansion: excluded objects greyed.
  (c) Final plan: truck/airplane routes taken, overlaid on the pruned graph.

Usage (from /home/leedk/flax/):
    conda run -n flax python scripts/generate_difficultlogistics_vis.py --problem_idx 12
Output: paper/figures/vis/vis_difficultlogistics_{a,b,c}.{pdf,png}
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
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pddlgym
from pddlgym.structs import LiteralConjunction
from planning import FlaxPlanner
from guidance import NoSearchGuidance
from my_utils.pddl_utils import _create_planner

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures", "vis")

CMPL_RULES = "config/difficultlogistics_complementary_rules_gemma3-12b.json"
RELX_RULES = "config/difficultlogistics_relaxation_rules_gemma3-12b.json"

C_LOC       = "#BFC9CA"
C_LOCKED    = "#8E1B1B"
C_AIRPORT   = "#5B2C6F"
C_TRUCK     = "#1F618D"
C_AIRPLANE  = "#6C3483"
C_PKG       = "#B7950B"
C_PKG_GOAL  = "#1E8449"
C_PKG_KEY   = "#CA6F1E"
C_EXCLUDE   = "#D5D8DC"
C_ROAD      = "#AAB7B8"
C_AIRLINK   = "#D2B4DE"


def build_state(size_n=15):
    link = os.path.abspath("pddlgym/pddl/difficultlogistics_test")
    target = os.path.abspath("pddl_files/problems/difficultlogistics_problems/pddl_test")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(target, link)
    domain_link = os.path.abspath("pddlgym/pddl/difficultlogistics.pddl")
    if os.path.islink(domain_link) or os.path.exists(domain_link):
        os.unlink(domain_link)
    os.symlink(os.path.abspath("pddl_files/domains/difficultlogistics.pddl"), domain_link)

    env = pddlgym.make("PDDLEnvDifficultlogisticsTest-v0")
    return env


def _n(v):
    """Object display name, stripping pddlgym's ':type' suffix (e.g. 'city1:default')."""
    return str(v).split(":")[0]


def extract_world(state, domain):
    """Extract city/location/road/truck/package layout from a pddlgym state."""
    lits = {str(l.predicate.name): [] for l in state.literals}
    by_pred = {}
    for lit in state.literals:
        by_pred.setdefault(lit.predicate.name, []).append(lit.variables)

    def names(pred):
        return [_n(v[0]) for v in by_pred.get(pred, [])]

    cities = names("city")
    locations = names("location")
    trucks = names("truck")
    airplanes = names("airplane")
    airports = names("airport")
    objs = names("obj")
    key_packages = {_n(v[0]) for v in by_pred.get("key-package", [])}
    locked = {_n(v[0]) for v in by_pred.get("locked", [])}

    in_city = {_n(v[0]): _n(v[1]) for v in by_pred.get("in-city", [])}
    roads = [(_n(v[0]), _n(v[1])) for v in by_pred.get("road", [])]
    air_links = [(_n(v[0]), _n(v[1])) for v in by_pred.get("air-link", [])]
    at = {_n(v[0]): _n(v[1]) for v in by_pred.get("at", [])}
    on = {_n(v[0]): _n(v[1]) for v in by_pred.get("on", [])}
    switch_for = [(_n(v[0]), _n(v[1])) for v in by_pred.get("switch-for", [])]

    goal_locs = {}
    for lit in state.goal.literals:
        if lit.predicate.name == "at":
            goal_locs[_n(lit.variables[0])] = _n(lit.variables[1])

    return dict(cities=cities, locations=locations, trucks=trucks,
               airplanes=airplanes, airports=airports, objs=objs,
               key_packages=key_packages, locked=locked, in_city=in_city,
               roads=roads, air_links=air_links, at=at, on=on,
               switch_for=switch_for, goal_locs=goal_locs)


def layout(world):
    """City-clustered layout: cities on a big circle, locations within each
    city on a small circle around the city center."""
    pos = {}
    cities = sorted(world["cities"])
    n_c = len(cities)
    city_center = {}
    R_CITY = 6.0
    for i, c in enumerate(cities):
        angle = 2 * math.pi * i / max(n_c, 1)
        city_center[c] = (R_CITY * math.cos(angle), R_CITY * math.sin(angle))

    for c in cities:
        locs_c = sorted([l for l in world["locations"] if world["in_city"].get(l) == c])
        n_l = len(locs_c)
        cx, cy = city_center[c]
        r_loc = 1.3
        for j, l in enumerate(locs_c):
            a = 2 * math.pi * j / max(n_l, 1)
            pos[l] = (cx + r_loc * math.cos(a), cy + r_loc * math.sin(a))
    return pos, city_center


def draw_panel(ax, world, pos, city_center, excluded=None, title="", plan_routes=None):
    excluded = excluded or set()

    for c, (cx, cy) in city_center.items():
        ax.add_patch(plt.Circle((cx, cy), 1.9, facecolor="#F4F6F6",
                                edgecolor="#AAB7B8", linewidth=1.0, zorder=0))
        ax.text(cx, cy + 2.05, c, ha="center", fontsize=8, fontweight="bold", color="#566573")

    for a, b in world["roads"]:
        if a in pos and b in pos:
            grey = a in excluded or b in excluded
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                   color=C_EXCLUDE if grey else C_ROAD, linewidth=1.0, zorder=1)
    for a, b in world["air_links"]:
        if a in pos and b in pos:
            grey = a in excluded or b in excluded
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                   color=C_EXCLUDE if grey else C_AIRLINK, linewidth=1.3,
                   linestyle="--", zorder=1)

    for l in world["locations"]:
        if l not in pos:
            continue
        grey = l in excluded
        color = C_EXCLUDE if grey else (C_LOCKED if l in world["locked"] else
                                        (C_AIRPORT if l in world["airports"] else C_LOC))
        ax.add_patch(plt.Circle(pos[l], 0.17, facecolor=color, edgecolor="white",
                                linewidth=0.6, zorder=2))

    # trucks / airplanes
    for t in world["trucks"]:
        loc = world["at"].get(t)
        if loc in pos:
            grey = t in excluded
            ax.add_patch(plt.Rectangle((pos[loc][0] - 0.12, pos[loc][1] - 0.12), 0.24, 0.24,
                                       facecolor=C_EXCLUDE if grey else C_TRUCK,
                                       edgecolor="white", linewidth=0.5, zorder=3))
    for p in world["airplanes"]:
        loc = world["at"].get(p)
        if loc in pos:
            grey = p in excluded
            ax.plot(pos[loc][0], pos[loc][1], marker="^",
                   color=C_EXCLUDE if grey else C_AIRPLANE, markersize=8, zorder=3)

    # packages
    for o in world["objs"]:
        loc = world["at"].get(o) or world["on"].get(o)
        if loc is None or loc not in pos:
            continue
        grey = o in excluded
        is_key = o in world["key_packages"]
        is_goal = o in world["goal_locs"]
        color = C_EXCLUDE if grey else (C_PKG_KEY if is_key else
                                        (C_PKG_GOAL if is_goal else C_PKG))
        jitter = (np.random.RandomState(hash(o) % (2**31)).uniform(-0.25, 0.25, 2))
        ax.plot(pos[loc][0] + jitter[0], pos[loc][1] + jitter[1], marker="o",
               color=color, markersize=4, zorder=4, markeredgecolor="white", markeredgewidth=0.3)

    if plan_routes:
        for (mover, path) in plan_routes:
            pts = [pos[l] for l in path if l in pos]
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                ax.annotate("", xy=pts[i+1], xytext=pts[i],
                           arrowprops=dict(arrowstyle="-|>", color="#B3005E",
                                           lw=1.4, mutation_scale=10, alpha=0.85),
                           zorder=5)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def replay_plan_routes(plan, world):
    """Track truck/airplane location over the plan to draw route arrows."""
    cur_loc = dict(world["at"])
    routes = {m: [cur_loc[m]] for m in world["trucks"] + world["airplanes"] if m in cur_loc}
    for act in plan:
        name = act.predicate.name.lower()
        args = [_n(v) for v in act.variables]
        if name == "drive-truck":
            truck, loc_from, loc_to, city = args
            if truck in routes:
                routes[truck].append(loc_to)
        elif name == "fly-airplane":
            plane, loc_from, loc_to = args
            if plane in routes:
                routes[plane].append(loc_to)
    return [(m, path) for m, path in routes.items() if len(path) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_idx", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    env = build_state()
    env.fix_problem_index(args.problem_idx)
    state, _ = env.reset()
    if type(state.goal).__name__ == "Literal":
        state = state.with_goal(LiteralConjunction([state.goal]))

    world = extract_world(state, env.domain)
    pos, city_center = layout(world)

    guidance = NoSearchGuidance()
    guidance.seed(0)
    planner = FlaxPlanner(is_strips_domain=True, base_planner=_create_planner("fd-lama-first"),
                          search_guider=guidance, seed=0,
                          complementary_rules=CMPL_RULES, relaxation_rules=RELX_RULES)

    plan, vis_info = planner(env.domain, state, timeout=args.timeout)
    print(f"Solved problem {args.problem_idx}: plan length {len(plan)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    excluded_after_gnn = {o.name for o in (vis_info.get("gnn_ignored_objects") or set())}
    excluded_after_cmpl = {o.name for o in (vis_info.get("cmpl_ignored_objects") or set())}
    plan_routes = replay_plan_routes(plan, world)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_panel(ax, world, pos, city_center, excluded=set(),
              title=f"(a) Full Problem\n{len(state.objects)} objects, {len(world['cities'])} cities")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_difficultlogistics_a.{ext}"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_panel(ax, world, pos, city_center, excluded=excluded_after_cmpl,
              title=f"(b) After Pruning + Rule Expansion\n"
                    f"{len(state.objects) - len(excluded_after_cmpl)} objects retained")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_difficultlogistics_b.{ext}"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_panel(ax, world, pos, city_center, excluded=excluded_after_cmpl,
              title=f"(c) Final Plan\nlength {len(plan)}",
              plan_routes=plan_routes)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_difficultlogistics_c.{ext}"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Legend figure element (combined into a small separate strip for the paper)
    fig, ax = plt.subplots(figsize=(6, 0.6))
    ax.axis("off")
    handles = [
        mpatches.Patch(facecolor=C_LOC, label="Location"),
        mpatches.Patch(facecolor=C_LOCKED, label="Locked hub"),
        mpatches.Patch(facecolor=C_AIRPORT, label="Airport"),
        mpatches.Patch(facecolor=C_TRUCK, label="Truck"),
        mpatches.Patch(facecolor=C_PKG, label="Package"),
        mpatches.Patch(facecolor=C_PKG_KEY, label="Key-package"),
        mpatches.Patch(facecolor=C_PKG_GOAL, label="Goal package"),
        mpatches.Patch(facecolor=C_EXCLUDE, label="Excluded"),
    ]
    ax.legend(handles=handles, loc="center", ncol=8, fontsize=7.5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"vis_difficultlogistics_legend.{ext}"),
                   dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 3 panels + legend -> {OUT_DIR}/vis_difficultlogistics_{{a,b,c,legend}}.{{pdf,png}}")


if __name__ == "__main__":
    main()
