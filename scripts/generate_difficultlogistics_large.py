"""
Generate a handful of larger DifficultLogistics problems directly (skips the
slow FD-based difficulty classification loop in
src/generate_difficultlogistics_problems.py, which solves every candidate
world with a reference planner before accepting/rejecting it).

Purpose: test whether guided pruning (Manual/LLM-Flax rules) overtakes Pure FD
as object count increases, mirroring the scale-dependent crossover already
documented for MazeNamo (Appendix: Pure FD strong at small scale, weaker at
large scale) -- i.e. whether the same explanation replicates on a second
domain rather than being MazeNamo-specific.

Usage: python scripts/generate_difficultlogistics_large.py
Output: pddl_files/problems/difficultlogistics_problems/pddl_test_large/*.pddl
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from generate_difficultlogistics_problems import (
    generate_random_logistics_world, world_to_pddl,
)

OUT_DIR = "pddl_files/problems/difficultlogistics_problems/pddl_test_large"
N_PROBLEMS = 15

# Roughly 1.6x the default world (5 cities, ~10 locs/city, ~8 pkgs/city,
# ~100 objects total) -> ~160 objects total.
CONFIG = dict(
    num_cities=8,
    min_locs_per_city=8, max_locs_per_city=12,
    min_pkgs_per_city=6, max_pkgs_per_city=10,
    trucks_per_city=1,
    num_airplanes=3,
    cross_city_goal_prob=0.3,
    air_extra_prob=0.5,
    min_goal_pkgs=3, max_goal_pkgs=5,
    max_stack_height=3,
    locked_per_city=2,
    num_keys=3,
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = 0
    seed = 1000
    while written < N_PROBLEMS:
        try:
            world = generate_random_logistics_world(seed=seed, **CONFIG)
        except ValueError as e:
            print(f"  [seed {seed}] generation failed: {e}, retrying...")
            seed += 1
            continue
        total_objects = (
            sum(len(v) for v in world["city_locs"].values())
            + len(world["packages"]) + len(world["trucks"])
            + len(world["airplanes"]) + len(world["cities"])
        )
        pddl_str = world_to_pddl(world, problem_name="difficultlogistics_problem",
                                 domain_name="difficultlogistics")
        out_path = os.path.join(OUT_DIR, f"difficultlogistics_problem_{written}.pddl")
        with open(out_path, "w") as f:
            f.write(pddl_str)
        print(f"  [{written}] seed={seed} total_objects~={total_objects} -> {out_path}")
        written += 1
        seed += 1

    print(f"\nWrote {written} problems to {OUT_DIR}")


if __name__ == "__main__":
    main()
