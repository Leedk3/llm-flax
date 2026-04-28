"""
Run Pure FD and PLOI baseline experiments across all 8 MazeNamo benchmarks.

Configs:
  pure  — FD with no guidance, no rules (lower bound)
  ploi  — FD + GNN (IncrementalPlanner), no relaxation/complementary rules

Results saved to: results/baseline_<size>x<size>_<difficulty>_n<n>.json

Usage (from /home/leedk/flax/):
    conda run -n flax python scripts/run_baselines.py
    conda run -n flax python scripts/run_baselines.py --configs pure      # only pure FD
    conda run -n flax python scripts/run_baselines.py --configs ploi      # only PLOI
    conda run -n flax python scripts/run_baselines.py --dry_run           # print plan only
"""

import os
import sys
import json
import time
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pddlgym
from pddlgym.structs import LiteralConjunction
from planning import (PlanningTimeout, PlanningFailure,
                      validate_strips_plan, IncrementalPlanner)
from guidance import NoSearchGuidance, GNNSearchGuidance
from my_utils.pddl_utils import _create_planner

# ── Benchmark definitions ─────────────────────────────────────────────────────
BENCHMARKS = [
    dict(size=10, difficulty="easy",   n=50, timeout=10.0),
    dict(size=10, difficulty="medium", n=50, timeout=10.0),
    dict(size=10, difficulty="hard",   n=50, timeout=10.0),
    dict(size=12, difficulty="medium", n=50, timeout=30.0),
    dict(size=12, difficulty="hard",   n=50, timeout=30.0),
    dict(size=12, difficulty="expert", n=30, timeout=30.0),
    dict(size=15, difficulty="medium", n=30, timeout=40.0),
    dict(size=15, difficulty="hard",   n=30, timeout=40.0),
]

MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_symlink(size, difficulty):
    link   = os.path.abspath("pddlgym/pddl/mazenamo_test")
    target = os.path.abspath(
        f"pddl_files/problems/mazenamo_problems/pddl_{size}x{size}_{difficulty}")
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        shutil.rmtree(link)
    os.symlink(target, link)
    print(f"  Symlink: mazenamo_test → pddl_{size}x{size}_{difficulty}")


def _make_guider(config_name, seed):
    if config_name == "pure":
        return NoSearchGuidance()
    if config_name == "ploi":
        return GNNSearchGuidance(
            training_planner=_create_planner("fd-opt-lmcut"),
            num_train_problems=1,
            num_epochs=301,
            criterion_name="bce",
            bce_pos_weight=10,
            load_from_file=True,
            load_dataset_from_file=True,
            dataset_file_prefix=os.path.join(MODEL_DIR, "training_data"),
            save_model_prefix=os.path.join(MODEL_DIR, f"bce10_model_last_seed{seed}"),
            is_strips_domain=True,
        )
    raise ValueError(f"Unknown config: {config_name}")


def run_one_benchmark(bm, config_name, seed=0, dry_run=False):
    """Run config on one benchmark. Returns result dict."""
    size, diff, n, timeout = bm["size"], bm["difficulty"], bm["n"], bm["timeout"]

    _set_symlink(size, diff)

    base_planner = _create_planner("fd-lama-first")
    guider = _make_guider(config_name, seed)
    guider.seed(seed)
    if config_name != "pure":
        guider.train("Mazenamo", timeout=120)

    if config_name == "pure":
        planner = base_planner          # FD called directly (no vis_info)
    else:  # ploi
        planner = IncrementalPlanner(
            is_strips_domain=True,
            base_planner=base_planner,
            search_guider=guider,
            seed=seed)

    env = pddlgym.make("PDDLEnvMazenamoTest-v0")
    num_problems = min(n, len(env.problems))
    if dry_run:
        print(f"  [DRY RUN] would run {num_problems} problems, timeout={timeout}s")
        return None

    success, times, lengths, failures = 0, [], [], []

    for i in range(num_problems):
        env.fix_problem_index(i)
        state, _ = env.reset()
        if type(state.goal).__name__ == "Literal":
            state = state.with_goal(LiteralConjunction([state.goal]))

        t0 = time.time()
        try:
            if config_name == "pure":
                plan = planner(env.domain, state, timeout=timeout)
            else:
                plan, _ = planner(env.domain, state, timeout=timeout)
        except (PlanningTimeout, PlanningFailure) as e:
            elapsed = time.time() - t0
            failures.append(i)
            print(f"    [{i+1}/{num_problems}] FAIL ({elapsed:.1f}s): {e}", flush=True)
            continue

        elapsed = time.time() - t0

        if not validate_strips_plan(
                domain_file=env.domain.domain_fname,
                problem_file=env.problems[i].problem_fname,
                plan=plan):
            failures.append(i)
            print(f"    [{i+1}/{num_problems}] INVALID plan", flush=True)
            continue

        success += 1
        times.append(elapsed)
        lengths.append(len(plan))
        print(f"    [{i+1}/{num_problems}] OK  len={len(plan)}  t={elapsed:.2f}s",
              flush=True)

    sr  = success / num_problems
    avg_t   = sum(times)   / success if success else timeout
    avg_len = sum(lengths) / success if success else 0.0

    print(f"  → SR={sr:.3f}  avg_time={avg_t:.3f}s  avg_len={avg_len:.1f}  "
          f"({success}/{num_problems} solved)")
    return dict(
        config=config_name,
        size=size, difficulty=diff, num_problems=num_problems,
        success_rate=sr,
        avg_time=round(avg_t, 3),
        avg_plan_length=round(avg_len, 1),
        num_solved=success,
        failed_indices=failures,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["pure", "ploi"],
                        choices=["pure", "ploi"])
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Subset e.g. '10x10_easy 12x12_expert'")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    bm_filter = set(args.benchmarks) if args.benchmarks else None

    for bm in BENCHMARKS:
        key = f"{bm['size']}x{bm['size']}_{bm['difficulty']}"
        if bm_filter and key not in bm_filter:
            continue

        for cfg in args.configs:
            out_file = os.path.join(
                RESULT_DIR,
                f"baseline_{bm['size']}x{bm['size']}_{bm['difficulty']}"
                f"_n{bm['n']}_{cfg}.json")

            if os.path.exists(out_file) and not args.dry_run:
                print(f"\n[SKIP] {out_file} already exists")
                continue

            print(f"\n{'='*60}")
            print(f"  Config={cfg}  {bm['size']}x{bm['size']} {bm['difficulty']}"
                  f"  n={bm['n']}  timeout={bm['timeout']}s")
            print(f"{'='*60}")

            result = run_one_benchmark(bm, cfg, seed=args.seed,
                                       dry_run=args.dry_run)

            if result and not args.dry_run:
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_file}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
