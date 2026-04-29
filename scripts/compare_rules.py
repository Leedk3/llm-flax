"""
Ablation study: compare manual rules vs LLM-generated rules in the Flax pipeline.

Runs the same Flax planner with different rule configurations and reports:
  - Success rate
  - Average planning time
  - Average plan length

Usage:
    python scripts/compare_rules.py
    python scripts/compare_rules.py --size 10 --difficulty easy --num_problems 50
    python scripts/compare_rules.py --size 10 --difficulty medium --num_problems 100
"""

import os
import sys
import json
import time
import argparse
import subprocess
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAIN_NAME   = "MazeNamo"
TRAIN_PLANNER = "fd-opt-lmcut"
TEST_PLANNER  = "fd-lama-first"
DEFAULT_GUIDER = "gnn-bce-10"
NUM_SEEDS     = 1
NUM_TRAIN     = 200
NUM_EPOCHS    = 301

# Rule configurations to compare.
# Each entry may override "guider" and "planner_type"; omitted fields use defaults.
RULE_CONFIGS = {
    "manual": {
        "planner_type": "flax",
        "cmpl_rules": "config/mazenamo_complementary_rules.json",
        "relx_rules": "config/mazenamo_relaxation_rules_1.json",
    },
    "llm_rules": {
        "planner_type": "flax",
        "cmpl_rules": "config/mazenamo_complementary_rules_gemma3-12b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_gemma3-12b.json",
    },
    "llm_rules+recovery": {
        "planner_type": "llmflax",
        "cmpl_rules": "config/mazenamo_complementary_rules_gemma3-12b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_gemma3-12b.json",
    },
    # ── Qwen2.5-14B (original Stage 1 experiments) ────────────────────────
    "llm_rules_qwen2.5-14b": {
        "planner_type": "flax",
        "cmpl_rules": "config/mazenamo_complementary_rules_qwen2.5-14b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_qwen2.5-14b.json",
    },
    # ── Model comparison (Stage 1 only, same GNN) ─────────────────────────
    "llm_rules_llama3.1-8b": {
        "planner_type": "flax",
        "cmpl_rules": "config/mazenamo_complementary_rules_llama3.1-8b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_llama3.1-8b.json",
    },
    "llm_rules_mistral-7b": {
        "planner_type": "flax",
        "cmpl_rules": "config/mazenamo_complementary_rules_mistral-7b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_mistral-7b.json",
    },
    "llm_rules_gemma3-12b": {
        "planner_type": "flax",
        "cmpl_rules": "config/mazenamo_complementary_rules_gemma3-12b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_gemma3-12b.json",
    },
    # Stage 3: Full LLM-Flax — no GNN training, no manual rules
    "full_llm_flax": {
        "planner_type": "llmflax",
        "guider": "llm-zero-shot",
        "cmpl_rules": "config/mazenamo_complementary_rules_qwen2.5-14b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_qwen2.5-14b.json",
    },
    # Stage 3: goal-biased facts + 2-turn CoT
    "full_llm_flax_cot": {
        "planner_type": "llmflax",
        "guider": "llm-cot",
        "cmpl_rules": "config/mazenamo_complementary_rules_qwen2.5-14b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_qwen2.5-14b.json",
    },
    # Stage 3: goal-biased facts + 1-turn CoT-lite (analysis + scores in one call)
    "full_llm_flax_cot_lite": {
        "planner_type": "llmflax",
        "guider": "llm-cot-lite",
        "cmpl_rules": "config/mazenamo_complementary_rules_qwen2.5-14b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_qwen2.5-14b.json",
    },
    # Stage 3: top-K scoring — score only K most important objects, rest get 0.1
    "full_llm_flax_topk": {
        "planner_type": "llmflax",
        "guider": "llm-topk",
        "cmpl_rules": "config/mazenamo_complementary_rules_qwen2.5-14b.json",
        "relx_rules": "config/mazenamo_relaxation_rules_qwen2.5-14b.json",
    },
}


def setup_pddlgym_symlink(size: int, difficulty: str):
    """Point pddlgym/pddl/mazenamo_test to the right problem set."""
    link = os.path.abspath("pddlgym/pddl/mazenamo_test")
    target = os.path.abspath(
        f"pddl_files/problems/mazenamo_problems/pddl_{size}x{size}_{difficulty}"
    )
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        import shutil; shutil.rmtree(link)
    os.symlink(target, link)
    print(f"  Symlink: pddlgym/pddl/mazenamo_test -> {target}")


def parse_results(output: str) -> dict:
    """Extract metrics from main.py stdout."""
    results = {}

    m = re.search(r"total avg planning time:\s*([\d.]+)", output)
    results["avg_time"] = float(m.group(1)) if m else None

    m = re.search(r"total avg success rate:\s*([\d.]+)", output)
    results["success_rate"] = float(m.group(1)) if m else None

    m = re.search(r"total avg plan length:\s*([\d.]+)", output)
    results["avg_plan_length"] = float(m.group(1)) if m else None

    return results


def run_flax(config_name: str, config: dict, size: int, difficulty: str,
             num_problems: int, test_timeout: float) -> dict:
    """Run Flax with a given rule config and return metrics."""
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  Rules : cmpl={config['cmpl_rules']}, relx={config['relx_rules']}")
    print(f"{'='*60}")

    guider = config.get("guider", DEFAULT_GUIDER)
    cmd = [
        "python", "-u", "src/main.py",
        "--domain_name",       DOMAIN_NAME,
        "--train_planner_name", TRAIN_PLANNER,
        "--test_planner_name",  TEST_PLANNER,
        "--guider_name",        guider,
        "--num_seeds",          str(NUM_SEEDS),
        "--num_train_problems", str(NUM_TRAIN),
        "--num_test_problems",  str(num_problems),
        "--planner_type",       config.get("planner_type", "flax"),
        "--train_timeout",      "120",
        "--test_timeout",       str(test_timeout),
        "--num_epochs",         str(NUM_EPOCHS),
        "--cmpl_rules",         config["cmpl_rules"],
        "--relx_rules",         config["relx_rules"],
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

    start = time.time()
    proc = subprocess.run(
        cmd, capture_output=False, text=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    elapsed = time.time() - start

    output = proc.stdout
    print(output)

    metrics = parse_results(output)
    metrics["wall_time_s"] = round(elapsed, 1)
    metrics["config"] = config_name
    return metrics


def print_comparison_table(all_results: list[dict]):
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    header = f"{'Config':<25} {'Success':>10} {'Avg Time':>12} {'Plan Len':>10} {'Wall(s)':>10}"
    print(header)
    print("-" * 70)
    for r in all_results:
        sr  = f"{r['success_rate']:.4f}"  if r['success_rate']  is not None else "N/A"
        at  = f"{r['avg_time']:.4f}"      if r['avg_time']       is not None else "N/A"
        pl  = f"{r['avg_plan_length']:.1f}" if r['avg_plan_length'] is not None else "N/A"
        wt  = f"{r['wall_time_s']:.1f}"
        print(f"{r['config']:<25} {sr:>10} {at:>12} {pl:>10} {wt:>10}")
    print("=" * 70)


def save_results(all_results: list[dict], size: int, difficulty: str,
                 num_problems: int, suffix: str = ""):
    tag = f"_{suffix}" if suffix else ""
    out_path = f"results/ablation_{size}x{size}_{difficulty}_n{num_problems}{tag}.json"
    os.makedirs("results", exist_ok=True)
    # Merge with existing file: new results overwrite old entries for the same config
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        new_configs = {r["config"] for r in all_results}
        kept = [r for r in existing.get("results", []) if r["config"] not in new_configs]
        merged = kept + all_results
        existing["results"] = merged
        meta = existing
    else:
        meta = {
            "size": size, "difficulty": difficulty,
            "num_problems": num_problems, "num_seeds": NUM_SEEDS,
            "results": all_results,
        }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",         type=int,   default=10,
                        help="Grid size (default: 10)")
    parser.add_argument("--difficulty",   type=str,   default="easy",
                        choices=["easy", "medium", "hard", "expert"],
                        help="Problem difficulty (default: easy)")
    parser.add_argument("--num_problems", type=int,   default=50,
                        help="Number of test problems (default: 50)")
    parser.add_argument("--test_timeout", type=float, default=10.0,
                        help="Per-problem timeout in seconds (default: 10)")
    parser.add_argument("--configs",      nargs="+",  default=None,
                        help="Subset of configs to run (default: all)")
    parser.add_argument("--output_suffix", type=str,  default="",
                        help="Optional suffix for result filename to avoid overwriting")
    args = parser.parse_args()

    setup_pddlgym_symlink(args.size, args.difficulty)

    configs_to_run = {
        k: v for k, v in RULE_CONFIGS.items()
        if args.configs is None or k in args.configs
    }

    all_results = []
    for name, config in configs_to_run.items():
        metrics = run_flax(
            name, config,
            args.size, args.difficulty,
            args.num_problems, args.test_timeout
        )
        all_results.append(metrics)

    print_comparison_table(all_results)
    save_results(all_results, args.size, args.difficulty, args.num_problems,
                 suffix=args.output_suffix)
