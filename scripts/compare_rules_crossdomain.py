"""
Cross-domain generalization ablation: Manual vs. LLM-generated (Gemma3-12B)
rules vs. Pure FD, on domains beyond MazeNamo (DifficultLogistics, SokomindPlus).

Unlike scripts/compare_rules.py (which uses a MazeNamo-specific GNN trained on
200 problems), this script uses guider_name="no-guidance" (uniform random
scoring) for the "flax" planner_type runs. Training a domain-specific GNN
would require solving 200 problems with optimal FD first, which is not
tractable in-session for these domains (e.g. SokomindPlus 20x20 problems take
60-300s each with FD, per scripts/run_sokomindplus.sh's own comment). Using a
fixed no-guidance scorer isolates *rule quality* -- exactly what Stage 1
claims to improve -- consistent with this paper's own Appendix finding that
"PLOI (GNN without rules) is frequently the weakest configuration...the rules
are the critical component, not the scoring model."

Usage:
    python scripts/compare_rules_crossdomain.py --domain difficultlogistics --num_problems 5 --test_timeout 30
    python scripts/compare_rules_crossdomain.py --domain sokomindplus --num_problems 5 --test_timeout 40
"""

import os
import sys
import json
import time
import argparse
import subprocess
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_PLANNER = "fd-opt-lmcut"
TEST_PLANNER  = "fd-lama-first"
GUIDER        = "no-guidance"

DOMAINS = {
    "difficultlogistics": {
        "domain_name": "DifficultLogistics",
        "domain_pddl": "pddl_files/domains/difficultlogistics.pddl",
        "pddl_test_dir": "pddl_files/problems/difficultlogistics_problems/pddl_test",
        "link_name": "difficultlogistics_test",
        "manual_cmpl": "config/difficultlogistics_complementary_rules.json",
        "manual_relx": "config/difficultlogistics_relaxation_rules_1.json",
        "llm_cmpl": "config/difficultlogistics_complementary_rules_gemma3-12b.json",
        "llm_relx": "config/difficultlogistics_relaxation_rules_gemma3-12b.json",
    },
    "difficultlogistics_large": {
        "domain_name": "DifficultLogistics",
        "domain_pddl": "pddl_files/domains/difficultlogistics.pddl",
        "pddl_test_dir": "pddl_files/problems/difficultlogistics_problems/pddl_test_large",
        "link_name": "difficultlogistics_test",
        "manual_cmpl": "config/difficultlogistics_complementary_rules.json",
        "manual_relx": "config/difficultlogistics_relaxation_rules_1.json",
        "llm_cmpl": "config/difficultlogistics_complementary_rules_gemma3-12b.json",
        "llm_relx": "config/difficultlogistics_relaxation_rules_gemma3-12b.json",
    },
    "sokomindplus": {
        "domain_name": "SokomindPlus",
        "domain_pddl": "pddl_files/domains/sokomindplus.pddl",
        "pddl_test_dir": "pddl_files/problems/sokomindplus_problems/pddl_test_15x15",
        "link_name": "sokomindplus_test",
        "manual_cmpl": "config/sokomindplus_complementary_rules.json",
        "manual_relx": "config/sokomindplus_relaxation_rules_1.json",
        "llm_cmpl": "config/sokomindplus_complementary_rules_gemma3-12b.json",
        "llm_relx": "config/sokomindplus_relaxation_rules_gemma3-12b.json",
    },
}


def setup_pddlgym_symlink(link_name: str, target_dir: str):
    link = os.path.abspath(f"pddlgym/pddl/{link_name}")
    target = os.path.abspath(target_dir)
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        import shutil; shutil.rmtree(link)
    os.symlink(target, link)
    print(f"  Symlink: pddlgym/pddl/{link_name} -> {target}")


def setup_domain_file_symlink(domain_name: str, domain_pddl: str):
    """pddlgym expects the domain file at pddlgym/pddl/<name.lower()>.pddl."""
    link = os.path.abspath(f"pddlgym/pddl/{domain_name.lower()}.pddl")
    target = os.path.abspath(domain_pddl)
    if os.path.islink(link) or os.path.exists(link):
        os.unlink(link)
    os.symlink(target, link)
    print(f"  Symlink: pddlgym/pddl/{domain_name.lower()}.pddl -> {target}")


def parse_results(output: str) -> dict:
    results = {}
    m = re.search(r"total avg planning time:\s*([\d.]+)", output)
    results["avg_time"] = float(m.group(1)) if m else None
    m = re.search(r"total avg success rate:\s*([\d.]+)", output)
    results["success_rate"] = float(m.group(1)) if m else None
    m = re.search(r"total avg plan length:\s*([\d.]+)", output)
    results["avg_plan_length"] = float(m.group(1)) if m else None
    return results


def run_config(config_name: str, domain_name: str, planner_type: str,
               cmpl_rules: str, relx_rules: str,
               num_problems: int, test_timeout: float,
               guider: str = GUIDER) -> dict:
    print(f"\n{'='*60}")
    print(f"  Domain: {domain_name}  Config: {config_name}  planner_type={planner_type}  guider={guider}")
    if cmpl_rules:
        print(f"  Rules : cmpl={cmpl_rules}, relx={relx_rules}")
    print(f"{'='*60}")

    cmd = [
        "python", "-u", "src/main.py",
        "--domain_name",       domain_name,
        "--test_planner_name", TEST_PLANNER,
        "--guider_name",       guider,
        "--num_seeds",         "1",
        "--num_test_problems", str(num_problems),
        "--planner_type",      planner_type,
        "--test_timeout",      str(test_timeout),
    ]
    if planner_type != "pure":
        cmd += ["--train_planner_name", TRAIN_PLANNER,
                "--cmpl_rules", cmpl_rules,
                "--relx_rules", relx_rules]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

    start = time.time()
    proc = subprocess.run(cmd, text=True, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    print(proc.stdout)
    metrics = parse_results(proc.stdout)
    metrics["wall_time_s"] = round(elapsed, 1)
    metrics["config"] = config_name
    return metrics


def save_results(all_results: list, domain_key: str, num_problems: int):
    out_path = f"results/crossdomain_{domain_key}_n{num_problems}.json"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        new_configs = {r["config"] for r in all_results}
        kept = [r for r in existing.get("results", []) if r["config"] not in new_configs]
        merged = kept + all_results
        existing["results"] = merged
        meta = existing
    else:
        meta = {"domain": domain_key, "num_problems": num_problems, "results": all_results}
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(DOMAINS.keys()))
    parser.add_argument("--num_problems", type=int, default=30)
    parser.add_argument("--test_timeout", type=float, default=30.0)
    parser.add_argument("--configs", nargs="+", default=["pure", "manual", "llm_rules"])
    parser.add_argument("--guider", type=str, default="no-guidance",
                        choices=["no-guidance", "llm-zero-shot"],
                        help="Object scorer for 'flax' planner_type configs "
                             "(pure ignores this). llm-zero-shot uses "
                             "LLMObjectGuidance (Stage 3) instead of a fixed "
                             "uniform-random scorer.")
    args = parser.parse_args()

    d = DOMAINS[args.domain]
    setup_domain_file_symlink(d["domain_name"], d["domain_pddl"])
    setup_pddlgym_symlink(d["link_name"], d["pddl_test_dir"])

    config_suffix = "" if args.guider == "no-guidance" else f"_{args.guider}"
    all_results = []
    for cfg in args.configs:
        if cfg == "pure":
            metrics = run_config("pure", d["domain_name"], "pure", None, None,
                                 args.num_problems, args.test_timeout)
        elif cfg == "manual":
            metrics = run_config(f"manual{config_suffix}", d["domain_name"], "flax",
                                 d["manual_cmpl"], d["manual_relx"],
                                 args.num_problems, args.test_timeout,
                                 guider=args.guider)
        elif cfg == "llm_rules":
            metrics = run_config(f"llm_rules_gemma3-12b{config_suffix}", d["domain_name"], "flax",
                                 d["llm_cmpl"], d["llm_relx"],
                                 args.num_problems, args.test_timeout,
                                 guider=args.guider)
        else:
            raise ValueError(f"Unknown config '{cfg}'")
        all_results.append(metrics)

    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON RESULTS —", d["domain_name"])
    print("=" * 70)
    for r in all_results:
        sr = f"{r['success_rate']:.4f}" if r['success_rate'] is not None else "N/A"
        at = f"{r['avg_time']:.4f}" if r['avg_time'] is not None else "N/A"
        print(f"{r['config']:<25} SR={sr:>8} Time={at:>10} Wall(s)={r['wall_time_s']:.1f}")

    save_results(all_results, args.domain, args.num_problems)
