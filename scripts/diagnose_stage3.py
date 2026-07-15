"""
Stage 3 failure diagnostic: instruments the direct-mode zero-shot LLM object
scorer (guidance/llm_object_guidance.py, the same code path used for Fig. 4b)
to determine WHY scores collapse to near-constant 0.50 on large problems.

Reviewers asked whether: (a) important (plan-critical) objects are missing
from high scores, (b) irrelevant objects are scored too highly, or (c) the
near-constant 0.50 is a real scoring signal vs. an artifact of incomplete
LLM output. This script answers all three by capturing the RAW parsed score
dict (before guidance/llm_object_guidance.py's `.get(name, 0.5)` fallback
fills in missing objects) and cross-referencing it against a known-good
plan's object set (from the Manual Flax/GNN pipeline, which succeeds on
these benchmarks).

Usage:
    python scripts/diagnose_stage3.py --size 12 --difficulty hard --num_problems 5
    python scripts/diagnose_stage3.py --size 15 --difficulty hard --num_problems 5
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pddlgym
from pddlgym.structs import LiteralConjunction
from planning import PlanningTimeout, PlanningFailure, FlaxPlanner
from guidance import GNNSearchGuidance
from guidance.llm_object_guidance import (
    DIRECT_SYSTEM, DIRECT_USER_TEMPLATE, _format_goal, _format_objects,
    _format_facts_alphabetical, _parse_scores,
)
from my_utils.pddl_utils import _create_planner
from openai import OpenAI


def setup_symlink(size, difficulty):
    link = os.path.abspath("pddlgym/pddl/mazenamo_test")
    target = os.path.abspath(f"pddl_files/problems/mazenamo_problems/pddl_{size}x{size}_{difficulty}")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(target, link)


def get_manual_plan_objects(domain, state, timeout=30.0):
    """Run Manual Flax (GNN + manual rules) to get a known-good plan's object set."""
    planner = _create_planner("fd-lama-first")
    guidance = GNNSearchGuidance(
        training_planner=_create_planner("fd-opt-lmcut"),
        num_train_problems=200, num_epochs=301, criterion_name="bce", bce_pos_weight=10,
        load_from_file=True, load_dataset_from_file=True,
        dataset_file_prefix="model/training_data",
        save_model_prefix="model/bce10_model_last_seed0",
        is_strips_domain=True,
    )
    guidance.seed(0)
    guidance.train("Mazenamo", timeout=120)
    flax = FlaxPlanner(is_strips_domain=True, base_planner=planner, search_guider=guidance,
                       seed=0, complementary_rules="config/mazenamo_complementary_rules.json",
                       relaxation_rules="config/mazenamo_relaxation_rules_1.json")
    try:
        plan, _ = flax(domain, state, timeout=timeout)
        return {o.name for act in plan for o in act.variables}
    except (PlanningTimeout, PlanningFailure):
        return None


def raw_direct_score(client, model, state, max_facts=80):
    """Replicate LLMObjectGuidance._score_direct but expose raw diagnostics."""
    facts_str = _format_facts_alphabetical(state, max_facts=max_facts)
    prompt = DIRECT_USER_TEMPLATE.format(
        goal=_format_goal(state), objects=_format_objects(state), facts=facts_str,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": DIRECT_SYSTEM},
                   {"role": "user",   "content": prompt}],
        temperature=0.0,
    )
    raw_text = resp.choices[0].message.content.strip()
    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    try:
        raw_scores = _parse_scores(raw_text, debug=False, fallback_objects=state.objects)
        parse_ok = True
    except Exception as e:
        raw_scores = {}
        parse_ok = False
    return {
        "finish_reason": finish_reason,
        "parse_ok": parse_ok,
        "raw_text_len": len(raw_text),
        "raw_scores": raw_scores,  # only objects the LLM actually mentioned
    }


def diagnose(size, difficulty, num_problems, model="qwen2.5:14b"):
    setup_symlink(size, difficulty)
    env = pddlgym.make("PDDLEnvMazenamoTest-v0")
    num_problems = min(num_problems, len(env.problems))
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    results = []
    for idx in range(num_problems):
        env.fix_problem_index(idx)
        state, _ = env.reset()
        if type(state.goal).__name__ == "Literal":
            state = state.with_goal(LiteralConjunction([state.goal]))

        n_objects = len(state.objects)
        diag = raw_direct_score(client, model, state)
        n_scored = len(diag["raw_scores"])
        n_missing = n_objects - n_scored

        # objects the LLM scored high (>=0.7) vs how many were even mentioned
        high_scored = {k for k, v in diag["raw_scores"].items() if v >= 0.7}

        plan_objects = get_manual_plan_objects(env.domain, state, timeout=30.0)
        recall = None
        if plan_objects:
            plan_objects_present = plan_objects & {o.name for o in state.objects}
            if plan_objects_present:
                # An object counts as "recalled" if it was scored (by the LLM)
                # at all with a non-trivial score, OR is missing from the LLM's
                # response entirely (which silently becomes 0.5 downstream).
                hit = plan_objects_present & set(diag["raw_scores"].keys())
                recall = len(hit) / len(plan_objects_present)
                missing_plan_objs = plan_objects_present - set(diag["raw_scores"].keys())
            else:
                missing_plan_objs = set()
        else:
            missing_plan_objs = None

        row = {
            "problem_idx": idx,
            "n_objects": n_objects,
            "n_scored_by_llm": n_scored,
            "n_silently_defaulted_0.5": n_missing,
            "pct_defaulted": round(100 * n_missing / n_objects, 1),
            "finish_reason": diag["finish_reason"],
            "raw_response_chars": diag["raw_text_len"],
            "n_scored_high(>=0.7)": len(high_scored),
            "plan_object_recall_in_llm_response": recall,
            "plan_objects_silently_defaulted": sorted(missing_plan_objs) if missing_plan_objs else [],
        }
        print(json.dumps(row, indent=2))
        results.append(row)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--difficulty", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=5)
    parser.add_argument("--model", type=str, default="qwen2.5:14b")
    args = parser.parse_args()

    out = diagnose(args.size, args.difficulty, args.num_problems, args.model)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/stage3_diagnostic_{args.size}x{args.size}_{args.difficulty}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")
