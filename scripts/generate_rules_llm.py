"""
LLM-based automatic rule generation for Flax planner.

Given a PDDL domain file, uses a local LLM (via Ollama) to generate:
  1. relaxation_rules: which objects can be safely removed to simplify the problem
  2. complementary_rules: which object pairs must always appear together

Usage:
    python scripts/generate_rules_llm.py --domain pddl_files/domains/mazenamo.pddl
    python scripts/generate_rules_llm.py --domain pddl_files/domains/mazenamo.pddl --debug
    python scripts/generate_rules_llm.py --domain pddl_files/domains/mazenamo.pddl --model llama3.1:8b
"""

import os
import re
import json
import argparse
from openai import OpenAI

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# PDDL predicate parsing
# ---------------------------------------------------------------------------

def parse_predicates(domain_text: str) -> dict[str, int]:
    """Extract predicate names and their arities from a PDDL domain.

    Returns: {"predicatename": arity, ...}  (names lowercased)
    """
    # Grab the (:predicates ...) block
    match = re.search(r"\(:predicates(.*?)\n\s*\)", domain_text, re.DOTALL)
    if not match:
        return {}

    block = match.group(1)
    predicates = {}

    # Each predicate looks like: (name ?a - type ?b - type ...)
    for pred_match in re.finditer(r"\(\s*(\w+)(.*?)\)", block, re.DOTALL):
        name = pred_match.group(1).lower()
        params = pred_match.group(2).strip()
        # Count parameters: tokens starting with ?
        arity = len(re.findall(r"\?\w+", params))
        predicates[name] = arity

    return predicates

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in AI planning and PDDL (Planning Domain Definition Language).
Your task is to analyze a PDDL domain and generate rule configurations for the
Flax neuro-symbolic planner.

CRITICAL FORMAT RULES:
- All index values must be INTEGERS (0, 1, 2, ...), NEVER strings or predicate names.
- An index refers to the position of an argument in a predicate's parameter list (0-based).
- Output valid JSON only. No explanation, no markdown fences, no extra text.
"""

RELAXATION_RULES_PROMPT = """\
Analyze this PDDL domain and generate relaxation rules in the exact JSON format below.

=== WHAT ARE RELAXATION RULES? ===
A relaxation rule removes a type of object from the problem to make it simpler.
For example, if a light box is blocking the path but is not a goal object,
we can pretend it was already moved away (remove it and mark its position as empty).

=== EXACT JSON FORMAT ===
{{
  "rule0": {{
    "pre_compute": {{
      "PREDICATE_NAME": [FROM_INDEX, TO_INDEX]
    }},
    "precond": {{
      "PREDICATE_NAME": [ARG_INDEX]
    }},
    "delete_objects": [ARG_INDEX],
    "delete_effects": {{
      "PREDICATE_NAME": [ARG_INDEX, ...]
    }},
    "add_effects": {{
      "PREDICATE_NAME": [ARG_INDEX]
    }}
  }}
}}

=== INDEX MEANING ===
Each predicate has numbered arguments starting at 0.
For example, oAt(?o - obstacle, ?p - pos):
  - argument 0 = ?o (the obstacle)
  - argument 1 = ?p (the position)

So "oat": [0, 1] means the predicate oAt with arg0=obstacle, arg1=position.
And "oat": [0] means only argument 0 (the obstacle) of oAt.
And "posempty": [1] means argument 1 of posEmpty (the position freed up).

NEVER use strings like "obstacle" or "pos" as index values. Only integers.

=== FIELD MEANINGS ===
- pre_compute: binary predicate used to look up a related object (e.g. oAt maps obstacle->pos)
- precond: unary predicate that identifies which objects to remove (e.g. isLight marks removable objects)
- delete_objects: index of the object to remove from the problem
- delete_effects: predicates (and their arg indices) to remove from the state
- add_effects: predicates (and their arg indices) to add after removal (e.g. mark position as empty)

=== PDDL DOMAIN ===
{domain}

=== CORRECT EXAMPLE ===
For a domain where isLight(?o) marks boxes that can be removed,
oAt(?o - obstacle, ?p - pos) gives their position,
and posEmpty(?p) marks free cells:

{{
  "rule0": {{
    "pre_compute": {{"oat": [0, 1]}},
    "precond": {{"islight": [0]}},
    "delete_objects": [0],
    "delete_effects": {{"islight": [0], "ismoveable": [0], "oat": [0, 1]}},
    "add_effects": {{"posempty": [1]}}
  }}
}}

Read: "For any obstacle[0] that satisfies isLight[0], pre-compute its position
via oAt[0,1], then remove obstacle[0] from the problem, delete its isLight/isMoveable/oAt
literals, and add posEmpty for its former position[1]."

=== WRONG EXAMPLE (DO NOT DO THIS) ===
{{
  "rule0": {{
    "pre_compute": {{"oat": ["obstacle", "pos"]}},   <- WRONG: use integers [0, 1]
    "precond": {{"islight": ["obstacle"]}},           <- WRONG: use integers [0]
    "delete_objects": ["obstacle"],                   <- WRONG: use integers [0]
    ...
  }}
}}

Now generate relaxation rules for the PDDL domain above.
Use lowercase predicate names. Output valid JSON only.
"""

COMPLEMENTARY_RULES_PROMPT = """\
Analyze this PDDL domain and generate complementary rules in the exact JSON format below.

=== WHAT ARE COMPLEMENTARY RULES? ===
Complementary rules ensure that when an object is included in a simplified problem,
its tightly-related partner objects are also included.
For example: if an obstacle is included, its grid position must also be included.

=== EXACT JSON FORMAT ===
{{
  "predicate_name": {{
    "cond": [[ARG_INDEX], [ARG_INDEX]],
    "cmpl": [[ARG_INDEX], [ARG_INDEX]]
  }}
}}

=== INDEX MEANING ===
Each predicate has numbered arguments starting at 0.
For example, oAt(?o - obstacle, ?p - pos):
  - argument 0 = ?o (the obstacle)
  - argument 1 = ?p (the position)

"cond": [[0], [1]] means:
  - condition 0: if argument 0 (obstacle) is in current objects
  - condition 1: if argument 1 (position) is in current objects

"cmpl": [[1], [0]] means:
  - complement for condition 0: also include argument 1 (position)
  - complement for condition 1: also include argument 0 (obstacle)

So cond[i] and cmpl[i] are paired: "if cond[i] objects are included, also include cmpl[i] objects."

NEVER use strings like "obstacle" or "pos" as index values. Only integers.

=== PDDL DOMAIN ===
{domain}

=== CORRECT EXAMPLE ===
For oAt(?o - obstacle, ?p - pos) — obstacle and position must appear together:

{{
  "oat": {{
    "cond": [[0], [1]],
    "cmpl": [[1], [0]]
  }}
}}

Read: "If obstacle(arg 0) is included, also include its position(arg 1).
       If position(arg 1) is included, also include its obstacle(arg 0)."

=== WRONG EXAMPLE (DO NOT DO THIS) ===
{{
  "oat": {{
    "cond": [["oat", "obstacle", "pos"], ["on_ground", "obstacle"]],  <- WRONG: use [[0],[1]]
    "cmpl": [["oat", "obstacle", "pos"], ["upon", "obstacle", "x"]]   <- WRONG: use [[1],[0]]
  }}
}}

=== SELECTION CRITERIA ===
Only include predicates where both arguments are domain objects (not just properties).
Focus on binary predicates that represent spatial or structural relationships
(e.g., "object is at position", "object is on top of another object").
Skip unary predicates (isLight, isHeavy, etc.) — those don't need complementary rules.

Now generate complementary rules for the PDDL domain above.
Use lowercase predicate names. Output valid JSON only.
"""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_indices(indices, field_desc: str) -> str | None:
    """Return an error string if indices is not a list of ints, else None."""
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        return f"{field_desc} must be a list of integers, got: {indices}"
    return None


def validate_relaxation_rules(rules: dict,
                               known_predicates: dict[str, int] | None = None) -> list[str]:
    """Returns a list of error messages, empty if valid."""
    errors = []
    required_keys = {"pre_compute", "precond", "delete_objects", "delete_effects", "add_effects"}

    # --- duplicate detection ---
    seen_contents = []
    for rule_name, rule in rules.items():
        canonical = json.dumps(rule, sort_keys=True)
        if canonical in seen_contents:
            errors.append(f"[{rule_name}] is a duplicate of an earlier rule")
        else:
            seen_contents.append(canonical)

    for rule_name, rule in rules.items():
        prefix = f"[{rule_name}]"
        missing = required_keys - set(rule.keys())
        if missing:
            errors.append(f"{prefix} Missing keys: {missing}")
            continue

        all_pred_sections = [
            ("pre_compute", rule["pre_compute"]),
            ("precond",     rule["precond"]),
            ("delete_effects", rule["delete_effects"]),
            ("add_effects", rule["add_effects"]),
        ]

        for section, mapping in all_pred_sections:
            for pred, indices in mapping.items():
                # index format check
                err = _check_indices(indices, f"{prefix} {section}['{pred}']")
                if err:
                    errors.append(err)
                # predicate name check
                if known_predicates is not None and pred not in known_predicates:
                    errors.append(
                        f"{prefix} {section}: '{pred}' is not a predicate in the domain. "
                        f"Known predicates: {sorted(known_predicates)}"
                    )

        err = _check_indices(rule["delete_objects"],
                             f"{prefix} delete_objects")
        if err:
            errors.append(err)

    return errors


def validate_complementary_rules(rules: dict,
                                  known_predicates: dict[str, int] | None = None) -> list[str]:
    """Returns a list of error messages, empty if valid."""
    errors = []

    for pred_name, rule in rules.items():
        prefix = f"[{pred_name}]"

        # predicate name check
        if known_predicates is not None and pred_name not in known_predicates:
            errors.append(
                f"{prefix} '{pred_name}' is not a predicate in the domain. "
                f"Known predicates: {sorted(known_predicates)}"
            )

        if "cond" not in rule or "cmpl" not in rule:
            errors.append(f"{prefix} Must have 'cond' and 'cmpl' keys")
            continue
        if len(rule["cond"]) != len(rule["cmpl"]):
            errors.append(f"{prefix} 'cond' and 'cmpl' must have the same length")
            continue

        for i, (cond_item, cmpl_item) in enumerate(zip(rule["cond"], rule["cmpl"])):
            err = _check_indices(cond_item, f"{prefix} cond[{i}]")
            if err:
                errors.append(err)
            err = _check_indices(cmpl_item, f"{prefix} cmpl[{i}]")
            if err:
                errors.append(err)

    return errors


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def dedup_relaxation_rules(rules: dict) -> dict:
    """Remove duplicate rules (same content, different key)."""
    seen, deduped = set(), {}
    for name, rule in rules.items():
        canonical = json.dumps(rule, sort_keys=True)
        if canonical not in seen:
            seen.add(canonical)
            deduped[f"rule{len(deduped)}"] = rule
    removed = len(rules) - len(deduped)
    if removed:
        print(f"  Removed {removed} duplicate rule(s).")
    return deduped


def filter_unknown_predicates(rules: dict, known_predicates: dict[str, int],
                               rule_type: str) -> dict:
    """Drop any rule/entry that references an unknown predicate."""
    if rule_type == "relaxation":
        valid = {}
        for name, rule in rules.items():
            sections = ["pre_compute", "precond", "delete_effects", "add_effects"]
            bad = [p for s in sections for p in rule.get(s, {}) if p not in known_predicates]
            if bad:
                print(f"  Dropped [{name}]: unknown predicate(s) {bad}")
            else:
                valid[name] = rule
        # Re-number after drops
        return {f"rule{i}": r for i, r in enumerate(valid.values())}

    elif rule_type == "complementary":
        valid = {}
        for pred_name, rule in rules.items():
            if pred_name not in known_predicates:
                print(f"  Dropped [{pred_name}]: not in domain predicates")
            else:
                valid[pred_name] = rule
        return valid

    return rules


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model: str, messages: list, debug: bool = False) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    if debug:
        print("\n[LLM raw response]\n", text, "\n")
    return text


def parse_json_response(response: str) -> dict:
    cleaned = re.sub(r"^```[a-z]*\n?", "", response.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    return json.loads(cleaned)


def generate_with_retry(client: OpenAI, model: str, prompt: str,
                        validate_fn, rule_type: str,
                        known_predicates: dict[str, int] | None = None,
                        debug: bool = False) -> dict:
    """Call LLM and retry with error feedback if validation fails."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Attempt {attempt}/{MAX_RETRIES}...")
        raw = call_llm(client, model, messages, debug)

        try:
            rules = parse_json_response(raw)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {e}"
            print(f"  Parse error: {error_msg}")
        else:
            errors = validate_fn(rules, known_predicates)
            if not errors:
                print(f"  Validation passed.")
                return rules
            error_msg = "Format errors:\n" + "\n".join(f"  - {e}" for e in errors)
            print(f"  Validation failed:\n{error_msg}")

        if attempt < MAX_RETRIES:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your {rule_type} output had the following errors:\n{error_msg}\n\n"
                    "Remember: ALL index values must be plain integers (0, 1, 2, ...). "
                    "Never use strings as index values. "
                    "Please fix and return valid JSON only."
                ),
            })

    raise ValueError(
        f"Failed to generate valid {rule_type} after {MAX_RETRIES} attempts."
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def generate_rules(domain_path: str, model: str, debug: bool = False):
    domain_text = open(domain_path).read()
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    known_predicates = parse_predicates(domain_text)
    print(f"Model : {model}")
    print(f"Domain: {domain_path}")
    print(f"Predicates found: {sorted(known_predicates)}\n")

    print("Generating relaxation rules...")
    relaxation_rules = generate_with_retry(
        client, model,
        RELAXATION_RULES_PROMPT.format(domain=domain_text),
        validate_relaxation_rules,
        "relaxation rules",
        known_predicates,
        debug,
    )
    relaxation_rules = dedup_relaxation_rules(relaxation_rules)
    relaxation_rules = filter_unknown_predicates(
        relaxation_rules, known_predicates, "relaxation"
    )

    print("\nGenerating complementary rules...")
    complementary_rules = generate_with_retry(
        client, model,
        COMPLEMENTARY_RULES_PROMPT.format(domain=domain_text),
        validate_complementary_rules,
        "complementary rules",
        known_predicates,
        debug,
    )
    complementary_rules = filter_unknown_predicates(
        complementary_rules, known_predicates, "complementary"
    )

    return relaxation_rules, complementary_rules


def save_rules(relaxation_rules: dict, complementary_rules: dict,
               domain_path: str, model: str):
    domain_name = os.path.splitext(os.path.basename(domain_path))[0]
    model_tag = model.replace(":", "-").replace("/", "-")
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(domain_path), "../../config")
    )
    os.makedirs(config_dir, exist_ok=True)

    relx_path = os.path.join(config_dir, f"{domain_name}_relaxation_rules_{model_tag}.json")
    cmpl_path = os.path.join(config_dir, f"{domain_name}_complementary_rules_{model_tag}.json")

    with open(relx_path, "w") as f:
        json.dump(relaxation_rules, f, indent=4)
    with open(cmpl_path, "w") as f:
        json.dump(complementary_rules, f, indent=4)

    print(f"\nSaved relaxation rules    -> {relx_path}")
    print(f"Saved complementary rules -> {cmpl_path}")
    return relx_path, cmpl_path


def compare_with_manual(generated: dict, manual_path: str, rule_type: str):
    if not os.path.exists(manual_path):
        print(f"  [skip] manual file not found: {manual_path}")
        return
    with open(manual_path) as f:
        manual = json.load(f)
    print(f"\n--- {rule_type} ---")
    print(f"  Manual    : {json.dumps(manual)}")
    print(f"  Generated : {json.dumps(generated)}")
    print(f"  Match     : {'YES (exact)' if generated == manual else 'NO (differs)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Path to PDDL domain file")
    parser.add_argument("--model", default="qwen2.5:14b",
                        help="Ollama model name (default: qwen2.5:14b)")
    parser.add_argument("--debug", action="store_true", help="Print raw LLM responses")
    args = parser.parse_args()

    relaxation_rules, complementary_rules = generate_rules(
        args.domain, args.model, args.debug
    )

    print("\n[Generated relaxation rules]")
    print(json.dumps(relaxation_rules, indent=2))
    print("\n[Generated complementary rules]")
    print(json.dumps(complementary_rules, indent=2))

    save_rules(relaxation_rules, complementary_rules, args.domain, args.model)

    domain_name = os.path.splitext(os.path.basename(args.domain))[0]
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(args.domain), "../../config")
    )
    print("\n=== Comparison with manual rules ===")
    compare_with_manual(
        relaxation_rules,
        os.path.join(config_dir, f"{domain_name}_relaxation_rules_1.json"),
        "relaxation",
    )
    compare_with_manual(
        complementary_rules,
        os.path.join(config_dir, f"{domain_name}_complementary_rules.json"),
        "complementary",
    )
