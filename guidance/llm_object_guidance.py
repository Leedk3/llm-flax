"""
Stage 3: LLM zero-shot object importance scoring.

Replaces the GNN (which requires domain-specific training data) with an LLM
that scores object importance zero-shot from the PDDL state and goal.

Scoring modes (use_cot):
  "none"    — direct: 1 call, score ALL objects, alphabetical facts (baseline)
  "lite"    — CoT-lite: 1 call, analysis + score ALL objects, goal-biased facts
  "full"    — CoT-full: 2 calls, separate reasoning + scoring, goal-biased facts
  "topk"    — Top-K: 1 call, score only top-K important objects; rest get 0.1
              Much faster output (K items vs N items in JSON).

Key design choices:
  - Single LLM call per state (all scoring done at once), then cached.
  - train() is a no-op — no training data needed.
  - Fallback: uniform 0.5 scores on LLM failure.
"""

import re
import json
from openai import OpenAI
from guidance.base_guidance import BaseSearchGuidance


# ── Direct mode prompts ────────────────────────────────────────────────────────

DIRECT_SYSTEM = """\
You are an expert AI planning assistant. Given a PDDL planning state and goal,
rank all objects by how important they are for achieving the goal.

Respond with a JSON object mapping each object name to a score from 0.0 to 1.0:
  1.0  = definitely needed (e.g. robot, goal position, blocking obstacles)
  0.7  = probably useful (e.g. objects near the path)
  0.4  = possibly needed (e.g. objects that might be relevant)
  0.1  = very unlikely to be needed (e.g. objects far from goal)

Format: {"object_name": score, ...}
No explanation, no markdown, just the JSON object.
"""

DIRECT_USER_TEMPLATE = """\
=== GOAL ===
{goal}

=== OBJECTS ===
{objects}

=== STATE FACTS ===
{facts}

Score each object's importance for achieving the goal (0.0 = irrelevant, 1.0 = essential).
Consider: what is blocking the robot's path? What objects must be moved or interacted with?
Objects that appear in goal literals should score high.

Respond with JSON only: {{"object_name": score, ...}}
"""


# ── Top-K mode prompts ─────────────────────────────────────────────────────────

TOPK_SYSTEM = """\
You are an expert AI planning assistant. Your job is to identify which objects
a robot NEEDS to achieve its goal so that irrelevant objects can be pruned.
"""

TOPK_USER_TEMPLATE = """\
=== GOAL ===
{goal}

=== ALL OBJECTS ({n_objects} total) ===
{objects}

=== KEY STATE FACTS (goal-biased, {n_facts} shown) ===
{facts}

Select the {k} most important objects for achieving the goal.
Include: robot, goal objects, objects blocking the path, objects that must be moved.
Exclude: objects far from the path with no role in the goal.

Respond with JSON only — only the selected objects, others will default to 0.1:
{{"object_name": score, ...}}

Scoring: 1.0 = essential, 0.8 = on path/blocking, 0.6 = possibly needed.
No explanation, no markdown, just JSON with exactly {k} or fewer entries.
"""


# ── CoT full (2-turn) prompts ──────────────────────────────────────────────────

COT_SYSTEM = """\
You are an expert AI planning assistant analyzing a PDDL planning problem.
Your goal is to identify which objects the robot actually needs to interact with
to achieve its goal, so irrelevant objects can be pruned from the search space.
"""

COT_REASONING_TEMPLATE = """\
=== GOAL ===
{goal}

=== ALL OBJECTS ({n_objects} total) ===
{objects}

=== KEY STATE FACTS (goal-biased selection, {n_facts} shown) ===
{facts}

Analyze this planning problem step by step:
1. Where is the robot currently? (look for rAt or similar position predicate)
2. What is the target state described by the goal?
3. What is the likely path or sequence of actions the robot must take?
4. Which specific objects are ON or NEAR the path and must be interacted with?
5. Which objects are clearly irrelevant (far from path, no role in goal)?

Be specific about object names. Think carefully before answering.
"""

COT_SCORING_TEMPLATE = """\
Based on your analysis, assign an importance score (0.0–1.0) to EVERY object.

Scoring guide:
  1.0 = essential (robot, goal objects, objects that MUST be moved or passed)
  0.8 = on the likely path or directly blocking
  0.5 = possibly relevant (near the path, uncertain)
  0.1 = clearly irrelevant (far from path, no role)

You MUST include all {n_objects} objects. Missing objects default to 0.5.
Respond with JSON only — no explanation, no markdown:
{{"object_name": score, ...}}
"""

# ── CoT-lite (1-turn) prompts ──────────────────────────────────────────────────

COT_LITE_SYSTEM = """\
You are an expert AI planning assistant. Given a PDDL planning state and goal,
identify which objects the robot needs to achieve its goal and score their importance.
"""

COT_LITE_USER_TEMPLATE = """\
=== GOAL ===
{goal}

=== ALL OBJECTS ({n_objects} total) ===
{objects}

=== KEY STATE FACTS (goal-biased, {n_facts} shown) ===
{facts}

Step 1 — Brief analysis (2-3 lines only):
Where is the robot? What path does it need to take? Which objects are blocking or essential?

Step 2 — Score every object (0.0–1.0):
  1.0 = essential (robot, goal objects, must-interact)
  0.8 = on path or blocking
  0.5 = possibly relevant
  0.1 = clearly irrelevant

Output format — analysis first, then JSON on the last line:
ANALYSIS: <your 2-3 line reasoning here>
JSON: {{"object_name": score, ...}}

You MUST include all {n_objects} objects in the JSON.
"""


# ── Fact selection helpers ─────────────────────────────────────────────────────

def _format_goal(state) -> str:
    lines = []
    for lit in sorted(state.goal.literals, key=str):
        args = ", ".join(str(v) for v in lit.variables)
        lines.append(f"  {lit.predicate.name}({args})")
    return "\n".join(lines)


def _format_objects(state) -> str:
    lines = []
    for obj in sorted(state.objects, key=str):
        lines.append(f"  {obj.name} (type: {obj.var_type})")
    return "\n".join(lines)


def _format_facts_alphabetical(state, max_facts: int = 80) -> str:
    """Original: alphabetical order, hard truncation."""
    lines = []
    for lit in sorted(state.literals, key=str):
        args = ", ".join(str(v) for v in lit.variables)
        lines.append(f"  {lit.predicate.name}({args})")
    if len(lines) > max_facts:
        lines = lines[:max_facts]
        lines.append("  ... (more facts omitted)")
    return "\n".join(lines)


def _format_facts_goal_biased(state, max_facts: int = 150) -> tuple:
    """Goal-biased fact selection:
      P1 — facts involving goal objects (always included)
      P2 — binary position-like facts (high-frequency object involved)
      P3 — remaining facts (fill to budget)
    Returns (formatted_string, n_facts_shown).
    """
    goal_objects = set()
    for lit in state.goal.literals:
        goal_objects |= set(lit.variables)

    obj_freq: dict = {}
    for lit in state.literals:
        for v in lit.variables:
            obj_freq[v] = obj_freq.get(v, 0) + 1
    n_obj = len(state.objects)
    freq_threshold = max(3, n_obj // 10)

    def lit_to_line(lit) -> str:
        args = ", ".join(str(v) for v in lit.variables)
        return f"  {lit.predicate.name}({args})"

    p1, p2, p3 = [], [], []
    seen = set()
    for lit in sorted(state.literals, key=str):
        key = str(lit)
        if key in seen:
            continue
        seen.add(key)
        line = lit_to_line(lit)
        involves_goal = any(v in goal_objects for v in lit.variables)
        is_positional = any(obj_freq.get(v, 0) >= freq_threshold
                            for v in lit.variables)
        if involves_goal:
            p1.append(line)
        elif is_positional:
            p2.append(line)
        else:
            p3.append(line)

    result = list(p1)
    remaining = max_facts - len(result)
    if remaining > 0:
        result += p2[:remaining]
    remaining = max_facts - len(result)
    if remaining > 0:
        result += p3[:remaining]

    total = len(p1) + len(p2) + len(p3)
    if total > max_facts:
        result.append(f"  ... ({total - len(result)} more facts omitted)")

    return "\n".join(result), min(len(result), max_facts)


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _parse_scores(raw: str, debug: bool, fallback_objects) -> dict:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    scores_raw = json.loads(raw)
    scores = {}
    for name, val in scores_raw.items():
        try:
            scores[name] = float(val)
        except (TypeError, ValueError):
            if debug:
                print(f"  [LLMObjectGuidance] Non-numeric score for '{name}': {val}")

    if not scores:
        return scores

    # If LLM returned ranks/integers outside [0,1], normalize to [0,1]
    max_val = max(scores.values())
    min_val = min(scores.values())
    if max_val > 1.0 or min_val < 0.0:
        span = max_val - min_val if max_val > min_val else 1.0
        scores = {k: (v - min_val) / span for k, v in scores.items()}
        if debug:
            print(f"  [LLMObjectGuidance] Normalized scores "
                  f"(original range [{min_val:.2f}, {max_val:.2f}])")

    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    if debug:
        for name, s in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {s:.2f}  {name}")
    return scores


# ── Main class ─────────────────────────────────────────────────────────────────

class LLMObjectGuidance(BaseSearchGuidance):
    """Zero-shot object importance scoring using an LLM.

    Drop-in replacement for GNNSearchGuidance. No training required.

    Args:
        use_cot: scoring mode.
            "none"  — direct single-turn, all objects, alphabetical facts
            "lite"  — single-turn CoT, all objects, goal-biased facts
            "full"  — two-turn CoT, all objects, goal-biased facts
            "topk"  — single-turn, score only top-K objects; rest get topk_default
        topk:  number of objects to select in "topk" mode (default: 40)
        topk_default: score assigned to unselected objects (default: 0.1)
    """

    def __init__(self, model: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434/v1",
                 debug: bool = False,
                 use_cot: str = "none",
                 topk: int = 40,
                 topk_default: float = 0.1):
        self._client = OpenAI(base_url=ollama_url, api_key="ollama")
        self._model = model
        self._debug = debug
        self._use_cot = use_cot
        self._topk = topk
        self._topk_default = topk_default
        self._last_state = None
        self._last_scores: dict = {}

    def train(self, train_env_name, timeout=120):
        print(f"LLMObjectGuidance [{self._use_cot}]: zero-shot mode, "
              f"skipping training for domain '{train_env_name}'.")

    def seed(self, seed):
        pass

    def score_object(self, obj, state) -> float:
        if state != self._last_state:
            self._last_scores = self._score_all_objects(state)
            self._last_state = state
        return self._last_scores.get(obj.name, 0.5)

    def _score_all_objects(self, state) -> dict:
        if self._use_cot == "full":
            return self._score_cot_full(state)
        if self._use_cot == "lite":
            return self._score_cot_lite(state)
        if self._use_cot == "topk":
            return self._score_topk(state)
        return self._score_direct(state)

    # ── Direct ────────────────────────────────────────────────────────────────

    def _score_direct(self, state) -> dict:
        facts_str = _format_facts_alphabetical(state, max_facts=80)
        prompt = DIRECT_USER_TEMPLATE.format(
            goal=_format_goal(state),
            objects=_format_objects(state),
            facts=facts_str,
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": DIRECT_SYSTEM},
                           {"role": "user",   "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if self._debug:
                print(f"\n[LLMObjectGuidance direct]\n{raw}\n")
            return _parse_scores(raw, self._debug, state.objects)
        except Exception as e:
            if self._debug:
                print(f"  [LLMObjectGuidance direct] Failed: {e}")
            return {obj.name: 0.5 for obj in state.objects}

    # ── Top-K ─────────────────────────────────────────────────────────────────

    def _score_topk(self, state) -> dict:
        """Score only the top-K most important objects; assign topk_default to rest.

        Output JSON is ~K entries instead of ~N, cutting generation time
        from O(N) to O(K) tokens — critical for large problems (N=240+).
        """
        goal_str = _format_goal(state)
        obj_str = _format_objects(state)
        facts_str, n_facts = _format_facts_goal_biased(state)
        n_objects = len(state.objects)
        k = min(self._topk, n_objects)

        prompt = TOPK_USER_TEMPLATE.format(
            goal=goal_str, objects=obj_str, facts=facts_str,
            n_objects=n_objects, n_facts=n_facts, k=k,
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": TOPK_SYSTEM},
                           {"role": "user",   "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if self._debug:
                print(f"\n[LLMObjectGuidance top-k raw]\n{raw}\n")

            topk_scores = _parse_scores(raw, self._debug, state.objects)

            # Fill unselected objects with topk_default
            scores = {obj.name: self._topk_default for obj in state.objects}
            scores.update(topk_scores)

            if self._debug:
                n_high = sum(1 for s in scores.values() if s > self._topk_default)
                print(f"  [Top-K] {n_high} objects scored above default "
                      f"({self._topk_default}), {n_objects - n_high} defaulted.")
            return scores

        except Exception as e:
            if self._debug:
                print(f"  [LLMObjectGuidance top-k] Failed: {e}")
            return {obj.name: 0.5 for obj in state.objects}

    # ── CoT full (2-turn) ──────────────────────────────────────────────────────

    def _score_cot_full(self, state) -> dict:
        goal_str = _format_goal(state)
        obj_str = _format_objects(state)
        facts_str, n_facts = _format_facts_goal_biased(state)
        n_objects = len(state.objects)

        reasoning_prompt = COT_REASONING_TEMPLATE.format(
            goal=goal_str, objects=obj_str, facts=facts_str,
            n_objects=n_objects, n_facts=n_facts,
        )
        try:
            r1 = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": COT_SYSTEM},
                           {"role": "user",   "content": reasoning_prompt}],
                temperature=0.0,
            )
            reasoning = r1.choices[0].message.content.strip()
            if self._debug:
                print(f"\n[LLMObjectGuidance CoT-full reasoning]\n{reasoning}\n")

            scoring_prompt = COT_SCORING_TEMPLATE.format(n_objects=n_objects)
            r2 = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system",    "content": COT_SYSTEM},
                           {"role": "user",      "content": reasoning_prompt},
                           {"role": "assistant", "content": reasoning},
                           {"role": "user",      "content": scoring_prompt}],
                temperature=0.0,
            )
            raw = r2.choices[0].message.content.strip()
            if self._debug:
                print(f"\n[LLMObjectGuidance CoT-full scores]\n{raw}\n")
            return _parse_scores(raw, self._debug, state.objects)
        except Exception as e:
            if self._debug:
                print(f"  [LLMObjectGuidance CoT-full] Failed: {e}")
            return {obj.name: 0.5 for obj in state.objects}

    # ── CoT lite (1-turn) ──────────────────────────────────────────────────────

    def _score_cot_lite(self, state) -> dict:
        goal_str = _format_goal(state)
        obj_str = _format_objects(state)
        facts_str, n_facts = _format_facts_goal_biased(state)
        n_objects = len(state.objects)

        prompt = COT_LITE_USER_TEMPLATE.format(
            goal=goal_str, objects=obj_str, facts=facts_str,
            n_objects=n_objects, n_facts=n_facts,
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": COT_LITE_SYSTEM},
                           {"role": "user",   "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if self._debug:
                print(f"\n[LLMObjectGuidance CoT-lite raw]\n{raw}\n")
            json_match = re.search(r"JSON:\s*(\{.*\})", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1)
            return _parse_scores(raw, self._debug, state.objects)
        except Exception as e:
            if self._debug:
                print(f"  [LLMObjectGuidance CoT-lite] Failed: {e}")
            return {obj.name: 0.5 for obj in state.objects}
