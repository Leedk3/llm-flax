"""
LLM-guided object recovery for failed planning attempts.

When Step 1 (GNN pruning) fails to find a plan, instead of blindly
lowering the threshold (gamma decay), we ask an LLM which specific
objects are likely missing from the current object set.

This replaces the heuristic gamma=0.9 decay with semantic reasoning
about why the plan might have failed.
"""

import re
import json
import os
from openai import OpenAI


SYSTEM_PROMPT = """\
You are an expert AI planner assistant. Your task is to analyze a PDDL \
planning problem and identify which objects are missing from the current \
simplified problem that are needed to find a valid plan.

Always respond with a JSON object in this exact format:
{"objects": ["name1", "name2", ...]}

If no additional objects are needed, return: {"objects": []}
No explanation, no markdown, just the JSON object.
"""

RECOVERY_PROMPT_TEMPLATE = """\
A symbolic planner failed to find a plan using only the objects listed below.
Analyze the problem and identify which EXCLUDED objects should be added to \
make the problem solvable.

=== GOAL ===
{goal}

=== CURRENT OBJECTS (already included) ===
{current_objects}

=== EXCLUDED OBJECTS (candidates to add) ===
{excluded_objects}

=== RELEVANT STATE FACTS ===
{state_facts}

=== INSTRUCTIONS ===
Look at the goal and the current objects. Think about what path the robot \
needs to take and what obstacles might be blocking it.
Identify which excluded objects are likely blocking the path or are needed \
to achieve the goal.
Return only the object names that MUST be added. Be selective — adding too \
many objects slows down planning.

Respond with JSON only: {{"objects": ["name1", "name2", ...]}}
"""


def _format_goal(state) -> str:
    lines = []
    for lit in sorted(state.goal.literals, key=str):
        args = ", ".join(str(v) for v in lit.variables)
        lines.append(f"  {lit.predicate.name}({args})")
    return "\n".join(lines)


def _format_objects(objects) -> str:
    lines = []
    for obj in sorted(objects, key=str):
        lines.append(f"  {obj.name} (type: {obj.var_type})")
    return "\n".join(lines) if lines else "  (none)"


def _format_state_facts(state, excluded_objects, max_facts: int = 60) -> str:
    """Show literals that involve excluded objects (most relevant for recovery)."""
    excluded_names = {obj.name for obj in excluded_objects}
    relevant = []
    for lit in sorted(state.literals, key=str):
        var_names = {v.name for v in lit.variables}
        if var_names & excluded_names:
            args = ", ".join(str(v) for v in lit.variables)
            relevant.append(f"  {lit.predicate.name}({args})")

    if not relevant:
        # Fall back to all literals if nothing involves excluded objects
        for lit in sorted(state.literals, key=str):
            args = ", ".join(str(v) for v in lit.variables)
            relevant.append(f"  {lit.predicate.name}({args})")

    if len(relevant) > max_facts:
        relevant = relevant[:max_facts]
        relevant.append(f"  ... ({len(relevant)} more facts omitted)")
    return "\n".join(relevant) if relevant else "  (no relevant facts)"


class LLMRecoveryGuidance:
    """Suggests which objects to add after a failed planning attempt.

    Drop-in replacement for the gamma-decay object selection in Step 1
    of the Flax pipeline.
    """

    def __init__(self, model: str = "qwen2.5:14b",
                 ollama_url: str = "http://localhost:11434/v1",
                 max_objects_per_call: int = 5,
                 debug: bool = False):
        self._client = OpenAI(base_url=ollama_url, api_key="ollama")
        self._model = model
        self._max_objects = max_objects_per_call
        self._debug = debug

    def suggest_objects(self, state, cur_objects: set,
                        all_objects: set) -> set:
        """Return a set of objects to add from (all_objects - cur_objects).

        Args:
            state:       Current PDDL state (with .literals, .goal, .objects)
            cur_objects: Objects already included in the simplified problem
            all_objects: Full object set from the original problem

        Returns:
            Set of TypedEntity objects to add (subset of all_objects - cur_objects)
        """
        excluded = all_objects - cur_objects
        if not excluded:
            return set()

        obj_name_to_entity = {obj.name: obj for obj in excluded}

        prompt = RECOVERY_PROMPT_TEMPLATE.format(
            goal=_format_goal(state),
            current_objects=_format_objects(cur_objects),
            excluded_objects=_format_objects(excluded),
            state_facts=_format_state_facts(state, excluded),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            if self._debug:
                print(f"\n[LLMRecovery raw]\n{raw}\n")

            # Strip markdown fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

            parsed = json.loads(raw)
            suggested_names = parsed.get("objects", [])

            # Limit and validate
            suggested_names = suggested_names[:self._max_objects]
            result = set()
            for name in suggested_names:
                if name in obj_name_to_entity:
                    result.add(obj_name_to_entity[name])
                else:
                    if self._debug:
                        print(f"  [LLMRecovery] Unknown object '{name}', skipping")
            return result

        except Exception as e:
            if self._debug:
                print(f"  [LLMRecovery] Failed: {e}, falling back to empty set")
            return set()
