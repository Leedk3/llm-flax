"""An incremental planner that samples more and more objects until
it finds a plan.
"""

import time
import tempfile
import numpy as np
import json
from pddlgym.structs import State, Literal
from pddlgym.spaces import LiteralSpace
from pddlgym.parser import PDDLProblemParser
from planning import Planner, PlanningFailure, PlanningTimeout, validate_strips_plan

def apply_complementary_rules(state, cur_objects, complementary_rules):
    new_cur_objects = cur_objects.copy()

    # First apply rules with single-object conditions to recover important objects with special roles
    for lit in state.literals:
        p_name = lit.predicate.name
        if p_name in complementary_rules:
            p_cmpl_rule = complementary_rules[p_name]
            if len(p_cmpl_rule["cond"]) == 1:
                for v_idx_list_cond, v_idx_list_cmpl in zip(p_cmpl_rule["cond"], p_cmpl_rule["cmpl"]):
                    if cur_objects.isdisjoint(set([lit.variables[v_idx] for v_idx in v_idx_list_cond])):
                        new_cur_objects.update([lit.variables[v_idx] for v_idx in v_idx_list_cmpl])

    # Then apply rules with two-object conditions
    for lit in state.literals:
        p_name = lit.predicate.name
        if p_name in complementary_rules:
            p_cmpl_rule = complementary_rules[p_name]
            if len(p_cmpl_rule["cond"]) == 2:
                for v_idx_list_cond, v_idx_list_cmpl in zip(p_cmpl_rule["cond"], p_cmpl_rule["cmpl"]):
                    if not cur_objects.isdisjoint(set([lit.variables[v_idx] for v_idx in v_idx_list_cond])):
                        new_cur_objects.update([lit.variables[v_idx] for v_idx in v_idx_list_cmpl])

    new_cur_lits = set()
    for lit in state.literals:
        if all(var in new_cur_objects for var in lit.variables):
            new_cur_lits.add(lit)
    dummy_state = State(new_cur_lits, new_cur_objects, state.goal)

    return new_cur_objects, dummy_state

def apply_relaxation_rules(state, relaxation_rules, domain, force_include_goal_objects=True):
    relaxed_objects = set(state.objects)
    relaxed_literals = set(state.literals)

    goal_objects = set()
    if force_include_goal_objects:
        for lit in state.goal.literals:
            goal_objects |= set(lit.variables)

    for rule_name in relaxation_rules:
        rule = relaxation_rules[rule_name]
        pre_compute_relation = {}
        try:
            for lit in state.literals:
                p_name = lit.predicate.name
                if p_name in rule["pre_compute"]:
                    if p_name not in pre_compute_relation:
                        pre_compute_relation[p_name] = {}
                    v0_idx = rule["pre_compute"][p_name][0]
                    v1_idx = rule["pre_compute"][p_name][1]
                    v0, v1 = lit.variables[v0_idx], lit.variables[v1_idx]
                    pre_compute_relation[p_name][v0] = v1
        except (IndexError, KeyError):
            continue  # skip rules with invalid pre_compute indices

        for lit in state.literals:
            p_name = lit.predicate.name
            if p_name in rule["precond"]:
                try:
                    v_idx_2_obj = {}
                    for v_idx in rule["precond"][p_name]:
                        v_idx_2_obj[v_idx] = lit.variables[v_idx]
                    for v_idx in rule["delete_objects"]:
                        if v_idx_2_obj[v_idx] not in goal_objects:
                            relaxed_objects.discard(v_idx_2_obj[v_idx])
                    for del_p_name in rule["delete_effects"]:
                        del_p = domain.predicates[del_p_name]
                        v_idx_list = rule["delete_effects"][del_p_name]
                        if len(v_idx_list) == 1:
                            v_idx = v_idx_list[0]
                            if v_idx_2_obj[v_idx] not in goal_objects:
                                relaxed_literals.discard(Literal(del_p, [v_idx_2_obj[v_idx]]))
                        elif len(v_idx_list) == 2:
                            for v_idx in v_idx_list:
                                if v_idx not in v_idx_2_obj:
                                    v0 = v_idx_2_obj[0]
                                    try:
                                        v_idx_2_obj[v_idx] = pre_compute_relation[del_p_name][v0]
                                    except (KeyError, IndexError):
                                        continue
                            try:
                                candidate_vars = [v_idx_2_obj[v_idx] for v_idx in v_idx_list]
                                # Do not delete literals that involve goal objects
                                if not any(v in goal_objects for v in candidate_vars):
                                    relaxed_literals.discard(Literal(del_p, candidate_vars))
                            except (KeyError, IndexError):
                                continue
                    for add_p_name in rule["add_effects"]:
                        add_p = domain.predicates[add_p_name]
                        v_idx_list = rule["add_effects"][add_p_name]
                        if len(v_idx_list) == 1:
                            v_idx = v_idx_list[0]
                            relaxed_literals.add(Literal(add_p, [v_idx_2_obj[v_idx]]))
                except (IndexError, KeyError):
                    continue  # skip this literal application if indices are invalid
    
    # Clean up relaxed_literals, if any literal contains object not in relaxed_objects, remove it
    cleaned_relaxed_literals = set()
    for lit in relaxed_literals:
        if all(var in relaxed_objects for var in lit.variables):
            cleaned_relaxed_literals.add(lit)
    relaxed_literals = cleaned_relaxed_literals

    dummy_state = State(relaxed_literals, relaxed_objects, state.goal)
    return relaxed_objects, dummy_state


class IncrementalPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold.
    """
    def __init__(self, is_strips_domain, base_planner, search_guider, seed,
                 gamma=0.9, # parameter for incrementing by score
                 max_iterations=1000,
                 force_include_goal_objects=True):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print("Initializing {} with base planner {}, "
              "guidance {}".format(self.__class__.__name__,
                                   base_planner.__class__.__name__,
                                   search_guider.__class__.__name__))
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(
                state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        cur_objects = set()
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_to_score = {obj: self._guidance.score_object(obj, state)
                           for obj in state.objects if obj not in cur_objects}
        # Initialize threshold.
        threshold = self._gamma
        vis_info = {
            "force_include_goal_objects": cur_objects.copy(),
            "object_to_score": object_to_score,
            "gnn_ignored_objects": None,
            "gnn_ignored_objects_threshold_dict": {},
        }
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lowering threshold.
            unused_objs = sorted(list(state.objects-cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                threshold *= self._gamma
                # See if there are any new objects.
                new_objs = {o for o in unused_objs
                            if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
                    break
            cur_objects |= new_objs
            # Keep only literals referencing currently considered objects.
            cur_lits = set()
            for lit in state.literals:
                if all(var in cur_objects for var in lit.variables):
                    cur_lits.add(lit)
            dummy_state = State(cur_lits, cur_objects, state.goal)
            # Try planning with only this object set.
            print("[Trying to plan with {} objects of {} total, "
                  "threshold is {}...]".format(len(cur_objects), len(state.objects), threshold), flush=True)
            vis_info["gnn_ignored_objects"] = state.objects - cur_objects
            vis_info["gnn_ignored_objects_threshold_dict"][threshold] = state.objects - cur_objects
            try:
                time_elapsed = time.time()-start_time
                # Get a plan from base planner & validate it.
                plan = self._planner(domain, dummy_state, timeout-time_elapsed)
                if not validate_strips_plan(domain_file=dom_file,
                                            problem_file=prob_file,
                                            plan=plan):
                    raise PlanningFailure("Invalid plan")
            except PlanningFailure:
                # Try again with more objects.
                if len(cur_objects) == len(state.objects):
                    # We already tried with all objects, give up.
                    break
                continue
            return plan, vis_info
        raise PlanningFailure("Plan not found! Reached max_iterations.")


class ComplementaryPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold.
    """
    def __init__(self, is_strips_domain, base_planner, search_guider, seed,
                 gamma=0.9, # parameter for incrementing by score
                 max_iterations=1000,
                 force_include_goal_objects=True,
                 complementary_rules=None):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print("Initializing {} with base planner {}, "
              "guidance {}".format(self.__class__.__name__,
                                   base_planner.__class__.__name__,
                                   search_guider.__class__.__name__))
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects
        with open(complementary_rules, "r") as file:
            self._complementary_rules = json.load(file)

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(
                state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        cur_objects = set()
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_to_score = {obj: self._guidance.score_object(obj, state)
                           for obj in state.objects if obj not in cur_objects}
        # Initialize threshold.
        threshold = self._gamma
        vis_info = {
            "force_include_goal_objects": cur_objects.copy(),
            "object_to_score": object_to_score,
            "gnn_ignored_objects": None,
            "gnn_ignored_objects_threshold_dict": {},
            "cmpl_ignored_objects": None,
        }
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lowering threshold.
            unused_objs = sorted(list(state.objects-cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                threshold *= self._gamma
                # See if there are any new objects.
                new_objs = {o for o in unused_objs
                            if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
                    break
            cur_objects |= new_objs
            # Keep only literals referencing currently considered objects.
            cur_lits = set()
            for lit in state.literals:
                if all(var in cur_objects for var in lit.variables):
                    cur_lits.add(lit)
            dummy_state = State(cur_lits, cur_objects, state.goal)
            # Try planning with only this object set.
            print("[Trying to plan with {} objects of {} total, "
                  "threshold is {}...]".format(len(cur_objects), len(state.objects), threshold), flush=True)
            vis_info["gnn_ignored_objects"] = state.objects - cur_objects
            vis_info["gnn_ignored_objects_threshold_dict"][threshold] = state.objects - cur_objects
            try:
                time_elapsed = time.time()-start_time
                # Get a plan from base planner & validate it.
                plan = self._planner(domain, dummy_state, timeout/2-time_elapsed)
                if not validate_strips_plan(domain_file=dom_file,
                                            problem_file=prob_file,
                                            plan=plan):
                    raise PlanningFailure("Invalid plan")
            except PlanningFailure:
                # Try again with more objects.
                # print("time spent:", time.time()-start_time)
                if len(cur_objects) == len(state.objects):
                    # We already tried with all objects, give up.
                    break
                continue
            except PlanningTimeout:
                # Try with enhanced objects.
                new_cur_objects, dummy_state = apply_complementary_rules(state, cur_objects, self._complementary_rules)
                print("[Trying to plan with {} enhanced objects of {} total, "
                  "threshold is {}...]".format(len(new_cur_objects), len(state.objects), threshold), flush=True)
                vis_info["cmpl_ignored_objects"] = state.objects - new_cur_objects
                try:
                    time_elapsed = time.time()-start_time
                    plan = self._planner(domain, dummy_state, timeout-time_elapsed)
                except PlanningTimeout:
                    print("time spent:", time.time()-start_time)
                    raise PlanningTimeout("Planning timed out!")
         
            return plan, vis_info
        raise PlanningFailure("Plan not found! Reached max_iterations.")


class PureRelaxationPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold.
    """
    def __init__(self, is_strips_domain, base_planner, search_guider, seed,
                 gamma=0.9, # parameter for incrementing by score
                 max_iterations=1000,
                 force_include_goal_objects=True,
                 relaxation_rules=None):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print("Initializing {} with base planner {}, "
              "guidance {}".format(self.__class__.__name__,
                                   base_planner.__class__.__name__,
                                   search_guider.__class__.__name__))
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects
        with open(relaxation_rules, "r") as file:
            self._relaxation_rules = json.load(file)

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(
                state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        cur_objects = set()
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_to_score = {obj: self._guidance.score_object(obj, state)
                           for obj in state.objects if obj not in cur_objects}
        # Initialize threshold.
        threshold = self._gamma
        vis_info = {
            "force_include_goal_objects": cur_objects.copy(),
            "object_to_score": object_to_score,
            "gnn_ignored_objects": None,
            "gnn_ignored_objects_threshold_dict": {},
            "relx_ignored_objects": None,
            "relaxed_plan": None,
            "cmpl_ignored_objects": None,
        }
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lowering threshold.
            unused_objs = sorted(list(state.objects-cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                threshold *= self._gamma
                # See if there are any new objects.
                new_objs = {o for o in unused_objs
                            if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
                    break
            cur_objects |= new_objs
            # Keep only literals referencing currently considered objects.
            cur_lits = set()
            for lit in state.literals:
                if all(var in cur_objects for var in lit.variables):
                    cur_lits.add(lit)
            dummy_state = State(cur_lits, cur_objects, state.goal)
            # Try planning with only this object set.
            print("[Trying to plan with {} objects of {} total, "
                  "threshold is {}...]".format(
                      len(cur_objects), len(state.objects), threshold),
                  flush=True)
            vis_info["gnn_ignored_objects"] = state.objects - cur_objects
            vis_info["gnn_ignored_objects_threshold_dict"][threshold] = state.objects - cur_objects
            try:
                time_elapsed = time.time()-start_time
                # Get a plan from base planner & validate it.
                plan = self._planner(domain, dummy_state, timeout/6-time_elapsed)
                if not validate_strips_plan(domain_file=dom_file,
                                            problem_file=prob_file,
                                            plan=plan):
                    raise PlanningFailure("Invalid plan")
            except PlanningFailure:
                # Try again with more objects.
                # print("time spent:", time.time()-start_time)
                if len(cur_objects) == len(state.objects):
                    # We already tried with all objects, give up.
                    break
                continue
            except PlanningTimeout:
                # Try with enhanced objects.
                # Apply relaxation rules and solve the relaxed problem
                relaxed_objects, dummy_state = apply_relaxation_rules(state, self._relaxation_rules, domain, self._force_include_goal_objects)
                print("[Trying to plan the rule-relaxed problem with {} objects of {} total...]".format(len(relaxed_objects), len(state.objects)), flush=True)
                vis_info["relx_ignored_objects"] = state.objects - relaxed_objects
                try:
                    time_elapsed = time.time()-start_time
                    relaxed_plan = self._planner(domain, dummy_state, timeout/2-time_elapsed)
                    vis_info["relaxed_plan"] = relaxed_plan
                except PlanningTimeout:
                    raise PlanningTimeout("Rule-relaxed problem planning timed out!")

                objects_in_relaxed_plan = {o for act in relaxed_plan for o in act.variables}
                cur_objects.update(objects_in_relaxed_plan)

                new_cur_objects = cur_objects.copy()
                new_cur_lits = set()
                for lit in state.literals:
                    if all(var in new_cur_objects for var in lit.variables):
                        new_cur_lits.add(lit)
                dummy_state = State(new_cur_lits, new_cur_objects, state.goal)
                print("[Trying to plan with {} enhanced objects of {} total, "
                  "threshold is {}...]".format(len(new_cur_objects), len(state.objects), threshold), flush=True)
                vis_info["cmpl_ignored_objects"] = state.objects - new_cur_objects
                try:
                    time_elapsed = time.time()-start_time
                    plan = self._planner(domain, dummy_state, timeout-time_elapsed)
                except PlanningTimeout:
                    print("time spent:", time.time()-start_time)
                    raise PlanningTimeout("Planning timed out!")
         
            return plan, vis_info
        raise PlanningFailure("Plan not found! Reached max_iterations.")


class FlaxPlanner(Planner):
    """Sample objects by incrementally lowering a score threshold.
    """
    def __init__(self, is_strips_domain, base_planner, search_guider, seed,
                 gamma=0.9, # parameter for incrementing by score
                 max_iterations=1000,
                 force_include_goal_objects=True,
                 complementary_rules=None,
                 relaxation_rules=None):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print("Initializing {} with base planner {}, "
              "guidance {}".format(self.__class__.__name__,
                                   base_planner.__class__.__name__,
                                   search_guider.__class__.__name__))
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects
        with open(complementary_rules, "r") as file:
            self._complementary_rules = json.load(file)
        with open(relaxation_rules, "r") as file:
            self._relaxation_rules = json.load(file)

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(
                state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)
        cur_objects = set()
        start_time = time.time()
        if self._force_include_goal_objects:
            # Always start off considering objects in the goal.
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)
        # Get scores once.
        object_to_score = {obj: self._guidance.score_object(obj, state)
                           for obj in state.objects if obj not in cur_objects}
        # Initialize threshold.
        threshold = self._gamma
        vis_info = {
            "force_include_goal_objects": cur_objects.copy(),
            "object_to_score": object_to_score,
            "gnn_ignored_objects": None,
            "gnn_ignored_objects_threshold_dict": {},
            "relx_ignored_objects": None,
            "relaxed_plan": None,
            "cmpl_ignored_objects": None,
        }
        for _ in range(self._max_iterations):
            # Find new objects by incrementally lowering threshold.
            unused_objs = sorted(list(state.objects-cur_objects))
            new_objs = set()
            while unused_objs:
                # Geometrically lower threshold.
                threshold *= self._gamma
                # See if there are any new objects.
                new_objs = {o for o in unused_objs
                            if object_to_score[o] > threshold}
                # If there are, try planning with them.
                if new_objs:
                    break
            cur_objects |= new_objs
            # Keep only literals referencing currently considered objects.
            cur_lits = set()
            for lit in state.literals:
                if all(var in cur_objects for var in lit.variables):
                    cur_lits.add(lit)
            dummy_state = State(cur_lits, cur_objects, state.goal)
            # Try planning with only this object set.
            print("[Trying to plan with {} objects of {} total, "
                  "threshold is {}...]".format(
                      len(cur_objects), len(state.objects), threshold),
                  flush=True)
            vis_info["gnn_ignored_objects"] = state.objects - cur_objects
            vis_info["gnn_ignored_objects_threshold_dict"][threshold] = state.objects - cur_objects
            try:
                time_elapsed = time.time()-start_time
                # Get a plan from base planner & validate it.
                plan = self._planner(domain, dummy_state, timeout/6-time_elapsed)
                if not validate_strips_plan(domain_file=dom_file,
                                            problem_file=prob_file,
                                            plan=plan):
                    raise PlanningFailure("Invalid plan")
            except PlanningFailure:
                # Try again with more objects.
                # print("time spent:", time.time()-start_time)
                if len(cur_objects) == len(state.objects):
                    # We already tried with all objects, give up.
                    break
                continue
            except PlanningTimeout:
                # Try with enhanced objects.
                # Apply relaxation rules and solve the relaxed problem
                relaxed_objects, dummy_state = apply_relaxation_rules(state, self._relaxation_rules, domain, self._force_include_goal_objects)
                print("[Trying to plan the rule-relaxed problem with {} objects of {} total...]".format(len(relaxed_objects), len(state.objects)), flush=True)
                vis_info["relx_ignored_objects"] = state.objects - relaxed_objects
                try:
                    time_elapsed = time.time()-start_time
                    relaxed_plan = self._planner(domain, dummy_state, timeout/2-time_elapsed)
                    vis_info["relaxed_plan"] = relaxed_plan
                except PlanningTimeout:
                    raise PlanningTimeout("Rule-relaxed problem planning timed out!")

                objects_in_relaxed_plan = {o for act in relaxed_plan for o in act.variables}
                cur_objects.update(objects_in_relaxed_plan)

                # Apply complementary rules
                new_cur_objects, dummy_state = apply_complementary_rules(state, cur_objects, self._complementary_rules)
                print("[Trying to plan with {} enhanced objects of {} total, "
                  "threshold is {}...]".format(len(new_cur_objects), len(state.objects), threshold), flush=True)
                vis_info["cmpl_ignored_objects"] = state.objects - new_cur_objects
                try:
                    time_elapsed = time.time()-start_time
                    plan = self._planner(domain, dummy_state, timeout-time_elapsed)
                except PlanningTimeout:
                    print("time spent:", time.time()-start_time)
                    raise PlanningTimeout("Planning timed out!")
         
            return plan, vis_info
        raise PlanningFailure("Plan not found! Reached max_iterations.")


class LLMFlaxPlanner(Planner):
    """Flax planner where Step 1 uses LLM-guided object recovery instead of
    blind gamma-decay threshold lowering.

    When the GNN-pruned simplified task times out, instead of geometrically
    lowering the score threshold, we ask an LLM which specific excluded
    objects are likely needed to solve the problem.

    Steps:
      Step 1: GNN scores objects. If plan found → return.
              On timeout → LLM suggests which objects to add → retry once.
      Step 2: If still failing, fall back to standard relaxation rules.
      Step 3: Apply complementary rules.
    """

    def __init__(self, is_strips_domain, base_planner, search_guider, seed,
                 gamma=0.9,
                 max_iterations=1000,
                 force_include_goal_objects=True,
                 complementary_rules=None,
                 relaxation_rules=None,
                 llm_recovery=None):
        super().__init__()
        assert isinstance(base_planner, Planner)
        print("Initializing {} with base planner {}, guidance {}, "
              "LLM recovery {}".format(
                  self.__class__.__name__,
                  base_planner.__class__.__name__,
                  search_guider.__class__.__name__,
                  llm_recovery.__class__.__name__ if llm_recovery else "None"))
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._planner = base_planner
        self._guidance = search_guider
        self._rng = np.random.RandomState(seed=seed)
        self._force_include_goal_objects = force_include_goal_objects
        self._llm_recovery = llm_recovery
        with open(complementary_rules, "r") as f:
            self._complementary_rules = json.load(f)
        with open(relaxation_rules, "r") as f:
            self._relaxation_rules = json.load(f)

    def __call__(self, domain, state, timeout):
        act_preds = [domain.predicates[a] for a in list(domain.actions)]
        act_space = LiteralSpace(
            act_preds, type_to_parent_types=domain.type_to_parent_types)
        dom_file = tempfile.NamedTemporaryFile(delete=False).name
        prob_file = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(dom_file)
        lits = set(state.literals)
        if not domain.operators_as_actions:
            lits |= set(act_space.all_ground_literals(state, valid_only=False))
        PDDLProblemParser.create_pddl_file(
            prob_file, state.objects, lits, "myproblem",
            domain.domain_name, state.goal, fast_downward_order=True)

        cur_objects = set()
        start_time = time.time()
        if self._force_include_goal_objects:
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)

        # Score objects once with GNN
        object_to_score = {
            obj: self._guidance.score_object(obj, state)
            for obj in state.objects if obj not in cur_objects
        }
        threshold = self._gamma

        vis_info = {
            "force_include_goal_objects": cur_objects.copy(),
            "object_to_score": object_to_score,
            "gnn_ignored_objects": None,
            "gnn_ignored_objects_threshold_dict": {},
            "llm_recovery_added": None,
            "relx_ignored_objects": None,
            "relaxed_plan": None,
            "cmpl_ignored_objects": None,
        }

        # ── Step 1: GNN-guided incremental pruning ──────────────────────────
        step1_timed_out = False
        for _ in range(self._max_iterations):
            unused_objs = sorted(list(state.objects - cur_objects))
            new_objs = set()
            while unused_objs:
                threshold *= self._gamma
                new_objs = {o for o in unused_objs
                            if object_to_score.get(o, 0) > threshold}
                if new_objs:
                    break
            cur_objects |= new_objs

            cur_lits = {lit for lit in state.literals
                        if all(v in cur_objects for v in lit.variables)}
            dummy_state = State(cur_lits, cur_objects, state.goal)

            print("[Step1] Planning with {}/{} objects, threshold={:.3f}...".format(
                len(cur_objects), len(state.objects), threshold), flush=True)
            vis_info["gnn_ignored_objects"] = state.objects - cur_objects
            vis_info["gnn_ignored_objects_threshold_dict"][threshold] = \
                state.objects - cur_objects

            try:
                time_elapsed = time.time() - start_time
                plan = self._planner(domain, dummy_state,
                                     timeout / 6 - time_elapsed)
                if not validate_strips_plan(domain_file=dom_file,
                                            problem_file=prob_file,
                                            plan=plan):
                    raise PlanningFailure("Invalid plan")
                return plan, vis_info
            except PlanningFailure:
                if len(cur_objects) == len(state.objects):
                    break
                continue
            except PlanningTimeout:
                step1_timed_out = True
                break

        # ── LLM recovery (between Step 1 and Step 2) ────────────────────────
        # Budget design:
        #   - LLM API calls take ~3–5 s each (Qwen2.5-14b via Ollama).
        #   - We must leave at least STEP2_MIN seconds for Step 2.
        #   - Feasibility check: only attempt recovery if the gap between now
        #     and the timeout/2 Step-2 deadline is large enough to cover an
        #     estimated LLM call + a minimum recovery replan + STEP2_MIN.
        RECOVERY_FRACTION  = 0.15   # cap replan at 15 % of total timeout
        LLM_CALL_ESTIMATE  = 5.0    # conservative API call budget (seconds)
        STEP2_MIN          = 5.0    # guaranteed minimum for Step 2

        if step1_timed_out and self._llm_recovery is not None:
            time_elapsed = time.time() - start_time
            time_before_step2 = timeout / 2 - time_elapsed  # slack before Step-2 deadline
            # Minimum needed: LLM call + 1 s replan + STEP2_MIN
            if time_before_step2 >= LLM_CALL_ESTIMATE + 1.0 + STEP2_MIN:
                recovery_budget = min(
                    timeout * RECOVERY_FRACTION,
                    time_before_step2 - LLM_CALL_ESTIMATE - STEP2_MIN
                )
                print("[LLM Recovery] Asking LLM for missing objects "
                      "(budget={:.1f}s)...".format(recovery_budget), flush=True)
                llm_added = self._llm_recovery.suggest_objects(
                    state, cur_objects, state.objects)
                if llm_added:
                    print("[LLM Recovery] Adding: {}".format(
                        [o.name for o in llm_added]), flush=True)
                    cur_objects |= llm_added
                    vis_info["llm_recovery_added"] = llm_added

                    cur_lits = {lit for lit in state.literals
                                if all(v in cur_objects for v in lit.variables)}
                    dummy_state = State(cur_lits, cur_objects, state.goal)
                    # Recompute budget using actual elapsed time after LLM call
                    time_elapsed = time.time() - start_time
                    actual_budget = min(
                        recovery_budget,
                        max(0.0, timeout / 2 - time_elapsed - STEP2_MIN / 2)
                    )
                    print("[LLM Recovery] Replanning with {}/{} objects "
                          "(budget={:.1f}s)...".format(
                              len(cur_objects), len(state.objects),
                              actual_budget), flush=True)
                    if actual_budget > 0.2:
                        try:
                            plan = self._planner(domain, dummy_state,
                                                 actual_budget)
                            if validate_strips_plan(domain_file=dom_file,
                                                    problem_file=prob_file,
                                                    plan=plan):
                                return plan, vis_info
                        except (PlanningFailure, PlanningTimeout):
                            print("[LLM Recovery] Still failed, proceeding "
                                  "to relaxation.", flush=True)
            else:
                print("[LLM Recovery] Skipping — only {:.1f}s before Step-2 "
                      "deadline (need ≥{:.1f}s).".format(
                          time_before_step2,
                          LLM_CALL_ESTIMATE + 1.0 + STEP2_MIN),
                      flush=True)

        # ── Step 2: Relaxation rules + rough plan ────────────────────────────
        # Guarantee Step 2 always gets at least STEP2_MIN seconds, even if
        # recovery used up most of the first-half budget.
        relaxed_objects, dummy_state = apply_relaxation_rules(
            state, self._relaxation_rules, domain,
            self._force_include_goal_objects)
        print("[Step2] Rule-relaxed problem with {}/{} objects...".format(
            len(relaxed_objects), len(state.objects)), flush=True)
        vis_info["relx_ignored_objects"] = state.objects - relaxed_objects
        try:
            time_elapsed = time.time() - start_time
            step2_budget = max(STEP2_MIN, timeout / 2 - time_elapsed)
            relaxed_plan = self._planner(domain, dummy_state, step2_budget)
            vis_info["relaxed_plan"] = relaxed_plan
        except PlanningTimeout:
            raise PlanningTimeout("Rule-relaxed problem planning timed out!")

        objects_in_relaxed_plan = {o for act in relaxed_plan
                                   for o in act.variables}
        cur_objects.update(objects_in_relaxed_plan)

        # ── Step 3: Complementary rules ──────────────────────────────────────
        new_cur_objects, dummy_state = apply_complementary_rules(
            state, cur_objects, self._complementary_rules)
        print("[Step3] Complementary expansion: {}/{} objects...".format(
            len(new_cur_objects), len(state.objects)), flush=True)
        vis_info["cmpl_ignored_objects"] = state.objects - new_cur_objects
        try:
            time_elapsed = time.time() - start_time
            plan = self._planner(domain, dummy_state, timeout - time_elapsed)
        except PlanningTimeout:
            raise PlanningTimeout("Planning timed out!")

        return plan, vis_info

