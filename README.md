# LLM-Augmented Flax: Toward Domain-Agnostic Neuro-Symbolic Task Planning

**Core goal:** Given only a PDDL domain file, deploy a competent neuro-symbolic planner in seconds — no hand-crafted rules, no training data collection.

The original Flax planner requires three types of domain-specific manual effort per new domain:

1. Hand-crafted relaxation rules (domain expert, hours)
2. Hand-crafted complementary rules (domain expert, hours)
3. 200 solved training problems for GNN (planner + compute, hours)

This project replaces all three with LLM calls.

---

## Research Roadmap

```
Stage 1 [DONE]   LLM auto-generates relaxation & complementary rules
                 → Eliminates manual rule engineering
                 → 8 benchmarks evaluated; avg SR gap vs manual: -0.008

Stage 2 [IMPL]   LLM guides object recovery after Step 1 failure
                 → Replaces blind γ=0.9 threshold decay with semantic reasoning
                 → Code done; experiments running on 12×12 Hard, 15×15 Medium

Stage 3 [IMPL]   LLM scores object importance zero-shot
                 → Replaces domain-specific GNN training entirely
                 → Code done; experiments pending

Full LLM-Flax    Stage 1 + 2 + 3
                 → PDDL file in → working planner out, zero manual effort
```

### Ablation Table

| Configuration | SR (avg, 8 tasks) | Training data | Manual rules |
|---|---|---|---|
| Flax baseline (manual rules + GNN) | **0.803** | ✓ 200 problems | ✓ |
| + LLM rules (Stage 1) | 0.795 | ✓ 200 problems | ✗ |
| + LLM failure recovery (Stage 2) | TBD | ✓ 200 problems | ✗ |
| Full LLM-Flax (Stage 1+2+3) | TBD | **✗** | ✗ |

> **Key reframing**: the goal is not to match manual performance.
> It is to enable deployment on a **new domain in seconds**, without any domain-expert involvement.
> A slight average SR reduction is acceptable if it eliminates hours of manual engineering.

---

## Stage 1 (DONE): LLM Rule Generation

### What It Replaces

Two hand-crafted JSON files per domain:

- **Relaxation rules** — which objects to remove to simplify the problem, and what effects to update
- **Complementary rules** — which object pairs must always appear together

### Pipeline

```
PDDL domain file
      ↓
[LLM (Qwen2.5-14b via Ollama)]
      ↓  ← format validation + retry (up to 3×)
      ↓  ← duplicate removal
      ↓  ← unknown predicate filtering
Generated rules (JSON)  →  Flax planner (unchanged)
```

### Key Files

- `scripts/generate_rules_llm.py` — rule generation with validation & retry
- `scripts/compare_rules.py` — ablation experiment runner
- `config/mazenamo_relaxation_rules_qwen2.5-14b.json` — generated output
- `config/mazenamo_complementary_rules_qwen2.5-14b.json` — generated output

### Results (n=50 unless noted)

| Grid | Difficulty | Manual SR | LLM SR | Δ SR | Manual Time (s) | LLM Time (s) |
|------|------------|-----------|--------|------|-----------------|--------------|
| 10×10 | Easy | **1.000** | 0.940 | −0.060 | 0.932 | 0.853 |
| 10×10 | Medium | **0.960** | 0.900 | −0.060 | 1.997 | 1.925 |
| 10×10 | Hard | **0.960** | 0.900 | −0.060 | 2.200 | 2.020 |
| 12×12 | Medium | **0.960** | 0.920 | −0.040 | 1.914 | 1.773 |
| 12×12 | Hard | **0.680** | 0.500 | −0.180 | 4.594 | 6.967 |
| 12×12 | Expert (n=30) | 0.000 | **1.000** | **+1.000** | timeout | 1.574 |
| 15×15 | Medium (n=30) | **0.967** | 0.200 | −0.767 | 9.132 | 6.342 |
| 15×15 | Hard (n=30) | 0.900 | **1.000** | +0.100 | 11.901 | 1.931 |
| **Average** | | **0.803** | 0.795 | −0.008 | | |

### Key Findings

- **Non-monotonic pattern**: LLM rules fail on stacking-heavy tasks (Medium) but excel on navigation-heavy tasks (Hard/Expert)
- **12×12 Expert reversal**: Manual times out 100% (SR 0.000); LLM solves all 30 at 1.57 s
- **Root cause**: LLM adds `clear:[0]` precondition → cannot remove stacked light boxes → stacking-heavy tasks fail
- **Richer complementary rules**: LLM generates 3 rules (`oat`+`rat`+`upon`) vs manual's 1, benefiting large navigation problems
- **Avg gap near zero**: −0.008 across 8 benchmarks when full difficulty spectrum is included

---

## Stage 2 (IMPL): LLM-Guided Failure Recovery

### What It Replaces

When Step 1 (GNN pruning) times out, the current pipeline blindly lowers the score threshold:

```
Current:  plan fails → threshold *= 0.9 → include more objects → retry (blind)
Proposed: plan fails → LLM(state, goal, excluded_objects) → add specific objects → retry
```

### Implementation

- **`guidance/llm_recovery_guidance.py`** — `LLMRecoveryGuidance` class
  - Input: PDDL state literals, goal, current and excluded object sets
  - Prompt: goal + current objects + excluded objects + relevant state facts
  - Output: set of objects to add
  - Fallback: returns empty set if LLM call fails (pipeline continues to Step 2)

- **`planning/my_planner.py`** — `LLMFlaxPlanner` class
  - Runs GNN Step 1; on timeout → calls `LLMRecoveryGuidance.suggest_objects()` → replans
  - If still failing → falls back to standard Step 2 (relaxation rules)

### Results (final, v3 budget policy)

| Config | 12×12 Hard SR | 15×15 Medium SR |
| --- | --- | --- |
| Manual | **0.880** | **0.967** |
| LLM rules | 0.800 | 0.833 |
| LLM rules + recovery | 0.800 | 0.833 |

**Key finding: LLM recovery is neutral** — SR is unchanged from the LLM rules baseline.

The v3 feasibility check correctly identifies that a 30 s timeout leaves only ~10 s before the Step-2 deadline (below the 11 s threshold), so the LLM call is skipped. At 40 s timeout (15×15 Medium) the LLM call proceeds but relaxation already handles these problems reliably.

Budget policy design history:
- **v1** (naive): shared budget → SR 0.633 (regression)
- **v2** (cap without LLM latency): SR 0.02 (catastrophic)
- **v3** (feasibility check + cap + guarantee): SR 0.800/0.833 (neutral, no regression)

### Status

Complete (both 12×12 Hard and 15×15 Medium). v1/v2 design lessons documented in paper appendix.

---

## Stage 3 (IMPL): LLM Zero-Shot Object Scoring

### What It Replaces

The GNN requires 200 domain-specific solved problems for training. Stage 3 replaces it entirely:

```
Current:  GNN(trained on 200 domain problems) → score(o) ∈ [0,1]
Proposed: LLM(PDDL state + goal, zero-shot)  → score(o) ∈ [0,1]
```

### Implementation

- **`guidance/llm_object_guidance.py`** — `LLMObjectGuidance` class
  - Drop-in replacement for `GNNSearchGuidance` (same `score_object(obj, state)` API)
  - `train()` is a no-op — no training data needed
  - Single LLM call per planning problem (all objects scored at once, then cached)
  - Prompt: goal + object list + state facts → `{"obj_name": 0.0–1.0, ...}` JSON
  - Fallback: scores everything 0.5 on failure (threshold decay still works)

### Full LLM-Flax Pipeline

With Stage 1+2+3, the complete system requires only:

```bash
# 1. Generate rules (once per domain, ~10 s)
python scripts/generate_rules_llm.py --domain pddl_files/domains/newdomain.pddl

# 2. Run Full LLM-Flax (no training, no manual rules)
python scripts/compare_rules.py --size 12 --difficulty hard \
    --num_problems 30 --test_timeout 30 --configs full_llm_flax
```

### Results

| Config | 12×12 Hard SR | 12×12 Hard Time (s) | 15×15 Hard SR | 15×15 Hard Time (s) |
|---|---|---|---|---|
| Manual | 0.880 | 3.97 | 0.900 | 11.90 |
| LLM rules | 0.800 | 3.22 | **1.000** | 1.93 |
| **Full LLM-Flax** | **0.720** | **22.2** | **0.200** | **23.1** |

**Key findings:**
- 12×12 Hard: SR 0.720 with zero training data — feasible, but 5.6× slower
- 15×15 Hard: SR collapses to 0.200 — scale bottleneck (240+ objects, 80-fact prompt cap)

**Root cause of 15×15 collapse:** GNN trained on 200 problems learns efficient navigation pruning; LLM zero-shot scoring with truncated state context can't replicate this at scale → Step 1 timeout rate spikes → 40s budget insufficient.

### Status

Complete (12×12 Hard, 15×15 Hard). Scale limitation identified: prompt context cap needs to be addressed for large problems.

---

## Setup

```bash
# Install Ollama and model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b
pip install openai

# Stage 1: Generate rules for a domain
python scripts/generate_rules_llm.py \
    --domain pddl_files/domains/mazenamo.pddl \
    --model qwen2.5:14b

# Stage 1+2 ablation (manual vs LLM rules vs LLM rules+recovery)
python scripts/compare_rules.py \
    --size 12 --difficulty hard --num_problems 50 --test_timeout 30

# Full LLM-Flax (Stage 1+2+3): zero training, zero manual rules
python scripts/compare_rules.py \
    --size 12 --difficulty hard --num_problems 30 --test_timeout 30 \
    --configs full_llm_flax
```

---

## File Structure

```
scripts/
  generate_rules_llm.py       # Stage 1: LLM rule generation
  compare_rules.py            # Ablation experiment runner (all stages)

guidance/
  gnn_guidance.py             # Baseline: GNN object scorer (requires training)
  llm_recovery_guidance.py    # Stage 2: LLM failure recovery
  llm_object_guidance.py      # Stage 3: LLM zero-shot object scorer

planning/
  my_planner.py               # FlaxPlanner + LLMFlaxPlanner (Stage 2)

config/
  mazenamo_relaxation_rules_1.json              # Manual (baseline)
  mazenamo_complementary_rules.json             # Manual (baseline)
  mazenamo_relaxation_rules_qwen2.5-14b.json    # Stage 1 output
  mazenamo_complementary_rules_qwen2.5-14b.json # Stage 1 output

results/
  ablation_*.json             # Experiment results

paper/
  main.tex                    # Paper draft
  refs.bib                    # References
```

---

## Acknowledgments

This project builds directly on **Flax** — the neuro-symbolic task planner developed by Du et al. (2026).
The core three-step planning loop (GNN pruning → relaxation → complementary expansion), the MazeNamo benchmark, and the GNN training infrastructure are all from the original Flax system.
Our contribution is three stages of LLM-based automation layered on top of the unmodified Flax planner.

```bibtex
@article{du2026fast,
  title={Fast Task Planning with Neuro-Symbolic Relaxation},
  author={Du, Qiwei and Li, Bowen and Du, Yi and Su, Shaoshu and Fu, Taimeng
          and Zhan, Zitong and Zhao, Zhipeng and Wang, Chen},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  publisher={IEEE},
  note={arXiv:2507.15975}
}
```
