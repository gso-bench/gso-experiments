# Measuring Optimization Capability Scaling

**Question**: Are models getting exponentially better at optimizing code — and how would we measure that?

METR's time-horizon benchmark shows that the task-length at which models have 50% success doubles every ~7 months. But that measures *long-horizon coding ability* — the ability to write more code for longer. It doesn't measure anything specific to *optimization*: understanding where bottlenecks are, knowing which technique to apply, trading off implementation complexity vs. speedup.

## What existing GSO data already tells us

### 1. Optimization is continuous, not binary

GSO currently scores tasks as binary (did the model match expert speedup?). But the underlying data is continuous: `model_speedup / expert_speedup` gives an **optimization level** between 0 and 1+.

Key finding: **even when models "fail", they achieve 65-85% of the expert's speedup**. Only 4-24% of test-passing attempts produce zero speedup. Models almost always find *something* to optimize — they just don't find the *best* optimization.

| Model | Median Optimization Level | 50% OPL |
|-------|--------------------------|---------|
| Claude Opus 4.6 | 90% | 94% |
| Claude Opus 4.5 | 89% | 89% |
| GPT-5.2 | 86% | 85% |
| Gemini 3 Pro | 83% | 82% |
| Claude Sonnet 4.5 | 72% | 80% |
| GPT-5.1 | 68% | 68% |
| o3 | 48% | 64% |
| Gemini 3 Flash | 52% | 59% |

The **50% Optimization Proficiency Level (OPL)** is the fraction of expert speedup at which a model's probability of achieving it drops to 50%. This is a logistic fit:

```
P(model_speedup / expert_speedup >= t) = σ(α + β·t)
50% OPL = -α/β
```

### 2. Technique type is the real capability discriminator

Binary solve rates by optimization technique (top model, Claude Opus 4.6):

| Technique | Solve Rate | What it requires |
|-----------|-----------|------------------|
| Caching/avoidance | 60% | Identify redundant work, add memoization |
| C/C++ optimization | 45% | Write/modify compiled code in the right place |
| Python-level | 42% | Better algorithms, data structures, pandas idioms |
| Rust rewrite | 40% | Rewrite hot path in Rust |
| SIMD/vectorization | 29% | AVX/SSE intrinsics, hardware-aware optimization |
| Cython | 22% | Typed Cython extensions |
| ufunc/C-loop | 15% | NumPy ufunc internals, C iteration machinery |

The gap between "caching" (60%) and "ufunc/C-loop" (15%) is **not about coding ability** — it's about optimization-specific knowledge. Models can write C code, but they struggle to write the *right* C code in the *right place* within NumPy's ufunc dispatch machinery.

### 3. Target speedup has no predictive power

We confirmed that `log2(target_speedup)` has AIC *worse than the null model* (750 vs 749). The magnitude of speedup an optimization achieves tells you nothing about how hard it is to implement. A 1.3x optimization via a 500-line C++ patch is much harder than an 8x optimization via a 20-line Python change.

## Proposal: Optimization Proficiency Level (OPL) as the scaling metric

### The core metric

Instead of METR's time-horizon, track the **50% OPL** over model generations:

```
50% OPL = the fraction of expert-level speedup at which
          the model's success probability crosses 50%
```

- Current best model (Opus 4.6): 94% OPL
- Current weakest (Gemini Flash): 59% OPL
- Expert level: 100%

**As models improve, 50% OPL should grow toward and past 100%.** If it grows at a consistent rate, that's your exponential scaling law for optimization capability.

### Why this is different from METR

| | METR Time-Horizon | OPL |
|--|---|---|
| **Measures** | How long a task can the model complete? | How close to expert optimization can the model get? |
| **x-axis** | log₂(human minutes) | optimization level (model/expert speedup) |
| **Captures** | Long-horizon coding, planning, debugging | Optimization insight, technique knowledge, domain expertise |
| **Saturates at** | Months/years of autonomous work | Exceeding expert-level optimization |
| **Unique signal** | Can the model persist on a long task? | Does the model know WHERE and HOW to optimize? |

A model could have a long time-horizon (can write code for hours) but a low OPL (doesn't know which optimization technique to apply). OPL captures the **optimization-specific** part of capability.

### What's needed to make the scaling claim

**With existing data** (what we have now):
- 8 models at one point in time
- OPL ranges from 59% to 94%
- Clean logistic fits, technique breakdowns

**To claim exponential scaling** (what's needed):
- 4+ model generations from the same family (e.g., GPT-4 → 4o → 5.1 → 5.2 → 6) evaluated on the same benchmark
- Plot 50% OPL vs. release date
- Fit exponential: OPL(t) = OPL₀ · 2^(t/τ)
- τ = the "OPL doubling time"

## Medium-term: making the eval more powerful

### Multi-level optimization targets

Current GSO has one target per task (the expert solution). A richer eval would define **multiple optimization levels per codebase**, from easy to hard:

- **Level 1**: Obvious improvement (add caching, remove redundant copy) → ~1.5x
- **Level 2**: Algorithmic improvement (better data structure, smarter indexing) → ~3x
- **Level 3**: Language-level optimization (Cython, compiled extension) → ~8x
- **Level 4**: Hardware-aware optimization (SIMD, memory layout, parallelism) → ~20x+

Source these from **real commit histories** where optimization happened incrementally across multiple PRs. Many open-source projects have this naturally (e.g., NumPy's ufunc optimization happened in stages across dozens of commits).

Then fit: `P(model reaches level L) = σ(α + β·L)`

The **50% depth horizon** = the optimization depth at which the model's success drops to 50%. Track this over model generations: "models can handle one deeper optimization level every N months."

### Technique-specific scaling curves

Track solve rate per technique category over model generations:

```
SIMD/vectorization:  GPT-4: 0% → GPT-5.1: 10% → GPT-5.2: 25% → GPT-6: ???
Python-level:        GPT-4: 5% → GPT-5.1: 20% → GPT-5.2: 42% → GPT-6: ???
```

This tells a richer story: **which optimization capabilities are scaling fastest?** If SIMD is growing faster than Python-level, models are gaining domain knowledge, not just coding ability.

### Agentic iteration component

Real optimization is iterative: profile → hypothesize → implement → measure → repeat. A next-gen eval could:

1. Give the model a codebase + profiling tools + N iterations
2. Measure speedup achieved after each iteration
3. Better models would: converge faster, reach higher final speedup, make fewer wasted iterations

The **iteration efficiency** = how many iterations to reach X% of expert speedup. This captures the "optimization insight" dimension that one-shot evaluation misses.

## Summary

| Timeframe | What to track | What it measures |
|-----------|--------------|-----------------|
| **Now** (existing data) | 50% OPL per model | Optimization proficiency snapshot |
| **Short-term** (evaluate older models) | 50% OPL over time | OPL growth rate |
| **Medium-term** (multi-level targets) | 50% depth horizon over time | Optimization depth scaling |
| **Longer-term** (agentic iteration) | Iteration efficiency over time | Optimization insight scaling |

The key insight: **optimization capability is not the same as coding ability**. METR measures the latter. OPL and its extensions measure the former. Both are needed for a complete picture of AI capability scaling.
