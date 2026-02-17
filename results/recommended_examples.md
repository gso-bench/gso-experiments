# GSO Benchmark: Recommended Unsaturated Examples

12 tasks where all top-8 models fail to match the expert's optimization. Selected from the 102-instance [GSO benchmark](https://huggingface.co/datasets/gso-bench/gso) using: expert speedup > 1.5x, best model closeness < 0.85, and at least 3 models producing correct patches.

### How GSO scoring works

Each task has an **expert commit** (the ground-truth optimization) and a **base** (the unoptimized code). A model's patch is evaluated on two axes:

- **Speedup** (geometric mean): How much faster is the model's patch vs the unoptimized base? Higher is better.
- **Closeness** (harmonic mean of patch time / expert commit time): How close is the model's patch to the expert's speed? A value of 1.0 means the model matches the expert exactly; < 1.0 means slower than the expert; values below 0.95 count as unsolved.

A model "solves" a task only if its patch is both (a) at least 1.2x faster than base and (b) within 5% of the expert's speed (closeness >= 0.95).

## Summary Table

| Instance | API | Expert Speedup | Best Model | Model Speedup | Closeness | Correct / Faster | Gap Type |
|---|---|---|---|---|---|---|---|
| pandas-fd43d4b | RangeIndex.take | 8.33x | gpt-5.2 | 6.48x | 0.73 | 3 / 3 | cross_module_tracing |
| numpy-ba89ef9 | numpy.add.at | 5.39x | claude-opus-4.6 | 4.31x | 0.78 | 5 / 5 | cross_module_tracing |
| numpy-19bfa3f | np.char.add | 5.23x | gemini-3-flash | 1.77x | 0.32 | 2 / 1 | compiled_code_barrier |
| numpy-22ab9aa | np.char.rfind | 3.78x | gpt-5.1 | 1.63x | 0.41 | 4 / 3 | compiled_code_barrier |
| numpy-83c780d | np.char.find | 3.26x | claude-sonnet-4.5 | 3.49x | 0.79 | 6 / 5 | compiled_code_barrier, algorithm_porting |
| numpy-cb0d7cd | numpy.strings.ljust | 3.18x | gpt-5.2 | 1.74x | 0.51 | 4 / 3 | compiled_code_barrier, kernel_fusion |
| pandas-bfaf917 | maybe_sequence_to_range | 2.67x | claude-opus-4.5 | 2.14x | 0.71 | 7 / 5 | compiled_code_barrier |
| pandas-2cdca01 | Period.strftime | 2.47x | gpt-5.1 | 1.39x | 0.54 | 8 / 7 | compiled_code_barrier |
| tornado-4d4c1e0 | Future.set_exception | 2.40x | claude-opus-4.6 | 1.57x | 0.55 | 8 / 3 | architectural_substitution |
| pydantic-addf1f9 | BaseModel.__setattr__ | 2.33x | gpt-5.2 | 1.81x | 0.76 | 6 / 4 | memoization_pattern |
| tornado-ac13ee5 | Future.done | 2.06x | gemini-3-pro | 1.49x | 0.70 | 7 / 4 | architectural_substitution |
| pydantic-ac9e6ee | TypeAdapter.validate_python | 1.80x | gpt-5.2 | 1.61x | 0.85 | 6 / 5 | compiled_code_barrier |

**Correct / Faster** = models passing correctness tests / models that also produce a speedup >= 1.2x (but still fail to match the expert).

## Capability Gap Taxonomy

- **compiled_code_barrier** (7/12): Expert writes new C/C++/Cython/Rust; model limited to Python-level optimizations
- **cross_module_tracing** (2/12): Expert traces performance across files/layers; model fixes the obvious bottleneck only
- **architectural_substitution** (2/12): Expert replaces an abstraction entirely; model optimizes within it
- **memoization_pattern** (1/12): Expert caches a deterministic dispatch decision; model reduces but doesn't eliminate per-call overhead
- **kernel_fusion** (1/12): Expert fuses multiple passes into one native kernel; model composes existing ops across passes
- **algorithm_porting** (1/12): Expert ports a known-optimal algorithm; model writes a naive implementation

## Deep Dives

### pandas-dev__pandas-fd43d4b -- RangeIndex.take (8.33x target)
- **Expert**: Arithmetic `start + step * indices` instead of materializing 100M elements. Cross-module fix in `cast.py` to prevent accidental `_data` materialization.
- **Model**: Same arithmetic approach but single-file only. Misses secondary materialization paths in adjacent modules.
- **Key insight**: Primary optimization gets 78% of speedup; remaining 22% from preventing `_values` materialization in call sites the model never examines.

### numpy__numpy-ba89ef9 -- numpy.add.at (5.39x target)
- **Expert**: Fast path bypassing NpyIter buffering + SIMD dispatch bypass for count=1 in a separate C file.
- **Model**: Same NpyIter bypass (correct), plus extra iterator inlining. Misses the 2-line SIMD dispatch fix.
- **Key insight**: 20% gap from a 2-line change in a different file that the model never traces into.

### numpy__numpy-19bfa3f -- np.char.add (5.23x target)
- **Expert**: New C++ `string_add` using `memcpy` on raw buffers. Zero Python objects. Registered as ufunc.
- **Model**: `arr.astype(object) + arr2.astype(object)` then convert back. 3 passes with full Python object boxing.
- **Key insight**: Most dramatic gap. Model cannot write C++ ufunc loops.

### numpy__numpy-83c780d -- np.char.find (3.26x target)
- **Expert**: C++ ufunc porting CPython's fastsearch (Boyer-Moore-Horspool + Two-Way). 724 lines.
- **Model**: Naive O(n*m) search inside `_vec_string`. GM exceeds target (3.49x) but HM fails (0.785x) due to regressions.
- **Key insight**: Model wrote naive algorithm instead of porting CPython's well-known fastsearch.

### numpy__numpy-22ab9aa -- np.char.rfind (3.78x target)
- **Expert**: Same C++ ufunc with fastsearch as np.char.find, using FAST_RSEARCH mode.
- **Model**: `tolist()` + list comprehension `[s.rfind(...) for s in seq]`. Optimal pure-Python but 10K Python calls vs zero.
- **Key insight**: Largest absolute gap (2.15x). The Python/C boundary is insurmountable.

### numpy__numpy-cb0d7cd -- numpy.strings.ljust (3.18x target)
- **Expert**: Single-pass C++ `string_pad` using `buffer_memset` + `buffer_memcpy`.
- **Model**: Composes 5 existing ufuncs: str_len, subtract, multiply, copy, add. Five passes with temp arrays.
- **Key insight**: Classic fusion gap -- composing primitives vs implementing a fused kernel.

### pandas-dev__pandas-2cdca01 -- Period.strftime (2.47x target)
- **Expert**: Bypass `c_strftime` in Cython with f-string formatting from pre-computed `npy_datetimestruct`.
- **Model**: Same strategy but extra function call frame, no pre-computed dts passthrough, redundant decomposition.
- **Key insight**: 8/8 pass tests, 7/8 achieve speedup, 0/8 match expert. Models know WHAT but not HOW at Cython level.

### pandas-dev__pandas-bfaf917 -- maybe_sequence_to_range (2.67x target)
- **Expert**: 73-line Cython loop with `boundscheck=False`. Short-circuits at first mismatch. Zero allocations.
- **Model**: `np.diff()` + `.all()`. Computes all 999,999 differences even when answer is "no" at element 1.
- **Key insight**: Vectorization vs short-circuit tradeoff. Models stay in NumPy's paradigm even when a C loop is better.

### tornadoweb__tornado-4d4c1e0 -- Future.set_exception (2.40x target)
- **Expert**: Replace custom Future with `asyncio.Future` (C-implemented). 30-file refactor.
- **Model**: Most sophisticated pure-Python optimization seen: conditional `__slots__`, inlined methods, local var caching.
- **Key insight**: Model hits the Python optimization ceiling but can't break through it.

### pydantic__pydantic-addf1f9 -- BaseModel.__setattr__ (2.33x target)
- **Expert**: Memoized handler dispatch -- `__pydantic_setattr_handlers__` dict caches handler per attribute name. 1 dict lookup after first call.
- **Model**: Specialized closure with if/elif chain. Still 3-4 dict lookups per call.
- **Key insight**: Model reduces checks; expert caches the entire dispatch decision.

### tornadoweb__tornado-ac13ee5 -- Future.done (2.06x target)
- **Expert**: Same `asyncio.Future` substitution across 19 files.
- **Model**: `__slots__`, lazy callbacks, reordered fast-path checks on existing Future.
- **Key insight**: Model optimizes WITHIN the abstraction; expert REPLACES it.

### pydantic__pydantic-ac9e6ee -- TypeAdapter.validate_python (1.80x target)
- **Expert**: Upgrades pydantic-core to get Rust-implemented EnumValidator.
- **Model**: Optimizes Python `to_enum()`: cached dict lookups, identity checks, skipped `_missing_`. Gets within 90%.
- **Key insight**: Closest gap (0.19x). Clever Python can approach but not match a compiled-code upgrade.
