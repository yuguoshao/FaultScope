# faultscope-fusion-blossom Optimization Notes

This file records optimization work, discarded attempts, and observations for the
`faultscope-fusion-blossom` backend. It is meant to prevent repeated experiments and
to keep future performance work focused on the real bottlenecks.

## Current Position

`faultscope-fusion-blossom` is a minimal serial beta adapter. It validates the
post-install native backend path and uses the real upstream fusion-blossom
serial solver, but it is not yet a production-tuned backend.

The current target comparison is:

```bash
.venv/bin/python benchmarks/surface_code_decoder_performance.py \
  --distances 5 7 9 12 --rates 0.01 --shots 50000 \
  --paths faultscope-dem-pymatching-native,faultscope-dem-fusion-blossom \
  --split-native-baseline --same-seed-across-paths
```

The PyCapsule fast path is working: `python_decode_calls` should stay at `0`.
Remaining performance gaps are dominated by solver work, not Python callback
overhead.

## Improvements Kept

### External Native Backend Path

- The backend is installed as a separate package and discovered through the
  `faultscope.native_decoders` entry point.
- Decoder objects expose a PyCapsule native handle to FaultScope core.
- The hot path uses host-allocated correction buffers and does not create
  Python batch dictionaries, correction dictionaries, or NumPy syndrome
  matrices.
- FaultScope calls the backend through the native decoder ABI and keeps
  `python_decode_calls == 0` in the fast path.

### Real Serial Solver Adapter

- Construction compiles a graphlike FaultScope DEM problem into a fusion-blossom
  solver initializer.
- Detector ids map to real fusion-blossom vertices.
- One-detector DEM edges create virtual boundary vertices.
- Two-detector DEM edges create graph edges.
- Solver paths are mapped back to DEM edge effects and observable corrections.

### Safe Parallel-Edge Handling

- Same endpoint and same logical effect parallel DEM edges are merged using the
  independent odd-parity probability.
- Same endpoint with different logical effects is rejected. This avoids silently
  choosing an incorrect observable correction.
- One-detector boundary alternatives remain valid because they map to distinct
  virtual boundary vertices.

### General Hot-Path Work

- Per-worker solver state and scratch buffers are reused across batches.
- `SyndromePattern` and defect buffers are reused instead of rebuilt from
  scratch for every shot.
- Panic handling is done once per worker instead of around every small decode
  block.
- Profile environment lookup is hoisted out of inner loops.
- Profiling splits the backend work into:
  - `collect_s`
  - `clear_s`
  - `grow_s`
  - `extract_s`
  - `apply_s`
  - `total_s`

### Useful Diagnostics

The following environment variables are still useful for controlled experiments:

- `NPSIM_FUSION_BLOSSOM_THREADS`
- `NPSIM_FUSION_BLOSSOM_BLOCK_ROWS`
- `NPSIM_FUSION_BLOSSOM_PROFILE`

Earlier experiments also used `NPSIM_FUSION_BLOSSOM_BOUNDARY_VIRTUALS`,
`NPSIM_FUSION_BLOSSOM_PREDICTION`, and diagnostic benchmark paths such as
`faultscope-dem-fusion-blossom-*`. Those public hooks were removed from the cleaned
backend path; keep the observations below as history unless new evidence
justifies reintroducing them.

## Attempts Not Kept As Defaults

### Shared Boundary Vertices

`shared-boundary` was faster for some `d=5/7/9` runs, but regressed at `d=12`.
It remains a diagnostic option only. Do not make it default without repeated
large-distance evidence.

### Lower Weight Scales

`weight_scale=100` can reduce integer weights and sometimes speeds up growth,
but it changes quantization and did not consistently win on same-seed runs.

`weight_scale=1000` improved some `d=5` runs, but was not a stable improvement
for `d=7/9/12`.

`weight_scale=1_000_000` preserves more precision but is not a performance
default.

Lower scales should be treated as approximate solver modes and validated for
logical-loss behavior before being recommended.

### Subgraph Prediction Path

The subgraph extraction path was correct, but 50k-shot runs showed it was slower
at `d=5/9/12`. It remains useful only as a diagnostic comparison.

### Fixed Thread Counts

Manual `t1`, `t4`, and `t8` caps were tried. `t1` and `t4` were slower in the
target benchmark, while `t8` was noisy and not consistently better. The default
continues to use available host parallelism, with an environment override for
experiments.

### Fixed Block Sizes

Different distances preferred different block sizes, and timing variance was
high. Keep `NPSIM_FUSION_BLOSSOM_BLOCK_ROWS` as an experiment knob instead of a
hardcoded tuning rule.

### `max_tree_size`

Small tree-size caps can severely regress performance or trigger very slow
solver behavior. Treat this as experimental only.

### Upstream Parallel Solver

Generic coordinate partitioning was investigated.

- Spatial-axis partitions were often rejected by upstream edge-locality checks.
- Time-axis partitions could construct, but were much slower than the serial
  solver for `d=7/9/12`.

Do not switch to the parallel solver without a geometry-aware partitioner with
buffer layers and a clear benchmark win.

### Legacy Solver Variants

`LegacySolverSerial` showed small gains on tiny cases but regressed at larger
distances. It was removed.

`SolverDualParallel` was much slower in the tested configuration. It was
removed.

### Coordinate Vertex Ordering

Coordinate-based vertex reordering produced small or noisy effects: slight
`d=7` gain, flat `d=9`, tiny `d=12` gain, and worse `d=15` behavior. It was
removed to avoid extra complexity.

### Upstream Unsafe Pointer Features

The upstream `dangerous_pointer` / `unsafe_pointer` path requires nightly Rust
because it depends on `#![feature(get_mut_unchecked)]`. It is not viable for the
stable build.

### `i32_weight`

The `i32_weight` feature built, but it did not produce a material target-scale
speedup and narrows the available weight range. It was reverted.

## Observations

### Main Bottleneck

Profiling shows the dominant time is fusion-blossom growth, not FaultScope data
movement.

For example, on a `d=9`, 1024-shot profile with 128-shot blocks, typical block
timing looked like:

- `collect_s`: about `0.00007s`
- `clear_s`: about `0.00001s`
- `grow_s`: about `0.08s` to `0.10s`
- `extract_s`: about `0.001s`
- `apply_s`: usually about `0.003s` to `0.004s`

This means optimizing Python interaction, observable bit application, or basic
packed-mask scanning will not close the main gap by itself.

### Native PyMatching Remains The Strong Baseline

After the current native PyMatching work, typical 50k-shot results are in this
range:

| distance | native PyMatching | fusion-blossom |
| --- | ---: | ---: |
| 5 | 400k+ shots/s | 220k-270k shots/s |
| 7 | 120k+ shots/s | 35k-50k shots/s |
| 9 | around 50k shots/s | 14k-18k shots/s |
| 12 | around 17k shots/s | 4k-5k shots/s |

Exact values are noisy and machine-dependent. Use repeated runs and medians for
decisions.

### Seed Policy Matters

The benchmark offsets seeds by path unless `--same-seed-across-paths` is used.
Use same-seed mode when comparing logical loss or debugging correction
differences.

### Weight Scale Behavior

The default weight scale is currently conservative. At `p=0.01`, surface-code
DEM weights commonly land around the tens of thousands. GCD reduction did not
meaningfully reduce current target graphs.

Do not blindly divide all weights by two. Upstream debug assertions assume even
remaining lengths in parts of the dual module, and simple scaling changes can
violate solver expectations or alter correction choices.

### Upstream Revision

The backend is pinned to upstream fusion-blossom commit
`c536a4b...`. At the time of inspection, upstream `HEAD`/`main` pointed to the
same commit, so there was no newer public revision to pull for performance.

## Recommended Next Steps

1. Focus on solver-growth reduction rather than Python boundary work.
2. If testing lower weight scales, treat them as approximate modes and validate
   both performance and logical-loss behavior with same-seed runs.
3. Build a geometry-aware partitioner only if it can include safe buffer layers
   and fusion edges. The generic coordinate partition experiment was not enough.
4. Add a repeated-run benchmark protocol that reports medians and variance.
5. Keep native PyMatching as the primary performance baseline for graphlike
   surface-code decoding.

Useful profiling command:

```bash
NPSIM_FUSION_BLOSSOM_PROFILE=1 .venv/bin/python benchmarks/surface_code_decoder_performance.py \
  --distances 9 --rates 0.01 --shots 1024 \
  --paths faultscope-dem-fusion-blossom --split-native-baseline
```

## Do Not Revive Without New Evidence

Avoid reintroducing these changes unless a new benchmark clearly supports them:

- shared boundary as the default
- very low default weight scale
- subgraph prediction as the default
- fixed thread-count defaults
- fixed block-size defaults
- small `max_tree_size` caps
- generic coordinate partitioning for the parallel solver
- `LegacySolverSerial`
- `SolverDualParallel`
- coordinate vertex ordering
- nightly-only unsafe pointer features
- `i32_weight`
