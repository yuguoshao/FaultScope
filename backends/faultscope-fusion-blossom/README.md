# faultscope-fusion-blossom

Official FaultScope fusion-blossom backend.

This package provides a minimal serial-solver beta fusion-blossom MWPM adapter. It
converts FaultScope graphlike detector error problems into fusion-blossom solver
state. Current FaultScope problems use the zero-copy native graphlike
construction PyCapsule ABI v1; older or third-party objects retain the Python
attribute compatibility path. The adapter safely compresses identical boundary
and two-detector parallel edges and exposes the decoder through the separate
FaultScope native decoder PyCapsule ABI V4. The factory declares `[Packed]` and
workers use one tagged callback; hotspot edge events remain in FaultScope's
independent attribution sidecar. It is not
yet the production partitioned or streaming adapter, and it does not support
erasure or dynamic weights. The development build uses fusion-blossom's compact
vertex/edge index mode and rejects graphs that exceed that backend index range.
`NativeFusionBlossomDecoder.from_graphlike_problem(...)` accepts a precompiled
view. Components that share a canonical `dem_edge_index` remain an uncorrelated
matching approximation and are never sampled by this package. Construction
summaries distinguish canonical `dem_edge_count`, pre-merge
`graphlike_edge_count`, and post-merge `solver_edge_count`.
Dynamic native error views remain valid until the associated decoder state is
dropped. The backend retains every dynamic error message in state-owned storage
to keep concurrent callback consumption safe, so repeated errors retain memory
until state drop. ABI consumers should copy error text promptly.
The default integer conversion uses `weight_scale=10_000`. After scaling,
solver weights are normalized by their common even-preserving divisor,
preserving the integer MWPM objective while reducing solver weight magnitudes
when possible. The ABI V4 factory shares immutable graph/path data, while each
private worker owns one exclusive mutable solver. Collection caches workers per
thread and task; the backend needs no solver mutex, scheduler, or state pool.
Set `NPSIM_FUSION_BLOSSOM_PROFILE=1` to print
per-batch native timing split into defect collection, solver clear, solver
growth, matching extraction, and correction application.

See `OPTIMIZATION_NOTES.md` for optimization experiments that were not kept in
the public backend path.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/faultscope-fusion-blossom
```
