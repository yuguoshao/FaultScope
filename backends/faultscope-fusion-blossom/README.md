# faultscope-fusion-blossom

Official FaultScope fusion-blossom backend.

This package provides a minimal serial-solver beta fusion-blossom MWPM adapter. It
converts FaultScope graphlike detector error matrix problem views into
fusion-blossom solver state, safely compresses identical boundary and
two-detector parallel edges, and
exposes the decoder through the FaultScope native decoder PyCapsule ABI. It is not
yet the production partitioned or streaming adapter, and it does not support
erasure or dynamic weights. The development build uses fusion-blossom's compact
vertex/edge index mode and rejects graphs that exceed that backend index range.
The default integer conversion uses `weight_scale=10_000`. After scaling,
solver weights are normalized by their common even-preserving divisor,
preserving the integer MWPM objective while reducing solver weight magnitudes
when possible. Each decoder descriptor owns one mutex-protected solver state and
path cache that it reuses across calls. Collection parallelism comes from fresh,
independent worker descriptors rather than a backend-owned scheduler or state
pool. Set `NPSIM_FUSION_BLOSSOM_PROFILE=1` to print
per-batch native timing split into defect collection, solver clear, solver
growth, matching extraction, and correction application.

See `OPTIMIZATION_NOTES.md` for optimization experiments that were not kept in
the public backend path.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/faultscope-fusion-blossom
```
