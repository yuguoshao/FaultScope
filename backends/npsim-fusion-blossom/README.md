# npsim-fusion-blossom

Official NPSim fusion-blossom backend.

This package provides a minimal serial-solver beta fusion-blossom MWPM adapter. It
converts NPSim graphlike DEM problem views into fusion-blossom solver state,
safely compresses identical boundary and two-detector parallel edges, and
exposes the decoder through the NPSim native decoder PyCapsule ABI. It is not
yet the production partitioned or streaming adapter, and it does not support
erasure or dynamic weights. The development build uses fusion-blossom's compact
vertex/edge index mode and rejects graphs that exceed that backend index range.
The default integer conversion uses `weight_scale=10_000`. After scaling,
solver weights are normalized by their common even-preserving divisor,
preserving the integer MWPM objective while reducing solver weight magnitudes
when possible. Packed batch decoding reuses per-worker solver state and path
caches across calls.
Set `NPSIM_FUSION_BLOSSOM_THREADS=<n>` to cap the packed batch worker count
when diagnosing host-specific scheduling behavior; by default the backend uses
the available native parallelism. Set `NPSIM_FUSION_BLOSSOM_PROFILE=1` to print
per-batch native timing split into defect collection, solver clear, solver
growth, matching extraction, and correction application.
`NPSIM_FUSION_BLOSSOM_BLOCK_ROWS=<n>` overrides the packed-row scheduler block
size for load-balancing experiments.

See `OPTIMIZATION_NOTES.md` for optimization experiments that were not kept in
the public backend path.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/npsim-fusion-blossom
```
