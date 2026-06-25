# npsim-fusion-blossom

Official NPSim fusion-blossom backend.

This package provides a minimal serial beta fusion-blossom MWPM adapter. It
converts NPSim graphlike DEM problem views into fusion-blossom solver state,
safely compresses identical two-detector parallel edges, and exposes the
decoder through the NPSim native decoder PyCapsule ABI. It is not yet the
production parallel or streaming adapter, and it does not support erasure or
dynamic weights.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/npsim-fusion-blossom
```
