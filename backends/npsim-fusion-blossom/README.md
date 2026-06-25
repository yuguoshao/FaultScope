# npsim-fusion-blossom

Official NPSim fusion-blossom backend.

This package provides a minimal serial fusion-blossom MWPM adapter. It converts
NPSim graphlike DEM problem views into fusion-blossom solver state and exposes
the decoder through the NPSim native decoder PyCapsule ABI. It is not yet the
production parallel or streaming adapter.

Install locally during development:

```bash
.venv/bin/python -m pip install -e backends/npsim-fusion-blossom
```
