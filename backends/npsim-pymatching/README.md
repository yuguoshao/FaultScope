# npsim-pymatching

Official optional NPSim PyMatching backend.

This package builds a native decoder from NPSim graphlike DEM problem views and
exposes it through the NPSim native decoder PyCapsule ABI. It uses pinned
PyMatching sparse-blossom C++ source internally; it does not call the Python
`PyMatchingBatchDecoder` hot path.

Development install from the NPSim repository root:

```bash
.venv/bin/python -m pip install -e backends/npsim-pymatching --no-build-isolation
```

If the PyMatching source is already checked out, set
`NPSIM_PYMATCHING_SOURCE_DIR=/path/to/PyMatching` before building. Otherwise the
build script clones PyMatching `v2.4.0` into Cargo's build output directory.
