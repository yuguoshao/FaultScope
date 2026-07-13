# faultscope-pymatching

Official optional FaultScope PyMatching backend.

This package builds a native decoder from FaultScope graphlike detector error
matrix problem views and exposes it through the FaultScope native decoder
PyCapsule ABI. It uses pinned
PyMatching sparse-blossom C++ source internally; it does not call the Python
`PyMatchingDecoder` hot path.

Dynamic native error views remain valid until the associated decoder state is
dropped. The backend retains every dynamic error message in state-owned storage
to keep concurrent callback consumption safe, so repeated errors retain memory
until state drop. ABI consumers should copy error text promptly.

Development install from the FaultScope repository root:

```bash
.venv/bin/python -m pip install -e backends/faultscope-pymatching --no-build-isolation
```

If the PyMatching source is already checked out, set
`NPSIM_PYMATCHING_SOURCE_DIR=/path/to/PyMatching` before building. Otherwise the
build script clones PyMatching `v2.4.0` into Cargo's build output directory.
