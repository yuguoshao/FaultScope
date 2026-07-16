# faultscope-pymatching

Official optional FaultScope PyMatching backend.

This package builds a native decoder from FaultScope graphlike detector error
problems. Current FaultScope problems cross the extension boundary through the
zero-copy native graphlike construction PyCapsule ABI v1; older or third-party
problem objects retain the Python attribute compatibility path. The resulting
decoder is exposed through the separate FaultScope native decoder PyCapsule ABI
v3. The public Python object is a factory handle; each private worker owns an
exclusive native solver. It uses pinned
PyMatching sparse-blossom C++ source internally; it does not call the Python
`PyMatchingDecoder` hot path. Independent parallel edges are merged only when
they have the same detector endpoints and logical-observable support. The
adapter rejects conflicting logical effects because PyMatching cannot preserve
both effects on one simple-graph edge.

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
