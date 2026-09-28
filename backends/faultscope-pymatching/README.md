# faultscope-pymatching

Official optional FaultScope PyMatching backend.

This package builds a native decoder from FaultScope graphlike detector error
problems. Current FaultScope problems cross the extension boundary through the
zero-copy native graphlike construction PyCapsule ABI v1; older or third-party
problem objects retain the Python attribute compatibility path. The resulting
decoder is exposed through the separate FaultScope native decoder PyCapsule ABI
V4. The factory declares `[Packed]`; each private worker owns an exclusive
native solver and receives all batches through one tagged callback. It uses pinned
PyMatching sparse-blossom C++ source internally; it does not call the Python
`PyMatchingDecoder` hot path. Independent parallel edges are merged only when
they have the same detector endpoints and logical-observable support. The
adapter rejects conflicting logical effects because PyMatching cannot preserve
both effects on one simple-graph edge.

`NativePyMatchingDecoder.from_graphlike_problem(...)` accepts a precompiled
view, including validated multi-component hints that share one canonical
`dem_edge_index`. These components are an uncorrelated matching approximation;
canonical DEM sampling remains outside this backend. Construction summaries
report canonical `dem_edge_count`, pre-merge `graphlike_edge_count`, and
post-merge `solver_edge_count` separately.

Hotspot attribution is not part of the decoder input. FaultScope records the
edge-event trace in its own sidecar while this backend receives only packed
detector syndrome rows.

Dynamic native error views remain valid until the associated decoder state is
dropped. The backend retains every dynamic error message in state-owned storage
to keep concurrent callback consumption safe, so repeated errors retain memory
until state drop. ABI consumers should copy error text promptly.

## Installation

Install FaultScope first using the
[installation guide](../../docs/getting_started.md#installation).
With the same virtual environment active, install the native backend from PyPI:

```bash
pip install faultscope_pymatching
python -I -m faultscope.backends status
```

Prebuilt wheels require no Rust or C++ compiler. The Python `pymatching` package
is separate from this native backend and does not provide
`NativePyMatchingDecoder`.

## Build from Source for Development

For backend development or a platform without a compatible wheel, use Rust, Git,
and a C++20 compiler. From the FaultScope repository root with your virtual
environment active and the base package installed:

```bash
python -m pip install "maturin>=1.7,<2"
python -m pip install -e backends/faultscope-pymatching --no-build-isolation
python -I -m faultscope.backends status
```

If the PyMatching source is already checked out, set
`NPSIM_PYMATCHING_SOURCE_DIR=/path/to/PyMatching` before building. Otherwise the
build script clones PyMatching `v2.4.0` into Cargo's build output directory.
