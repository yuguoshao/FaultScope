# Decoder Development

This page is for developers who want to prototype a decoder in Python or add a
native decoder backend to NPSim. For user-facing workflows, see the
[User Guide](user_guide.md). For exact signatures, see the
[API Reference](api_reference.md).

## Mental Model

Decoder integration has two phases:

```text
Circuit -> DetectorErrorModel -> decoding problem -> decoder object
```

and then, on the hot path:

```text
packed detector syndrome masks -> decoder -> packed observable correction masks
```

A decoder can and usually should use circuit-derived information, but only
during construction. The decoder object should compile that information into
its own graph, matrices, weights, or lookup tables. Batch decoding should not
read the original circuit or DEM again.

The runtime expects detector and observable ids to be stable:

- `detector_ids` define the syndrome order consumed by the decoder.
- `observable_ids` define the logical correction ids that the decoder may
  return.
- A correction mask bit `k` is the decoder's predicted logical correction for
  shot `k`.

The default residual logical loss is:

```text
sampled observable mask XOR decoder correction mask
```

over all declared logical observables.

## Python Prototype Decoders

Python decoders are the quickest way to validate an algorithm. They use the
compatibility path, where NPSim materializes a Python batch object and calls
`decode_batch_masks(batch)`.

Recommended shape:

```python
from npsim.runtime import generate_native_dem


class MyDecoder:
    @classmethod
    def from_circuit(cls, circuit, detectors=None, observables=None):
        dem = generate_native_dem(
            circuit,
            detectors=detectors,
            observables=observables,
        )
        return cls.from_dem(dem)

    @classmethod
    def from_dem(cls, dem):
        # Use graphlike problems for MWPM/fusion-blossom-style decoders.
        problem = dem.compile_graphlike_problem()

        # Use this instead for BP+OSD/LDPC-style decoders:
        # problem = dem.compile_binary_linear_problem()

        return cls(
            detector_ids=problem.detector_ids,
            observable_ids=problem.observable_ids,
            problem=problem,
        )

    def __init__(self, detector_ids, observable_ids, problem):
        self.detector_ids = tuple(detector_ids)
        self.observable_ids = tuple(observable_ids)
        self.problem = problem

        # Compile the problem into the decoder's internal representation here.
        # For example: matching graph, sparse parity-check matrix, weights, etc.

    def decode_batch_masks(self, batch):
        syndrome_masks = [
            batch.detectors.get(detector_id, 0)
            for detector_id in self.detector_ids
        ]

        # Replace this with the actual algorithm.
        # Return one packed correction mask per corrected observable id.
        correction_for_first_observable = 0

        return {
            self.observable_ids[0]: correction_for_first_observable,
        }
```

Use the decoder like any other Python decoder:

```python
decoder = MyDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
)

result = sampler.estimate(
    shots=10_000,
    decoder=decoder,
)
```

`batch.detectors` is a mapping from detector id to a Python integer containing
packed syndrome bits. The least significant bit is shot 0. The return value
must be `dict[int, int]`, mapping observable id to a packed correction mask.

Python decoders are ideal for correctness prototypes and small experiments.
They are not the final high-performance path, because detector masks and
correction masks cross the Python boundary.

## Getting Circuit And DEM Information

When starting from a circuit, first generate a detector error model:

```python
from npsim.runtime import generate_native_dem

dem = generate_native_dem(
    circuit,
    detectors=detectors,
    observables=observables,
)
```

The DEM can then be compiled into decoder-ready views:

```python
indexed = dem.compile_indexed()
graphlike = dem.compile_graphlike_problem()
binary = dem.compile_binary_linear_problem()
```

Use the view that matches the decoder family:

| View | Intended backend |
| --- | --- |
| `IndexedDem` | General DEM indexing, edge metadata, and stable order inspection |
| `GraphlikeDecodingProblem` | MWPM-style decoders, including future fusion-blossom adapters |
| `BinaryLinearDecodingProblem` | BP+OSD/LDPC-style decoders using sparse binary `H` and `F` |

`GraphlikeDecodingProblem` rejects hyperedges and undetectable pure logical
edges, because matching-style backends cannot infer those errors from syndrome
data. `BinaryLinearDecodingProblem` does not require graphlike edges; each DEM
edge is an independent binary error variable.

The problem views expose ids, counts, `edge_summary`, probabilities, weights,
and sparse binary matrix entries for construction-time inspection. They
intentionally do not expose `to_numpy_*` hot-path helpers. A native decoder
should consume the native problem representation instead of moving batch data
through Python.

## Native Decoder Backends

Native decoders are Python-owned handles around Rust decoder objects. They are
the intended path for production backends because the batch syndrome masks,
correction masks, default residual loss, and hotspot aggregation stay in native
memory.

The recommended ownership model is:

```text
Python passes circuit / DEM / options
        -> PyO3 factory reads those Python objects
        -> Rust compiles a native decoder backend
        -> Python receives an opaque decoder handle
        -> estimate(..., decoder=decoder) uses the native fast path
```

Python is responsible for selecting and configuring the decoder. Rust is
responsible for constructing the backend and running batch decode. After the
decoder is constructed, Python should not participate in the syndrome or
correction-mask hot path.

### Python Input, Rust Construction

A native decoder should expose constructors that accept Python objects:

```python
decoder = MyNativeDecoder.from_dem(
    dem,
    option_a=...,
    option_b=...,
)

decoder = MyNativeDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
    option_a=...,
)
```

`from_circuit(...)` is a convenience constructor. It should generate or compile
a DEM first, then delegate to the same Rust construction path as
`from_dem(...)`.

The PyO3 implementation should convert Python input into Rust core structures
once:

```rust
#[pyclass(name = "MyNativeDecoder", module = "npsim._npsim_native")]
pub struct PyMyNativeDecoder {
    inner: Arc<dyn npsim_core::NativeBatchDecoder>,
}

#[pymethods]
impl PyMyNativeDecoder {
    #[staticmethod]
    pub fn from_dem(
        py: Python<'_>,
        dem: &Bound<'_, PyDetectorErrorModel>,
        option_a: Option<f64>,
    ) -> PyResult<Self> {
        let core_dem = dem.borrow().to_core_dem(py)?;

        let problem = core_dem
            .compile_graphlike_problem()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let backend = MyNativeDecoder::from_graphlike_problem(
            problem,
            option_a,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(Self {
            inner: Arc::new(backend),
        })
    }
}
```

For BP+OSD/LDPC-style decoders, use
`core_dem.compile_binary_linear_problem()` instead. For a decoder that needs
more DEM metadata, use `compile_indexed()` or extend the construction-time DEM
view. Do not add per-batch circuit or DEM reads to `decode_batch(...)`.

The Rust backend stores all construction-time information:

```rust
pub struct MyNativeDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    // Compiled graph / H matrix / F matrix / weights / LLR / backend state.
}
```

The Python object only owns the handle. It does not own batch detector masks or
correction masks during `estimate(...)`.

The core Rust contract is `NativeBatchDecoder`:

```rust
use npsim_core::{
    CorrectionMaskBatch,
    DetectorMaskBatchView,
    Mask,
    NativeBatchDecoder,
    NpResult,
};

pub struct MyNativeDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    // Store the compiled graph, matrix, weights, or backend object here.
}

impl NativeBatchDecoder for MyNativeDecoder {
    fn name(&self) -> &'static str {
        "my-decoder"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> NpResult<CorrectionMaskBatch> {
        let words = npsim_core::word_count(detectors.shots);

        // detectors.masks are ordered exactly like self.detector_ids().
        // Run the backend and fill one Mask per corrected observable id.
        let correction_for_observable_0 = Mask::zero(words);

        CorrectionMaskBatch::new(
            vec![self.observable_ids[0]],
            vec![correction_for_observable_0],
            detectors.shots,
        )
    }
}
```

The PyO3 layer should expose a Python handle with constructors such as:

```python
decoder = MyNativeDecoder.from_dem(dem)
# or
decoder = MyNativeDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
)
```

Internally, those constructors should compile the DEM/problem once and store an
`Arc<dyn NativeBatchDecoder + Send + Sync>`. The runtime recognizes native
handles explicitly; a Python subclass that only implements
`decode_batch_masks(batch)` remains a Python decoder and does not enter the
native fast path.

Native backends are discovered through built-in handles and post-install plugin
entry points. The V1 plugin contract is intentionally scoped to NPSim-owned
backend packages:

```text
entry point group: npsim.native_decoders
ABI name: npsim.native_decoder_plugin.v1
```

The plugin package returns decoder classes that construct native handles. The
runtime still recognizes only native handles; a Python subclass that implements
`decode_batch_masks(batch)` remains a slow-path Python decoder.

Official backend installation metadata lives in the built-in catalog. Each
entry records the backend name, backend package, proxy class name, target
problem view, source repository, default revision, installability, and a short
description. The catalog includes `pymatching` and `fusion-blossom` for
graphlike MWPM-style decoding and reserves `bposd` for binary-linear
BP+OSD/LDPC decoding.

Python can inspect compiled native backend names:

```python
from npsim.decoders import available_native_decoders

print(available_native_decoders())
```

The default build currently exposes:

```text
no-correction
graphlike-detector-copy
```

These are smoke-test and template backends, not production decoders.
Post-install backend packages may add more names. NPSim never clones, builds, or
installs backend code during `import npsim` or `estimate(...)`; installation is
an explicit command.

Python can use either friendly proxy classes or a generic resolver:

```python
from npsim.decoders import (
    NativeFusionBlossomDecoder,
    NativePyMatchingDecoder,
    create_native_decoder,
    get_native_decoder_class,
)

decoder = NativePyMatchingDecoder.from_dem(dem)
decoder = create_native_decoder("pymatching", dem=dem)
decoder = NativeFusionBlossomDecoder.from_dem(dem)
decoder = create_native_decoder("fusion-blossom", dem=dem)
Decoder = get_native_decoder_class("fusion-blossom")
```

### Example Native Backend

`NativeGraphlikeDetectorCopyDecoder` is an in-tree example backend that shows
the intended construction pattern without introducing third-party dependencies.
It is useful for testing the native path and for copying when adding a real
backend.

```python
from npsim.decoders import NativeGraphlikeDetectorCopyDecoder

decoder = NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

decoder = NativeGraphlikeDetectorCopyDecoder.from_circuit(
    circuit,
    detectors=detectors,
    observables=observables,
)
```

Construction happens in Rust:

1. The PyO3 factory receives the Python `DetectorErrorModel` or `Circuit`.
2. Rust compiles a `GraphlikeDecodingProblem`.
3. The backend finds single-detector, single-observable graphlike edges.
4. The Python object stores only a native decoder handle.

On the hot path, it copies the mapped detector syndrome mask into the
corresponding observable correction mask. Observables without a unique mapped
edge receive zero correction. If more than one single-detector candidate edge
maps to the same observable, construction fails with `ValueError`.

This backend is intentionally not a general decoder. It does not replace
fusion-blossom, MWPM, BP+OSD, or LDPC decoding. Its purpose is to demonstrate
how Python can pass construction information while Rust owns the native decode
backend.

### Post-Install Native Backends

Optional native backends are installed after the core NPSim package. Users can
inspect backend status:

```bash
python -m npsim.backends status
```

Installation helpers are uniform for catalog entries:

```bash
python -m npsim.backends install fusion-blossom --dry-run
python -m npsim.backends install pymatching --dry-run
python -m npsim.backends install bposd --dry-run
```

The command reserves the install workflow and prints the clone/build/install
steps. Until official backend packages are available, non-dry-run installation
fails with a clear package-unavailable or reserved-backend message. `bposd` is
currently a catalog reservation only; it does not imply a BP+OSD backend exists.

When the backend package is missing, the public proxy remains importable but
construction raises an install hint:

```python
from npsim.decoders import NativeFusionBlossomDecoder, NativePyMatchingDecoder

decoder = NativeFusionBlossomDecoder.from_dem(dem)  # raises until installed
decoder = NativePyMatchingDecoder.from_dem(dem)  # raises until installed
```

The post-install plugin ABI is not a general third-party stable ABI. It is a
versioned contract for official NPSim backend packages so the core package can
reject mismatched backend builds before any hot-path decoding begins.

### Official Backend Packages

The repository includes official optional backend packages at:

```text
backends/npsim-pymatching/
backends/npsim-fusion-blossom/
```

Install them in editable mode during development from an activated project
virtual environment, or with `.venv/bin` explicitly on `PATH`:

```bash
.venv/bin/python -m pip install -e backends/npsim-pymatching --no-build-isolation
.venv/bin/python -m pip install -e backends/npsim-fusion-blossom
```

The package declares:

```toml
[project.entry-points."npsim.native_decoders"]
pymatching = "npsim_pymatching:backend_manifest"
fusion-blossom = "npsim_fusion_blossom:backend_manifest"
```

Its manifest returns the current NPSim native decoder plugin ABI, package
metadata, and one or more decoder classes. The class implements
`from_dem(...)` and `from_circuit(...)`; construction compiles the DEM to a
`GraphlikeDecodingProblem`, passes that metadata to the package's Rust/PyO3
extension, and stores external native decoder state in a PyCapsule.

The PyMatching backend links pinned PyMatching sparse-blossom C++ source in the
`npsim-pymatching` package. It does not call the Python
`PyMatchingBatchDecoder` hot path and does not depend on the PyPI wheel exposing
a stable native SDK. The backend uses the same native PyCapsule boundary as the
other official packages: Python passes construction metadata, while hot-path
`DetectorMaskBatchView` and `CorrectionMaskBatch` buffers stay native.

The current backend is a minimal serial beta fusion-blossom MWPM adapter. It
maps NPSim detector indices to fusion-blossom vertices, converts graphlike DEM
edges to weighted solver edges, runs the serial solver for each shot, and maps
the selected edge paths back to observable correction masks. It safely
compresses identical two-detector parallel edges before constructing the solver
graph. It is not yet the production parallel or streaming adapter.

The backend decoder object exposes:

```python
decoder.__npsim_native_decoder_capsule__()
decoder.name
decoder.detector_ids
decoder.observable_ids
decoder.edge_count
decoder.solver_vertex_count
decoder.solver_edge_count
decoder.boundary_vertex_count
decoder.build_summary
decoder.decode_batch_masks(batch)  # debug fallback only
```

NPSim only calls the capsule method on the native fast path. The plugin ABI is
host-allocated for hot outputs: NPSim allocates correction mask words for the
decoder's declared observables, and the backend writes into those buffers.
Backend packages must not allocate correction masks and ask NPSim to free them
across the dynamic-library boundary.

### Fusion-Blossom Adapter Status

Fusion Blossom is a MWPM decoder route for QEC. The
[paper](https://arxiv.org/abs/2305.08307) describes a parallel MWPM decoder and
stream decoding support. The public
[repository](https://github.com/yuewuo/fusion-blossom) presents the project as
a fast MWPM solver for QEC, ships Rust code plus a Python binding, and currently
declares Rust package version `0.2.13`. NPSim pins the backend dependency to a
concrete git revision because that version is not available from crates.io. The
public Rust source exposes types such as `SolverInitializer` and
`SyndromePattern` and helper functions such as `fusion_mwpm(...)` and
`detailed_matching(...)`.

NPSim reserves the public constructor through a post-install proxy:

```python
from npsim.decoders import NativeFusionBlossomDecoder

decoder = NativeFusionBlossomDecoder.from_dem(dem)
decoder = NativeFusionBlossomDecoder.from_circuit(circuit)
```

The default package does not ship the fusion-blossom solver. After
`npsim-fusion-blossom` is installed, the proxy delegates construction to that
package while preserving the native fast path.

The implemented minimal beta adapter is:

1. Compile `DetectorErrorModel` to `GraphlikeDecodingProblem`.
2. Map each NPSim detector index to a fusion-blossom vertex.
3. Convert one-detector DEM edges to boundary or virtual-vertex edges.
4. Merge two-detector parallel DEM edges only when they share both endpoints
   and the same fault-observable set. The merged probability is the independent
   odd-parity probability.
5. Reject two-detector parallel edges with different fault-observable sets,
   because choosing one correction would be ambiguous.
6. Preserve each solver edge's contributing DEM edge indices and
   fault-observable indices so the solver prediction
   can be converted back into observable correction masks.
7. Convert each hot-path `DetectorMaskBatchView` shot into the solver syndrome
   representation without touching Python.
8. Return a checked `CorrectionMaskBatch`.

The construction summary is intentionally lightweight and safe to inspect from
Python:

```python
summary = decoder.build_summary
summary["dem_edge_count"]
summary["solver_edge_count"]
summary["merged_parallel_edge_count"]
summary["edges"][0]["dem_edge_indices"]
summary["edges"][0]["fault_observables"]
```

The remaining productionization items are solver reuse, parallel/streaming
execution, erasure/dynamic weights, compression for ambiguous parallel logical
effects, and large-scale performance tuning.

The primary beta evaluation entry point is the surface-code decoder performance
benchmark:

```bash
.venv/bin/python benchmarks/surface_code_decoder_performance.py --distances 3 5 7 --shots 10000
```

It compares Stim DEM + PyMatching, NPSim DEM + Python PyMatching, NPSim DEM +
native PyMatching, and NPSim DEM + fusion-blossom native decoding when the
optional backend packages are installed. The benchmark uses a local graphlike
Stim DEM converter that splits separator groups into NPSim DEM edges for native
paths; this does not change the threshold benchmark. It reports construction
time, sampling time where separable, native estimate time, solver-edge
metadata, merged parallel edges, and whether native paths stayed out of Python
callbacks.

For native backend diagnosis, add `--split-native-baseline`. The native rows
then report `sample_s` as a no-decoder native mean-loss baseline and
`decode_or_estimate_s` as the additional decoder cost. This makes it clear
whether a gap is in NPSim sampling/aggregation or in the backend decode loop.

## Validation And Performance Rules

NPSim validates native decoder output before using it:

- correction `shots` must match the sampled batch;
- correction mask word count must match the shot count;
- correction observable ids must be unique;
- correction observable ids must be declared by the decoder;
- missing observable corrections are treated as all-zero correction masks.

The native fast path is used only when:

- `decoder` is a native decoder handle;
- no `loss_mask_fn` is supplied;
- no `correction_mask_fn` is supplied.

If a Python loss or correction callback is supplied, NPSim uses the
compatibility path and may call `decoder.decode_batch_masks(batch)`. This is
useful for debugging and comparison, but it moves batch data through Python.

`materialize_dem=False` creates a light DEM sampler without full DEM metadata.
That mode can sample, but APIs that need DEM metadata, including hotspot result
construction and decoder problem construction, require a materialized DEM.

## Development Checklist

For a new decoder family:

1. Prototype with a Python class implementing `from_dem(...)` and
   `decode_batch_masks(batch)`.
2. Choose the construction view: graphlike for MWPM/fusion-blossom-style
   decoders, binary linear for BP+OSD/LDPC-style decoders.
3. Add or update an official backend catalog entry with package name, problem
   kind, source repo, installability, and proxy class name.
4. Implement an official backend package that exposes a
   `npsim.native_decoders` entry point manifest with the matching ABI.
5. Expose a friendly proxy class when the backend should be importable from
   `npsim.decoders`.
6. Test both paths: Python compatibility behavior and native no-callback fast
   path.
