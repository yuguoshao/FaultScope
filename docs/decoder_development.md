# Decoder Development

This page is for developers who want to prototype a decoder in Python or add a
native decoder backend to FaultScope. For user-facing workflows, see the
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
during construction. In detector error model language, construction consumes
the detector error matrix \(H=D\Omega\), logical fault matrix, probabilities,
weights, coordinates, or edge metadata and compiles them into its own graph,
matrices, weights, or lookup tables. Batch decoding should not read the original
circuit or DEM again.

The runtime expects detector and observable ids to be stable:

- `detector_ids` define the syndrome order consumed by the decoder and must be
  unique within that decoder's input layout.
- `observable_ids` must be the sampler's complete canonical logical-id sequence;
  native decoders may not omit, add, or reorder ids.
- A correction mask bit `k` is the decoder's predicted logical correction for
  shot `k`.

The default residual logical loss is:

```text
sampled observable mask XOR decoder correction mask
```

over all declared logical observables.

## Python Prototype Decoders

Python decoders are the quickest way to validate an algorithm. They use the
compatibility path, where FaultScope materializes a Python batch object and calls
`decode_batch_masks(batch)`.

Recommended shape:

```python
from faultscope.runtime import generate_native_dem


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
        # For example: matching graph, sparse detector error matrix H, weights, etc.

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
They are not the final high-performance path, because detector syndrome masks and
correction masks cross the Python boundary.

Forward-circuit decoders that need only selected measurement records should
request them once with `batch.measurement_masks(keys)`. This performs one
Python/Rust boundary call and materializes only those packed integer masks:

```python
def decode_batch_masks(self, batch):
    measurements = batch.measurement_masks(self.measurement_keys)
    masks = tuple(measurements[key] for key in self.measurement_keys)
    # Reuse masks and prefer whole-mask bitwise operations. Do not read
    # batch.measurements or shift a growing Python integer in a shot loop.
    ...
```

The existing `batch.measurements` mapping and `batch.measurement_mask(key)`
accessor remain available. The bulk selector is preferred when a decoder knows
its required keys in advance; ordinary Python batch-like objects without this
method remain supported by decoders that fall back to reading `measurements`
once.

### Stable collection identity

Native decoder objects passed to `CollectionTask.decoder` must implement a
stable identity protocol:

```python
def strong_id_payload(self):
    return {
        "backend": "my-decoder",
        "backend_version": "1.2.3",
        "implementation_version": 1,
        "detector_ids": list(self.detector_ids),
        "observable_ids": list(self.observable_ids),
        "parameters": dict(self.normalized_options),
        "solver": self.canonical_solver_problem,
    }
```

The return value must be a JSON-serializable mapping derived from effective
decoder state, not `repr(...)` or object identity. Include every parameter that
can alter corrections, the detector and observable layouts, the effective
graph or matrix, backend/ABI version information, and ordered payloads for all
children of a composite decoder. Equivalent mapping insertion orders must
produce equivalent payloads. Increment `implementation_version` when bundled
decoder behavior changes without an otherwise visible configuration change.

Collection rejects native decoder objects without this method before reading
resume data or starting workers. There is intentionally no caller-supplied
identity override, because it could recreate silent cross-decoder merges.
Python prototype decoders remain usable with simulator `estimate(...)`, but are
not accepted by native collection.

## Getting Circuit And DEM Information

When starting from a circuit, first generate a detector error model:

```python
from faultscope.runtime import generate_native_dem

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
| `GraphlikeDecodingProblem` | MWPM-style decoders using graphlike columns of the detector error matrix |
| `BinaryLinearDecodingProblem` | BP+OSD/LDPC-style decoders using sparse binary detector error matrix `H` and logical fault matrix `F` |

`GraphlikeDecodingProblem` rejects hyperedges and undetectable pure logical
edges, because matching-style backends cannot infer those errors from syndrome
data. `BinaryLinearDecodingProblem` does not require graphlike edges; each DEM
edge is an independent binary error variable.

When an external format supplies a valid graphlike decomposition for a
canonical hyperedge, bind it to the canonical model once:

```python
from faultscope.dem import GeneratedDetectorErrorModel, GraphlikeDecompositionHints

hints = GraphlikeDecompositionHints(
    dem,
    {0: (((10, 20), (0,)), ((30,), ()))},
)
artifact = GeneratedDetectorErrorModel(dem, graphlike_hints=hints)
graphlike = artifact.compile_graphlike_problem()
```

The typed sidecar is a decoder construction hint, not another DEM. Every component
keeps the parent probability and `dem_edge_index`, while native sampling still
uses one Bernoulli variable per canonical edge. The compiler rejects invalid
ids, empty or pure-logical components, components with more than two detectors,
and decompositions whose GF(2) XOR differs from the parent support.

The problem views expose ids, detector coordinates, counts, `edge_summary`,
probabilities, weights, and sparse binary matrix entries for construction-time
inspection. `detector_coords` is ordered exactly like `detector_ids`; backend
builders can use it for geometry-aware graph compilation or partition planning.
They intentionally do not expose `to_numpy_*` hot-path helpers.

A native problem view uses the same canonical ids as DEM sampling: declarations
first, then ids first referenced by raw edges. Per-edge detector and observable
supports are reduced over GF(2), so sparse matrices never encode duplicate
coordinates with numeric values greater than one. Decoder builders should
consume these views instead of rebuilding ids or matrices from raw DEM fields.

## Native Decoder Backends

Native decoders are Python-owned handles around Rust decoder objects. They are
the intended path for production backends because the detector syndrome masks,
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
once. `NativeBatchDecoder` remains the public Python factory-handle class name;
it is not the name of a Rust trait. A backend-specific Python class is also a
factory handle and stores the immutable Rust factory:

```rust
use std::sync::Arc;

use faultscope_core::NativeDecoderFactory;

#[pyclass(name = "MyNativeDecoder", module = "faultscope._native")]
pub struct PyMyNativeDecoder {
    inner: Arc<dyn NativeDecoderFactory>,
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

        let factory = MyNativeDecoderFactory::from_graphlike_problem(
            problem,
            option_a,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(Self {
            inner: Arc::new(factory),
        })
    }
}
```

For BP+OSD/LDPC-style decoders, use
`core_dem.compile_binary_linear_problem()` instead. For a decoder that needs
more DEM metadata, use `compile_indexed()` or extend the construction-time DEM
view. Do not add per-batch circuit or DEM reads to `decode_batch(...)`.

The Rust factory stores immutable construction-time information. Mutable
solver state belongs only to a worker created from that information:

```rust
pub struct MyNativeDecoderFactory {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    // Share immutable compiled graph / H / F / weights / LLR data with workers.
    compiled: Arc<MyCompiledProblem>,
}

pub struct MyNativeDecoderWorker {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    // This worker's exclusive, mutable backend solver and scratch state.
    solver: MySolver,
}
```

The Python object only owns the handle. It does not own batch detector syndrome
masks or correction masks during `estimate(...)`.

The core Rust contracts are the distinct factory and worker traits below. These
signatures are normative: `NativeDecoderFactory: Send + Sync` is immutable and
creates fresh workers, while `NativeDecoderWorker: Send` owns mutable state and
decodes through `&mut self`:

```rust
pub trait NativeDecoderFactory: Send + Sync {
    fn name(&self) -> &str;
    fn detector_ids(&self) -> &[i64];
    fn observable_ids(&self) -> &[i64];
    fn batch_formats(&self) -> &[DetectorBatchFormat];
    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>>;
}

pub trait NativeDecoderWorker: Send {
    fn name(&self) -> &str;
    fn detector_ids(&self) -> &[i64];
    fn observable_ids(&self) -> &[i64];
    fn batch_formats(&self) -> &[DetectorBatchFormat];
    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch>;

    fn decode_batch_checked(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch>;
}
```

`batch_formats()` is a stable, non-empty, duplicate-free preference list using
`Masks`, `Packed`, or `Events`. FaultScope calls the single checked wrapper to
validate input shape, detector ID order, declared support, output variant,
observable layout, shot count, and padding. Masks input must return mask-major
corrections; Packed and Events input must return packed-row corrections. A
minimal Packed-native implementation has this ownership shape:

```rust
use std::sync::Arc;

use faultscope_core::{
    DecoderCorrectionBatch,
    DetectorBatchFormat,
    DetectorBatchView,
    NativeDecoderFactory,
    NativeDecoderWorker,
    NpError,
    NpResult,
    PackedObservableShotBatch,
};

const DECODER_NAME: &str = "my-decoder";
const BATCH_FORMATS: [DetectorBatchFormat; 1] = [DetectorBatchFormat::Packed];

struct MyCompiledProblem;
struct MySolver;

impl MySolver {
    fn from_compiled(_compiled: &MyCompiledProblem) -> NpResult<Self> {
        Ok(Self)
    }
}

struct MyNativeDecoderFactory {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    compiled: Arc<MyCompiledProblem>,
}

impl NativeDecoderFactory for MyNativeDecoderFactory {
    fn name(&self) -> &str {
        DECODER_NAME
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &BATCH_FORMATS
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        Ok(Box::new(MyNativeDecoderWorker {
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            solver: MySolver::from_compiled(&self.compiled)?,
        }))
    }
}

struct MyNativeDecoderWorker {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    solver: MySolver,
}

impl NativeDecoderWorker for MyNativeDecoderWorker {
    fn name(&self) -> &str {
        DECODER_NAME
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &BATCH_FORMATS
    }

    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        let DetectorBatchView::Packed(detectors) = detectors else {
            return Err(NpError::new("my-decoder only accepts Packed batches"));
        };
        let _solver = &mut self.solver; // Decode row-major packed shots here.
        Ok(DecoderCorrectionBatch::Packed(PackedObservableShotBatch::zero(
            self.observable_ids.clone(),
            detectors.shots,
        )?))
    }
}
```

Every worker must report the same `name`, `detector_ids`, `observable_ids`, and
ordered `batch_formats` as its factory for its entire lifetime. `decode_batch`
is the only dispatch method. Events may contain repeated detector indices and
the decoder must combine them by XOR. Mutable solver state remains exclusively
inside the worker.

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
`Arc<dyn NativeDecoderFactory>`. FaultScope owns the workers: it uses a
temporary worker for debug and estimate paths, a per-thread/task worker cache
for collection (including a one-worker collection), and
recursive composite workers whose children are fresh workers from the child
factories. Backend code therefore needs
no backend decoder pool or worker mutex. A Python subclass that only implements
`decode_batch_masks(batch)` remains a Python decoder and does not enter the
native fast path.

Native backends are discovered through built-in handles and post-install plugin
entry points. FaultScope 0.2.10 uses the strict factory/worker ABI V4:

```text
entry point group: faultscope.native_decoders
ABI and capsule name: faultscope.native_decoder_plugin.v4
numeric ABI: 4
capsule method: __faultscope_native_decoder_capsule__
```

The plugin package returns public Python decoder classes that are factory
handles. The factory cannot decode; FaultScope creates private exclusive workers
for collection, estimate, debug, and composite decoding. Collection caches a
worker per thread and task, including a one-worker collection, so a backend does
not need a solver pool or a mutex around mutable solver state. See
[Native Decoder ABI V4](native_decoder_abi.md) for tagged layouts, format
negotiation, fixed output mapping, and lifecycle rules.

The capsule returned by `__faultscope_native_decoder_capsule__()` must have a
non-null capsule destructor. That destructor owns the descriptor and factory
state, calls `drop_factory_state` exactly once, and then releases the
descriptor. FaultScope rejects V4 capsules without this ownership hook.

V4 is deliberately incompatible with V3: remove `supports_*` and the packed/
event-specific callbacks, add a stable ordered `batch_formats` metadata
callback, and route all layouts through one tagged `decode_batch`. The public
Python class names, constructors, entry-point group, and
`decode_batch_masks()` comparison helper remain unchanged. A problem view only
constructs the decoder; runtime layout is chosen later by format negotiation.

Official backend installation metadata lives in the built-in catalog. Each
entry records the backend name, backend package, proxy class name, target
problem view, source repository, default revision, installability, and a short
description. `pymatching` and `fusion-blossom` are installable.
`bpdecoder` is temporarily non-installable while awaiting its V4 migration.
`mwpm` remains discoverable but unavailable because its package is ABI v1 and
not yet migrated to FaultScope native decoder ABI V4. `bposd` is a reserved,
unimplemented, non-installable catalog/status entry. A generic install request
for any unavailable entry reports why it is unavailable and returns no
install plan or steps.

Python can inspect compiled native backend names:

```python
from faultscope.decoders import available_native_decoders

print(available_native_decoders())
```

The default build currently exposes:

```text
no-correction
graphlike-detector-copy
```

These are smoke-test and template backends, not production decoders.
Post-install backend packages may add more names. FaultScope never clones, builds, or
installs backend code during `import faultscope` or `estimate(...)`; installation is
an explicit command.

Python can use either friendly proxy classes or a generic resolver:

```python
from faultscope.decoders import (
    NativeFusionBlossomDecoder,
    NativeMwpmDecoder,
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
from faultscope.decoders import NativeGraphlikeDetectorCopyDecoder

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

Optional native backends are installed after the core FaultScope package. Users can
inspect backend status:

```bash
python -m faultscope.backends status
```

Installable catalog entries expose explicit installation helpers:

```bash
python -m faultscope.backends install fusion-blossom --dry-run
python -m faultscope.backends install pymatching --dry-run
```

These commands print clone/build/install steps for the two installable
entries. `bpdecoder` and `mwpm` are shown by `status` as unavailable pending
ABI V4 migration, and generic install requests return no plan. `bposd` is a reserved,
unimplemented, non-installable catalog/status entry; the generic
`python -m faultscope.backends install bposd --dry-run` command is accepted but
only reports its unavailability, emits no install steps, and installs nothing.

When the backend package is missing, the public proxy remains importable but
construction raises an install hint:

```python
from faultscope.decoders import (
    NativeBpDecoder,
    NativeFusionBlossomDecoder,
    NativeMwpmDecoder,
    NativePyMatchingDecoder,
)

decoder = NativeBpDecoder.from_dem(dem)  # temporarily unavailable pending ABI V4
decoder = NativeFusionBlossomDecoder.from_dem(dem)  # raises until installed
decoder = NativeMwpmDecoder.from_dem(dem)  # unavailable pending ABI V4 migration
decoder = NativePyMatchingDecoder.from_dem(dem)  # raises until installed
```

The post-install plugin ABI is not a general third-party stable ABI. It is a
versioned contract for official FaultScope backend packages so the core package can
reject mismatched backend builds before any hot-path decoding begins.

### Official Backend Packages

The repository includes official optional backend packages at:

```text
backends/faultscope-pymatching/
backends/faultscope-fusion-blossom/
```

Install them in editable mode during development from an activated project
virtual environment, or with `.venv/bin` explicitly on `PATH`:

```bash
.venv/bin/python -m pip install -e backends/faultscope-pymatching --no-build-isolation
.venv/bin/python -m pip install -e backends/faultscope-fusion-blossom
```

The `bpdecoder` backend remains in the official catalog but is intentionally
non-installable until its external package implements ABI V4.

The in-repo backend packages declare:

```toml
[project.entry-points."faultscope.native_decoders"]
pymatching = "faultscope_pymatching:backend_manifest"
fusion-blossom = "faultscope_fusion_blossom:backend_manifest"
```

Each manifest returns the current FaultScope native decoder plugin ABI, package
metadata, and one or more decoder classes. The matching classes implement
`from_dem(...)`, `from_graphlike_problem(...)`, and `from_circuit(...)`;
`from_dem(...)` compiles the ordinary no-hint `GraphlikeDecodingProblem`, while
the problem constructor accepts a precompiled decomposition view. Construction
passes that metadata to the package's Rust/PyO3 extension and stores external
native decoder state in a PyCapsule.

The PyMatching backend links pinned PyMatching sparse-blossom C++ source in the
`faultscope-pymatching` package. It does not call the Python
`PyMatchingDecoder` hot path and does not depend on the PyPI wheel exposing
a stable native SDK. The backend uses the same native PyCapsule boundary as the
other official packages: Python passes construction metadata, while hot-path
detector/correction buffers stay native. It declares `[Packed]`, so ordinary
collection and hotspot collection both use the single tagged callback with a
row-major input matching PyMatching's
`decode_batch(..., bit_packed_shots=True)` convention. FaultScope holds the
edge-event attribution trace in an independent sidecar.

The current backend is a minimal serial-solver beta fusion-blossom MWPM
adapter. It maps FaultScope detector indices to fusion-blossom vertices, converts
graphlike DEM edges to weighted solver edges, runs a serial solver per shot,
and maps matched vertex pairs through cached shortest paths back to observable
correction masks. It safely compresses identical one-detector boundary edges
and identical two-detector parallel edges before constructing the solver graph.
It is not yet the production partitioned or streaming adapter. The backend
build uses fusion-blossom's compact vertex/edge index mode and rejects graphs
that exceed that backend index range. The default integer conversion uses
`weight_scale=10_000`; after scaling, solver weights are normalized by their
common even-preserving divisor, preserving the integer MWPM objective while
reducing solver weight magnitudes when possible. The factory retains immutable
graph and path metadata; every private worker owns one exclusive mutable solver.
Collection reuses that worker through its thread/task cache. The backend does
not maintain a solver pool, decoder mutex, or packed-row scheduler.

`NPSIM_FUSION_BLOSSOM_PROFILE=1` prints a native per-batch timing
split for defect collection, solver clear, solver growth, matching extraction,
and correction application. This diagnostic is environment-variable gated and
does not change the public decoder API.

Current profiling on surface-code DEMs shows the fusion backend time is
dominated by upstream `solver.solve(...)`; defect collection, packed mask
layout, matching extraction, and correction application are secondary costs.
Further large-scale improvement should therefore focus on geometry-aware
partitioning or upstream solver strategy rather than Python callback removal or
small detector-layout special cases.

The backend decoder object exposes:

```python
decoder.__faultscope_native_decoder_capsule__()
decoder.name
decoder.detector_ids
decoder.observable_ids
decoder.edge_count
decoder.solver_vertex_count
decoder.solver_edge_count
decoder.boundary_vertex_count
decoder.build_summary
decoder.decode_batch_masks(batch)  # explicit Python comparison helper
```

FaultScope only calls the capsule method on the native fast path. The plugin ABI is
host-allocated for hot outputs: FaultScope allocates correction mask words for the
decoder's declared observables and initializes every correction word or byte to
zero before the callback. A backend may leave zero corrections untouched or XOR
its corrections into those buffers.
Backend packages must not allocate correction masks and ask FaultScope to free them
across the dynamic-library boundary.

### Fusion-Blossom Adapter Status

Fusion Blossom is a MWPM decoder route for QEC. The
[paper](https://arxiv.org/abs/2305.08307) describes a parallel MWPM decoder and
stream decoding support. The public
[repository](https://github.com/yuewuo/fusion-blossom) presents the project as
a fast MWPM solver for QEC, ships Rust code plus a Python binding, and currently
declares Rust package version `0.2.13`. FaultScope pins the backend dependency to a
concrete git revision because that version is not available from crates.io. The
public Rust source exposes types such as `SolverInitializer` and
`SyndromePattern` and helper functions such as `fusion_mwpm(...)` and
`detailed_matching(...)`.

FaultScope reserves the public constructor through a post-install proxy:

```python
from faultscope.decoders import NativeFusionBlossomDecoder

decoder = NativeFusionBlossomDecoder.from_dem(dem)
decoder = NativeFusionBlossomDecoder.from_circuit(circuit)
```

The default package does not ship the fusion-blossom solver. After
`faultscope-fusion-blossom` is installed, the proxy delegates construction to that
package while preserving the native fast path.

The implemented minimal beta adapter is:

1. Compile `DetectorErrorModel` to `GraphlikeDecodingProblem`.
2. Map each FaultScope detector index to a fusion-blossom vertex.
3. Convert one-detector DEM edges to boundary or virtual-vertex edges.
4. Merge one-detector boundary edges only when they share the same detector and
   fault-observable set; merge two-detector parallel DEM edges only when they
   share both endpoints and the same fault-observable set. The merged
   probability is the independent odd-parity probability.
5. Keep one virtual vertex per boundary edge group.
6. Reject two-detector parallel edges with different fault-observable sets,
   because choosing one correction would be ambiguous.
7. Preserve each solver edge's contributing DEM edge indices and
   fault-observable indices so the solver prediction
   can be converted back into observable correction masks.
8. Read each hot-path Packed row into the solver syndrome representation, run
   serial MWPM, and recover matched-pair paths through the exclusive worker's
   cache without touching Python.
9. Return a checked packed-row correction through the V4 tagged callback.

The construction summary is intentionally lightweight and safe to inspect from
Python:

```python
summary = decoder.build_summary
summary["dem_edge_count"]
summary["graphlike_edge_count"]
summary["solver_edge_count"]
summary["merged_parallel_edge_count"]
summary["edges"][0]["dem_edge_indices"]
summary["edges"][0]["fault_observables"]
```

The remaining productionization items are partitioned/streaming solver execution,
erasure/dynamic weights, compression for ambiguous parallel logical effects, and
large-scale performance tuning.

The primary beta evaluation entry point is the surface-code decoder performance
benchmark:

```bash
.venv/bin/maturin develop --release --skip-install
.venv/bin/python benchmarks/surface_code_decoder_performance.py --distances 3 5 7 --shots 10000
```

It compares Stim DEM + PyMatching, Stim bit-packed DEM + PyMatching bit-packed
batch decode, FaultScope DEM + Python PyMatching, FaultScope DEM + native
PyMatching, and FaultScope DEM + fusion-blossom native decoding when the optional
backend packages are installed. The bit-packed Stim/PyMatching row is the
official-style maximum-throughput baseline. The converter creates exactly one
canonical FaultScope edge per effective Stim `error` instruction and packages
actual multi-component separator groups in sparse, model-bound graphlike hints.
Every FaultScope path
samples that same canonical DEM; native matching backends consume the compiled
problem as an uncorrelated graphlike approximation. A backend without
`from_graphlike_problem(...)` is reported as `skip` instead of being constructed
from a transformed sampling model. The benchmark reports construction time,
sampling time where separable, native estimate time, solver-edge metadata,
merged parallel edges, and whether native paths stayed out of Python callbacks.

For native backend diagnosis, add `--split-native-baseline`. Native PyMatching
rows then report `sample_s` using a native no-correction decoder through the
same packed-row sampler path, and `decode_or_estimate_s` as the additional
decoder cost. This makes it clearer whether a gap is in FaultScope sampling or in
the backend decode loop.

Use `--same-seed-across-paths` when comparing mean-loss differences between
native decoder configurations; otherwise the benchmark preserves the historical
per-path seed offset.

## Validation And Performance Rules

FaultScope validates native decoder output before using it:

- correction `shots` must match the sampled batch;
- correction mask word count must match the shot count;
- correction observable ids must be unique;
- correction observable ids must exactly equal the decoder and sampler
  canonical sequence;
- packed output padding bits must be zero.

Layout mismatches are rejected before sampling or worker creation whenever the
decoder is bound to a sampler. There is no native ID-alignment fallback and no
implicit zero correction for a missing native observable. Only the explicit
no-decoder path supplies an all-zero correction batch.

The native fast path is used only when:

- `decoder` is a native decoder handle;
- no `loss_mask_fn` is supplied;
- no `correction_mask_fn` is supplied.

For DEM estimates, `aggregate_hotspots=False` also allows backends that support
the packed-row callback to receive `shots x ceil(detectors/8)` syndrome bytes
and return `shots x ceil(observables/8)` correction bytes. With
`aggregate_hotspots=True`, FaultScope preserves the existing hotspot-capable
detector-major path.

If a Python loss or correction callback is supplied, FaultScope uses the
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
   `faultscope.native_decoders` entry point manifest with the matching ABI.
5. Expose a friendly proxy class when the backend should be importable from
   `faultscope.decoders`.
6. Test both paths: Python compatibility behavior and native no-callback fast
   path.
