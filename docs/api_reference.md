# FaultScope API Reference

FaultScope is a Rust Cargo workspace with a Python API. The product runtime lives in
`faultscope-core` and is exposed to Python through the private extension module
`faultscope._native`. DEM APIs use detector error model terminology: detector
declarations represent detector matrix rows, generated DEM edges are columns of
the detector error matrix \(H=D\Omega\), and decoder-ready views expose that
sparse binary structure. User code should import from the public Python modules:
`faultscope`, `faultscope.core`, `faultscope.runtime`, `faultscope.dem`,
`faultscope.collection`, `faultscope.decoders`, `faultscope.io`, and
`faultscope.viz`.

The package is pre-1.0. Within `0.2.x`, names listed in public Python module
`__all__` values and the documented root APIs of `faultscope-core` and
`faultscope-collection` are compatibility contracts. Compatible additions may
land in patch releases. Removal or renaming requires deprecation before a later
minor release. Private modules and names beginning with `_` are implementation
details.

`faultscope.__version__` reports the installed distribution version. The Python
package includes `py.typed` and a generated structural stub for the private
PyO3 extension so public wrappers remain type-checkable.

## Import Surface

Common runtime, circuit, DEM, decoder, IO, and visualization objects are
re-exported from `faultscope`:

```python
from faultscope import (
    BernoulliPauliNoise,
    DemFaultScopeSimulator,
    FaultScopeSimulator,
    Circuit,
    Detector,
    DetectorErrorModelGenerator,
    LogicalObservable,
    NoiseLocation,
    Operation,
    NativeMwpmDecoder,
    NativePyMatchingDecoder,
    PyMatchingDecoder,
    Collector,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    Progress,
    TaskStats,
    collect,
    iter_collect,
    iter_progress,
)
```

`PauliFrame` and `StabilizerState` are not top-level exports. Import helper
types and functions from `faultscope.core`:

```python
from faultscope.core import PauliFrame, StabilizerState, pauli_string_to_xz
```

Lower-level subsystem modules expose grouped APIs:

```python
from faultscope.runtime import compile_native_sampler, generate_native_dem
from faultscope.dem import DemFaultScopeSimulator, DetectorErrorModel, DemHotspotEstimator
```

## Core Circuit Objects

`Circuit(n_qubits, operations)` stores an ordered stabilizer-compatible circuit.

Read-only attributes:

- `n_qubits: int`
- `operations: tuple[Operation, ...]`

Methods:

- `noise_locations() -> dict[str, NoiseLocation]`

`NoiseLocation(id, model, rate, qubits, tags=None)` names one stochastic noise
source. The rate must be in `[0, 1]`.

Read-only attributes:

- `id: str`
- `model: object`
- `rate: float`
- `qubits: tuple[int, ...]`
- `tags: dict[str, object]`

`NoiseLocation` is frozen. The constructor copies `tags`, and the getter returns
a shallow copy, so add tags at construction time.

`Operation` can be constructed directly, but application code should use the
static constructors:

```text
Operation.h(qubit, **metadata)
Operation.s(qubit, **metadata)
Operation.s_dag(qubit, **metadata)
Operation.x(qubit, **metadata)
Operation.y(qubit, **metadata)
Operation.z(qubit, **metadata)
Operation.cx(control, target, **metadata)
Operation.cz(left, right, **metadata)
Operation.swap(left, right, **metadata)
Operation.pauli_gate(qubits, pauli, **metadata)
Operation.noise(location, **metadata)
Operation.measure(qubit, *, key=None, basis="Z", noise=None, **metadata)
Operation.measure_pauli(qubits, pauli, *, key=None, noise=None, **metadata)
Operation.reset(qubit, *, key=None, basis="Z", **metadata)
Operation.detector(measurement_keys, *, detector_id=None, coords=None, **metadata)
Operation.observable_include(observable_id, measurement_keys, **metadata)
```

`Operation.pauli` is a read-only attribute, not the Pauli-gate constructor.

Read-only operation attributes include `kind`, `qubits`, `key`, `basis`,
`pauli`, `measurement_keys`, `observable_id`, `noise_location`, and `metadata`.

## Noise Models

FaultScope provides these stochastic noise model classes:

```text
BernoulliPauliNoise(pauli)
PauliChannel(weights)
SingleQubitDepolarizing()
TwoQubitDepolarizing(_events=None)
MeasurementBitFlip()
```

Common methods:

- `sample(rng, rate)` samples an event from a Python RNG object.
- `score(event, rate)` returns the log-derivative score used by hotspot
  estimators.
- `apply(event, state, frame, qubits)` applies an event to Python helper state.

Additional attributes and methods:

- `BernoulliPauliNoise.pauli`
- `PauliChannel.weights: dict[str, float]`
- `PauliChannel.event_length`
- `PauliChannel.total_weight`
- `MeasurementBitFlip.apply_to_bit(bit, event)`

## Pauli And Stabilizer Helpers

`faultscope.core` exports helper functions for Pauli representations:

```text
pauli_to_xz(pauli)
xz_to_pauli(x, z)
pauli_string_to_xz(pauli_string, n_qubits=None)
sparse_pauli_to_xz(n_qubits, qubits, paulis)
symplectic_product(x1, z1, x2, z2)
multiply_pauli_rows(left_x, left_z, left_sign, right_x, right_z, right_sign)
```

`PauliFrame` and `StabilizerState` are helper classes exposed from
`faultscope.core`. They are useful for tests and low-level workflows; the packed
runtime APIs below are the normal product path.

## Forward Runtime

`FaultScopeSimulator(circuit, *, observables=None)` compiles a
circuit for packed batch simulation.

Read-only attributes:

- `circuit`
- `observables`
- `locations: dict[str, NoiseLocation]`

Methods:

```text
run_batch(*, shots, rng=None, seed=None) -> SampleBatch
sample(shots, seed=None, rng=None) -> SampleBatch
sample_measurements(shots, seed=None, rng=None) -> dict[str, int]
estimate(
    *,
    shots,
    loss_mask_fn=None,
    decoder=None,
    correction_mask_fn=None,
    seed=None,
    baseline=None,
    top_k=10,
) -> FailureEstimate
```

`SampleBatch` stores bit-packed integer masks:

- `shots: int`
- `all_mask: int`
- `x_frame: tuple[int, ...]`
- `z_frame: tuple[int, ...]`
- `measurements: dict[str, int]`
- `detectors: dict[int, int]`
- `observables: dict[int, int]`
- `noise_event_masks: dict[str, int]`

Bit helpers:

```text
bit(mask, shot)
measurement_bit(key, shot)
measurement_mask(key)
detector_bit(detector_id, shot)
observable_bit(observable_id, shot)
x_bit(qubit, shot)
x_mask(qubit)
z_bit(qubit, shot)
z_mask(qubit)
```

`FailureEstimate` exposes:

- `shots`
- `mean_loss`
- `logical_failure_rate`
- `baseline`
- `sensitivities: dict[str, float]`
- `hotspots: dict[str, float]`
- `by_qubit`, `by_round`, `by_gate`, `by_operation`
- `locations: dict[str, NoiseLocation]`
- `top_hotspots(top_k=10)`
- `hotspot_table(top_k=10)`

Example:

```python
from faultscope import (
    BernoulliPauliNoise,
    FaultScopeSimulator,
    Circuit,
    NoiseLocation,
    Operation,
)

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.25, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
    ),
)

sim = FaultScopeSimulator(circuit)
batch = sim.run_batch(shots=32, seed=1)
result = sim.estimate(
    shots=128,
    seed=2,
    loss_mask_fn=lambda batch: batch.measurements["m0"],
)

print(batch.measurement_bit("m0", 0))
print(result.top_hotspots(1)[0].location_id)
```

## Native Runtime Handles

The `faultscope.runtime` wrappers expose lower-level native handles while preserving
the public `UnsupportedNativeCircuitError` boundary:

```text
compile_native_sampler(circuit, *, observables=None) -> NativePackedSampler
generate_native_dem(circuit, *, detectors=None, observables=None) -> DetectorErrorModel
compile_native_dem_generator(circuit, *, detectors=None, observables=None) -> NativeDemGenerator
compile_native_dem_sampler(dem) -> NativeDemSampler
compile_native_dem_sampler_from_circuit(
    circuit,
    *,
    detectors=None,
    observables=None,
    materialize_dem=True,
) -> NativeDemSampler
```

`NativePackedSampler` methods:

```text
sample(shots, seed=None, rng=None) -> SampleBatch
sample_measurements(shots, seed=None, rng=None) -> dict[str, int]
run_native_batch(shots, seed=None) -> native batch handle
estimate(shots, loss_mask_fn=None, decoder=None, correction_mask_fn=None, seed=None, baseline=None, top_k=10)
estimate_hotspots(batch, loss_mask, baseline=None, top_k=10)
```

`NativeDemGenerator` methods:

```text
generate_dem() -> DetectorErrorModel
generate() -> DetectorErrorModel
compile_sampler(*, materialize_dem=True) -> NativeDemSampler
```

`materialize_dem=False` compiles a light DEM sampler without constructing a
Python `DetectorErrorModel`. In that mode, `sampler.dem is None`. Sampling
works, but APIs that need DEM metadata, including estimate and hotspot result
construction, raise `ValueError`.

## DEM Runtime From Circuit

`DemFaultScopeSimulator(circuit, *, detectors=None, observables=None,
materialize_dem=True)` is the DEM-level counterpart to `FaultScopeSimulator`.
It compiles a circuit into a detector error model sampler in Rust, then samples
DEM edges directly. It does not run the forward stabilizer trajectory and does
not expose measurement, `x_frame`, or `z_frame` masks.

Read-only attributes:

- `circuit`
- `dem: DetectorErrorModel | None`
- `edge_count: int`

Methods:

```text
run_batch(*, shots, rng=None, seed=None, return_edge_events=True) -> DemSampleBatch
sample(shots, seed=None, rng=None) -> DemSampleBatch
run_native_batch(shots, seed=None) -> native DEM batch handle
estimate_default(shots, seed=None, baseline=None, top_k=10) -> DemHotspotEstimate
estimate(
    *,
    shots,
    seed=None,
    decoder=None,
    correction_mask_fn=None,
    loss_mask_fn=None,
    baseline=None,
    top_k=10,
    aggregate_hotspots=True,
) -> DemHotspotEstimate
estimate_hotspots(batch, loss_mask, baseline=None, top_k=10) -> DemHotspotEstimate
```

`materialize_dem=False` uses the same light sampling path as
`compile_native_dem_sampler_from_circuit(..., materialize_dem=False)`. In that
mode `sim.dem is None`; `run_batch(...)`, `sample(...)`, and
`run_native_batch(...)` work, while metadata-dependent estimate/hotspot APIs
raise `ValueError`.

Example:

```python
from faultscope import BernoulliPauliNoise, Circuit, DemFaultScopeSimulator
from faultscope import NoiseLocation, Operation

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.125, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
        Operation.observable_include(0, ("m0",)),
    ),
)

sim = DemFaultScopeSimulator(circuit)
batch = sim.run_batch(shots=64, seed=5)
result = sim.estimate(shots=256, seed=6, top_k=1)

print(batch.detector_bit(0, 0))
print(result.top_edges(1)[0].edge_index)
```

Native decoder handles:

```python
from faultscope.decoders import (
    NativeBatchDecoder,
    NativeCompositeDecoder,
    NativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder,
    available_native_decoders,
)

available_native_decoders()
decoder = NativeNoCorrectionDecoder(observable_ids=(0,))
decoder.name
decoder.detector_ids
decoder.observable_ids
composite = NativeCompositeDecoder((decoder_a, decoder_b))
```

Native decoders are Python-owned handles around Rust decoder objects. When a
native decoder is passed to `estimate(..., decoder=decoder)` without
`loss_mask_fn` or `correction_mask_fn`, FaultScope uses the native fast path:
detector syndrome masks, correction masks, default residual loss, and hotspot
aggregation stay in Rust. If a Python loss or correction callback is supplied,
FaultScope uses the compatibility path and may call `decoder.decode_batch_masks(...)`.
Python classes that merely define or subclass `decode_batch_masks(...)` remain
ordinary Python decoders and do not enter the native fast path.

`available_native_decoders()` returns the names of compiled native decoder
backends. The default build exposes `"no-correction"` and
`"graphlike-detector-copy"`. Compatible post-install backends can add names such
as `"pymatching"`, `"fusion-blossom"`, and `"bpdecoder"` through the
`faultscope.native_decoders` entry point group. Use
`get_native_decoder_class(name)` or `create_native_decoder(name, dem=dem)` for a
uniform API. Friendly proxies such as `NativePyMatchingDecoder`,
`NativeFusionBlossomDecoder`, `NativeMwpmDecoder`, and `NativeBposdDecoder` remain importable;
construction raises a precise availability error when no compatible backend is
available. `mwpm` is discoverable but unavailable pending its ABI v2 migration.
`bposd` is a reserved, unimplemented, non-installable catalog/status entry. The
generic `python -m faultscope.backends install bposd --dry-run` subcommand only
reports that unavailability; it creates no install plan or steps and installs
nothing. A post-install backend enters the native fast path only
when the constructed decoder exposes the FaultScope native decoder PyCapsule
ABI; otherwise it remains a normal Python decoder.

## Detector Error Models

`Detector(id, measurement_keys, coords=None)` declares one detector, meaning one
parity constraint over measurement outcomes. A collection of `Detector`
instances is the API representation of detector matrix \(D\).

Read-only attributes:

- `id: int`
- `measurement_keys: tuple[str, ...]`
- `coords: tuple[float, ...]`

`LogicalObservable(id, measurement_keys=None, pauli_qubits=None, pauli="")`
declares one logical observable. Observables can be based on measurement keys,
final Pauli-frame projection, or both.

`DetectorErrorEdge(probability, detectors, observables, location_id, event, tags=None)`
stores one DEM edge. Its `detectors` field is the support of one detector error
matrix column; its `observables` field is the support of the corresponding
logical fault column.

Read-only edge attributes:

- `probability: float`
- `detectors: tuple[int, ...]`
- `observables: tuple[int, ...]`
- `location_id: str`
- `event: object`
- `tags: dict[str, object]`

Methods:

- `to_dem_line() -> str`

`DetectorErrorModel(detectors, observables, edges)` stores a typed detector
error model: detector declarations, logical observable declarations, and
materialized detector error matrix columns with probabilities.

Compilation and sampling use one canonical layout. Detector and observable ids
are ordered as explicit declarations followed by ids first encountered in the
raw edge sequence. Within each edge, repeated ids are reduced over GF(2): even
multiplicity cancels and odd multiplicity leaves one target in first-occurrence
order. The source `DetectorErrorModel` remains unchanged; compiled views and
samplers expose the canonical interpretation. Duplicate explicit declarations
are rejected.

Methods:

```text
to_dem_text(*, include_detector_coords=True) -> str
edges_by_location() -> dict[str, list[DetectorErrorEdge]]
compile_indexed() -> IndexedDem
compile_graphlike_problem() -> GraphlikeDecodingProblem
compile_binary_linear_problem() -> BinaryLinearDecodingProblem
is_graphlike() -> bool
project_hotspots_to_edges(hotspots) -> dict[tuple[str, object], float]
project_result_to_detector_graph(result) -> DetectorGraphHotspots
project_sensitivities_to_detector_graph(sensitivities) -> DetectorGraphHotspots
```

`project_hotspots_to_edges(...)` expects a mapping from location id to hotspot
value. It returns edge values keyed by `(location_id, event)`.

`DetectorErrorModelGenerator(circuit, *, detectors=None, observables=None)`
generates a DEM from a circuit. If declarations are omitted, FaultScope reads
`Operation.detector(...)` and `Operation.observable_include(...)` entries from
the circuit.

`IndexedDem`, `GraphlikeDecodingProblem`, and `BinaryLinearDecodingProblem` are
native decoder-ready views. They expose stable ids, detector coordinates,
counts, `edge_summary`, and compact `repr(...)` metadata for inspection.
`detector_coords` follows `detector_ids` order. `GraphlikeDecodingProblem`
targets MWPM-style backends such as future fusion-blossom adapters.
`BinaryLinearDecodingProblem` targets BP+OSD/LDPC-style backends with sparse
binary detector error matrix `H` and logical fault matrix `F`. These objects
intentionally do not expose `to_numpy_*` hot-path helpers; native decoders
should consume the native view without moving masks through Python.

Example:

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    DetectorErrorModelGenerator,
    NoiseLocation,
    Operation,
)

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.125, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
        Operation.observable_include(0, ("m0",)),
    ),
)

dem = DetectorErrorModelGenerator(circuit).generate()
print(dem.to_dem_text())
print(dem.edges_by_location()["x0"][0].event)
```

## DEM Batch Hotspot Simulation

`DemHotspotEstimator(dem)` samples directly from DEM edges.

Methods:

```text
run_batch(*, shots, rng=None, seed=None, return_edge_events=True) -> DemSampleBatch
estimate(
    *,
    shots,
    seed=None,
    decoder=None,
    correction_mask_fn=None,
    loss_mask_fn=None,
    baseline=None,
    top_k=10,
    aggregate_hotspots=True,
) -> DemHotspotEstimate
```

Set `aggregate_hotspots=False` for decoder benchmarks and logic-only estimates.
This skips DEM edge-event recording and hotspot metadata aggregation while
keeping the same result object shape; when `decoder` is native, syndrome and
correction masks stay in native memory. Native decoders that implement the
optional packed-row callback receive `shots x ceil(detectors/8)` syndrome bytes
on this path and return `shots x ceil(observables/8)` correction bytes.

`DemSampleBatch` stores:

- `shots`
- `all_mask`
- `detectors: dict[int, int]`
- `observables: dict[int, int]`
- `edge_event_masks: dict[int, int]`

Bit helpers:

```text
bit(mask, shot)
detector_bit(detector_id, shot)
observable_bit(observable_id, shot)
edge_event_bit(edge_index, shot)
```

`DemHotspotEstimate` exposes:

- `dem`
- `shots`
- `mean_loss`
- `logical_failure_rate`
- `baseline`
- `edge_sensitivities: dict[int, float]`
- `edge_hotspots: dict[int, float]`
- `sensitivities: dict[str, float]`
- `hotspots: dict[str, float]`
- `by_detector`, `by_round`, `by_gate`, `by_operation`
- `locations: dict[str, DemLocationMetadata]`
- `detector_graph_hotspots`
- `top_edges(top_k=10)`
- `top_hotspots(top_k=10)`
- `hotspot_table(top_k=10)`

Example:

```python
from faultscope import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.dem import DemHotspotEstimator

dem = DetectorErrorModel(
    detectors=(Detector(0, ()),),
    observables=(LogicalObservable(0),),
    edges=(DetectorErrorEdge(0.125, (0,), (0,), "edge0", "X"),),
)

dem_sim = DemHotspotEstimator(dem)
batch = dem_sim.run_batch(shots=32, seed=3)
result = dem_sim.estimate(shots=128, seed=4, top_k=1)

print(batch.detector_bit(0, 0))
print(result.top_edges(1)[0].edge_index)
```

## Collection API

`faultscope.collection` provides native-first logical error-rate collection for
threshold sweeps. The public Python API is:

```python
from faultscope.collection import (
    COLLECTION_CSV_FIELDS,
    COLLECTION_CSV_HEADER,
    CollectionData,
    CollectionOptions,
    CollectionTask,
    Progress,
    TaskStats,
    FiniteSizeScalingFit,
    PairwiseCrossing,
    ThresholdAnalysisResult,
    ThresholdEstimate,
    ThresholdPoint,
    analyze_thresholds,
    collect,
    iter_collect,
    plot_threshold_analysis,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)
```

The top-level `faultscope` collection exports are `Collector`,
`CollectionOptions`, `CollectionRunOptions`, `CollectionTask`, `Progress`,
`TaskStats`, `HotspotCollectionResult`, `collect`, `collect_hotspots`,
`iter_collect`, and `iter_progress`.
Threshold analysis types and helpers are exported from `faultscope.collection`
only, not from top-level `faultscope`.

`CollectionOptions` is a frozen dataclass:

```text
CollectionOptions(
    max_shots: int | None = None,
    max_errors: int | None = None,
    batch_size: int = 10_000,
    start_batch_size: int | None = None,
    max_batch_size: int | None = None,
    max_batch_seconds: float | None = None,
    *,
    min_shots: int = 0,
)
```

`max_shots`, `batch_size`, `start_batch_size`, `max_batch_size`, and
`max_batch_seconds` must be positive when set. `max_errors` must be
non-negative when set; `min_shots` must be non-negative and no larger than
`max_shots`. Collection stops at `max_shots`, or when both `min_shots` and
`max_errors` have been reached.

`CollectionRunOptions` is a frozen dataclass:

```text
CollectionRunOptions(
    seed: int | None = None,
    num_workers: int = 1,
    existing_data_filepaths: tuple[str | Path, ...] = (),
    save_resume_filepath: str | Path | None = None,
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
    custom_error_count_key: str | None = None,
    decoders: tuple[str | object, ...] = (),
)
```

`num_workers` must be positive. The run seed is passed once to the Rust
scheduler, which derives deterministic task-local streams from each strong id.

`CollectionTask` is a frozen dataclass:

```text
CollectionTask(
    circuit: Circuit | None = None,
    dem: DetectorErrorModel | None = None,
    detectors: tuple[Detector, ...] | None = None,
    observables: tuple[LogicalObservable, ...] | None = None,
    decoder: object | str | None = None,
    decoder_options: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    collection_options: CollectionOptions | None = None,
    task_id: str | None = None,
    postselection_mask: bytes | bytearray | memoryview | None = None,
    postselected_observables_mask: bytes | bytearray | memoryview | None = None,
)
```

Exactly one of `circuit` or `dem` is required. Circuit tasks compile a
materialized native DEM sampler, using embedded declarations unless explicit
`detectors` or `observables` are supplied. String decoders are resolved with
`create_native_decoder(name, dem=dem, options=decoder_options)`. Object decoders
must be native decoder handles; Python decoders are rejected by collection.
Every decoder object must also implement
`strong_id_payload() -> Mapping[str, object]` and return JSON-serializable stable
identity data. Missing or invalid payloads fail before resume lookup or native
scheduling.

Collection strong ids use schema version 2. The canonical payload contains all
behavioral circuit/DEM fields, the resolved decoder payload, metadata, and
postselection masks. It preserves sequence order, sorts mapping keys, and does
not use `repr(...)`. Decoder payloads include effective normalized options and
solver structure, so constructing the same decoder by registered name or as an
equivalent object produces the same id. Runtime limits and display task ids are
not included.

Schema-v1 resume rows are intentionally not reused: the old id did not contain
enough decoder state for a safe migration. Archive the old CSV and use a new
resume file after upgrading. The CSV header itself is unchanged.

`TaskStats` is a frozen dataclass:

```text
TaskStats(
    task_id: str,
    shots: int,
    errors: int,
    discards: int,
    seconds: float,
    decoder: str | None,
    metadata: Mapping[str, object],
    strong_id: str = "",
    custom_counts: Mapping[str, int] = {},
)
```

Properties:

- `accepted_shots = shots - discards`
- `raw_error_rate = errors / shots`
- `accepted_error_rate = errors / accepted_shots`
- `logical_error_rate = errors / accepted_shots`
- `accepted_error_rate_stderr = sqrt(p * (1 - p) / accepted_shots)`
- `logical_error_rate_stderr = accepted_error_rate_stderr`

If `accepted_shots` is zero, accepted/logical rates and their standard errors
return `nan`. If `shots` is zero, `raw_error_rate` returns `nan`.
`TaskStats` also provides `with_edits(...)`, `to_csv_row()`,
`to_csv_line()`, `from_csv_row(...)`, and `__add__` for validated merging by
`strong_id`, decoder, and metadata. Normal `TaskStats` equality is the frozen
dataclass field equality. `__add__` treats `task_id` as display-only: stats may
merge with different display ids when `strong_id`, decoder, and metadata match.

`Progress` is a frozen dataclass used by streaming collection:

```text
Progress(
    new_stats: tuple[TaskStats, ...],
    status_message: str,
)
```

`new_stats` contains committed batch-delta `TaskStats` objects, not detector or
correction batch data.

Collection functions:

```text
Collector(*, options=None, run_options=None)

Collector.collect(tasks) -> list[TaskStats]
Collector.collect_hotspots(tasks) -> list[HotspotCollectionResult]
Collector.iter_collect(tasks) -> Iterator[TaskStats]
Collector.iter_progress(tasks) -> Iterator[Progress]

iter_collect(
    tasks,
    *,
    options=None,
    run_options=None,
) -> Iterator[TaskStats]

iter_progress(tasks, *, options=None, run_options=None) -> Iterator[Progress]
collect(tasks, *, options=None, run_options=None) -> list[TaskStats]
collect_hotspots(tasks, *, options=None, run_options=None) -> list[HotspotCollectionResult]
```

The functions are one-shot wrappers around `Collector`. Sampling options come
from the Collector and are overlaid by each task's `collection_options`. The
final batch is capped to the remaining shot budget. `max_errors` and
`custom_error_count_key` stopping are checked after each completed batch.

`num_workers` defaults to `1`. With fixed batch settings, the Rust scheduler can
parallelize both multiple tasks and a single large task. Fixed seed plus fixed
batch settings gives deterministic stats, ordered hotspot batches, and edge
sensitivities independent of worker count.
Adaptive tasks using `max_batch_seconds` execute two or three serial calibration
batches, freeze the median-throughput batch estimate, and parallelize the
remaining fixed-size batches through the same worker pool. Calibration and
parallel deltas are committed in order and obey the same shot/error limits.
Because calibration uses elapsed time, adaptive runs do not promise identical
error counts across worker counts or machine loads.

`iter_collect(...)` yields final `TaskStats` only. `iter_progress(...)` yields
committed batch-delta `Progress` values only. Stream deltas are emitted after
Rust commits batches in task-local ordinal order and after configured resume
CSV output is flushed. Deltas that complete after a stop condition are not
emitted or counted.

`CollectionRunOptions.decoders` is a tuple of decoder names or native decoder
objects. Tasks with `decoder is None` are expanded once per fanout decoder;
tasks that already specify `decoder=` keep their own decoder. String decoders
are resolved through `create_native_decoder(...)`.

`save_resume_filepath` and `existing_data_filepaths` use CSV rows with this
header:

```text
shots,errors,discards,seconds,decoder,strong_id,json_metadata,custom_counts
```

CSV/resume orchestration is Python-owned and outside the native sampling hot
path. Existing rows are merged by `strong_id`; mismatched decoder or metadata
for the same `strong_id` raises `ValueError`. A completed resume task is not
sampled again, and only newly collected deltas are appended. CSV rows do not
persist a display `task_id`, so `TaskStats.from_csv_row(...)` reconstructs
`task_id` from `strong_id`.

Public CSV utilities:

```text
COLLECTION_CSV_FIELDS
COLLECTION_CSV_HEADER
CollectionData(stats=())
read_stats_from_csv_files(*filepaths) -> list[TaskStats]
write_stats_to_csv_file(filepath, stats, *, append=False) -> None
```

`CollectionData` merges samples by `strong_id` using the same validation as
`TaskStats.__add__`. `read_stats_from_csv_files(...)` rejects malformed headers,
negative `shots`/`errors`/`discards`/`seconds`, negative custom counts,
non-object `custom_counts`, and non-integer custom count values.

Analysis helpers are available from `faultscope.collection.analysis`:

```text
error_rate_points(stats, *, x_key, group_key=None, count_key=None)
fit_log_error_rate_lines(points, *, x_key, group_key)
predict_error_rate(fit, x)
plot_error_rates(stats, *, x_key, group_key=None, output=None, ax=None, count_key=None)
```

Plotting lazily imports matplotlib and raises an install hint when the optional
collection plotting dependencies are unavailable.

Threshold analysis helpers are available from `faultscope.collection` and
`faultscope.collection.threshold`:

```text
analyze_thresholds(
    stats,
    *,
    x_key,
    distance_key,
    series_keys=(),
    count_key=None,
    bootstrap_samples=1000,
    confidence_level=0.95,
    seed=0,
    scaling_order=2,
) -> tuple[ThresholdAnalysisResult, ...]

plot_threshold_analysis(results, *, output=None, axes=None, log_y=True) -> (figure, axes)
```

`ThresholdPoint.rate` is the raw logical rate `errors / accepted_shots`, and
`ThresholdPoint.stderr` is its binomial standard error. Pairwise interpolation,
finite-size logit fitting, and bootstrap resampling use the private continuity
correction `(errors + 0.5) / (accepted_shots + 1)` so zero- and one-rate points
remain finite. Data is grouped by `series_keys`.
`PairwiseCrossing.status` is `"ok"`, `"no_crossing"`, or `"ambiguous"`.
`FiniteSizeScalingFit.status` is `"ok"`, `"insufficient_data"`,
`"fit_failed"`, or `"bootstrap_unstable"`. These statuses are diagnostics, not
exceptions; invalid inputs and missing optional dependencies still raise.

`plot_threshold_analysis(...)` accepts one result or an iterable of results and
rejects empty input. With `axes=None`, it creates an `n x 2` grid. Supplied axes
must have exact shape `(n, 2)`. The left panel plots raw-rate curves by
distance with pairwise crossings and scaling threshold diagnostics; the right
panel plots finite-size collapse when a scaling threshold and critical exponent
exist, otherwise it annotates the scaling status. `output=` saves with
`bbox_inches="tight"`. With `log_y=True`, a series containing raw zero-rate
points uses a symmetric-log scale so those observations remain visible;
strictly positive series use a logarithmic scale. Matplotlib is imported lazily
and missing dependencies raise an install hint for `faultscope[collection]`.

Postselection masks are bytes-like bit-packed masks over the native detector or
observable order. Detector postselection discards any shot where a selected
detector fired. Observable postselection discards any shot where the decoder
residual is nonzero on a selected observable. Logical errors are counted only on
accepted shots and only on non-postselected residual observables.

Custom counts:

- `count_observable_error_combos=True` records accepted residual observable
  combinations under keys such as `obs_mistake_mask=E_E__`.
- `count_detection_events=True` records `detection_events` and
  `detectors_checked`.
- `custom_error_count_key="..."` makes `max_errors` use that custom count
  instead of `errors`.

## Callback Contracts

Forward estimate callbacks use bit-packed integer masks:

- `decoder.decode_batch_masks(batch) -> dict[int, int]`
- `correction_mask_fn(batch) -> dict[int, int]`
- `loss_mask_fn(batch) -> int`
- `loss_mask_fn(batch, corrections) -> int`

Native forward batches also provide
`batch.measurement_masks(keys) -> dict[str, int]`. Python decoders should call
it once when they need a known subset of measurement records; it materializes
only the requested packed masks. `batch.measurements` remains the compatible
full mapping.

DEM estimate callbacks also use bit-packed masks. The DEM loss callback is
always called with a correction mapping:

- `decoder.decode_batch_masks(batch) -> dict[int, int]`
- `correction_mask_fn(batch) -> dict[int, int]`
- `loss_mask_fn(batch, corrections) -> int`

Returned masks use one bit per shot. A set bit means the condition is true for
that shot. Do not pass both `decoder` and `correction_mask_fn` in the same
estimate call.

## Optional Integrations

PyMatching:

```python
from faultscope import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.decoders import PyMatchingDecoder

dem = DetectorErrorModel(
    detectors=(Detector(0, ()),),
    observables=(LogicalObservable(0),),
    edges=(DetectorErrorEdge(0.1, (0,), (0,), "e0", "X"),),
)
decoder = PyMatchingDecoder.from_dem(dem)
print(decoder.decode_batch_masks({0: 0b1010}, shots=4))
```

`PyMatchingDecoder.from_dem(...)` builds a Python compatibility decoder
from a graphlike DEM. The resulting decoder implements
`decode_batch_masks(batch)`, so it can be passed to either
`FaultScopeSimulator.estimate(..., decoder=decoder)` or
`DemHotspotEstimator.estimate(..., decoder=decoder)`, but its hot batch
data crosses Python.

For the native PyMatching hot path, install the optional backend and use:

```python
from faultscope.decoders import NativePyMatchingDecoder

decoder = NativePyMatchingDecoder.from_dem(dem)
```

Stim import:

```python
from faultscope.io import parse_stim_circuit

imported = parse_stim_circuit("X_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]\n")
print(imported.measurement_keys)
```

Visualization helpers require Pillow and write image files:

```text
write_repetition_hotspot_heatmap(result, path, *, distance, rounds)
write_repetition_gate_structure_hotspot_map(result, path, *, distance, rounds)
write_rotated_surface_code_spatial_hotspot_map(result, path, *, distance)
```

## Rust API

The Rust core crate is `faultscope-core`. It is Python-independent and re-exports
its main types from the crate root:

```rust
use faultscope_core::{
    FaultScopeSimulator, Circuit, DemHotspotEstimator,
    Detector, DetectorErrorModelGenerator, LogicalObservable, NoiseLocation,
    NoiseModel, Operation,
};
```

Core data types include:

- `Circuit { n_qubits, operations }`
- `Operation`
- `NoiseLocation { id, model, rate, qubits, tags }`
- `NoiseModel`
- `Detector`
- `LogicalObservable`
- `DetectorErrorEdge`
- `DetectorErrorModel`
- `SamplerProgram` and `RuntimeState` (integer-indexed forward-sampler IR and batch state)
- `DemBatch`
- `HotspotEstimate`
- `DemHotspotEstimate`
- `NpError` and `NpResult<T>`

Example:

```rust
use std::collections::HashMap;

use faultscope_core::{
    FaultScopeSimulator, Circuit, NoiseLocation, NoiseModel, Operation,
};

let noise = NoiseLocation {
    id: "x0".to_string(),
    model: NoiseModel::BernoulliPauli("X".to_string()),
    rate: 1.0,
    qubits: vec![0],
    tags: HashMap::new(),
};

let circuit = Circuit {
    n_qubits: 1,
    operations: vec![
        Operation::Noise(noise),
        Operation::Measure {
            qubit: 0,
            key: Some("m0".to_string()),
            basis: "Z".to_string(),
            noise: None,
        },
    ],
};

let simulator = FaultScopeSimulator::new(circuit, Vec::new())?;
let batch = simulator.run_batch(1024, Some(1), true)?;
let estimate = simulator.estimate_from_loss(&batch, &batch.measurements["m0"], None, 10);
```

Validation failures return `NpError` in Rust and usually become `ValueError` or
`UnsupportedNativeCircuitError` through the public Python wrappers.
