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

The package is pre-1.0. Python and Rust API compatibility is not guaranteed
between releases, including patch releases. Public module `__all__` values and
documented crate-root exports describe the current supported surface, while
private modules and names beginning with `_` are implementation details. The
native decoder ABI is versioned separately.

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

`TwoQubitDepolarizing` always uses the canonical ordered set of 15 non-identity
two-qubit Pauli events. The compatibility `_events` argument may be omitted or
set to that exact sequence; custom subsets, duplicates, and reorderings are
rejected.

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
```

`PauliFrame(x, z)` validates matching binary x/z vectors once and owns the
validated data. `PauliFrame.zero(n_qubits)` constructs an empty frame.
`StabilizerState` has no raw-tableau constructor; create it with
`StabilizerState.zero(n_qubits)` and use its invariant-preserving operations.

Dense x/z arguments must match the state's `n_qubits` and contain only binary
integer values. Sparse Pauli and gate targets must be unique non-negative
integers in `0 <= qubit < n_qubits`. Malformed calls raise `ValueError` before
computation, RNG use, or state mutation. Zero-qubit empty supports and
correctly-sized identity supports remain valid.

Noise-model `apply(...)` requires a `StabilizerState` and `PauliFrame` with the
same `n_qubits`; event validation completes before both objects are updated as
one operation.

`PauliFrame` and `StabilizerState` are helper classes exposed from
`faultscope.core`. They are useful for tests and low-level workflows; the packed
runtime APIs below are the normal product path.

The Rust crate root exposes owning `PauliFrame` and zero-state-only
`ConcreteStabilizer`. Their public gates and Pauli operations return `NpResult`;
low-level conversion, symplectic, row/word, validator, and frame free functions
are private implementation details. `ConcreteStabilizer::measure_pauli_with`
validates first and calls its RNG closure only when the measurement is random.
`ConcreteStabilizer::apply_pauli_event` validates a sparse event once and then
atomically updates the state and matching `PauliFrame`.

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
generate_native_dem(circuit, *, detectors=None, observables=None,
                    approximate_disjoint_errors=False) -> DetectorErrorModel
compile_native_dem_generator(circuit, *, detectors=None, observables=None,
                             approximate_disjoint_errors=False) -> NativeDemGenerator
compile_native_dem_sampler(dem) -> NativeDemSampler
compile_native_dem_sampler_from_circuit(
    circuit,
    *,
    detectors=None,
    observables=None,
    approximate_disjoint_errors=False,
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

`estimate_hotspots(...)` accepts only a native batch produced by the same
compiled sampler. The batch must contain recorded per-location event masks;
foreign batches and invalid loss-mask widths raise `ValueError`.

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
approximate_disjoint_errors=False, materialize_dem=True)` is the DEM-level
counterpart to `FaultScopeSimulator`.
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

DEM hotspot estimation likewise requires a native batch from the same compiled
sampler with edge-event recording enabled. Missing events, incompatible batch
identity, zero shots, or invalid loss-mask widths raise `ValueError`.

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
decoder_from_circuit = NativeNoCorrectionDecoder.from_circuit(circuit)
decoder.name
decoder.detector_ids
decoder.observable_ids
composite = NativeCompositeDecoder((decoder_a, decoder_b))
```

`NativeNoCorrectionDecoder.from_circuit(...)` and
`create_native_decoder("no-correction", circuit=circuit)` infer omitted detector
and observable layouts from `Operation.detector(...)` and
`Operation.observable_include(...)` declarations embedded in the circuit.
Passing an explicit empty sequence, such as `detectors=()`, requests an empty
layout instead of inference.

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
as `"pymatching"` and `"fusion-blossom"` through the
`faultscope.native_decoders` entry point group. Use
`get_native_decoder_class(name)` or `create_native_decoder(name, dem=dem)` for a
uniform API. Friendly proxies such as `NativePyMatchingDecoder`,
`NativeFusionBlossomDecoder`, `NativeMwpmDecoder`, and `NativeBposdDecoder` remain importable;
construction raises a precise availability error when no compatible backend is
available. Native plugins use ABI V4: each factory declares an ordered
Masks/Packed/Events preference list and each worker exposes one tagged callback.
DEM hotspot attribution is held in a separate FaultScope sidecar, not passed to
the decoder. `bpdecoder` and `mwpm` are discoverable but temporarily unavailable
pending ABI V4 migration.
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
compile_graphlike_problem(*, decomposition=None) -> GraphlikeDecodingProblem
compile_binary_linear_problem() -> BinaryLinearDecodingProblem
is_graphlike() -> bool
project_hotspots_to_edges(hotspots) -> dict[tuple[str, object], float]
project_result_to_detector_graph(result) -> DetectorGraphHotspots
project_sensitivities_to_detector_graph(sensitivities) -> DetectorGraphHotspots
```

`project_hotspots_to_edges(...)` expects a mapping from location id to hotspot
value. It returns edge values keyed by `(location_id, event)`.

`compile_graphlike_problem(decomposition=...)` accepts an optional mapping from
canonical DEM edge index to an ordered sequence of `(detector_ids,
observable_ids)` components. Each component inherits the parent edge's
probability and `dem_edge_index`. Component targets are reduced over GF(2), and
the XOR of all components must exactly reproduce the parent edge's detector and
observable support. Components must be nonempty, touch one or two detectors,
and reference valid model ids; pure logical components are rejected. Edges
without a hint retain the ordinary graphlike validation behavior. This
argument compiles a decoder view only: DEM sampling always samples the original
canonical edges.

The mapping argument is the backward-compatible low-level interface. New code
should package it once as typed, model-bound metadata:

```python
hints = GraphlikeDecompositionHints(dem, components_by_edge)
artifact = GeneratedDetectorErrorModel(dem, graphlike_hints=hints)
problem = artifact.compile_graphlike_problem()
```

`GraphlikeDecompositionHints` validates and compiles the sparse hint mapping at
construction time. It is bound to the exact immutable `DetectorErrorModel`
instance supplied to its constructor; attaching it to another model raises
`ValueError`. Its read-only `components_by_edge` attribute exposes the validated
sparse mapping.

`GeneratedDetectorErrorModel` packages the canonical `dem` and optional
`graphlike_hints`. `compile_graphlike_problem()` uses the hints when present and
otherwise delegates to ordinary graphlike compilation. This is an API-level
bundle only: sampling still consumes `artifact.dem`, never its components.

`DetectorErrorModelGenerator(circuit, *, detectors=None, observables=None,
approximate_disjoint_errors=0.0)` generates a DEM from a circuit. If declarations
are omitted, FaultScope reads `Operation.detector(...)` and
`Operation.observable_include(...)` entries from the circuit.

`generate()` returns the backward-compatible canonical DEM.
`generate_artifact()` returns a `GeneratedDetectorErrorModel`. The native
circuit generator does not synthesize graphlike hints yet, so its artifact
currently has `graphlike_hints is None`; this stable return type allows hints to
be added later without changing sampler or decoder APIs.

`approximate_disjoint_errors` matches Stim's circuit-to-DEM policy. Before using
this option, a one-qubit `PauliChannel` attempts Stim's numerical conversion into
independent X/Y/Z mechanisms; a single positive component is trivially exact,
and some multi-component channels are exact as well. `False` or `0.0` rejects a
categorical channel when that conversion is unavailable. `True` enables
independent approximation for all valid component probabilities; a float in
`[0, 1]` is the maximum accepted component probability. `PAULI_CHANNEL_2` and
wider channels have no general exact conversion attempt. Depolarizing channels
use an exact independent reparameterization and do not require this option, but
exact conversion is limited to rates `<=3/4` for one qubit and `<=15/16` for two
qubits. The option does not alter hand-built DEM semantics: every DEM edge
remains an independent Bernoulli instruction. During an opted-in `PauliChannel`
conversion, disjoint Pauli components with identical propagated
detector/observable support are first combined by summing their probabilities;
only distinct effect classes are approximated as independent.

The one-qubit solver follows Stim's `1e-14` absolute residual criterion. A tiny
non-factorable channel can therefore be accepted as numerically exact without
the option, potentially dropping effects beneath that tolerance. Forward
sampling remains the exact categorical reference.

`IndexedDem`, `GraphlikeDecodingProblem`, and `BinaryLinearDecodingProblem` are
native decoder-ready views. They expose stable ids, detector coordinates,
counts, `edge_summary`, and compact `repr(...)` metadata for inspection.
`detector_coords` follows `detector_ids` order. `GraphlikeDecodingProblem`
targets MWPM-style backends, including the optional PyMatching and
fusion-blossom adapters.
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
    COLLECTION_COUNTER_SCHEMA_VERSION,
    COLLECTION_CSV_FIELDS,
    COLLECTION_CSV_HEADER,
    CollectionCounterSchema,
    CollectionData,
    Collector,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    HotspotCollectionResult,
    Progress,
    TaskStats,
    FiniteSizeScalingFit,
    PairwiseCrossing,
    ThresholdAnalysisResult,
    ThresholdEstimate,
    ThresholdPoint,
    analyze_thresholds,
    collect,
    collect_hotspots,
    iter_collect,
    iter_progress,
    plot_threshold_analysis,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)
```

The top-level `faultscope` collection exports are `Collector`,
`CollectionCounterSchema`, `CollectionOptions`, `CollectionRunOptions`,
`CollectionTask`, `Progress`, `TaskStats`, `HotspotCollectionResult`, `collect`,
`collect_hotspots`, `iter_collect`, and `iter_progress`.
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
`max_errors` have been reached. Shot, error, and batch counts must be integers;
booleans and floating-point counts are rejected. `max_batch_seconds` must be a
finite positive number.

When used as `CollectionTask.collection_options`, only arguments explicitly
passed to `CollectionOptions` override the Collector defaults. Passing a
default value such as `batch_size=10_000` or `min_shots=0` restores that
default, while an explicit `None` clears an inherited nullable value. Clearing
`max_shots` is allowed during option merging but collection then fails with
`max_shots is required` before compiling the task.

Use `options.with_edits(batch_size=10_000, max_errors=None)` to derive options
without losing which fields are explicit overrides. The method preserves prior
override intent and marks every supplied field explicit, including values equal
to their defaults. `dataclasses.replace()` is not supported for
`CollectionOptions` because it replays every dataclass field and cannot preserve
this distinction.

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

`num_workers` must be a positive non-boolean integer. `seed`, when set, must be
an integer in the unsigned 64-bit range. The run seed is passed once to the
Rust scheduler, which derives deterministic task-local streams from each
internal sampling id.

`CollectionTask` is a frozen dataclass:

```text
CollectionTask(
    circuit: Circuit,
    *,
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

`circuit` is required. Passing `dem=` raises a migration error pointing to
`faultscope.collection.dem.DemCollectionTask`. `detectors=None` and
`observables=None` use declarations embedded in the circuit. An explicit
sequence replaces the corresponding embedded declarations, and an explicit
empty sequence clears them. The resulting declarations are compiled into the
`SamplerProgram`; their stable declaration order defines the detector and
observable layouts used by decoder validation, postselection, and counters.

Collection shots execute the circuit directly with the packed Forward runtime.
No DEM is generated when the task has no decoder or supplies an already
constructed native decoder. A string decoder causes one circuit-derived DEM to
be generated and cached during task preparation solely to construct the
decoder's static problem; all shots still use the Forward sampler. Object
decoders must be native decoder handles; Python decoders are rejected by
collection. Every decoder object must also implement
`strong_id_payload() -> Mapping[str, object]` and return JSON-serializable stable
identity data. Missing or invalid payloads fail before resume lookup or native
scheduling.

Collection identity uses the v3 CSV/resume contract and is split into a sampling
id (schema v2) and a public `strong_id` (schema v4). A Forward source digest has
the `forward_circuit` kind and contains the circuit plus the `None`/explicit
detector and observable declarations; `None` and an explicit empty sequence are
therefore distinct. The sampling id combines that source digest with the
resolved decoder digest, metadata, and both postselection masks. Rust derives
task-local random streams from this id. The `strong_id` hashes the sampling id
together with the complete counter schema. Changing either count flag creates a
separate resume identity while preserving the same seeded random samples.
`custom_error_count_key`, task id, seed, shot/error limits, batch sizing, and
worker count are excluded from both identities. Decoder payloads include
effective normalized options and solver structure; canonical encoding preserves
sequence order, sorts mapping keys, and never uses `repr(...)`.

`CollectionCounterSchema` is a public frozen dataclass:

```text
CollectionCounterSchema(
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
)

schema.schema_version == 1
```

`COLLECTION_COUNTER_SCHEMA_VERSION` is `1`. Native collection creates the
schema from the two `CollectionRunOptions` count flags; callers do not pass a
schema object into `collect`. Increment this schema version whenever counter
meaning, key encoding, or per-shot coverage changes. A change to resume
identity or CSV layout requires a new collection resume-contract version as
well; no subset/superset counter backfill is inferred across versions.

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
    counter_schema: CollectionCounterSchema | None = None,
)
```

Native collection totals and progress deltas always carry a non-`None`
`counter_schema`. `None` is reserved for manually constructed analysis data;
those values may contain arbitrary custom counters but cannot be passed back as
resume data. Versioned stats must contain both fixed detection counters when
detection counting is enabled, including explicit zero values, and must omit
them when it is disabled.

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
merge with different display ids when `strong_id`, decoder, metadata, and
counter schema match exactly.

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

`HotspotCollectionResult` is a frozen dataclass with Forward semantics:

```text
HotspotCollectionResult(
    stats: TaskStats,
    batch_stats: tuple[TaskStats, ...],
    location_sensitivities: Mapping[str, float],
)
```

`location_sensitivities` is keyed by physical noise-location id and is
shot-weighted across all committed batches. Forward hotspot collection records
noise-event masks; ordinary collection does not. Hotspot collection does not
support CSV partial resume or adaptive batch sizing.

The functions are one-shot wrappers around `Collector`. Sampling options come
from the Collector and are overlaid by each task's `collection_options`. The
final batch is capped to the remaining shot budget. `max_errors` and
`custom_error_count_key` stopping are checked after each completed batch.

`num_workers` defaults to `1`. With fixed batch settings, the Rust scheduler can
parallelize both multiple tasks and a single large task. Fixed seed plus fixed
batch settings gives deterministic stats, ordered hotspot batches, and location
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

### Legacy DEM Collection

Explicit detector-error-model sampling is isolated in
`faultscope.collection.dem` and is not re-exported from
`faultscope.collection` or top-level `faultscope`. It is a library-only legacy
path; the collection CLI accepts Forward circuit tasks only.

```python
from faultscope.collection.dem import (
    DemCollectionTask,
    DemCollector,
    DemHotspotCollectionResult,
    collect,
    collect_hotspots,
    iter_collect,
    iter_progress,
)
```

`DemCollectionTask` is a frozen dataclass:

```text
DemCollectionTask(
    dem: DetectorErrorModel,
    decoder: object | str | None = None,
    decoder_options: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    collection_options: CollectionOptions | None = None,
    task_id: str | None = None,
    postselection_mask: bytes | bytearray | memoryview | None = None,
    postselected_observables_mask: bytes | bytearray | memoryview | None = None,
)
```

`DemCollector` exposes the same `collect`, `collect_hotspots`, `iter_collect`,
and `iter_progress` methods as `Collector` but samples the supplied DEM directly.
Its hotspot result retains edge semantics:

```text
DemHotspotCollectionResult(
    stats: TaskStats,
    batch_stats: tuple[TaskStats, ...],
    edge_sensitivities: tuple[float, ...],
)
```

The legacy module reuses `CollectionOptions`, `CollectionRunOptions`,
`CollectionCounterSchema`, `TaskStats`, and the CSV helpers. Explicit DEM source
payloads retain their existing identity, while a Forward `forward_circuit`
source cannot collide with an old circuit-to-DEM identity.

`save_resume_filepath` and `existing_data_filepaths` use CSV rows with this
header:

```text
shots,errors,discards,seconds,decoder,strong_id,json_metadata,json_counter_schema,custom_counts
```

CSV/resume orchestration is Python-owned and outside the native sampling hot
path. `json_counter_schema` stores the complete canonical schema object.
Existing rows are merged by `strong_id`; mismatched decoder, metadata, or
counter schema for the same `strong_id` raises `ValueError`. Resume accepts only
the exact current v3 schema and rejects missing/unsupported schemas or missing
fixed counters before native workers start. A completed resume task is not
sampled again, and only newly collected deltas are appended. CSV rows do not
persist a display `task_id`, so `TaskStats.from_csv_row(...)` reconstructs
`task_id` from `strong_id`. When those rows are used for collection resume,
validated historical stats are rebound to the current task identity before
completion checks, so returned stats use the current task's display `task_id`
even when no additional sampling is needed.

The v3 header is deliberately incompatible with v2 CSV. Reading a v2 file or
attempting to append to it raises `ValueError`; append validates the existing
header before opening it for writes. There is no in-place migration or
compatibility adapter. Archive the old file or convert it with tooling outside
this API before starting a v3 resume run.

Public CSV utilities:

```text
COLLECTION_CSV_FIELDS
COLLECTION_CSV_HEADER
COLLECTION_COUNTER_SCHEMA_VERSION
CollectionCounterSchema(...)
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
- `custom_error_count_key=None` makes `max_errors` use the main `errors` count.
- `custom_error_count_key="detection_events"` or `"detectors_checked"`
  requires `count_detection_events=True`. A missing fixed key is invalid, not
  zero.
- `custom_error_count_key="obs_mistake_mask=<mask>"` requires
  `count_observable_error_combos=True`. For every expanded task, `<mask>` must
  have exactly one `E`/`_` character per observable, include at least one `E`
  on a non-postselected observable, and never put `E` on a postselected
  observable. A valid combo that is absent from `custom_counts` is a true zero.

Every non-`None` stop key is validated for every expanded task even when
`max_errors` is unset or the task is already complete. Serial, parallel,
adaptive, streaming, and hotspot collection use the same validation and
batch-commit stopping rules; invalid keys fail before sampling, resume writes,
or collection worker creation.

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
- `SamplerProgram` and `RuntimeState` (immutable integer-indexed forward-sampler
  IR with read-only getters, and batch state)
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
let estimate = simulator.estimate_from_loss(&batch, &batch.measurements["m0"], None, 10)?;
```

The Rust collection crate is `faultscope-collection`. Its primary public API
accepts a compiled `SamplerProgram` or `ForwardLogicalCollectionTask` and keeps
all shot sampling on the Forward runtime:

```rust
use faultscope_collection::{
    collect_forward_hotspot_tasks,
    collect_forward_logical_error_stats,
    collect_forward_logical_error_tasks,
    collect_forward_logical_error_tasks_with_progress,
    sample_forward_logical_error_stats,
    ForwardHotspotCollectionResult,
    ForwardLogicalCollectionTask,
    LogicalCollectionOptions,
    LogicalCollectionRunOptions,
    LogicalCollectionStats,
    LogicalCounterSchema,
    LOGICAL_COUNTER_SCHEMA_VERSION,
};
```

`ForwardHotspotCollectionResult` contains `stats`, ordered `batch_stats`, and
`location_sensitivities: HashMap<String, f64>`. The neutral `Logical*` names are
public aliases for the options, run options, stats, and counter schema shared by
both collection runtimes.

The explicit DEM APIs remain available as a legacy Rust path:

```text
DemLogicalCollectionTask
collect_dem_logical_error_stats(...)
sample_dem_logical_error_stats(...)
collect_dem_logical_error_tasks(...)
collect_dem_logical_error_tasks_with_progress(...)
collect_dem_hotspot_tasks(...) -> Vec<DemHotspotCollectionResult>
```

`DemHotspotCollectionResult` retains `edge_sensitivities: Vec<f64>`. Forward
callers should use the `collect_forward_*` functions; the scheduler, stop
conditions, progress protocol, decoder worker cache, and shared `Logical*`
schemas are common to both paths.

Validation failures return `NpError` in Rust and usually become `ValueError` or
`UnsupportedNativeCircuitError` through the public Python wrappers.
Both Rust `estimate_from_loss` methods and the low-level
`compute_packed_estimate` return `NpResult`. Runtime states and DEM batches are
bound to their compiled layout; clones preserve that identity, while a
separately compiled estimator is intentionally incompatible even if its source
model is equal. Their hotspot-layout fields are private. Read them through
methods such as `noise_locations()`, `shots()`, `all_mask()`, `event_masks()`,
and `edge_event_masks()`. Unrelated public result fields such as detector,
observable, and loss masks remain directly accessible. This prevents callers
from invalidating an already validated event layout; public estimate methods
validate once and dispatch to a trusted internal aggregation loop. The
standalone low-level `compute_packed_estimate` remains a checked boundary for
callers that pair a `RuntimeState` with raw locations and a catalog.
