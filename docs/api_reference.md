# NPSim API Reference

NPSim is a Rust Cargo workspace with a Python API. The product runtime lives in
`npsim-core` and is exposed to Python through the private extension module
`npsim._npsim_native`. User code should import from the public Python modules:
`npsim`, `npsim.core`, `npsim.runtime`, `npsim.dem`, `npsim.decoders`,
`npsim.io`, and `npsim.viz`.

The package is pre-1.0. Python source-level compatibility is the main user
compatibility target. The Rust core API is public and typed, but may still move
while the core stabilizes.

## Import Surface

Common runtime, circuit, DEM, decoder, IO, and visualization objects are
re-exported from `npsim`:

```python
from npsim import (
    BernoulliPauliNoise,
    BatchForwardNoiseAwareSimulator,
    Circuit,
    Detector,
    DetectorErrorModelGenerator,
    LogicalObservable,
    NoiseLocation,
    Operation,
    PyMatchingBatchDecoder,
)
```

`PauliFrame` and `StabilizerState` are not top-level exports. Import helper
types and functions from `npsim.core`:

```python
from npsim.core import PauliFrame, StabilizerState, pauli_string_to_xz
```

Lower-level subsystem modules expose grouped APIs:

```python
from npsim.runtime import compile_native_sampler, generate_native_dem
from npsim.dem import DetectorErrorModel, DemBatchHotspotSimulator
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

NPSim provides these stochastic noise model classes:

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

`npsim.core` exports helper functions for Pauli representations:

```text
pauli_to_xz(pauli)
xz_to_pauli(x, z)
pauli_string_to_xz(pauli_string, n_qubits=None)
sparse_pauli_to_xz(n_qubits, qubits, paulis)
symplectic_product(x1, z1, x2, z2)
multiply_pauli_rows(left_x, left_z, left_sign, right_x, right_z, right_sign)
```

`PauliFrame` and `StabilizerState` are helper classes exposed from
`npsim.core`. They are useful for tests and low-level workflows; the packed
runtime APIs below are the normal product path.

## Forward Runtime

`BatchForwardNoiseAwareSimulator(circuit, *, observables=None)` compiles a
circuit for packed batch simulation.

Read-only attributes:

- `circuit`
- `observables`
- `locations: dict[str, NoiseLocation]`

Methods:

```text
run_batch(*, shots, rng=None, seed=None) -> BatchTrajectory
sample(shots, seed=None, rng=None) -> BatchTrajectory
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
) -> SimulationResult
```

`BatchTrajectory` stores bit-packed integer masks:

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

`SimulationResult` exposes:

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
from npsim import (
    BernoulliPauliNoise,
    BatchForwardNoiseAwareSimulator,
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

sim = BatchForwardNoiseAwareSimulator(circuit)
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

The `npsim.runtime` wrappers expose lower-level native handles while preserving
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
sample(shots, seed=None, rng=None) -> BatchTrajectory
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

## Detector Error Models

`Detector(id, measurement_keys, coords=None)` declares one detector.

Read-only attributes:

- `id: int`
- `measurement_keys: tuple[str, ...]`
- `coords: tuple[float, ...]`

`LogicalObservable(id, measurement_keys=None, pauli_qubits=None, pauli="")`
declares one logical observable. Observables can be based on measurement keys,
final Pauli-frame projection, or both.

`DetectorErrorEdge(probability, detectors, observables, location_id, event, tags=None)`
stores one DEM edge.

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
error model.

Methods:

```text
to_dem_text(*, include_detector_coords=True) -> str
edges_by_location() -> dict[str, list[DetectorErrorEdge]]
project_hotspots_to_edges(hotspots) -> dict[tuple[str, object], float]
project_result_to_detector_graph(result) -> DetectorGraphHotspots
project_sensitivities_to_detector_graph(sensitivities) -> DetectorGraphHotspots
```

`project_hotspots_to_edges(...)` expects a mapping from location id to hotspot
value. It returns edge values keyed by `(location_id, event)`.

`DetectorErrorModelGenerator(circuit, *, detectors=None, observables=None)`
generates a DEM from a circuit. If declarations are omitted, NPSim reads
`Operation.detector(...)` and `Operation.observable_include(...)` entries from
the circuit.

Example:

```python
from npsim import (
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

`DemBatchHotspotSimulator(dem)` samples directly from DEM edges.

Methods:

```text
run_batch(*, shots, rng=None, seed=None, return_edge_events=True) -> DemBatchTrajectory
estimate(
    *,
    shots,
    seed=None,
    decoder=None,
    correction_mask_fn=None,
    loss_mask_fn=None,
    baseline=None,
    top_k=10,
) -> DemHotspotResult
```

`DemBatchTrajectory` stores:

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

`DemHotspotResult` exposes:

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
from npsim import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.dem import DemBatchHotspotSimulator

dem = DetectorErrorModel(
    detectors=(Detector(0, ()),),
    observables=(LogicalObservable(0),),
    edges=(DetectorErrorEdge(0.125, (0,), (0,), "edge0", "X"),),
)

dem_sim = DemBatchHotspotSimulator(dem)
batch = dem_sim.run_batch(shots=32, seed=3)
result = dem_sim.estimate(shots=128, seed=4, top_k=1)

print(batch.detector_bit(0, 0))
print(result.top_edges(1)[0].edge_index)
```

## Callback Contracts

Forward estimate callbacks use bit-packed integer masks:

- `decoder.decode_batch_masks(batch) -> dict[int, int]`
- `correction_mask_fn(batch) -> dict[int, int]`
- `loss_mask_fn(batch) -> int`
- `loss_mask_fn(batch, corrections) -> int`

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
from npsim import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.decoders import PyMatchingBatchDecoder

dem = DetectorErrorModel(
    detectors=(Detector(0, ()),),
    observables=(LogicalObservable(0),),
    edges=(DetectorErrorEdge(0.1, (0,), (0,), "e0", "X"),),
)
decoder = PyMatchingBatchDecoder.from_dem(dem)
print(decoder.decode_batch_masks({0: 0b1010}, shots=4))
```

Stim import:

```python
from npsim.io import parse_stim_circuit

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

The Rust core crate is `npsim-core`. It is Python-independent and re-exports
its main types from the crate root:

```rust
use npsim_core::{
    BatchForwardNoiseAwareSimulator, Circuit, DemBatchHotspotSimulator,
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
- `PackedBatch`
- `DemBatch`
- `HotspotEstimate`
- `DemHotspotEstimate`
- `NpError` and `NpResult<T>`

Example:

```rust
use std::collections::HashMap;

use npsim_core::{
    BatchForwardNoiseAwareSimulator, Circuit, NoiseLocation, NoiseModel, Operation,
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

let simulator = BatchForwardNoiseAwareSimulator::new(circuit, Vec::new())?;
let batch = simulator.run_batch(1024, Some(1), true)?;
let estimate = simulator.estimate_from_loss(&batch, &batch.measurements["m0"], None, 10);
```

Validation failures return `NpError` in Rust and usually become `ValueError` or
`UnsupportedNativeCircuitError` through the public Python wrappers.
