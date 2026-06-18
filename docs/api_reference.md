# NPSim API Reference

NPSim is a Rust Cargo workspace with a Python API. The main algorithms live in
`npsim-core`; Python users normally import from `npsim`, `npsim.core`,
`npsim.runtime`, and `npsim.dem`. The private extension module
`npsim._npsim_native` backs those public modules, but direct imports from it are
not the recommended API.

The package is currently pre-1.0. Python source-level compatibility is the main
compatibility target. The Rust core API is public and typed, but may still move
as the core package stabilizes.

## Python API

### Import Surface

Common objects are re-exported from the top-level package:

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

Lower-level modules provide the same objects grouped by subsystem:

```python
from npsim.core import Circuit, Operation, NoiseLocation, PauliFrame, StabilizerState
from npsim.runtime import compile_native_sampler, generate_native_dem
from npsim.dem import DetectorErrorModel, DemBatchHotspotSimulator
```

### Core Circuit Objects

`Circuit(n_qubits, operations)` stores a stabilizer-compatible circuit.

Read-only attributes:

- `n_qubits: int`
- `operations: tuple[Operation, ...]`

Methods:

- `noise_locations() -> dict[str, NoiseLocation]`

`NoiseLocation(id, model, rate, qubits, tags=None)` names one stochastic noise
source.

Read-only attributes:

- `id: str`
- `model: object`
- `rate: float`
- `qubits: tuple[int, ...]`
- `tags: dict[str, object]`

`Operation` can be constructed directly, but most code should use the static
constructors:

```python
Operation.h(qubit, **metadata)
Operation.s(qubit, **metadata)
Operation.s_dag(qubit, **metadata)
Operation.x(qubit, **metadata)
Operation.y(qubit, **metadata)
Operation.z(qubit, **metadata)
Operation.cx(control, target, **metadata)
Operation.cz(left, right, **metadata)
Operation.swap(left, right, **metadata)
Operation.pauli(qubits, pauli, **metadata)
Operation.noise(location, **metadata)
Operation.measure(qubit, *, key=None, basis="Z", noise=None, **metadata)
Operation.measure_pauli(qubits, pauli, *, key=None, noise=None, **metadata)
Operation.reset(qubit, *, key=None, basis="Z", **metadata)
Operation.detector(measurement_keys, *, detector_id=None, coords=None, **metadata)
Operation.observable_include(observable_id, measurement_keys, **metadata)
```

Read-only attributes include `kind`, `qubits`, `key`, `basis`, `pauli`,
`measurement_keys`, `observable_id`, `noise_location`, and `metadata`.

Example:

```python
from npsim import BernoulliPauliNoise, Circuit, NoiseLocation, Operation

x_noise = NoiseLocation(
    id="x0",
    model=BernoulliPauliNoise("X"),
    rate=0.01,
    qubits=(0,),
    tags={"round": 0, "gate": "idle"},
)

circuit = Circuit(
    n_qubits=1,
    operations=(
        Operation.noise(x_noise),
        Operation.measure(0, key="m0", basis="Z"),
    ),
)
```

### Noise Models

NPSim provides five stochastic noise models:

```python
BernoulliPauliNoise(pauli)
PauliChannel(weights)
SingleQubitDepolarizing()
TwoQubitDepolarizing(_events=None)
MeasurementBitFlip()
```

Common methods:

- `sample(rng, rate)` samples an event using a Python RNG object that provides
  `random()` and, where needed, `randrange()`.
- `score(event, rate)` returns the log-derivative score used by hotspot
  estimators.
- `apply(event, state, frame, qubits)` applies the sampled event to Python
  stabilizer/frame helpers.

Additional attributes and methods:

- `BernoulliPauliNoise.pauli`
- `PauliChannel.weights: dict[str, float]`
- `PauliChannel.event_length`
- `PauliChannel.total_weight`
- `MeasurementBitFlip.apply_to_bit(bit, event)`

### Pauli And Stabilizer Helpers

`npsim.core` exports helper functions for Pauli representation conversion:

```python
pauli_to_xz(pauli)
xz_to_pauli(x, z)
pauli_string_to_xz(pauli)
sparse_pauli_to_xz(n_qubits, qubits, pauli)
symplectic_product(x1, z1, x2, z2)
multiply_pauli_rows(left_x, left_z, right_x, right_z)
```

`PauliFrame` and `StabilizerState` are PyO3-backed Python classes used by the
runtime and by reference tests. They expose forward Clifford, Pauli, reset, and
measurement helpers for source-level Python workflows.

### Forward Runtime

`BatchForwardNoiseAwareSimulator(circuit, *, observables=None)` compiles a
circuit for packed batch simulation.

Read-only attributes:

- `circuit`
- `observables`
- `locations: dict[str, NoiseLocation]`

Methods:

```python
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

- `shots`
- `all_mask`
- `x_frame`, `z_frame`
- `measurements: dict[str, int]`
- `detectors: dict[int, int]`
- `observables: dict[int, int]`
- `noise_event_masks: dict[str, int]`

It also provides bit helpers such as `measurement_bit(key, shot)`,
`detector_bit(detector_id, shot)`, `observable_bit(observable_id, shot)`,
`x_bit(qubit, shot)`, and `z_bit(qubit, shot)`.

`SimulationResult` contains `shots`, `mean_loss`,
`logical_failure_rate`, `baseline`, `sensitivities`, `hotspots`, aggregation
maps, `top_hotspots(top_k)`, and `hotspot_table(top_k)`.

Example:

```python
from npsim import BatchForwardNoiseAwareSimulator

sim = BatchForwardNoiseAwareSimulator(circuit)
batch = sim.run_batch(shots=1024, seed=123)

loss_mask = batch.measurements["m0"]
result = sim.estimate(
    shots=4096,
    loss_mask_fn=lambda trajectory, corrections: trajectory.measurements["m0"],
    seed=123,
    top_k=5,
)

print(batch.measurement_bit("m0", 0))
print(result.hotspot_table(top_k=5))
```

### Native Runtime Wrappers

The `npsim.runtime.native` wrappers expose lower-level native handles while
preserving the public Python error boundary:

```python
compile_native_sampler(circuit, *, observables=None) -> NativePackedSampler
generate_native_dem(circuit, *, detectors=None, observables=None) -> DetectorErrorModel
compile_native_dem_sampler(dem) -> NativeDemSampler
```

These functions always use the Rust native path. They raise
`UnsupportedNativeCircuitError` when the extension cannot be imported, the
circuit cannot be compiled, DEM generation fails, or sampling is unsupported.

`NativePackedSampler` and `NativeDemSampler` are advanced handles used by the
public simulator facades. They expose packed batch and hotspot methods similar
to the facade classes, including `run_native_batch(...)` and
`estimate_hotspots(...)`.

Example:

```python
from npsim.runtime import (
    compile_native_dem_sampler,
    compile_native_sampler,
    generate_native_dem,
)

sampler = compile_native_sampler(circuit)
trajectory = sampler.sample(1024, seed=7)

dem = generate_native_dem(circuit)
dem_sampler = compile_native_dem_sampler(dem)
dem_result = dem_sampler.estimate_default(4096, seed=7, top_k=5)
```

### Detector Error Models

`Detector(id, measurement_keys, coords=None)` declares one detector.

`LogicalObservable(id, measurement_keys=None, pauli_qubits=None, pauli="")`
declares one logical observable. Observables can be measurement-key based,
Pauli-frame based, or both.

`DetectorErrorEdge(probability, detectors, observables, location_id, event, tags=None)`
stores one DEM edge.

Read-only edge attributes:

- `probability`
- `detectors`
- `observables`
- `location_id`
- `event`
- `tags`

Methods:

- `to_dem_line() -> str`

`DetectorErrorModel(detectors, observables, edges)` stores a typed detector
error model.

Methods:

```python
to_dem_text(include_detector_coords=True) -> str
edges_by_location() -> dict[str, tuple[DetectorErrorEdge, ...]]
project_hotspots_to_edges(result) -> DemHotspotResult
project_result_to_detector_graph(result) -> DetectorGraphHotspots
project_sensitivities_to_detector_graph(sensitivities) -> DetectorGraphHotspots
```

`DetectorErrorModelGenerator(circuit, *, detectors=None, observables=None)`
generates a DEM from a circuit. If detectors or observables are omitted, NPSim
uses detector and observable declarations embedded in circuit operations.

Example:

```python
from npsim import Detector, DetectorErrorModelGenerator, LogicalObservable

detectors = (Detector(id=0, measurement_keys=("m0",)),)
observables = (LogicalObservable(id=0, measurement_keys=("m0",)),)

dem = DetectorErrorModelGenerator(
    circuit,
    detectors=detectors,
    observables=observables,
).generate()

print(dem.to_dem_text())
```

### DEM Batch Hotspot Simulation

`DemBatchHotspotSimulator(dem)` samples directly from detector error model
edges.

Methods:

```python
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

`DemBatchTrajectory` exposes `shots`, `all_mask`, `detectors`, `observables`,
`edge_event_masks`, and bit helpers for detector, observable, and edge-event
masks.

`DemHotspotResult` exposes edge-level and location-level sensitivities,
detector graph projections, aggregation maps, `top_edges(top_k)`,
`top_hotspots(top_k)`, and `hotspot_table(top_k)`.

Example:

```python
from npsim.dem import DemBatchHotspotSimulator

dem_sim = DemBatchHotspotSimulator(dem)
dem_batch = dem_sim.run_batch(shots=1024, seed=5)
dem_result = dem_sim.estimate(shots=4096, seed=5, top_k=5)

print(dem_batch.detector_bit(0, 0))
print(dem_result.hotspot_table(top_k=5))
```

### Callback Contracts

Forward runtime callbacks work with bit-packed integer masks:

- `decoder.decode_batch_masks(detector_masks, *, shots=None) -> dict[int, int]`
- `correction_mask_fn(batch) -> dict[int, int]`
- `loss_mask_fn(batch, corrections) -> int`

DEM runtime callbacks use the same packed-mask convention:

- `correction_mask_fn(batch) -> dict[int, int]`
- `loss_mask_fn(batch, corrections) -> int`

The returned integer mask has one bit per shot. A set bit means the condition is
true for that shot.

### Optional Integrations

PyMatching:

```python
from npsim.decoders import PyMatchingBatchDecoder

decoder = PyMatchingBatchDecoder.from_dem(dem)
corrections = decoder.decode_batch_masks(dem_batch)
```

Stim import:

```python
from npsim.io import load_stim_file, parse_stim_circuit

imported = parse_stim_circuit("H 0\nM 0\nDETECTOR rec[-1]\n")
circuit = imported.circuit
```

Visualization helpers require Pillow and write image files:

```python
from npsim.viz import (
    write_repetition_gate_structure_hotspot_map,
    write_repetition_hotspot_heatmap,
    write_rotated_surface_code_spatial_hotspot_map,
)
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

### Data Model

Core public data types include:

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

The Rust API uses typed structs and enums instead of Python dictionaries.
Validation failures return `NpError` with human-readable messages.

### Forward Batch Sampling

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
            key: Some("m".to_string()),
            basis: "Z".to_string(),
            noise: None,
        },
    ],
};

let simulator = BatchForwardNoiseAwareSimulator::new(circuit, Vec::new())?;
let batch = simulator.run_batch(1024, Some(1), true)?;
let estimate = simulator.estimate_from_loss(&batch, &batch.measurements["m"], None, 10);
```

### DEM Generation

```rust
use npsim_core::{DetectorErrorModelGenerator, Operation};

let circuit = Circuit {
    n_qubits: 1,
    operations: vec![
        Operation::Measure {
            qubit: 0,
            key: Some("m".to_string()),
            basis: "Z".to_string(),
            noise: None,
        },
        Operation::Detector {
            detector_id: Some(0),
            measurement_keys: vec!["m".to_string()],
            coords: vec![0.0],
        },
        Operation::ObservableInclude {
            observable_id: 0,
            measurement_keys: vec!["m".to_string()],
        },
    ],
};

let dem = DetectorErrorModelGenerator::new(circuit, None, None)?.generate()?;
```

### DEM Hotspot Sampling

```rust
use npsim_core::{DemBatchHotspotSimulator, Detector, LogicalObservable};

let dem = DetectorErrorModelGenerator::new(
    circuit,
    Some(vec![Detector {
        id: 0,
        measurement_keys: vec!["m".to_string()],
        coords: Vec::new(),
    }]),
    Some(vec![LogicalObservable {
        id: 0,
        measurement_keys: vec!["m".to_string()],
        pauli_qubits: Vec::new(),
        pauli: String::new(),
    }]),
)?
.generate()?;

let simulator = DemBatchHotspotSimulator::new(dem)?;
let estimate = simulator.estimate_default(4096, Some(2), None, 10)?;
```

## Error Boundaries

Public Python wrapper functions in `npsim.runtime` convert extension import,
compile, DEM generation, and native sampling failures into
`UnsupportedNativeCircuitError`.

Direct construction of Python API objects may raise `TypeError` for wrong
signatures and `ValueError` for invalid values such as unsupported rates,
qubits, Pauli strings, or inconsistent observable declarations.

The Rust core returns `NpResult<T>` and uses `NpError` for validation and
unsupported-circuit diagnostics.
