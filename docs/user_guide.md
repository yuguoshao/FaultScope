# NPSim User Guide

NPSim is a Rust-core stabilizer simulator with a Python API. It is designed
for forward noise-aware sampling, detector error model generation, and noise
hotspot estimation in quantum error correction workflows.

This guide is task-oriented. For symbol-by-symbol signatures and attribute
lists, see [NPSim API Reference](api_reference.md). For the mathematical
score-function estimator background, see
[Forward Noise-aware Stabilizer Simulator](forward_noise_aware_stabilizer.md).

## Contents

- [What NPSim Is For](#what-npsim-is-for)
- [Installation And Build](#installation-and-build)
- [Package Layout And Imports](#package-layout-and-imports)
- [Core Mental Model](#core-mental-model)
- [Quickstart: One-qubit Hotspot Estimate](#quickstart-one-qubit-hotspot-estimate)
- [Workflow: Forward Sampling](#workflow-forward-sampling)
- [Workflow: Forward Hotspot Estimation](#workflow-forward-hotspot-estimation)
- [Workflow: Detector Error Model Generation](#workflow-detector-error-model-generation)
- [Workflow: DEM Sampling And Hotspots](#workflow-dem-sampling-and-hotspots)
- [Workflow: PyMatching Decoding](#workflow-pymatching-decoding)
- [Workflow: Stim Import](#workflow-stim-import)
- [Workflow: Repetition-code Examples](#workflow-repetition-code-examples)
- [Workflow: Visualization](#workflow-visualization)
- [Interpreting Results](#interpreting-results)
- [Packed Mask Basics](#packed-mask-basics)
- [Choosing A Workflow](#choosing-a-workflow)
- [Benchmarks](#benchmarks)
- [Rust Core Usage](#rust-core-usage)
- [Troubleshooting And FAQ](#troubleshooting-and-faq)
- [Best Practices](#best-practices)

## What NPSim Is For

NPSim estimates which local noise sources matter most for a correction
workflow. A typical output is a ranked list of noise locations with estimated
sensitivity and absolute hotspot score.

Use NPSim when you want to:

- Execute stabilizer-compatible circuits in forward time order.
- Sample bit-packed measurement, detector, observable, and noise-event masks.
- Generate detector error models from circuits.
- Estimate logical-failure sensitivity with respect to local noise rates.
- Compare forward-circuit and detector-error-model workflows.
- Integrate detector-level decoders such as PyMatching.

NPSim is not a general quantum state-vector simulator. It targets Clifford,
Pauli, reset, measurement, stochastic Pauli noise, detector, and observable
workflows.

## Installation And Build

From a source checkout, create a virtual environment and build the Python
extension with maturin:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip maturin
.venv/bin/python -m maturin develop --release
```

Optional Python integrations are installed separately:

```bash
.venv/bin/python -m pip install numpy scipy pymatching pillow stim
```

Useful verification commands:

```bash
.venv/bin/python -c "import npsim; print(npsim.Circuit)"
cargo test --workspace
.venv/bin/python -m unittest discover -s tests -q
```

If you only changed Rust extension code and want to refresh the current
checkout artifact without reinstalling metadata, use:

```bash
.venv/bin/python -m maturin develop --release --skip-install
```

Common optional dependencies:

| Feature | Packages |
| --- | --- |
| PyMatching decoding | `numpy`, `scipy`, `pymatching` |
| Visualization | `pillow` |
| Stim import/comparison | `stim` |
| Benchmarks | `numpy`, optionally `stim`, `pymatching`, `scipy` |

## Package Layout And Imports

The repository is a Cargo workspace with a Python package surface:

- `crates/npsim-core`: Python-independent typed Rust core.
- `crates/npsim-python`: PyO3 binding crate that builds
  `npsim._npsim_native`.
- `npsim/`: Python import surface, adapters, decoders, visualization, and
  helpers.

Use public Python modules in application code:

```python
from npsim import Circuit, Operation, NoiseLocation
from npsim.core import BernoulliPauliNoise, MeasurementBitFlip
from npsim.runtime import BatchForwardNoiseAwareSimulator, generate_native_dem
from npsim.dem import Detector, LogicalObservable, DemBatchHotspotSimulator
```

`npsim._npsim_native` is the private PyO3 backing module. Do not import it
directly in user code unless you are debugging the binding layer.

## Core Mental Model

### Circuits And Operations

A `Circuit` stores `n_qubits` and an ordered tuple of `Operation` objects.
Operations execute in forward time order.

```python
from npsim import Circuit, Operation

circuit = Circuit(
    n_qubits=1,
    operations=(
        Operation.measure(0, key="m0", basis="Z"),
    ),
)
```

Use operation constructors instead of manually constructing operation objects:

```python
Operation.h(0)
Operation.cx(0, 1)
Operation.noise(location)
Operation.measure(0, key="m0", basis="Z")
Operation.detector(("m0",), detector_id=0)
Operation.observable_include(0, ("m0",))
```

Measurement keys are stable string identifiers used by detectors,
observables, callbacks, decoders, and result accessors. Prefer descriptive
keys such as `"r3_c2"` or `"final_data_0"`.

### Noise Locations

Every stochastic source is a `NoiseLocation`. Its `id` must be unique within
the circuit. Its `tags` are optional metadata used by result aggregation and
visualization.

```python
from npsim import BernoulliPauliNoise, NoiseLocation

x_noise = NoiseLocation(
    id="data_x0",
    model=BernoulliPauliNoise("X"),
    rate=0.02,
    qubits=(0,),
    tags={"round": 0, "gate": "idle", "qubit": 0},
)
```

Supported Python noise models:

- `BernoulliPauliNoise("X")`, `"Y"`, `"Z"`, or a multi-qubit Pauli string.
- `SingleQubitDepolarizing()`.
- `TwoQubitDepolarizing()`.
- `PauliChannel({"X": 1.0, "Z": 0.5})`.
- `MeasurementBitFlip()`.

### Detectors, Observables, And DEMs

Detectors are parity checks over measurement keys:

```python
from npsim import Detector

detectors = (
    Detector(id=0, measurement_keys=("m0",), coords=(0.0, 0.0)),
)
```

Logical observables can be measurement-key based, final Pauli-frame based, or
both:

```python
from npsim import LogicalObservable

measurement_observable = LogicalObservable(id=0, measurement_keys=("m0",))
frame_observable = LogicalObservable(id=0, pauli_qubits=(0,), pauli="Z")
```

A detector error model, or DEM, is a compact edge model with edge
probabilities, detector flips, observable flips, source location ids, and
event labels. DEM workflows are useful when your loss and decoder are
detector-level rather than full circuit-state-level.

### Packed Masks

Batch results use Python integers as bit-packed masks. Shot `k` is stored in
bit `k`.

```python
def bit(mask: int, shot: int) -> int:
    return (int(mask) >> shot) & 1
```

This representation keeps callback data compact and fast. Most result objects
also provide helper methods such as `measurement_bit(...)`,
`detector_bit(...)`, `observable_bit(...)`, and `edge_event_bit(...)`.

### Hotspots And Sensitivities

NPSim estimates sensitivity of a loss expectation with respect to each local
noise rate. The signed `sensitivity` indicates the direction of the local
gradient. The `hotspot` score is the absolute value used for ranking.

Use signed sensitivity when you need the gradient direction. Use hotspot
ranking when you need the most influential locations or edges.

## Quickstart: One-qubit Hotspot Estimate

This complete example builds a one-qubit circuit, samples it, and estimates a
measurement-loss hotspot.

```python
from npsim import (
    BernoulliPauliNoise,
    BatchForwardNoiseAwareSimulator,
    Circuit,
    NoiseLocation,
    Operation,
)

x_noise = NoiseLocation(
    id="data_x0",
    model=BernoulliPauliNoise("X"),
    rate=0.02,
    qubits=(0,),
    tags={"round": 0, "gate": "idle", "qubit": 0},
)

circuit = Circuit(
    n_qubits=1,
    operations=(
        Operation.noise(x_noise),
        Operation.measure(0, key="m0", basis="Z"),
    ),
)

simulator = BatchForwardNoiseAwareSimulator(circuit)
batch = simulator.run_batch(shots=1024, seed=1)

result = simulator.estimate(
    shots=20_000,
    seed=2,
    loss_mask_fn=lambda batch: batch.measurements["m0"],
    top_k=5,
)

print(batch.measurement_bit("m0", 0))
print(result.logical_failure_rate)
print(result.hotspot_table(top_k=5))
```

The `loss_mask_fn` returns a packed integer. A set bit means that shot
contributes loss.

## Workflow: Forward Sampling

Forward sampling executes the full circuit and returns packed masks for
measurements, detectors, observables, noise events, and final Pauli frames.

```python
from npsim import BatchForwardNoiseAwareSimulator

simulator = BatchForwardNoiseAwareSimulator(circuit)
batch = simulator.run_batch(shots=4096, seed=123)

measurement_mask = batch.measurements["m0"]
first_shot_bit = batch.measurement_bit("m0", 0)
noise_mask = batch.noise_event_masks["data_x0"]
```

Use forward sampling when:

- Your loss depends on circuit-level measurement history or Pauli frames.
- You need custom Python callback logic.
- You want to inspect measurement records or noise-event masks directly.
- You are comparing full measurement samples against another simulator.

`run_batch(...)` and `sample(...)` both return a `BatchTrajectory`.
`sample_measurements(...)` returns only measurement masks and is useful for
throughput-oriented sampling.

## Workflow: Forward Hotspot Estimation

Forward hotspot estimation combines batch sampling with a loss mask and
score-function aggregation.

### Explicit Loss Mask

Use an explicit `loss_mask_fn` when your loss is not simply the residual of
declared logical observables.

```python
result = simulator.estimate(
    shots=20_000,
    seed=4,
    loss_mask_fn=lambda batch: batch.measurements["m0"],
    top_k=10,
)
```

The callback can accept either `batch` or `(batch, corrections)`.

### Default Logical Loss

If the simulator is constructed with observables, `estimate(...)` can compute
the default residual logical loss:

```python
from npsim import LogicalObservable

observables = (LogicalObservable(id=0, pauli_qubits=(0,), pauli="Z"),)
simulator = BatchForwardNoiseAwareSimulator(circuit, observables=observables)

result = simulator.estimate(
    shots=20_000,
    seed=5,
    decoder=None,
    top_k=10,
)
```

With no decoder, the correction map is empty. If a decoder is supplied, the
loss is computed from observable masks XOR correction masks.

### Custom Correction And Loss

Use custom callbacks when your correction representation is not provided by a
decoder object.

```python
def correction_mask_fn(batch):
    return {0: 0}

def loss_mask_fn(batch, corrections):
    observed = batch.observables.get(0, 0)
    corrected = corrections.get(0, 0)
    return observed ^ corrected

result = simulator.estimate(
    shots=20_000,
    seed=6,
    correction_mask_fn=correction_mask_fn,
    loss_mask_fn=loss_mask_fn,
    top_k=10,
)
```

Callback conventions:

- `decoder.decode_batch_masks(batch) -> dict[int, int]`.
- `correction_mask_fn(batch) -> dict[int, int]`.
- `loss_mask_fn(batch) -> int`.
- `loss_mask_fn(batch, corrections) -> int`.
- Returned integer masks use one bit per shot.

Do not supply both `decoder` and `correction_mask_fn` in the same estimate
call. Use one correction source.

## Workflow: Detector Error Model Generation

DEM generation propagates single-error effects through the circuit and returns
a detector-level model.

```python
from npsim import Detector, LogicalObservable
from npsim.runtime import generate_native_dem

detectors = (Detector(id=0, measurement_keys=("m0",)),)
observables = (LogicalObservable(id=0, measurement_keys=("m0",)),)

dem = generate_native_dem(
    circuit,
    detectors=detectors,
    observables=observables,
)

print(dem.to_dem_text())
```

You can also embed declarations in the circuit:

```python
circuit_with_declarations = Circuit(
    n_qubits=1,
    operations=(
        Operation.noise(x_noise),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
        Operation.observable_include(0, ("m0",)),
    ),
)

dem = generate_native_dem(circuit_with_declarations)
```

Use DEM generation when:

- You want a compact detector-level representation.
- You want edge-level diagnostic output.
- You want repeated detector-level sampling after one generation step.
- You want PyMatching integration.

DEM generation requires deterministic detector and observable effects in the
ideal and single-error circuits. If a referenced measurement is random in the
ideal circuit, use forward sampling or change the detector parity so it is
deterministic.

## Workflow: DEM Sampling And Hotspots

DEM sampling uses edge probabilities from a `DetectorErrorModel` instead of
executing the full circuit.

```python
from npsim.dem import DemBatchHotspotSimulator

dem_simulator = DemBatchHotspotSimulator(dem)
batch = dem_simulator.run_batch(shots=4096, seed=7)

result = dem_simulator.estimate(
    shots=20_000,
    seed=8,
    top_k=10,
)

print(batch.detector_bit(0, 0))
print(result.top_edges(top_k=5))
print(result.hotspot_table(top_k=5))
```

Use DEM sampling when:

- Your decoder consumes detector masks.
- Your loss is based on detector and observable masks.
- You want edge-level hotspot ranking.
- You want a fast path for repeated estimates after DEM generation.

DEM estimates support the same decoder and callback shape as forward
estimates, but batch objects contain detector, observable, and edge-event masks
instead of full circuit measurement/frame state.

## Workflow: PyMatching Decoding

PyMatching integration is optional and requires `numpy`, `scipy`, and
`pymatching`.

```python
from npsim.decoders import PyMatchingBatchDecoder

decoder = PyMatchingBatchDecoder.from_dem(dem)

forward_result = BatchForwardNoiseAwareSimulator(
    circuit,
    observables=observables,
).estimate(
    shots=50_000,
    seed=9,
    decoder=decoder,
)

dem_result = DemBatchHotspotSimulator(dem).estimate(
    shots=50_000,
    seed=10,
    decoder=decoder,
)
```

The DEM must be graphlike: every edge may touch at most two detectors.
Observable flips are passed to PyMatching through its faults matrix.

For custom decoders, implement:

```python
class MyDecoder:
    def decode_batch_masks(self, batch):
        return {0: 0}
```

The return value maps observable ids to correction masks.

## Workflow: Stim Import

NPSim includes a subset importer for flattened Stim text circuits.

```python
from npsim.io import parse_stim_circuit
from npsim.runtime import generate_native_dem

stim_text = """
X_ERROR(0.01) 0
M 0
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

imported = parse_stim_circuit(stim_text)
dem = generate_native_dem(
    imported.circuit,
    detectors=imported.detectors,
    observables=imported.observables,
)
```

The importer supports common Clifford gates, Pauli noise, depolarizing noise,
measurements, resets, detectors, and observable declarations. `REPEAT` blocks
are not supported by the subset importer. Flatten Stim circuits before import.

Use `load_stim_file(path)` when the circuit lives in a file.

## Workflow: Repetition-code Examples

The repetition-code builder is useful for smoke tests, tutorials, and quick
decoder experiments.

```python
from npsim import BatchForwardNoiseAwareSimulator
from npsim.experiments import make_repetition_code_experiment

experiment = make_repetition_code_experiment(
    distance=5,
    rounds=3,
    data_error_rate=0.01,
    measurement_error_rate=0.01,
)

result = BatchForwardNoiseAwareSimulator(
    experiment.circuit,
    observables=experiment.observables,
).estimate(
    shots=50_000,
    seed=11,
    decoder=experiment.decoder,
    top_k=10,
)

print(result.hotspot_table(top_k=10))
```

`data_error_rate` and `measurement_error_rate` can be scalars or mappings
keyed by `(round, index)`. This makes it easy to create intentionally uneven
noise profiles for hotspot smoke tests.

## Workflow: Visualization

Visualization helpers require Pillow.

```python
from npsim.viz import write_repetition_hotspot_heatmap

write_repetition_hotspot_heatmap(
    result,
    "repetition_hotspots.png",
    distance=5,
    rounds=3,
)
```

For repetition-code plots, noise tags should include fields such as `round`,
`operation`, `qubit`, and `check`.

For rotated surface-code plots, use:

```python
from npsim.viz import write_rotated_surface_code_spatial_hotspot_map

write_rotated_surface_code_spatial_hotspot_map(
    result,
    "surface_hotspots.png",
    distance=5,
)
```

Surface-code spatial plots expect tags such as `layout`, `role`, `row`, `col`,
`x`, and `y`. Missing layout tags will not break the estimator, but the plotter
may not be able to place hotspots on the expected geometry.

## Interpreting Results

### Forward Results

`SimulationResult` summarizes a forward estimate:

- `shots`: number of Monte Carlo shots.
- `mean_loss`: average value of the loss mask.
- `logical_failure_rate`: alias for the logical failure estimate.
- `baseline`: baseline used by the score-function estimator.
- `sensitivities`: signed gradient estimate by location id.
- `hotspots`: absolute sensitivity by location id.
- `by_qubit`, `by_round`, `by_gate`, `by_operation`: tag aggregations.
- `locations`: metadata for noise locations.
- `top_hotspots(top_k)`: ranked hotspot rows.
- `hotspot_table(top_k)`: text table for logs and reports.

Example:

```python
for row in result.top_hotspots(top_k=5):
    print(row.location_id, row.sensitivity, row.hotspot, row.tags)
```

### DEM Results

`DemHotspotResult` includes both edge-level and location-level values:

- `edge_sensitivities` and `edge_hotspots`: arrays indexed by DEM edge.
- `sensitivities` and `hotspots`: location-level maps.
- `by_detector`, `by_round`, `by_gate`, `by_operation`: aggregations.
- `detector_graph_hotspots`: projected graph-level values.
- `top_edges(top_k)`: ranked DEM edge rows.
- `top_hotspots(top_k)`: ranked location rows.
- `hotspot_table(top_k)`: text table.

Use `top_edges(...)` to debug specific DEM edges. Use `top_hotspots(...)` to
rank original noise locations.

### Sensitivity Versus Hotspot

Sensitivity is signed. A positive value means increasing the rate tends to
increase the loss under the estimator. A negative value can appear for custom
losses or finite-sample effects. Hotspot is `abs(sensitivity)` and is used for
ranking impact.

Monte Carlo estimates have variance. If two locations have similar hotspot
values, increase `shots` before treating their ordering as meaningful.

### Tag Aggregations

Aggregations are driven by `NoiseLocation.tags`. If you want per-round,
per-gate, per-qubit, or layout-aware summaries, attach those tags at circuit
construction time:

```python
tags={
    "round": 3,
    "gate": "cx",
    "operation": "syndrome_extract",
    "qubit": 7,
}
```

Tag values should be simple values such as `None`, `bool`, `int`, `float`, or
`str`.

## Packed Mask Basics

NPSim stores batch values as Python integers. Bit `k` belongs to shot `k`.

```python
def mask_bit(mask: int, shot: int) -> int:
    return (int(mask) >> shot) & 1
```

For a trajectory with `shots=4`, the mask `0b1010` means the value is true for
shots 1 and 3.

Common packed fields:

- `BatchTrajectory.measurements: dict[str, int]`.
- `BatchTrajectory.detectors: dict[int, int]`.
- `BatchTrajectory.observables: dict[int, int]`.
- `BatchTrajectory.noise_event_masks: dict[str, int]`.
- `DemBatchTrajectory.detectors: dict[int, int]`.
- `DemBatchTrajectory.observables: dict[int, int]`.
- `DemBatchTrajectory.edge_event_masks: dict[int, int]`.

Packed masks keep callback data compact. For occasional inspection, prefer
helper methods such as `measurement_bit(...)` or `edge_event_bit(...)`.

## Choosing A Workflow

| Need | Recommended workflow |
| --- | --- |
| Inspect raw measurement masks | Forward sampling |
| Custom loss over measurement history | Forward estimate with `loss_mask_fn` |
| Custom decoder over detector masks | Forward or DEM estimate with decoder |
| Graphlike matching decoder | DEM + PyMatching |
| Edge-level hotspot ranking | DEM hotspot estimate |
| Full circuit-state behavior | Forward sampling |
| Fast repeated detector-level sampling | Generate DEM once, then DEM sampling |
| Rust application integration | `npsim-core` |

Forward and DEM workflows answer related but different questions. Forward
sampling preserves more circuit-level information. DEM sampling is usually the
better fit once the analysis has been reduced to detector and observable
masks.

## Benchmarks

Run benchmarks from the repository root after building the native extension.

```bash
.venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31 --rounds 3
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20
.venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21 --rounds 3
.venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21 --rounds 3 --shots 100000
.venv/bin/python benchmarks/surface_code_threshold.py --distances 3 5 7 --shots 10000
```

Stim comparisons are reported when `stim` is installed. Threshold comparisons
require `numpy`, `scipy`, `pymatching`, and `stim`.

Benchmark output is tab-separated. Use the `status` column to distinguish
successful comparisons from optional dependency skips.
`dem_throughput.py` reports native full DEM generation, native detector-only
DEM generation, reusable native generator compile time, compiled-generator DEM
generation time, light sampler compile time, Stim DEM generation, and detector
sampling throughput. The light sampler column uses `materialize_dem=False`;
that path is intended for detector/observable sampling throughput and returns
a sampler with `dem is None`, so APIs that need full DEM metadata reject it
with `ValueError`.

## Rust Core Usage

Rust users can depend on `npsim-core` directly inside the workspace. The core
API is typed and Python-independent.

```rust
use std::collections::HashMap;

use npsim_core::{
    BatchForwardNoiseAwareSimulator, Circuit, NoiseLocation, NoiseModel, Operation,
};

let noise = NoiseLocation {
    id: "x0".to_string(),
    model: NoiseModel::BernoulliPauli("X".to_string()),
    rate: 0.02,
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
```

The Rust crate is pre-1.0. Prefer the Python API for user-facing stability
unless you specifically need Rust integration.

## Troubleshooting And FAQ

### `ModuleNotFoundError: npsim._npsim_native`

The Rust extension has not been built for the active Python environment. Run:

```bash
.venv/bin/python -m maturin develop --release
```

Make sure the same `.venv/bin/python` is used to build and run your script.

### `UnsupportedNativeCircuitError`

The public native wrappers use this error boundary for extension import,
compile, DEM generation, and sampling failures. Common causes:

- Duplicate noise location ids.
- Noise rates outside `[0, 1]`.
- Unsupported operation kinds.
- Invalid Pauli strings or qubit counts.
- DEM generation through an ideal-random measurement.
- Missing or inconsistent detector/observable measurement keys.

### DEM Generation Fails On A Random Measurement

DEM generation needs deterministic detector and observable effects in the
ideal and single-error circuits. If a measurement is intentionally random, do
not reference it directly as a detector. Add stabilizer structure that makes
the detector parity deterministic, or use forward sampling with a custom
`loss_mask_fn`.

### PyMatching Is Unavailable

Install optional dependencies:

```bash
.venv/bin/python -m pip install numpy scipy pymatching
```

If PyMatching rejects a DEM, check whether any edge touches more than two
detectors. Non-graphlike DEMs need a custom decoder or another decoding path.

### Visualization Is Unavailable

Install Pillow:

```bash
.venv/bin/python -m pip install pillow
```

Visualization helpers also expect layout tags. The estimator can run without
those tags, but plotters need them to place hotspots.

### Results Change With The Seed

Hotspot estimates are Monte Carlo estimates. Small shot counts can reorder
similar locations. Increase `shots`, fix `seed`, and compare confidence by
rerunning with independent seeds.

### The Default Loss Raises About Missing Observables

Default logical loss requires observable masks in the batch. Construct the
simulator with observables or supply `loss_mask_fn` explicitly.

### Should I Use `rng` Or `seed`?

Native runtime paths are seed-oriented. Prefer `seed=` for reproducibility.
Python RNG objects are accepted by high-level facades for compatibility, but
native samplers use deterministic Rust RNG state internally.

### Can I Import From `npsim._npsim_native`?

Avoid it in user code. The private extension module backs the public Python
classes, but public compatibility is provided through `npsim`, `npsim.core`,
`npsim.runtime`, and `npsim.dem`.

## Best Practices

- Use unique, stable noise location ids.
- Add useful tags at construction time: `round`, `gate`, `operation`, `qubit`,
  `row`, `col`, `role`, and layout coordinates when relevant.
- Keep measurement keys stable and descriptive.
- Use `seed=` for reproducible native sampling.
- Start with small `shots` for debugging and increase shots for final
  estimates.
- Use `hotspot_table(top_k)` for reports and `top_hotspots(top_k)` for
  programmatic analysis.
- Use DEM sampling for detector-level studies and forward sampling for custom
  circuit-level losses.
- Keep optional dependencies out of core workflows unless you need the
  corresponding adapter.
- Treat `npsim._npsim_native` as private; import from public modules instead.
