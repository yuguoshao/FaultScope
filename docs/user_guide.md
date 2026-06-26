# FaultScope User Guide

FaultScope is a Rust-core stabilizer simulator with a Python API. It is designed for
forward noise-aware batch sampling, detector error model generation, detector
level sampling, decoder integration, and noise hotspot estimation for quantum
error correction workflows.

For exact signatures, see [FaultScope API Reference](api_reference.md). For the
score-function estimator and DEM background, see
[FaultScope Theory](theory.md).

## Contents

- [Installation And Build](#installation-and-build)
- [Imports And Package Layout](#imports-and-package-layout)
- [Core Concepts](#core-concepts)
- [Quickstart: Forward Hotspots](#quickstart-forward-hotspots)
- [Forward Sampling And Estimates](#forward-sampling-and-estimates)
- [Detector Error Models](#detector-error-models)
- [DEM Sampling And Hotspots](#dem-sampling-and-hotspots)
- [PyMatching Decoding](#pymatching-decoding)
- [Stim Import](#stim-import)
- [Repetition-code Experiments](#repetition-code-experiments)
- [Packed Mask Basics](#packed-mask-basics)
- [Choosing A Workflow](#choosing-a-workflow)
- [Benchmarks](#benchmarks)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Installation And Build

From a source checkout, create a virtual environment and install the package:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install .
```

`pip install .` reads `pyproject.toml`, installs the build dependency
`maturin>=1.7,<2` in an isolated build environment, and builds the
`faultscope._native` extension. Offline installs and `--no-build-isolation`
workflows must provide `maturin` ahead of time.

Install optional integrations only when needed:

```bash
.venv/bin/python -m pip install ".[pymatching,visualization]"
.venv/bin/python -m pip install ".[test]"
```

Useful checks:

```bash
.venv/bin/python -c "import faultscope; print(faultscope.Circuit)"
cargo test --workspace
.venv/bin/python -m unittest discover -s tests -q
```

Optional dependency groups:

| Feature | Packages |
| --- | --- |
| PyMatching decoding | `numpy`, `scipy`, `pymatching` |
| Visualization | `pillow` |
| Stim import/comparison | `stim` |
| Benchmarks | `numpy`, optionally `stim`, `pymatching`, `scipy` |

## Imports And Package Layout

The repository has three main layers:

- `crates/faultscope-core`: Python-independent Rust core.
- `crates/faultscope-python`: PyO3 binding crate for `faultscope._native`.
- `faultscope/`: public Python import surface plus adapters and examples.

Use public modules in application code:

```python
from faultscope import Circuit, NoiseLocation, Operation
from faultscope.core import BernoulliPauliNoise, PauliFrame, StabilizerState
from faultscope.runtime import FaultScopeSimulator, generate_native_dem
from faultscope.dem import Detector, LogicalObservable, DemHotspotEstimator
```

`PauliFrame` and `StabilizerState` are available from `faultscope.core`, not from
the top-level `faultscope` package. Avoid importing from `faultscope._native`
directly unless you are debugging the binding layer.

## Core Concepts

A `Circuit` stores `n_qubits` and an ordered tuple of `Operation` objects.
Operations execute in forward time order.

Use static operation constructors:

```text
Operation.h(0)
Operation.cx(0, 1)
Operation.pauli_gate((0, 1), "XZ")
Operation.noise(location)
Operation.measure(0, key="m0", basis="Z")
Operation.detector(("m0",), detector_id=0)
Operation.observable_include(0, ("m0",))
```

Every stochastic source is a `NoiseLocation`. Its `id` should be unique in the
circuit. `tags` are optional metadata used by hotspot aggregation and
visualization.

Supported noise model classes:

- `BernoulliPauliNoise("X")`, `"Y"`, `"Z"`, or a multi-qubit Pauli string.
- `SingleQubitDepolarizing()`.
- `TwoQubitDepolarizing()`.
- `PauliChannel({"X": 1.0, "Z": 0.5})`.
- `MeasurementBitFlip()`.

Detectors are parity checks over measurement keys. Logical observables can be
measurement-key based, final Pauli-frame based, or both. DEM workflows use
detector and observable masks instead of full circuit measurement history.

## Quickstart: Forward Hotspots

This example builds a one-qubit circuit, samples it, and estimates the
sensitivity of a measurement-loss mask to an X-noise rate.

```python
from faultscope import (
    BernoulliPauliNoise,
    FaultScopeSimulator,
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

simulator = FaultScopeSimulator(circuit)
batch = simulator.run_batch(shots=1024, seed=1)

result = simulator.estimate(
    shots=2048,
    seed=2,
    loss_mask_fn=lambda batch: batch.measurements["m0"],
    top_k=5,
)

print(batch.measurement_bit("m0", 0))
print(result.logical_failure_rate)
print(result.hotspot_table(top_k=5))
```

`loss_mask_fn` returns a packed integer. A set bit means the corresponding shot
contributes loss.

## Forward Sampling And Estimates

Forward sampling executes the circuit and returns packed masks for
measurements, detectors, observables, noise events, and final Pauli frames.

Use `run_batch(...)` or `sample(...)` when you need full packed trajectory
data. Use `sample_measurements(...)` when you only need measurement masks for
throughput-oriented sampling.

```python
from faultscope import (
    BernoulliPauliNoise,
    FaultScopeSimulator,
    Circuit,
    NoiseLocation,
    Operation,
)

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.1, (0,))
circuit = Circuit(1, (Operation.noise(noise), Operation.measure(0, key="m0")))

simulator = FaultScopeSimulator(circuit)
batch = simulator.sample(shots=64, seed=3)
measurement_masks = simulator.sample_measurements(shots=64, seed=3)

print(batch.measurements["m0"])
print(measurement_masks["m0"])
print(batch.noise_event_masks["x0"])
```

If the simulator is constructed with observables, `estimate(...)` can compute
the default residual logical loss. With no decoder, the correction map is
empty. With a decoder, observable masks are XORed with correction masks.

```python
from faultscope import (
    BernoulliPauliNoise,
    FaultScopeSimulator,
    Circuit,
    LogicalObservable,
    NoiseLocation,
    Operation,
)

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.1, (0,))
circuit = Circuit(1, (Operation.noise(noise), Operation.measure(0, key="m0")))
observables = (LogicalObservable(id=0, pauli_qubits=(0,), pauli="Z"),)

result = FaultScopeSimulator(
    circuit,
    observables=observables,
).estimate(shots=256, seed=4, top_k=1)

print(result.top_hotspots(1)[0].location_id)
```

Forward custom correction and loss callbacks use packed masks:

```text
decoder.decode_batch_masks(batch) -> dict[int, int]
correction_mask_fn(batch) -> dict[int, int]
loss_mask_fn(batch) -> int
loss_mask_fn(batch, corrections) -> int
```

Do not supply both `decoder` and `correction_mask_fn` in the same estimate
call.

## Detector Error Models

DEM generation propagates single-error effects through a circuit and returns a
detector-level model. You can pass detector/observable declarations explicitly
or embed them as circuit operations.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    Detector,
    LogicalObservable,
    NoiseLocation,
    Operation,
)
from faultscope.runtime import generate_native_dem

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.125, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
    ),
)
detectors = (Detector(id=0, measurement_keys=("m0",)),)
observables = (LogicalObservable(id=0, measurement_keys=("m0",)),)

dem = generate_native_dem(
    circuit,
    detectors=detectors,
    observables=observables,
)

print(dem.to_dem_text())
print(dem.edges_by_location()["x0"][0].event)
```

Embedded declarations let the circuit carry its detector and observable
semantics:

```python
from faultscope import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from faultscope.runtime import generate_native_dem

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

dem = generate_native_dem(circuit)
print(dem.to_dem_text())
```

For repeated DEM generation or sampler compilation, compile a reusable native
generator:

```python
from faultscope import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from faultscope.runtime import compile_native_dem_generator

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.125, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
    ),
)

generator = compile_native_dem_generator(circuit)
dem = generator.generate_dem()
light_sampler = generator.compile_sampler(materialize_dem=False)

print(len(dem.edges))
print(light_sampler.dem)
```

DEM generation requires deterministic detector and observable effects in the
ideal and single-error circuits. If a referenced measurement is random in the
ideal circuit, use forward sampling with a custom loss mask or change the
detector parity.

## DEM Sampling And Hotspots

DEM sampling uses edge probabilities from a `DetectorErrorModel` instead of
executing the full circuit.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    NoiseLocation,
    Operation,
)
from faultscope.dem import DemHotspotEstimator
from faultscope.runtime import generate_native_dem

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

dem = generate_native_dem(circuit)
simulator = DemHotspotEstimator(dem)
batch = simulator.run_batch(shots=64, seed=5)
result = simulator.estimate(shots=256, seed=6, top_k=1)

print(batch.detector_bit(0, 0))
print(result.edge_sensitivities[0])
print(result.top_edges(1)[0].edge_index)
```

`edge_sensitivities` and `edge_hotspots` are dictionaries keyed by DEM edge
index. `sensitivities` and `hotspots` are dictionaries keyed by original
location id.

DEM custom loss callbacks always receive both the batch and a correction map:

```text
loss_mask_fn(batch, corrections) -> int
```

If no decoder or `correction_mask_fn` is supplied, `corrections` is an empty
mapping.

`materialize_dem=False` is a sampling-only path:

```python
from faultscope import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from faultscope.runtime import compile_native_dem_sampler_from_circuit

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

light_sampler = compile_native_dem_sampler_from_circuit(
    circuit,
    materialize_dem=False,
)
batch = light_sampler.run_batch(shots=64, seed=7)

print(light_sampler.dem)
print(batch.detectors[0])
```

Light samplers return `dem is None`; APIs that require DEM metadata reject them
with `ValueError`.

## PyMatching Decoding

PyMatching integration is optional and requires `numpy`, `scipy`, and
`pymatching`. The DEM must be graphlike: every edge may touch at most two
detectors. Pure logical edges with no detectors are rejected because a matching
decoder cannot infer them from syndrome data.

FaultScope integrates with PyMatching through `PyMatchingDecoder`. The decoder
is built from a graphlike `DetectorErrorModel`, because PyMatching needs a
check matrix and logical fault matrix. After construction, the decoder can be
used in either workflow:

- Forward workflow: sample the original circuit with
  `FaultScopeSimulator`, then pass `decoder=decoder` to
  `estimate(...)`.
- DEM workflow: sample the detector error model with `DemHotspotEstimator`,
  then pass the same `decoder=decoder` to `estimate(...)`.

In other words, a DEM is needed to construct the PyMatching decoder, but the
sampling path can still be forward circuit sampling.

`PyMatchingDecoder` is the Python compatibility path. It is useful for
prototyping and for environments that only install the PyMatching Python wheel,
but FaultScope packed detector masks must still be converted through Python/NumPy
before PyMatching decodes them. For the native hot path, install the optional
`faultscope-pymatching` backend and use `NativePyMatchingDecoder`:

```bash
python -m faultscope.backends install pymatching --dry-run
```

```python
from faultscope.decoders import NativePyMatchingDecoder

decoder = NativePyMatchingDecoder.from_dem(dem)
result = sampler.estimate(shots=1024, seed=1, decoder=decoder)
```

External graphlike MWPM backends can use the same post-install mechanism. A
`faultscope-mwpm` package that registers the `mwpm` entry point can be used
through `NativeMwpmDecoder` or `create_native_decoder("mwpm", dem=dem)`.

```python
from faultscope import (
    BernoulliPauliNoise,
    FaultScopeSimulator,
    Circuit,
    NoiseLocation,
    Operation,
)
from faultscope.decoders import PyMatchingDecoder
from faultscope.dem import DemHotspotEstimator
from faultscope.runtime import generate_native_dem

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.1, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
        Operation.observable_include(0, ("m0",)),
    ),
)

dem = generate_native_dem(circuit)
decoder = PyMatchingDecoder.from_dem(dem)

forward_result = FaultScopeSimulator(circuit).estimate(
    shots=256,
    seed=10,
    decoder=decoder,
)

dem_result = DemHotspotEstimator(dem).estimate(
    shots=256,
    seed=10,
    decoder=decoder,
)

print(forward_result.mean_loss)
print(dem_result.mean_loss)
```

You can also call the decoder directly when you already have packed detector
masks:

```python
from faultscope import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.decoders import PyMatchingDecoder

dem = DetectorErrorModel(
    detectors=(Detector(0, ()),),
    observables=(LogicalObservable(0),),
    edges=(DetectorErrorEdge(0.1, (0,), (0,), "edge0", "X"),),
)

decoder = PyMatchingDecoder.from_dem(dem)
corrections = decoder.decode_batch_masks({0: 0b1010}, shots=4)

print(corrections[0])
```

Custom decoders only need to implement:

```text
decode_batch_masks(batch) -> dict[int, int]
```

The return value maps observable ids to correction masks.

## Native Decoder Fast Path

Python decoders remain supported for prototypes, but native decoder handles can
avoid moving detector and correction masks through Python. A native decoder is
still created and selected from Python:

```python
from faultscope import (
    FaultScopeSimulator,
    BernoulliPauliNoise,
    Circuit,
    LogicalObservable,
    NativeNoCorrectionDecoder,
    NoiseLocation,
    Operation,
)
from faultscope.decoders import available_native_decoders

noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.05, (0,))
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0", basis="Z"),
    ),
)
decoder = NativeNoCorrectionDecoder(observable_ids=(0,))
assert decoder.name in available_native_decoders()

result = FaultScopeSimulator(
    circuit,
    observables=(LogicalObservable(0, measurement_keys=("m0",)),),
).estimate(shots=1024, seed=1, decoder=decoder)
```

When `loss_mask_fn` and `correction_mask_fn` are omitted, this uses the native
fast path: detector masks, decoder output, default residual loss, and hotspot
aggregation all stay in Rust. If you supply a Python loss or correction
callback, FaultScope falls back to the compatibility path and calls
`decode_batch_masks(batch)`. A Python class or subclass that only implements
`decode_batch_masks(batch)` is still a Python decoder and does not enter the
native fast path.

DEM objects can also compile decoder-ready native views:

```python
indexed = dem.compile_indexed()
matching_problem = dem.compile_graphlike_problem()
binary_problem = dem.compile_binary_linear_problem()
```

`compile_graphlike_problem()` is for future MWPM/fusion-blossom style backends.
`compile_binary_linear_problem()` is for future BP+OSD/LDPC style backends.
These native views expose stable ids, counts, `edge_summary`, and compact reprs
for inspection, but intentionally avoid public `to_numpy_*` hot-path helpers.

Optional native decoder backends are managed explicitly after installing FaultScope.
Inspect the built-in backend catalog and installed backend status with:

```bash
python -m faultscope.backends status
```

Inspect backend installation steps with:

```bash
python -m faultscope.backends install pymatching --dry-run
python -m faultscope.backends install fusion-blossom --dry-run
python -m faultscope.backends install bposd --dry-run
```

For local development, activate the project virtual environment, or otherwise
ensure `.venv/bin` is on `PATH`, then install optional backend packages from the
source checkout:

```bash
.venv/bin/python -m pip install -e backends/faultscope-pymatching --no-build-isolation
.venv/bin/python -m pip install -e backends/faultscope-fusion-blossom
.venv/bin/python -c "from faultscope.decoders import available_native_decoders; print(available_native_decoders())"
```

FaultScope does not clone, compile, or install backend code during `import faultscope` or
`estimate(...)`. The default build includes only smoke-test/template native
backends, not fusion-blossom or BP+OSD. The local `faultscope-fusion-blossom`
package is a minimal serial-solver beta adapter: it enters the native PyCapsule
fast path, exposes construction metadata such as `solver_edge_count`, and
safely compresses identical boundary and two-detector parallel edges. It uses
`weight_scale=10_000` by default for the fusion-blossom integer solver and
normalizes scaled integer weights by a common even-preserving divisor without
changing that integer MWPM objective. It still rejects ambiguous parallel
logical effects and has not implemented production partitioning, streaming
execution, or production performance tuning. For local diagnostics,
`NPSIM_FUSION_BLOSSOM_THREADS=<n>` caps the packed batch worker count; by
default the backend uses available native parallelism. Set
`NPSIM_FUSION_BLOSSOM_PROFILE=1` to print the native timing split used for
backend performance diagnosis, including solver clear/growth/extraction costs.

The local `faultscope-pymatching` package links pinned PyMatching sparse-blossom C++
source and exposes `NativePyMatchingDecoder`. It is graphlike-only and keeps
hot-path detector/correction masks out of Python; the existing
`PyMatchingDecoder` remains available as the Python compatibility adapter.

For the full developer contract, including `from_circuit(...)`,
`from_dem(...)`, Python prototype decoders, and native backend skeletons, see
[Decoder Development](decoder_development.md).

## Stim Import

FaultScope includes a subset importer for flattened Stim text circuits. The importer
also embeds `DETECTOR` and `OBSERVABLE_INCLUDE` as circuit operations, so the
returned circuit can be used directly by batch and DEM workflows.

```python
from faultscope.io import parse_stim_circuit
from faultscope.runtime import generate_native_dem

stim_text = """
X_ERROR(0.01) 0
M 0
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

imported = parse_stim_circuit(stim_text)
dem = generate_native_dem(imported.circuit)

print(imported.measurement_keys)
print(dem.to_dem_text())
```

Supported instructions include common Clifford gates, Pauli gates, resets,
measurements, MPP, Pauli/depolarizing noise, `PAULI_CHANNEL_1`,
`PAULI_CHANNEL_2`, `DETECTOR`, and `OBSERVABLE_INCLUDE`. `REPEAT` blocks and
full Stim feedback/correlated-error semantics are outside the subset.

Use `load_stim_file(path)` when the circuit lives in a file.

## Repetition-code Experiments

The repetition-code builder is useful for smoke tests, tutorials, and decoder
experiments.

```python
from faultscope import FaultScopeSimulator
from faultscope.experiments import make_repetition_code_experiment

experiment = make_repetition_code_experiment(
    distance=3,
    rounds=1,
    data_error_rate=0.01,
    measurement_error_rate=0.01,
)

result = FaultScopeSimulator(
    experiment.circuit,
    observables=experiment.observables,
).estimate(
    shots=256,
    seed=8,
    decoder=experiment.decoder,
    top_k=3,
)

print(result.hotspot_table(top_k=3))
```

`data_error_rate` and `measurement_error_rate` can be scalars or mappings keyed
by `(round, index)`.

## Packed Mask Basics

FaultScope stores batch values as Python integers. Bit `k` belongs to shot `k`.

```python
def mask_bit(mask: int, shot: int) -> int:
    return (int(mask) >> shot) & 1


print(mask_bit(0b1010, 1))
print(mask_bit(0b1010, 2))
```

Common packed fields:

- `SampleBatch.measurements: dict[str, int]`
- `SampleBatch.detectors: dict[int, int]`
- `SampleBatch.observables: dict[int, int]`
- `SampleBatch.noise_event_masks: dict[str, int]`
- `DemSampleBatch.detectors: dict[int, int]`
- `DemSampleBatch.observables: dict[int, int]`
- `DemSampleBatch.edge_event_masks: dict[int, int]`

For occasional inspection, prefer helper methods such as
`measurement_bit(...)`, `detector_bit(...)`, and `edge_event_bit(...)`.

## Choosing A Workflow

| Need | Recommended workflow |
| --- | --- |
| Inspect raw measurement masks | Forward sampling |
| Custom loss over measurement history | Forward estimate with `loss_mask_fn` |
| Custom decoder over detector masks | Forward or DEM estimate with decoder |
| Graphlike matching decoder | DEM + PyMatching |
| Edge-level hotspot ranking | DEM hotspot estimate |
| Fast repeated detector-level sampling | Generate DEM once, then DEM sampling |
| Rust application integration | `faultscope-core` |

Forward and DEM workflows answer related but different questions. Forward
sampling preserves more circuit-level information. DEM sampling is usually the
better fit once the analysis has been reduced to detector and observable masks.

FaultScope's product runtime is a packed batch engine. It does not expose a general
per-shot adaptive branching simulator.

## Benchmarks

Run performance benchmarks with a release build of the native extension:

```bash
.venv/bin/maturin develop --release --skip-install
```

Then run benchmarks from the repository root:

```bash
.venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31 --rounds 3
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20
.venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21 --rounds 3
.venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21 --rounds 3 --shots 100000
.venv/bin/python benchmarks/native_decoder_fast_path.py
.venv/bin/python benchmarks/surface_code_decoder_performance.py --distances 3 5 7 --shots 10000
.venv/bin/python benchmarks/surface_code_threshold.py --distances 3 5 7 --shots 10000
```

Stim comparisons are reported when `stim` is installed. Threshold comparisons
require `numpy`, `scipy`, `pymatching`, and `stim`. The surface-code decoder
performance benchmark compares PyMatching with the optional
`faultscope-pymatching` and `faultscope-fusion-blossom` backends when those backends are
installed; unavailable native backends are reported as `skip:<reason>` rows.
The `stim-dem-pymatching-bitpacked` row uses Stim bit-packed DEM sampling plus
PyMatching bit-packed batch decode as the official-style maximum-throughput
baseline.
Pass `--split-native-baseline` to show the native no-correction packed-row
mean-loss baseline separately from the native decoder delta.
Add `--same-seed-across-paths` when comparing mean-loss differences across
native decoder configurations.

## Troubleshooting

### `ModuleNotFoundError: faultscope._native`

The Rust extension has not been built for the active Python environment. From
the repository root, run:

```bash
.venv/bin/python -m pip install .
```

Make sure the same `.venv/bin/python` is used to install and run your script.
After changing Rust extension code, reinstall with `--force-reinstall` if
needed.

### `UnsupportedNativeCircuitError`

Common causes:

- Duplicate noise location ids.
- Noise rates outside `[0, 1]`.
- Unsupported operation kinds.
- Invalid Pauli strings or qubit counts.
- DEM generation through an ideal-random measurement.
- Missing or inconsistent detector/observable measurement keys.

### DEM Generation Fails On A Random Measurement

DEM generation needs deterministic detector and observable effects in the ideal
and single-error circuits. If a measurement is intentionally random, do not
reference it directly as a detector. Add stabilizer structure that makes the
detector parity deterministic, or use forward sampling with a custom
`loss_mask_fn`.

### PyMatching Is Unavailable

Install optional dependencies:

```bash
.venv/bin/python -m pip install ".[pymatching]"
```

If PyMatching rejects a DEM, check whether any edge touches more than two
detectors or whether a pure logical edge has no detector support.

### The Default Loss Raises About Missing Observables

Default logical loss requires observable masks in the batch. Construct the
simulator with observables, embed observable declarations where appropriate, or
supply `loss_mask_fn` explicitly.

### Should I Use `rng` Or `seed`?

Prefer `seed=` for reproducible native sampling. High-level facades accept
Python RNG objects for compatibility by deriving a native seed from
`rng.getrandbits(64)`.

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
- Treat `faultscope._native` as private; import from public modules instead.
