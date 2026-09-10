# FaultScope User Guide

FaultScope is a Rust-core stabilizer simulator with a Python API. It is designed for
forward noise-aware batch sampling, detector error model generation, detector
syndrome sampling, decoder integration, and noise hotspot estimation for quantum
error correction workflows. Its DEM terminology follows the detector error model
formalism: detectors are parity constraints on measurement outcomes,
\(D\) is the detector matrix, \(\Omega\) is the measurement syndrome matrix, and
\(H=D\Omega\) is the detector error matrix.

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
- [Logical Error-Rate Collection](#logical-error-rate-collection)
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
.venv/bin/python -m pip install ".[collection]"
.venv/bin/python -m pip install ".[test]"
```

Useful checks:

```bash
.venv/bin/python -c "import faultscope; print(faultscope.Circuit)"
cargo test --workspace
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m pytest -q tests/test_threshold_analysis.py
```

Optional dependency groups:

| Feature | Packages |
| --- | --- |
| PyMatching decoding | `numpy`, `scipy`, `pymatching` |
| Visualization | `pillow` |
| Stim import/comparison | `stim` |
| Benchmarks | `numpy`, optionally `stim`, `pymatching`, `scipy` |
| Collection plotting/fitting | `numpy`, `scipy`, `matplotlib` |

## Imports And Package Layout

The repository has three main layers:

- `crates/faultscope-core`: Python-independent Rust core.
- `crates/faultscope-python`: PyO3 binding crate for `faultscope._native`.
- `faultscope/`: public Python import surface plus adapters and examples.

Use public modules in application code:

```python
from faultscope import Circuit, NoiseLocation, Operation
from faultscope.core import BernoulliPauliNoise, PauliFrame, StabilizerState
from faultscope.runtime import DemFaultScopeSimulator, FaultScopeSimulator, generate_native_dem
from faultscope.dem import Detector, LogicalObservable, DemHotspotEstimator
```

`PauliFrame` and `StabilizerState` are available from `faultscope.core`, not from
the top-level `faultscope` package. Avoid importing from `faultscope._native`
directly unless you are debugging the binding layer.

Construct a frame with validated x/z vectors or `PauliFrame.zero(...)`.
`StabilizerState` intentionally has no raw-tableau constructor; use
`StabilizerState.zero(...)`. Invalid Pauli supports and targets raise
`ValueError` before random-number generation or state mutation.

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

Detectors are parity constraints over measurement keys. A set of detector
declarations is the API representation of detector matrix rows. Logical
observables can be measurement-key based, final Pauli-frame based, or both. DEM
workflows use detector syndrome masks and logical observable flip masks instead
of full circuit measurement history.

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
empty. With a decoder, logical observable flip masks are XORed with correction
masks.

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
detector error model. Conceptually, the generator computes the measurement
syndrome matrix \(\Omega\), multiplies it by the detector matrix \(D\), and
materializes columns of the detector error matrix \(H=D\Omega\) as DEM edges
with optional logical observable flips. You can pass detector/observable
declarations explicitly or embed them as circuit operations.

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
artifact = generator.generate_artifact()
dem = artifact.dem
light_sampler = generator.compile_sampler(materialize_dem=False)

print(len(dem.edges))
print(artifact.graphlike_hints)  # Native circuit generation currently returns None.
print(light_sampler.dem)
```

`GeneratedDetectorErrorModel` keeps one canonical sampling model and an
optional, sparse `GraphlikeDecompositionHints` sidecar. Use
`artifact.compile_graphlike_problem()` for decoder construction and
`artifact.dem` for sampling. Stim `^` groups can populate this sidecar; native
circuit generation does not synthesize such hints yet. Consequently, an
unhinted edge touching more than two detectors is rejected by graphlike MWPM
compilation instead of being split silently.

DEM generation requires deterministic detector and observable effects in the
ideal and single-error circuits. If a referenced measurement is random in the
ideal circuit, use forward sampling with a custom loss mask or change the
detector parity.

Circuit-to-DEM noise conversion follows Stim's policy:

- `SingleQubitDepolarizing` and `TwoQubitDepolarizing` use an exact independent
  reparameterization, so DEM sampling preserves the detector/observable joint
  distribution. Exact conversion rejects rates above `3/4` and `15/16`,
  respectively; use forward sampling above those mixing limits.
- A one-qubit `PauliChannel` first attempts Stim's conversion into independent
  X/Y/Z mechanisms. A single positive component is trivially exact, and some
  multi-component channels are also accepted when this conversion succeeds.
  A channel that cannot use this path is categorical and is rejected by
  default. `PAULI_CHANNEL_2` and wider FaultScope channels do not have a general
  exact factorization attempt.
- To accept Stim's independent approximation, pass
  `approximate_disjoint_errors=True` to `DetectorErrorModelGenerator`,
  `generate_native_dem`, `compile_native_dem_generator`,
  `compile_native_dem_sampler_from_circuit`, or `DemFaultScopeSimulator`.
  A float such as `0.01` enables the approximation only when every component
  probability is at most that threshold.

After propagation, mutually exclusive Pauli outcomes with identical
detector/observable effects are first combined exactly by summing their
probabilities. Distinct effect classes are then approximated as independent, so
two originally exclusive classes can occur in one shot with probability equal
to the product of their class probabilities. FaultScope does not currently
store categorical or correlated error groups in `DetectorErrorModel`. Use
forward sampling when that joint distribution must remain exact. DEM location
hotspot values are summaries of independent edge derivatives; they are not
automatically chain-rule derivatives with respect to the original
depolarizing/channel rate.

As in Stim 1.16, the one-qubit conversion is numerical: it accepts an X/Y/Z
independent representation when the reconstructed disjoint probabilities have
total absolute residual below `1e-14`. Consequently, a sufficiently small
non-factorable one-qubit channel can be treated as numerically exact even when
`approximate_disjoint_errors=False`; tiny effects below that absolute tolerance
may disappear from the DEM. Use forward sampling when those probabilities are
material.

## DEM Sampling And Hotspots

DEM sampling uses edge probabilities from a detector error model instead of
executing the full circuit. Use `DemFaultScopeSimulator(circuit)` when you want a
`FaultScopeSimulator`-shaped entry point that compiles the circuit to a DEM
sampler internally.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    DemFaultScopeSimulator,
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

simulator = DemFaultScopeSimulator(circuit)
batch = simulator.run_batch(shots=64, seed=5)
result = simulator.estimate(shots=256, seed=6, top_k=1)

print(batch.detector_bit(0, 0))
print(result.edge_sensitivities[0])
print(result.top_edges(1)[0].edge_index)
```

`DemFaultScopeSimulator` returns a `DemSampleBatch`, not a forward
`SampleBatch`: it contains detector masks, observable masks, and optional DEM
edge-event masks, but no measurement record or Pauli-frame masks. Internally, it
uses the same Rust path as `compile_native_dem_sampler_from_circuit(...)`.

Hotspot estimation requires the per-location or per-edge event masks used to
correlate noise events with failed shots. A native batch may be passed only to
the same compiled sampler that produced it; missing event recording, a foreign
sampler batch, zero shots, or an invalid loss-mask width raises `ValueError`
instead of producing an estimate.

If you already have a `DetectorErrorModel`, use `DemHotspotEstimator(dem)`:

```python
from faultscope.dem import DemHotspotEstimator
from faultscope.runtime import generate_native_dem

dem = generate_native_dem(circuit)
dem_simulator = DemHotspotEstimator(dem)
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
from faultscope.runtime import DemFaultScopeSimulator, compile_native_dem_sampler_from_circuit

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

The same light path is available through the high-level simulator:

```python
light_simulator = DemFaultScopeSimulator(circuit, materialize_dem=False)
batch = light_simulator.run_batch(shots=64, seed=7)
```

Light samplers and light simulators return `dem is None`; APIs that require DEM
metadata reject them with `ValueError`.

## Logical Error-Rate Collection

`faultscope.collection` is the threshold-style sampling entry point. It collects
`shots`, `errors`, `discards`, and elapsed seconds by executing one or more
circuits directly with the packed Forward runtime. The hot path stays native:
Rust owns circuit sampling, native decoder calls, batch scheduling,
postselection, and counting. Python owns task construction, strong ids, CSV
resume files, and `TaskStats` wrappers.

Create a circuit task and configure a reusable `Collector`:

```python
from faultscope import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from faultscope.collection import (
    Collector,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
)

circuit = Circuit(
    1,
    (
        Operation.noise(
            NoiseLocation("logical_x", BernoulliPauliNoise("X"), 0.125, (0,))
        ),
        Operation.measure(0, key="m"),
        Operation.detector(("m",), detector_id=0),
        Operation.observable_include(0, ("m",)),
    ),
)

collector = Collector(
    options=CollectionOptions(
        max_shots=10_000,
        min_shots=2_000,
        max_errors=200,
        batch_size=1_000,
    ),
    run_options=CollectionRunOptions(seed=1),
)
stats = collector.collect(
    [CollectionTask(circuit=circuit, task_id="p=0.125", metadata={"p": 0.125})]
)[0]

print(
    stats.shots,
    stats.errors,
    stats.raw_error_rate,
    stats.logical_error_rate,
    stats.logical_error_rate_stderr,
)
```

`CollectionTask` requires `circuit`. `detectors=None` and `observables=None`
retain declarations embedded in the circuit; an explicit sequence replaces the
corresponding embedded declarations, and an explicit empty sequence clears
them. The collection adapter derives detector and observable ids from the
compiled sampler once, preserving declaration order for decoder validation,
postselection, and custom counters. Collection does not generate a DEM when no
decoder is used or when an already constructed native decoder is supplied. A
decoder name may generate one cached DEM during task preparation solely to
construct its static decoding problem; every shot still executes the Forward
circuit.

`max_shots` is required after combining the Collector's base options with each
task's `collection_options`. The final batch is capped so collection never exceeds
`max_shots`. `max_errors` stops only after both it and `min_shots` are reached;
the returned error count can exceed the threshold by one committed batch.

Task options override only fields explicitly passed to `CollectionOptions`.
An empty `CollectionOptions()` inherits every Collector default. Explicit
defaults such as `batch_size=10_000` and `min_shots=0` reset inherited values;
explicit `None` clears nullable values, for example `max_errors=None` disables
an inherited error limit and `max_batch_seconds=None` disables inherited
adaptive batching. Collection count options reject booleans and fractional
values, and adaptive batch seconds must be finite and positive.

Derive an existing option set with `options.with_edits(...)`; supplied fields
remain explicit even when equal to their defaults. Do not use
`dataclasses.replace()` with `CollectionOptions`, because it cannot preserve the
explicit-override information.

Use `collect_hotspots(...)` (or `Collector.collect_hotspots(...)`) when the same
run must also return ordered per-batch `TaskStats` and shot-weighted physical
location sensitivities in `HotspotCollectionResult.location_sensitivities`.
This path records Forward noise-event masks and keeps decoding, residual
counting, and sensitivity calculation inside Rust. CSV partial resume is
intentionally unsupported for hotspot collection.

Native decoders can be passed directly or resolved by name. A string decoder
uses a circuit-derived DEM only as its construction-time static problem:

```python
from faultscope.collection import CollectionTask, collect

stats = collect(
    [CollectionTask(circuit=circuit, decoder="graphlike-detector-copy")],
    options=CollectionOptions(max_shots=10_000, batch_size=1_000),
    run_options=CollectionRunOptions(seed=3),
)
```

The default collection API is native-only. Python decoders are still useful for
`estimate(...)` prototypes, but `faultscope.collection` rejects them because it
does not move detector batches or correction masks through Python.

### Legacy Explicit DEM Collection

Direct DEM sampling is intentionally isolated from the main collection API and
has no CLI command. Existing DEM workflows can use the legacy library module:

```python
from faultscope.collection import CollectionOptions, CollectionRunOptions
from faultscope.collection.dem import DemCollectionTask, DemCollector

dem_stats = DemCollector(
    options=CollectionOptions(max_shots=10_000, batch_size=1_000),
    run_options=CollectionRunOptions(seed=3),
).collect([DemCollectionTask(dem=dem)])
```

The legacy module also provides `collect`, `iter_collect`, `iter_progress`, and
`collect_hotspots`; its hotspot result retains `edge_sensitivities`. DEM task
types are not re-exported from `faultscope.collection` or top-level
`faultscope`. Passing `dem=` to `CollectionTask` raises a migration error that
points to `faultscope.collection.dem.DemCollectionTask`.

Use `num_workers` to enable the Rust global worker pool. With fixed-size
batches, one large task and many independent tasks both share the same worker
cap. With a fixed seed and fixed batch settings, `num_workers=1`, `2`, and `4`
produce the same `shots`, `errors`, `discards`, and `custom_counts`. If
`max_batch_seconds` is set, each adaptive task runs two or three serial
calibration batches, freezes the estimated batch size, and sends the remaining
work through the same global worker pool. Adaptive runs preserve ordered
commits and shot/stop limits, but elapsed-time feedback means different worker
counts or machine loads need not produce identical error counts for the same
seed.

```python
stats = collect(
    tasks,
    options=CollectionOptions(max_shots=100_000, batch_size=2_000),
    run_options=CollectionRunOptions(seed=4, num_workers=4),
)
```

Resume files are CSV files managed in Python. `save_resume_filepath` reads any
existing file, skips tasks already satisfying their stop condition, and appends
only newly collected deltas. `existing_data_filepaths` contributes prior data to
stop decisions without appending to those files.

```python
stats = collect(
    tasks,
    options=CollectionOptions(max_shots=100_000, batch_size=5_000),
    run_options=CollectionRunOptions(
        seed=5,
        existing_data_filepaths=("previous_threshold.csv",),
        save_resume_filepath="threshold_resume.csv",
    ),
)
```

The CSV header is:

```text
shots,errors,discards,seconds,decoder,strong_id,json_metadata,json_counter_schema,custom_counts
```

This is the collection v3 CSV contract. `json_counter_schema` records the full
versioned counter schema for every total and batch delta. Collection rejects a
v2 header when reading or appending; it does not migrate or append v3 rows to an
old file. Archive v2 files or convert them separately before choosing a v3
resume path. CSV rows do not store the display-only `task_id`, so direct CSV
reads temporarily expose `strong_id` as the task id. A collection resume
rebinds validated history to the current `CollectionTask.task_id` before stop
checks and returns the current label even when the saved task is already
complete.

`iter_collect(...)` yields only final `TaskStats`. Use `iter_progress(...)` to
receive committed batch deltas. Resume CSV rows are appended and flushed before
each `Progress(new_stats=(...), status_message=...)` value is yielded:

```python
from faultscope.collection import (
    CollectionOptions,
    CollectionRunOptions,
    Progress,
    iter_progress,
)

for progress in iter_progress(
    tasks,
    options=CollectionOptions(max_shots=100_000, batch_size=2_000),
    run_options=CollectionRunOptions(
        save_resume_filepath="threshold_resume.csv",
    ),
):
    assert isinstance(progress, Progress)
    for delta in progress.new_stats:
        print(delta.task_id, delta.shots, delta.errors)
```

Use the public CSV utilities when you want to summarize, merge, or write
collection data yourself:

```python
from faultscope.collection import (
    CollectionData,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)

stats = read_stats_from_csv_files("old.csv", "new.csv")
data = CollectionData(stats)
write_stats_to_csv_file("merged.csv", data.values())
```

### Seeds, Repeats, and Resume Identity

Collection uses the `explicit-v1` seed policy. `CollectionTask.seed` and
`DemCollectionTask.seed` override `CollectionRunOptions.seed`. A task seed of
`None` inherits the run seed unchanged. Both seed fields accept `None` or a
non-boolean integer from `0` through `2**64 - 1`. When the run seed is `None`,
Rust draws one random root seed for that collection run; tasks without an
explicit seed inherit that root. An explicit task seed still takes precedence
in an otherwise unseeded run.

For a reproducible run, fix the seed and batch settings:

```python
stats = collect(
    [CollectionTask(circuit=circuit)],
    options=CollectionOptions(max_shots=10_000, batch_size=1_000),
    run_options=CollectionRunOptions(seed=41, num_workers=4),
)
```

The effective task seed is a root for the existing batch and resume stream
derivation; it is not passed unchanged to every batch. A direct sampler call
with the same numeric seed therefore need not produce the same shots. Batch
ordinals and the resume stream continue to select deterministic substreams. With unchanged
sampling input, fixed batch settings, and the same continuation state, task
order and worker count do not change the sampled stream. A resumed collection
uses its continuation stream, so splitting a run into resume sessions does not
promise the same shot sequence as one uninterrupted run.

Metadata, decoder identity, and backend fingerprints do not alter this seed
selection. Tasks with identical sampling input, effective seed, batching, and
continuation state share the same underlying noise stream even if their labels
or decoders differ. Different decoders can still produce different corrections
and logical error counts; different postselection can also change discards and
accepted-shot counts. Changing metadata alone does not create independent
repetitions. Give repeats different task seeds:

```python
repeat_stats = collect(
    [
        CollectionTask(circuit=circuit, task_id="repeat-1", seed=101),
        CollectionTask(circuit=circuit, task_id="repeat-2", seed=102),
    ],
    options=CollectionOptions(max_shots=10_000, batch_size=1_000),
    run_options=CollectionRunOptions(num_workers=4),
)
```

Earlier collection versions mixed the run seed with a stable hash of each
`sampling_id` when no task seed was set. This automatically separated tasks
without depending on their list positions or worker assignment. Because the
id covered decoder fingerprints and metadata, changing either also changed the
noise stream. An explicit Rust task seed already bypassed that mix. This
deterministic identity salt was separate from unseeded-run randomness: commit
`4a72ce1` retained identity-based splitting while resolving
an omitted run seed to one random root per run. `explicit-v1` removes the
identity salt and makes independent repeats an explicit seed choice; it keeps
unseeded runs random and preserves batch/resume stream derivation.

Python collection still separates an internal `sampling_id` (schema 2) from the
public `strong_id` (schema 5). Despite its historical name, `sampling_id` is
now a configuration fingerprint, not an RNG input. Forward source payloads
use the `forward_circuit` kind and contain the circuit plus the
`None`/explicit detector and observable declarations. Explicit DEM tasks use
their distinct DEM source payload. The sampling id also hashes the resolved
decoder fingerprint, metadata, and postselection masks.

The strong id hashes the sampling id, full counter schema, configured
`task_seed` (`None` or an integer), and `seed_policy="explicit-v1"`. Thus an
explicit task seed changes resume identity, even if it equals the run seed a
task would otherwise inherit. Changing either count flag also starts a
separate identity while preserving seeded samples. Mapping keys are sorted,
and metadata must be JSON serializable. The run seed, display-only `task_id`,
shot/error limits, batch sizing, worker count, and a legal
`custom_error_count_key` remain excluded from identity. Excluding the run seed
preserves aggregation across runs of the same configuration. For independent
repetitions, change the run seed for tasks whose `seed` is `None`; for tasks
with an explicit seed, change that task seed instead.

The CSV header remains v3, but earlier Python-generated strong ids do not match
schema-5 tasks. Their rows are not reused as resume history for the new seed
policy. Use a new resume path when upgrading. Matching explicit seeds does not guarantee
shot-for-shot agreement after changes to dependencies, sampling algorithms, or
batch settings. Removing identity salt does not reconstruct samples from a
previously completed experiment; reproducing such a run requires its original
implementation, seed policy, and settings.

These identity schemas are generated by the Python collection adapter. Public
Rust collection callers supply their own `strong_id` strings and must assign
new identities when moving from the old seed policy to `explicit-v1`; an
arbitrary caller-provided old id is not automatically invalidated.

Expanded tasks in one logical collection call must have unique `strong_id`
values. Collection rejects a duplicate before starting sampling workers,
emitting progress, or appending resume data. `task_id` is only a display label,
so changing it does not create a separate collection identity. Distinct task
seeds allow otherwise identical repeats in one call. Different circuits,
declaration overrides, decoders, metadata, postselection masks, and counter
schemas also distinguish identities, but do not by themselves request distinct
random streams. A resume CSV produced by an older version from duplicate
identities cannot be separated reliably and should be regenerated.

Decoder objects must provide `strong_id_payload()` returning a JSON-serializable
mapping. Bundled native decoders and official backend packages implement this
protocol. The payload covers the effective backend, implementation fingerprint
version, normalized parameters, detector/observable layout, solver problem, and
all composite children. Collection rejects opaque decoder objects before
sampling because their results cannot be identified for safe resume and merge.

The collection counter schema is exposed as the frozen
`CollectionCounterSchema` type. Its current `schema_version` is `1`, and its two
boolean fields mirror `count_observable_error_combos` and
`count_detection_events`. Native `TaskStats` always carries this schema.
`counter_schema=None` is allowed only for manually constructed analysis data;
such stats can retain arbitrary custom counters but cannot be resumed. Counter
schema versions increase whenever counter meaning, key encoding, or coverage
changes. Resume identity/CSV contract versions increase when identity or file
layout changes.

Pre-v3 collection resume files cannot be migrated safely in place. After
upgrading, archive an old CSV and start a new file. Resume requires an exact v3
schema match and rejects a missing or unsupported schema, a mixed schema under
one strong id, or a missing fixed detection counter. It does not backfill a
counter subset or superset.

Postselection masks are bytes-like bit-packed masks over the sampler's canonical
detector or observable declaration order. A fired postselected detector
discards the shot before logical error counting. A nonzero residual on a
postselected observable also discards the shot. Logical error rate uses accepted
shots:

```python
task = CollectionTask(
    circuit=circuit,
    postselection_mask=bytes([0b0000_0001]),
    postselected_observables_mask=None,
)
```

Optional custom counts are accumulated in `TaskStats.custom_counts`:

```python
stats = collect(
    [CollectionTask(circuit=circuit)],
    options=CollectionOptions(max_shots=10_000, batch_size=1_000),
    run_options=CollectionRunOptions(
        count_observable_error_combos=True,
        count_detection_events=True,
    ),
)[0]

print(stats.custom_counts)
```

`count_observable_error_combos=True` records accepted residual observable masks
with keys such as `obs_mistake_mask=E_E__`. `count_detection_events=True`
records total detection events and detector checks. When enabled, every native
stat and progress delta includes both `detection_events` and
`detectors_checked`, even when either value is zero.

`custom_error_count_key` selects the `max_errors` stop counter:

- `None` uses the main logical `errors` count.
- `detection_events` and `detectors_checked` require
  `count_detection_events=True`; missing fixed keys are errors.
- `obs_mistake_mask=<mask>` requires
  `count_observable_error_combos=True`. The mask must match every expanded
  task's observable width, contain only `E` and `_`, include at least one
  non-postselected `E`, and never mark a postselected observable with `E`.
  A legal combo absent from `custom_counts` means zero occurrences.

Every nonempty key is validated even when `max_errors` is not set. Invalid keys
fail before sampling, CSV writes, or native worker startup. Fixed, parallel,
adaptive, streaming, and hotspot collection share these rules.

Decoder fanout expands tasks that do not already specify a decoder:

```python
stats = collect(
    [CollectionTask(circuit=circuit, task_id="surface-d3")],
    options=CollectionOptions(max_shots=20_000, batch_size=1_000),
    run_options=CollectionRunOptions(
        decoders=("no-correction", "graphlike-detector-copy"),
    ),
)
```

Tasks with their own `decoder=` keep it. When multiple fanout decoders are
provided, generated task ids append the decoder name, for example
`surface-d3:graphlike-detector-copy`.

The module also has a small Forward-only command line interface. The task
factory must return `CollectionTask(circuit=...)`; there is no DEM CLI:

```bash
python -m faultscope.collection collect \
  --tasks-factory experiments.threshold:make_tasks \
  --factory-arg distance=5 \
  --max-shots 100000 \
  --batch-size 2000 \
  --decoder graphlike-detector-copy \
  --save-resume-filepath stats.csv

python -m faultscope.collection summarize stats.csv
python -m faultscope.collection merge merged.csv stats-a.csv stats-b.csv
python -m faultscope.collection fit stats.csv --x-key p --group-key d
python -m faultscope.collection plot stats.csv --x-key p --group-key d --out plot.png
python -m faultscope.collection threshold stats.csv \
  --x-key p \
  --distance-key d \
  --series-key decoder \
  --bootstrap-samples 50 \
  --format text \
  --plot-out threshold.png
```

Task factories are called as `factory(**factory_kwargs)` and may return either
a single `CollectionTask` or an iterable of `CollectionTask` values. Each
`--factory-arg key=value` value is JSON-decoded when possible, so inputs such
as numbers, booleans, arrays, and objects arrive at the factory as structured
Python values.

Use `threshold` when a CSV contains a sweep over physical error rate and code
distance. Text output has a stable tabular header plus pairwise and global
diagnostic rows. JSON output is an array of `ThresholdAnalysisResult.to_dict()`
objects. Statistical statuses such as `no_crossing`, `ambiguous`,
`insufficient_data`, `fit_failed`, and `bootstrap_unstable` are ordinary
diagnostics and return exit code 0; invalid CSV, invalid arguments, or missing
optional dependencies return a nonzero subprocess exit.

The same analysis is available in Python:

```python
from faultscope.collection import (
    analyze_thresholds,
    plot_threshold_analysis,
    read_stats_from_csv_files,
)

stats = read_stats_from_csv_files("stats.csv")
results = analyze_thresholds(
    stats,
    x_key="p",
    distance_key="d",
    series_keys=("decoder",),
    bootstrap_samples=50,
)
plot_threshold_analysis(results, output="threshold.png")
```

The public threshold points report the raw rate `errors / accepted_shots` and
its binomial standard error. Crossing interpolation, scaling logits, and
bootstrap resampling apply a continuity correction internally so records with
zero or all accepted shots in error remain numerically finite.
When `plot_threshold_analysis(..., log_y=True)` receives a zero-rate point, the
rate panel uses a symmetric-log axis so the raw zero remains visible.

Plotting and finite-size scaling require the optional collection extra:

```bash
python -m pip install "faultscope[collection]"
```

## PyMatching Decoding

PyMatching integration is optional and requires `numpy`, `scipy`, and
`pymatching`. The DEM must be graphlike: every edge may touch at most two
detectors. Pure logical edges with no detectors are rejected because a matching
decoder cannot infer them from syndrome data.

FaultScope integrates with PyMatching through `PyMatchingDecoder`. The decoder
is built from a graphlike `DetectorErrorModel`, because PyMatching needs the
detector error matrix \(H\) and a logical fault matrix. After construction, the
decoder can be used in either workflow:

- Forward workflow: sample the original circuit with
  `FaultScopeSimulator`, then pass `decoder=decoder` to
  `estimate(...)`.
- DEM workflow: sample the detector error model with `DemHotspotEstimator`,
  then pass the same `decoder=decoder` to `estimate(...)`.

In other words, a DEM is needed to construct the PyMatching decoder, but the
sampling path can still be forward circuit sampling.

`PyMatchingDecoder` is the Python compatibility path. It is useful for
prototyping and for environments that only install the PyMatching Python wheel,
but FaultScope packed detector syndrome masks must still be converted through
Python/NumPy before PyMatching decodes them. For the native hot path, install the optional
`faultscope-pymatching` backend and use `NativePyMatchingDecoder`:

```bash
python -m faultscope.backends install pymatching --dry-run
```

```python
from faultscope.decoders import NativePyMatchingDecoder

decoder = NativePyMatchingDecoder.from_dem(dem)
result = sampler.estimate(shots=1024, seed=1, decoder=decoder)
```

The `mwpm` proxy is currently unavailable. Its external `faultscope-mwpm`
package still implements ABI v1 and is not yet migrated to the strict
FaultScope native decoder ABI V4, so it remains non-installable. `bpdecoder` is
also temporarily non-installable while awaiting V4 migration. `bposd` is a
reserved, unimplemented, non-installable catalog/status entry. The generic
install subcommand accepts these names but only reports their unavailability;
it returns no install plan or steps and installs nothing.

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
fast path: detector syndrome masks, decoder output, default residual loss, and
hotspot aggregation all stay in Rust. If you supply a Python loss or correction
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

`compile_graphlike_problem()` is for MWPM/fusion-blossom style backends. Its
recommended hinted path packages
`GraphlikeDecompositionHints(dem, components_by_edge)` inside a
`GeneratedDetectorErrorModel`; calling `artifact.compile_graphlike_problem()`
then compiles canonical hyperedges into graphlike decoder components. The raw
`decomposition={dem_edge_index: ((detector_ids, observable_ids), ...)}` argument
remains available as a low-level compatibility API.
The components must XOR back to the parent support and retain the parent's
probability/index; they are never sampled independently.
`compile_binary_linear_problem()` is for future BP+OSD/LDPC style backends.
These native views expose stable ids, counts, `edge_summary`, and compact reprs
for inspection, but intentionally avoid public `to_numpy_*` hot-path helpers.

Optional native decoder backends are managed explicitly after installing FaultScope.
Inspect the built-in backend catalog and installed backend status with:

```bash
python -m faultscope.backends status
```

Inspect backend installation steps, or check the unavailable `bpdecoder` and
reserved `bposd` entries with the same generic subcommand:

```bash
python -m faultscope.backends install pymatching --dry-run
python -m faultscope.backends install fusion-blossom --dry-run
python -m faultscope.backends install bpdecoder --dry-run
python -m faultscope.backends install bposd --dry-run
```

The PyMatching and fusion-blossom commands produce install plans. The
`bpdecoder` command reports that its package is temporarily disabled until it
migrates to native decoder ABI V4; it intentionally returns no install plan or
package command. The `bposd` command likewise reports that the reserved entry
is unavailable and installs nothing.

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
execution, or production performance tuning. Its public Python object is a
factory handle; each private collection worker owns one exclusive mutable solver
while immutable graph data is shared. The backend needs no decoder mutex or
worker pool. Set
`NPSIM_FUSION_BLOSSOM_PROFILE=1` for local diagnostics that print the native
timing split, including solver clear/growth/extraction costs.

The local `faultscope-pymatching` package links pinned PyMatching sparse-blossom C++
source and exposes `NativePyMatchingDecoder`. It is graphlike-only and keeps
hot-path detector/correction masks out of Python; the existing
`PyMatchingDecoder` remains available as the Python compatibility adapter.
Both optional native matching packages expose
`from_graphlike_problem(problem, *, options=None)` for a precompiled view;
`from_dem(...)` delegates through the ordinary no-hint compilation path.

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
full Stim feedback/correlated-error semantics are outside the subset. Imported
`PAULI_CHANNEL_1/2` remains categorical in forward sampling. Circuit→DEM first
attempts Stim's one-qubit independent conversion for `PAULI_CHANNEL_1`; other
multi-component cases are rejected unless `approximate_disjoint_errors` is
explicitly enabled.

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
| Custom decoder over detector syndrome masks | Forward or DEM estimate with decoder |
| Graphlike matching decoder | DEM + PyMatching |
| Threshold-style logical error rates | `faultscope.collection.collect` |
| Edge-level hotspot ranking | `DemFaultScopeSimulator` or DEM hotspot estimate |
| Fast repeated detector syndrome sampling | `DemFaultScopeSimulator` or generate DEM once, then DEM sampling |
| Rust application integration | `faultscope-core` |

Forward and DEM workflows answer related but different questions. Forward
sampling preserves more circuit-level information. DEM sampling is usually the
better fit once the analysis has been reduced to detector syndrome and logical
observable flip masks.

FaultScope's product runtime is a packed batch engine. It does not expose a general
per-shot adaptive branching simulator.

## Benchmarks

Run performance benchmarks with a release build of the native extension:

```bash
.venv/bin/maturin develop --release --skip-install
```

Then run benchmarks from the repository root:

```bash
.venv/bin/python benchmarks/compiler_throughput.py --distances 5 10 15 20 --input-form repeat --representation m-plus-r --json-out compiler-throughput.json
.venv/bin/python benchmarks/sampling_throughput.py --distances 15 21 31 --rounds 3
.venv/bin/python benchmarks/sampling_throughput.py --family random-clifford --qubits 128 256 512 --depth 20
.venv/bin/python benchmarks/dem_throughput.py --distances 9 13 21 --rounds 3
.venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21 --rounds 3 --shots 100000
.venv/bin/python benchmarks/collection_throughput.py --shots 10000 --batch-size 1000 --adaptive-start-batch-size 100 --adaptive-max-batch-size 1000 --max-batch-seconds 0.25 --workers 1 2 4
.venv/bin/python benchmarks/native_decoder_fast_path.py
.venv/bin/python benchmarks/surface_code_decoder_performance.py --distances 3 5 7 --shots 10000
.venv/bin/python benchmarks/surface_code_threshold.py --distances 3 5 7 --shots 10000
```

Stim comparisons are reported when `stim` is installed. Threshold comparisons
require `numpy`, `scipy`, `pymatching`, and `stim`. The surface-code decoder
performance benchmark compares PyMatching with the optional
`faultscope-pymatching` and `faultscope-fusion-blossom` backends when those backends are
installed; unavailable native backends are reported as `skip:<reason>` rows.
It samples one canonical FaultScope edge per Stim `error` instruction. Stim `^`
separator groups are used only to compile the decoder problem, whose current
matching interpretation is an uncorrelated graphlike approximation.
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

Default logical loss requires logical observable flip masks in the batch. Construct the
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
- Use DEM sampling for detector-syndrome studies and forward sampling for custom
  circuit-level losses.
- Treat `faultscope._native` as private; import from public modules instead.
