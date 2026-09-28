# Getting Started

This tutorial runs a distance-3 surface-code memory experiment: construct a noisy
circuit, sample detector syndromes, decode them, and estimate the logical failure
rate. It also shows how the same run identifies sensitive noise locations.

Three terms used throughout the guides:

- A **shot** is one execution of the noisy circuit.
- A **detector** reports a measurement parity. Choose checks that are 0 without
  noise; a value of 1 means the check fired.
- A **logical failure** occurs when the decoder's predicted logical flips
  disagree with the sampled logical observable record.

## Installation

Use **64-bit CPython 3.10–3.14**. Pip installs a prebuilt wheel on the
[supported platforms](release.md#supported-toolchains), including Linux, macOS,
and Windows. You do not need Rust or a source checkout to install these wheels.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, use `.venv\Scripts\Activate.ps1` in PowerShell instead. Install
FaultScope from PyPI in the active environment:

```bash
pip install faultscope
```

For this tutorial, also install the optional PyMatching integration and Stim:

```bash
pip install "faultscope[pymatching]" stim
```

The example uses each dependency for a specific task:

| Package | Role |
| --- | --- |
| FaultScope | Samples the noisy circuit and estimates logical failure and noise sensitivities |
| Stim | Constructs the example circuit and a decomposed DEM for the decoding graph |
| PyMatching | Predicts logical flips from detector syndromes |

The `pymatching` extra installs PyMatching, NumPy, and SciPy. The optional native
PyMatching backend is a separate package; see
[Decoding](guides/decoding.md#install-a-native-backend).

Check that the package imports and print the installed version:

```bash
python -I -c "import faultscope; print(faultscope.__version__)"
```

<span id="install-from-source"></span>

### Source builds

For platforms without a compatible wheel, or to edit FaultScope itself, follow
[Development](development.md#set-up-a-development-environment). Source builds
require Python 3.10+ and Rust 1.85+.

## Simulate a surface-code memory

Save the following as `surface_code.py`. The circuit has three rounds and
depolarizing noise with rate 0.005 after Clifford gates.

```python
import pymatching
import stim

from faultscope import FaultScopeSimulator, PyMatchingDecoder, parse_stim_circuit

stim_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.005,
)
imported = parse_stim_circuit(str(stim_circuit))
simulator = FaultScopeSimulator(imported.circuit)

# Inspect a batch of circuit samples.
batch = simulator.run_batch(shots=1024, seed=1)
detector_ids = tuple(detector.id for detector in imported.detectors)
observable_ids = tuple(observable.id for observable in imported.observables)
print(f"Sampled {batch.shots} shots with {len(batch.detectors)} detectors")
print("First-shot syndrome:", [batch.detector_bit(d, 0) for d in detector_ids])

# Use the same detector and observable ordering for the decoder.
stim_dem = stim_circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(stim_dem)
decoder = PyMatchingDecoder(
    matching=matching,
    detector_ids=detector_ids,
    observable_ids=observable_ids,
    edge_count=matching.num_edges,
)

# Sample a new batch and estimate failure after decoding.
result = simulator.estimate(shots=10_000, seed=2, decoder=decoder)
print(f"Logical failure rate: {result.logical_failure_rate:.4%}")
```

Run it in the activated environment:

```bash
python -I surface_code.py
```

The `-I` option makes Python use the installed package even when the script is
inside the source checkout. Use this invocation for the other standalone guide
examples as well.

## Read the results

The first batch contains 1,024 shots, 24 detectors, and one logical observable.
The printed syndrome is the 24 detector bits for the first shot. A zero syndrome
means no detector fired; it does not by itself prove that decoding succeeded.

The second batch estimates the fraction of 10,000 shots that failed after
decoding. This is a Monte Carlo estimate: the rate and hotspot rankings can vary,
and more shots reduce sampling uncertainty. A seed makes a run reproducible under
the same implementation and settings; it does not promise identical samples
across releases.

Stim's decomposed DEM supplies the **decoding graph**. FaultScope still samples
the original circuit and its physical noise channels. Constructing the decoder
from a DEM does not switch the experiment to DEM sampling.

Batch fields store packed integer masks, with one bit per shot. Use helpers such
as `batch.detector_bit(detector_id, shot)` for inspection. Reuse the simulator
for additional batches so that compilation is not repeated.

## Inspect noise sensitivities

Append this line to the same script:

```python
print(result.hotspot_table(top_k=5))
```

The table ranks circuit noise locations by the absolute value of their estimated
sensitivity. A positive sensitivity means increasing that location's noise rate
increases the expected loss; a negative value means it decreases the loss. The
decoder is held fixed. These estimates describe a local change in rate, not the
number of failures caused by a location.

This example reports **physical noise-rate sensitivities**. DEM sampling instead
reports sensitivities to independent edge probabilities. Read
[Noise Sensitivity](guides/noise_sensitivity.md) before comparing or grouping them.

## Next steps

- [Circuit Sampling](guides/sampling.md): build circuits and inspect packed results.
- [Decoding](guides/decoding.md): choose a decoder and check its detector layout.
- [Collection and Thresholds](guides/collection.md): collect statistics across many tasks.
- [Detector Error Models](guides/dem.md): understand conversion and sampling limits.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError: faultscope._native` | Install with the same environment's Python, then run the script with `python -I`. Source contributors should use `maturin develop`; see [Development](development.md). |
| Pip tries to build from source | Upgrade pip with `pip install --upgrade pip` and check the [supported platforms](release.md#supported-toolchains). Source builds require Rust 1.85+. |
| PyMatching cannot be imported | Install the `pymatching` extra in the active environment. |
| A different circuit cannot produce a DEM | Check the [DEM conversion limits](guides/dem.md); the Forward sampler accepts some circuits that the DEM generator rejects. |
| Default loss reports missing observables | Declare the circuit's logical observables or supply a custom loss as described in [Circuit Sampling](guides/sampling.md). |
