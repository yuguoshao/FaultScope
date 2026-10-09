# FaultScope

FaultScope is a stabilizer circuit simulator for quantum error correction, with a
Rust core and a Python API. It simulates noisy Clifford circuits, samples detector
syndromes, and estimates logical error rates with external decoders. It also
generates detector error models and estimates how individual noise rates affect
logical failure.

[Documentation](https://yuguoshao.github.io/FaultScope/) |
[User Guide](docs/user_guide.md) | [API Reference](docs/api_reference.md)

- Sample measurements, detector syndromes, and logical observables from noisy
  Clifford circuits.
- Generate detector error models (DEMs) and sample their independent error
  mechanisms.
- Integrate decoders and collect logical error rates across code distances and
  noise rates.
- Estimate noise sensitivities and rank fault locations by their effect on a
  chosen loss.

## Installation

Use **64-bit CPython 3.10–3.14**. Prebuilt wheels are available for
[supported Linux, macOS, and Windows platforms](docs/release.md#supported-toolchains),
so installing on these platforms does not require Rust.

Create and activate a virtual environment, then install from PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install faultscope
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead. For the quickstart below, add the example dependencies:

```bash
pip install "faultscope[pymatching]" stim
```

The `pymatching` extra supplies PyMatching, NumPy, and SciPy; Stim supplies the
example circuit. See the [installation guide](docs/getting_started.md#installation)
for more details, or [Development](docs/development.md) for source builds and
contributions.

## Quickstart: Simulate a Surface-Code Memory

This example runs a distance-3 rotated surface-code Z-memory experiment for three
rounds, with depolarizing noise after Clifford gates. Stim constructs the circuit
and a decomposed DEM for the decoding graph. FaultScope samples the original
circuit, and PyMatching predicts logical flips from the detector syndromes.

Save the following code as `surface_code.py` and run `python -I surface_code.py`
in the activated environment. The `-I` option imports the installed package even
when the script is inside the source checkout.

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

# Construct a matching decoder for this circuit's detector layout.
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

A logical failure occurs when the decoder's predicted logical flips disagree with
the sampled logical observable record. The reported rate is a Monte Carlo
estimate; increase the shot count to reduce sampling uncertainty.

Batch fields such as `batch.detectors` hold packed integer masks, with one bit per
shot. Helpers such as `detector_bit(detector_id, shot)` expose individual values.
The same compiled simulator can be reused for additional batches.

## Simulation Workflows

| Task | Public entry point | Result |
| --- | --- | --- |
| Sample a noisy circuit | `FaultScopeSimulator.run_batch(...)` | Measurements, detectors, observables, and noise-event masks |
| Generate a DEM | `generate_native_dem(circuit)` | Detector and logical effects of error mechanisms |
| Sample a DEM | `DemFaultScopeSimulator(circuit)` or `DemHotspotEstimator(dem)` | Detector syndromes and logical flips from independent error mechanisms |
| Estimate decoded failure | `simulator.estimate(..., decoder=decoder)` | Logical failure rate and noise sensitivities |
| Run repeated QEC experiments | `Collector` with `CollectionTask(circuit=...)` | Logical error-rate statistics with parallel collection and CSV resume |

Circuit-based collection uses forward circuit sampling. A DEM used to construct
its decoder does not change the sampling path. Collection also supports stopping
conditions and analysis of code-distance and noise-rate sweeps; see
[logical error-rate collection](docs/guides/collection.md).

DEM sampling operates on independent error mechanisms. Some circuit noise
channels convert exactly; others require explicitly enabled approximation.
Use forward sampling when the original channel's mutually exclusive outcomes
must be preserved. See [detector error models](docs/guides/dem.md).

The simulator supports Clifford gates, Pauli measurements and resets, and the
supported stochastic Pauli noise models. Non-Clifford gates, general non-Pauli
channels, and general per-shot adaptive branching are outside its scope.

## Noise Sensitivity Analysis

The quickstart's estimate also contains sensitivities for the physical noise
locations in the circuit. Append this line to `surface_code.py` to print the five
highest-ranked locations from the same estimate:

```python
print(result.hotspot_table(top_k=5))
```

A sensitivity estimates how the chosen loss changes when a location's noise rate
increases, with the decoder held fixed. A positive value indicates increasing
loss; a negative value indicates decreasing loss. Hotspot rankings use the
absolute sensitivity, so they highlight large effects in either direction.

Location tags allow results to be grouped by qubit, round, gate, or operation.
Forward sensitivities refer to physical noise rates. DEM sensitivities refer to
DEM edge probabilities; their location summaries are not generally derivatives
with respect to the original circuit noise rates. See the
[sensitivity guide](docs/guides/noise_sensitivity.md) for interpretation and
[theory](docs/theory.md) for the estimator's statistical assumptions.

## Performance

FaultScope processes many shots together using packed bit arrays in Rust. Reuse
a compiled simulator when sampling the same circuit repeatedly.

The quickstart uses the Python PyMatching integration. Optional native PyMatching
and fusion-blossom backends keep batch decoding in native code; see
[native decoder integration](docs/guides/decoding.md#native-backends).

For reproducible measurements, use a release build and report the circuit, shot
count, decoder, and hardware alongside the timings. Start with the
[sampling benchmark](benchmarks/sampling_throughput.py) or the
[surface-code decoder benchmark](benchmarks/surface_code_decoder_performance.py).
The [benchmark guide](docs/development.md#measure-performance) documents build instructions,
commands, and comparison settings.

## Documentation and Development

- [Getting Started](docs/getting_started.md): installation and a complete surface-code example.
- [User Guide](docs/user_guide.md): sampling, decoding, collection, and noise sensitivities.
- [API Reference](docs/api_reference.md): Python and Rust interfaces and result fields.
- [Theory](docs/theory.md): sampling models, sensitivity estimators, and DEM semantics.
- [Decoder Development](docs/decoder_development.md): custom Python and native decoders.
- [Release and Compatibility](docs/release.md): supported platforms and version policy.
- [Development](docs/development.md): build, test, lint, documentation, and benchmarks.

FaultScope is pre-1.0; Python and Rust APIs may change between releases.
Contributions should pass the relevant development checks above.

## License

FaultScope is distributed under the [GNU Affero General Public License v3.0](LICENSE).
