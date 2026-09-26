# FaultScope Documentation

FaultScope is a stabilizer circuit simulator for quantum error correction, with a
Rust core and a Python API. It simulates noisy Clifford circuits, samples detector
syndromes, and estimates logical error rates with external decoders. It also
generates detector error models and estimates how individual noise rates affect
logical failure.

## Start Here

<span id="quick-build"></span>

Follow [Getting Started](getting_started.md) to install FaultScope and run a
complete surface-code memory experiment. You will use Stim to construct a circuit,
FaultScope to sample it, and PyMatching to decode the detector syndromes.

The basic workflow is:

1. Build a circuit or import one from Stim text.
2. Compile it once and sample batches of shots.
3. Decode the detector syndromes and count logical failures.
4. Inspect noise sensitivities to see which rates most affect the chosen loss.

## Choose a Guide

| I want to… | Read |
| --- | --- |
| Inspect measurements, detectors, or logical observables | [Circuit Sampling](guides/sampling.md) |
| Connect PyMatching or choose a native decoder | [Decoding](guides/decoding.md) |
| Generate or sample a detector error model (DEM) | [Detector Error Models](guides/dem.md) |
| Run parameter sweeps, resume collection, or fit a threshold | [Collection and Thresholds](guides/collection.md) |
| Rank noise locations and interpret sensitivities | [Noise Sensitivity](guides/noise_sensitivity.md) |

Use [the guide overview](user_guide.md) for terminology and help choosing between
circuit and DEM sampling. Forward circuit sampling preserves the physical noise
channels; DEM sampling treats error mechanisms as independent events. Their
sensitivity results refer to different parameters.

FaultScope supports Clifford circuits with supported stochastic Pauli noise.
It does not provide general per-shot adaptive branching. See
[supported inputs](guides/sampling.md) for details.

## Reference and Development

<span id="documentation-development"></span>

- [API Reference](api_reference.md): public interfaces, arguments, and result fields.
- [Theory](theory.md): sampling assumptions and sensitivity estimators.
- [Architecture](implementation_overview.md): compilation, execution, and aggregation.
- [Decoder Development](decoder_development.md): custom Python and native decoders.
- [Native Decoder ABI](native_decoder_abi.md): the exact plugin contract.
- [Compatibility](release.md): supported toolchains and upgrade notes.
- [Development and Benchmarks](development.md): build, test, measure, and preview the docs.
