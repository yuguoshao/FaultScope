# Decoder Development

Use this page to implement a decoder. To use an existing decoder, start with
[Decoding](guides/decoding.md). The [Native Decoder ABI V4](native_decoder_abi.md)
defines the exact plugin contract; this page explains how to build against it.

Choose the route that matches your work:

| Goal | Route |
| --- | --- |
| Try an algorithm with Python and packed masks | [Python prototype](#python-prototype-decoders) |
| Ship a native solver in a separate package | [Standalone native plugin](#standalone-native-plugin) |
| Change FaultScope's built-in decoder support | [Core integration](#core-integration) |

## Mental Model

Construct the solver once from a circuit, DEM, or compiled decoding problem.
For each batch, give it detector syndromes and return predicted logical flips:

```text
Construction: circuit / DEM -> decoding problem -> decoder
Each batch:   detector syndromes -> decoder -> observable corrections
```

FaultScope compares each correction with the sampled logical observable. A shot
fails if any observable disagrees. The decoder must not read sampled noise events
or observable truth to choose its correction. Those records belong to sampling
and attribution, not decoding.

Build graph structure, matrices, weights, coordinates, and lookup tables during
construction. Reuse that state instead of reading the circuit or DEM per batch.

## Python Prototype Decoders

A Python decoder implements `decode_batch_masks(batch)`. The method receives
`batch.detectors`, a mapping from detector ID to an integer mask. Bit 0 is shot 0.
It returns a mapping from observable ID to an integer correction mask, using the
same shot order.

This complete example needs only the base FaultScope package. Detector 0 and
observable 0 are the same measurement, so copying the detector mask is an exact
decoder for this circuit. The class is specific to this example, not a general
error-correcting algorithm.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    FaultScopeSimulator,
    NoiseLocation,
    Operation,
)


class CopyDetectorDecoder:
    detector_ids = (0,)
    observable_ids = (0,)

    def decode_batch_masks(self, batch):
        return {0: batch.detectors[0]}


circuit = Circuit(
    1,
    (
        Operation.noise(NoiseLocation("x_error", BernoulliPauliNoise("X"), 0.1, (0,))),
        Operation.measure(0, key="m"),
        Operation.detector(("m",), detector_id=0),
        Operation.observable_include(0, ("m",)),
    ),
)

simulator = FaultScopeSimulator(circuit)
result = simulator.estimate(shots=1024, seed=7, decoder=CopyDetectorDecoder())
print(f"Logical failure rate: {result.logical_failure_rate:.2%}")
assert result.logical_failure_rate == 0.0
```

Save this as `decoder_example.py` and run `python -I decoder_example.py` in the
environment where FaultScope is installed. `-I` avoids importing an unbuilt
package from the source checkout.

Python decoders work with simulator `estimate(...)`; native collection does not
accept them. For real algorithms, add constructors such as `from_dem(...)` or
`from_circuit(...)` and compile the solver there. Prefer whole-mask operations
to Python loops over shots. A forward decoder that needs selected measurements
can fetch them together with `batch.measurement_masks(keys)`; DEM batches have
no measurement records. See the [callback API](api_reference.md#callback-contracts).

## Getting Circuit And DEM Information

Choose a construction view for the solver:

| DEM method | Provides | Typical use |
| --- | --- | --- |
| `compile_indexed()` | Stable IDs, indexed edge supports, probabilities, and original edge indices | General decoder construction |
| `compile_graphlike_problem()` | One- or two-detector edges with logical effects and weights | MWPM and fusion-blossom |
| `compile_binary_linear_problem()` | Sparse binary matrices `H` and `F`, probabilities, and log-likelihood ratios | BP/OSD and other binary solvers |

These views preserve declared ID order, then append IDs first encountered in raw
edges. Edge supports use GF(2) parity: an ID repeated twice cancels. Detector
coordinates follow detector ID order. Use the compiled views rather than
rebuilding these conventions in the backend.

Graphlike compilation rejects hyperedges and pure logical edges with no detector.
When a source supplies a valid decomposition, use a `GeneratedDetectorErrorModel`
with `GraphlikeDecompositionHints` and call its `compile_graphlike_problem()`.
Every component must have one or two detectors, and the XOR of all components
must reproduce the parent's detector and observable support. Components retain
the parent's probability and `dem_edge_index`. They only describe the decoding
graph: sampling still draws the canonical parent edge once. Current matching
backends use the components as an uncorrelated approximation.

FaultScope's native circuit-to-DEM generator does not create these hints and
requires each measurement encountered during propagation to be deterministic in
the ideal reference. Do not assume it can construct a matching graph for every
circuit that forward sampling supports. The [surface-code example](getting_started.md)
uses Stim to prepare its decoding graph. See [DEMs](guides/dem.md) for conversion
options and [DEM API](api_reference.md#detector-error-models) for signatures.

<span id="native-decoder-backends"></span>

## Standalone Native Plugin

A standalone backend lives in its own Python extension package. Python builds
and owns a decoder handle; FaultScope calls the plugin through the versioned C
ABI. This route does not require adding private PyO3 classes to `faultscope._native`.

<span id="development-checklist"></span>

Follow these steps:

1. Choose the construction view above and compile immutable solver data.
2. Register a `faultscope.native_decoders` entry point that returns the
   [plugin manifest](native_decoder_abi.md#manifest-and-official-packages).
   Expose constructors such as `from_dem(...)`, `from_graphlike_problem(...)`,
   and, where supported, `from_circuit(...)`.
3. Return a handle implementing `__faultscope_native_decoder_capsule__()` with
   the current ABI descriptor. The capsule owns the factory; each factory
   creates private workers with exclusive mutable solver state.
4. Implement the ABI's single tagged `decode_batch` callback. Read the borrowed
   detector batch and write corrections into FaultScope's output buffers.
   Follow the [format](native_decoder_abi.md#format-tags-and-fixed-correction-mapping),
   [callback](native_decoder_abi.md#callback-contract), and
   [lifetime](native_decoder_abi.md#ownership-and-lifetime) rules.
5. Provide stable collection identity as described below, then validate the
   native path against the Python prototype or a trusted solver.

Factories share immutable data across threads. Workers own solver state and
scratch buffers; FaultScope calls a given worker serially and caches workers by
task and thread during collection. A backend does not need a second worker pool
or a mutex around a shared mutable solver.

The graphlike construction capsule uses ABI V1; the decoder execution capsule
uses ABI V4. They are separate interfaces. A compiled problem chooses solver
construction data, while the factory's `batch_formats` determines runtime layout.
Use the [ABI reference](native_decoder_abi.md) for versions, sizes, and validation.

### Stable collection identity

Native decoder objects supplied directly to `CollectionTask.decoder` must
implement `strong_id_payload()`. Return a JSON-serializable mapping describing
the effective decoder state. Include:

- backend, ABI, and implementation versions;
- ordered detector and observable IDs;
- normalized options and every parameter that can change corrections;
- the effective graph or matrix;
- ordered child payloads for a composite decoder.

Do not derive identity from `repr(...)` or object addresses. Equivalent mapping
insertion orders must have equivalent meaning. Increment the implementation
version when behavior changes without a visible configuration change. Collection
rejects a missing identity method before reading resume files or starting workers;
there is no caller-supplied identity override. See [collection identity](guides/collection.md)
for how this protects resumed results.

<span id="python-input-rust-construction"></span>

## Core Integration

Use this route when changing FaultScope itself. Built-in Rust decoders implement
`NativeDecoderFactory: Send + Sync` and `NativeDecoderWorker: Send`.
`NativeBatchDecoder` is the Python factory-handle class, not a Rust trait.
The PyO3 layer holds the factory and converts construction input once; the worker
owns mutable state and decodes through `&mut self`.

Start with these implementation points:

- [Core decoder traits and batch types](https://github.com/yuguoshao/FaultScope/blob/main/crates/faultscope-core/src/decoder.rs).
- [PyO3 decoder binding and capsule validation](https://github.com/yuguoshao/FaultScope/blob/main/crates/faultscope-python/src/decoder_api.rs).
- [Native decoder ABI tests](https://github.com/yuguoshao/FaultScope/blob/main/tests/test_native_decoder_abi_v4.py).

When adding an official backend, also update the
[backend catalog](https://github.com/yuguoshao/FaultScope/blob/main/faultscope/backends/registry.py)
and its public proxy in `faultscope.decoders`. Keep installation and solver setup
out of imports and batch execution. Repository build and test instructions are
in [Development](development.md).

<span id="example-native-backend"></span>
<span id="post-install-native-backends"></span>
<span id="official-backend-packages"></span>
<span id="fusion-blossom-adapter-status"></span>

## Backend Examples

The two in-repository packages demonstrate standalone integration:

| Backend | Implementation and limits |
| --- | --- |
| [PyMatching](https://github.com/yuguoshao/FaultScope/blob/main/backends/faultscope-pymatching/README.md) | Native C++ sparse-blossom solver, graphlike construction, packed batches |
| [fusion-blossom](https://github.com/yuguoshao/FaultScope/blob/main/backends/faultscope-fusion-blossom/README.md) | Serial beta MWPM adapter, integer weight conversion, profiling and current limitations |

Their READMEs describe solver-specific behavior and build requirements.
[Decoding](guides/decoding.md#install-a-native-backend) covers installation and
backend status. The built-in no-correction and detector-copy decoders are useful
integration checks, not substitutes for a general QEC decoder.

## Validation And Performance Rules

Before using a new backend, check:

- zero, single, and multiple detector events against a trusted implementation;
- detector order, complete observable order, batch size, and padding;
- empty batches, multiple observables, and every declared batch format;
- worker creation, failure cleanup, and concurrent workers;
- stable identity and collection resume behavior.

Use FaultScope's checked dispatch. The incoming detector IDs must match the
factory's declared order, and corrections must cover the sampler's complete
canonical observable sequence. Missing or reordered native observables are errors.
For native `FaultScopeSimulator.estimate(...)` in 0.2.12, supply the observable
declarations through `FaultScopeSimulator(circuit, observables=...)`; embedded
`observable_include` operations alone do not populate that validation layout.

With a native handle and no Python `loss_mask_fn` or `correction_mask_fn`, the
runtime keeps decoding, residual loss, and hotspot aggregation in native memory.
A Python callback uses the compatibility path and may call
`decode_batch_masks(batch)`. Merely implementing that Python method does not make
a decoder native.

DEM sampling uses the decoder's first preferred batch format. Enabling
`aggregate_hotspots` records an attribution sidecar in the same sampling pass;
it does not force detector-major input. The decoder receives only syndromes.
The [ABI format table](native_decoder_abi.md#format-tags-and-fixed-correction-mapping)
defines the correction layout for each input format.

`materialize_dem=False` creates a sampler without full DEM metadata. It can
sample, but APIs that construct hotspot results or decoding problems from that
metadata require a materialized DEM. Keep sampling, decoding, and attribution
timings separate when measuring performance; solver-specific diagnostics belong
in the backend package.
