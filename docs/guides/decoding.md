# Decoding

A decoder predicts logical flips from detector syndromes. FaultScope compares
those predictions with the sampled observables to count logical failures.
Choose a Python decoder for a first experiment, or a native decoder for collection
and runs that should keep batch data out of Python.

| Choice | Use it for | Requirements |
| --- | --- | --- |
| `PyMatchingDecoder` | Simulator experiments with a matching decoder | PyMatching, NumPy, and SciPy |
| Your own `decode_batch_masks(batch)` class | Prototyping or a custom Python algorithm | Base FaultScope package |
| `NativePyMatchingDecoder` or `NativeFusionBlossomDecoder` | Native decoding and collection | A separately installed backend |

Python decoders work with simulator `estimate(...)`. The main collection API
accepts native decoders or no decoder. See [Collection](collection.md).

## A Complete PyMatching Example

Install FaultScope with the `pymatching` extra as described in
[Getting started](../getting_started.md). This example uses the Python PyMatching
package and needs no optional native backend.

The circuit has one possible X error. Its detector and logical observable both
read the resulting measurement, so the decoder can identify every error. This
small circuit makes the interface visible; for a QEC experiment, use the
[surface-code example](../getting_started.md).

Save this as `decoding_example.py`:

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    FaultScopeSimulator,
    NoiseLocation,
    Operation,
    PyMatchingDecoder,
)
from faultscope.runtime import generate_native_dem

circuit = Circuit(
    1,
    (
        Operation.noise(NoiseLocation("x_error", BernoulliPauliNoise("X"), 0.1, (0,))),
        Operation.measure(0, key="m"),
        Operation.detector(("m",), detector_id=0),
        Operation.observable_include(0, ("m",)),
    ),
)

dem = generate_native_dem(circuit)
decoder = PyMatchingDecoder.from_dem(dem)
simulator = FaultScopeSimulator(circuit, observables=dem.observables)

uncorrected = simulator.estimate(shots=4096, seed=7)
corrected = simulator.estimate(shots=4096, seed=7, decoder=decoder)
print(dem.to_dem_text())
print(f"Without decoding: {uncorrected.logical_failure_rate:.2%}")
print(f"With decoding:    {corrected.logical_failure_rate:.2%}")
assert corrected.logical_failure_rate == 0.0
```

Run `python -I decoding_example.py` in the activated environment. `-I` imports
the installed package even if the script is in the source checkout. The
uncorrected rate is about 10%; the corrected rate is zero for this particular
circuit. The DEM builds the decoder, while `FaultScopeSimulator` still samples
the original circuit.

## Which DEMs Can Matching Decode?

The example's DEM is graphlike: its edge touches one detector. Direct matching
construction accepts edges with one or two detectors after GF(2) cancellation.
It rejects hyperedges and logical-only edges with no detector. PyMatching also
rejects parallel edges with the same detector endpoints but conflicting logical
effects.

Some sources provide a decomposition into graphlike components. A
`GeneratedDetectorErrorModel` can carry these in `GraphlikeDecompositionHints`;
use `artifact.compile_graphlike_problem()` to build a native matching decoder
from them. The hints only affect decoder construction. They do not split the
canonical sampling event into independent events, and current matching backends
do not implement correlated matching from the shared parent index.

FaultScope's native generator currently does not synthesize hints and requires
deterministic ideal measurements. A circuit supported by forward sampling may
therefore need another route to its decoding graph. The surface-code quickstart
uses Stim's `detector_error_model(decompose_errors=True)` and constructs the
Python matching object from that model. See [DEMs](dem.md) for conversion limits
and the [DEM API](../api_reference.md#detector-error-models) for compiled views.

<span id="native-backends"></span>

## Install a Native Backend

Start from the FaultScope repository root with your virtual environment active
and the base package installed. You can inspect discovery without installing:

```bash
python -I -m faultscope.backends status
python -I -m faultscope.backends install pymatching --dry-run
```

`--dry-run` only prints a plan. It does not install or verify package availability.
The generic plan may end with a package-index install. To build the backend from
this source checkout, use the local package instead.

For native PyMatching, install the build tool, then build the backend:

```bash
python -m pip install "maturin>=1.7,<2"
python -m pip install -e backends/faultscope-pymatching --no-build-isolation
python -I -m faultscope.backends status
```

This needs Rust, Git, and a C++20 compiler. The build fetches pinned PyMatching
source unless `NPSIM_PYMATCHING_SOURCE_DIR` points to an existing checkout.
See the [backend README](https://github.com/yuguoshao/FaultScope/blob/main/backends/faultscope-pymatching/README.md)
for details. Installing the Python `pymatching` wheel alone does not install
`NativePyMatchingDecoder`.

Alternatively, build the fusion-blossom backend from the same repository root
and active environment:

```bash
python -m pip install -e backends/faultscope-fusion-blossom
python -I -m faultscope.backends status
```

This is a serial beta MWPM adapter. Its
[README](https://github.com/yuguoshao/FaultScope/blob/main/backends/faultscope-fusion-blossom/README.md)
describes build requirements, integer weights, and unsupported features.

In FaultScope 0.2.11, `bpdecoder` and `mwpm` await ABI V4 migration, and `bposd`
is reserved. These catalog entries are not installable. `status` distinguishes
an absent package from one that is installed but cannot load. FaultScope never
installs a backend during import or sampling.

## Use the Native Path

After installing native PyMatching, change the decoder construction in the
complete example above to:

```python
from faultscope.decoders import NativePyMatchingDecoder

decoder = NativePyMatchingDecoder.from_dem(dem)
```

Keep the same `simulator.estimate(..., decoder=decoder)` call. A native handle
with no Python loss or correction callback keeps decoding and default residual
loss in native memory. Adding `loss_mask_fn` or `correction_mask_fn` uses the
Python compatibility path. A custom Python class is not a native decoder merely
because it defines `decode_batch_masks`.

Native decoder IDs must agree with the sampled layout. Construct the decoder
from the same declarations as the sampler; do not drop or reorder observables.
For native forward estimates in 0.2.11, pass `observables` explicitly, as the
complete example does. Embedded `observable_include` operations alone can fail
the native observable-ID check.
When using a hinted artifact, call
`NativePyMatchingDecoder.from_graphlike_problem(artifact.compile_graphlike_problem())`
instead of compiling its raw DEM without the hints.

Hotspot attribution does not change the decoder's input format. FaultScope can
record events separately while the native decoder receives its preferred batch
layout. See [Noise sensitivity](noise_sensitivity.md) for interpreting the result,
or [Decoder development](../decoder_development.md) to implement a backend.
