# Sample a Noisy Circuit

Use forward sampling when you need measurement records, detector syndromes, or
logical observables from the original noisy circuit. The circuit can also be
reused for decoding and noise sensitivity estimates.

Install FaultScope using the [installation instructions](../getting_started.md#install-from-source).
This page needs no optional packages. The Python blocks below continue the same
example; save them in order as `sampling_example.py` and run:

```bash
python -I sampling_example.py
```

## Build and Sample a Circuit

The simulator starts the qubit in `|0>`. This circuit applies an X error with
probability 0.1, measures in the Z basis, and records that measurement as both a
detector and a logical observable. It is a small sampling example, not an
error-correcting code.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    FaultScopeSimulator,
    NoiseLocation,
    Operation,
)

circuit = Circuit(
    1,
    (
        Operation.noise(NoiseLocation("x0", BernoulliPauliNoise("X"), 0.1, (0,))),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
        Operation.observable_include(0, ("m0",)),
    ),
)
simulator = FaultScopeSimulator(circuit)
batch = simulator.run_batch(shots=4096, seed=7)

print("shots:", batch.shots)
print("measured-one fraction:", batch.measurements["m0"].bit_count() / batch.shots)
print("first-shot detector:", batch.detector_bit(0, 0))
```

The measured-one fraction should be close to 0.1. It varies between random
samples. A detector records the declared parity over measurement records. Here
the ideal noiseless parity is zero, so a detector value of one indicates a
change; an arbitrary nonzero ideal parity is not automatically subtracted.
A logical observable identifies the logical quantity that a decoder will try
to predict.

## Read Packed Results

Each integer mask stores one bit per shot. Bit zero is the first shot. Use the
bit helpers when inspecting individual shots, and integer operations when
working with whole batches.

```python
print("first-shot measurement:", batch.measurement_bit("m0", 0))
print("first-shot observable:", batch.observable_bit(0, 0))
print("sampled X events:", batch.noise_event_masks["x0"].bit_count())
```

| Field | Contents |
| --- | --- |
| `measurements` | Measurement-key to packed outcome mask |
| `detectors` | Detector-id to packed syndrome mask |
| `observables` | Observable-id to packed logical flip mask |
| `noise_event_masks` | Noise-location-id to packed event mask |
| `x_frame`, `z_frame` | Packed Pauli-frame differences for each qubit |

Mask bits beyond `batch.shots` are not samples. Custom loss and correction
callbacks must also return masks with the batch's shot width.

## Reuse the Simulator

Reuse the compiled simulator for additional batches. `sample(...)` is another
entry point for a full batch; `sample_measurements(...)` returns only measurement
masks when you do not need events or detector records.

```python
next_batch = simulator.sample(shots=1024, seed=8)
measurement_masks = simulator.sample_measurements(shots=1024, seed=9)
print("next batch:", next_batch.shots)
print("measurement keys:", tuple(measurement_masks))
```

Fix the seed and call settings when reproducing a run. Different APIs and
batching schemes do not promise identical shots for the same numeric seed.
Use `seed=` for native sampling; `rng=` is a compatibility input on the facades.

## Import Stim Text

FaultScope's parser accepts a supported subset of Stim text, including `REPEAT`
blocks and `SHIFT_COORDS`. It embeds detector and observable declarations in the
returned circuit. Parsing text itself does not require the `stim` Python package.
The [surface-code quickstart](../getting_started.md#simulate-a-surface-code-memory)
uses Stim to generate a complete QEC experiment.

```python
from faultscope import parse_stim_circuit

imported = parse_stim_circuit("""
R 0
REPEAT 2 {
    X_ERROR(0.1) 0
    M 0
    DETECTOR(0) rec[-1]
    SHIFT_COORDS(1)
    R 0
}
OBSERVABLE_INCLUDE(0) rec[-1]
""")
imported_batch = FaultScopeSimulator(imported.circuit).run_batch(shots=64, seed=10)
print("imported detectors:", tuple(imported_batch.detectors))
```

The subset includes common Clifford and Pauli gates, resets, measurements,
`MPP`, Pauli/depolarizing noise, `PAULI_CHANNEL_1/2`, and record-based detector
and observable declarations. It does not provide general Stim feedback or
correlated-error instruction semantics. Use `load_stim_file(path)` to read a
text file. Unsupported syntax raises `StimImportError`.

## Choose the Next Step

- [Decode a batch](decoding.md) to measure failure after error correction.
- [Collect logical error rates](collection.md) across many circuits or noise rates.
- [Estimate noise sensitivities](noise_sensitivity.md) to identify influential locations.
- [Generate and sample a DEM](dem.md) when you only need detector and logical flips.

Forward sampling supports the implemented Clifford operations, Pauli
measurements/resets, and stochastic Pauli noise models. General non-Clifford
operations, non-Pauli channels, and per-shot adaptive branching are outside this
runtime's scope. See the [API reference](../api_reference.md#forward-runtime)
for method signatures and native batch handles. For a small built-in QEC
experiment, see the [repetition-code builder](../api_reference.md#repetition-code-experiments).
