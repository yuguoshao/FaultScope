# Estimate Noise Sensitivities

A logical error rate tells you how often an experiment fails. A sensitivity
estimates how that rate changes when one noise parameter changes, while the
circuit and decoder are held fixed. Hotspots rank the magnitude of these
sensitivities to help identify influential locations.

This page needs only the base FaultScope installation. The Python blocks
continue one example. Save them as `sensitivity_example.py` and run:

```bash
python -I sensitivity_example.py
```

## Estimate a Physical Noise-Rate Sensitivity

This circuit measures a qubit after X noise and treats a measured one as a
logical failure. With no decoder, the expected loss is the noise probability
and its derivative is one. That makes the example easy to check.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    FaultScopeSimulator,
    NoiseLocation,
    Operation,
)

noise = NoiseLocation(
    "data_x0",
    BernoulliPauliNoise("X"),
    0.1,
    (0,),
    tags={"round": 0, "gate": "idle", "operation": "noise"},
)
circuit = Circuit(
    1,
    (
        Operation.noise(noise),
        Operation.measure(0, key="m0"),
        Operation.observable_include(0, ("m0",)),
    ),
)
simulator = FaultScopeSimulator(circuit)
result = simulator.estimate(shots=8192, seed=3, baseline=0.0)

print("logical error rate:", result.logical_failure_rate)
print("rate sensitivity:", result.sensitivities["data_x0"])
print(result.hotspot_table(top_k=5))
```

The logical rate should be near 0.1 and the sensitivity near 1. A sensitivity
of 1 means that a small rate increase of 0.001 would increase the expected loss
by about 0.001 locally. It does not predict the result of a large parameter
change. Monte Carlo estimates fluctuate, so compare values using enough shots
for the experiment.

## Read Rankings and Groups

The sign carries the direction of the effect: a positive sensitivity means
increasing the noise rate increases loss; a negative value means it decreases
loss under the fixed model and decoder. A hotspot is the absolute sensitivity,
so a high rank can correspond to either sign.

```python
print("signed sensitivities:", result.sensitivities)
print("absolute hotspots:", result.hotspots)
print("by qubit:", result.by_qubit)
print("by gate:", result.by_gate)
```

`by_qubit`, `by_round`, `by_gate`, and `by_operation` sum absolute hotspots over
locations with the corresponding qubits or tags. They are summaries of
influence, not signed derivatives for a shared parameter. Keep location ids
unique and attach consistent tags when constructing larger circuits.

## Use a Decoder or a Custom Loss

For a QEC circuit with logical observables, pass a decoder to
`simulator.estimate(..., decoder=decoder)`. The default loss is one when any
logical residual remains after applying the decoder's predicted flips. The
[decoding guide](decoding.md) shows a complete example.

For another binary objective, supply a callback returning a packed loss mask.
This continuation uses the measured-one mask and therefore estimates the same
objective as the example above:

```python
custom = simulator.estimate(
    shots=8192,
    seed=4,
    baseline=0.0,
    loss_mask_fn=lambda batch: batch.measurements["m0"],
)
print("custom mean loss:", custom.mean_loss)
```

With a custom loss, `logical_failure_rate` is an alias for that mean loss; it
need not represent a decoded logical failure. Forward loss callbacks may
accept `(batch)` or `(batch, corrections)`. DEM loss callbacks always accept
`(batch, corrections)`. Do not supply both `decoder` and `correction_mask_fn`.

A native decoder keeps the default decoding and loss path in Rust. Supplying
Python loss or correction callbacks uses the compatibility path.

## Understand the Estimator

The estimator correlates loss with the sampled noise events using a
score-function derivative. It does not rerun the experiment once per noise
location. For channel noise, the derivative changes the total location rate
while keeping the conditional Pauli-component weights fixed.

The example sets `baseline=0.0`, a fixed baseline. The default baseline uses
the same batch's mean loss; it can reduce variance but introduces a finite-batch
bias. See [the estimator derivation](../theory.md) for assumptions, baseline
choices, and endpoint behavior. These estimates hold the decoder fixed; they
do not differentiate through rebuilding or retuning the decoder as noise changes.

## Keep Forward and DEM Meanings Separate

| Estimator | Parameter being varied |
| --- | --- |
| Forward | The physical `NoiseLocation.rate` in the original circuit |
| DEM edge | The probability of an independent DEM edge |
| DEM location summary | A weighted summary of its edge derivatives |

A DEM location summary does not automatically apply the circuit-to-DEM
parameter transformation's chain rule. In particular, exact depolarizing
factorization can introduce several latent edge probabilities for one physical
rate. Choose the forward estimator when the question concerns that physical
rate. See [DEM sampling](dem.md) for conversion and approximation limits.

Event recording is required for hotspot estimation. Native batches must belong
to the same compiled sampler that estimates them; missing event masks or a
foreign batch are rejected. For larger jobs, use
[`Collector.collect_hotspots`](collection.md#other-collection-paths) to combine
shot-weighted sensitivities across fixed-size batches. This path does not
support CSV partial resume.
