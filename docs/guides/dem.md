# Generate and Sample a Detector Error Model

A detector error model (DEM) describes how error mechanisms flip detectors and
logical observables. Use it when you need repeated syndrome sampling or a
static problem for a decoder. DEM sampling draws independent error mechanisms;
it does not replay the circuit's measurement trajectory.

This page needs only the base FaultScope installation. The Python blocks
continue one example. Save them as `dem_example.py` and run:

```bash
python -I dem_example.py
```

## Generate a DEM

Here an X error flips a Z measurement, one detector, and one logical observable.
The detector and observable declarations are embedded in the circuit.

```python
from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    DemFaultScopeSimulator,
    NoiseLocation,
    Operation,
    generate_native_dem,
)

circuit = Circuit(
    1,
    (
        Operation.noise(NoiseLocation("x0", BernoulliPauliNoise("X"), 0.125, (0,))),
        Operation.measure(0, key="m0"),
        Operation.detector(("m0",), detector_id=0),
        Operation.observable_include(0, ("m0",)),
    ),
)
dem = generate_native_dem(circuit)
print(dem.to_dem_text())
print("effects at x0:", dem.edges_by_location()["x0"])
```

Each DEM edge stores an event probability, the detector ids it flips, and the
logical observable ids it flips. Location ids and tags retain attribution to
the circuit's noise locations. Multiple effects may come from one location.

You can also pass `detectors=` and `observables=` explicitly. `None` uses the
embedded declarations; an explicit sequence replaces them, and an empty
sequence clears them.

## Sample Detector and Logical Flips

`DemFaultScopeSimulator(circuit)` compiles and samples a DEM through an interface
similar to the forward simulator.

```python
dem_simulator = DemFaultScopeSimulator(circuit)
batch = dem_simulator.run_batch(shots=4096, seed=5)
print("detector event fraction:", batch.detectors[0].bit_count() / batch.shots)
print("first-shot logical flip:", batch.observable_bit(0, 0))

result = dem_simulator.estimate(shots=4096, seed=6, baseline=0.0)
print("uncorrected logical error rate:", result.mean_loss)
print("edge sensitivities:", result.edge_sensitivities)
```

Both rates should be close to 0.125. This example has no correction; pass a
decoder to `estimate(..., decoder=decoder)` to measure decoded residual failures.
See [decoding](decoding.md) for construction and layout requirements.

A `DemSampleBatch` contains detector masks, observable masks, and optionally
edge-event masks. It has no measurement record or Pauli-frame masks. If you
already have a DEM, reuse it with `DemHotspotEstimator`:

```python
from faultscope.dem import DemHotspotEstimator

existing_dem_sampler = DemHotspotEstimator(dem)
existing_batch = existing_dem_sampler.run_batch(shots=64, seed=7)
print("existing DEM shots:", existing_batch.shots)
```

## Choose Forward or DEM Sampling

Use [forward sampling](sampling.md) when you need measurement history or the
original joint distribution of a categorical noise channel. DEM sampling is
appropriate when an independent-mechanism model represents the experiment you
intend to run.

Circuit-to-DEM conversion has these limits:

- The current generator requires each measurement in the ideal noiseless
  circuit to be deterministic. A random intermediate measurement is rejected
  even if it is unused or cancels in a deterministic final detector parity.
  Changing only detector declarations cannot bypass this restriction. Use
  forward sampling for such circuits.
- Depolarizing noise uses an exact independent reparameterization up to rates
  `3/4` for one qubit and `15/16` for two qubits. Above those mixing limits,
  use forward sampling.
- One-qubit `PauliChannel` first attempts an independent X/Y/Z conversion.
  Channels that cannot be converted are rejected by default. Enable
  `approximate_disjoint_errors=True` only when the independent approximation
  is suitable; a float such as `0.01` permits it only below that component
  probability threshold. There is no general exact conversion attempt for
  wider Pauli channels.

For approximated channels, outcomes with the same detector/logical effect are
combined exactly first. Different effect classes are then treated as
independent. They can co-occur in a DEM sample even if they were mutually
exclusive in the original channel. The one-qubit numerical conversion uses an
absolute residual tolerance of `1e-14`, so sufficiently small effects can be
lost even without approximation enabled. The
[DEM reference](../api_reference.md#detector-error-models) records the full
conversion contract.

## Reuse Generation and Decoder Views

A compiled generator can produce a sampling DEM and decoder-ready views without
repeating circuit preparation:

```python
from faultscope.runtime import compile_native_dem_generator

generator = compile_native_dem_generator(circuit)
artifact = generator.generate_artifact()
print("canonical edges:", len(artifact.dem.edges))
print("graphlike hints:", artifact.graphlike_hints)
```

`artifact.dem` is the canonical model to sample. Optional
`GraphlikeDecompositionHints` describe how to construct a graphlike decoder
problem; they do not create extra independent sampling edges. Stim DEM `^`
groups can populate these hints. Native circuit generation currently does not
synthesize them. A graphlike decoder rejects an unhinted edge touching more
than two detectors instead of splitting it silently.

## Use a Sampling-Only Handle

If you only need samples, `materialize_dem=False` avoids building Python DEM
metadata:

```python
from faultscope.runtime import compile_native_dem_sampler_from_circuit

light_sampler = compile_native_dem_sampler_from_circuit(circuit, materialize_dem=False)
light_batch = light_sampler.run_batch(shots=64, seed=8)
print("materialized DEM:", light_sampler.dem)
print("light batch shots:", light_batch.shots)
```

`light_sampler.dem` is `None`. Sampling works, but estimates and hotspot APIs
that require DEM metadata raise `ValueError`. The same option is available on
`DemFaultScopeSimulator`.

## Interpret Sensitivities

DEM `edge_sensitivities` are derivatives with respect to independent edge
probabilities. They are dictionaries keyed by edge index; location
`sensitivities` summarize these edge derivatives. Location summaries do not
apply the chain rule for circuit-to-DEM probability conversion and are not
generally derivatives with respect to physical circuit rates. Use the forward
estimator for that question; see [noise sensitivity analysis](noise_sensitivity.md).

Hotspot analysis needs recorded event masks. A native batch must come from the
same compiled sampler used to estimate it. Missing events, a foreign sampler
batch, zero shots, or an invalid loss-mask width raise `ValueError`.

Next, [construct a decoder](decoding.md),
[collect many experiments](collection.md), or consult the
[DEM runtime API](../api_reference.md#dem-runtime-from-circuit).
