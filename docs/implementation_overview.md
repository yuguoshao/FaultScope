<span id="faultscope"></span>

# Architecture

FaultScope samples noisy stabilizer circuits and estimates how a chosen loss
changes with each noise rate. Python defines circuits, decoders, and loss
callbacks. Rust compiles the circuit, runs batches, and aggregates the results.

This page follows the execution path. See [Theory](theory.md) for the estimators
and [API reference](api_reference.md) for method signatures.

## From a circuit to a result

```text
Circuit + optional detector/observable declarations
    |
    v
Validate and expand operations; assign integer IDs
    |
    v
Compile ideal measurements with a symbolic stabilizer
    |
    v
Immutable SamplerProgram
    |
    v
Forward batch: Pauli frames + measurement/detector/observable masks
    |
    v
Optional decoder -> correction masks -> loss mask
    |
    +--> Loss counts
    |
    +--> Physical-location sensitivities and hotspot rankings
         (requires recorded noise-event masks)
```

A detector is a parity of measurement results. An observable records a logical
flip. The decoder predicts observable corrections from detector data; the loss
rule decides which shots count as failures.

## Circuit compilation

The compiler validates the supported operations and their qubit targets, then
expands the operation tree. Measurement keys and noise-location labels become
dense integer IDs. Names and tags remain in boundary tables for input resolution
and returned results.

A shared symbolic stabilizer tracks the ideal circuit. Each ideal measurement
becomes an expression containing a constant bit and an XOR of independent random
bits. A deterministic measurement may therefore depend on earlier random
measurements without introducing a new random source. Ideal Pauli gates affect
this symbolic state and need no separate instruction in the batch sampler.

The resulting `SamplerProgram` stores the executable instructions, measurement
expressions, observable definitions, noise locations, and allocation sizes. It
is immutable and can be reused for many batches.

### Repeated circuits

`REPEAT` bodies are expanded into individual operations, measurements, and noise
locations. Record lookbacks resolve against the expanded measurement history;
repeated explicit labels receive instance-specific names.

During symbolic compilation, suitable repeated bodies can reuse an affine
expression template when the stabilizer support cycles after one or two
iterations. This reduces compilation work. The executable sampler still contains
expanded instructions; it does not run a compressed loop. The
`loop_kernel_count` statistic describes this compilation optimization.

<span id="batch-runtime"></span>

## Forward batches

`FaultScopeSimulator` runs the compiled program using packed masks. Bit `k` in a
mask belongs to shot `k`; a Rust `u64` word holds 64 shots. Python exposes masks as
integers.

| Runtime data | Meaning of a set bit |
| --- | --- |
| `X_frame[q]`, `Z_frame[q]` | The shot has an X or Z Pauli-frame component on qubit `q`. |
| Measurement mask | The recorded measurement result is 1. |
| Detector mask | The declared measurement parity is 1. |
| Observable mask | The declared logical observable is 1. |
| Noise-event mask | A non-identity error or measurement flip occurred at that location. |

The sampler first generates masks for the ideal measurement random sources.
Clifford instructions then update the frame masks with swaps and XORs. For
example, H swaps the X and Z components. A measurement combines its compiled
ideal expression with the frame's anticommutation bit and any attached
measurement noise. Reset clears the local frame after optionally recording a
measurement result.

Detectors take XORs of recorded measurements. Observable declarations can use
measurement parity, a final Pauli-frame projection, or both.

Noise sampling preserves the channel's semantics. A depolarizing or weighted
Pauli channel first decides whether an event occurs, then chooses one Pauli
component. Those components are mutually exclusive within a physical location.
Low-rate Bernoulli sampling skips runs of shots with no event instead of drawing
once for every shot.

Ordinary sampling can omit noise-event masks. Hotspot estimation records one
such mask per location so it can compare event occurrence with the loss.

<span id="pymatching-decoder"></span>
<span id="_1"></span>

## Decoders and loss

The decoder receives detector data and returns logical correction masks. The
default loss marks a shot when any observable differs from its correction:

```text
residual[a] = observable[a] XOR correction[a]
loss_mask = OR over all residual[a]
```

Without a decoder, missing corrections are zero. A custom loss callback can
replace this rule; its returned integer still uses one bit per shot. Callback
signatures and decoder adapters are described in the
[API reference](api_reference.md).

Native decoders keep the batch path in Rust. Python callbacks cross the binding
boundary to inspect batch masks. Both paths produce a loss mask for the same
aggregation step. Decoder parameters are held fixed by the score estimator; the
estimator does not differentiate decoder construction or training.

Matching decoders use a graphlike view of a DEM. Each solver component must touch
one or two detectors, and a logical-only error with no detector cannot be
represented by this matching interface. Imported graphlike decomposition hints
can split a parent edge for the solver while leaving the sampler's parent event
intact. See [Decoder development](decoder_development.md) for that boundary.

<span id="_2"></span>
<span id="_3"></span>

## Hotspot aggregation

Rust counts loss shots, event shots, and shots containing both with bitwise AND
and `popcount`. It uses these counts to estimate one signed sensitivity per
physical noise rate. The absolute value is the hotspot used for ranking.

A positive sensitivity means that a small increase in the rate increases the
chosen loss; a negative value means the opposite. A hotspot is neither an error
probability nor a count of observed faults. Raising a location's error rate does
not necessarily raise its hotspot.

Results include the mean loss, signed sensitivities, absolute hotspots, top
locations, and sums by qubit, round, gate, and operation. The default baseline is
the same batch's mean loss. Its finite-sample bias and the limits at probabilities
near 0 or 1 are explained in [Theory](theory.md#baseline-and-finite-samples).

Estimation requires positive shots, recorded events, a state from the same
compiled program or a clone, and a loss mask of the expected width. The program
checks these conditions before entering the aggregation loop.

<span id="collection-runtime"></span>

## Collection scheduling

The main collection API accepts `CollectionTask(circuit=...)` and samples through
the Forward runtime. Optional detector and observable sequences replace the
corresponding circuit declarations; `None` keeps them, while an explicit empty
sequence removes them.

Tasks sharing a source reuse the compiled sampler during one collection call.
The Rust scheduler manages batches, workers, stopping conditions, seeds, and
decoder worker caches. With a fixed seed and fixed batch layout, batch contents
do not depend on the worker count.

A decoder selected by name may need a DEM during task preparation to build its
static decoding problem. This does not switch sampling to the DEM runtime.
Without a decoder, or with an already constructed native decoder, ordinary
collection does not generate a DEM.

Forward hotspot collection enables event recording and combines batch
sensitivities using shot-count weights. Ordinary collection leaves event
recording off. Explicit DEM collection remains a separate legacy API under
`faultscope.collection.dem`; it is not exported from the main collection module
or exposed through the collection CLI.

<span id="dem-hotspot-mode"></span>

## Forward and DEM execution

A DEM is an alternative sampling model that keeps detector and observable effects
of errors. It does not retain the full measurement history or Pauli frames.

| Question | Forward runtime | DEM runtime |
| --- | --- | --- |
| Input | Circuit compiled to `SamplerProgram` | `DetectorErrorModel` compiled to independent edge instructions |
| Sampled object | Physical noise event at each circuit location | Bernoulli event for each canonical DEM edge |
| Retained results | Measurements, detectors, observables, frames, optional location events | Detectors, observables, optional edge events |
| Sensitivity parameter | Original physical location rate | DEM edge probability |
| Location result | Physical-rate derivative estimate | Probability-weighted summary of edge derivatives |
| General categorical Pauli channels | Preserves the mutually exclusive component choice | Needs an exact conversion or an explicitly enabled approximation |

`DemHotspotEstimator` samples an existing DEM. `DemFaultScopeSimulator(circuit)`
first generates a DEM and then uses that same DEM sampler. Neither class returns
to Forward circuit sampling during a batch. Use Forward hotspots when the
quantity of interest is the derivative with respect to the circuit's physical
noise rate.

<span id="detector-error-model"></span>

## Circuit-to-DEM compilation

`DetectorErrorModelGenerator` propagates individual error mechanisms to find
which detectors and observables they flip. Each canonical DEM edge records a
probability, detector support, observable support, and source attribution.
Mechanisms with no detector or observable effect can be omitted.

The current generator requires every explicit Pauli measurement to have a
deterministic ideal result, including measurements unused by detectors. Resets
that record a measurement result have the same requirement. A circuit with
random intermediate measurements is therefore not accepted merely because their
final parity is deterministic. Forward sampling supports such randomness. The
generator does not implement gauge-detector elimination.

Uniform depolarizing noise uses an exact independent factorization within its
supported rate range. One-qubit `PauliChannel` conversion also tries to find
independent X/Y/Z factors. Other multicomponent channels require
`approximate_disjoint_errors` when no exact path is available. The approximation
can lose the original channel's mutually exclusive event choices. Conversion
formulas, numerical tolerances, and attribution rules are in
[Theory](theory.md#circuit-to-dem-conversion).

Canonical edges determine the sampling distribution. Graphlike hints belong to
a decoder view and do not create new independent sampling events. The native
circuit-to-DEM generator currently produces no graphlike hints.

<span id="_4"></span>

## Supported circuit model

The Forward runtime supports H, S, S_DAG, CX, CZ, SWAP, ideal Pauli strings,
X/Y/Z and Pauli-string measurements, basis resets, Bernoulli Pauli noise,
weighted Pauli channels, one- and two-qubit depolarizing noise, and attached
measurement bit-flip noise. Detectors and observables declare how results are
combined.

Operations follow a fixed circuit for all shots. General per-shot adaptive
branching, non-Clifford gates, and non-Pauli channels such as amplitude damping
are outside this runtime. Such channels need a suitable Pauli approximation
before they can enter the model.

<span id="stim-import-subset"></span>

### Stim import

`parse_stim_circuit(...)` and `load_stim_file(...)` import the supported Stim text
subset, including nested `REPEAT` blocks, measurement record lookbacks,
`SHIFT_COORDS`, and detector coordinates. The returned circuit contains its
detector and observable declarations and can enter either compilation path,
subject to that path's restrictions.

The importer is not a complete Stim interpreter. Feedback targets and
`CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` are examples of unsupported features;
unsupported syntax raises `StimImportError`.

<span id="_5"></span>

## Validation and source map

Useful checks compare sampling distributions with analytic toy circuits, verify
seeded equivalence of repeated and explicitly expanded circuits, and compare
hotspot estimates with analytic derivatives or finite differences. Monte Carlo
comparisons need statistical tolerances and must account for the chosen
baseline. Increasing a rate alone is not a valid test of hotspot rank.

| Responsibility | Main Rust source |
| --- | --- |
| Operation expansion and integer IDs | `faultscope-core/src/program.rs` |
| Symbolic compilation and repeated-body reuse | `faultscope-core/src/compile.rs`, `faultscope-core/src/stabilizer.rs`, `faultscope-core/src/expr.rs` |
| Forward batch execution | `faultscope-core/src/packed.rs`, `faultscope-core/src/sampling.rs` |
| Circuit-to-DEM generation | `faultscope-core/src/dem/` |
| DEM batches | `faultscope-core/src/dem_sampling.rs` |
| Sensitivity and hotspot aggregation | `faultscope-core/src/hotspot.rs` |
| Batch scheduling and collection | `faultscope-collection/src/` |

These paths are relative to `crates/`.
