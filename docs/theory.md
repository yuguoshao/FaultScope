<span id="faultscope-detector-error-model"></span>

# Theory

FaultScope estimates how a chosen failure probability changes when a noise rate
changes. The Forward estimator differentiates physical circuit rates. The DEM
estimator differentiates probabilities in an independent-edge model. Their
sampling distributions and parameters must be kept distinct.

For the execution path, see [Architecture](implementation_overview.md). This page
defines the quantities, estimators, and conversion limits.

## Start with one qubit

Prepare a qubit in `|0>`, apply X with probability \(p\), then measure Z. Let the
loss be 1 when the measurement is 1. For this circuit,

\[
J(p)=\Pr(\text{loss}=1)=p,
\qquad
\frac{dJ}{dp}=1.
\]

Changing the rate from 0.1 to 0.2 raises the failure probability, but the
sensitivity stays 1. A hotspot measures the magnitude of a local derivative; it
is not the noise rate, the number of observed errors, or the probability that a
location caused a failure. Increasing a rate does not guarantee a higher hotspot
rank.

For a small change \(\delta\lambda_l\) at a physical location \(l\),

\[
\Delta J\approx g_l\,\delta\lambda_l,
\qquad
g_l=\frac{\partial J}{\partial\lambda_l}.
\]

FaultScope reports an estimate of the signed \(g_l\) and ranks locations by its
absolute value. The sign matters when deciding which direction to change a rate.

<span id="_1"></span>

## Notation

| Symbol | Meaning |
| --- | --- |
| \(N\), \(k\) | Number of shots and a shot index, with \(0\leq k<N\). |
| \(l\), \(\lambda_l\) | Physical noise location and its total event rate. |
| \(\tau_k\) | One shot's noise events, measurements, detectors, and observables. |
| \(F_k\) | Boolean loss for shot \(k\). |
| \(E_{kl}\) | 1 if an error or measurement flip occurred at location \(l\) in shot \(k\). |
| \(F\), \(E_l\) | Packed masks containing all \(F_k\) or \(E_{kl}\) bits. |
| \(O_a\), \(C_a\) | Packed observable and decoder correction masks for observable \(a\). |
| \(A\) | Mask with the low \(N\) bits set; it removes unused bits. |
| \(p_j\), \(f_j\) | Probability and occurrence variable of canonical DEM edge \(j\). |

A packed mask stores shot `k` in bit `k`. Python exposes the mask as an integer;
Rust stores it in `Mask` words. `popcount` counts its set bits.

<span id="_2"></span>

## Objective and loss

The target is the expected loss under the circuit's noise distribution:

\[
J(\lambda)=\mathbb E_{\tau\sim P_\lambda}[L(\tau)],
\qquad F_k=L(\tau_k)\in\{0,1\}.
\]

The default loss is residual logical failure. It XORs each observable with its
decoder correction, then ORs the residuals:

\[
R_a=O_a\oplus C_a,
\qquad
F=\bigvee_a R_a,
\qquad
\overline F=\frac{\operatorname{popcount}(F\mathbin{\&}A)}{N}.
\]

A missing correction is zero. An observable mask is therefore distinct from a
correction mask and from the final loss mask. Custom loss callbacks return the
same packed Boolean representation; their API signatures are listed in the
[API reference](api_reference.md).

The derivation below assumes independent shots, independently sampled physical
noise locations, and a fixed loss rule applied to each shot. It holds the Pauli
mixture weights and decoder behavior fixed. It does not include derivatives of
decoder construction, training, or an explicitly rate-dependent loss. A callback
that couples different shots also falls outside this per-shot derivation.

<span id="score-function"></span>

## Score-function derivation

At location \(l\), an event \(e_l\) has distribution
\(p_l(e_l;\lambda_l)\). Its score is

\[
s_l(\tau)=\frac{\partial\log p_l(e_l;\lambda_l)}{\partial\lambda_l}.
\]

The locations are independent, and ideal measurement randomness has no explicit
dependence on the noise rates. Differentiating the trajectory probability gives

\[
\begin{aligned}
g_l
&=\frac{\partial}{\partial\lambda_l}
  \sum_\tau P_\lambda(\tau)L(\tau)\\
&=\sum_\tau P_\lambda(\tau)L(\tau)
  \frac{\partial\log P_\lambda(\tau)}{\partial\lambda_l}\\
&=\mathbb E[L(\tau)s_l(\tau)].
\end{aligned}
\]

For the supported channels, the rate controls whether an event occurs; the
conditional choice of Pauli component is fixed. At an interior probability,

\[
s_l(\tau)=
\begin{cases}
1/\lambda_l,&E_{kl}=1,\\
-1/(1-\lambda_l),&E_{kl}=0.
\end{cases}
\]

This is why a single batch can estimate every location's rate sensitivity from
its event mask and the shared loss mask. No circuit rerun with a perturbed rate
is needed for each location.

## Baseline and finite samples

For a fixed baseline \(b\), or one independent of the sampled shot,
\(\mathbb E[s_l]=0\) implies

\[
g_l=\mathbb E[(L-b)s_l].
\]

The finite-sample estimator is

\[
\widehat g_l=\frac1N\sum_{k=0}^{N-1}(F_k-b)s_{kl}.
\]

FaultScope defaults to \(b=\overline F\), the mean loss of the same batch. This
baseline depends on the samples, so the fixed-baseline argument does not make
that estimate exactly unbiased. For iid shots and an unclipped interior rate,

\[
\mathbb E[\overline F\,\overline s_l]=\frac{g_l}{N},
\qquad
\mathbb E[\widehat g_l]=\left(1-\frac1N\right)g_l.
\]

The implementation divides by \(N\) and does not apply an \(N/(N-1)\) correction.
The bias vanishes as the batch size grows; at \(N=1\), the default estimate is
zero. An explicitly supplied fixed baseline, including zero, avoids this
particular bias, though its variance can differ. Baselines can also change a
finite batch's ranking.

### Probability limits and rare events

Sampling uses the model's actual rate. Score evaluation uses

\[
\widetilde p=\operatorname{clamp}(\lambda_l,10^{-12},1-10^{-12}),
\qquad
s_1=\frac1{\widetilde p},
\qquad
s_0=-\frac1{1-\widetilde p}.
\]

Clipping prevents division by zero; it does not provide an exact derivative at
an endpoint. If clipping changes the rate, the score is no longer the exact
log-probability derivative of the sampled distribution.

At rate 0 or 1, the event mask is constant. With the default same-batch baseline,
the estimate cancels to zero, up to floating-point error. This does not imply a
zero one-sided derivative: the one-qubit example still has \(dJ/dp=1\). The same
cancellation occurs in a finite batch that happens to contain no events or only
events. Rare-event estimates therefore need enough shots to observe both cases;
a zero estimate alone is not evidence of zero sensitivity.

<span id="score"></span>

## Noise models

Every row below has the same rate score. The derivative is with respect to
\(\lambda_l\), holding any conditional Pauli weights fixed.

| Model | Event distribution at rate \(\lambda\) | Meaning of a set event bit |
| --- | --- | --- |
| <span id="bernoulli-pauli"></span>`BernoulliPauliNoise(P)` | \(\Pr(I)=1-\lambda\), \(\Pr(P)=\lambda\). | The specified Pauli error occurred. |
| <span id="measurement-bit-flip"></span>`MeasurementBitFlip()` | No flip with probability \(1-\lambda\); flip with probability \(\lambda\). | The recorded measurement bit was flipped. |
| <span id="single-qubit-depolarizing"></span>`SingleQubitDepolarizing()` | Identity with probability \(1-\lambda\); each of X, Y, Z with probability \(\lambda/3\). | Any non-identity Pauli occurred. |
| <span id="two-qubit-depolarizing"></span>`TwoQubitDepolarizing()` | II with probability \(1-\lambda\); each of the other 15 Paulis with probability \(\lambda/15\). | Any non-identity two-qubit Pauli occurred. |
| <span id="pauli-channel"></span>`PauliChannel({P_i: w_i})` | Identity with probability \(1-\lambda\); component \(P_i\) with probability \(\lambda w_i/\sum_jw_j\). | One channel component was selected. |

Forward sampling selects at most one component per location. In particular,
Pauli-channel sensitivities are not derivatives with respect to the individual
\(w_i\).

<span id="packed-forward"></span>

## Packed counts

For a location's event mask \(E=E_l\), define three counts:

\[
F_A=F\mathbin{\&}A,
\qquad
n_F=\operatorname{popcount}(F_A),
\qquad
n_E=\operatorname{popcount}(E),
\qquad
n_{FE}=\operatorname{popcount}(F_A\mathbin{\&}E).
\]

The two score sums and the estimate are

\[
\begin{aligned}
\operatorname{sum\_loss\_score}
 &=n_{FE}s_1+(n_F-n_{FE})s_0,\\
\operatorname{sum\_score}
 &=n_Es_1+(N-n_E)s_0,\\
\widehat g_l
 &=\frac{\operatorname{sum\_loss\_score}
         -b\,\operatorname{sum\_score}}N,\\
h_l&=|\widehat g_l|.
\end{aligned}
\]

These are exactly the per-shot sums above, evaluated with bit operations and
`popcount`. `mean_loss` is \(n_F/N\), `sensitivities` contains \(\widehat g_l\),
and `hotspots` contains \(h_l\).

<span id="packed-masks-measurements-detector-syndromes-observables"></span>

### Measurements, detectors, and observables

<span id="measurement-masks"></span>

For measurement key \(m\), the packed mask is

\[
M_m=\sum_{k=0}^{N-1}m_{k,m}2^k.
\]

For example, `0b1010` means that shots 1 and 3 measured 1.
Keys identify individual records and must be unique after
repeat expansion. Attached measurement noise is applied before recording the
mask, so all downstream parities see the noisy result.

<span id="detector-masks"></span>

For a detector declared on measurement keys \(K_i\),

\[
S_i=\bigoplus_{m\in K_i}M_m.
\]

The Forward runtime evaluates this parity directly. A declaration should have
zero ideal parity when a set detector bit is intended to mean a violation.

<span id="observable-masks"></span>

An observable can combine measurement keys \(K_a\) with a final Pauli-frame
projection \(Q_a\):

\[
O_a=\left(\bigoplus_{m\in K_a}M_m\right)\oplus Q_a.
\]

Circuit `observable_include` operations XOR their contributions into the same
observable ID. Separately supplied `LogicalObservable` definitions are evaluated
at the end of the batch. The [objective](#objective-and-loss) combines observable
and correction masks into the loss.

<span id="tag"></span>

### Grouping physical hotspots

Forward results sum absolute hotspots by qubit and metadata tag:

\[
\operatorname{by\_qubit}[q]=\sum_{l:q\in\operatorname{qubits}(l)}h_l,
\qquad
\operatorname{by\_round}[r]=\sum_{l:\operatorname{tags}(l)[\text{round}]=r}h_l.
\]

`by_gate` and `by_operation` use the same rule for their tags. A two-qubit
location contributes its full hotspot to each qubit. These are sums of absolute
values, so they do not cancel opposing signed sensitivities.

<span id="detector-error-model"></span>
<span id="detector-formalism-packed-masks"></span>

## Detector error models

A detector error model describes the joint distribution of detector and logical
observable flips. It does not retain the full stabilizer state, measurement
history, or final Pauli frame.

Let \(D\) contain the declared measurement parities, one row per detector. Let
\(\Omega\) contain the measurement flips caused by each error mechanism, one
column per mechanism. All matrix operations here are over \(\mathbb F_2\):

\[
H=D\Omega,
\qquad
H_{ij}=1\iff\text{mechanism }j\text{ flips detector }i.
\]

The logical fault matrix \(G\) similarly records observable flips. A canonical
DEM edge stores its probability \(p_j\), the support of column \(j\) of \(H\)
and \(G\), and source metadata such as `location_id`, event label, and tags.
Effects are defined relative to the ideal reference result.

```text
error(p_j) D0 D3 L0
```

This is one mechanism: when it occurs, it toggles both detector bits 0 and 3 and
observable bit 0. An edge is a model instruction, not a gate or a fault that has
already occurred in a shot.

<span id="dem-edge-sampling"></span>

### Independent edge sampling

The DEM sampler draws one independent Bernoulli variable for each canonical
edge:

\[
f_j\sim\operatorname{Bernoulli}(p_j),
\qquad
s=Hf,
\qquad
o=Gf.
\]

Thus \(s_i=\bigoplus_{j:H_{ij}=1}f_j\) and
\(o_a=\bigoplus_{j:G_{aj}=1}f_j\). The default residual logical loss is the same
\(F=\bigvee_a(O_a\oplus C_a)\) used by Forward sampling.

Handwritten and imported `error(p)` instructions follow this independent-edge
semantics. Shared location labels do not make different edges mutually
exclusive. FaultScope's canonical DEM has no general categorical error-group
schema.

<span id="dem"></span>

## Circuit-to-DEM conversion

`DetectorErrorModelGenerator` propagates individual error mechanisms through a
circuit. Conceptually, each edge's support is the XOR between an ideal reference
and a run with that mechanism injected:

```text
detector support = reference detectors XOR injected detectors
observable support = reference observables XOR injected observables
```

An empty support has no effect on the modeled detector/observable distribution
and can be omitted. How physical event probabilities become independent DEM
probabilities depends on the channel.

### Measurement restrictions

The current generator requires each explicit single-qubit or Pauli-string
measurement to have a deterministic ideal result. It checks this even when the
measurement is unused by a detector or observable. Resets that record a
measurement result have the same requirement. Deterministic final parities are
therefore not enough to admit random intermediate measurements.

Forward sampling supports ideal measurement randomness. The native DEM generator
does not implement Stim's gauge-detector elimination. This stricter conversion
boundary should be checked before choosing DEM sampling for a circuit.

### Depolarizing channels

For an \(n\)-qubit uniform depolarizing channel, write \(K=4^n\). The physical
channel has identity probability \(1-\lambda\) and probability
\(\lambda/(K-1)\) for each non-identity Pauli. Its exact independent
factorization assigns every non-identity Pauli the Bernoulli probability

\[
q_n(\lambda)=
\frac{1-\left(1-\frac{K}{K-1}\lambda\right)^{2/K}}2.
\]

Multiplying the independently sampled Pauli factors reproduces the original
categorical distribution. This real-valued factorization requires
\(0\leq\lambda\leq(K-1)/K\): the circuit-to-DEM limit is 3/4 for one qubit and
15/16 for two qubits. Forward sampling accepts the model's full rate range
\([0,1]\).

Factors with identical detector/observable support within a location combine by
parity, with probability \(a+b-2ab\). Their event label uses `P_i^P_j`. Factors
with empty support can be omitted. Parallel edges from different locations stay
separate to preserve attribution, so edge counts or text can differ from Stim's
global simplifications while representing the same joint distribution.

### Weighted Pauli channels

For `PauliChannel({P_i: w_i})`, component probabilities are
\(p_i=\lambda w_i/\sum_jw_j\). A channel with one positive-probability component
is exactly Bernoulli. With several components, the physical channel chooses at
most one, which independent DEM instructions cannot represent in general.

For a one-qubit channel, conversion first tries independent X/Y/Z Bernoulli
factors whose Pauli product reconstructs the categorical probabilities. The
solver accepts a sum of absolute reconstruction errors below \(10^{-14}\).
Consequently, very small effects can fall below this tolerance even if the
channel is not mathematically factorizable. Use Forward sampling if those
effects matter. General multiqubit Pauli channels do not use this exact solver.

When an exact path is unavailable, conversion rejects the channel unless
`approximate_disjoint_errors` is enabled. `True` permits the approximation; a
numeric threshold in \([0,1]\) permits it only when every component probability
is at most that threshold.

After propagation, mutually exclusive components with identical support combine
by adding their probabilities and use an event label such as `P_i|P_j`.
Different support classes then become independent instructions. Their original
mutual exclusion is lost: two classes can now occur together with probability
equal to the product of their probabilities. Shared `location_id` values retain
attribution but do not restore that dependence.

<span id="dem-hotspot"></span>

## DEM sensitivities

For the independent-edge model, the derivative parameter is \(p_j\):

\[
\widehat g^{\mathrm{DEM}}_j
=\frac1N\sum_k(F_k-b)
\left(\frac{f_{kj}}{\widetilde p_j}
      -\frac{1-f_{kj}}{1-\widetilde p_j}\right),
\qquad
h^{\mathrm{DEM}}_j=|\widehat g^{\mathrm{DEM}}_j|.
\]

Here \(\widetilde p_j\) is the same clipped probability used for Forward
scores. The [packed-count formula](#packed-counts), same-batch baseline bias,
and endpoint limits apply with an edge-event mask in place of a physical
location-event mask. The result's `edge_sensitivities` and `edge_hotspots` expose
these quantities.

<span id="dem-location"></span>

### Location summaries are not a chain rule

For the set \(\mathcal E_l\) of edges carrying a location ID, FaultScope defines

\[
P_l=\sum_{j\in\mathcal E_l}p_j,
\qquad
w_j=
\begin{cases}
p_j/P_l,&P_l>0,\\
1/|\mathcal E_l|,&P_l=0,
\end{cases}
\]

\[
\widehat g^{\mathrm{summary}}_l
 =\sum_{j\in\mathcal E_l}w_j\widehat g^{\mathrm{DEM}}_j,
\qquad
h^{\mathrm{summary}}_l=|\widehat g^{\mathrm{summary}}_l|.
\]

These probability-weighted summaries populate the DEM result's location
`sensitivities` and `hotspots`. Tag aggregation uses the location hotspots after
this signed averaging, so edge derivatives of opposite signs can cancel.

This summary does not apply the conversion's parameter chain rule. If
\(p_j=p_j(\lambda_l)\), the corresponding model derivative would instead involve

\[
\frac{\partial J_{\mathrm{DEM}}}{\partial\lambda_l}
=\sum_j\frac{\partial J_{\mathrm{DEM}}}{\partial p_j}
       \frac{\partial p_j}{\partial\lambda_l}.
\]

In particular, depolarizing factors use \(q_n(\lambda_l)\), not the physical
rate itself. For an approximate Pauli-channel conversion, the edge derivatives
also describe the approximate independent model. Use the Forward hotspot
estimator for derivatives of the original circuit's physical rates.

<span id="detector-graph"></span>

## Detector graph projection

`project_sensitivities_to_detector_graph(...)` distributes supplied location
sensitivities across their DEM edges using the same probability weights above.
The DEM result's `detector_graph_hotspots` starts from its edge sensitivities.
Both organize contributions by a `(detector_tuple, observable_tuple)` key:

```text
((3, 4), ())      # detector edge
((5,), (0,))      # one detector and a logical flip
```

| Result field | Aggregation |
| --- | --- |
| `by_detector_edge`, `signed_by_detector_edge` | Sum absolute or signed edge contributions with the same support key. |
| `by_detector`, `signed_by_detector` | Divide each edge contribution equally among the detectors it touches, then sum. |
| `by_observable`, `signed_by_observable` | Divide each edge contribution equally among the observables it touches, then sum. |
| `by_location`, `signed_by_location` | Sum edge contributions by source location. |

Absolute and signed summaries answer different questions. Summing absolute edge
contributions can exceed the absolute value of their signed sum; a projected
`by_location` value need not equal the DEM result's separately weighted location
hotspot. `project_hotspots_to_edges(...)` exposes the location-to-edge allocation
keyed by `(location_id, event)`.

<span id="pymatching"></span>

### Matching uses a decoder view

A matching decoder uses detector support matrix \(H\), logical fault matrix
\(G\), and log-odds weights

\[
w_j=\log\frac{1-p_j}{p_j}.
\]

Its solver components must touch one or two detectors. An edge with a logical
flip but no detector is rejected by this interface. Observable support is
passed as logical fault information so that decoding predicts correction masks,
not physical error locations.

Imported graphlike decomposition hints can describe solver components for a
canonical parent edge. Their detector and observable supports must XOR back to
the parent's full support. All components retain the parent's probability and
`dem_edge_index`. The sampler still draws the parent event once and toggles its
full support; it never samples those components independently.

The matching backend treats solver components as an uncorrelated graphlike
approximation. Shared parent metadata does not implement correlated matching.
The native circuit-to-DEM generator currently returns no graphlike hints;
unhinted edges with more than two detectors are rejected by this matching path.

<span id="_3"></span>

## Implementation references

The count formulas are implemented in `crates/faultscope-core/src/hotspot.rs`.
Forward event masks come from `packed.rs`; DEM edge events come from
`dem_sampling.rs`. Channel conversion is in `dem/event_plan.rs`.

Forward hotspot estimation checks that the batch has positive shots, event
recording, the correct compiled-program identity, and a loss mask of the expected
width before aggregation. Missing event records are an invalid input, not
observations of zero events. The [Architecture](implementation_overview.md) page
shows how these components fit together.
