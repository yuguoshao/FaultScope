# FaultScope User Guide

Start with [Getting Started](getting_started.md) to install FaultScope and run a
surface-code memory experiment. Then choose the guide for your next task.
Each guide has a complete example and explains its outputs and limits.

<span id="contents"></span>
<span id="choosing-a-workflow"></span>

## Choose a Task

| Goal | Guide |
| --- | --- |
| Sample measurement records, detectors, and logical observables | [Forward sampling](guides/sampling.md) |
| Estimate failure after decoding | [Decoder integration](guides/decoding.md) |
| Sweep circuits or noise rates, save statistics, and resume | [Logical error-rate collection](guides/collection.md) |
| Generate or sample detector error models | [DEM generation and sampling](guides/dem.md) |
| Find influential physical noise locations | [Noise sensitivity analysis](guides/noise_sensitivity.md) |

For exact signatures and types, use the [API Reference](api_reference.md).
For the estimators and their assumptions, use [Theory](theory.md).

## Installation and Circuit Basics

<span id="installation-and-build"></span>

- [Install from source](getting_started.md#install-from-source): supported tools,
  the virtual environment, and optional dependencies.

<span id="imports-and-package-layout"></span>
<span id="core-concepts"></span>

- [Build a noisy circuit](guides/sampling.md#build-and-sample-a-circuit): circuit
  operations, noise locations, detector declarations, and observables.
- [Public imports](api_reference.md#import-surface): supported Python modules.

<span id="stim-import"></span>

- [Import Stim text](guides/sampling.md#import-stim-text): the supported subset,
  including `REPEAT` and `SHIFT_COORDS`.

<span id="repetition-code-experiments"></span>

- [Run a surface-code experiment](getting_started.md#simulate-a-surface-code-memory),
  or use the built-in [repetition-code builder](api_reference.md#repetition-code-experiments).

## Sampling and Analysis

<span id="forward-sampling-and-estimates"></span>
<span id="packed-mask-basics"></span>

- [Sample and inspect packed results](guides/sampling.md): masks, bit helpers,
  seeds, and simulator reuse.

<span id="pymatching-decoding"></span>
<span id="native-decoder-fast-path"></span>

- [Choose and install a decoder](guides/decoding.md): Python prototypes, optional
  native backends, and the static decoding problem.

<span id="detector-error-models"></span>
<span id="dem-sampling-and-hotspots"></span>

- [Generate and sample a DEM](guides/dem.md): deterministic-measurement
  requirements, exact conversions, approximation, and light samplers.

<span id="quickstart-forward-hotspots"></span>

- [Estimate noise sensitivities](guides/noise_sensitivity.md): signed derivatives,
  absolute hotspot rankings, custom loss, and Forward versus DEM interpretation.

<span id="logical-error-rate-collection"></span>
<span id="seeds-repeats-and-resume-identity"></span>

- [Collect logical error rates](guides/collection.md): limits, parallel workers,
  progress, CSV resume, reproducibility, and threshold sweeps.

<span id="legacy-explicit-dem-collection"></span>

- [Explicit DEM collection](api_reference.md#legacy-dem-collection): the
  library-only interface for tasks that already contain a DEM.

## Troubleshooting and Development

<span id="troubleshooting"></span>
<span id="modulenotfounderror-faultscope_native"></span>
<span id="unsupportednativecircuiterror"></span>
<span id="dem-generation-fails-on-a-random-measurement"></span>
<span id="pymatching-is-unavailable"></span>
<span id="the-default-loss-raises-about-missing-observables"></span>
<span id="should-i-use-rng-or-seed"></span>

- [Troubleshooting](getting_started.md#troubleshooting): installation and common
  runtime errors. DEM-specific limits are explained in the
  [DEM guide](guides/dem.md#choose-forward-or-dem-sampling); seed behavior is in
  [sampling](guides/sampling.md#reuse-the-simulator) and
  [collection](guides/collection.md#reproduce-a-run).

<span id="benchmarks"></span>

- [Measure performance](development.md#measure-performance) with release builds
  and the repository benchmark scripts.

<span id="best-practices"></span>

- Reuse compiled simulators, keep location ids unique, use explicit seeds for
  experiments you need to reproduce, and check model assumptions before using a
  DEM approximation. See [development checks](development.md#run-checks) when
  contributing changes.

The anchors on this page preserve links from older versions of the user guide.
