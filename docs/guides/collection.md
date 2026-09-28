# Collect Logical Error Rates

Use `Collector` to run circuit tasks in batches, stop at a shot or error limit,
and save statistics for later analysis. Rust performs sampling, native decoding,
postselection, scheduling, and counting. Python prepares tasks and manages CSV
files.

Install FaultScope using the
[installation instructions](../getting_started.md#installation).
The collection example needs no optional packages and uses no decoder. The
optional threshold example below also needs the collection extra. Python
blocks run in page order; the threshold example uses its own synthetic data. Save them as `collection_example.py`
and run them in the installed environment:

```bash
python -I collection_example.py
```

## Run a Noise-Rate Sweep

The simple circuit below has one logical observable and no correction. Its
logical error rate should be close to the X-noise probability. For a decoded
experiment, replace `make_circuit` with your QEC circuit and supply an appropriate
native decoder in each task.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from faultscope import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from faultscope.collection import (
    Collector,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    read_stats_from_csv_files,
)


def make_circuit(probability):
    return Circuit(
        1,
        (
            Operation.noise(
                NoiseLocation("x0", BernoulliPauliNoise("X"), probability, (0,))
            ),
            Operation.measure(0, key="m0"),
            Operation.detector(("m0",), detector_id=0),
            Operation.observable_include(0, ("m0",)),
        ),
    )


tasks = [
    CollectionTask(
        circuit=make_circuit(p),
        task_id=f"p={p}",
        metadata={"p": p},
        seed=100 + index,
    )
    for index, p in enumerate((0.01, 0.03))
]
collector = Collector(
    options=CollectionOptions(max_shots=4096, batch_size=512),
    run_options=CollectionRunOptions(num_workers=2),
)
stats = collector.collect(tasks)
for stat in stats:
    print(stat.task_id, stat.shots, stat.errors, stat.logical_error_rate)
```

`logical_error_rate` is `errors / (shots - discards)`. A shot is an error when
at least one non-postselected logical residual is nonzero. With a decoder, a
residual is the sampled logical flip XOR the decoder's prediction.
`raw_error_rate` uses all shots as its denominator. The corresponding logical
rate standard error is available as `logical_error_rate_stderr`.

## Add Decoding and Stop Conditions

Pass a native decoder handle or name in `CollectionTask.decoder`. The collection
API accepts native decoders; Python prototype decoders belong in
`simulator.estimate(...)`. See [decoding](decoding.md) for installation and
construction of matching decoders.

Circuit tasks always use forward sampling. A decoder name may generate a DEM
once to construct the decoder, but the shots still execute the original
circuit. `CollectionRunOptions(decoders=(...))` expands tasks that have no
decoder into one task per decoder; explicitly configured tasks keep theirs.

| Option | Meaning |
| --- | --- |
| `max_shots` | Required total shot cap, including resumed shots |
| `max_errors` | Stop when this count is reached, after `min_shots` |
| `min_shots` | Minimum total shots before error-count stopping is allowed |
| `batch_size` | Fixed batch size; the final batch respects the shot cap |
| `num_workers` | Worker limit shared by all tasks, including batches of one large task |

Error limits are checked after committed batches, so the count may exceed
`max_errors` by a batch. Task `collection_options` override only fields explicitly
set on that task. Use `options.with_edits(...)` to edit options while preserving
that intent; do not use `dataclasses.replace()` for `CollectionOptions`.

## Save, Resume, and Stream Progress

This example uses a temporary directory so rerunning the page starts clean.
For a real experiment, use a persistent path such as `Path("stats.csv")`.

```python
with TemporaryDirectory() as directory:
    csv_path = Path(directory) / "stats.csv"
    run_options = CollectionRunOptions(num_workers=2, save_resume_filepath=csv_path)
    first = Collector(
        options=CollectionOptions(max_shots=1024, batch_size=256),
        run_options=run_options,
    )
    for progress in first.iter_progress(tasks):
        for delta in progress.new_stats:
            print("committed:", delta.task_id, delta.shots)

    resumed = Collector(
        options=CollectionOptions(max_shots=2048, batch_size=256),
        run_options=run_options,
    ).collect(tasks)
    saved = read_stats_from_csv_files(csv_path)
    print("resumed totals:", [stat.shots for stat in resumed])
    print("saved totals:", [stat.shots for stat in saved])
```

The second collection continues to 2048 total shots per task. Completed tasks
are skipped. `save_resume_filepath` reads prior data and appends only new
statistics; `existing_data_filepaths` reads other files without changing them.
`iter_progress()` yields committed deltas and flushes configured CSV output
before each yield. `iter_collect()` yields final task totals, not live progress.

CSV rows merge by `strong_id`, a fingerprint of the source, decoder,
metadata, postselection, counter schema, and explicit task seed. Changing only
`task_id` changes the label, not the experiment identity. Duplicate identities
within one call are rejected. Use distinct task seeds for independent repeats.
See [stored-result compatibility](../release.md#upgrading-stored-results) when
reusing files written by older versions.

<span id="seeds-and-resume"></span>

## Reproduce a Run

A task seed overrides the run seed; `None` inherits the run seed. When neither
is set, the run draws one random root shared by its tasks. Metadata and decoder
identity do not select independent noise streams.

With unchanged sampling input, effective seed, fixed batch settings, and the
same continuation state, changing task order or worker count does not change
the sampled stream. Changing the decoder can still change the logical errors.
A resumed session uses a continuation stream and need not match the shot
sequence of one uninterrupted run.

Setting `max_batch_seconds` enables adaptive batching. Each task uses two or
three serial calibration batches, freezes a throughput-based batch size, and
then runs the remaining batches in parallel. Elapsed-time feedback means
adaptive runs do not promise identical error counts across worker counts or
machine loads. Fix the batch settings when that reproducibility matters.

## Postselect and Inspect Counters

Postselection discards shots with selected detector events or selected logical
residuals. Masks follow the sampler's detector or observable declaration order.
For the one-detector example, `postselection_mask=bytes([1])` selects detector
0. Discarded shots do not enter the logical error-rate denominator.

Enable `count_observable_error_combos` or `count_detection_events` on run options
to populate `TaskStats.custom_counts`. A legal `custom_error_count_key` can
select the counter used by `max_errors`. See the
[postselection and counter reference](../api_reference.md#postselection-and-custom-counters)
for mask layout, key syntax, and validation rules.

## Analyze Threshold Sweeps

For a threshold experiment, collect several physical error rates at several
code distances and record them in metadata, for example `{"p": 0.01, "d": 3}`.
Then `analyze_thresholds(stats, x_key="p", distance_key="d")` groups the points,
finds crossings between adjacent distances, and fits finite-size scaling.
`plot_threshold_analysis(...)` plots rates and scaling collapse. The toy sweep
above has no code-distance axis and is not a threshold experiment.

Plotting, bootstrap intervals, and scaling fits use the optional collection
packages. After [installing FaultScope](../getting_started.md#installation),
add the extra in the same active environment:

```bash
pip install "faultscope[collection]"
```

The following independent example demonstrates the analysis interface with
synthetic counts. These counts follow a chosen scaling curve; they are not
measurements of a physical code and do not establish a QEC threshold.

```python
import math

from faultscope.collection import TaskStats, analyze_thresholds

synthetic_stats = []
for distance in (3, 5, 7):
    for probability in (0.015, 0.02, 0.025):
        logit_rate = -2 + 12 * (probability - 0.02) * distance ** (1 / 1.5)
        rate = 1 / (1 + math.exp(-logit_rate))
        synthetic_stats.append(
            TaskStats(
                task_id=f"synthetic-d{distance}-p{probability}",
                shots=100_000,
                errors=round(100_000 * rate),
                discards=0,
                seconds=0.0,
                decoder=None,
                metadata={"p": probability, "d": distance},
            )
        )

analysis = analyze_thresholds(
    synthetic_stats,
    x_key="p",
    distance_key="d",
    bootstrap_samples=0,
    scaling_order=1,
)[0]
print("synthetic pairwise crossing:", analysis.pairwise_threshold.value)
print("synthetic scaling status:", analysis.scaling_fit.status)
```

The pairwise crossing should be near the chosen value 0.02. This example disables
bootstrap intervals; use a positive `bootstrap_samples` and a fixed analysis
`seed` to estimate intervals from experimental counts. A scaling fit requires
at least three distances, two noise-rate points per distance, overlapping rate
ranges, and enough data for positive fit degrees of freedom. Missing or
ambiguous crossings and insufficient fits are reported as statuses.

See the [threshold API](../api_reference.md#threshold-analysis) for the full
contract and the [benchmark guide](../development.md#measure-performance) for
the surface-code threshold comparison. The CLI offers
`collect`, `summarize`, `merge`, `fit`, `plot`, and `threshold`; inspect it with:

```bash
python -I -m faultscope.collection --help
```

## Other Collection Paths

`Collector.collect_hotspots(tasks)` also returns ordered batch statistics and
shot-weighted physical location sensitivities. It requires fixed batch sizes
and does not support CSV partial resume.

For an existing DEM, the library-only `faultscope.collection.dem` module provides
`DemCollectionTask(dem=...)` and `DemCollector`. Its sampling uses independent
DEM edges and its hotspot result contains `edge_sensitivities`. The main
collection module and CLI accept circuit tasks only. See the
[legacy DEM reference](../api_reference.md#legacy-dem-collection) for this path.

For throughput measurements, use the
[collection benchmark](../development.md#measure-performance) with a release
build. For all signatures and persistence contracts, see the
[collection API](../api_reference.md#collection-api).
