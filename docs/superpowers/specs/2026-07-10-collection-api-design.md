# Collection API Consolidation Design

## Status

Approved for implementation on 2026-07-10.

## Goals

- Make `Collector` the primary Python interface for logical error-rate collection.
- Reduce the functional API from eighteen keyword arguments to immutable option objects.
- Give final-result and streaming-progress iteration distinct return types.
- Replace ambiguous `TaskStats.error_rate` and `TaskStats.stderr` properties with names that expose their denominators.
- Preserve the native Rust scheduler, batch-delta resume safety, CSV schema, strong-id algorithm, and counting semantics.

## Non-Goals

- No Rust collection public API changes.
- No scheduler, decoder, sampling, CSV schema, or strong-id changes.
- No Python decoder compatibility path.
- No new persistence backend or progress event schema.

## Public API

### Collection Options

`CollectionOptions` remains the sampling and stopping configuration:

```python
@dataclass(frozen=True)
class CollectionOptions:
    max_shots: int | None = None
    max_errors: int | None = None
    batch_size: int = 10_000
    start_batch_size: int | None = None
    max_batch_size: int | None = None
    max_batch_seconds: float | None = None
```

`seed` is removed from `CollectionOptions`. A task-level `collection_options` value continues to override the Collector defaults for these fields.

### Run Options

```python
@dataclass(frozen=True)
class CollectionRunOptions:
    seed: int | None = None
    num_workers: int = 1
    existing_data_filepaths: tuple[str | Path, ...] = ()
    save_resume_filepath: str | Path | None = None
    count_observable_error_combos: bool = False
    count_detection_events: bool = False
    custom_error_count_key: str | None = None
    decoders: tuple[str | object, ...] = ()
```

`num_workers` must be positive. File path and decoder collections are tuples so a reusable Collector never retains a one-shot iterable. A task with an explicit decoder is not expanded by run-level decoder fanout.

The run seed is sent to the Rust scheduler. Python task options do not duplicate it. The scheduler therefore derives deterministic task-local streams from the run seed and task strong id.

### Collector

```python
class Collector:
    def __init__(
        self,
        *,
        options: CollectionOptions | None = None,
        run_options: CollectionRunOptions | None = None,
    ) -> None: ...

    def collect(self, tasks: Iterable[CollectionTask]) -> list[TaskStats]: ...
    def iter_collect(self, tasks: Iterable[CollectionTask]) -> Iterator[TaskStats]: ...
    def iter_progress(self, tasks: Iterable[CollectionTask]) -> Iterator[Progress]: ...
```

Collector instances contain only immutable configuration. Each method invocation starts an independent collection run and may safely re-read configured resume data.

- `collect` returns final totals in expanded task order.
- `iter_collect` yields final `TaskStats` values and never yields `Progress`.
- `iter_progress` yields committed batch-delta `Progress` values and never yields `TaskStats` directly.

### Functional Wrappers

```python
def collect(tasks, *, options=None, run_options=None) -> list[TaskStats]: ...
def iter_collect(tasks, *, options=None, run_options=None) -> Iterator[TaskStats]: ...
def iter_progress(tasks, *, options=None, run_options=None) -> Iterator[Progress]: ...
```

Each wrapper constructs a Collector and delegates to the corresponding method. The existing individual collection kwargs, `progress_mode`, public `progress_callback`, and `print_progress` are removed before the first release.

The CLI uses the same private progress sink as `iter_progress` when progress output is needed, while retaining the final totals returned by the collection run. Internal native callbacks remain private and continue to flush resume CSV deltas before acknowledging committed batches. This avoids a second sampling run and does not expose a public callback API.

## Task Statistics

`TaskStats` exposes these rate properties:

```python
accepted_shots = shots - discards
raw_error_rate = errors / shots
accepted_error_rate = errors / accepted_shots
logical_error_rate = accepted_error_rate
accepted_error_rate_stderr = sqrt(p * (1 - p) / accepted_shots)
logical_error_rate_stderr = accepted_error_rate_stderr
```

Raw rate returns `math.nan` when `shots == 0`. Accepted/logical rates and their standard error return `math.nan` when `accepted_shots == 0`.

The ambiguous `error_rate` and `stderr` properties are removed. `ThresholdPoint.rate` and `ThresholdPoint.stderr` are separate threshold-analysis values and remain unchanged.

## Internal Data Flow

1. A functional wrapper creates a Collector, or user code creates one directly.
2. Collector merges its base `CollectionOptions` with each task's `collection_options`.
3. Python expands decoder fanout, computes strong ids, reads existing CSV totals, and prepares native task dictionaries.
4. The PyO3 bridge releases the GIL while the existing Rust scheduler samples and decodes.
5. Committed native deltas reacquire the GIL only for the existing aggregate-stat callback.
6. Python appends and flushes resume rows before acknowledging the delta.
7. Collector converts native totals or deltas into `TaskStats` or `Progress` according to the selected method.

No detector, correction, or mask batches cross the Python boundary.

## Error Handling

- Invalid option values fail during option construction where possible.
- A missing effective `max_shots` still fails before native collection begins.
- Python decoders continue to raise `TypeError("collection requires a native decoder")`.
- CSV identity collisions and malformed resume data retain their current errors.
- Closing or abandoning `iter_progress` cancels collection and joins native worker threads through the existing cancellation path.

## Exports And Documentation

`Collector`, `CollectionRunOptions`, and `iter_progress` are exported from both `faultscope.collection` and top-level `faultscope`. Existing collection exports remain except for API elements explicitly removed above.

Documentation and examples use Collector for reusable or production workflows and the functional wrappers for one-shot scripts.

## Testing

- API contract snapshot covers the new exports, signatures, dataclass fields, Collector methods, and removed names.
- Functional wrappers and Collector methods return equivalent results.
- `iter_collect` has a `TaskStats`-only runtime and static type.
- `iter_progress` preserves committed-delta ordering, cancellation, and per-batch resume flush.
- Task options override base sampling options; run options control workers, seed, persistence, counting, and fanout.
- Same run seed remains repeatable, while separate task strong ids receive separate deterministic streams.
- TaskStats rates cover no discards, partial discards, all discards, and zero shots.
- CLI, benchmarks, docs, stubs, mypy, unittest, pytest, and Rust workspace tests remain green.
