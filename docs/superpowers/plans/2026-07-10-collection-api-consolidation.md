# Collection API Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the eighteen-argument collection API with immutable `Collector` configuration and expose denominator-specific `TaskStats` rate properties before the first release.

**Architecture:** Python continues to own task preparation, strong ids, CSV resume, and aggregate conversion. A stateless `Collector` stores `CollectionOptions` and `CollectionRunOptions`, while the existing PyO3 bridge and Rust scheduler remain unchanged. Functional APIs become thin Collector wrappers, and streaming uses a dedicated `iter_progress` method.

**Tech Stack:** Python 3.10 dataclasses and typing, PyO3 native bridge, Rust `faultscope-collection`, unittest/pytest, mypy, Ruff.

## Global Constraints

- Do not modify the Rust collection scheduler, PyO3 bridge signature, CSV header, strong-id algorithm, or counting semantics.
- `CollectionOptions` contains sampling/stopping fields and no seed.
- `CollectionRunOptions` contains run seed, workers, persistence, custom counting, and decoder fanout.
- Remove public `progress_mode`, `progress_callback`, `print_progress`, `TaskStats.error_rate`, and `TaskStats.stderr` before release.
- `iter_collect` yields only `TaskStats`; `iter_progress` yields only `Progress`.
- Resume CSV deltas must be flushed before streaming progress is yielded.
- Keep all detector, correction, and mask batches inside native code.
- Preserve unrelated release-stabilization changes already present in the dirty worktree.

---

### Task 1: Option And Statistics Contracts

**Files:**
- Modify: `faultscope/collection/_types.py`
- Modify: `tests/test_collection.py`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Produces: `CollectionRunOptions`, revised `CollectionOptions`, and explicit `TaskStats` rate properties.
- Consumes: Existing `CollectionTask`, `TaskStats`, CSV utilities, and immutable dataclass conventions.

- [x] **Step 1: Write failing option and rate tests**

Add tests that assert:

```python
options = CollectionRunOptions(seed=5, num_workers=2, decoders=("mwpm",))
assert options.seed == 5
assert options.num_workers == 2
assert options.decoders == ("mwpm",)
with pytest.raises(ValueError, match="num_workers must be positive"):
    CollectionRunOptions(num_workers=0)
assert "seed" not in {field.name for field in dataclasses.fields(CollectionOptions)}

stats = TaskStats(
    task_id="rates",
    strong_id="rates",
    shots=100,
    errors=10,
    discards=20,
    seconds=1.0,
    decoder=None,
    metadata={},
)
assert stats.raw_error_rate == 0.1
assert stats.accepted_error_rate == 0.125
assert stats.logical_error_rate == 0.125
assert stats.accepted_error_rate_stderr == math.sqrt(0.125 * 0.875 / 80)
assert stats.logical_error_rate_stderr == stats.accepted_error_rate_stderr
assert not hasattr(stats, "error_rate")
assert not hasattr(stats, "stderr")
```

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_collection.CollectionTests.test_collection_run_options_validate_workers tests.test_collection.CollectionTests.test_task_stats_rates_have_explicit_denominators -v
```

Expected: import/property failures because the new API does not exist.

- [x] **Step 3: Implement immutable option and rate types**

Implement:

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

    def __post_init__(self) -> None:
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
```

Remove `seed` from `CollectionOptions`. Replace `TaskStats.error_rate` and `TaskStats.stderr` with the properties approved in the design, preserving `math.nan` for zero denominators.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

---

### Task 2: Collector And Thin Functional API

**Files:**
- Modify: `faultscope/collection/_collect.py`
- Modify: `tests/test_collection.py`

**Interfaces:**
- Consumes: `CollectionOptions`, `CollectionRunOptions`, `CollectionTask`, `TaskStats`, `Progress`.
- Produces: `Collector.collect`, `Collector.iter_collect`, `Collector.iter_progress`, and matching module functions.

- [x] **Step 1: Write failing Collector contract tests**

Add tests equivalent to:

```python
collector = Collector(
    options=CollectionOptions(max_shots=8, batch_size=2),
    run_options=CollectionRunOptions(seed=7, num_workers=2),
)
direct = collector.collect([CollectionTask(dem=_logical_edge_dem())])
wrapped = collect(
    [CollectionTask(dem=_logical_edge_dem())],
    options=CollectionOptions(max_shots=8, batch_size=2),
    run_options=CollectionRunOptions(seed=7, num_workers=2),
)
assert direct == wrapped
assert all(isinstance(item, TaskStats) for item in collector.iter_collect(tasks))
assert all(isinstance(item, Progress) for item in collector.iter_progress(tasks))
assert tuple(inspect.signature(collect).parameters) == ("tasks", "options", "run_options")
```

Also assert old keyword arguments raise `TypeError` and `iter_collect` has return annotation `Iterator[TaskStats]`.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_collection.CollectionTests.test_collector_matches_functional_wrappers tests.test_collection.CollectionTests.test_collection_functions_have_consolidated_signatures -v
```

Expected: `Collector`/`iter_progress` import failures or old signatures.

- [x] **Step 3: Refactor collection orchestration**

Implement a stateless Collector:

```python
class Collector:
    def __init__(self, *, options=None, run_options=None) -> None:
        self.options = options or CollectionOptions()
        self.run_options = run_options or CollectionRunOptions()

    def collect(self, tasks):
        return _run_collect(tasks, self.options, self.run_options, stream=False)

    def iter_collect(self, tasks):
        yield from self.collect(tasks)

    def iter_progress(self, tasks):
        yield from _iter_collect_stream(tasks, self.options, self.run_options)
```

Use final annotated signatures from the design. Keep the existing condition-variable cancellation handshake for `iter_progress`. Refactor `_run_collect` to read workers, seed, persistence, counting, and fanout exclusively from `CollectionRunOptions`. Remove public callback, print, and progress-mode branches while preserving private native callbacks needed for streaming and CSV flush.

Functional wrappers construct a Collector and delegate directly.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

---

### Task 3: Streaming, Resume, Seed, Fanout, And CLI Migration

**Files:**
- Modify: `tests/test_collection.py`
- Modify: `faultscope/collection/__main__.py`
- Modify: `benchmarks/collection_throughput.py`
- Modify: `scripts/wheel_smoke.py`

**Interfaces:**
- Consumes: Consolidated `collect`, `Collector.iter_progress`, `CollectionRunOptions`.
- Produces: Migrated CLI, benchmark, smoke test, and regression coverage.

- [x] **Step 1: Convert existing behavior tests to the new API**

Replace calls such as:

```python
collect(tasks, max_shots=256, batch_size=64, seed=123, num_workers=4)
```

with:

```python
collect(
    tasks,
    options=CollectionOptions(max_shots=256, batch_size=64),
    run_options=CollectionRunOptions(seed=123, num_workers=4),
)
```

Replace stream-mode callback tests with direct `iter_progress` iteration and explicit iterator close/cancellation tests. Preserve assertions for committed order, resume rows, interruption recovery, decoder fanout order, custom stop counters, adaptive batches, and worker determinism.

- [x] **Step 2: Run collection tests and verify RED migration failures**

Run:

```bash
.venv/bin/python -m unittest tests.test_collection -q
```

Expected: failures in CLI, benchmark-facing helpers, or remaining old API call sites.

- [x] **Step 3: Migrate production consumers**

Update CLI collect construction to create `CollectionOptions` and `CollectionRunOptions`. When progress printing is requested, use the package-private progress sink shared with `Collector.iter_progress` so the CLI receives live deltas and the final totals from one native run; otherwise call `Collector.collect`. Update benchmark and wheel smoke calls to use the two option objects. Do not change CLI flags or benchmark arguments.

- [x] **Step 4: Run collection and benchmark tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_collection tests.test_benchmarks -q
```

Expected: PASS, with only existing environment-dependent skips.

---

### Task 4: Exports, Contract Snapshot, Typing, And Documentation

**Files:**
- Modify: `faultscope/collection/__init__.py`
- Modify: `faultscope/__init__.py`
- Modify: `tests/api_contract_v0_1.json`
- Modify: `tests/test_release_contract.py`
- Modify: `docs/user_guide.md`
- Modify: `docs/api_reference.md`
- Modify: `README.md`
- Regenerate: `faultscope/_native.pyi` only if native runtime discovery changes

**Interfaces:**
- Produces: Public exports for `Collector`, `CollectionRunOptions`, and `iter_progress`.
- Removes: Public old kwargs and ambiguous TaskStats property names.

- [x] **Step 1: Write/update release contract expectations first**

Update snapshots and signature assertions to require:

```python
assert faultscope.Collector is faultscope.collection.Collector
assert faultscope.CollectionRunOptions is faultscope.collection.CollectionRunOptions
assert faultscope.iter_progress is faultscope.collection.iter_progress
assert tuple(inspect.signature(iter_progress).parameters) == (
    "tasks", "options", "run_options"
)
```

Update dataclass fields and TaskStats public method/property snapshots to match the approved design.

- [x] **Step 2: Run release contracts and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_release_contract -q
```

Expected: export and snapshot mismatches.

- [x] **Step 3: Update exports and documentation**

Export the three new names from `faultscope.collection` and top-level `faultscope`. Rewrite examples around Collector and consolidated wrappers. Remove documentation for `progress_mode`, callbacks, `print_progress`, `error_rate`, and ambiguous `stderr`; retain threshold-analysis `ThresholdPoint.stderr` documentation.

- [x] **Step 4: Run contracts, type checks, and formatting**

Run:

```bash
.venv/bin/python -m unittest tests.test_release_contract -q
.venv/bin/mypy
.venv/bin/ruff check faultscope tests benchmarks scripts
.venv/bin/ruff format --check faultscope tests benchmarks scripts
```

Expected: all commands PASS.

---

### Task 5: Full Verification

**Files:**
- Verify only; fix failures in the owning task's files.

**Interfaces:**
- Consumes: Complete consolidated Python API.
- Produces: Release-ready verified workspace.

- [x] **Step 1: Rebuild the editable ABI3 extension**

```bash
env -u CONDA_PREFIX \
  VIRTUAL_ENV=/Users/yuguo/Documents/project/FaultScope/.venv \
  PATH=/Users/yuguo/Documents/project/FaultScope/.venv/bin:/Users/yuguo/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin \
  .venv/bin/maturin develop
```

Expected: `faultscope-0.1.0` installs successfully with `abi3-py310`.

- [x] **Step 2: Run full Python and Rust verification**

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m pytest -q
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

Expected: all commands PASS, apart from documented environment-dependent test skips.

- [x] **Step 3: Run API and artifact smoke checks**

```bash
.venv/bin/python scripts/generate_native_stub.py --check
.venv/bin/python -m mypy.stubtest faultscope._native --ignore-missing-stub
.venv/bin/python benchmarks/collection_throughput.py \
  --shots 10000 --batch-size 1000 --workers 1 2 4 \
  --json-out /tmp/faultscope-collection-throughput.json
```

Expected: stub checks pass and benchmark emits valid JSON with fixed and adaptive scenarios.
