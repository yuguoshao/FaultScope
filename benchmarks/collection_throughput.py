"""FaultScope logical collection throughput benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/collection_throughput.py --shots 10000 --batch-size 1000 --workers 1 2

The benchmark exercises faultscope.collection with fixed-size batches. It
reports single-task throughput, which should use batch-granular Rust workers
when num_workers > 1, and multi-task throughput through the same global worker
pool.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faultscope import DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.collection import CollectionTask, collect


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _validate_positive("shots", args.shots)
    _validate_positive("batch-size", args.batch_size)
    _validate_positive("small-batch-size", args.small_batch_size)
    _validate_positive("tasks", args.tasks)
    _validate_positive("repeats", args.repeats)
    for workers in args.workers:
        _validate_positive("workers", workers)
    if not 0.0 <= args.probability <= 1.0:
        raise SystemExit("probability must be between 0 and 1")

    print(
        "mode\ttasks\tworkers\tshots\tbatch_size\tseconds\tshots_per_second\t"
        "speedup\terrors\tdiscards\tstatus",
        flush=True,
    )
    scenarios = (
        ("single", 1, args.batch_size),
        ("multi", args.tasks, args.batch_size),
        ("multi-small", args.tasks, args.small_batch_size),
    )
    for mode, task_count, batch_size in scenarios:
        _run_worker_sweep(
            mode=mode,
            task_count=task_count,
            workers=args.workers,
            shots=args.shots,
            batch_size=batch_size,
            probability=args.probability,
            seed=args.seed,
            repeats=args.repeats,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--small-batch-size", type=int, default=1_000)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--probability", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def _run_worker_sweep(
    *,
    mode: str,
    task_count: int,
    workers: list[int],
    shots: int,
    batch_size: int,
    probability: float,
    seed: int,
    repeats: int,
) -> None:
    baseline_seconds: float | None = None
    for worker_count in workers:
        seconds, errors, discards = _median_collection_time(
            task_count=task_count,
            workers=worker_count,
            shots=shots,
            batch_size=batch_size,
            probability=probability,
            seed=seed,
            repeats=repeats,
        )
        if baseline_seconds is None:
            baseline_seconds = seconds
        total_shots = shots * task_count
        shots_per_second = total_shots / seconds if seconds > 0 else float("inf")
        speedup = baseline_seconds / seconds if seconds > 0 else float("inf")
        print(
            f"{mode}\t{task_count}\t{worker_count}\t{total_shots}\t"
            f"{batch_size}\t{seconds:.6f}\t{shots_per_second:.3f}\t"
            f"{speedup:.3f}\t{errors}\t{discards}\tok",
            flush=True,
        )


def _median_collection_time(
    *,
    task_count: int,
    workers: int,
    shots: int,
    batch_size: int,
    probability: float,
    seed: int,
    repeats: int,
) -> tuple[float, int, int]:
    timings: list[float] = []
    last_errors = 0
    last_discards = 0
    for repeat in range(repeats):
        tasks = [
            CollectionTask(
                dem=_logical_edge_dem(probability),
                task_id=f"task-{index}",
                metadata={"index": index},
            )
            for index in range(task_count)
        ]
        started = time.perf_counter()
        stats = collect(
            tasks,
            max_shots=shots,
            batch_size=batch_size,
            seed=seed + repeat,
            num_workers=workers,
        )
        timings.append(time.perf_counter() - started)
        last_errors = sum(stat.errors for stat in stats)
        last_discards = sum(stat.discards for stat in stats)
    return statistics.median(timings), last_errors, last_discards


def _logical_edge_dem(probability: float) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=probability,
                detectors=(),
                observables=(0,),
                location_id="logical_edge",
                event="L",
            ),
        ),
    )


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"{name} must be positive; got {value}")


if __name__ == "__main__":
    main()
