"""Logical error-rate collection loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from faultscope._native import _collect_dem_logical_error_stats_many
from faultscope.decoders import create_native_decoder
from faultscope.runtime import (
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
)

from faultscope.collection._types import CollectionOptions, CollectionTask, TaskStats


# Python owns task parsing, strong-id construction, and CSV resume orchestration.
# Native sampling, decoding, batch scheduling, and counting stay in Rust.
CSV_HEADER = [
    "shots",
    "errors",
    "discards",
    "seconds",
    "decoder",
    "strong_id",
    "json_metadata",
    "custom_counts",
]


def iter_collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    max_shots: int | None = None,
    max_errors: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    start_batch_size: int | None = None,
    max_batch_size: int | None = None,
    max_batch_seconds: float | None = None,
    num_workers: int | None = None,
    existing_data_filepaths: Iterable[str | Path] = (),
    save_resume_filepath: str | Path | None = None,
    progress_callback: Callable[[TaskStats], object] | None = None,
    print_progress: bool = False,
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
    custom_error_count_key: str | None = None,
) -> Iterator[TaskStats]:
    call_options = _call_options(
        options=options,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        seed=seed,
        start_batch_size=start_batch_size,
        max_batch_size=max_batch_size,
        max_batch_seconds=max_batch_seconds,
    )
    task_list = list(tasks)
    native_tasks = []
    for index, task in enumerate(task_list):
        effective = _merge_options(call_options, task.collection_options)
        if effective.max_shots is None:
            raise ValueError("max_shots is required")
        native_tasks.append(_native_task(task, index, effective))

    existing = _read_existing_stats(existing_data_filepaths, save_resume_filepath)
    native_stats = _collect_dem_logical_error_stats_many(
        native_tasks,
        num_workers=1 if num_workers is None else num_workers,
        seed=seed,
        count_observable_error_combos=count_observable_error_combos,
        count_detection_events=count_detection_events,
        custom_error_count_key=custom_error_count_key,
        existing_stats=list(existing.values()),
    )

    resume_path = Path(save_resume_filepath) if save_resume_filepath is not None else None
    if resume_path is not None:
        _ensure_resume_header(resume_path)

    for item in native_stats:
        stat = _task_stats_from_native(item)
        if resume_path is not None:
            delta = _native_stats_delta(item, existing.get(item["strong_id"]))
            if delta is not None:
                _append_native_stats(resume_path, delta)
                existing[item["strong_id"]] = _merge_native_stats(
                    existing.get(item["strong_id"]), item
                )
        if print_progress:
            print(
                f"{stat.task_id}: shots={stat.shots} errors={stat.errors} "
                f"discards={stat.discards}",
                file=sys.stderr,
            )
        if progress_callback is not None:
            progress_callback(stat)
        yield stat


def collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    max_shots: int | None = None,
    max_errors: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    start_batch_size: int | None = None,
    max_batch_size: int | None = None,
    max_batch_seconds: float | None = None,
    num_workers: int | None = None,
    existing_data_filepaths: Iterable[str | Path] = (),
    save_resume_filepath: str | Path | None = None,
    progress_callback: Callable[[TaskStats], object] | None = None,
    print_progress: bool = False,
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
    custom_error_count_key: str | None = None,
) -> list[TaskStats]:
    return list(
        iter_collect(
            tasks,
            options=options,
            max_shots=max_shots,
            max_errors=max_errors,
            batch_size=batch_size,
            seed=seed,
            start_batch_size=start_batch_size,
            max_batch_size=max_batch_size,
            max_batch_seconds=max_batch_seconds,
            num_workers=num_workers,
            existing_data_filepaths=existing_data_filepaths,
            save_resume_filepath=save_resume_filepath,
            progress_callback=progress_callback,
            print_progress=print_progress,
            count_observable_error_combos=count_observable_error_combos,
            count_detection_events=count_detection_events,
            custom_error_count_key=custom_error_count_key,
        )
    )


def _merge_options(
    base: CollectionOptions,
    *overlays: CollectionOptions | None,
) -> CollectionOptions:
    default_batch_size = CollectionOptions().batch_size
    max_shots = base.max_shots
    max_errors = base.max_errors
    batch_size = base.batch_size
    seed = base.seed
    start_batch_size = base.start_batch_size
    max_batch_size = base.max_batch_size
    max_batch_seconds = base.max_batch_seconds
    for overlay in overlays:
        if overlay is None:
            continue
        if overlay.max_shots is not None:
            max_shots = overlay.max_shots
        if overlay.max_errors is not None:
            max_errors = overlay.max_errors
        if overlay.seed is not None:
            seed = overlay.seed
        if overlay.batch_size != default_batch_size:
            batch_size = overlay.batch_size
        if overlay.start_batch_size is not None:
            start_batch_size = overlay.start_batch_size
        if overlay.max_batch_size is not None:
            max_batch_size = overlay.max_batch_size
        if overlay.max_batch_seconds is not None:
            max_batch_seconds = overlay.max_batch_seconds
    return CollectionOptions(
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        seed=seed,
        start_batch_size=start_batch_size,
        max_batch_size=max_batch_size,
        max_batch_seconds=max_batch_seconds,
    )


def _call_options(
    *,
    options: CollectionOptions | None,
    max_shots: int | None,
    max_errors: int | None,
    batch_size: int | None,
    seed: int | None,
    start_batch_size: int | None,
    max_batch_size: int | None,
    max_batch_seconds: float | None,
) -> CollectionOptions:
    effective = _merge_options(CollectionOptions(), options)
    return CollectionOptions(
        max_shots=max_shots if max_shots is not None else effective.max_shots,
        max_errors=max_errors if max_errors is not None else effective.max_errors,
        batch_size=batch_size if batch_size is not None else effective.batch_size,
        seed=seed if seed is not None else effective.seed,
        start_batch_size=(
            start_batch_size
            if start_batch_size is not None
            else effective.start_batch_size
        ),
        max_batch_size=(
            max_batch_size if max_batch_size is not None else effective.max_batch_size
        ),
        max_batch_seconds=(
            max_batch_seconds
            if max_batch_seconds is not None
            else effective.max_batch_seconds
        ),
    )


def _native_task(
    task: CollectionTask,
    index: int,
    options: CollectionOptions,
) -> dict[str, object]:
    sampler, dem = _compile_task_sampler(task)
    decoder = _resolve_decoder(task, dem)
    decoder_name = _decoder_name(decoder if decoder is not None else task.decoder)
    metadata = dict(task.metadata or {})
    metadata_json = _canonical_json(metadata)
    task_id = task.task_id or f"task-{index}"
    postselection_mask = _bytes_or_none(task.postselection_mask)
    postselected_observables_mask = _bytes_or_none(task.postselected_observables_mask)
    strong_id = _strong_id(
        task=task,
        dem=dem,
        decoder_name=decoder_name,
        metadata=metadata,
        metadata_json=metadata_json,
        postselection_mask=postselection_mask,
        postselected_observables_mask=postselected_observables_mask,
    )
    return {
        "task_id": task_id,
        "strong_id": strong_id,
        "sampler": sampler,
        "decoder": decoder,
        "decoder_name": decoder_name,
        "metadata_json": metadata_json,
        "max_shots": int(options.max_shots),
        "max_errors": options.max_errors,
        "batch_size": options.batch_size,
        "seed": options.seed,
        "start_batch_size": options.start_batch_size,
        "max_batch_size": options.max_batch_size,
        "max_batch_seconds": options.max_batch_seconds,
        "postselection_mask": postselection_mask,
        "postselected_observables_mask": postselected_observables_mask,
    }


def _compile_task_sampler(task: CollectionTask) -> tuple[Any, Any]:
    if task.dem is not None:
        sampler = compile_native_dem_sampler(task.dem)
        return sampler, task.dem
    sampler = compile_native_dem_sampler_from_circuit(
        task.circuit,
        detectors=task.detectors,
        observables=task.observables,
        materialize_dem=True,
    )
    return sampler, sampler.dem


def _resolve_decoder(task: CollectionTask, dem: Any) -> object | None:
    if isinstance(task.decoder, str):
        return create_native_decoder(
            task.decoder,
            dem=dem,
            options=task.decoder_options,
        )
    return task.decoder


def _decoder_name(decoder: object | str | None) -> str | None:
    if decoder is None:
        return None
    if isinstance(decoder, str):
        return decoder
    name = getattr(decoder, "name", None)
    if isinstance(name, str):
        return name
    return type(decoder).__name__


def _strong_id(
    *,
    task: CollectionTask,
    dem: Any,
    decoder_name: str | None,
    metadata: Mapping[str, object],
    metadata_json: str,
    postselection_mask: bytes | None,
    postselected_observables_mask: bytes | None,
) -> str:
    payload = {
        "source_kind": "circuit" if task.circuit is not None else "dem",
        "circuit": repr(task.circuit) if task.circuit is not None else None,
        "dem": repr(dem),
        "decoder": decoder_name,
        "decoder_options": _jsonable(task.decoder_options or {}),
        "metadata": metadata,
        "metadata_json": metadata_json,
        "postselection_mask": (
            None if postselection_mask is None else postselection_mask.hex()
        ),
        "postselected_observables_mask": (
            None
            if postselected_observables_mask is None
            else postselected_observables_mask.hex()
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _jsonable(value: object) -> object:
    json.loads(_canonical_json(value))
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bytes_or_none(value: bytes | bytearray | memoryview | None) -> bytes | None:
    if value is None:
        return None
    return bytes(value)


def _task_stats_from_native(item: Mapping[str, object]) -> TaskStats:
    metadata = item.get("metadata")
    if metadata is None:
        metadata = json.loads(str(item["metadata_json"]))
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    return TaskStats(
        task_id=str(item["task_id"]),
        strong_id=str(item["strong_id"]),
        shots=int(item["shots"]),
        errors=int(item["errors"]),
        discards=int(item["discards"]),
        seconds=float(item["seconds"]),
        decoder=item["decoder"] if isinstance(item["decoder"], str) else None,
        metadata=dict(metadata),
        custom_counts=dict(item.get("custom_counts", {})),
    )


def _read_existing_stats(
    existing_data_filepaths: Iterable[str | Path],
    save_resume_filepath: str | Path | None,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    paths = [Path(path) for path in existing_data_filepaths]
    if save_resume_filepath is not None:
        resume = Path(save_resume_filepath)
        if resume.exists() and resume not in paths:
            paths.append(resume)
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_HEADER:
                raise ValueError(f"collection CSV header does not match in {path}")
            for row in reader:
                stats = _native_stats_from_csv_row(row)
                out[stats["strong_id"]] = _merge_native_stats(
                    out.get(stats["strong_id"]), stats
                )
    return out


def _native_stats_from_csv_row(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "task_id": row["strong_id"],
        "strong_id": row["strong_id"],
        "decoder": row["decoder"] or None,
        "metadata_json": row["json_metadata"],
        "shots": int(row["shots"]),
        "errors": int(row["errors"]),
        "discards": int(row["discards"]),
        "seconds": float(row["seconds"]),
        "custom_counts": json.loads(row["custom_counts"] or "{}"),
    }


def _ensure_resume_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def _append_native_stats(path: Path, stats: Mapping[str, object]) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                int(stats["shots"]),
                int(stats["errors"]),
                int(stats["discards"]),
                f"{float(stats['seconds']):.12g}",
                stats["decoder"] or "",
                stats["strong_id"],
                stats["metadata_json"],
                _canonical_json(stats.get("custom_counts", {})),
            ]
        )


def _merge_native_stats(
    left: Mapping[str, object] | None,
    right: Mapping[str, object],
) -> dict[str, object]:
    if left is None:
        return dict(right)
    if left["strong_id"] != right["strong_id"]:
        raise ValueError("cannot merge collection stats with different strong_id")
    if left["decoder"] != right["decoder"] or left["metadata_json"] != right["metadata_json"]:
        raise ValueError("collection stats strong_id collision has different identity")
    custom_counts = dict(left.get("custom_counts", {}))
    for key, value in dict(right.get("custom_counts", {})).items():
        custom_counts[key] = int(custom_counts.get(key, 0)) + int(value)
    return {
        "task_id": left.get("task_id") or right.get("task_id"),
        "strong_id": left["strong_id"],
        "decoder": left["decoder"],
        "metadata_json": left["metadata_json"],
        "shots": int(left["shots"]) + int(right["shots"]),
        "errors": int(left["errors"]) + int(right["errors"]),
        "discards": int(left["discards"]) + int(right["discards"]),
        "seconds": float(left["seconds"]) + float(right["seconds"]),
        "custom_counts": custom_counts,
    }


def _native_stats_delta(
    total: Mapping[str, object],
    existing: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if existing is None:
        return dict(total)
    custom_counts: dict[str, int] = {}
    total_counts = dict(total.get("custom_counts", {}))
    existing_counts = dict(existing.get("custom_counts", {}))
    for key, value in total_counts.items():
        delta = int(value) - int(existing_counts.get(key, 0))
        if delta:
            custom_counts[key] = delta
    delta = {
        "task_id": total["task_id"],
        "strong_id": total["strong_id"],
        "decoder": total["decoder"],
        "metadata_json": total["metadata_json"],
        "shots": int(total["shots"]) - int(existing["shots"]),
        "errors": int(total["errors"]) - int(existing["errors"]),
        "discards": int(total["discards"]) - int(existing["discards"]),
        "seconds": max(0.0, float(total["seconds"]) - float(existing["seconds"])),
        "custom_counts": custom_counts,
    }
    if (
        delta["shots"] == 0
        and delta["errors"] == 0
        and delta["discards"] == 0
        and not custom_counts
    ):
        return None
    return delta
