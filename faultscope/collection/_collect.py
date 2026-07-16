"""Logical error-rate collection loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from faultscope._native import _collect_dem_hotspots_many, _collect_dem_logical_error_stats_many
from faultscope.decoders import create_native_decoder
from faultscope.runtime import (
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
)

from faultscope.collection._types import (
    CollectionData,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    HotspotCollectionResult,
    Progress,
    TaskStats,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)
from faultscope.collection._identity import (
    STRONG_ID_SCHEMA_VERSION,
    canonical_json as _identity_canonical_json,
    decoder_identity_payload,
    source_identity_payload,
)


# Python owns task parsing, strong-id construction, and CSV resume orchestration.
# Native sampling, decoding, batch scheduling, and counting stay in Rust.


@dataclass(frozen=True, init=False)
class Collector:
    """Reusable immutable configuration for native logical-error collection."""

    options: CollectionOptions
    run_options: CollectionRunOptions

    def __init__(
        self,
        *,
        options: CollectionOptions | None = None,
        run_options: CollectionRunOptions | None = None,
    ) -> None:
        object.__setattr__(self, "options", options or CollectionOptions())
        object.__setattr__(self, "run_options", run_options or CollectionRunOptions())

    def collect(self, tasks: Iterable[CollectionTask]) -> list[TaskStats]:
        """Collect and return one final total per expanded task."""

        return _run_collect(tasks, self.options, self.run_options)

    def iter_collect(self, tasks: Iterable[CollectionTask]) -> Iterator[TaskStats]:
        """Yield final task totals in expanded task order."""

        yield from self.collect(tasks)

    def collect_hotspots(self, tasks: Iterable[CollectionTask]) -> list[HotspotCollectionResult]:
        """Collect logical statistics and shot-weighted edge sensitivities."""

        return _run_collect_hotspots(tasks, self.options, self.run_options)

    def iter_progress(self, tasks: Iterable[CollectionTask]) -> Iterator[Progress]:
        """Yield committed native batch deltas as progress events."""

        yield from _iter_collect_stream(tasks, self.options, self.run_options)

    def _collect_with_progress(
        self,
        tasks: Iterable[CollectionTask],
        progress_sink: Callable[[Progress], object],
    ) -> list[TaskStats]:
        return _run_collect(
            tasks,
            self.options,
            self.run_options,
            progress_sink=progress_sink,
        )


def iter_collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> Iterator[TaskStats]:
    """Collect native logical-error statistics and yield final task totals."""

    yield from Collector(options=options, run_options=run_options).iter_collect(tasks)


def collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> list[TaskStats]:
    """Collect native logical-error statistics and return one total per task."""

    return Collector(options=options, run_options=run_options).collect(tasks)


def collect_hotspots(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> list[HotspotCollectionResult]:
    """Collect logical statistics and shot-weighted edge sensitivities."""

    return Collector(options=options, run_options=run_options).collect_hotspots(tasks)


def iter_progress(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> Iterator[Progress]:
    """Collect native logical-error statistics and yield committed batch deltas."""

    yield from Collector(options=options, run_options=run_options).iter_progress(tasks)


def _run_collect(
    tasks: Iterable[CollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
    *,
    progress_sink: Callable[[Progress], object] | None = None,
) -> list[TaskStats]:
    task_list = _expand_tasks_for_decoders(list(tasks), run_options.decoders)
    native_tasks = []
    for index, task in enumerate(task_list):
        effective = _merge_options(options, task.collection_options)
        if effective.max_shots is None:
            raise ValueError("max_shots is required")
        native_tasks.append(_native_task(task, index, effective))

    existing = _read_existing_stats(
        run_options.existing_data_filepaths,
        run_options.save_resume_filepath,
    )
    resume_path = (
        Path(run_options.save_resume_filepath)
        if run_options.save_resume_filepath is not None
        else None
    )
    existing_data = CollectionData(existing.values())

    def on_stream_delta(item: Mapping[str, object]) -> None:
        stat = _task_stats_from_native(item)
        if resume_path is not None:
            write_stats_to_csv_file(resume_path, [stat], append=True)
        existing_data.add_sample(stat)
        progress = Progress((stat,), _status_message(stat))
        if progress_sink is not None:
            progress_sink(progress)

    native_stats = _collect_dem_logical_error_stats_many(
        native_tasks,
        num_workers=run_options.num_workers,
        seed=run_options.seed,
        count_observable_error_combos=run_options.count_observable_error_combos,
        count_detection_events=run_options.count_detection_events,
        custom_error_count_key=run_options.custom_error_count_key,
        existing_stats=[_native_stats_from_task_stats(stat) for stat in existing.values()],
        progress_callback=on_stream_delta if progress_sink is not None else None,
    )
    final_stats = [_task_stats_from_native(item) for item in native_stats]

    if progress_sink is None:
        for stat in final_stats:
            if resume_path is not None:
                delta = _stats_delta(stat, existing.get(stat.strong_id))
                if delta is not None:
                    write_stats_to_csv_file(resume_path, [delta], append=True)

    return final_stats


def _run_collect_hotspots(
    tasks: Iterable[CollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
) -> list[HotspotCollectionResult]:
    if run_options.existing_data_filepaths or run_options.save_resume_filepath is not None:
        raise ValueError("hotspot collection does not support CSV partial resume")
    task_list = _expand_tasks_for_decoders(list(tasks), run_options.decoders)
    native_tasks = []
    for index, task in enumerate(task_list):
        effective = _merge_options(options, task.collection_options)
        if effective.max_shots is None:
            raise ValueError("max_shots is required")
        native_tasks.append(_native_task(task, index, effective))
    native_results = _collect_dem_hotspots_many(
        native_tasks,
        num_workers=run_options.num_workers,
        seed=run_options.seed,
        count_observable_error_combos=run_options.count_observable_error_combos,
        count_detection_events=run_options.count_detection_events,
        custom_error_count_key=run_options.custom_error_count_key,
    )
    return [
        HotspotCollectionResult(
            stats=_task_stats_from_native(item["stats"]),
            batch_stats=tuple(_task_stats_from_native(stat) for stat in item["batch_stats"]),
            edge_sensitivities=tuple(float(value) for value in item["edge_sensitivities"]),
        )
        for item in native_results
    ]


class _CollectStreamCancelled(Exception):
    pass


def _iter_collect_stream(
    tasks: Iterable[CollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
) -> Iterator[Progress]:
    condition = threading.Condition()
    pending: Progress | None = None
    failure: BaseException | None = None
    done = False
    cancelled = False

    def progress_bridge(progress: Progress) -> None:
        nonlocal pending
        with condition:
            if cancelled:
                raise _CollectStreamCancelled
            while pending is not None and not cancelled:
                condition.wait()
            if cancelled:
                raise _CollectStreamCancelled
            pending = progress
            condition.notify_all()
            while pending is progress and not cancelled:
                condition.wait()
            if cancelled:
                raise _CollectStreamCancelled

    def worker() -> None:
        nonlocal failure, done
        try:
            _run_collect(
                tasks,
                options,
                run_options,
                progress_sink=progress_bridge,
            )
        except _CollectStreamCancelled:
            with condition:
                done = True
                condition.notify_all()
        except BaseException as exc:  # Propagate callback and native errors.
            with condition:
                if cancelled:
                    done = True
                else:
                    failure = exc
                condition.notify_all()
        else:
            with condition:
                done = True
                condition.notify_all()

    thread = threading.Thread(target=worker, name="faultscope-collect-stream")
    thread.start()
    try:
        while True:
            with condition:
                while pending is None and failure is None and not done:
                    condition.wait()
                if pending is not None:
                    progress = pending
                elif failure is not None:
                    raise failure
                else:
                    return
            try:
                yield progress
            except GeneratorExit:
                with condition:
                    cancelled = True
                    if pending is progress:
                        pending = None
                    condition.notify_all()
                raise
            finally:
                with condition:
                    if not cancelled and pending is progress:
                        pending = None
                    condition.notify_all()
    finally:
        with condition:
            cancelled = True
            pending = None
            condition.notify_all()
        thread.join()


def _merge_options(
    base: CollectionOptions,
    *overlays: CollectionOptions | None,
) -> CollectionOptions:
    default_batch_size = CollectionOptions().batch_size
    max_shots = base.max_shots
    min_shots = base.min_shots
    max_errors = base.max_errors
    batch_size = base.batch_size
    start_batch_size = base.start_batch_size
    max_batch_size = base.max_batch_size
    max_batch_seconds = base.max_batch_seconds
    for overlay in overlays:
        if overlay is None:
            continue
        if overlay.max_shots is not None:
            max_shots = overlay.max_shots
        if getattr(overlay, "_min_shots_explicit", False):
            min_shots = overlay.min_shots
        if overlay.max_errors is not None:
            max_errors = overlay.max_errors
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
        min_shots=min_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        start_batch_size=start_batch_size,
        max_batch_size=max_batch_size,
        max_batch_seconds=max_batch_seconds,
    )


def _native_task(
    task: CollectionTask,
    index: int,
    options: CollectionOptions,
) -> dict[str, object]:
    if options.max_shots is None:
        raise ValueError("max_shots is required")
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
        decoder=decoder,
        decoder_name=decoder_name,
        metadata=metadata,
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
        "min_shots": options.min_shots,
        "max_errors": options.max_errors,
        "batch_size": options.batch_size,
        "seed": None,
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
    circuit = task.circuit
    if circuit is None:
        raise ValueError("collection task requires a circuit or DEM")
    sampler = compile_native_dem_sampler_from_circuit(
        circuit,
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
    decoder: object | None,
    decoder_name: str | None,
    metadata: Mapping[str, object],
    postselection_mask: bytes | None,
    postselected_observables_mask: bytes | None,
) -> str:
    payload = {
        "schema": "faultscope.collection.strong_id",
        "schema_version": STRONG_ID_SCHEMA_VERSION,
        "source": source_identity_payload(circuit=task.circuit, dem=dem),
        "decoder": decoder_identity_payload(decoder, decoder_name=decoder_name),
        "metadata": dict(metadata),
        "postselection_mask": (None if postselection_mask is None else postselection_mask.hex()),
        "postselected_observables_mask": (
            None if postselected_observables_mask is None else postselected_observables_mask.hex()
        ),
    }
    return hashlib.sha256(_identity_canonical_json(payload).encode("utf-8")).hexdigest()


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
    raw_custom_counts = item.get("custom_counts", {})
    if not isinstance(raw_custom_counts, Mapping):
        raise TypeError("native custom_counts must be a mapping")
    custom_counts = {str(key): int(str(value)) for key, value in raw_custom_counts.items()}
    return TaskStats(
        task_id=str(item["task_id"]),
        strong_id=str(item["strong_id"]),
        shots=int(str(item["shots"])),
        errors=int(str(item["errors"])),
        discards=int(str(item["discards"])),
        seconds=float(str(item["seconds"])),
        decoder=item["decoder"] if isinstance(item["decoder"], str) else None,
        metadata=dict(metadata),
        custom_counts=custom_counts,
    )


def _read_existing_stats(
    existing_data_filepaths: Iterable[str | Path],
    save_resume_filepath: str | Path | None,
) -> dict[str, TaskStats]:
    paths = [Path(path) for path in existing_data_filepaths]
    if save_resume_filepath is not None:
        resume = Path(save_resume_filepath)
        if resume.exists() and resume not in paths:
            paths.append(resume)
    return {stat.strong_id: stat for stat in read_stats_from_csv_files(paths)}


def _native_stats_from_task_stats(stat: TaskStats) -> dict[str, object]:
    return {
        "task_id": stat.task_id,
        "strong_id": stat.strong_id,
        "decoder": stat.decoder,
        "metadata_json": _canonical_json(dict(stat.metadata)),
        "shots": int(stat.shots),
        "errors": int(stat.errors),
        "discards": int(stat.discards),
        "seconds": float(stat.seconds),
        "custom_counts": {str(key): int(value) for key, value in stat.custom_counts.items()},
    }


def _stats_delta(total: TaskStats, existing: TaskStats | None) -> TaskStats | None:
    if existing is None:
        return total
    custom_counts: dict[str, int] = {}
    for key, value in total.custom_counts.items():
        count_delta = int(value) - int(existing.custom_counts.get(key, 0))
        if count_delta:
            custom_counts[key] = count_delta
    stats_delta = total.with_edits(
        shots=total.shots - existing.shots,
        errors=total.errors - existing.errors,
        discards=total.discards - existing.discards,
        seconds=max(0.0, total.seconds - existing.seconds),
        custom_counts=custom_counts,
    )
    if (
        stats_delta.shots == 0
        and stats_delta.errors == 0
        and stats_delta.discards == 0
        and not custom_counts
    ):
        return None
    return stats_delta


def _expand_tasks_for_decoders(
    tasks: list[CollectionTask],
    decoders: Iterable[str | object] | str | object | None,
) -> list[CollectionTask]:
    if decoders is None:
        return tasks
    decoder_list = _normalize_decoders(decoders)
    if not decoder_list:
        return tasks
    out: list[CollectionTask] = []
    for task_index, task in enumerate(tasks):
        if task.decoder is not None:
            out.append(task)
            continue
        base_id = task.task_id or f"task-{task_index}"
        for decoder_index, decoder in enumerate(decoder_list):
            task_id = base_id
            if len(decoder_list) > 1:
                name = _decoder_name(decoder) or f"decoder-{decoder_index}"
                task_id = f"{base_id}:{name}"
            out.append(replace(task, decoder=decoder, task_id=task_id))
    return out


def _normalize_decoders(
    decoders: Iterable[str | object] | str | object,
) -> list[str | object]:
    if isinstance(decoders, str):
        return [decoders]
    if isinstance(decoders, Iterable):
        return list(decoders)
    return [decoders]


def _status_message(stat: TaskStats) -> str:
    return f"{stat.task_id}: shots={stat.shots} errors={stat.errors} discards={stat.discards}"
