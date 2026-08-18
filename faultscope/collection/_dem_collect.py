"""Legacy detector-error-model logical-error collection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
import threading
from pathlib import Path
from typing import Any

from faultscope._native import _collect_dem_hotspots_many, _collect_dem_logical_error_stats_many
from faultscope.collection._collect import (
    _bytes_or_none,
    _canonical_json,
    _counter_schema_from_run_options,
    _decoder_name,
    _merge_options,
    _normalize_decoders,
    _native_stats_from_task_stats,
    _read_existing_stats,
    _stats_delta,
    _status_message,
    _task_stats_from_native,
)
from faultscope.collection._dem_types import DemCollectionTask, DemHotspotCollectionResult
from faultscope.collection._identity import (
    SAMPLING_ID_SCHEMA_VERSION,
    STRONG_ID_SCHEMA_VERSION,
    decoder_identity_digest,
    decoder_identity_payload,
    domain_digest,
    source_identity_digest,
    source_identity_payload,
)
from faultscope.collection._types import (
    _CollectionCsvAppender,
    CollectionCounterSchema,
    CollectionOptions,
    CollectionRunOptions,
    Progress,
    TaskStats,
)
from faultscope.decoders import create_native_decoder
from faultscope.runtime import compile_native_dem_sampler


@dataclass(frozen=True)
class _PreparedDemSource:
    sampler: Any
    dem: Any
    source_digest: str


class _DemPreparationContext:
    def __init__(self) -> None:
        self._sources: dict[int, list[tuple[object, _PreparedDemSource]]] = {}

    def prepare_source(self, task: DemCollectionTask) -> _PreparedDemSource:
        key = id(task.dem)
        entries = self._sources.setdefault(key, [])
        for held_dem, source in entries:
            if task.dem is held_dem:
                return source
        source = _PreparedDemSource(
            sampler=_compile_task_sampler(task),
            dem=task.dem,
            source_digest=source_identity_digest(
                source_identity_payload(circuit=None, dem=task.dem)
            ),
        )
        entries.append((task.dem, source))
        return source


@dataclass(frozen=True, init=False)
class DemCollector:
    """Reusable immutable configuration for legacy DEM collection."""

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

    def collect(self, tasks: Iterable[DemCollectionTask]) -> list[TaskStats]:
        return _run_collect(tasks, self.options, self.run_options)

    def iter_collect(self, tasks: Iterable[DemCollectionTask]) -> Iterator[TaskStats]:
        yield from self.collect(tasks)

    def collect_hotspots(
        self,
        tasks: Iterable[DemCollectionTask],
    ) -> list[DemHotspotCollectionResult]:
        return _run_collect_hotspots(tasks, self.options, self.run_options)

    def iter_progress(self, tasks: Iterable[DemCollectionTask]) -> Iterator[Progress]:
        yield from _iter_collect_stream(tasks, self.options, self.run_options)

    def _collect_with_progress(
        self,
        tasks: Iterable[DemCollectionTask],
        progress_sink: Callable[[Progress], object],
    ) -> list[TaskStats]:
        return _run_collect(
            tasks,
            self.options,
            self.run_options,
            progress_sink=progress_sink,
        )


def collect(
    tasks: Iterable[DemCollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> list[TaskStats]:
    return DemCollector(options=options, run_options=run_options).collect(tasks)


def iter_collect(
    tasks: Iterable[DemCollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> Iterator[TaskStats]:
    yield from DemCollector(options=options, run_options=run_options).iter_collect(tasks)


def collect_hotspots(
    tasks: Iterable[DemCollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> list[DemHotspotCollectionResult]:
    return DemCollector(options=options, run_options=run_options).collect_hotspots(tasks)


def iter_progress(
    tasks: Iterable[DemCollectionTask],
    *,
    options: CollectionOptions | None = None,
    run_options: CollectionRunOptions | None = None,
) -> Iterator[Progress]:
    yield from DemCollector(options=options, run_options=run_options).iter_progress(tasks)


def _run_collect(
    tasks: Iterable[DemCollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
    *,
    progress_sink: Callable[[Progress], object] | None = None,
) -> list[TaskStats]:
    return _run_collect_materialized(
        list(tasks),
        options,
        run_options,
        progress_sink=progress_sink,
    )


def _run_collect_materialized(
    tasks: list[DemCollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
    *,
    progress_sink: Callable[[Progress], object] | None = None,
) -> list[TaskStats]:
    counter_schema = _counter_schema_from_run_options(run_options)
    native_tasks = _prepare_native_tasks(
        tasks,
        options,
        run_options.decoders,
        counter_schema,
    )
    existing = _read_existing_stats(
        run_options.existing_data_filepaths,
        run_options.save_resume_filepath,
    )
    current_strong_ids = {str(task["strong_id"]) for task in native_tasks}
    existing = {
        strong_id: stat for strong_id, stat in existing.items() if strong_id in current_strong_ids
    }
    for stat in existing.values():
        if stat.counter_schema != counter_schema:
            raise ValueError(
                "existing stats strong_id matched but counter schema differs or is missing"
            )
    resume_path = (
        Path(run_options.save_resume_filepath)
        if run_options.save_resume_filepath is not None
        else None
    )
    resume_writer: _CollectionCsvAppender | None = None

    def append_resume(stat: TaskStats) -> None:
        nonlocal resume_writer
        if resume_path is None:
            return
        if resume_writer is None:
            resume_writer = _CollectionCsvAppender(resume_path)
        resume_writer.write((stat,))

    def on_stream_delta(item: Mapping[str, object]) -> None:
        stat = _task_stats_from_native(item)
        append_resume(stat)
        if progress_sink is not None:
            progress_sink(Progress((stat,), _status_message(stat)))

    try:
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
                delta = _stats_delta(stat, existing.get(stat.strong_id))
                if delta is not None:
                    append_resume(delta)
        return final_stats
    finally:
        if resume_writer is not None:
            resume_writer.close()


def _run_collect_hotspots(
    tasks: Iterable[DemCollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
) -> list[DemHotspotCollectionResult]:
    if run_options.existing_data_filepaths or run_options.save_resume_filepath is not None:
        raise ValueError("hotspot collection does not support CSV partial resume")
    native_tasks = _prepare_native_tasks(
        list(tasks),
        options,
        run_options.decoders,
        _counter_schema_from_run_options(run_options),
    )
    native_results = _collect_dem_hotspots_many(
        native_tasks,
        num_workers=run_options.num_workers,
        seed=run_options.seed,
        count_observable_error_combos=run_options.count_observable_error_combos,
        count_detection_events=run_options.count_detection_events,
        custom_error_count_key=run_options.custom_error_count_key,
    )
    return [
        DemHotspotCollectionResult(
            stats=_task_stats_from_native(item["stats"]),
            batch_stats=tuple(_task_stats_from_native(stat) for stat in item["batch_stats"]),
            edge_sensitivities=tuple(float(value) for value in item["edge_sensitivities"]),
        )
        for item in native_results
    ]


class _CollectStreamCancelled(Exception):
    pass


def _iter_collect_stream(
    tasks: Iterable[DemCollectionTask],
    options: CollectionOptions,
    run_options: CollectionRunOptions,
) -> Iterator[Progress]:
    task_list = list(tasks)
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
            _run_collect_materialized(
                task_list,
                options,
                run_options,
                progress_sink=progress_bridge,
            )
        except _CollectStreamCancelled:
            with condition:
                done = True
                condition.notify_all()
        except BaseException as exc:
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

    thread = threading.Thread(target=worker, name="faultscope-dem-collect-stream")
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


def _prepare_native_tasks(
    tasks: list[DemCollectionTask],
    options: CollectionOptions,
    decoders: Iterable[str | object] | str | object | None,
    counter_schema: CollectionCounterSchema,
) -> list[dict[str, object]]:
    if any(not isinstance(task, DemCollectionTask) for task in tasks):
        raise TypeError("faultscope.collection.dem accepts only DemCollectionTask(dem=...) objects")
    context = _DemPreparationContext()
    task_list = _expand_dem_tasks_for_decoders(tasks, decoders)
    native_tasks: list[dict[str, object]] = []
    for index, task in enumerate(task_list):
        effective = _merge_options(options, task.collection_options)
        if effective.max_shots is None:
            raise ValueError("max_shots is required")
        native_tasks.append(
            _native_task(
                task,
                index,
                effective,
                counter_schema,
                source=context.prepare_source(task),
            )
        )
    return native_tasks


def _expand_dem_tasks_for_decoders(
    tasks: list[DemCollectionTask],
    decoders: Iterable[str | object] | str | object | None,
) -> list[DemCollectionTask]:
    if decoders is None:
        return tasks
    decoder_list = _normalize_decoders(decoders)
    if not decoder_list:
        return tasks
    out: list[DemCollectionTask] = []
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


def _compile_task_sampler(task: DemCollectionTask) -> Any:
    return compile_native_dem_sampler(task.dem)


def _native_task(
    task: DemCollectionTask,
    index: int,
    options: CollectionOptions,
    counter_schema: CollectionCounterSchema,
    *,
    source: _PreparedDemSource | None = None,
) -> dict[str, object]:
    if options.max_shots is None:
        raise ValueError("max_shots is required")
    if source is None:
        source = _PreparedDemSource(
            _compile_task_sampler(task),
            task.dem,
            source_identity_digest(source_identity_payload(circuit=None, dem=task.dem)),
        )
    decoder = (
        create_native_decoder(task.decoder, dem=source.dem, options=task.decoder_options)
        if isinstance(task.decoder, str)
        else task.decoder
    )
    decoder_name = _decoder_name(decoder if decoder is not None else task.decoder)
    metadata = dict(task.metadata or {})
    postselection_mask = _bytes_or_none(task.postselection_mask)
    postselected_observables_mask = _bytes_or_none(task.postselected_observables_mask)
    sampling_id, strong_id = _task_identities(
        source_digest=source.source_digest,
        decoder=decoder,
        decoder_name=decoder_name,
        metadata=metadata,
        postselection_mask=postselection_mask,
        postselected_observables_mask=postselected_observables_mask,
        counter_schema=counter_schema,
    )
    return {
        "task_id": task.task_id or f"task-{index}",
        "strong_id": strong_id,
        "sampling_id": sampling_id,
        "sampler": source.sampler,
        "decoder": decoder,
        "decoder_name": decoder_name,
        "metadata_json": _canonical_json(metadata),
        "max_shots": options.max_shots,
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


def _task_identities(
    *,
    source_digest: str,
    decoder: object | None,
    decoder_name: str | None,
    metadata: Mapping[str, object],
    postselection_mask: bytes | None,
    postselected_observables_mask: bytes | None,
    counter_schema: CollectionCounterSchema,
) -> tuple[str, str]:
    decoder_digest = decoder_identity_digest(
        decoder_identity_payload(decoder, decoder_name=decoder_name)
    )
    sampling_id = domain_digest(
        schema="faultscope.collection.sampling_id",
        schema_version=SAMPLING_ID_SCHEMA_VERSION,
        payload={
            "source_digest": source_digest,
            "decoder_digest": decoder_digest,
            "metadata": dict(metadata),
            "postselection_mask": (
                None if postselection_mask is None else postselection_mask.hex()
            ),
            "postselected_observables_mask": (
                None
                if postselected_observables_mask is None
                else postselected_observables_mask.hex()
            ),
        },
    )
    strong_id = domain_digest(
        schema="faultscope.collection.strong_id",
        schema_version=STRONG_ID_SCHEMA_VERSION,
        payload={
            "sampling_id": sampling_id,
            "counter_schema": counter_schema._to_payload(),
        },
    )
    return sampling_id, strong_id


__all__ = [
    "DemCollector",
    "collect",
    "collect_hotspots",
    "iter_collect",
    "iter_progress",
]
