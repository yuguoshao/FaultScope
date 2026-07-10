"""Logical error-rate collection loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Any, Literal

from faultscope._native import _collect_dem_logical_error_stats_many
from faultscope.decoders import create_native_decoder
from faultscope.runtime import (
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
)

from faultscope.collection._types import (
    CollectionData,
    CollectionOptions,
    CollectionTask,
    Progress,
    TaskStats,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)


# Python owns task parsing, strong-id construction, and CSV resume orchestration.
# Native sampling, decoding, batch scheduling, and counting stay in Rust.
ProgressMode = Literal["final", "stream"]


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
    progress_callback: Callable[[TaskStats | Progress], object] | None = None,
    print_progress: bool = False,
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
    custom_error_count_key: str | None = None,
    progress_mode: ProgressMode = "final",
    decoders: Iterable[str | object] | str | object | None = None,
) -> Iterator[TaskStats | Progress]:
    _validate_progress_mode(progress_mode)
    if progress_mode == "stream":
        yield from _iter_collect_stream(
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
            decoders=decoders,
        )
        return
    final_stats = _run_collect(
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
        progress_mode="final",
        decoders=decoders,
    )
    yield from final_stats


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
    progress_callback: Callable[[TaskStats | Progress], object] | None = None,
    print_progress: bool = False,
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
    custom_error_count_key: str | None = None,
    progress_mode: ProgressMode = "final",
    decoders: Iterable[str | object] | str | object | None = None,
) -> list[TaskStats]:
    final_stats = _run_collect(
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
        progress_mode=progress_mode,
        decoders=decoders,
    )
    return final_stats


def _run_collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None,
    max_shots: int | None,
    max_errors: int | None,
    batch_size: int | None,
    seed: int | None,
    start_batch_size: int | None,
    max_batch_size: int | None,
    max_batch_seconds: float | None,
    num_workers: int | None,
    existing_data_filepaths: Iterable[str | Path],
    save_resume_filepath: str | Path | None,
    progress_callback: Callable[[TaskStats | Progress], object] | None,
    print_progress: bool,
    count_observable_error_combos: bool,
    count_detection_events: bool,
    custom_error_count_key: str | None,
    progress_mode: str,
    decoders: Iterable[str | object] | str | object | None,
) -> list[TaskStats]:
    _validate_progress_mode(progress_mode)
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
    task_list = _expand_tasks_for_decoders(list(tasks), decoders)
    native_tasks = []
    for index, task in enumerate(task_list):
        effective = _merge_options(call_options, task.collection_options)
        if effective.max_shots is None:
            raise ValueError("max_shots is required")
        native_tasks.append(_native_task(task, index, effective))

    existing = _read_existing_stats(existing_data_filepaths, save_resume_filepath)
    resume_path = Path(save_resume_filepath) if save_resume_filepath is not None else None
    existing_data = CollectionData(existing.values())

    def on_stream_delta(item: Mapping[str, object]) -> None:
        stat = _task_stats_from_native(item)
        if resume_path is not None:
            write_stats_to_csv_file(resume_path, [stat], append=True)
        existing_data.add_sample(stat)
        progress = Progress((stat,), _status_message(stat))
        if print_progress:
            print(progress.status_message, file=sys.stderr)
        if progress_callback is not None:
            progress_callback(progress)

    native_stats = _collect_dem_logical_error_stats_many(
        native_tasks,
        num_workers=1 if num_workers is None else num_workers,
        seed=seed,
        count_observable_error_combos=count_observable_error_combos,
        count_detection_events=count_detection_events,
        custom_error_count_key=custom_error_count_key,
        existing_stats=[_native_stats_from_task_stats(stat) for stat in existing.values()],
        progress_callback=on_stream_delta if progress_mode == "stream" else None,
    )
    final_stats = [_task_stats_from_native(item) for item in native_stats]

    if progress_mode == "final":
        for stat in final_stats:
            if resume_path is not None:
                delta = _stats_delta(stat, existing.get(stat.strong_id))
                if delta is not None:
                    write_stats_to_csv_file(resume_path, [delta], append=True)
            if print_progress:
                print(_status_message(stat), file=sys.stderr)
            if progress_callback is not None:
                progress_callback(stat)

    return final_stats


class _CollectStreamCancelled(Exception):
    pass


def _iter_collect_stream(
    tasks: Iterable[CollectionTask],
    **kwargs: object,
) -> Iterator[Progress]:
    user_callback = kwargs.pop("progress_callback")
    condition = threading.Condition()
    pending: Progress | None = None
    failure: BaseException | None = None
    done = False
    cancelled = False

    def progress_bridge(progress: TaskStats | Progress) -> None:
        nonlocal pending
        if not isinstance(progress, Progress):
            raise TypeError("stream collection expected Progress callbacks")
        if user_callback is not None:
            user_callback(progress)  # type: ignore[misc]
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
                progress_callback=progress_bridge,
                progress_mode="stream",
                **kwargs,  # type: ignore[arg-type]
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


def _validate_progress_mode(progress_mode: str) -> ProgressMode:
    if progress_mode == "final":
        return "final"
    if progress_mode == "stream":
        return "stream"
    raise ValueError('progress_mode must be "final" or "stream"')


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
        "custom_counts": {
            str(key): int(value) for key, value in stat.custom_counts.items()
        },
    }


def _stats_delta(total: TaskStats, existing: TaskStats | None) -> TaskStats | None:
    if existing is None:
        return total
    custom_counts: dict[str, int] = {}
    for key, value in total.custom_counts.items():
        delta = int(value) - int(existing.custom_counts.get(key, 0))
        if delta:
            custom_counts[key] = delta
    delta = total.with_edits(
        shots=total.shots - existing.shots,
        errors=total.errors - existing.errors,
        discards=total.discards - existing.discards,
        seconds=max(0.0, total.seconds - existing.seconds),
        custom_counts=custom_counts,
    )
    if (
        delta.shots == 0
        and delta.errors == 0
        and delta.discards == 0
        and not custom_counts
    ):
        return None
    return delta


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
    try:
        return list(decoders)  # type: ignore[arg-type]
    except TypeError:
        return [decoders]


def _status_message(stat: TaskStats) -> str:
    return (
        f"{stat.task_id}: shots={stat.shots} errors={stat.errors} "
        f"discards={stat.discards}"
    )
