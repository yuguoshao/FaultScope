"""Logical error-rate collection loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from faultscope._native import _collect_dem_logical_error_stats
from faultscope.decoders import create_native_decoder
from faultscope.runtime import (
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
)

from faultscope.collection._types import CollectionOptions, CollectionTask, TaskStats


def iter_collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    max_shots: int | None = None,
    max_errors: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> Iterator[TaskStats]:
    call_options = _call_options(
        options=options,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        seed=seed,
    )
    for index, task in enumerate(tasks):
        effective = _merge_options(call_options, task.collection_options)
        if effective.max_shots is None:
            raise ValueError("max_shots is required")
        yield _collect_one(task, index, effective)


def collect(
    tasks: Iterable[CollectionTask],
    *,
    options: CollectionOptions | None = None,
    max_shots: int | None = None,
    max_errors: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> list[TaskStats]:
    return list(
        iter_collect(
            tasks,
            options=options,
            max_shots=max_shots,
            max_errors=max_errors,
            batch_size=batch_size,
            seed=seed,
        )
    )


def _merge_options(
    base: CollectionOptions,
    *overlays: CollectionOptions | None,
    batch_size_override: int | None = None,
) -> CollectionOptions:
    default_batch_size = CollectionOptions().batch_size
    max_shots = base.max_shots
    max_errors = base.max_errors
    batch_size = base.batch_size
    seed = base.seed
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
    if batch_size_override is not None:
        batch_size = batch_size_override
    return CollectionOptions(
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        seed=seed,
    )


def _call_options(
    *,
    options: CollectionOptions | None,
    max_shots: int | None,
    max_errors: int | None,
    batch_size: int | None,
    seed: int | None,
) -> CollectionOptions:
    effective = _merge_options(CollectionOptions(), options)
    return CollectionOptions(
        max_shots=max_shots if max_shots is not None else effective.max_shots,
        max_errors=max_errors if max_errors is not None else effective.max_errors,
        batch_size=batch_size if batch_size is not None else effective.batch_size,
        seed=seed if seed is not None else effective.seed,
    )


def _collect_one(
    task: CollectionTask,
    index: int,
    options: CollectionOptions,
) -> TaskStats:
    sampler, dem = _compile_task_sampler(task)
    decoder = _resolve_decoder(task, dem)
    shots_done, errors, seconds = _collect_dem_logical_error_stats(
        sampler,
        max_shots=int(options.max_shots),
        max_errors=options.max_errors,
        batch_size=options.batch_size,
        seed=options.seed,
        decoder=decoder,
    )

    return TaskStats(
        task_id=task.task_id or f"task-{index}",
        shots=int(shots_done),
        errors=int(errors),
        discards=0,
        seconds=float(seconds),
        decoder=_decoder_name(decoder if decoder is not None else task.decoder),
        metadata=dict(task.metadata or {}),
    )


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
