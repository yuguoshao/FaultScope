"""Typed collection inputs and logical error-rate statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class CollectionOptions:
    max_shots: int | None = None
    max_errors: int | None = None
    batch_size: int = 10_000
    seed: int | None = None
    start_batch_size: int | None = None
    max_batch_size: int | None = None
    max_batch_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_shots is not None and self.max_shots <= 0:
            raise ValueError("max_shots must be positive")
        if self.max_errors is not None and self.max_errors < 0:
            raise ValueError("max_errors must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.start_batch_size is not None and self.start_batch_size <= 0:
            raise ValueError("start_batch_size must be positive")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_batch_seconds is not None and self.max_batch_seconds <= 0:
            raise ValueError("max_batch_seconds must be positive")


@dataclass(frozen=True)
class CollectionTask:
    circuit: Any | None = None
    dem: Any | None = None
    detectors: tuple[Any, ...] | None = None
    observables: tuple[Any, ...] | None = None
    decoder: object | str | None = None
    decoder_options: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None
    collection_options: CollectionOptions | None = None
    task_id: str | None = None
    postselection_mask: bytes | bytearray | memoryview | None = None
    postselected_observables_mask: bytes | bytearray | memoryview | None = None

    def __post_init__(self) -> None:
        if (self.circuit is None) == (self.dem is None):
            raise ValueError("CollectionTask requires exactly one of circuit or dem")
        if self.collection_options is not None and not isinstance(
            self.collection_options, CollectionOptions
        ):
            raise TypeError("collection_options must be a CollectionOptions instance")


@dataclass(frozen=True)
class TaskStats:
    task_id: str
    shots: int
    errors: int
    discards: int
    seconds: float
    decoder: str | None
    metadata: Mapping[str, object]
    strong_id: str = ""
    custom_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return self.errors / self.shots

    @property
    def accepted_shots(self) -> int:
        return self.shots - self.discards

    @property
    def logical_error_rate(self) -> float:
        if self.accepted_shots == 0:
            return math.nan
        return self.errors / self.accepted_shots

    @property
    def stderr(self) -> float:
        if self.accepted_shots == 0:
            return math.nan
        p = self.logical_error_rate
        return math.sqrt(p * (1.0 - p) / self.accepted_shots)
