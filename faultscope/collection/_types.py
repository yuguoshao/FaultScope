"""Typed collection inputs and logical error-rate statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class CollectionOptions:
    max_shots: int | None = None
    max_errors: int | None = None
    batch_size: int = 10_000
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_shots is not None and self.max_shots <= 0:
            raise ValueError("max_shots must be positive")
        if self.max_errors is not None and self.max_errors < 0:
            raise ValueError("max_errors must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


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

    @property
    def error_rate(self) -> float:
        return self.errors / self.shots

    @property
    def logical_error_rate(self) -> float:
        return self.error_rate

    @property
    def stderr(self) -> float:
        p = self.error_rate
        return math.sqrt(p * (1.0 - p) / self.shots)
