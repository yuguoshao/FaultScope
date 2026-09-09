"""Legacy detector-error-model collection inputs and results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from faultscope.collection._types import CollectionOptions, TaskStats, _validate_seed


@dataclass(frozen=True)
class DemCollectionTask:
    """A legacy collection task sampled directly from a detector error model."""

    dem: Any
    decoder: object | str | None = None
    decoder_options: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None
    collection_options: CollectionOptions | None = None
    task_id: str | None = None
    postselection_mask: bytes | bytearray | memoryview | None = None
    postselected_observables_mask: bytes | bytearray | memoryview | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.dem is None:
            raise ValueError("DemCollectionTask requires dem")
        _validate_seed(self.seed)
        if self.collection_options is not None and not isinstance(
            self.collection_options, CollectionOptions
        ):
            raise TypeError("collection_options must be a CollectionOptions instance")


@dataclass(frozen=True)
class DemHotspotCollectionResult:
    stats: TaskStats
    batch_stats: tuple[TaskStats, ...]
    edge_sensitivities: tuple[float, ...]


__all__ = ["DemCollectionTask", "DemHotspotCollectionResult"]
