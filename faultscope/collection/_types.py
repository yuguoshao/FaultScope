"""Typed collection inputs and logical error-rate statistics."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from dataclasses import dataclass, field, replace
import math
from collections.abc import Iterable, Iterator
from typing import Any, Mapping


COLLECTION_COUNTER_SCHEMA_VERSION = 1
OBSERVABLE_COMBO_PREFIX = "obs_mistake_mask="
DETECTION_EVENTS_KEY = "detection_events"
DETECTORS_CHECKED_KEY = "detectors_checked"

COLLECTION_CSV_FIELDS = (
    "shots",
    "errors",
    "discards",
    "seconds",
    "decoder",
    "strong_id",
    "json_metadata",
    "json_counter_schema",
    "custom_counts",
)
COLLECTION_CSV_HEADER = ",".join(COLLECTION_CSV_FIELDS)


class _UnsetMinShots:
    def __repr__(self) -> str:
        return "0"


_UNSET_MIN_SHOTS = _UnsetMinShots()


@dataclass(frozen=True)
class CollectionCounterSchema:
    """Versioned definition of the custom counters collected for every shot."""

    schema_version: int = field(default=COLLECTION_COUNTER_SCHEMA_VERSION, init=False)
    count_observable_error_combos: bool = False
    count_detection_events: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.count_observable_error_combos, bool):
            raise TypeError("count_observable_error_combos must be a bool")
        if not isinstance(self.count_detection_events, bool):
            raise TypeError("count_detection_events must be a bool")

    def _to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "count_observable_error_combos": self.count_observable_error_combos,
            "count_detection_events": self.count_detection_events,
        }

    @classmethod
    def _from_payload(cls, value: object) -> "CollectionCounterSchema":
        if not isinstance(value, Mapping):
            raise ValueError("collection counter schema must be a JSON object")
        raw_version = value.get("schema_version")
        if (
            isinstance(raw_version, bool)
            or not isinstance(raw_version, int)
            or raw_version != COLLECTION_COUNTER_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported collection counter schema version "
                f"{raw_version!r}; expected {COLLECTION_COUNTER_SCHEMA_VERSION}"
            )
        expected = {
            "schema_version",
            "count_observable_error_combos",
            "count_detection_events",
        }
        if set(value) != expected:
            raise ValueError("collection counter schema fields do not match the v3 contract")
        return cls(
            count_observable_error_combos=_require_bool(
                value["count_observable_error_combos"],
                field_name="count_observable_error_combos",
            ),
            count_detection_events=_require_bool(
                value["count_detection_events"],
                field_name="count_detection_events",
            ),
        )


@dataclass(frozen=True, init=False)
class CollectionOptions:
    max_shots: int | None = None
    max_errors: int | None = None
    batch_size: int = 10_000
    start_batch_size: int | None = None
    max_batch_size: int | None = None
    max_batch_seconds: float | None = None
    min_shots: int = 0

    def __init__(
        self,
        max_shots: int | None = None,
        max_errors: int | None = None,
        batch_size: int = 10_000,
        start_batch_size: int | None = None,
        max_batch_size: int | None = None,
        max_batch_seconds: float | None = None,
        *,
        min_shots: int | _UnsetMinShots = _UNSET_MIN_SHOTS,
    ) -> None:
        explicit_min_shots = min_shots is not _UNSET_MIN_SHOTS
        resolved_min_shots = 0 if not explicit_min_shots else min_shots
        object.__setattr__(self, "max_shots", max_shots)
        object.__setattr__(self, "max_errors", max_errors)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "start_batch_size", start_batch_size)
        object.__setattr__(self, "max_batch_size", max_batch_size)
        object.__setattr__(self, "max_batch_seconds", max_batch_seconds)
        object.__setattr__(self, "min_shots", resolved_min_shots)
        object.__setattr__(self, "_min_shots_explicit", explicit_min_shots)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.max_shots is not None and self.max_shots <= 0:
            raise ValueError("max_shots must be positive")
        if not isinstance(self.min_shots, int) or isinstance(self.min_shots, bool):
            raise TypeError("min_shots must be an integer")
        if self.min_shots < 0:
            raise ValueError("min_shots must be non-negative")
        if self.max_shots is not None and self.min_shots > self.max_shots:
            raise ValueError("min_shots must not exceed max_shots")
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
    counter_schema: CollectionCounterSchema | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("shots", self.shots),
            ("errors", self.errors),
            ("discards", self.discards),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.discards > self.shots:
            raise ValueError("discards must not exceed shots")
        if self.errors > self.shots - self.discards:
            raise ValueError("errors count must not exceed accepted shots")
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or not math.isfinite(self.seconds)
            or self.seconds < 0
        ):
            raise ValueError("seconds must be finite and non-negative")
        if not isinstance(self.custom_counts, Mapping):
            raise ValueError("custom_counts must be a mapping")
        for value in self.custom_counts.values():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("custom_counts values must be integers")
            if value < 0:
                raise ValueError("custom_counts values must be non-negative")
        if self.counter_schema is not None and not isinstance(
            self.counter_schema, CollectionCounterSchema
        ):
            raise TypeError("counter_schema must be a CollectionCounterSchema or None")
        self._validate_custom_counts_against_schema()

    def _validate_custom_counts_against_schema(self) -> None:
        schema = self.counter_schema
        if schema is None:
            return
        for fixed_key in (DETECTION_EVENTS_KEY, DETECTORS_CHECKED_KEY):
            present = fixed_key in self.custom_counts
            if present != schema.count_detection_events:
                expectation = "contain" if schema.count_detection_events else "not contain"
                raise ValueError(
                    f"custom_counts must {expectation} {fixed_key!r} for their counter schema"
                )
        for raw_key in self.custom_counts:
            key = str(raw_key)
            if key in {DETECTION_EVENTS_KEY, DETECTORS_CHECKED_KEY}:
                continue
            if not key.startswith(OBSERVABLE_COMBO_PREFIX):
                raise ValueError(f"unsupported versioned custom counter {key!r}")
            if not schema.count_observable_error_combos:
                raise ValueError(
                    f"observable combo counter {key!r} is disabled by the counter schema"
                )
            _validate_observable_combo_key_syntax(key)

    @property
    def raw_error_rate(self) -> float:
        if self.shots == 0:
            return math.nan
        return self.errors / self.shots

    @property
    def accepted_shots(self) -> int:
        return self.shots - self.discards

    @property
    def accepted_error_rate(self) -> float:
        if self.accepted_shots == 0:
            return math.nan
        return self.errors / self.accepted_shots

    @property
    def logical_error_rate(self) -> float:
        return self.accepted_error_rate

    @property
    def accepted_error_rate_stderr(self) -> float:
        if self.accepted_shots == 0:
            return math.nan
        p = self.accepted_error_rate
        return math.sqrt(p * (1.0 - p) / self.accepted_shots)

    @property
    def logical_error_rate_stderr(self) -> float:
        return self.accepted_error_rate_stderr

    def with_edits(self, **edits: object) -> "TaskStats":
        return replace(self, **edits)  # type: ignore[arg-type]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "shots": str(int(self.shots)),
            "errors": str(int(self.errors)),
            "discards": str(int(self.discards)),
            "seconds": f"{float(self.seconds):.12g}",
            "decoder": self.decoder or "",
            "strong_id": self.strong_id,
            "json_metadata": _canonical_json(dict(self.metadata)),
            "json_counter_schema": _canonical_json(
                None if self.counter_schema is None else self.counter_schema._to_payload()
            ),
            "custom_counts": _canonical_json(
                {str(key): int(value) for key, value in self.custom_counts.items()}
            ),
        }

    def to_csv_line(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=COLLECTION_CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writerow(self.to_csv_row())
        return buffer.getvalue().rstrip("\r\n")

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "TaskStats":
        metadata = json.loads(row.get("json_metadata") or "null")
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        custom_counts = json.loads(row.get("custom_counts") or "{}")
        if not isinstance(custom_counts, Mapping):
            raise ValueError("collection CSV custom_counts must be a JSON object")
        strong_id = row.get("strong_id", "")
        raw_counter_schema = json.loads(row.get("json_counter_schema") or "null")
        counter_schema = (
            None
            if raw_counter_schema is None
            else CollectionCounterSchema._from_payload(raw_counter_schema)
        )
        return cls(
            task_id=strong_id,
            strong_id=strong_id,
            shots=_parse_non_negative_int(row.get("shots", "0"), field_name="shots"),
            errors=_parse_non_negative_int(row.get("errors", "0"), field_name="errors"),
            discards=_parse_non_negative_int(row.get("discards", "0"), field_name="discards"),
            seconds=_parse_non_negative_float(row.get("seconds", "0"), field_name="seconds"),
            decoder=row.get("decoder") or None,
            metadata=dict(metadata),
            custom_counts=_parse_custom_counts(custom_counts),
            counter_schema=counter_schema,
        )

    def __add__(self, other: "TaskStats") -> "TaskStats":
        if not isinstance(other, TaskStats):
            return NotImplemented
        if self.strong_id != other.strong_id:
            raise ValueError("cannot merge stats with different strong_id values")
        if self.decoder != other.decoder or _canonical_json(dict(self.metadata)) != _canonical_json(
            dict(other.metadata)
        ):
            raise ValueError("stats with the same strong_id have different decoder or metadata")
        if self.counter_schema != other.counter_schema:
            raise ValueError("stats with the same strong_id have different counter schemas")
        custom_counts = {str(key): int(value) for key, value in self.custom_counts.items()}
        for key, value in other.custom_counts.items():
            custom_counts[str(key)] = int(custom_counts.get(str(key), 0)) + int(value)
        return TaskStats(
            task_id=self.task_id or other.task_id,
            strong_id=self.strong_id,
            shots=self.shots + other.shots,
            errors=self.errors + other.errors,
            discards=self.discards + other.discards,
            seconds=self.seconds + other.seconds,
            decoder=self.decoder,
            metadata=dict(self.metadata),
            custom_counts=custom_counts,
            counter_schema=self.counter_schema,
        )


@dataclass(frozen=True)
class HotspotCollectionResult:
    stats: TaskStats
    batch_stats: tuple[TaskStats, ...]
    edge_sensitivities: tuple[float, ...]


@dataclass(frozen=True)
class Progress:
    new_stats: tuple[TaskStats, ...]
    status_message: str


class CollectionData:
    def __init__(self, stats: Iterable[TaskStats] = ()) -> None:
        self.data: dict[str, TaskStats] = {}
        for stat in stats:
            self.add_sample(stat)

    def add_sample(self, stat: TaskStats) -> None:
        existing = self.data.get(stat.strong_id)
        self.data[stat.strong_id] = stat if existing is None else existing + stat

    def values(self) -> tuple[TaskStats, ...]:
        return tuple(self.data.values())

    def __iter__(self) -> Iterator[TaskStats]:
        return iter(self.data.values())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, strong_id: str) -> TaskStats:
        return self.data[strong_id]


def read_stats_from_csv_files(*filepaths: str | Path | Iterable[str | Path]) -> list[TaskStats]:
    data = CollectionData()
    for path in _flatten_filepaths(filepaths):
        path = Path(path)
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != COLLECTION_CSV_FIELDS:
                raise ValueError(f"collection CSV header does not match in {path}")
            for row in reader:
                data.add_sample(TaskStats.from_csv_row(row))
    return list(data.values())


def write_stats_to_csv_file(
    filepath: str | Path,
    stats: Iterable[TaskStats],
    *,
    append: bool = False,
) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    if append and not write_header:
        with path.open(newline="") as existing_file:
            existing_header = tuple(next(csv.reader(existing_file), ()))
        if existing_header != COLLECTION_CSV_FIELDS:
            raise ValueError(f"collection CSV header does not match in {path}")
    with path.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=COLLECTION_CSV_FIELDS,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        for stat in stats:
            writer.writerow(stat.to_csv_row())


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_non_negative_int(value: str, *, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"collection CSV {field_name} must be non-negative")
    return parsed


def _parse_non_negative_float(value: str, *, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"collection CSV {field_name} must be finite and non-negative")
    return parsed


def _parse_custom_counts(custom_counts: Mapping[str, object]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for key, value in custom_counts.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("collection CSV custom_counts values must be integers")
        if value < 0:
            raise ValueError("collection CSV custom_counts values must be non-negative")
        parsed[str(key)] = value
    return parsed


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"collection counter schema {field_name} must be a bool")
    return value


def _validate_observable_combo_key_syntax(key: str) -> None:
    mask = key.removeprefix(OBSERVABLE_COMBO_PREFIX)
    if not mask or set(mask) - {"E", "_"} or "E" not in mask:
        raise ValueError(
            f"observable combo key {key!r} must contain a non-empty E/_ mask with at least one E"
        )


def _flatten_filepaths(
    values: tuple[str | Path | Iterable[str | Path], ...],
) -> Iterator[str | Path]:
    for value in values:
        if isinstance(value, (str, Path)):
            yield value
        else:
            yield from value
