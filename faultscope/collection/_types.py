"""Typed collection inputs and logical error-rate statistics."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from dataclasses import dataclass, field, replace
import math
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar, Mapping


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


class _UnsetOptionDefault:
    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


_UNSET_NONE: Any = _UnsetOptionDefault(None)
_UNSET_BATCH_SIZE: Any = _UnsetOptionDefault(10_000)
_UNSET_MIN_SHOTS: Any = _UnsetOptionDefault(0)
_MAX_U64 = (1 << 64) - 1


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
    _MAX_SHOTS_EXPLICIT: ClassVar[int] = 1 << 0
    _MAX_ERRORS_EXPLICIT: ClassVar[int] = 1 << 1
    _BATCH_SIZE_EXPLICIT: ClassVar[int] = 1 << 2
    _START_BATCH_SIZE_EXPLICIT: ClassVar[int] = 1 << 3
    _MAX_BATCH_SIZE_EXPLICIT: ClassVar[int] = 1 << 4
    _MAX_BATCH_SECONDS_EXPLICIT: ClassVar[int] = 1 << 5
    _MIN_SHOTS_EXPLICIT: ClassVar[int] = 1 << 6
    _explicit_mask: ClassVar[int] = 0

    max_shots: int | None = None
    max_errors: int | None = None
    batch_size: int = 10_000
    start_batch_size: int | None = None
    max_batch_size: int | None = None
    max_batch_seconds: float | None = None
    min_shots: int = 0

    def __init__(
        self,
        max_shots: int | None = _UNSET_NONE,
        max_errors: int | None = _UNSET_NONE,
        batch_size: int = _UNSET_BATCH_SIZE,
        start_batch_size: int | None = _UNSET_NONE,
        max_batch_size: int | None = _UNSET_NONE,
        max_batch_seconds: float | None = _UNSET_NONE,
        *,
        min_shots: int = _UNSET_MIN_SHOTS,
    ) -> None:
        explicit_mask = 0
        if max_shots is _UNSET_NONE:
            max_shots = None
        else:
            explicit_mask |= self._MAX_SHOTS_EXPLICIT
        if max_errors is _UNSET_NONE:
            max_errors = None
        else:
            explicit_mask |= self._MAX_ERRORS_EXPLICIT
        if batch_size is _UNSET_BATCH_SIZE:
            batch_size = 10_000
        else:
            explicit_mask |= self._BATCH_SIZE_EXPLICIT
        if start_batch_size is _UNSET_NONE:
            start_batch_size = None
        else:
            explicit_mask |= self._START_BATCH_SIZE_EXPLICIT
        if max_batch_size is _UNSET_NONE:
            max_batch_size = None
        else:
            explicit_mask |= self._MAX_BATCH_SIZE_EXPLICIT
        if max_batch_seconds is _UNSET_NONE:
            max_batch_seconds = None
        else:
            explicit_mask |= self._MAX_BATCH_SECONDS_EXPLICIT
        if min_shots is _UNSET_MIN_SHOTS:
            min_shots = 0
        else:
            explicit_mask |= self._MIN_SHOTS_EXPLICIT

        object.__setattr__(self, "max_shots", max_shots)
        object.__setattr__(self, "max_errors", max_errors)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "start_batch_size", start_batch_size)
        object.__setattr__(self, "max_batch_size", max_batch_size)
        object.__setattr__(self, "max_batch_seconds", max_batch_seconds)
        object.__setattr__(self, "min_shots", min_shots)
        object.__setattr__(self, "_explicit_mask", explicit_mask)
        self.__post_init__()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_shots", self.max_shots),
            ("max_errors", self.max_errors),
            ("start_batch_size", self.start_batch_size),
            ("max_batch_size", self.max_batch_size),
        ):
            if value is not None:
                _require_integer(value, field_name=field_name)
        _require_integer(self.batch_size, field_name="batch_size")
        _require_integer(self.min_shots, field_name="min_shots")

        if self.max_shots is not None and self.max_shots <= 0:
            raise ValueError("max_shots must be positive")
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
        if self.max_batch_seconds is not None:
            seconds = _require_finite_number(
                self.max_batch_seconds,
                field_name="max_batch_seconds",
            )
            if seconds <= 0:
                raise ValueError("max_batch_seconds must be finite and positive")
            object.__setattr__(self, "max_batch_seconds", seconds)


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
        if self.seed is not None:
            _require_integer(self.seed, field_name="seed")
            if self.seed < 0 or self.seed > _MAX_U64:
                raise ValueError(f"seed must be between 0 and {_MAX_U64}")
        _require_integer(self.num_workers, field_name="num_workers")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")


@dataclass(frozen=True, init=False)
class CollectionTask:
    circuit: Any
    detectors: tuple[Any, ...] | None = None
    observables: tuple[Any, ...] | None = None
    decoder: object | str | None = None
    decoder_options: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None
    collection_options: CollectionOptions | None = None
    task_id: str | None = None
    postselection_mask: bytes | bytearray | memoryview | None = None
    postselected_observables_mask: bytes | bytearray | memoryview | None = None

    def __init__(
        self,
        circuit: Any = None,
        *,
        detectors: tuple[Any, ...] | None = None,
        observables: tuple[Any, ...] | None = None,
        decoder: object | str | None = None,
        decoder_options: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        collection_options: CollectionOptions | None = None,
        task_id: str | None = None,
        postselection_mask: bytes | bytearray | memoryview | None = None,
        postselected_observables_mask: bytes | bytearray | memoryview | None = None,
        **legacy: object,
    ) -> None:
        if "dem" in legacy:
            raise TypeError(
                "CollectionTask no longer accepts dem; use "
                "faultscope.collection.dem.DemCollectionTask(dem=...)"
            )
        if legacy:
            name = next(iter(legacy))
            raise TypeError(f"CollectionTask got an unexpected keyword argument {name!r}")
        object.__setattr__(self, "circuit", circuit)
        object.__setattr__(self, "detectors", None if detectors is None else tuple(detectors))
        object.__setattr__(self, "observables", None if observables is None else tuple(observables))
        object.__setattr__(self, "decoder", decoder)
        object.__setattr__(self, "decoder_options", decoder_options)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "collection_options", collection_options)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "postselection_mask", postselection_mask)
        object.__setattr__(
            self,
            "postselected_observables_mask",
            postselected_observables_mask,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.circuit is None:
            raise ValueError("CollectionTask requires circuit")
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
    location_sensitivities: Mapping[str, float]


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
    if append:
        with _CollectionCsvAppender(path) as appender:
            appender.write(stats)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=COLLECTION_CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for stat in stats:
            writer.writerow(stat.to_csv_row())


class _CollectionCsvAppender:
    """Validate an append target once and keep it open for streamed deltas."""

    def __init__(self, filepath: str | Path) -> None:
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", newline="")
        try:
            self._file.seek(0)
            existing_header = tuple(next(csv.reader(self._file), ()))
            if existing_header and existing_header != COLLECTION_CSV_FIELDS:
                raise ValueError(f"collection CSV header does not match in {self.path}")
            self._file.seek(0, io.SEEK_END)
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=COLLECTION_CSV_FIELDS,
                extrasaction="ignore",
            )
            if not existing_header:
                self._writer.writeheader()
        except BaseException:
            self._file.close()
            raise

    def write(self, stats: Iterable[TaskStats]) -> None:
        for stat in stats:
            self._writer.writerow(stat.to_csv_row())
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> _CollectionCsvAppender:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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


def _require_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _require_finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be finite and positive") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite and positive")
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
