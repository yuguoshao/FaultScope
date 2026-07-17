import builtins
from collections.abc import Iterable, Iterator
import csv
from dataclasses import fields
import gc
import io
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import weakref
from unittest import mock
from typing import get_type_hints

import faultscope
import faultscope.collection._collect as collection_collect_module
from faultscope import (
    Circuit,
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    LogicalObservable,
    NativeCompositeDecoder,
    NativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder,
    Operation,
    BernoulliPauliNoise,
    NoiseLocation,
)
from faultscope.collection import (
    COLLECTION_COUNTER_SCHEMA_VERSION,
    COLLECTION_CSV_FIELDS,
    COLLECTION_CSV_HEADER,
    Collector,
    CollectionCounterSchema,
    CollectionData,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    Progress,
    TaskStats,
    analyze_thresholds,
    collect as public_collect,
    iter_collect as public_iter_collect,
    iter_progress as public_iter_progress,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)
from faultscope.collection.analysis import (
    error_rate_points,
    fit_log_error_rate_lines,
    plot_error_rates,
    predict_error_rate,
)


def _logical_edge_dem(probability: float = 1.0) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=probability,
                detectors=(),
                observables=(0,),
                location_id="logical_edge",
                event="L",
            ),
        ),
    )


def _graphlike_dem(probability: float = 1.0) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(Detector(id=0, measurement_keys=()),),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=probability,
                detectors=(0,),
                observables=(0,),
                location_id="edge0",
                event="X",
            ),
        ),
    )


def _two_observable_dem(probability: float = 1.0) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(),
        observables=(LogicalObservable(id=0), LogicalObservable(id=1)),
        edges=(
            DetectorErrorEdge(
                probability=probability,
                detectors=(),
                observables=(0,),
                location_id="logical_edge_0",
                event="L0",
            ),
        ),
    )


def _native_counter_schema(
    *,
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": COLLECTION_COUNTER_SCHEMA_VERSION,
        "count_observable_error_combos": count_observable_error_combos,
        "count_detection_events": count_detection_events,
    }


def _collect(
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
    count_observable_error_combos: bool = False,
    count_detection_events: bool = False,
    custom_error_count_key: str | None = None,
    decoders: Iterable[str | object] | str | object | None = None,
) -> list[TaskStats]:
    base = options or CollectionOptions()
    effective_options = CollectionOptions(
        max_shots=max_shots if max_shots is not None else base.max_shots,
        max_errors=max_errors if max_errors is not None else base.max_errors,
        batch_size=batch_size if batch_size is not None else base.batch_size,
        start_batch_size=(
            start_batch_size if start_batch_size is not None else base.start_batch_size
        ),
        max_batch_size=(max_batch_size if max_batch_size is not None else base.max_batch_size),
        max_batch_seconds=(
            max_batch_seconds if max_batch_seconds is not None else base.max_batch_seconds
        ),
    )
    if decoders is None:
        decoder_tuple: tuple[str | object, ...] = ()
    elif isinstance(decoders, str) or not isinstance(decoders, Iterable):
        decoder_tuple = (decoders,)
    else:
        decoder_tuple = tuple(decoders)
    run_options = CollectionRunOptions(
        seed=seed,
        num_workers=1 if num_workers is None else num_workers,
        existing_data_filepaths=tuple(existing_data_filepaths),
        save_resume_filepath=save_resume_filepath,
        count_observable_error_combos=count_observable_error_combos,
        count_detection_events=count_detection_events,
        custom_error_count_key=custom_error_count_key,
        decoders=decoder_tuple,
    )
    return public_collect(tasks, options=effective_options, run_options=run_options)


class CollectionTests(unittest.TestCase):
    def _threshold_stat(
        self,
        *,
        x: float,
        distance: int,
        errors: int,
        shots: int = 1000,
    ) -> TaskStats:
        task_id = f"threshold-{distance}-{x}-{errors}"
        return TaskStats(
            task_id=task_id,
            shots=shots,
            errors=errors,
            discards=0,
            seconds=0.0,
            decoder="mwpm",
            metadata={"p": x, "d": distance},
            strong_id=task_id,
        )

    def _scaling_threshold_stats(self, *, shots: int = 8000) -> list[TaskStats]:
        stats = []
        threshold = 0.031
        nu = 1.4
        for distance in (5, 7, 9, 11):
            for x in (0.021, 0.026, 0.031, 0.036, 0.041):
                z = (x - threshold) * distance ** (1.0 / nu)
                rate = 1.0 / (1.0 + math.exp(-(-1.1 + 52.0 * z + 4.0 * z * z)))
                stats.append(
                    self._threshold_stat(
                        x=x,
                        distance=distance,
                        errors=round(rate * shots),
                        shots=shots,
                    )
                )
        return stats

    def _run_collection_cli(
        self,
        temp: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
        env["PYTHONNOUSERSITE"] = "1"
        return subprocess.run(
            [sys.executable, "-m", "faultscope.collection", *args],
            cwd=temp,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def _write_threshold_cli_stats(self, temp: Path) -> Path:
        stats_path = temp / "threshold.csv"
        write_stats_to_csv_file(
            stats_path,
            [
                TaskStats(
                    "d3-p01",
                    1000,
                    100,
                    0,
                    0.0,
                    "mwpm",
                    {"p": 0.01, "d": 3, "family": "rotated"},
                    strong_id="d3-p01",
                ),
                TaskStats(
                    "d3-p03",
                    1000,
                    300,
                    0,
                    0.0,
                    "mwpm",
                    {"p": 0.03, "d": 3, "family": "rotated"},
                    strong_id="d3-p03",
                ),
                TaskStats(
                    "d5-p01",
                    1000,
                    300,
                    0,
                    0.0,
                    "mwpm",
                    {"p": 0.01, "d": 5, "family": "rotated"},
                    strong_id="d5-p01",
                ),
                TaskStats(
                    "d5-p03",
                    1000,
                    100,
                    0,
                    0.0,
                    "mwpm",
                    {"p": 0.03, "d": 5, "family": "rotated"},
                    strong_id="d5-p03",
                ),
                TaskStats(
                    "b-d3-p01",
                    1000,
                    90,
                    0,
                    0.0,
                    "bposd",
                    {"p": 0.01, "d": 3, "family": "rotated"},
                    strong_id="b-d3-p01",
                ),
                TaskStats(
                    "b-d3-p03",
                    1000,
                    320,
                    0,
                    0.0,
                    "bposd",
                    {"p": 0.03, "d": 3, "family": "rotated"},
                    strong_id="b-d3-p03",
                ),
                TaskStats(
                    "b-d5-p01",
                    1000,
                    310,
                    0,
                    0.0,
                    "bposd",
                    {"p": 0.01, "d": 5, "family": "rotated"},
                    strong_id="b-d5-p01",
                ),
                TaskStats(
                    "b-d5-p03",
                    1000,
                    110,
                    0,
                    0.0,
                    "bposd",
                    {"p": 0.03, "d": 5, "family": "rotated"},
                    strong_id="b-d5-p03",
                ),
            ],
        )
        return stats_path

    def _write_scaling_threshold_cli_stats(self, temp: Path) -> Path:
        stats_path = temp / "scaling-threshold.csv"
        stats = [
            item.with_edits(metadata={**item.metadata, "family": "rotated"})
            for item in self._scaling_threshold_stats()
        ]
        write_stats_to_csv_file(stats_path, stats)
        return stats_path

    def test_threshold_core_is_covered_by_unittest_discovery(self) -> None:
        rates_by_distance = {
            3: (100, 200, 400, 500),
            5: (50, 200, 500, 600),
            7: (30, 200, 600, 700),
            9: (100, 300, 600, 600),
        }
        stats = [
            self._threshold_stat(x=x, distance=distance, errors=errors)
            for distance, errors_by_x in rates_by_distance.items()
            for x, errors in zip((0.01, 0.02, 0.04, 0.05), errors_by_x)
        ]

        (result,) = analyze_thresholds(
            stats,
            x_key="p",
            distance_key="d",
            bootstrap_samples=0,
        )

        self.assertAlmostEqual(result.pairwise_threshold.value, 0.03)

        (raw_result,) = analyze_thresholds(
            [self._threshold_stat(x=0.01, distance=3, errors=0, shots=10)],
            x_key="p",
            distance_key="d",
            bootstrap_samples=0,
        )
        self.assertEqual(raw_result.points[0].rate, 0.0)
        self.assertEqual(raw_result.points[0].stderr, 0.0)

    def test_threshold_scaling_core_is_covered_by_unittest_discovery(self) -> None:
        stats = self._scaling_threshold_stats()
        (valid,) = analyze_thresholds(
            stats,
            x_key="p",
            distance_key="d",
            bootstrap_samples=0,
        )
        self.assertEqual(valid.scaling_fit.status, "ok")
        self.assertAlmostEqual(valid.scaling_fit.threshold.value, 0.031, delta=0.003)

        incomplete = stats + [self._threshold_stat(x=0.031, distance=13, errors=250)]
        (rejected,) = analyze_thresholds(
            incomplete,
            x_key="p",
            distance_key="d",
            bootstrap_samples=0,
        )
        self.assertEqual(rejected.scaling_fit.status, "insufficient_data")

    def test_threshold_bootstrap_is_deterministic_under_unittest(self) -> None:
        stats = [
            self._threshold_stat(x=0.01, distance=3, errors=100),
            self._threshold_stat(x=0.02, distance=3, errors=300),
            self._threshold_stat(x=0.03, distance=3, errors=500),
            self._threshold_stat(x=0.01, distance=5, errors=500),
            self._threshold_stat(x=0.02, distance=5, errors=300),
            self._threshold_stat(x=0.03, distance=5, errors=100),
        ]
        kwargs = dict(
            x_key="p",
            distance_key="d",
            bootstrap_samples=40,
            seed=123,
        )

        first = analyze_thresholds(stats, **kwargs)
        second = analyze_thresholds(list(reversed(stats)), **kwargs)

        self.assertEqual(first[0].to_dict(), second[0].to_dict())

    def test_options_validate_positive_limits(self) -> None:
        CollectionOptions(max_shots=1, max_errors=0, batch_size=1)
        CollectionOptions(start_batch_size=1, max_batch_size=2, max_batch_seconds=0.25)

        with self.assertRaisesRegex(ValueError, "max_shots"):
            CollectionOptions(max_shots=0)
        with self.assertRaisesRegex(ValueError, "max_errors"):
            CollectionOptions(max_errors=-1)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            CollectionOptions(batch_size=0)
        with self.assertRaisesRegex(ValueError, "start_batch_size"):
            CollectionOptions(start_batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_batch_size"):
            CollectionOptions(max_batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_batch_seconds"):
            CollectionOptions(max_batch_seconds=0)

    def test_collection_run_options_validate_workers(self) -> None:
        options = CollectionRunOptions(seed=5, num_workers=2, decoders=("mwpm",))

        self.assertEqual(options.seed, 5)
        self.assertEqual(options.num_workers, 2)
        self.assertEqual(options.decoders, ("mwpm",))
        self.assertNotIn("seed", {field.name for field in fields(CollectionOptions)})
        with self.assertRaisesRegex(ValueError, "num_workers must be positive"):
            CollectionRunOptions(num_workers=0)

    def test_task_requires_exactly_one_source(self) -> None:
        dem = _logical_edge_dem()
        CollectionTask(dem=dem)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            CollectionTask()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            CollectionTask(dem=dem, circuit=Circuit(1, (Operation.measure(0, key="m0"),)))

    def test_task_stats_rates_have_explicit_denominators(self) -> None:
        stats = TaskStats(
            task_id="case",
            shots=100,
            errors=25,
            discards=0,
            seconds=0.5,
            decoder=None,
            metadata={"d": 3},
            strong_id="strong",
            custom_counts={},
        )

        self.assertEqual(stats.raw_error_rate, 0.25)
        self.assertEqual(stats.accepted_error_rate, 0.25)
        self.assertEqual(stats.logical_error_rate, 0.25)
        self.assertEqual(
            stats.accepted_error_rate_stderr,
            math.sqrt(0.25 * 0.75 / 100),
        )
        self.assertEqual(
            stats.logical_error_rate_stderr,
            stats.accepted_error_rate_stderr,
        )
        self.assertEqual(stats.accepted_shots, 100)
        self.assertFalse(hasattr(stats, "error_rate"))
        self.assertFalse(hasattr(stats, "stderr"))

    def test_task_stats_rates_use_accepted_shots_for_logical_rate(self) -> None:
        stats = TaskStats(
            task_id="case",
            shots=100,
            errors=10,
            discards=20,
            seconds=0.5,
            decoder=None,
            metadata={},
            strong_id="strong",
            custom_counts={},
        )

        self.assertEqual(stats.raw_error_rate, 0.1)
        self.assertEqual(stats.accepted_error_rate, 0.125)
        self.assertEqual(stats.logical_error_rate, 0.125)
        self.assertEqual(
            stats.accepted_error_rate_stderr,
            math.sqrt(0.125 * 0.875 / 80),
        )

    def test_task_stats_rejects_invalid_counts_and_elapsed_time(self) -> None:
        base = {
            "task_id": "case",
            "shots": 10,
            "errors": 2,
            "discards": 1,
            "seconds": 0.5,
            "decoder": None,
            "metadata": {},
            "strong_id": "strong",
            "custom_counts": {},
        }
        invalid_edits = (
            {"shots": -1},
            {"errors": -1},
            {"discards": -1},
            {"discards": 11},
            {"errors": 10},
            {"seconds": -0.1},
            {"seconds": math.nan},
            {"seconds": math.inf},
            {"custom_counts": {"bad": -1}},
            {"custom_counts": {"bad": True}},
        )
        for edits in invalid_edits:
            with self.subTest(edits=edits):
                with self.assertRaises(ValueError):
                    TaskStats(**(base | edits))

        valid = TaskStats(**base)
        with self.assertRaisesRegex(ValueError, "accepted shots"):
            valid.with_edits(errors=10)

    def test_task_stats_csv_utilities_round_trip_and_merge(self) -> None:
        stats = TaskStats(
            task_id="case",
            shots=10,
            errors=2,
            discards=1,
            seconds=0.5,
            decoder="native",
            metadata={"d": 3},
            strong_id="strong",
            custom_counts={"obs_mistake_mask=E": 2},
        )
        row_data = stats.to_csv_row()
        line = stats.to_csv_line()
        row = next(csv.DictReader(io.StringIO(COLLECTION_CSV_HEADER + "\n" + line)))
        roundtrip = TaskStats.from_csv_row(row)

        self.assertEqual(COLLECTION_CSV_FIELDS[0], "shots")
        self.assertEqual(tuple(row_data), COLLECTION_CSV_FIELDS)
        self.assertEqual(roundtrip.task_id, stats.strong_id)
        self.assertNotEqual(roundtrip, stats)
        self.assertEqual(
            stats.with_edits(shots=5, errors=1).shots,
            5,
        )
        merged = stats + stats.with_edits(
            task_id="other-display-id",
            shots=3,
            errors=1,
            discards=0,
            seconds=0.25,
            custom_counts={"obs_mistake_mask=E": 1, "detectors_checked": 10},
        )
        self.assertEqual(merged.task_id, "case")
        self.assertEqual(merged.shots, 13)
        self.assertEqual(merged.errors, 3)
        self.assertEqual(merged.custom_counts["obs_mistake_mask=E"], 3)
        self.assertEqual(merged.custom_counts["detectors_checked"], 10)
        with self.assertRaisesRegex(ValueError, "same strong_id"):
            _ = stats + stats.with_edits(metadata={"d": 5})

    def test_versioned_counter_schema_round_trips_and_must_match_when_merging(self) -> None:
        detection_schema = CollectionCounterSchema(count_detection_events=True)
        stats = TaskStats(
            task_id="versioned",
            strong_id="versioned-strong",
            shots=0,
            errors=0,
            discards=0,
            seconds=0.0,
            decoder=None,
            metadata={},
            custom_counts={"detection_events": 0, "detectors_checked": 0},
            counter_schema=detection_schema,
        )
        row = next(csv.DictReader(io.StringIO(COLLECTION_CSV_HEADER + "\n" + stats.to_csv_line())))

        self.assertEqual(TaskStats.from_csv_row(row), stats.with_edits(task_id=stats.strong_id))
        with self.assertRaisesRegex(ValueError, "counter schemas"):
            _ = stats + stats.with_edits(
                counter_schema=CollectionCounterSchema(),
                custom_counts={},
            )

    def test_v2_csv_read_and_append_are_rejected_without_modifying_the_file(self) -> None:
        v2_header = "shots,errors,discards,seconds,decoder,strong_id,json_metadata,custom_counts\n"
        v2_row = "1,0,0,0,,legacy,{},{}\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy-v2.csv"
            original = v2_header + v2_row
            path.write_text(original)

            with self.assertRaisesRegex(ValueError, "header does not match"):
                read_stats_from_csv_files(path)
            with self.assertRaisesRegex(ValueError, "header does not match"):
                write_stats_to_csv_file(
                    path,
                    [
                        TaskStats(
                            task_id="new",
                            strong_id="new",
                            shots=1,
                            errors=0,
                            discards=0,
                            seconds=0.0,
                            decoder=None,
                            metadata={},
                        )
                    ],
                    append=True,
                )

            self.assertEqual(path.read_text(), original)

    def test_read_stats_from_csv_files_rejects_invalid_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            malformed_header = base / "malformed_header.csv"
            malformed_header.write_text(
                "shots,errors,discards,seconds,decoder,strong_id,json_metadata\n"
            )
            with self.assertRaisesRegex(ValueError, "header does not match"):
                read_stats_from_csv_files(malformed_header)

            invalid_rows = [
                (
                    "negative_shots",
                    {
                        "shots": "-1",
                        "errors": "0",
                        "discards": "0",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "negative_errors",
                    {
                        "shots": "1",
                        "errors": "-1",
                        "discards": "0",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "negative_discards",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "-1",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "negative_seconds",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "0",
                        "seconds": "-0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "non_finite_seconds",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "0",
                        "seconds": "nan",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "discards_exceed_shots",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "2",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "errors_exceed_accepted_shots",
                    {
                        "shots": "2",
                        "errors": "2",
                        "discards": "1",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": "{}",
                    },
                ),
                (
                    "negative_custom_count",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "0",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": '{"obs":-1}',
                    },
                ),
                (
                    "non_object_custom_counts",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "0",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": '["obs"]',
                    },
                ),
                (
                    "non_integer_custom_count",
                    {
                        "shots": "1",
                        "errors": "0",
                        "discards": "0",
                        "seconds": "0.5",
                        "decoder": "native",
                        "strong_id": "strong",
                        "json_metadata": "{}",
                        "custom_counts": '{"obs":1.5}',
                    },
                ),
            ]

            for name, row in invalid_rows:
                path = base / f"{name}.csv"
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=COLLECTION_CSV_FIELDS)
                    writer.writeheader()
                    writer.writerow(row)
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        read_stats_from_csv_files(path)

    def test_collection_data_reads_writes_and_merges_csv(self) -> None:
        first = TaskStats(
            task_id="case",
            shots=10,
            errors=2,
            discards=0,
            seconds=0.5,
            decoder="native",
            metadata={"d": 3},
            strong_id="strong",
            custom_counts={},
        )
        second = first.with_edits(shots=5, errors=1, seconds=0.25)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stats.csv"
            write_stats_to_csv_file(path, [first, second])
            stats = read_stats_from_csv_files(path)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].shots, 15)
        self.assertEqual(stats[0].errors, 3)
        data = CollectionData(stats)
        self.assertEqual(data["strong"].shots, 15)

    def test_collect_reports_all_logical_edge_failures(self) -> None:
        stats = _collect(
            [CollectionTask(dem=_logical_edge_dem(), task_id="logical")],
            max_shots=16,
            batch_size=5,
            seed=1,
        )

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].task_id, "logical")
        self.assertEqual(stats[0].shots, 16)
        self.assertEqual(stats[0].errors, 16)
        self.assertEqual(stats[0].discards, 0)
        self.assertEqual(stats[0].raw_error_rate, 1.0)
        self.assertTrue(stats[0].strong_id)

    def test_counter_flags_change_strong_id_but_not_sampling_identity_or_samples(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(0.375), task_id="schema-identity")
        options = CollectionOptions(max_shots=256, batch_size=32)
        schemas = [
            CollectionCounterSchema(
                count_observable_error_combos=count_combos,
                count_detection_events=count_detection,
            )
            for count_combos in (False, True)
            for count_detection in (False, True)
        ]
        native_tasks = [
            collection_collect_module._native_task(task, 0, options, schema) for schema in schemas
        ]
        stats = [
            _collect(
                [task],
                options=options,
                seed=1234,
                count_observable_error_combos=schema.count_observable_error_combos,
                count_detection_events=schema.count_detection_events,
            )[0]
            for schema in schemas
        ]

        self.assertEqual(len({item["sampling_id"] for item in native_tasks}), 1)
        self.assertEqual(len({item["strong_id"] for item in native_tasks}), 4)
        self.assertEqual(len({stat.strong_id for stat in stats}), 4)
        self.assertEqual(
            {(stat.shots, stat.errors, stat.discards) for stat in stats},
            {(256, stats[0].errors, 0)},
        )
        self.assertEqual([stat.counter_schema for stat in stats], schemas)

    def test_custom_stop_key_does_not_change_strong_id(self) -> None:
        task = CollectionTask(dem=_graphlike_dem(0.25), task_id="stop-key-identity")
        common = dict(
            max_shots=32,
            batch_size=8,
            seed=91,
            count_detection_events=True,
        )
        default_stop = _collect([task], **common)[0]
        detection_stop = _collect(
            [task],
            custom_error_count_key="detection_events",
            **common,
        )[0]

        self.assertEqual(default_stop.strong_id, detection_stop.strong_id)
        self.assertEqual(default_stop.custom_counts, detection_stop.custom_counts)

    def test_native_graphlike_decoder_can_remove_all_failures(self) -> None:
        dem = _graphlike_dem()
        decoder = NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

        stats = _collect(
            [CollectionTask(dem=dem, decoder=decoder, task_id="corrected")],
            max_shots=32,
            batch_size=8,
            seed=2,
        )[0]

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(stats.shots, 32)
        self.assertEqual(stats.errors, 0)

    def test_strong_id_distinguishes_composite_decoder_behavior(self) -> None:
        dem = _graphlike_dem()
        no_correction = NativeCompositeDecoder(
            (
                NativeNoCorrectionDecoder(
                    observable_ids=(0,),
                    detector_ids=(0,),
                ),
            )
        )
        correcting = NativeCompositeDecoder((NativeGraphlikeDetectorCopyDecoder.from_dem(dem),))

        stats = _collect(
            [
                CollectionTask(dem=dem, decoder=no_correction, task_id="uncorrected"),
                CollectionTask(dem=dem, decoder=correcting, task_id="corrected"),
            ],
            max_shots=4,
            batch_size=4,
            seed=22,
        )

        self.assertEqual([stat.errors for stat in stats], [4, 0])
        self.assertNotEqual(stats[0].strong_id, stats[1].strong_id)
        self.assertEqual(len(CollectionData(stats)), 2)

    def test_native_decoder_payload_tracks_id_layout(self) -> None:
        first = NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(1,))
        second = NativeNoCorrectionDecoder(observable_ids=(2,), detector_ids=(1,))

        self.assertNotEqual(first.strong_id_payload(), second.strong_id_payload())

    def test_native_graphlike_payload_tracks_effective_mapping(self) -> None:
        def mapped_dem(detector_id: int) -> DetectorErrorModel:
            return DetectorErrorModel(
                detectors=(
                    Detector(id=0, measurement_keys=()),
                    Detector(id=1, measurement_keys=()),
                ),
                observables=(LogicalObservable(id=0),),
                edges=(
                    DetectorErrorEdge(
                        probability=0.25,
                        detectors=(detector_id,),
                        observables=(0,),
                        location_id="mapped",
                        event="X",
                    ),
                ),
            )

        first = NativeGraphlikeDetectorCopyDecoder.from_dem(mapped_dem(0))
        second = NativeGraphlikeDetectorCopyDecoder.from_dem(mapped_dem(1))

        self.assertEqual(first.detector_ids, second.detector_ids)
        self.assertEqual(first.observable_ids, second.observable_ids)
        self.assertNotEqual(first.strong_id_payload(), second.strong_id_payload())

    def test_string_and_equivalent_object_decoder_share_strong_id(self) -> None:
        dem = _graphlike_dem()
        by_name = _collect(
            [CollectionTask(dem=dem, decoder="graphlike-detector-copy")],
            max_shots=1,
            seed=26,
        )[0]
        by_object = _collect(
            [
                CollectionTask(
                    dem=dem,
                    decoder=NativeGraphlikeDetectorCopyDecoder.from_dem(dem),
                )
            ],
            max_shots=1,
            seed=26,
        )[0]

        self.assertEqual(by_name.strong_id, by_object.strong_id)

    def test_strong_id_canonicalizes_dem_mapping_order(self) -> None:
        def tagged_dem(tags: dict[str, int]) -> DetectorErrorModel:
            return DetectorErrorModel(
                detectors=(),
                observables=(LogicalObservable(id=0),),
                edges=(
                    DetectorErrorEdge(
                        probability=1.0,
                        detectors=(),
                        observables=(0,),
                        location_id="logical",
                        event="L",
                        tags=tags,
                    ),
                ),
            )

        first = _collect(
            [CollectionTask(dem=tagged_dem({"round": 1, "gate": 2}))],
            max_shots=1,
            seed=23,
        )[0]
        second = _collect(
            [CollectionTask(dem=tagged_dem({"gate": 2, "round": 1}))],
            max_shots=1,
            seed=23,
        )[0]

        self.assertEqual(first.strong_id, second.strong_id)

    def test_strong_id_canonicalizes_circuit_mapping_order(self) -> None:
        def tagged_circuit(tags: dict[str, int], metadata: dict[str, int]) -> Circuit:
            noise = NoiseLocation(
                "x0",
                BernoulliPauliNoise("X"),
                1.0,
                (0,),
                tags=tags,
            )
            return Circuit(
                1,
                (
                    Operation.noise(noise, **metadata),
                    Operation.measure(0, key="m0"),
                    Operation.observable_include(0, ("m0",)),
                ),
            )

        first = _collect(
            [
                CollectionTask(
                    circuit=tagged_circuit(
                        {"round": 1, "gate": 2},
                        {"layer": 3, "moment": 4},
                    )
                )
            ],
            max_shots=1,
            seed=24,
        )[0]
        second = _collect(
            [
                CollectionTask(
                    circuit=tagged_circuit(
                        {"gate": 2, "round": 1},
                        {"moment": 4, "layer": 3},
                    )
                )
            ],
            max_shots=1,
            seed=24,
        )[0]

        self.assertEqual(first.strong_id, second.strong_id)

    def test_strong_id_requires_valid_decoder_payload(self) -> None:
        class MissingPayload:
            pass

        class InvalidPayload:
            def strong_id_payload(self):
                return {"unstable": object()}

        class IncompletePayload:
            def strong_id_payload(self):
                return {}

        for decoder, message in (
            (MissingPayload(), "provide strong_id_payload"),
            (InvalidPayload(), "JSON-serializable mapping"),
            (IncompletePayload(), "missing required field"),
        ):
            with self.subTest(decoder=type(decoder).__name__):
                with self.assertRaisesRegex(TypeError, message):
                    _collect(
                        [CollectionTask(dem=_graphlike_dem(), decoder=decoder)],
                        max_shots=1,
                    )

    def test_max_errors_stops_after_completed_batch(self) -> None:
        stats = _collect(
            [CollectionTask(dem=_logical_edge_dem())],
            max_shots=10,
            max_errors=1,
            batch_size=4,
            seed=3,
        )[0]

        self.assertEqual(stats.shots, 4)
        self.assertEqual(stats.errors, 4)

    def test_task_options_do_not_reset_call_batch_size_by_default(self) -> None:
        stats = _collect(
            [
                CollectionTask(
                    dem=_logical_edge_dem(),
                    collection_options=CollectionOptions(max_errors=1),
                )
            ],
            max_shots=10,
            batch_size=4,
            seed=33,
        )[0]

        self.assertEqual(stats.shots, 4)
        self.assertEqual(stats.errors, 4)

    def test_native_collection_delegates_loop_to_sampler(self) -> None:
        dem = _logical_edge_dem()

        class RustCoreSampler:
            pass

        sampler = RustCoreSampler()
        with (
            mock.patch(
                "faultscope.collection._collect._compile_task_sampler",
                return_value=(sampler, dem),
            ),
            mock.patch(
                "faultscope.collection._collect._collect_dem_logical_error_stats_many",
                return_value=[
                    {
                        "task_id": "native-core",
                        "strong_id": "strong",
                        "shots": 7,
                        "errors": 3,
                        "discards": 0,
                        "seconds": 0.25,
                        "decoder": None,
                        "metadata": {},
                        "counter_schema": _native_counter_schema(),
                        "custom_counts": {},
                    }
                ],
            ) as collect_native,
        ):
            stats = _collect(
                [CollectionTask(dem=dem, task_id="native-core")],
                max_shots=11,
                max_errors=5,
                batch_size=4,
                seed=9,
            )[0]

        self.assertEqual(collect_native.call_count, 1)
        native_tasks = collect_native.call_args.args[0]
        self.assertEqual(len(native_tasks), 1)
        self.assertIs(native_tasks[0]["sampler"], sampler)
        self.assertEqual(native_tasks[0]["max_shots"], 11)
        self.assertEqual(native_tasks[0]["max_errors"], 5)
        self.assertEqual(native_tasks[0]["batch_size"], 4)
        self.assertIsNone(native_tasks[0]["seed"])
        self.assertEqual(collect_native.call_args.kwargs["seed"], 9)
        self.assertEqual(stats.task_id, "native-core")
        self.assertEqual(stats.shots, 7)
        self.assertEqual(stats.errors, 3)
        self.assertEqual(stats.seconds, 0.25)

    def test_same_seed_repeats_probabilistic_collection(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(probability=0.375))

        first = _collect([task], max_shots=256, batch_size=64, seed=123)[0]
        second = _collect([task], max_shots=256, batch_size=64, seed=123)[0]

        self.assertEqual(first.errors, second.errors)
        self.assertEqual(first.shots, second.shots)

    def test_python_decoder_is_rejected_by_default_collection(self) -> None:
        class CopyDecoder:
            def __init__(self) -> None:
                self.calls = 0

            def strong_id_payload(self):
                return {
                    "backend": "test-python",
                    "decoder": "copy",
                    "implementation_version": 1,
                    "parameters": {},
                }

            def decode_batch_masks(self, batch):
                self.calls += 1
                return {0: batch.detectors[0]}

        decoder = CopyDecoder()

        with self.assertRaisesRegex(TypeError, "native decoder"):
            _collect(
                [CollectionTask(dem=_graphlike_dem(), decoder=decoder)],
                max_shots=8,
                batch_size=8,
                seed=4,
            )
        self.assertEqual(decoder.calls, 0)

    def test_circuit_task_compiles_to_dem_sampler(self) -> None:
        noise = NoiseLocation("x0", BernoulliPauliNoise("X"), 1.0, (0,))
        circuit = Circuit(
            1,
            (
                Operation.noise(noise),
                Operation.measure(0, key="m0"),
                Operation.detector(("m0",), detector_id=0),
                Operation.observable_include(0, ("m0",)),
            ),
        )

        stats = _collect(
            [
                CollectionTask(
                    circuit=circuit,
                    task_id="circuit",
                    metadata={"kind": "one_qubit"},
                )
            ],
            max_shots=6,
            batch_size=4,
            seed=5,
        )[0]

        self.assertEqual(stats.task_id, "circuit")
        self.assertEqual(stats.metadata, {"kind": "one_qubit"})
        self.assertEqual(stats.shots, 6)
        self.assertEqual(stats.errors, 6)
        self.assertTrue(stats.strong_id)

    def test_iter_collect_yields_each_task(self) -> None:
        tasks = [
            CollectionTask(dem=_logical_edge_dem(), task_id="a"),
            CollectionTask(dem=_logical_edge_dem(), task_id="b"),
        ]

        stats = list(
            public_iter_collect(
                tasks,
                options=CollectionOptions(max_shots=3, batch_size=2),
                run_options=CollectionRunOptions(seed=6),
            )
        )

        self.assertEqual([stat.task_id for stat in stats], ["a", "b"])
        self.assertEqual([stat.errors for stat in stats], [3, 3])

    def test_collector_matches_functional_wrappers(self) -> None:
        tasks = [CollectionTask(dem=_logical_edge_dem(), task_id="collector")]
        options = CollectionOptions(max_shots=8, batch_size=2)
        run_options = CollectionRunOptions(seed=7, num_workers=2)

        direct = Collector(options=options, run_options=run_options).collect(tasks)
        wrapped = public_collect(tasks, options=options, run_options=run_options)
        final_items = list(Collector(options=options, run_options=run_options).iter_collect(tasks))
        progress_items = list(
            Collector(options=options, run_options=run_options).iter_progress(tasks)
        )

        self.assertEqual(
            [item.with_edits(seconds=0.0) for item in direct],
            [item.with_edits(seconds=0.0) for item in wrapped],
        )
        self.assertEqual(
            [item.with_edits(seconds=0.0) for item in final_items],
            [item.with_edits(seconds=0.0) for item in direct],
        )
        self.assertTrue(progress_items)
        self.assertTrue(all(isinstance(item, Progress) for item in progress_items))

    def test_collection_functions_have_consolidated_signatures(self) -> None:
        expected = ("tasks", "options", "run_options")

        self.assertEqual(tuple(inspect.signature(public_collect).parameters), expected)
        self.assertEqual(tuple(inspect.signature(public_iter_collect).parameters), expected)
        self.assertEqual(tuple(inspect.signature(public_iter_progress).parameters), expected)
        with self.assertRaises(TypeError):
            public_collect([CollectionTask(dem=_logical_edge_dem())], max_shots=1)

    def test_num_workers_keeps_result_order(self) -> None:
        tasks = [
            CollectionTask(dem=_logical_edge_dem(probability=0.25), task_id="a"),
            CollectionTask(dem=_logical_edge_dem(probability=0.5), task_id="b"),
        ]

        stats = _collect(tasks, max_shots=128, batch_size=16, seed=8, num_workers=2)

        self.assertEqual([stat.task_id for stat in stats], ["a", "b"])
        self.assertEqual([stat.shots for stat in stats], [128, 128])

    def test_num_workers_parallelizes_single_task_without_changing_stats(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(probability=0.375), task_id="single")

        serial = _collect([task], max_shots=256, batch_size=16, seed=123, num_workers=1)[0]
        parallel = _collect([task], max_shots=256, batch_size=16, seed=123, num_workers=4)[0]

        self.assertEqual(parallel.task_id, serial.task_id)
        self.assertEqual(parallel.shots, serial.shots)
        self.assertEqual(parallel.errors, serial.errors)
        self.assertEqual(parallel.discards, serial.discards)
        self.assertEqual(parallel.custom_counts, serial.custom_counts)

    def test_adaptive_num_workers_completes_single_task_with_exact_shot_cap(self) -> None:
        stats = _collect(
            [CollectionTask(dem=_logical_edge_dem(), task_id="adaptive-single")],
            max_shots=64,
            batch_size=8,
            start_batch_size=1,
            max_batch_size=8,
            max_batch_seconds=0.001,
            seed=125,
            num_workers=4,
        )[0]

        self.assertEqual(stats.shots, 64)
        self.assertEqual(stats.errors, 64)
        self.assertEqual(stats.discards, 0)

    def test_num_workers_parallelizes_multiple_tasks_without_changing_stats(self) -> None:
        tasks = [
            CollectionTask(dem=_logical_edge_dem(probability=0.25), task_id="a"),
            CollectionTask(dem=_logical_edge_dem(probability=0.375), task_id="b"),
            CollectionTask(dem=_logical_edge_dem(probability=0.5), task_id="c"),
        ]

        serial = _collect(tasks, max_shots=256, batch_size=16, seed=124, num_workers=1)
        parallel = _collect(tasks, max_shots=256, batch_size=16, seed=124, num_workers=4)

        self.assertEqual(
            [
                (stat.task_id, stat.shots, stat.errors, stat.discards, stat.custom_counts)
                for stat in parallel
            ],
            [
                (stat.task_id, stat.shots, stat.errors, stat.discards, stat.custom_counts)
                for stat in serial
            ],
        )

    def test_postselection_discards_detector_events(self) -> None:
        stats = _collect(
            [
                CollectionTask(
                    dem=_graphlike_dem(),
                    task_id="post",
                    postselection_mask=bytes([0b1]),
                )
            ],
            max_shots=8,
            batch_size=8,
            seed=9,
        )[0]

        self.assertEqual(stats.shots, 8)
        self.assertEqual(stats.discards, 8)
        self.assertEqual(stats.errors, 0)
        self.assertTrue(math.isnan(stats.logical_error_rate))

    def test_custom_counts_are_reported(self) -> None:
        stats = _collect(
            [CollectionTask(dem=_graphlike_dem(), task_id="counts")],
            max_shots=4,
            batch_size=4,
            seed=10,
            count_observable_error_combos=True,
            count_detection_events=True,
        )[0]

        self.assertEqual(stats.custom_counts["obs_mistake_mask=E"], 4)
        self.assertEqual(stats.custom_counts["detection_events"], 4)
        self.assertEqual(stats.custom_counts["detectors_checked"], 4)

    def test_detection_schema_preserves_fixed_zero_counters(self) -> None:
        stats = _collect(
            [CollectionTask(dem=_logical_edge_dem(0.0), task_id="zero-detection")],
            max_shots=4,
            batch_size=2,
            seed=10,
            count_detection_events=True,
        )[0]

        self.assertEqual(
            stats.custom_counts,
            {"detection_events": 0, "detectors_checked": 0},
        )
        self.assertEqual(
            stats.counter_schema,
            CollectionCounterSchema(count_detection_events=True),
        )

    def test_zero_error_limit_returns_and_persists_schema_complete_empty_stats(self) -> None:
        task = CollectionTask(dem=_graphlike_dem(), task_id="zero-limit")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "zero-limit.csv"
            fixed = _collect(
                [task],
                max_shots=8,
                max_errors=0,
                batch_size=4,
                seed=16,
                num_workers=4,
                save_resume_filepath=path,
                count_detection_events=True,
            )[0]
            adaptive = _collect(
                [task],
                max_shots=8,
                max_errors=0,
                batch_size=4,
                start_batch_size=1,
                max_batch_size=4,
                max_batch_seconds=0.01,
                seed=16,
                num_workers=4,
                count_detection_events=True,
            )[0]
            persisted = read_stats_from_csv_files(path)[0]

        for stats in (fixed, adaptive, persisted):
            self.assertEqual(stats.shots, 0)
            self.assertEqual(
                stats.custom_counts,
                {"detection_events": 0, "detectors_checked": 0},
            )
            self.assertEqual(
                stats.counter_schema,
                CollectionCounterSchema(count_detection_events=True),
            )

    def test_invalid_custom_stop_keys_fail_before_resume_write_even_without_max_errors(
        self,
    ) -> None:
        base = CollectionTask(dem=_two_observable_dem(), task_id="invalid-stop")
        cases = (
            ("typo", base, "detection_event", False, False),
            ("detection-disabled", base, "detection_events", False, False),
            ("combo-disabled", base, "obs_mistake_mask=E_", False, False),
            ("bad-prefix", base, "mistake_mask=E_", True, False),
            ("bad-width", base, "obs_mistake_mask=E", True, False),
            ("bad-character", base, "obs_mistake_mask=EX", True, False),
            ("empty", base, "obs_mistake_mask=", True, False),
            ("all-underscores", base, "obs_mistake_mask=__", True, False),
            (
                "postselected-error",
                CollectionTask(
                    dem=_two_observable_dem(),
                    task_id="postselected-stop",
                    postselected_observables_mask=b"\x01",
                ),
                "obs_mistake_mask=E_",
                True,
                False,
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, task, key, count_combos, count_detection in cases:
                path = Path(temp_dir) / f"{name}.csv"
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        _collect(
                            [task],
                            max_shots=4,
                            batch_size=2,
                            seed=17,
                            num_workers=4,
                            save_resume_filepath=path,
                            count_observable_error_combos=count_combos,
                            count_detection_events=count_detection,
                            custom_error_count_key=key,
                        )
                    self.assertFalse(path.exists())

    def test_valid_absent_combo_is_zero_and_fixed_detection_keys_can_stop(self) -> None:
        combo_stats = _collect(
            [CollectionTask(dem=_two_observable_dem(), task_id="zero-combo")],
            max_shots=5,
            max_errors=1,
            batch_size=2,
            seed=18,
            count_observable_error_combos=True,
            custom_error_count_key="obs_mistake_mask=_E",
        )[0]
        detector_stats = _collect(
            [CollectionTask(dem=_graphlike_dem(), task_id="detector-stop")],
            max_shots=10,
            max_errors=3,
            batch_size=2,
            seed=18,
            count_detection_events=True,
            custom_error_count_key="detectors_checked",
        )[0]

        self.assertEqual(combo_stats.shots, 5)
        self.assertNotIn("obs_mistake_mask=_E", combo_stats.custom_counts)
        self.assertEqual(detector_stats.shots, 4)
        self.assertEqual(detector_stats.custom_counts["detectors_checked"], 4)

    def test_implicit_dem_ids_match_fast_detailed_and_postselection_paths(self) -> None:
        dem = DetectorErrorModel(
            detectors=(),
            observables=(),
            edges=(DetectorErrorEdge(1.0, (7,), (9,), "implicit", "X"),),
        )

        fast = _collect(
            [CollectionTask(dem=dem, task_id="fast")],
            max_shots=4,
            batch_size=4,
            seed=11,
        )[0]
        detailed = _collect(
            [CollectionTask(dem=dem, task_id="detailed")],
            max_shots=4,
            batch_size=4,
            seed=11,
            count_observable_error_combos=True,
            count_detection_events=True,
        )[0]
        selected = _collect(
            [
                CollectionTask(
                    dem=dem,
                    task_id="selected",
                    postselection_mask=bytes([1]),
                )
            ],
            max_shots=4,
            batch_size=4,
            seed=11,
            count_observable_error_combos=True,
            count_detection_events=True,
        )[0]

        self.assertEqual((fast.errors, fast.discards), (4, 0))
        self.assertEqual((detailed.errors, detailed.discards), (4, 0))
        self.assertEqual(detailed.custom_counts["obs_mistake_mask=E"], 4)
        self.assertEqual(detailed.custom_counts["detection_events"], 4)
        self.assertEqual(detailed.custom_counts["detectors_checked"], 4)
        self.assertEqual((selected.errors, selected.discards), (0, 4))

    def test_save_resume_filepath_skips_completed_task(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="resume")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            first = _collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=11,
                save_resume_filepath=path,
            )[0]
            second = _collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=11,
                save_resume_filepath=path,
            )[0]

        self.assertEqual(first.shots, 6)
        self.assertEqual(second.shots, 6)
        self.assertEqual(second.errors, 6)

    def test_switching_counter_schema_collects_a_fresh_full_sample(self) -> None:
        task = CollectionTask(dem=_graphlike_dem(0.5), task_id="schema-switch")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "schema-switch.csv"
            plain = _collect(
                [task],
                max_shots=12,
                batch_size=4,
                seed=44,
                save_resume_filepath=path,
            )[0]
            detection = _collect(
                [task],
                max_shots=12,
                batch_size=4,
                seed=44,
                save_resume_filepath=path,
                count_detection_events=True,
            )[0]
            persisted = read_stats_from_csv_files(path)

        self.assertNotEqual(plain.strong_id, detection.strong_id)
        self.assertEqual((plain.shots, detection.shots), (12, 12))
        self.assertEqual(plain.errors, detection.errors)
        self.assertEqual(len(persisted), 2)
        self.assertEqual(
            detection.counter_schema,
            CollectionCounterSchema(count_detection_events=True),
        )

    def test_matching_resume_identity_rejects_missing_schema_before_native_collection(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="missing-schema")
        collected = _collect([task], max_shots=4, batch_size=4, seed=45)[0]
        forged = collected.with_edits(counter_schema=None)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-schema.csv"
            write_stats_to_csv_file(path, [forged])
            with mock.patch.object(
                collection_collect_module,
                "_collect_dem_logical_error_stats_many",
            ) as native_collect:
                with self.assertRaisesRegex(ValueError, "counter schema"):
                    _collect(
                        [task],
                        max_shots=4,
                        batch_size=4,
                        seed=45,
                        existing_data_filepaths=(path,),
                    )
                native_collect.assert_not_called()

    def test_csv_rejects_wrong_schema_version_missing_fixed_counter_and_mixed_schema(self) -> None:
        plain_schema = CollectionCounterSchema()
        detection_schema = CollectionCounterSchema(count_detection_events=True)
        plain = TaskStats(
            task_id="plain",
            strong_id="shared",
            shots=1,
            errors=0,
            discards=0,
            seconds=0.0,
            decoder=None,
            metadata={},
            custom_counts={},
            counter_schema=plain_schema,
        )
        detection = plain.with_edits(
            custom_counts={"detection_events": 0, "detectors_checked": 0},
            counter_schema=detection_schema,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            wrong_version = base / "wrong-version.csv"
            wrong_version_row = detection.to_csv_row()
            schema_payload = json.loads(wrong_version_row["json_counter_schema"])
            schema_payload["schema_version"] = 999
            wrong_version_row["json_counter_schema"] = json.dumps(schema_payload)
            with wrong_version.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLLECTION_CSV_FIELDS)
                writer.writeheader()
                writer.writerow(wrong_version_row)

            missing_fixed = base / "missing-fixed.csv"
            missing_fixed_row = detection.to_csv_row()
            missing_fixed_row["custom_counts"] = "{}"
            with missing_fixed.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLLECTION_CSV_FIELDS)
                writer.writeheader()
                writer.writerow(missing_fixed_row)

            mixed = base / "mixed.csv"
            write_stats_to_csv_file(mixed, [plain, detection])

            with self.assertRaisesRegex(ValueError, "unsupported collection counter schema"):
                read_stats_from_csv_files(wrong_version)
            with self.assertRaisesRegex(ValueError, "must contain"):
                read_stats_from_csv_files(missing_fixed)
            with self.assertRaisesRegex(ValueError, "counter schemas"):
                read_stats_from_csv_files(mixed)

    def test_resume_file_header_and_second_run_appends_no_duplicate_rows(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="resume-header")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            _collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=11,
                save_resume_filepath=path,
            )
            with open(path, newline="") as f:
                first_rows = list(csv.reader(f))

            _collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=999,
                save_resume_filepath=path,
            )
            with open(path, newline="") as f:
                second_rows = list(csv.reader(f))

        self.assertEqual(
            first_rows[0],
            [
                "shots",
                "errors",
                "discards",
                "seconds",
                "decoder",
                "strong_id",
                "json_metadata",
                "json_counter_schema",
                "custom_counts",
            ],
        )
        self.assertEqual(len(first_rows), 2)
        self.assertEqual(second_rows, first_rows)

    def test_existing_data_filepaths_count_towards_limits(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="existing")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            _collect([task], max_shots=5, batch_size=5, seed=12, save_resume_filepath=path)
            stats = _collect(
                [task],
                max_shots=5,
                batch_size=5,
                seed=999,
                existing_data_filepaths=[path],
            )[0]

        self.assertEqual(stats.shots, 5)
        self.assertEqual(stats.errors, 5)

    def test_v1_strong_id_rows_are_not_resumed(self) -> None:
        dem = DetectorErrorModel(
            detectors=(),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=1.0,
                    detectors=(),
                    observables=(0,),
                    location_id="logical",
                    event="L",
                ),
            ),
        )
        metadata = {"d": 3, "p": 0.01}
        legacy_id = "4875ec4d3a1fbc3fc3bbc4a769d7b56c14ea84583af5cab01860c6d4b8793bd4"
        legacy = TaskStats(
            task_id=legacy_id,
            strong_id=legacy_id,
            shots=100,
            errors=100,
            discards=0,
            seconds=1.0,
            decoder=None,
            metadata=metadata,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "v1.csv"
            write_stats_to_csv_file(path, [legacy])
            stats = _collect(
                [CollectionTask(dem=dem, metadata=metadata)],
                max_shots=3,
                batch_size=3,
                seed=25,
                existing_data_filepaths=(path,),
            )[0]

        self.assertEqual(stats.shots, 3)
        self.assertEqual(stats.errors, 3)
        self.assertNotEqual(stats.strong_id, legacy_id)

    def test_resume_file_does_not_duplicate_parallel_collection(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="parallel-resume")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            first = _collect(
                [task],
                max_shots=8,
                batch_size=4,
                seed=15,
                num_workers=4,
                save_resume_filepath=path,
            )[0]
            second = _collect(
                [task],
                max_shots=8,
                batch_size=4,
                seed=999,
                num_workers=4,
                save_resume_filepath=path,
            )[0]

        self.assertEqual(first.shots, 8)
        self.assertEqual(second.shots, 8)
        self.assertEqual(second.errors, 8)

    def test_adaptive_stream_interruption_resumes_from_committed_calibration_delta(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="adaptive-stream-resume")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-resume.csv"
            options = CollectionOptions(
                max_shots=10,
                batch_size=4,
                start_batch_size=1,
                max_batch_size=4,
                max_batch_seconds=1.0,
            )
            run_options = CollectionRunOptions(
                seed=126,
                num_workers=4,
                save_resume_filepath=path,
            )
            iterator = public_iter_progress(
                [task],
                options=options,
                run_options=run_options,
            )
            first = next(iterator)
            iterator.close()
            partial = read_stats_from_csv_files(path)[0]
            resumed = _collect(
                [task],
                max_shots=10,
                batch_size=4,
                start_batch_size=1,
                max_batch_size=4,
                max_batch_seconds=1.0,
                seed=126,
                num_workers=4,
                save_resume_filepath=path,
            )[0]

        self.assertEqual(first.new_stats[0].shots, 1)
        self.assertEqual(partial.shots, 1)
        self.assertEqual(partial.errors, 1)
        self.assertEqual(resumed.shots, 10)
        self.assertEqual(resumed.errors, 10)

    def test_adaptive_parallel_collection_uses_custom_stop_counter(self) -> None:
        stats = _collect(
            [CollectionTask(dem=_graphlike_dem(), task_id="adaptive-custom-stop")],
            max_shots=20,
            max_errors=6,
            batch_size=4,
            start_batch_size=1,
            max_batch_size=4,
            max_batch_seconds=1.0,
            seed=127,
            num_workers=4,
            count_detection_events=True,
            custom_error_count_key="detection_events",
        )[0]

        self.assertEqual(stats.shots, 9)
        self.assertEqual(stats.custom_counts["detection_events"], 9)

    def test_invalid_stop_key_is_consistent_for_serial_parallel_adaptive_and_progress(self) -> None:
        task = CollectionTask(dem=_graphlike_dem(), task_id="invalid-paths")
        fixed = CollectionOptions(max_shots=4, batch_size=2)
        adaptive = CollectionOptions(
            max_shots=4,
            batch_size=2,
            start_batch_size=1,
            max_batch_size=2,
            max_batch_seconds=0.01,
        )
        for name, options, workers in (
            ("serial", fixed, 1),
            ("parallel", fixed, 4),
            ("adaptive", adaptive, 4),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "requires count_detection_events"):
                    public_collect(
                        [task],
                        options=options,
                        run_options=CollectionRunOptions(
                            seed=19,
                            num_workers=workers,
                            custom_error_count_key="detectors_checked",
                        ),
                    )

        with self.assertRaisesRegex(ValueError, "requires count_detection_events"):
            list(
                public_iter_progress(
                    [task],
                    options=fixed,
                    run_options=CollectionRunOptions(
                        seed=19,
                        custom_error_count_key="detectors_checked",
                    ),
                )
            )

    def test_iter_progress_receives_incremental_stats(self) -> None:
        progress = list(
            public_iter_progress(
                [CollectionTask(dem=_logical_edge_dem(), task_id="progress")],
                options=CollectionOptions(max_shots=4, batch_size=4),
                run_options=CollectionRunOptions(seed=13),
            )
        )

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].new_stats[0].task_id, "progress")
        self.assertEqual(progress[0].new_stats[0].shots, 4)

    def test_progress_api_annotations_are_disjoint(self) -> None:
        iter_hints = get_type_hints(collection_collect_module.iter_collect)
        progress_hints = get_type_hints(collection_collect_module.iter_progress)

        self.assertEqual(iter_hints["return"], Iterator[TaskStats])
        self.assertEqual(progress_hints["return"], Iterator[Progress])

    def test_stream_progress_yields_batch_deltas_and_writes_resume_rows(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="stream")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream.csv"
            progress = list(
                public_iter_progress(
                    [task],
                    options=CollectionOptions(max_shots=10, batch_size=4),
                    run_options=CollectionRunOptions(
                        seed=21,
                        save_resume_filepath=path,
                    ),
                )
            )
            with path.open(newline="") as f:
                rows = list(csv.reader(f))

        self.assertTrue(all(isinstance(item, Progress) for item in progress))
        self.assertEqual(
            [item.new_stats[0].shots for item in progress],
            [4, 4, 2],
        )
        self.assertEqual([item.new_stats[0].errors for item in progress], [4, 4, 2])
        self.assertEqual(rows[0], list(COLLECTION_CSV_FIELDS))
        self.assertEqual(len(rows), 4)

    def test_stream_detection_deltas_keep_fixed_zero_counters(self) -> None:
        progress = list(
            public_iter_progress(
                [CollectionTask(dem=_logical_edge_dem(0.0), task_id="stream-zero")],
                options=CollectionOptions(max_shots=3, batch_size=2),
                run_options=CollectionRunOptions(
                    seed=22,
                    count_detection_events=True,
                ),
            )
        )

        self.assertEqual([item.new_stats[0].shots for item in progress], [2, 1])
        self.assertTrue(
            all(
                item.new_stats[0].custom_counts == {"detection_events": 0, "detectors_checked": 0}
                for item in progress
            )
        )

    def test_stream_persists_before_yield(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="ordered-stream")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ordered.csv"
            iterator = public_iter_progress(
                [task],
                options=CollectionOptions(max_shots=1, batch_size=1),
                run_options=CollectionRunOptions(
                    seed=31,
                    save_resume_filepath=path,
                ),
            )
            try:
                progress = next(iterator)
                with path.open(newline="") as f:
                    rows = list(csv.reader(f))
            finally:
                iterator.close()

        self.assertEqual(rows[0], list(COLLECTION_CSV_FIELDS))
        self.assertEqual(len(rows), 2)
        self.assertEqual(progress.new_stats[0].shots, 1)

    def test_stream_resume_survives_iterator_interruption(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="interrupt")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "resume.csv"
            iterator = public_iter_progress(
                [task],
                options=CollectionOptions(max_shots=6, batch_size=3),
                run_options=CollectionRunOptions(
                    seed=22,
                    save_resume_filepath=path,
                ),
            )
            first = next(iterator)
            iterator.close()
            partial = read_stats_from_csv_files(path)[0]
            resumed = _collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=22,
                save_resume_filepath=path,
            )[0]

        self.assertEqual(first.new_stats[0].shots, 3)
        self.assertEqual(partial.shots, 3)
        self.assertEqual(resumed.shots, 6)
        self.assertEqual(resumed.errors, 6)

    def test_stream_progress_objects_are_not_retained_after_next_item(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="retention")

        def native_item(shots: int) -> dict[str, object]:
            return {
                "task_id": "retention",
                "strong_id": "retention-strong",
                "shots": shots,
                "errors": shots,
                "discards": 0,
                "seconds": 0.0,
                "decoder": None,
                "metadata": {},
                "counter_schema": _native_counter_schema(),
                "custom_counts": {},
            }

        def fake_native_collect(*args: object, **kwargs: object) -> list[dict[str, object]]:
            progress_callback = kwargs["progress_callback"]
            assert callable(progress_callback)
            progress_callback(native_item(1))
            gc.collect()
            progress_callback(native_item(1))
            return [native_item(2)]

        with mock.patch.object(
            collection_collect_module,
            "_collect_dem_logical_error_stats_many",
            side_effect=fake_native_collect,
        ):
            iterator = public_iter_progress(
                [task],
                options=CollectionOptions(max_shots=2, batch_size=1),
            )
            first = next(iterator)
            first_ref = weakref.ref(first)
            del first
            second = next(iterator)
            gc.collect()
            self.assertIsNone(first_ref())
            self.assertEqual(second.new_stats[0].shots, 1)
            iterator.close()

    def test_stream_close_while_yielded_cancels_before_future_callbacks(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="close-race")
        producer_returned_from_first = threading.Event()
        real_condition = threading.Condition

        def native_item(shots: int) -> dict[str, object]:
            return {
                "task_id": "close-race",
                "strong_id": "close-race-strong",
                "shots": shots,
                "errors": shots,
                "discards": 0,
                "seconds": 0.0,
                "decoder": None,
                "metadata": {},
                "counter_schema": _native_counter_schema(),
                "custom_counts": {},
            }

        class RaceCondition:
            def __init__(self, lock: object | None = None) -> None:
                self._condition = (
                    real_condition() if lock is None else real_condition(lock)  # type: ignore[arg-type]
                )
                self._notify_count = 0
                self._pause_on_exit = False

            def __enter__(self) -> "RaceCondition":
                self._condition.__enter__()
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
                result = self._condition.__exit__(exc_type, exc, tb)
                if self._pause_on_exit:
                    producer_returned_from_first.wait(timeout=0.2)
                    self._pause_on_exit = False
                return result

            def notify_all(self) -> None:
                self._notify_count += 1
                self._condition.notify_all()
                if self._notify_count == 2:
                    self._pause_on_exit = True

            def wait(self, timeout: float | None = None) -> bool:
                return self._condition.wait(timeout)

        def fake_native_collect(*args: object, **kwargs: object) -> list[dict[str, object]]:
            progress_callback = kwargs["progress_callback"]
            assert callable(progress_callback)
            items = [native_item(1), native_item(2)]
            progress_callback(items[0])
            producer_returned_from_first.set()
            progress_callback(items[1])
            return items

        with (
            mock.patch.object(
                collection_collect_module,
                "_collect_dem_logical_error_stats_many",
                side_effect=fake_native_collect,
            ),
            mock.patch.object(
                collection_collect_module.threading,
                "Condition",
                RaceCondition,
            ),
        ):
            iterator = public_iter_progress(
                [task],
                options=CollectionOptions(max_shots=2, batch_size=1),
            )
            first = next(iterator)
            self.assertEqual(first.new_stats[0].shots, 1)
            iterator.close()

    def test_stream_close_cancels_and_joins_producer(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="close-stream")
        created_threads: list[threading.Thread] = []
        real_thread = threading.Thread

        def native_item(shots: int) -> dict[str, object]:
            return {
                "task_id": "close-stream",
                "strong_id": "close-stream-strong",
                "shots": shots,
                "errors": shots,
                "discards": 0,
                "seconds": 0.0,
                "decoder": None,
                "metadata": {},
                "counter_schema": _native_counter_schema(),
                "custom_counts": {},
            }

        def fake_native_collect(*args: object, **kwargs: object) -> list[dict[str, object]]:
            progress_callback = kwargs["progress_callback"]
            assert callable(progress_callback)
            items = [native_item(1), native_item(2), native_item(3)]
            for item in items:
                progress_callback(item)
            return items

        def make_thread(*args: object, **kwargs: object) -> threading.Thread:
            thread = real_thread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        with (
            mock.patch.object(
                collection_collect_module,
                "_collect_dem_logical_error_stats_many",
                side_effect=fake_native_collect,
            ),
            mock.patch.object(
                collection_collect_module.threading,
                "Thread",
                side_effect=make_thread,
            ),
        ):
            iterator = public_iter_progress(
                [task],
                options=CollectionOptions(max_shots=3, batch_size=1),
            )
            first = next(iterator)
            self.assertEqual(first.new_stats[0].shots, 1)
            iterator.close()
            self.assertEqual(len(created_threads), 1)
            created_threads[0].join(timeout=0.2)
            self.assertFalse(created_threads[0].is_alive())

    def test_private_progress_sink_returns_final_stats(self) -> None:
        progress: list[Progress] = []
        collector = Collector(
            options=CollectionOptions(max_shots=4, batch_size=4),
            run_options=CollectionRunOptions(seed=14),
        )

        stats = collector._collect_with_progress(
            [CollectionTask(dem=_logical_edge_dem(), task_id="printed")],
            progress.append,
        )

        self.assertEqual(stats[0].shots, 4)
        self.assertEqual(progress[0].status_message, "printed: shots=4 errors=4 discards=0")

    def test_decoder_fanout_expands_tasks_in_order(self) -> None:
        dem = _graphlike_dem()
        no_correction = NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(0,))

        stats = _collect(
            [CollectionTask(dem=dem, task_id="fanout")],
            max_shots=8,
            batch_size=4,
            seed=23,
            decoders=[no_correction, "graphlike-detector-copy"],
        )

        self.assertEqual(
            [(stat.task_id, stat.decoder, stat.errors) for stat in stats],
            [
                ("fanout:no-correction", "no-correction", 8),
                ("fanout:graphlike-detector-copy", "graphlike-detector-copy", 0),
            ],
        )

    def test_analysis_helpers_fit_and_predict_rates(self) -> None:
        stats = [
            TaskStats("a", 100, 10, 0, 0.1, "native", {"p": 0.1, "d": 3}, "a"),
            TaskStats("b", 100, 20, 0, 0.1, "native", {"p": 0.2, "d": 3}, "b"),
            TaskStats("c", 100, 5, 0, 0.1, "native", {"p": 0.1, "d": 5}, "c"),
            TaskStats("d", 100, 10, 0, 0.1, "native", {"p": 0.2, "d": 5}, "d"),
        ]

        points = error_rate_points(stats, x_key="p", group_key="d")
        fits = fit_log_error_rate_lines(points, x_key="p", group_key="d")

        self.assertEqual({point["group"] for point in points}, {3, 5})
        self.assertEqual(set(fits), {3, 5})
        self.assertGreater(predict_error_rate(fits[3], 0.15), 0.0)

    def test_error_rate_points_custom_count_is_strict_and_binomial(self) -> None:
        stats = TaskStats(
            "a",
            100,
            20,
            10,
            0.1,
            "native",
            {"p": 0.1},
            "a",
            {"logical_x": 18, "detection_events": 120},
        )

        (point,) = error_rate_points([stats], x_key="p", count_key="logical_x")
        self.assertEqual(point["shots"], 90)
        self.assertEqual(point["errors"], 18)
        self.assertEqual(point["rate"], 0.2)

        with self.assertRaisesRegex(ValueError, "missing custom count"):
            error_rate_points([stats], x_key="p", count_key="typo")
        with self.assertRaisesRegex(ValueError, "accepted shots"):
            error_rate_points([stats], x_key="p", count_key="detection_events")

    def test_collection_cli_commands_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            factory = temp / "factory.py"
            collected = temp / "collected.csv"
            merged = temp / "merged.csv"
            plot = temp / "plot.png"
            factory.write_text(
                "from faultscope import DetectorErrorEdge, DetectorErrorModel, LogicalObservable\n"
                "from faultscope.collection import CollectionTask\n"
                "def make_task(probability=1.0, distance=3, enabled=True, config=None):\n"
                "    assert isinstance(probability, float), type(probability)\n"
                "    assert probability == 1.0, probability\n"
                "    assert isinstance(distance, int) and not isinstance(distance, bool), type(distance)\n"
                "    assert distance == 3, distance\n"
                "    assert enabled is True, enabled\n"
                "    assert isinstance(config, dict), type(config)\n"
                "    assert config == {'label': 'smoke'}, config\n"
                "    dem = DetectorErrorModel(detectors=(), observables=(LogicalObservable(0),), edges=(DetectorErrorEdge(probability, (), (0,), 'edge', 'L'),))\n"
                "    return CollectionTask(dem=dem, task_id='cli-single', metadata={'p': probability, 'd': distance, 'enabled': enabled, 'config': config})\n"
            )

            result = self._run_collection_cli(
                temp,
                "collect",
                "--tasks-factory",
                f"{factory}:make_task",
                "--factory-arg",
                "probability=1.0",
                "--factory-arg",
                "distance=3",
                "--factory-arg",
                "enabled=true",
                "--factory-arg",
                'config={"label":"smoke"}',
                "--max-shots",
                "4",
                "--batch-size",
                "2",
                "--save-resume-filepath",
                str(collected),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(collected.exists())
            self.assertIn("collected 1 task(s)", result.stderr)

            result = self._run_collection_cli(temp, "summarize", str(collected))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("shots", result.stdout)
            self.assertIn('"d": 3', result.stdout)
            self.assertIn('"enabled": true', result.stdout)
            self.assertIn('"config": {"label": "smoke"}', result.stdout)

            result = self._run_collection_cli(temp, "merge", str(merged), str(collected))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(merged.exists())

            result = self._run_collection_cli(
                temp,
                "fit",
                str(collected),
                "--x-key",
                "p",
                "--group-key",
                "d",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("group\tslope\tintercept\tpoints", result.stdout)
            self.assertIn("3\t", result.stdout)

            result = self._run_collection_cli(
                temp,
                "plot",
                str(collected),
                "--x-key",
                "p",
                "--group-key",
                "d",
                "--out",
                str(plot),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(plot.exists())
            self.assertGreater(plot.stat().st_size, 0)

    def test_collection_factory_assertions_catch_non_json_decoded_args(self) -> None:
        from faultscope.collection import __main__ as collection_main

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            factory = temp / "factory.py"
            collected = temp / "collected.csv"
            factory.write_text(
                "from faultscope import DetectorErrorEdge, DetectorErrorModel, LogicalObservable\n"
                "from faultscope.collection import CollectionTask\n"
                "def make_task(probability=1.0, distance=3, enabled=True, config=None):\n"
                "    assert isinstance(probability, float), type(probability)\n"
                "    assert probability == 1.0, probability\n"
                "    assert isinstance(distance, int) and not isinstance(distance, bool), type(distance)\n"
                "    assert distance == 3, distance\n"
                "    assert enabled is True, enabled\n"
                "    assert isinstance(config, dict), type(config)\n"
                "    assert config == {'label': 'smoke'}, config\n"
                "    dem = DetectorErrorModel(detectors=(), observables=(LogicalObservable(0),), edges=(DetectorErrorEdge(probability, (), (0,), 'edge', 'L'),))\n"
                "    return CollectionTask(dem=dem, task_id='cli-single', metadata={'p': probability, 'd': distance, 'enabled': enabled, 'config': config})\n"
            )
            args = collection_main._parser().parse_args(
                [
                    "collect",
                    "--tasks-factory",
                    f"{factory}:make_task",
                    "--factory-arg",
                    "probability=1.0",
                    "--factory-arg",
                    "distance=3",
                    "--factory-arg",
                    "enabled=true",
                    "--factory-arg",
                    'config={"label":"smoke"}',
                    "--max-shots",
                    "4",
                    "--batch-size",
                    "2",
                    "--save-resume-filepath",
                    str(collected),
                ]
            )

            with mock.patch.object(
                collection_main,
                "_factory_kwargs",
                return_value={
                    "probability": "1.0",
                    "distance": "3",
                    "enabled": "true",
                    "config": '{"label":"smoke"}',
                },
            ):
                with self.assertRaises(AssertionError):
                    collection_main._cmd_collect(args)

    def test_plot_error_rates_missing_matplotlib_has_install_hint(self) -> None:
        original_import = builtins.__import__

        def import_hook(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("mocked missing matplotlib")
            return original_import(name, globals, locals, fromlist, level)

        stats = [TaskStats("plot", 10, 1, 0, 0.1, None, {"p": 0.1}, "plot")]

        with mock.patch("builtins.__import__", side_effect=import_hook):
            with self.assertRaisesRegex(
                ImportError,
                r"faultscope\[collection\]",
            ):
                plot_error_rates(stats, x_key="p")
        self.assertIs(builtins.__import__, original_import)

    def test_collection_throughput_benchmark_parser_is_importable(self) -> None:
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "collection_throughput.py"
        spec = importlib.util.spec_from_file_location("collection_throughput_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            parser = module._build_parser()
            args = parser.parse_args(
                [
                    "--shots",
                    "10",
                    "--batch-size",
                    "5",
                    "--workers",
                    "1",
                    "2",
                    "--repeats",
                    "1",
                ]
            )
        finally:
            sys.modules.pop(spec.name, None)

        self.assertEqual(args.shots, 10)
        self.assertEqual(args.batch_size, 5)
        self.assertEqual(args.workers, [1, 2])
        self.assertEqual(args.repeats, 1)

    def test_public_exports(self) -> None:
        self.assertIs(faultscope.Collector, Collector)
        self.assertIs(faultscope.CollectionTask, CollectionTask)
        self.assertIs(faultscope.CollectionOptions, CollectionOptions)
        self.assertIs(faultscope.CollectionRunOptions, CollectionRunOptions)
        self.assertIs(faultscope.Progress, Progress)
        self.assertIs(faultscope.TaskStats, TaskStats)
        self.assertIs(faultscope.collect, public_collect)
        self.assertIs(faultscope.iter_collect, public_iter_collect)
        self.assertIs(faultscope.iter_progress, public_iter_progress)

    def test_collection_exports_threshold_api_without_top_level_faultscope_exports(self) -> None:
        import faultscope.collection as collection
        import faultscope.collection.threshold as threshold

        for name in (
            "FiniteSizeScalingFit",
            "PairwiseCrossing",
            "ThresholdAnalysisResult",
            "ThresholdEstimate",
            "ThresholdPoint",
            "analyze_thresholds",
            "plot_threshold_analysis",
        ):
            self.assertIs(getattr(collection, name), getattr(threshold, name))
            self.assertFalse(hasattr(faultscope, name))

    def test_threshold_cli_text_outputs_stable_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = self._write_threshold_cli_stats(temp)

            result = self._run_collection_cli(
                temp,
                "threshold",
                str(stats_path),
                "--x-key",
                "p",
                "--distance-key",
                "d",
                "--bootstrap-samples",
                "0",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(
            lines[0],
            "series\tkind\tlower_distance\tupper_distance\tstatus\testimate\tci_low\tci_high\tbootstrap_samples\tbootstrap_successes\tnu\tnu_ci_low\tnu_ci_high\tnu_bootstrap_samples\tnu_bootstrap_successes\tmessage",
        )
        pairwise_rows = [line.split("\t") for line in lines[1:] if "\tpairwise\t" in line]
        self.assertTrue(pairwise_rows)
        self.assertTrue(all(row[10:15] == ["", "", "", "0", "0"] for row in pairwise_rows))
        self.assertTrue(
            any("\tpairwise\t3\t5\tok\t0.0203158395151\t" in line for line in lines[1:])
        )
        self.assertTrue(
            any("\tpairwise_global\t\t\tok\t0.0203158395151\t" in line for line in lines[1:])
        )
        self.assertTrue(
            any(
                "\tscaling_global\t\t\tinsufficient_data\t\t\t\t0\t0\t\t\t\t0\t0\t" in line
                for line in lines[1:]
            )
        )

    def test_threshold_cli_text_scaling_row_includes_critical_exponent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = self._write_scaling_threshold_cli_stats(temp)

            result = self._run_collection_cli(
                temp,
                "threshold",
                str(stats_path),
                "--x-key",
                "p",
                "--distance-key",
                "d",
                "--bootstrap-samples",
                "0",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        scaling_row = next(
            row.split("\t")
            for row in result.stdout.strip().splitlines()[1:]
            if "\tscaling_global\t" in row
        )
        self.assertEqual(scaling_row[4], "ok")
        self.assertNotEqual(scaling_row[5], "")
        self.assertNotEqual(scaling_row[10], "")
        self.assertEqual(scaling_row[11:15], ["", "", "0", "0"])

    def test_threshold_cli_json_outputs_result_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = self._write_threshold_cli_stats(temp)

            result = self._run_collection_cli(
                temp,
                "threshold",
                str(stats_path),
                "--x-key",
                "p",
                "--distance-key",
                "d",
                "--bootstrap-samples",
                "0",
                "--format",
                "json",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["crossings"][0]["status"], "ok")
        self.assertAlmostEqual(payload[0]["pairwise_threshold"]["value"], 0.0203158395151)

    def test_threshold_cli_json_preserves_repeated_series_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = self._write_threshold_cli_stats(temp)

            result = self._run_collection_cli(
                temp,
                "threshold",
                str(stats_path),
                "--x-key",
                "p",
                "--distance-key",
                "d",
                "--series-key",
                "family",
                "--series-key",
                "decoder",
                "--bootstrap-samples",
                "0",
                "--format",
                "json",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(list(payload[0]["series"]), ["family", "decoder"])

    def test_threshold_cli_multi_series_preserves_repeated_series_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = self._write_threshold_cli_stats(temp)

            result = self._run_collection_cli(
                temp,
                "threshold",
                str(stats_path),
                "--x-key",
                "p",
                "--distance-key",
                "d",
                "--series-key",
                "family",
                "--series-key",
                "decoder",
                "--bootstrap-samples",
                "0",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        rows = result.stdout.strip().splitlines()[1:]
        series_cells = [row.split("\t", 1)[0] for row in rows]
        self.assertTrue(all(cell.startswith("family=rotated,decoder=") for cell in series_cells))
        self.assertTrue(any("decoder=bposd" in cell for cell in series_cells))
        self.assertTrue(any("decoder=mwpm" in cell for cell in series_cells))

    def test_threshold_cli_writes_plot_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stats_path = self._write_threshold_cli_stats(temp)
            output = temp / "threshold.png"

            result = self._run_collection_cli(
                temp,
                "threshold",
                str(stats_path),
                "--x-key",
                "p",
                "--distance-key",
                "d",
                "--bootstrap-samples",
                "0",
                "--plot-out",
                str(output),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
