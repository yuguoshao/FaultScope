import math
import io
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr

import faultscope
from faultscope import (
    Circuit,
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    LogicalObservable,
    NativeGraphlikeDetectorCopyDecoder,
    Operation,
    BernoulliPauliNoise,
    NoiseLocation,
)
from faultscope.collection import (
    CollectionOptions,
    CollectionTask,
    TaskStats,
    collect,
    iter_collect,
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


class CollectionTests(unittest.TestCase):
    def test_options_validate_positive_limits(self) -> None:
        CollectionOptions(max_shots=1, max_errors=0, batch_size=1, seed=123)
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

    def test_task_requires_exactly_one_source(self) -> None:
        dem = _logical_edge_dem()
        CollectionTask(dem=dem)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            CollectionTask()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            CollectionTask(dem=dem, circuit=Circuit(1, (Operation.measure(0, key="m0"),)))

    def test_task_stats_rates_match_binomial_formula(self) -> None:
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

        self.assertEqual(stats.error_rate, 0.25)
        self.assertEqual(stats.logical_error_rate, 0.25)
        self.assertEqual(stats.stderr, math.sqrt(0.25 * 0.75 / 100))
        self.assertEqual(stats.accepted_shots, 100)

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

        self.assertEqual(stats.error_rate, 0.1)
        self.assertEqual(stats.logical_error_rate, 0.125)
        self.assertEqual(stats.stderr, math.sqrt(0.125 * 0.875 / 80))

    def test_collect_reports_all_logical_edge_failures(self) -> None:
        stats = collect(
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
        self.assertEqual(stats[0].error_rate, 1.0)
        self.assertTrue(stats[0].strong_id)

    def test_native_graphlike_decoder_can_remove_all_failures(self) -> None:
        dem = _graphlike_dem()
        decoder = NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

        stats = collect(
            [CollectionTask(dem=dem, decoder=decoder, task_id="corrected")],
            max_shots=32,
            batch_size=8,
            seed=2,
        )[0]

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(stats.shots, 32)
        self.assertEqual(stats.errors, 0)

    def test_max_errors_stops_after_completed_batch(self) -> None:
        stats = collect(
            [CollectionTask(dem=_logical_edge_dem())],
            max_shots=10,
            max_errors=1,
            batch_size=4,
            seed=3,
        )[0]

        self.assertEqual(stats.shots, 4)
        self.assertEqual(stats.errors, 4)

    def test_task_options_do_not_reset_call_batch_size_by_default(self) -> None:
        stats = collect(
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
        with mock.patch(
            "faultscope.collection._collect._compile_task_sampler",
            return_value=(sampler, dem),
        ), mock.patch(
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
                    "custom_counts": {},
                }
            ],
        ) as collect_native:
            stats = collect(
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
        self.assertEqual(native_tasks[0]["seed"], 9)
        self.assertEqual(stats.task_id, "native-core")
        self.assertEqual(stats.shots, 7)
        self.assertEqual(stats.errors, 3)
        self.assertEqual(stats.seconds, 0.25)

    def test_same_seed_repeats_probabilistic_collection(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(probability=0.375))

        first = collect([task], max_shots=256, batch_size=64, seed=123)[0]
        second = collect([task], max_shots=256, batch_size=64, seed=123)[0]

        self.assertEqual(first.errors, second.errors)
        self.assertEqual(first.shots, second.shots)

    def test_python_decoder_is_rejected_by_default_collection(self) -> None:
        class CopyDecoder:
            def __init__(self) -> None:
                self.calls = 0

            def decode_batch_masks(self, batch):
                self.calls += 1
                return {0: batch.detectors[0]}

        decoder = CopyDecoder()

        with self.assertRaisesRegex(TypeError, "native decoder"):
            collect(
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

        stats = collect(
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

        stats = list(iter_collect(tasks, max_shots=3, batch_size=2, seed=6))

        self.assertEqual([stat.task_id for stat in stats], ["a", "b"])
        self.assertEqual([stat.errors for stat in stats], [3, 3])

    def test_num_workers_keeps_result_order(self) -> None:
        tasks = [
            CollectionTask(dem=_logical_edge_dem(probability=0.25), task_id="a"),
            CollectionTask(dem=_logical_edge_dem(probability=0.5), task_id="b"),
        ]

        stats = collect(tasks, max_shots=128, batch_size=16, seed=8, num_workers=2)

        self.assertEqual([stat.task_id for stat in stats], ["a", "b"])
        self.assertEqual([stat.shots for stat in stats], [128, 128])

    def test_num_workers_parallelizes_single_task_without_changing_stats(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(probability=0.375), task_id="single")

        serial = collect([task], max_shots=256, batch_size=16, seed=123, num_workers=1)[0]
        parallel = collect([task], max_shots=256, batch_size=16, seed=123, num_workers=4)[0]

        self.assertEqual(parallel.task_id, serial.task_id)
        self.assertEqual(parallel.shots, serial.shots)
        self.assertEqual(parallel.errors, serial.errors)
        self.assertEqual(parallel.discards, serial.discards)
        self.assertEqual(parallel.custom_counts, serial.custom_counts)

    def test_num_workers_parallelizes_multiple_tasks_without_changing_stats(self) -> None:
        tasks = [
            CollectionTask(dem=_logical_edge_dem(probability=0.25), task_id="a"),
            CollectionTask(dem=_logical_edge_dem(probability=0.375), task_id="b"),
            CollectionTask(dem=_logical_edge_dem(probability=0.5), task_id="c"),
        ]

        serial = collect(tasks, max_shots=256, batch_size=16, seed=124, num_workers=1)
        parallel = collect(tasks, max_shots=256, batch_size=16, seed=124, num_workers=4)

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
        stats = collect(
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
        stats = collect(
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

    def test_save_resume_filepath_skips_completed_task(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="resume")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            first = collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=11,
                save_resume_filepath=path,
            )[0]
            second = collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=11,
                save_resume_filepath=path,
            )[0]

        self.assertEqual(first.shots, 6)
        self.assertEqual(second.shots, 6)
        self.assertEqual(second.errors, 6)

    def test_resume_file_header_and_second_run_appends_no_duplicate_rows(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="resume-header")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            collect(
                [task],
                max_shots=6,
                batch_size=3,
                seed=11,
                save_resume_filepath=path,
            )
            with open(path, newline="") as f:
                first_rows = list(csv.reader(f))

            collect(
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
                "custom_counts",
            ],
        )
        self.assertEqual(len(first_rows), 2)
        self.assertEqual(second_rows, first_rows)

    def test_existing_data_filepaths_count_towards_limits(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="existing")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            collect([task], max_shots=5, batch_size=5, seed=12, save_resume_filepath=path)
            stats = collect(
                [task],
                max_shots=5,
                batch_size=5,
                seed=999,
                existing_data_filepaths=[path],
            )[0]

        self.assertEqual(stats.shots, 5)
        self.assertEqual(stats.errors, 5)

    def test_resume_file_does_not_duplicate_parallel_collection(self) -> None:
        task = CollectionTask(dem=_logical_edge_dem(), task_id="parallel-resume")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/stats.csv"
            first = collect(
                [task],
                max_shots=8,
                batch_size=4,
                seed=15,
                num_workers=4,
                save_resume_filepath=path,
            )[0]
            second = collect(
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

    def test_progress_callback_receives_incremental_stats(self) -> None:
        seen = []

        stats = collect(
            [CollectionTask(dem=_logical_edge_dem(), task_id="progress")],
            max_shots=4,
            batch_size=4,
            seed=13,
            progress_callback=seen.append,
        )

        self.assertEqual(len(stats), 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].task_id, "progress")

    def test_print_progress_writes_status_to_stderr(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            collect(
                [CollectionTask(dem=_logical_edge_dem(), task_id="printed")],
                max_shots=4,
                batch_size=4,
                seed=14,
                print_progress=True,
            )

        self.assertIn("printed: shots=4 errors=4 discards=0", stderr.getvalue())

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
        self.assertIs(faultscope.CollectionTask, CollectionTask)
        self.assertIs(faultscope.CollectionOptions, CollectionOptions)
        self.assertIs(faultscope.TaskStats, TaskStats)
        self.assertIs(faultscope.collect, collect)
        self.assertIs(faultscope.iter_collect, iter_collect)


if __name__ == "__main__":
    unittest.main()
