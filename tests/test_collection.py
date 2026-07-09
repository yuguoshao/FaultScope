import math
import unittest
from unittest import mock

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

        with self.assertRaisesRegex(ValueError, "max_shots"):
            CollectionOptions(max_shots=0)
        with self.assertRaisesRegex(ValueError, "max_errors"):
            CollectionOptions(max_errors=-1)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            CollectionOptions(batch_size=0)

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
        )

        self.assertEqual(stats.error_rate, 0.25)
        self.assertEqual(stats.logical_error_rate, 0.25)
        self.assertEqual(stats.stderr, math.sqrt(0.25 * 0.75 / 100))

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
            "faultscope.collection._collect._collect_dem_logical_error_stats",
            return_value=(7, 3, 0.25),
        ) as collect_native:
            stats = collect(
                [CollectionTask(dem=dem, task_id="native-core")],
                max_shots=11,
                max_errors=5,
                batch_size=4,
                seed=9,
            )[0]

        collect_native.assert_called_once_with(
            sampler,
            max_shots=11,
            max_errors=5,
            batch_size=4,
            seed=9,
            decoder=None,
        )
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

    def test_iter_collect_yields_each_task(self) -> None:
        tasks = [
            CollectionTask(dem=_logical_edge_dem(), task_id="a"),
            CollectionTask(dem=_logical_edge_dem(), task_id="b"),
        ]

        stats = list(iter_collect(tasks, max_shots=3, batch_size=2, seed=6))

        self.assertEqual([stat.task_id for stat in stats], ["a", "b"])
        self.assertEqual([stat.errors for stat in stats], [3, 3])

    def test_public_exports(self) -> None:
        self.assertIs(faultscope.CollectionTask, CollectionTask)
        self.assertIs(faultscope.CollectionOptions, CollectionOptions)
        self.assertIs(faultscope.TaskStats, TaskStats)
        self.assertIs(faultscope.collect, collect)
        self.assertIs(faultscope.iter_collect, iter_collect)


if __name__ == "__main__":
    unittest.main()
