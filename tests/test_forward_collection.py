from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    Detector,
    LogicalObservable,
    NoiseLocation,
    Operation,
    generate_native_dem,
)
from faultscope.collection import (
    CollectionCounterSchema,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    collect,
    collect_hotspots,
    iter_progress,
)
from faultscope.collection._collect import _prepare_native_tasks
from faultscope.collection._dem_collect import _prepare_native_tasks as _prepare_dem_tasks
from faultscope.collection.dem import DemCollectionTask, collect as collect_dem
from faultscope.runtime.native import (
    compile_native_collection_sampler,
    compile_native_sampler,
)


def forward_circuit(probability: float = 0.25) -> Circuit:
    return Circuit(
        1,
        (
            Operation.noise(
                NoiseLocation(
                    "x0",
                    BernoulliPauliNoise("X"),
                    probability,
                    (0,),
                )
            ),
            Operation.measure(0, key="m"),
            Operation.detector(("m",), detector_id=3),
            Operation.observable_include(7, ("m",)),
        ),
    )


class ForwardCollectionTests(unittest.TestCase):
    def test_default_collection_sampler_matches_existing_sampler(self) -> None:
        circuit = forward_circuit()
        existing = compile_native_sampler(circuit)
        collection = compile_native_collection_sampler(circuit)

        self.assertEqual(collection.n_qubits, existing.n_qubits)
        self.assertEqual(collection.operation_count, existing.operation_count)
        self.assertEqual(collection.stored_operation_count, existing.stored_operation_count)
        self.assertEqual(collection.detector_ids, [3])
        self.assertEqual(collection.observable_ids, [7])

        existing_batch = existing.sample(shots=129, seed=123)
        collection_batch = collection.sample(shots=129, seed=123)
        self.assertEqual(collection_batch.measurements, existing_batch.measurements)
        self.assertEqual(collection_batch.detectors, existing_batch.detectors)
        self.assertEqual(collection_batch.observables, existing_batch.observables)

    def test_main_and_legacy_task_types_do_not_cross_route(self) -> None:
        circuit = forward_circuit()
        dem_task = DemCollectionTask(generate_native_dem(circuit))
        forward_task = CollectionTask(circuit)
        options = CollectionOptions(max_shots=1, batch_size=1)

        with self.assertRaisesRegex(TypeError, "faultscope.collection.dem"):
            collect([dem_task], options=options)
        with self.assertRaisesRegex(TypeError, "only DemCollectionTask"):
            collect_dem([forward_task], options=options)

    def test_collection_declaration_overrides_and_layout_ids(self) -> None:
        circuit = forward_circuit()
        embedded = compile_native_collection_sampler(circuit)
        self.assertEqual(embedded.detector_ids, [3])
        self.assertEqual(embedded.observable_ids, [7])

        replaced = compile_native_collection_sampler(
            circuit,
            detectors=(Detector(id=5, measurement_keys=("m",)),),
            observables=(LogicalObservable(id=11, measurement_keys=("m",)),),
        )
        self.assertEqual(replaced.detector_ids, [5])
        self.assertEqual(replaced.observable_ids, [11])

        cleared = compile_native_collection_sampler(
            circuit,
            detectors=(),
            observables=(),
        )
        self.assertEqual(cleared.detector_ids, [])
        self.assertEqual(cleared.observable_ids, [])

        with self.assertRaisesRegex(ValueError, "duplicate detector id 5"):
            compile_native_collection_sampler(
                circuit,
                detectors=(
                    Detector(id=5, measurement_keys=("m",)),
                    Detector(id=5, measurement_keys=("m",)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "duplicate observable id 11"):
            compile_native_collection_sampler(
                circuit,
                observables=(
                    LogicalObservable(id=11, measurement_keys=("m",)),
                    LogicalObservable(id=11, measurement_keys=("m",)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "unknown measurement key.*missing"):
            compile_native_collection_sampler(
                circuit,
                detectors=(Detector(id=2, measurement_keys=("missing",)),),
            )
        with self.assertRaisesRegex(ValueError, "unknown measurement key.*missing"):
            compile_native_collection_sampler(
                circuit,
                observables=(LogicalObservable(id=2, measurement_keys=("missing",)),),
            )

    def test_forward_hotspots_are_location_mappings_and_worker_invariant(self) -> None:
        task = CollectionTask(forward_circuit(), task_id="hotspot")
        results = []
        for workers in (1, 4):
            (result,) = collect_hotspots(
                [task],
                options=CollectionOptions(max_shots=130, batch_size=64),
                run_options=CollectionRunOptions(seed=77, num_workers=workers),
            )
            results.append(result)
            self.assertEqual([batch.shots for batch in result.batch_stats], [64, 64, 2])
            self.assertEqual(tuple(result.location_sensitivities), ("x0",))
            self.assertTrue(math.isfinite(result.location_sensitivities["x0"]))

        self.assertEqual(results[0].stats.errors, results[1].stats.errors)
        self.assertEqual(results[0].location_sensitivities, results[1].location_sensitivities)

    def test_zero_limit_forward_hotspot_preserves_zero_location_layout(self) -> None:
        (result,) = collect_hotspots(
            [CollectionTask(forward_circuit())],
            options=CollectionOptions(max_shots=8, max_errors=0, batch_size=4),
            run_options=CollectionRunOptions(seed=3, num_workers=4),
        )
        self.assertEqual(result.stats.shots, 0)
        self.assertEqual(result.batch_stats, ())
        self.assertEqual(result.location_sensitivities, {"x0": 0.0})

    def test_forward_and_explicit_dem_identities_are_isolated(self) -> None:
        circuit = forward_circuit()
        options = CollectionOptions(max_shots=1, batch_size=1)
        schema = CollectionCounterSchema()
        forward = _prepare_native_tasks(
            [CollectionTask(circuit)],
            options,
            (),
            schema,
        )[0]
        explicit = _prepare_native_tasks(
            [
                CollectionTask(
                    circuit,
                    detectors=(Detector(id=3, measurement_keys=("m",)),),
                    observables=(LogicalObservable(id=7, measurement_keys=("m",)),),
                )
            ],
            options,
            (),
            schema,
        )[0]
        dem = _prepare_dem_tasks(
            [DemCollectionTask(generate_native_dem(circuit))],
            options,
            (),
            schema,
        )[0]

        self.assertNotEqual(forward["sampling_id"], explicit["sampling_id"])
        self.assertNotEqual(forward["sampling_id"], dem["sampling_id"])
        self.assertNotEqual(forward["strong_id"], dem["strong_id"])

    def test_forward_collection_uses_external_declarations_for_counting(self) -> None:
        circuit = forward_circuit(1.0)
        (stats,) = collect(
            [
                CollectionTask(
                    circuit,
                    detectors=(Detector(id=5, measurement_keys=("m",)),),
                    observables=(LogicalObservable(id=11, measurement_keys=("m",)),),
                    postselection_mask=b"\x01",
                )
            ],
            options=CollectionOptions(max_shots=8, batch_size=4),
            run_options=CollectionRunOptions(seed=9, count_detection_events=True),
        )
        self.assertEqual(stats.errors, 0)
        self.assertEqual(stats.discards, 8)
        self.assertEqual(stats.custom_counts["detection_events"], 8)

    def test_forward_progress_and_csv_resume_keep_batch_deltas(self) -> None:
        task = CollectionTask(forward_circuit(1.0), task_id="forward-resume")
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "forward.csv"
            progress = list(
                iter_progress(
                    [task],
                    options=CollectionOptions(max_shots=10, batch_size=4),
                    run_options=CollectionRunOptions(
                        seed=41,
                        num_workers=3,
                        save_resume_filepath=resume_path,
                    ),
                )
            )
            (resumed,) = collect(
                [task],
                options=CollectionOptions(max_shots=14, batch_size=4),
                run_options=CollectionRunOptions(
                    seed=41,
                    num_workers=3,
                    save_resume_filepath=resume_path,
                ),
            )

        self.assertEqual([item.new_stats[0].shots for item in progress], [4, 4, 2])
        self.assertEqual([item.new_stats[0].errors for item in progress], [4, 4, 2])
        self.assertEqual(resumed.shots, 14)
        self.assertEqual(resumed.errors, 14)

    def test_forward_duplicate_strong_ids_fail_before_resume_write(self) -> None:
        circuit = forward_circuit(1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_path = Path(temp_dir) / "duplicate.csv"

            with self.assertRaisesRegex(
                ValueError,
                r'duplicate collection strong_id .*task indices 0 \("forward-a"\) and 1 \("forward-b"\)',
            ):
                collect(
                    [
                        CollectionTask(circuit, task_id="forward-a"),
                        CollectionTask(circuit, task_id="forward-b"),
                    ],
                    options=CollectionOptions(max_shots=4, batch_size=2),
                    run_options=CollectionRunOptions(
                        seed=1,
                        save_resume_filepath=resume_path,
                    ),
                )

            self.assertFalse(resume_path.exists())

    def test_forward_metadata_distinguishes_otherwise_identical_tasks(self) -> None:
        circuit = forward_circuit(1.0)
        stats = collect(
            [
                CollectionTask(circuit, task_id="replica-a", metadata={"replica": 0}),
                CollectionTask(circuit, task_id="replica-b", metadata={"replica": 1}),
            ],
            options=CollectionOptions(max_shots=4, batch_size=2),
            run_options=CollectionRunOptions(seed=1, num_workers=2),
        )

        self.assertEqual([stat.shots for stat in stats], [4, 4])
        self.assertNotEqual(stats[0].strong_id, stats[1].strong_id)

    def test_forward_adaptive_batching_reaches_exact_shot_cap(self) -> None:
        (stats,) = collect(
            [CollectionTask(forward_circuit(), task_id="forward-adaptive")],
            options=CollectionOptions(
                max_shots=31,
                batch_size=16,
                start_batch_size=3,
                max_batch_size=16,
                max_batch_seconds=0.01,
            ),
            run_options=CollectionRunOptions(
                seed=51,
                num_workers=4,
                count_detection_events=True,
            ),
        )

        self.assertEqual(stats.shots, 31)
        self.assertEqual(stats.custom_counts["detectors_checked"], 31)
        self.assertEqual(stats.custom_counts["detection_events"], stats.errors)


if __name__ == "__main__":
    unittest.main()
