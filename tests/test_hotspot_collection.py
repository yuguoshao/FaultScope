from __future__ import annotations

import tempfile
import unittest

from faultscope import (
    CollectionCounterSchema,
    CollectionOptions,
    CollectionRunOptions,
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    LogicalObservable,
    NativeCompositeDecoder,
    NativeNoCorrectionDecoder,
)
from faultscope.collection.dem import DemCollectionTask, collect, collect_hotspots


def logical_dem(probability: float) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=probability,
                detectors=(),
                observables=(0,),
                location_id="logical",
                event="L",
            ),
        ),
    )


def detected_logical_dem(probability: float) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(Detector(id=0, measurement_keys=()),),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=probability,
                detectors=(0,),
                observables=(0,),
                location_id="detected_logical",
                event="X",
            ),
        ),
    )


class HotspotCollectionTests(unittest.TestCase):
    def test_dem_hotspot_rejects_duplicate_strong_ids(self) -> None:
        dem = logical_dem(0.25)
        with self.assertRaisesRegex(
            ValueError,
            r'duplicate collection strong_id .*task indices 0 \("dem-hotspot-a"\) '
            r'and 1 \("dem-hotspot-b"\)',
        ):
            collect_hotspots(
                [
                    DemCollectionTask(dem=dem, task_id="dem-hotspot-a"),
                    DemCollectionTask(dem=dem, task_id="dem-hotspot-b"),
                ],
                options=CollectionOptions(max_shots=8, batch_size=4),
                run_options=CollectionRunOptions(seed=5, num_workers=2),
            )

    def test_collection_options_preserve_old_positionals_and_explicit_zero_overlay(self) -> None:
        positional = CollectionOptions(20, 3, 4)
        self.assertEqual(positional.max_shots, 20)
        self.assertEqual(positional.max_errors, 3)
        self.assertEqual(positional.batch_size, 4)
        self.assertEqual(positional.min_shots, 0)

        (stats,) = collect(
            [
                DemCollectionTask(
                    dem=logical_dem(1.0),
                    collection_options=CollectionOptions(min_shots=0),
                )
            ],
            options=CollectionOptions(
                max_shots=20,
                min_shots=8,
                max_errors=1,
                batch_size=4,
            ),
            run_options=CollectionRunOptions(seed=7),
        )
        self.assertEqual(stats.shots, 4)

    def test_composite_validates_children_and_stable_id_unions(self) -> None:
        decoder = NativeCompositeDecoder(
            (
                NativeNoCorrectionDecoder(observable_ids=(2,), detector_ids=(10, 20)),
                NativeNoCorrectionDecoder(observable_ids=(5,), detector_ids=(20, 30)),
            )
        )
        self.assertEqual(decoder.detector_ids, (10, 20, 30))
        self.assertEqual(decoder.observable_ids, (2, 5))
        with self.assertRaisesRegex(ValueError, "at least one"):
            NativeCompositeDecoder(())
        with self.assertRaisesRegex(ValueError, "duplicate observable id 1"):
            NativeCompositeDecoder(
                (
                    NativeNoCorrectionDecoder(observable_ids=(1,)),
                    NativeNoCorrectionDecoder(observable_ids=(1,)),
                )
            )
        with self.assertRaisesRegex(TypeError, "not a native decoder"):
            NativeCompositeDecoder((object(),))

    def test_hotspot_collection_is_worker_invariant_and_batch_ordered(self) -> None:
        dem = logical_dem(0.25)
        task = DemCollectionTask(dem=dem)
        results = []
        for workers in (1, 2, 4):
            (result,) = collect_hotspots(
                [task],
                options=CollectionOptions(max_shots=40, batch_size=10),
                run_options=CollectionRunOptions(
                    seed=123,
                    num_workers=workers,
                    count_observable_error_combos=True,
                ),
            )
            results.append(result)
            self.assertEqual([batch.shots for batch in result.batch_stats], [10] * 4)
        expected = results[0]
        for result in results[1:]:
            self.assertEqual(result.stats.shots, expected.stats.shots)
            self.assertEqual(result.stats.errors, expected.stats.errors)
            self.assertEqual(result.stats.custom_counts, expected.stats.custom_counts)
            self.assertEqual(result.edge_sensitivities, expected.edge_sensitivities)

    def test_min_shots_gates_error_stop_and_discards_in_flight_batches(self) -> None:
        (result,) = collect_hotspots(
            [DemCollectionTask(dem=logical_dem(1.0))],
            options=CollectionOptions(
                max_shots=20,
                min_shots=8,
                max_errors=1,
                batch_size=4,
            ),
            run_options=CollectionRunOptions(seed=7, num_workers=4),
        )
        self.assertEqual(result.stats.shots, 8)
        self.assertEqual(result.stats.errors, 8)
        self.assertEqual([batch.shots for batch in result.batch_stats], [4, 4])

    def test_min_shots_also_gates_logical_collection_error_stop(self) -> None:
        (stats,) = collect(
            [DemCollectionTask(dem=logical_dem(1.0))],
            options=CollectionOptions(
                max_shots=20,
                min_shots=8,
                max_errors=1,
                batch_size=4,
            ),
            run_options=CollectionRunOptions(seed=7, num_workers=4),
        )
        self.assertEqual(stats.shots, 8)
        self.assertEqual(stats.errors, 8)

    def test_hotspot_collection_rejects_csv_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "does not support CSV partial resume"):
                collect_hotspots(
                    [DemCollectionTask(dem=logical_dem(0.1))],
                    options=CollectionOptions(max_shots=4, batch_size=4),
                    run_options=CollectionRunOptions(save_resume_filepath=f"{tmp}/resume.csv"),
                )

    def test_postselected_shots_do_not_contribute_to_sensitivity(self) -> None:
        (result,) = collect_hotspots(
            [DemCollectionTask(dem=detected_logical_dem(1.0), postselection_mask=b"\x01")],
            options=CollectionOptions(max_shots=8, batch_size=4),
            run_options=CollectionRunOptions(seed=5, num_workers=2),
        )
        self.assertEqual(result.stats.errors, 0)
        self.assertEqual(result.stats.discards, 8)
        self.assertEqual(result.edge_sensitivities, (0.0,))

    def test_zero_error_limit_can_complete_without_scheduling(self) -> None:
        (result,) = collect_hotspots(
            [DemCollectionTask(dem=logical_dem(0.5))],
            options=CollectionOptions(max_shots=8, max_errors=0, batch_size=4),
            run_options=CollectionRunOptions(seed=3, num_workers=4),
        )
        self.assertEqual(result.stats.shots, 0)
        self.assertEqual(result.batch_stats, ())
        self.assertEqual(result.edge_sensitivities, (0.0,))

    def test_hotspot_validates_stop_key_before_zero_limit_completion(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported custom_error_count_key"):
            collect_hotspots(
                [DemCollectionTask(dem=logical_dem(0.5))],
                options=CollectionOptions(max_shots=8, max_errors=0, batch_size=4),
                run_options=CollectionRunOptions(
                    seed=3,
                    num_workers=4,
                    custom_error_count_key="detection_event",
                ),
            )

    def test_hotspot_uses_versioned_detection_schema_and_shared_stop_counter(self) -> None:
        (result,) = collect_hotspots(
            [DemCollectionTask(dem=detected_logical_dem(1.0))],
            options=CollectionOptions(max_shots=10, max_errors=3, batch_size=2),
            run_options=CollectionRunOptions(
                seed=4,
                num_workers=4,
                count_detection_events=True,
                custom_error_count_key="detectors_checked",
            ),
        )

        expected_schema = CollectionCounterSchema(count_detection_events=True)
        self.assertEqual(result.stats.shots, 4)
        self.assertEqual(result.stats.custom_counts["detectors_checked"], 4)
        self.assertEqual(result.stats.counter_schema, expected_schema)
        self.assertTrue(
            all(batch.counter_schema == expected_schema for batch in result.batch_stats)
        )


if __name__ == "__main__":
    unittest.main()
