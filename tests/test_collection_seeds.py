"""Explicit collection seeds select streams independently of task identity."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from faultscope import (
    BernoulliPauliNoise,
    Circuit,
    NativeNoCorrectionDecoder,
    NoiseLocation,
    Operation,
    generate_native_dem,
)
from faultscope.collection import (
    CollectionCounterSchema,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    TaskStats,
    write_stats_to_csv_file,
)
import faultscope.collection._collect as forward
import faultscope.collection._dem_collect as dem
from faultscope.collection._dem_types import DemCollectionTask
from faultscope.collection._identity import domain_digest


def noisy_circuit(probability=0.375):
    return Circuit(
        1,
        (
            Operation.noise(NoiseLocation("x0", BernoulliPauliNoise("X"), probability, (0,))),
            Operation.measure(0, key="m"),
            Operation.detector(("m",), detector_id=0),
            Operation.observable_include(0, ("m",)),
        ),
    )


def task_factory(module, probability=0.375):
    source = noisy_circuit(probability)
    if module is dem:
        source = generate_native_dem(source)
        return lambda **kwargs: DemCollectionTask(source, **kwargs)
    return lambda **kwargs: CollectionTask(source, **kwargs)


def signature(stats):
    return (stats.shots, stats.errors, stats.discards, tuple(sorted(stats.custom_counts.items())))


def sample_batches(module, mode, tasks, *, run_seed=1234, workers=3, decoders=()):
    options = CollectionOptions(max_shots=257, batch_size=17)
    run_options = CollectionRunOptions(
        seed=run_seed,
        num_workers=workers,
        decoders=tuple(decoders),
        count_detection_events=True,
        count_observable_error_combos=True,
    )
    batches, identities, sensitivities = {}, {}, {}
    if mode == "hotspots":
        for result in module.collect_hotspots(tasks, options=options, run_options=run_options):
            name = result.stats.task_id
            batches[name] = tuple(signature(stats) for stats in result.batch_stats)
            identities[name] = result.stats.strong_id
            sensitivities[name] = (
                result.edge_sensitivities if module is dem else result.location_sensitivities
            )
    else:
        for progress in module.iter_progress(tasks, options=options, run_options=run_options):
            for stats in progress.new_stats:
                batches.setdefault(stats.task_id, []).append(signature(stats))
                identities[stats.task_id] = stats.strong_id
        batches = {name: tuple(rows) for name, rows in batches.items()}
    return batches, identities, sensitivities


class ExplicitCollectionSeedTests(unittest.TestCase):
    def test_task_seed_is_validated_before_native_preparation(self):
        for task_type, module in ((CollectionTask, forward), (DemCollectionTask, dem)):
            with (
                self.subTest(task_type=task_type.__name__),
                mock.patch.object(
                    module,
                    "_compile_task_sampler",
                    side_effect=AssertionError("native preparation reached"),
                ) as compile_sampler,
            ):
                for value in (True, False, 1.5, "1", b"1"):
                    with self.subTest(seed=value), self.assertRaisesRegex(TypeError, "seed"):
                        task_type(object(), seed=value)
                for value in (-1, 1 << 64):
                    with (
                        self.subTest(seed=value),
                        self.assertRaisesRegex(ValueError, "seed must be between"),
                    ):
                        task_type(object(), seed=value)
                for value in (None, 0, (1 << 64) - 1):
                    task = task_type(object(), seed=value)
                    self.assertEqual(task.seed, value)
                    self.assertEqual(replace(task, metadata={"changed": True}).seed, value)
                compile_sampler.assert_not_called()

    def test_seed_changes_only_strong_identity_and_is_passed_to_native(self):
        options = CollectionOptions(max_shots=1, batch_size=1)
        schema = CollectionCounterSchema()
        for module in (forward, dem):
            factory = task_factory(module)
            seeds = (None, 0, (1 << 64) - 1)
            native = module._prepare_native_tasks(
                [factory(seed=seed) for seed in seeds], options, (), schema
            )
            with self.subTest(route=module.__name__):
                self.assertEqual([item["seed"] for item in native], list(seeds))
                self.assertEqual(len({item["sampling_id"] for item in native}), 1)
                self.assertEqual(len({item["strong_id"] for item in native}), 3)
                for seed, item in zip(seeds, native):
                    expected = domain_digest(
                        schema="faultscope.collection.strong_id",
                        schema_version=5,
                        payload={
                            "sampling_id": item["sampling_id"],
                            "counter_schema": schema._to_payload(),
                            "task_seed": seed,
                            "seed_policy": "explicit-v1",
                        },
                    )
                    self.assertEqual(item["strong_id"], expected)

    def test_metadata_changes_identity_but_not_seeded_batch_samples(self):
        for module in (forward, dem):
            factory = task_factory(module)
            for mode in ("logical", "hotspots"):
                with self.subTest(route=module.__name__, mode=mode):
                    tasks = [factory(task_id=name, metadata={"label": name}) for name in ("a", "b")]
                    batches, identities, sensitivities = sample_batches(module, mode, tasks)
                    self.assertEqual(batches["a"], batches["b"])
                    self.assertNotEqual(identities["a"], identities["b"])
                    self.assertEqual(sum(row[0] for row in batches["a"]), 257)
                    if mode == "hotspots":
                        self.assertEqual(sensitivities["a"], sensitivities["b"])

    def test_decoder_version_changes_identity_but_not_seeded_batch_samples(self):
        for module in (forward, dem):
            factory = task_factory(module)
            decoder = NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(0,))
            original_payload = module.decoder_identity_payload
            for mode in ("logical", "hotspots"):
                outcomes = []
                for version in ("backend-before", "backend-after"):
                    # Builtin decoder types cannot be wrapped by a Python capsule
                    # provider. Change only their declared identity at the boundary;
                    # the real native decoder still performs all collection work.
                    def versioned_payload(*args, **kwargs):
                        payload = original_payload(*args, **kwargs)
                        payload["payload"]["implementation_version"] = version
                        return payload

                    with mock.patch.object(
                        module, "decoder_identity_payload", side_effect=versioned_payload
                    ):
                        outcomes.append(
                            sample_batches(module, mode, [factory(task_id="task", decoder=decoder)])
                        )
                with self.subTest(route=module.__name__, mode=mode):
                    self.assertEqual(outcomes[0][0], outcomes[1][0])
                    self.assertNotEqual(outcomes[0][1], outcomes[1][1])
                    self.assertEqual(outcomes[0][2], outcomes[1][2])

    def test_explicit_seed_overrides_run_seed_and_separates_replicas(self):
        for module in (forward, dem):
            factory = task_factory(module)
            for mode in ("logical", "hotspots"):
                with self.subTest(route=module.__name__, mode=mode):
                    tasks = [factory(task_id=f"seed-{seed}", seed=seed) for seed in (0, 999)]
                    first = sample_batches(module, mode, tasks, run_seed=11)
                    second = sample_batches(module, mode, tasks, run_seed=22)
                    self.assertEqual(first, second)
                    self.assertNotEqual(first[0]["seed-0"], first[0]["seed-999"])
                    self.assertNotEqual(first[1]["seed-0"], first[1]["seed-999"])
                    for seed in (0, 999):
                        inherited = sample_batches(
                            module, mode, [factory(task_id="inherited")], run_seed=seed
                        )
                        self.assertEqual(first[0][f"seed-{seed}"], inherited[0]["inherited"])
                        if mode == "hotspots":
                            self.assertEqual(first[2][f"seed-{seed}"], inherited[2]["inherited"])

    def test_decoder_fanout_retains_seed_and_shared_detector_samples(self):
        decoders = ("no-correction", "graphlike-detector-copy")
        schema = CollectionCounterSchema(count_detection_events=True)
        for module in (forward, dem):
            factory = task_factory(module)
            task = factory(task_id="fanout", seed=81)
            native = module._prepare_native_tasks(
                [task], CollectionOptions(max_shots=257, batch_size=17), decoders, schema
            )
            self.assertEqual([item["seed"] for item in native], [81, 81])
            self.assertIs(native[0]["sampler"], native[1]["sampler"])
            for mode in ("logical", "hotspots"):
                with self.subTest(route=module.__name__, mode=mode):
                    first = sample_batches(module, mode, [task], run_seed=1, decoders=decoders)
                    second = sample_batches(module, mode, [task], run_seed=2, decoders=decoders)
                    self.assertEqual(first, second)
                    no_correction = first[0]["fanout:no-correction"]
                    copied = first[0]["fanout:graphlike-detector-copy"]
                    self.assertEqual(
                        [dict(row[3])["detection_events"] for row in no_correction],
                        [dict(row[3])["detection_events"] for row in copied],
                    )
                    self.assertGreater(sum(row[1] for row in no_correction), 0)
                    self.assertEqual(sum(row[1] for row in copied), 0)

    def test_schema4_rows_are_not_resumed_under_explicit_seed_policy(self):
        options = CollectionOptions(max_shots=3, batch_size=3)
        schema = CollectionCounterSchema()
        for module in (forward, dem):
            task = task_factory(module, probability=1.0)(metadata={"paper": "old"})
            native = module._prepare_native_tasks([task], options, (), schema)[0]
            legacy_id = domain_digest(
                schema="faultscope.collection.strong_id",
                schema_version=4,
                payload={
                    "sampling_id": native["sampling_id"],
                    "counter_schema": schema._to_payload(),
                },
            )
            old_stats = TaskStats(
                task_id="legacy",
                strong_id=legacy_id,
                shots=100,
                errors=100,
                discards=0,
                seconds=1.0,
                decoder=None,
                metadata=task.metadata,
                counter_schema=schema,
            )
            with self.subTest(route=module.__name__), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "old.csv"
                write_stats_to_csv_file(path, [old_stats])
                (stats,) = module.collect(
                    [task],
                    options=options,
                    run_options=CollectionRunOptions(seed=5, existing_data_filepaths=(path,)),
                )
                self.assertEqual((stats.shots, stats.errors), (3, 3))
                self.assertEqual(stats.strong_id, native["strong_id"])
                self.assertNotEqual(stats.strong_id, legacy_id)

    def test_explicit_seed_isolates_resume_but_run_seed_does_not(self):
        for module in (forward, dem):
            factory = task_factory(module, probability=1.0)
            with self.subTest(route=module.__name__), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "resume.csv"
                task = factory(seed=11)
                (first,) = module.collect(
                    [task],
                    options=CollectionOptions(max_shots=7, batch_size=7),
                    run_options=CollectionRunOptions(seed=3, save_resume_filepath=path),
                )
                (different_seed,) = module.collect(
                    [factory(seed=12)],
                    options=CollectionOptions(max_shots=3, batch_size=3),
                    run_options=CollectionRunOptions(seed=3, existing_data_filepaths=(path,)),
                )
                (resumed,) = module.collect(
                    [task],
                    options=CollectionOptions(max_shots=7, batch_size=7),
                    run_options=CollectionRunOptions(seed=999, existing_data_filepaths=(path,)),
                )
                self.assertEqual((different_seed.shots, different_seed.errors), (3, 3))
                self.assertNotEqual(first.strong_id, different_seed.strong_id)
                self.assertEqual((resumed.shots, resumed.errors), (7, 7))
                self.assertEqual(resumed.strong_id, first.strong_id)


if __name__ == "__main__":
    unittest.main()
