import gc
import sys
import unittest
from pathlib import Path
from unittest import mock

from faultscope import _native as faultscope_native
from faultscope._native import NATIVE_DECODER_PLUGIN_ABI
from faultscope.backends import clear_native_decoder_plugin_cache
from faultscope.decoders import (
    NativePyMatchingDecoder,
    PyMatchingDecoder,
    available_native_decoders,
    create_native_decoder,
    get_native_decoder_class,
)
from faultscope.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.runtime import compile_native_dem_sampler
from tests.native_backend_v3_helpers import (
    assert_factory_failure_lifetimes,
    assert_v3_worker_contract,
)

BACKEND_SRC = Path(__file__).resolve().parents[1] / "backends" / "faultscope-pymatching" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import faultscope_pymatching  # noqa: E402

try:
    from faultscope_pymatching import _native as pymatching_native  # noqa: E402
except ImportError:  # pragma: no cover - exercised when backend is not built.
    pymatching_native = None


requires_native_backend = unittest.skipUnless(
    faultscope_pymatching.native_extension_available(),
    "faultscope-pymatching native extension is not built",
)


class _FakeEntryPoints:
    def __init__(self, entry_points):
        self._entry_points = tuple(entry_points)

    def select(self, *, group):
        if group == "faultscope.native_decoders":
            return self._entry_points
        return ()


class _FakeEntryPoint:
    name = "pymatching"

    def load(self):
        return faultscope_pymatching.backend_manifest


class _Batch:
    def __init__(self, shots, detectors):
        self.shots = shots
        self.detectors = detectors


class PyMatchingBackendPackageTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_native_decoder_plugin_cache()

    def test_backend_manifest_contract(self) -> None:
        manifest = faultscope_pymatching.backend_manifest()

        self.assertEqual(manifest["abi_version"], NATIVE_DECODER_PLUGIN_ABI)
        self.assertEqual(manifest["name"], "pymatching")
        self.assertEqual(manifest["version"], faultscope_pymatching.__version__)
        self.assertEqual(
            manifest["source"],
            "faultscope-pymatching native sparse-blossom adapter",
        )
        self.assertIs(
            manifest["decoders"]["pymatching"],
            faultscope_pymatching.NativePyMatchingDecoder,
        )

    @requires_native_backend
    def test_backend_from_dem_returns_external_native_solver_decoder(self) -> None:
        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(single_boundary_dem())

        self.assertEqual(decoder.name, "pymatching")
        self.assertEqual(decoder.detector_ids, (0,))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertEqual(decoder.edge_count, 1)
        self.assertEqual(decoder.solver_edge_count, 1)
        self.assertEqual(decoder.build_summary["dem_edge_count"], 1)
        self.assertEqual(decoder.build_summary["solver_edge_count"], 1)
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertIsNotNone(decoder.__faultscope_native_decoder_capsule__())
        payload = decoder.strong_id_payload()
        self.assertEqual(payload["backend"], "pymatching")
        self.assertEqual(payload["backend_version"], faultscope_pymatching.__version__)
        self.assertEqual(payload["native_decoder_abi"], NATIVE_DECODER_PLUGIN_ABI)
        self.assertEqual(payload["parameters"], {})
        self.assertEqual(payload["solver"], decoder.build_summary)

    @requires_native_backend
    def test_backend_from_dem_uses_canonical_ids_and_parity(self) -> None:
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=5, measurement_keys=()),
                Detector(id=2, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=8),),
            edges=(
                DetectorErrorEdge(0.1, (7, 7), (9, 9), "cancelled", "X"),
                DetectorErrorEdge(0.2, (4, 2, 4), (11, 11, 9), "odd", "Z"),
            ),
        )

        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(dem)

        self.assertEqual(decoder.detector_ids, (5, 2, 7, 4))
        self.assertEqual(decoder.observable_ids, (8, 9, 11))
        self.assertEqual(decoder.edge_count, 2)
        self.assertEqual(decoder.solver_edge_count, 1)
        self.assertEqual(
            decoder.decode_batch_masks(_Batch(shots=1, detectors={5: 0, 2: 1, 7: 0, 4: 0})),
            {8: 0, 9: 1, 11: 0},
        )

    @requires_native_backend
    def test_exact_v3_factory_creates_distinct_workers_and_fast_paths(self) -> None:
        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(single_boundary_dem())

        test_stats = assert_v3_worker_contract(self, decoder)
        self.assertEqual(test_stats.factory_drops, 0)
        del decoder
        gc.collect()
        self.assertEqual(test_stats.factory_drops, 1)
        gc.collect()
        self.assertEqual(test_stats.factory_drops, 1)

    @requires_native_backend
    def test_concurrent_collection_uses_one_factory_handle(self) -> None:
        dem = single_boundary_dem()
        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(dem)
        test_stats = decoder._inner._test_stats_for_test()
        test_stats.enable_decode_overlap()
        tasks = (
            {
                "task_id": "pymatching-v3-workers",
                "strong_id": "pymatching-v3-workers-strong",
                "sampler": compile_native_dem_sampler(dem),
                "decoder": decoder,
                "metadata_json": "{}",
                "max_shots": 128,
                "min_shots": 128,
                "batch_size": 8,
            },
        )

        (stats,) = faultscope_native._collect_dem_logical_error_stats_many(
            tasks,
            num_workers=4,
            seed=1234,
        )

        self.assertEqual(stats["shots"], 128)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertGreater(len(set(test_stats.worker_addresses)), 1)
        self.assertEqual(test_stats.worker_creates, test_stats.worker_drops)
        self.assertEqual(test_stats.active_decodes, 0)
        self.assertGreater(test_stats.max_active_decodes, 1)

    @requires_native_backend
    def test_factory_failures_retain_messages_across_concurrent_calls(self) -> None:
        assert_factory_failure_lifetimes(self, pymatching_native.InvalidNativeDecoderCapsule)

    @requires_native_backend
    def test_invalid_worker_outputs_are_dropped_once(self) -> None:
        sampler = compile_native_dem_sampler(single_boundary_dem())
        for kind in (
            "create-error-after-allocation",
            "missing-worker-decode",
            "invalid-worker-size",
        ):
            with self.subTest(kind=kind):
                decoder = pymatching_native.InvalidNativeDecoderCapsule(kind)
                with self.assertRaisesRegex(ValueError, "worker"):
                    sampler.estimate(shots=1, seed=19, decoder=decoder)
                self.assertEqual(decoder.worker_drops, 1)

    @requires_native_backend
    def test_temporary_worker_helper_drops_invalid_worker_once(self) -> None:
        for kind in (
            "create-error-after-allocation",
            "missing-worker-decode",
            "invalid-worker-size",
        ):
            with self.subTest(kind=kind):
                decoder = pymatching_native.InvalidNativeDecoderCapsule(kind)
                with self.assertRaisesRegex(ValueError, "worker"):
                    decoder._try_create_temporary_worker_for_test()
                self.assertEqual(decoder.worker_drops, 1)

    @requires_native_backend
    def test_backend_entry_point_integrates_with_registry_and_fast_path(self) -> None:
        dem = single_boundary_dem()
        sampler = compile_native_dem_sampler(dem)

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints((_FakeEntryPoint(),)),
        ):
            clear_native_decoder_plugin_cache()
            self.assertIn("pymatching", available_native_decoders())
            self.assertIs(
                get_native_decoder_class("pymatching"),
                faultscope_pymatching.NativePyMatchingDecoder,
            )
            decoder = create_native_decoder("pymatching", dem=dem)
            friendly_decoder = NativePyMatchingDecoder.from_dem(dem)
            result = sampler.estimate(shots=2048, seed=101, decoder=decoder)
            mean_loss_result = sampler.estimate(
                shots=2048,
                seed=101,
                decoder=decoder,
                aggregate_hotspots=False,
            )

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(friendly_decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)
        self.assertEqual(mean_loss_result.mean_loss, 0.0)
        self.assertEqual(mean_loss_result.edge_sensitivities, {})

    @requires_native_backend
    def test_decode_batch_masks_debug_fallback_decodes_two_detector_edge(self) -> None:
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=0, measurement_keys=()),
                Detector(id=1, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(0, 1),
                    observables=(0,),
                    location_id="edge0",
                    event="X",
                ),
            ),
        )
        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(dem)
        batch = _Batch(shots=4, detectors={0: 0b1010, 1: 0b1010})

        corrections = decoder.decode_batch_masks(batch)

        self.assertEqual(corrections, {0: 0b1010})
        self.assertEqual(decoder.python_decode_call_count, 1)

    @requires_native_backend
    def test_decode_batch_masks_reads_and_writes_multiple_packed_words(self) -> None:
        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(single_boundary_dem())
        mask = (1 << 0) | (1 << 65) | (1 << 129)
        batch = _Batch(shots=130, detectors={0: mask})

        corrections = decoder.decode_batch_masks(batch)

        self.assertEqual(corrections, {0: mask})
        self.assertEqual(decoder.python_decode_call_count, 1)

    @requires_native_backend
    def test_decode_batch_masks_handles_more_than_64_observables(self) -> None:
        decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(many_observable_dem(65))
        mask = 0b10101
        batch = _Batch(shots=5, detectors={0: mask})

        corrections = decoder.decode_batch_masks(batch)

        self.assertEqual(corrections, {observable_id: mask for observable_id in range(65)})
        self.assertEqual(decoder.python_decode_call_count, 1)

    @requires_native_backend
    def test_matches_python_pymatching_on_small_graph(self) -> None:
        try:
            python_decoder = PyMatchingDecoder.from_dem(two_edge_dem())
        except ImportError as exc:
            self.skipTest(str(exc))
        native_decoder = faultscope_pymatching.NativePyMatchingDecoder.from_dem(two_edge_dem())
        batch = _Batch(shots=6, detectors={0: 0b001011, 1: 0b101010})

        self.assertEqual(
            native_decoder.decode_batch_masks(batch),
            python_decoder.decode_batch_masks(batch),
        )

    @requires_native_backend
    def test_unknown_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown pymatching option"):
            faultscope_pymatching.NativePyMatchingDecoder.from_dem(
                single_boundary_dem(),
                options={"enable_correlations": True},
            )


def single_boundary_dem() -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(Detector(id=0, measurement_keys=()),),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=0.2,
                detectors=(0,),
                observables=(0,),
                location_id="edge0",
                event="X",
            ),
        ),
    )


def two_edge_dem() -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(
            Detector(id=0, measurement_keys=()),
            Detector(id=1, measurement_keys=()),
        ),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=0.2,
                detectors=(0,),
                observables=(0,),
                location_id="edge0",
                event="X",
            ),
            DetectorErrorEdge(
                probability=0.2,
                detectors=(0, 1),
                observables=(0,),
                location_id="edge1",
                event="X",
            ),
            DetectorErrorEdge(
                probability=0.2,
                detectors=(1,),
                observables=(),
                location_id="edge2",
                event="X",
            ),
        ),
    )


def many_observable_dem(observable_count: int) -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(Detector(id=0, measurement_keys=()),),
        observables=tuple(LogicalObservable(id=index) for index in range(observable_count)),
        edges=(
            DetectorErrorEdge(
                probability=0.2,
                detectors=(0,),
                observables=tuple(range(observable_count)),
                location_id="edge0",
                event="X",
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
