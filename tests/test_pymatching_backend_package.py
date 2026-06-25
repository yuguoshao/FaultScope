import sys
import unittest
from pathlib import Path
from unittest import mock

from npsim._npsim_native import NATIVE_DECODER_PLUGIN_ABI
from npsim.backends import clear_native_decoder_plugin_cache
from npsim.decoders import (
    NativePyMatchingDecoder,
    PyMatchingBatchDecoder,
    available_native_decoders,
    create_native_decoder,
    get_native_decoder_class,
)
from npsim.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.runtime import compile_native_dem_sampler

BACKEND_SRC = (
    Path(__file__).resolve().parents[1]
    / "backends"
    / "npsim-pymatching"
    / "src"
)
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import npsim_pymatching  # noqa: E402

try:
    from npsim_pymatching import _native as pymatching_native  # noqa: E402
except ImportError:  # pragma: no cover - exercised when backend is not built.
    pymatching_native = None


requires_native_backend = unittest.skipUnless(
    npsim_pymatching.native_extension_available(),
    "npsim-pymatching native extension is not built",
)


class _FakeEntryPoints:
    def __init__(self, entry_points):
        self._entry_points = tuple(entry_points)

    def select(self, *, group):
        if group == "npsim.native_decoders":
            return self._entry_points
        return ()


class _FakeEntryPoint:
    name = "pymatching"

    def load(self):
        return npsim_pymatching.backend_manifest


class _Batch:
    def __init__(self, shots, detectors):
        self.shots = shots
        self.detectors = detectors


class PyMatchingBackendPackageTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_native_decoder_plugin_cache()

    def test_backend_manifest_contract(self) -> None:
        manifest = npsim_pymatching.backend_manifest()

        self.assertEqual(manifest["abi_version"], NATIVE_DECODER_PLUGIN_ABI)
        self.assertEqual(manifest["name"], "pymatching")
        self.assertEqual(manifest["version"], npsim_pymatching.__version__)
        self.assertEqual(
            manifest["source"],
            "npsim-pymatching native sparse-blossom adapter",
        )
        self.assertIs(
            manifest["decoders"]["pymatching"],
            npsim_pymatching.NativePyMatchingDecoder,
        )

    @requires_native_backend
    def test_backend_from_dem_returns_external_native_solver_decoder(self) -> None:
        decoder = npsim_pymatching.NativePyMatchingDecoder.from_dem(single_boundary_dem())

        self.assertEqual(decoder.name, "pymatching")
        self.assertEqual(decoder.detector_ids, (0,))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertEqual(decoder.edge_count, 1)
        self.assertEqual(decoder.solver_edge_count, 1)
        self.assertEqual(decoder.build_summary["dem_edge_count"], 1)
        self.assertEqual(decoder.build_summary["solver_edge_count"], 1)
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertIsNotNone(decoder.__npsim_native_decoder_capsule__())

    @requires_native_backend
    def test_backend_entry_point_integrates_with_registry_and_fast_path(self) -> None:
        dem = single_boundary_dem()
        sampler = compile_native_dem_sampler(dem)

        with mock.patch(
            "npsim.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints((_FakeEntryPoint(),)),
        ):
            clear_native_decoder_plugin_cache()
            self.assertIn("pymatching", available_native_decoders())
            self.assertIs(
                get_native_decoder_class("pymatching"),
                npsim_pymatching.NativePyMatchingDecoder,
            )
            decoder = create_native_decoder("pymatching", dem=dem)
            friendly_decoder = NativePyMatchingDecoder.from_dem(dem)
            result = sampler.estimate(shots=2048, seed=101, decoder=decoder)

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(friendly_decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)

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
        decoder = npsim_pymatching.NativePyMatchingDecoder.from_dem(dem)
        batch = _Batch(shots=4, detectors={0: 0b1010, 1: 0b1010})

        corrections = decoder.decode_batch_masks(batch)

        self.assertEqual(corrections, {0: 0b1010})
        self.assertEqual(decoder.python_decode_call_count, 1)

    @requires_native_backend
    def test_matches_python_pymatching_on_small_graph(self) -> None:
        try:
            python_decoder = PyMatchingBatchDecoder.from_dem(two_edge_dem())
        except ImportError as exc:
            self.skipTest(str(exc))
        native_decoder = npsim_pymatching.NativePyMatchingDecoder.from_dem(two_edge_dem())
        batch = _Batch(shots=6, detectors={0: 0b001011, 1: 0b101010})

        self.assertEqual(
            native_decoder.decode_batch_masks(batch),
            python_decoder.decode_batch_masks(batch),
        )

    @requires_native_backend
    def test_unknown_options_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown pymatching option"):
            npsim_pymatching.NativePyMatchingDecoder.from_dem(
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


if __name__ == "__main__":
    unittest.main()
