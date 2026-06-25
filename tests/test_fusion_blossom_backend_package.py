import sys
import unittest
from pathlib import Path
from unittest import mock

from npsim._npsim_native import NATIVE_DECODER_PLUGIN_ABI
from npsim.backends import clear_native_decoder_plugin_cache
from npsim.core import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from npsim.decoders import (
    NativeFusionBlossomDecoder,
    available_native_decoders,
    create_native_decoder,
    get_native_decoder_class,
)
from npsim.dem import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from npsim.runtime import (
    BatchForwardNoiseAwareSimulator,
    compile_native_dem_sampler,
    generate_native_dem,
)

BACKEND_SRC = (
    Path(__file__).resolve().parents[1]
    / "backends"
    / "npsim-fusion-blossom"
    / "src"
)
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import npsim_fusion_blossom  # noqa: E402

try:
    from npsim_fusion_blossom import _native as fusion_native  # noqa: E402
except ImportError:  # pragma: no cover - exercised when backend is not built.
    fusion_native = None


requires_native_backend = unittest.skipUnless(
    npsim_fusion_blossom.native_extension_available(),
    "npsim-fusion-blossom native extension is not built",
)


class _FakeEntryPoints:
    def __init__(self, entry_points):
        self._entry_points = tuple(entry_points)

    def select(self, *, group):
        if group == "npsim.native_decoders":
            return self._entry_points
        return ()


class _FakeEntryPoint:
    name = "fusion-blossom"

    def load(self):
        return npsim_fusion_blossom.backend_manifest


class FusionBlossomBackendPackageTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_native_decoder_plugin_cache()

    def test_backend_manifest_contract(self) -> None:
        manifest = npsim_fusion_blossom.backend_manifest()

        self.assertEqual(manifest["abi_version"], NATIVE_DECODER_PLUGIN_ABI)
        self.assertEqual(manifest["name"], "fusion-blossom")
        self.assertEqual(manifest["version"], npsim_fusion_blossom.__version__)
        self.assertEqual(
            manifest["source"],
            "npsim-fusion-blossom minimal serial adapter",
        )
        self.assertIs(
            manifest["decoders"]["fusion-blossom"],
            npsim_fusion_blossom.NativeFusionBlossomDecoder,
        )

    @requires_native_backend
    def test_backend_from_dem_returns_external_native_solver_decoder(self) -> None:
        dem = DetectorErrorModel(
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

        decoder = npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

        self.assertEqual(decoder.name, "fusion-blossom")
        self.assertEqual(decoder.detector_ids, (0,))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertIsNotNone(decoder.__npsim_native_decoder_capsule__())

    @requires_native_backend
    def test_backend_entry_point_integrates_with_registry_and_fast_path(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.25,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )
        detector = Detector(id=0, measurement_keys=("m",))
        observable = LogicalObservable(id=0, measurement_keys=("m",))
        dem = generate_native_dem(
            circuit,
            detectors=(detector,),
            observables=(observable,),
        )
        simulator = BatchForwardNoiseAwareSimulator(circuit, observables=(observable,))

        with mock.patch(
            "npsim.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints((_FakeEntryPoint(),)),
        ):
            clear_native_decoder_plugin_cache()
            self.assertIn("fusion-blossom", available_native_decoders())
            self.assertIs(
                get_native_decoder_class("fusion-blossom"),
                npsim_fusion_blossom.NativeFusionBlossomDecoder,
            )
            decoder = create_native_decoder("fusion-blossom", dem=dem)
            friendly_decoder = NativeFusionBlossomDecoder.from_dem(dem)
            self.assertEqual(decoder.name, "fusion-blossom")
            result = simulator.estimate(shots=4096, seed=111, decoder=decoder)

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(friendly_decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)

    @requires_native_backend
    def test_backend_decodes_single_detector_boundary_edge(self) -> None:
        dem = DetectorErrorModel(
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
        decoder = npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

        result = compile_native_dem_sampler(dem).estimate(
            shots=4096,
            seed=201,
            decoder=decoder,
        )

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)

    @requires_native_backend
    def test_backend_decodes_two_detector_edge(self) -> None:
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=10, measurement_keys=()),
                Detector(id=20, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(10, 20),
                    observables=(0,),
                    location_id="edge0",
                    event="X",
                ),
            ),
        )
        decoder = npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

        result = compile_native_dem_sampler(dem).estimate(
            shots=4096,
            seed=202,
            decoder=decoder,
        )

        self.assertEqual(decoder.detector_ids, (10, 20))
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)

    @requires_native_backend
    def test_backend_returns_zero_when_edge_has_no_observable_flip(self) -> None:
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(0,),
                    observables=(),
                    location_id="edge0",
                    event="X",
                ),
            ),
        )
        decoder = npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

        class Batch:
            shots = 3
            detectors = {0: 0b111}

        self.assertEqual(decoder.decode_batch_masks(Batch()), {0: 0})
        self.assertEqual(decoder.python_decode_call_count, 1)

    @requires_native_backend
    def test_backend_slow_debug_fallback_with_loss_callback(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.25,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )
        detector = Detector(id=0, measurement_keys=("m",))
        observable = LogicalObservable(id=0, measurement_keys=("m",))
        dem = generate_native_dem(
            circuit,
            detectors=(detector,),
            observables=(observable,),
        )
        simulator = BatchForwardNoiseAwareSimulator(circuit, observables=(observable,))
        decoder = npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

        result = simulator.estimate(
            shots=128,
            seed=111,
            decoder=decoder,
            loss_mask_fn=lambda batch, corrections: batch.observables[0]
            ^ corrections[0],
        )

        self.assertEqual(decoder.python_decode_call_count, 1)
        self.assertEqual(result.mean_loss, 0.0)

    @requires_native_backend
    def test_backend_accepts_weight_scale_option(self) -> None:
        dem = DetectorErrorModel(
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

        decoder = npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(
            dem,
            options={"weight_scale": 10_000},
        )

        self.assertEqual(decoder.name, "fusion-blossom")

    @requires_native_backend
    def test_backend_rejects_invalid_options(self) -> None:
        dem = DetectorErrorModel(
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

        with self.assertRaisesRegex(ValueError, "unknown fusion-blossom option"):
            npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(
                dem,
                options={"solver": "real"},
            )
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(
                dem,
                options={"weight_scale": 0},
            )
        with self.assertRaisesRegex(ValueError, "scaled weight"):
            npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(
                dem,
                options={"weight_scale": 1e300},
            )

    @requires_native_backend
    def test_backend_rejects_duplicate_graph_endpoint(self) -> None:
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=10, measurement_keys=()),
                Detector(id=20, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(0.2, (10, 20), (0,), "edge0", "X"),
                DetectorErrorEdge(0.3, (20, 10), (), "edge1", "Z"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "duplicate graph endpoint"):
            npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

    def test_backend_from_dem_requires_built_native_extension(self) -> None:
        if npsim_fusion_blossom.native_extension_available():
            self.skipTest("native extension is built")
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(),
        )

        with self.assertRaisesRegex(ImportError, "native extension is not built"):
            npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(dem)

    def test_invalid_capsule_name_is_rejected(self) -> None:
        class BadCapsuleDecoder:
            def __npsim_native_decoder_capsule__(self):
                return object()

        simulator = BatchForwardNoiseAwareSimulator(Circuit(n_qubits=1, operations=[]))

        with self.assertRaisesRegex(ValueError, "capsule must be named"):
            simulator.estimate(shots=16, decoder=BadCapsuleDecoder())

    @requires_native_backend
    def test_invalid_capsule_descriptor_is_rejected(self) -> None:
        simulator = BatchForwardNoiseAwareSimulator(Circuit(n_qubits=1, operations=[]))

        for kind, message in (
            ("abi-mismatch", "ABI version"),
            ("missing-callback", "missing required callbacks"),
            ("not-thread-safe", "thread-safe"),
        ):
            with self.subTest(kind=kind):
                decoder = fusion_native.InvalidNativeDecoderCapsule(kind)
                with self.assertRaisesRegex(ValueError, message):
                    simulator.estimate(shots=16, decoder=decoder)

    @requires_native_backend
    def test_decode_error_status_is_reported(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )
        simulator = BatchForwardNoiseAwareSimulator(circuit)
        decoder = fusion_native.InvalidNativeDecoderCapsule("decode-error")

        with self.assertRaisesRegex(ValueError, "forced native decoder decode failure"):
            simulator.estimate(shots=16, decoder=decoder)


if __name__ == "__main__":
    unittest.main()
