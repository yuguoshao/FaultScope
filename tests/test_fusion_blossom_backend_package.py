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
from npsim.runtime import BatchForwardNoiseAwareSimulator, generate_native_dem

BACKEND_SRC = (
    Path(__file__).resolve().parents[1]
    / "backends"
    / "npsim-fusion-blossom"
    / "src"
)
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import npsim_fusion_blossom  # noqa: E402


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
        self.assertEqual(manifest["source"], "npsim-fusion-blossom scaffold")
        self.assertIs(
            manifest["decoders"]["fusion-blossom"],
            npsim_fusion_blossom.NativeFusionBlossomDecoder,
        )

    def test_backend_from_dem_returns_native_smoke_decoder(self) -> None:
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

        self.assertEqual(decoder.name, "graphlike-detector-copy")
        self.assertEqual(decoder.detector_ids, (0,))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertEqual(decoder.python_decode_call_count, 0)

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
            result = simulator.estimate(shots=4096, seed=111, decoder=decoder)

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(friendly_decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)

    def test_backend_rejects_options_until_real_solver_exists(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "does not accept options"):
            npsim_fusion_blossom.NativeFusionBlossomDecoder.from_dem(
                dem,
                options={"solver": "real"},
            )


if __name__ == "__main__":
    unittest.main()
