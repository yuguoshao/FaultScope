import importlib.util
import io
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import faultscope
from faultscope.runtime import DemFaultScopeSimulator, FaultScopeSimulator, SampleBatch
from faultscope.core import Circuit, NoiseLocation, Operation
from faultscope.dem import (
    BinaryLinearDecodingProblem,
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    DetectorErrorModelGenerator,
    DetectorGraphEdgeHotspot,
    DetectorGraphHotspots,
    GraphlikeDecodingProblem,
    IndexedDem,
    LogicalObservable,
    SparseBinaryMatrix,
)
from faultscope.dem import (
    DemHotspotEstimator,
    DemSampleBatch,
    DemEdgeHotspot,
    DemHotspotEstimate,
    DemLocationHotspot,
    DemLocationMetadata,
)
from faultscope.core import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from faultscope.runtime import (
    UnsupportedNativeCircuitError,
    NativeDemSampler,
    NativePackedSampler,
    compile_native_dem_generator,
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
    compile_native_sampler,
    generate_native_dem,
)
from faultscope.runtime.loss import logical_residual_loss_mask
from faultscope.decoders import (
    NativeBatchDecoder,
    NativeBpDecoder,
    NativeBposdDecoder,
    NativeDecoderBackendUnavailable,
    NativeFusionBlossomDecoder,
    NativeGraphlikeDetectorCopyDecoder,
    NativeMwpmDecoder,
    NativeNoCorrectionDecoder,
    NativePyMatchingDecoder,
    PyMatchingDecoder,
    RepetitionCodeDecoder,
    UnsupportedPyMatchingDemError,
    available_native_decoders,
    create_native_decoder,
    get_native_decoder_class,
)
from faultscope.backends import (
    clear_native_decoder_plugin_cache,
    native_decoder_backend_statuses,
    official_native_decoder_backend_catalog,
)
from faultscope.experiments import make_repetition_code_experiment
from faultscope.runtime import FailureEstimate
from faultscope.core import PauliFrame, StabilizerState
from faultscope.io import StimImportError, parse_stim_circuit
from faultscope.viz import (
    VisualizationUnavailableError,
    write_rotated_surface_code_spatial_hotspot_map,
    write_repetition_gate_structure_hotspot_map,
    write_repetition_hotspot_heatmap,
)
from tests.surface_code_examples import (
    _rotated_surface_code_checks,
    make_large_rotated_surface_code_memory_example,
)
from tests.stim_helpers import (
    dem_batch_from_stim_samples,
    final_data_measurement_circuit,
    measurement_batch_from_stim_samples,
    faultscope_dem_error_edges,
    stim_dem_error_edges,
    to_stim_circuit,
    with_dem_declarations,
)


def _assert_binomial_count_close(
    testcase: unittest.TestCase,
    observed: int,
    *,
    shots: int,
    probability: float,
) -> None:
    expected = shots * probability
    sigma = (shots * probability * (1.0 - probability)) ** 0.5
    testcase.assertLessEqual(abs(observed - expected), max(12.0, 6.0 * sigma))


class _FakeEntryPoints:
    def __init__(self, entry_points):
        self._entry_points = tuple(entry_points)

    def select(self, *, group):
        if group == "faultscope.native_decoders":
            return self._entry_points
        return ()


class _FakeEntryPoint:
    def __init__(self, name, manifest_factory):
        self.name = name
        self._manifest_factory = manifest_factory

    def load(self):
        return self._manifest_factory


class PublicRenameTests(unittest.TestCase):
    def test_faultscope_exports_new_public_names_only(self) -> None:
        self.assertIs(faultscope.FaultScopeSimulator, FaultScopeSimulator)
        self.assertIs(faultscope.SampleBatch, SampleBatch)
        self.assertIs(faultscope.FailureEstimate, FailureEstimate)
        self.assertIs(faultscope.PyMatchingDecoder, PyMatchingDecoder)
        self.assertFalse(hasattr(faultscope, "BatchForwardNoiseAwareSimulator"))
        self.assertFalse(hasattr(faultscope, "BatchTrajectory"))
        self.assertFalse(hasattr(faultscope, "SimulationResult"))
        self.assertFalse(hasattr(faultscope, "PyMatchingBatchDecoder"))
        self.assertIsNone(importlib.util.find_spec("npsim"))


class StabilizerStateTests(unittest.TestCase):
    def test_pauli_frame_public_methods(self) -> None:
        frame = PauliFrame.zero(2)
        frame.apply_pauli_string((0, 1), "XZ")
        self.assertEqual(frame.x, [1, 0])
        self.assertEqual(frame.z, [0, 1])
        self.assertEqual(frame.pauli_on((0, 1)), "XZ")
        self.assertEqual(frame.measurement_flip((0,), "Z"), 1)

    def test_measurement_after_pauli_error(self) -> None:
        rng = random.Random(1)
        state = StabilizerState.zero(1)
        self.assertEqual(state.measure_z(0, rng), 0)

        state = StabilizerState.zero(1)
        state.apply_pauli(0, "X")
        self.assertEqual(state.measure_z(0, rng), 1)

    def test_bell_stabilizer_measurements(self) -> None:
        rng = random.Random(2)
        state = StabilizerState.zero(2)
        state.apply_h(0)
        state.apply_cx(0, 1)

        self.assertEqual(state.measure_pauli([1, 1], [0, 0], rng), 0)
        self.assertEqual(state.measure_pauli([0, 0], [1, 1], rng), 0)

    def test_s_dag_undoes_s(self) -> None:
        rng = random.Random(3)
        state = StabilizerState.zero(1)
        state.apply_h(0)
        state.apply_s(0)
        state.apply_s_dag(0)
        self.assertEqual(state.measure_x(0, rng), 0)

    def test_cz_cluster_stabilizers(self) -> None:
        rng = random.Random(4)
        state = StabilizerState.zero(2)
        state.apply_h(0)
        state.apply_h(1)
        state.apply_cz(0, 1)
        self.assertEqual(state.measure_pauli([1, 0], [0, 1], rng), 0)
        self.assertEqual(state.measure_pauli([0, 1], [1, 0], rng), 0)

    def test_swap_moves_state(self) -> None:
        rng = random.Random(5)
        state = StabilizerState.zero(2)
        state.apply_pauli(0, "X")
        state.apply_swap(0, 1)
        self.assertEqual(state.measure_z(0, rng), 0)
        self.assertEqual(state.measure_z(1, rng), 1)


class NoiseAwareSimulatorTests(unittest.TestCase):
    def test_core_circuit_objects_are_extension_classes(self) -> None:
        import faultscope
        import faultscope.core as core
        import faultscope.decoders as decoders
        import faultscope.dem as dem_module
        import faultscope.io as io
        import faultscope.runtime as runtime_module
        import faultscope.viz as viz
        import faultscope._native as native
        from faultscope.runtime import FaultHotspot

        self.assertNotIn("BatchStabilizerState", core.__all__)
        self.assertFalse(hasattr(core, "BatchStabilizerState"))
        self.assertIs(Circuit, native.Circuit)
        self.assertIs(Operation, native.Operation)
        self.assertIs(NoiseLocation, native.NoiseLocation)
        self.assertIs(BernoulliPauliNoise, native.BernoulliPauliNoise)
        self.assertIs(PauliChannel, native.PauliChannel)
        self.assertIs(SingleQubitDepolarizing, native.SingleQubitDepolarizing)
        self.assertIs(TwoQubitDepolarizing, native.TwoQubitDepolarizing)
        self.assertIs(MeasurementBitFlip, native.MeasurementBitFlip)
        self.assertIs(PauliFrame, native.PauliFrame)
        self.assertIs(StabilizerState, native.StabilizerState)
        self.assertIs(Detector, native.Detector)
        self.assertIs(LogicalObservable, native.LogicalObservable)
        self.assertIs(DetectorErrorEdge, native.DetectorErrorEdge)
        self.assertIs(DetectorErrorModel, native.DetectorErrorModel)
        self.assertIs(DetectorErrorModelGenerator, native.DetectorErrorModelGenerator)
        self.assertIs(IndexedDem, native.IndexedDem)
        self.assertIs(GraphlikeDecodingProblem, native.GraphlikeDecodingProblem)
        self.assertIs(BinaryLinearDecodingProblem, native.BinaryLinearDecodingProblem)
        self.assertIs(SparseBinaryMatrix, native.SparseBinaryMatrix)
        self.assertIs(DetectorGraphEdgeHotspot, native.DetectorGraphEdgeHotspot)
        self.assertIs(DetectorGraphHotspots, native.DetectorGraphHotspots)
        self.assertIs(FaultScopeSimulator, native.FaultScopeSimulator)
        self.assertIs(DemFaultScopeSimulator, native.DemFaultScopeSimulator)
        self.assertIs(SampleBatch, native.SampleBatch)
        self.assertIs(DemSampleBatch, native.DemSampleBatch)
        self.assertIs(FaultHotspot, native.FaultHotspot)
        self.assertIs(FailureEstimate, native.FailureEstimate)
        self.assertIs(DemLocationMetadata, native.DemLocationMetadata)
        self.assertIs(DemLocationHotspot, native.DemLocationHotspot)
        self.assertIs(DemEdgeHotspot, native.DemEdgeHotspot)
        self.assertIs(DemHotspotEstimate, native.DemHotspotEstimate)
        self.assertIs(DemHotspotEstimator, native.DemHotspotEstimator)
        self.assertIs(NativePackedSampler, native.NativePackedSampler)
        self.assertIs(NativeDemSampler, native.NativeDemSampler)
        self.assertIs(NativeBatchDecoder, native.NativeBatchDecoder)
        self.assertIs(
            NativeGraphlikeDetectorCopyDecoder,
            native.NativeGraphlikeDetectorCopyDecoder,
        )
        self.assertIs(NativeNoCorrectionDecoder, native.NativeNoCorrectionDecoder)
        self.assertIs(faultscope.Circuit, native.Circuit)
        self.assertIs(faultscope.FaultScopeSimulator, native.FaultScopeSimulator)
        self.assertIs(faultscope.DemFaultScopeSimulator, native.DemFaultScopeSimulator)
        self.assertIs(faultscope.DetectorErrorModelGenerator, native.DetectorErrorModelGenerator)
        self.assertIs(faultscope.DemHotspotEstimator, native.DemHotspotEstimator)
        self.assertIs(
            faultscope.NativeGraphlikeDetectorCopyDecoder,
            native.NativeGraphlikeDetectorCopyDecoder,
        )
        self.assertIs(core.Circuit, native.Circuit)
        self.assertIs(runtime_module.FaultScopeSimulator, native.FaultScopeSimulator)
        self.assertIs(runtime_module.DemFaultScopeSimulator, native.DemFaultScopeSimulator)
        self.assertIs(dem_module.DetectorErrorModelGenerator, native.DetectorErrorModelGenerator)
        self.assertIs(dem_module.DemFaultScopeSimulator, native.DemFaultScopeSimulator)
        self.assertIs(dem_module.DemHotspotEstimator, native.DemHotspotEstimator)
        self.assertIs(dem_module.IndexedDem, native.IndexedDem)
        self.assertIs(
            decoders.NativeGraphlikeDetectorCopyDecoder,
            native.NativeGraphlikeDetectorCopyDecoder,
        )
        self.assertIs(decoders.NativeNoCorrectionDecoder, native.NativeNoCorrectionDecoder)
        self.assertIs(decoders.available_native_decoders, available_native_decoders)
        self.assertIs(faultscope.available_native_decoders, available_native_decoders)
        self.assertIs(decoders.NativeFusionBlossomDecoder, NativeFusionBlossomDecoder)
        self.assertIs(faultscope.NativeFusionBlossomDecoder, NativeFusionBlossomDecoder)
        self.assertIs(decoders.NativeBpDecoder, NativeBpDecoder)
        self.assertIs(faultscope.NativeBpDecoder, NativeBpDecoder)
        self.assertIs(decoders.NativeBposdDecoder, NativeBposdDecoder)
        self.assertIs(faultscope.NativeBposdDecoder, NativeBposdDecoder)
        self.assertIs(decoders.NativeMwpmDecoder, NativeMwpmDecoder)
        self.assertIs(faultscope.NativeMwpmDecoder, NativeMwpmDecoder)
        self.assertIs(decoders.create_native_decoder, create_native_decoder)
        self.assertIs(faultscope.create_native_decoder, create_native_decoder)
        self.assertIs(decoders.get_native_decoder_class, get_native_decoder_class)
        self.assertIs(faultscope.get_native_decoder_class, get_native_decoder_class)
        self.assertTrue(hasattr(io, "parse_stim_circuit"))
        self.assertTrue(hasattr(decoders, "PyMatchingDecoder"))
        self.assertFalse(hasattr(native, "NativeFusionBlossomDecoder"))
        self.assertTrue(hasattr(decoders, "NativeFusionBlossomDecoder"))
        self.assertTrue(hasattr(viz, "write_repetition_hotspot_heatmap"))

        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.2,
            qubits=(0,),
            tags={"qubit": 0, "gate": "idle"},
        )
        detector = Operation.detector(("m",), detector_id=7, coords=(1.0, 2.0))
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location, gate="idle"),
                Operation.measure(0, key="m", basis="Z", noise=location),
                detector,
            ],
        )

        self.assertEqual(circuit.n_qubits, 1)
        self.assertEqual(circuit.operations[0].metadata["gate"], "idle")
        self.assertEqual(detector.metadata["coords"], (1.0, 2.0))
        self.assertEqual(circuit.noise_locations()["x0"].id, "x0")
        self.assertEqual(circuit.noise_locations()["x0"].qubits, (0,))

        edge = DetectorErrorEdge(
            probability=0.125,
            detectors=(7,),
            observables=(2,),
            location_id="x0",
            event="X",
            tags={"gate": "idle"},
        )
        self.assertEqual(Detector(id=7, measurement_keys=("m",)).measurement_keys, ("m",))
        self.assertEqual(LogicalObservable(id=2, measurement_keys=("m",)).id, 2)
        self.assertEqual(edge.to_dem_line(), "error(0.125) D7 L2")
        self.assertEqual(edge.tags["gate"], "idle")

        observable = LogicalObservable(id=2, measurement_keys=("m",))
        native_circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location, gate="idle"),
                Operation.measure(0, key="m", basis="Z"),
                detector,
            ],
        )
        sampler = native.compile_sampler(native_circuit, (observable,))
        self.assertIsInstance(sampler, native.NativePackedSampler)
        sample_payload = sampler.sample(8, 123)
        self.assertIsInstance(sample_payload, SampleBatch)
        self.assertIn("m", sample_payload.measurements)

        native_dem = native.generate_dem(
            native_circuit,
            (Detector(id=7, measurement_keys=("m",)),),
            (observable,),
        )
        self.assertIsInstance(native_dem, DetectorErrorModel)
        self.assertEqual(native_dem.edges[0].location_id, "x0")
        self.assertEqual(native_dem.edges[0].event, "X")
        self.assertEqual(native_dem.edges[0].tags["gate"], "idle")

        dem = DetectorErrorModel(
            detectors=(Detector(id=7, measurement_keys=("m",)),),
            observables=(observable,),
            edges=(edge,),
        )
        dem_sampler = native.compile_dem_sampler(dem)
        self.assertIsInstance(dem_sampler, native.NativeDemSampler)
        self.assertEqual(dem_sampler.edge_count, 1)
        dem_batch = dem_sampler.run_batch(8, 123, True)
        self.assertIsInstance(dem_batch, DemSampleBatch)
        self.assertIn(7, dem_batch.detectors)
        dem_result = dem_sampler.estimate_default(8, 123, None, 1)
        self.assertIsInstance(dem_result, DemHotspotEstimate)
        self.assertIsInstance(dem_result.top_edges(1)[0], DemEdgeHotspot)

        with self.assertRaises((TypeError, ValueError)):
            native.compile_sampler({"n_qubits": 1, "operations": []})

    def test_native_decoder_introspection(self) -> None:
        decoder = NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(5,))
        base_decoder = NativeBatchDecoder.no_correction(
            observable_ids=(1,),
            detector_ids=(6,),
        )

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            self.assertEqual(
                available_native_decoders(),
                ("no-correction", "graphlike-detector-copy"),
            )
            self.assertNotIn("fusion-blossom", available_native_decoders())
        clear_native_decoder_plugin_cache()
        self.assertEqual(decoder.name, "no-correction")
        self.assertEqual(decoder.detector_ids, (5,))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertIn("no-correction", repr(decoder))
        self.assertEqual(base_decoder.name, "no-correction")
        self.assertEqual(base_decoder.detector_ids, (6,))
        self.assertEqual(base_decoder.observable_ids, (1,))
        self.assertIn("NativeBatchDecoder", repr(base_decoder))

    def test_native_backend_catalog_includes_reserved_decoders(self) -> None:
        catalog = {entry.name: entry for entry in official_native_decoder_backend_catalog()}

        self.assertIn("fusion-blossom", catalog)
        self.assertEqual(catalog["fusion-blossom"].problem_kind, "graphlike")
        self.assertTrue(catalog["fusion-blossom"].installable)
        self.assertEqual(catalog["fusion-blossom"].package_name, "faultscope-fusion-blossom")
        self.assertIn("pymatching", catalog)
        self.assertEqual(catalog["pymatching"].problem_kind, "graphlike")
        self.assertTrue(catalog["pymatching"].installable)
        self.assertEqual(catalog["pymatching"].package_name, "faultscope-pymatching")
        self.assertIn("mwpm", catalog)
        self.assertEqual(catalog["mwpm"].problem_kind, "graphlike")
        self.assertFalse(catalog["mwpm"].installable)
        self.assertEqual(catalog["mwpm"].package_name, "faultscope-mwpm")
        self.assertEqual(catalog["mwpm"].repo_url, "https://github.com/Quon-team/mwpm.rs.git")
        self.assertIn("ABI v1", catalog["mwpm"].description)
        self.assertIn("FaultScope native decoder ABI v2", catalog["mwpm"].description)
        self.assertIn("bpdecoder", catalog)
        self.assertEqual(catalog["bpdecoder"].problem_kind, "binary-linear")
        self.assertTrue(catalog["bpdecoder"].installable)
        self.assertEqual(catalog["bpdecoder"].package_name, "faultscope-bpdecoder")
        self.assertIn("bposd", catalog)
        self.assertEqual(catalog["bposd"].problem_kind, "binary-linear")
        self.assertFalse(catalog["bposd"].installable)

    def test_missing_fusion_blossom_backend_has_install_hint(self) -> None:
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

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            with self.assertRaisesRegex(
                NativeDecoderBackendUnavailable,
                "python -m faultscope.backends install fusion-blossom",
            ):
                NativeFusionBlossomDecoder.from_dem(dem)
        clear_native_decoder_plugin_cache()

    def test_missing_pymatching_backend_has_install_hint(self) -> None:
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

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            with self.assertRaisesRegex(
                NativeDecoderBackendUnavailable,
                "python -m faultscope.backends install pymatching",
            ):
                NativePyMatchingDecoder.from_dem(dem)
        clear_native_decoder_plugin_cache()

    def test_mwpm_backend_is_unavailable_pending_abi_v2_migration(self) -> None:
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

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            with self.assertRaises(NativeDecoderBackendUnavailable) as raised:
                NativeMwpmDecoder.from_dem(dem)
            message = str(raised.exception)
            self.assertIn("ABI v1", message)
            self.assertIn("FaultScope native decoder ABI v2", message)
            self.assertNotIn("python -m faultscope.backends install mwpm", message)
        clear_native_decoder_plugin_cache()

    def test_missing_bpdecoder_backend_has_install_hint(self) -> None:
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

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            with self.assertRaisesRegex(
                NativeDecoderBackendUnavailable,
                "python -m faultscope.backends install bpdecoder",
            ):
                NativeBpDecoder.from_dem(dem)
        clear_native_decoder_plugin_cache()

    def test_reserved_bposd_backend_has_install_hint(self) -> None:
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

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            with self.assertRaisesRegex(
                NativeDecoderBackendUnavailable,
                "python -m faultscope.backends install bposd",
            ):
                NativeBposdDecoder.from_dem(dem)
        clear_native_decoder_plugin_cache()

    def test_mock_post_install_native_decoder_plugin_uses_fast_path(self) -> None:
        class MockFusionBlossomDecoder:
            @staticmethod
            def from_dem(dem, *, options=None):
                self.assertIsNone(options)
                return NativeNoCorrectionDecoder(
                    observable_ids=tuple(observable.id for observable in dem.observables),
                    detector_ids=tuple(detector.id for detector in dem.detectors),
                )

            @staticmethod
            def from_circuit(circuit, *, detectors=None, observables=None, options=None):
                self.assertIsNone(options)
                return NativeNoCorrectionDecoder(
                    observable_ids=tuple(observable.id for observable in observables or ()),
                    detector_ids=tuple(detector.id for detector in detectors or ()),
                )

        def manifest():
            return {
                "name": "fusion-blossom",
                "version": "test",
                "source": "unit-test",
                "abi_version": "faultscope.native_decoder_plugin.v2",
                "decoders": {"fusion-blossom": MockFusionBlossomDecoder},
            }

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
        observable = LogicalObservable(id=0, measurement_keys=("m",))
        dem = generate_native_dem(
            circuit,
            detectors=(Detector(id=0, measurement_keys=("m",)),),
            observables=(observable,),
        )
        simulator = FaultScopeSimulator(circuit, observables=(observable,))
        entry_points = _FakeEntryPoints((_FakeEntryPoint("fusion-blossom", manifest),))

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=entry_points,
        ):
            clear_native_decoder_plugin_cache()
            self.assertIn("fusion-blossom", available_native_decoders())
            self.assertIs(get_native_decoder_class("fusion-blossom"), MockFusionBlossomDecoder)
            decoder = NativeFusionBlossomDecoder.from_dem(dem)
            generic_decoder = create_native_decoder("fusion-blossom", dem=dem)
            native_result = simulator.estimate(shots=4096, seed=111, decoder=decoder)
            default_result = simulator.estimate(shots=4096, seed=111)

        clear_native_decoder_plugin_cache()
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(generic_decoder.python_decode_call_count, 0)
        self.assertEqual(native_result.mean_loss, default_result.mean_loss)

    def test_backend_cli_status_and_install_dry_run(self) -> None:
        from faultscope.backends.__main__ import main

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(()),
        ):
            clear_native_decoder_plugin_cache()
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                self.assertEqual(main(["status"]), 0)
            self.assertIn("no-correction", stdout.getvalue())
            self.assertIn("fusion-blossom", stdout.getvalue())
            self.assertIn("mwpm", stdout.getvalue())
            self.assertIn("bpdecoder", stdout.getvalue())
            self.assertIn("bposd", stdout.getvalue())
            self.assertIn("not-installed", stdout.getvalue())

            stdout = io.StringIO()
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("sys.stdout", stdout):
                    self.assertEqual(
                        main(
                            [
                                "install",
                                "fusion-blossom",
                                "--dry-run",
                                "--target-dir",
                                tmpdir,
                            ]
                        ),
                        0,
                    )
                self.assertFalse(os.listdir(tmpdir))
            self.assertIn("git clone", stdout.getvalue())
            self.assertIn("faultscope-fusion-blossom", stdout.getvalue())

            stdout = io.StringIO()
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("sys.stdout", stdout):
                    self.assertEqual(
                        main(
                            [
                                "install",
                                "bpdecoder",
                                "--dry-run",
                                "--target-dir",
                                tmpdir,
                            ]
                        ),
                        0,
                    )
                self.assertFalse(os.listdir(tmpdir))
            self.assertIn("faultscope-bpdecoder", stdout.getvalue())
            self.assertIn("pip install --upgrade faultscope-bpdecoder", stdout.getvalue())

            stdout = io.StringIO()
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("sys.stdout", stdout):
                    self.assertEqual(
                        main(
                            [
                                "install",
                                "mwpm",
                                "--dry-run",
                                "--target-dir",
                                tmpdir,
                            ]
                        ),
                        0,
                    )
                self.assertFalse(os.listdir(tmpdir))
            self.assertIn("ABI v1", stdout.getvalue())
            self.assertIn("FaultScope native decoder ABI v2", stdout.getvalue())
            self.assertNotIn("FaultScope will install", stdout.getvalue())
            self.assertNotIn("python -m faultscope.backends install mwpm", stdout.getvalue())
            self.assertNotIn("pip install", stdout.getvalue())

            stdout = io.StringIO()
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch("sys.stdout", stdout):
                    self.assertEqual(
                        main(
                            [
                                "install",
                                "bposd",
                                "--dry-run",
                                "--target-dir",
                                tmpdir,
                            ]
                        ),
                        0,
                    )
                self.assertFalse(os.listdir(tmpdir))
            self.assertIn("reserved and not installable yet", stdout.getvalue())
            self.assertIn("faultscope-bposd", stdout.getvalue())
        clear_native_decoder_plugin_cache()

    def test_post_install_plugin_manifest_abi_is_strict_v2(self) -> None:
        expected = "faultscope.native_decoder_plugin.v2"
        for observed in ("faultscope.native_decoder_plugin.v1", "wrong", None):
            with self.subTest(observed=observed):
                manifest_data = {
                    "name": "fusion-blossom",
                    "version": "test",
                    "source": "unit-test",
                    "decoders": {"fusion-blossom": object},
                }
                if observed is not None:
                    manifest_data["abi_version"] = observed

                def manifest():
                    return manifest_data

                with mock.patch(
                    "faultscope.backends.registry.metadata.entry_points",
                    return_value=_FakeEntryPoints((_FakeEntryPoint("fusion-blossom", manifest),)),
                ):
                    clear_native_decoder_plugin_cache()
                    self.assertNotIn("fusion-blossom", available_native_decoders())
                    statuses = {status.name: status for status in native_decoder_backend_statuses()}
                    status = statuses["fusion-blossom"]
                    self.assertTrue(status.installed)
                    self.assertFalse(status.loadable)
                    self.assertIn(repr(observed), status.error)
                    self.assertIn(repr(expected), status.error)
        clear_native_decoder_plugin_cache()

    def test_installed_mwpm_v1_plugin_is_reported_as_incompatible(self) -> None:
        def manifest():
            return {
                "name": "mwpm",
                "version": "0.1.0",
                "source": "installed-test-plugin",
                "abi_version": "faultscope.native_decoder_plugin.v1",
                "decoders": {"mwpm": object},
            }

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints((_FakeEntryPoint("mwpm", manifest),)),
        ):
            clear_native_decoder_plugin_cache()
            statuses = {status.name: status for status in native_decoder_backend_statuses()}
            status = statuses["mwpm"]
            self.assertTrue(status.installed)
            self.assertFalse(status.loadable)
            self.assertIn("faultscope.native_decoder_plugin.v1", status.error)
            self.assertIn("faultscope.native_decoder_plugin.v2", status.error)
            self.assertNotIn("python -m faultscope.backends install mwpm", status.error)
        clear_native_decoder_plugin_cache()

    def test_post_install_plugin_duplicate_decoder_name_is_not_loadable(self) -> None:
        def first_manifest():
            return {
                "name": "backend-a",
                "version": "test",
                "source": "unit-test",
                "abi_version": "faultscope.native_decoder_plugin.v2",
                "decoders": {"fusion-blossom": object},
            }

        def second_manifest():
            return {
                "name": "backend-b",
                "version": "test",
                "source": "unit-test",
                "abi_version": "faultscope.native_decoder_plugin.v2",
                "decoders": {"fusion-blossom": object},
            }

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints(
                (
                    _FakeEntryPoint("backend-a", first_manifest),
                    _FakeEntryPoint("backend-b", second_manifest),
                )
            ),
        ):
            clear_native_decoder_plugin_cache()
            statuses = {status.name: status for status in native_decoder_backend_statuses()}
            self.assertTrue(statuses["backend-a"].loadable)
            self.assertFalse(statuses["backend-b"].loadable)
            self.assertIn("already provided", statuses["backend-b"].error)
        clear_native_decoder_plugin_cache()

    def test_post_install_plugin_missing_decoders_is_not_loadable(self) -> None:
        def manifest():
            return {
                "name": "missing-decoders",
                "version": "test",
                "source": "unit-test",
                "abi_version": "faultscope.native_decoder_plugin.v2",
            }

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints((_FakeEntryPoint("missing-decoders", manifest),)),
        ):
            clear_native_decoder_plugin_cache()
            statuses = {status.name: status for status in native_decoder_backend_statuses()}
            self.assertFalse(statuses["missing-decoders"].loadable)
            self.assertIn("did not declare decoders", statuses["missing-decoders"].error)
        clear_native_decoder_plugin_cache()

    def test_post_install_plugin_missing_required_field_is_not_loadable(self) -> None:
        def manifest():
            return {
                "name": "missing-version",
                "source": "unit-test",
                "abi_version": "faultscope.native_decoder_plugin.v2",
                "decoders": {"missing-version": object},
            }

        with mock.patch(
            "faultscope.backends.registry.metadata.entry_points",
            return_value=_FakeEntryPoints((_FakeEntryPoint("missing-version", manifest),)),
        ):
            clear_native_decoder_plugin_cache()
            statuses = {status.name: status for status in native_decoder_backend_statuses()}
            self.assertFalse(statuses["missing-version"].loadable)
            self.assertIn("did not declare version", statuses["missing-version"].error)
        clear_native_decoder_plugin_cache()

    def test_noise_models_are_extension_classes_with_public_methods(self) -> None:
        bernoulli = BernoulliPauliNoise("XZ")
        self.assertEqual(bernoulli.pauli, "XZ")
        self.assertEqual(bernoulli.sample(random.Random(1), 1.0), "XZ")
        self.assertEqual(bernoulli.sample(random.Random(1), 0.0), "II")
        self.assertAlmostEqual(bernoulli.score("XZ", 0.25), 4.0)
        self.assertAlmostEqual(bernoulli.score("II", 0.25), -1.0 / 0.75)
        self.assertIn("BernoulliPauliNoise", repr(bernoulli))

        state = StabilizerState.zero(2)
        frame = PauliFrame.zero(2)
        bernoulli.apply("XZ", state, frame, (0, 1))
        self.assertEqual(frame.pauli_on((0, 1)), "XZ")

        channel = PauliChannel({"X": 1.0, "Y": 3.0})
        self.assertEqual(channel.weights, {"X": 1.0, "Y": 3.0})
        self.assertEqual(channel.event_length, 1)
        self.assertEqual(channel.total_weight, 4.0)
        self.assertEqual(channel.sample(random.Random(1), 0.0), "I")
        self.assertAlmostEqual(channel.score("I", 0.2), -1.0 / 0.8)
        self.assertAlmostEqual(channel.score("Y", 0.2), 5.0)
        self.assertIn("PauliChannel", repr(channel))

        single = SingleQubitDepolarizing()
        self.assertEqual(single.sample(random.Random(1), 0.0), "I")
        state = StabilizerState.zero(1)
        frame = PauliFrame.zero(1)
        single.apply("X", state, frame, (0,))
        self.assertEqual(frame.pauli_on((0,)), "X")

        two = TwoQubitDepolarizing()
        self.assertIn("IX", two._events)
        self.assertEqual(two.sample(random.Random(1), 0.0), "II")
        state = StabilizerState.zero(2)
        frame = PauliFrame.zero(2)
        two.apply("YZ", state, frame, (0, 1))
        self.assertEqual(frame.pauli_on((0, 1)), "YZ")

        mflip = MeasurementBitFlip()
        self.assertIs(mflip.sample(random.Random(1), 1.0), True)
        self.assertEqual(mflip.apply_to_bit(0, True), 1)
        self.assertEqual(mflip.apply_to_bit(1, True), 0)
        self.assertAlmostEqual(mflip.score(True, 0.2), 5.0)
        self.assertAlmostEqual(mflip.score(False, 0.2), -1.0 / 0.8)

        with self.assertRaises(ValueError):
            PauliChannel({})
        with self.assertRaises(ValueError):
            PauliChannel({"I": 1.0})
        with self.assertRaises(ValueError):
            PauliChannel({"X": -1.0})
        with self.assertRaises(ValueError):
            PauliChannel({"X": 0.0})
        with self.assertRaises(ValueError):
            PauliChannel({"X": 1.0, "ZZ": 1.0})


class BatchNoiseAwareSimulatorTests(unittest.TestCase):
    def test_repetition_decoder_packed_masks_match_scalar_decode(self) -> None:
        rng = random.Random(90210)
        shots = 129
        for distance in range(1, 10):
            keys = tuple(f"m{index}" for index in range(distance - 1))
            masks = {key: rng.getrandbits(shots) for key in keys}
            decoder = RepetitionCodeDecoder(
                distance=distance,
                measurement_keys=keys,
                observable_id=distance,
            )

            class Batch:
                def __init__(self) -> None:
                    self.shots = shots

                def measurement_masks(self, selected: tuple[str, ...]) -> dict[str, int]:
                    return {key: masks[key] for key in selected}

            batch = Batch()
            packed = decoder.decode_batch_masks(batch)[distance]
            scalar = 0
            for shot in range(shots):
                syndrome = [(masks[key] >> shot) & 1 for key in keys]
                if decoder.decode(syndrome, {}, batch)[0]:
                    scalar |= 1 << shot
            with self.subTest(distance=distance):
                self.assertEqual(packed, scalar)

    def test_repetition_decoder_reads_selected_measurement_masks_once(self) -> None:
        decoder = RepetitionCodeDecoder(
            distance=3,
            measurement_keys=("m0", "m1"),
            observable_id=7,
        )
        masks = {"m0": 0b10101, "m1": 0b01110}

        class SelectedBatch:
            shots = 5

            def __init__(self) -> None:
                self.selections: list[tuple[str, ...]] = []

            @property
            def measurements(self) -> dict[str, int]:
                raise AssertionError("bulk measurements should not be materialized")

            def measurement_masks(self, keys: tuple[str, ...]) -> dict[str, int]:
                selected = tuple(keys)
                self.selections.append(selected)
                return {key: masks[key] for key in selected}

        selected_batch = SelectedBatch()
        selected_correction = decoder.decode_batch_masks(selected_batch)
        self.assertEqual(selected_batch.selections, [("m0", "m1")])

        expected = 0
        for shot in range(selected_batch.shots):
            syndrome = [(masks[key] >> shot) & 1 for key in decoder.measurement_keys]
            if decoder.decode(syndrome, {}, selected_batch)[0]:
                expected |= 1 << shot
        self.assertEqual(selected_correction, {7: expected})

        class LegacyBatch:
            shots = 5

            def __init__(self) -> None:
                self.measurement_reads = 0

            @property
            def measurements(self) -> dict[str, int]:
                self.measurement_reads += 1
                return masks

        legacy_batch = LegacyBatch()
        self.assertEqual(decoder.decode_batch_masks(legacy_batch), {7: expected})
        self.assertEqual(legacy_batch.measurement_reads, 1)

    def test_batch_score_function_estimates_single_x_noise_gradient(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.2,
            qubits=(0,),
            tags={"qubit": 0, "round": 0, "gate": "idle"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        result = FaultScopeSimulator(circuit).estimate(
            shots=30_000,
            seed=13,
            loss_mask_fn=lambda batch: batch.measurements["m"],
        )

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.sensitivities["x0"], 1.0, delta=0.08)
        self.assertEqual(result.losses, [])

    def test_batch_repetition_code_experiment_runs(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=3,
            rounds=1,
            data_error_rate={(0, 0): 0.15, (0, 1): 0.15, (0, 2): 0.01},
            measurement_error_rate=0.02,
        )
        result = FaultScopeSimulator(
            experiment.circuit,
            observables=experiment.observables,
        ).estimate(
            shots=10_000,
            seed=14,
            decoder=experiment.decoder,
        )

        self.assertGreaterEqual(result.mean_loss, 0.0)
        self.assertLessEqual(result.mean_loss, 1.0)
        self.assertTrue(result.top_hotspots(top_k=3))
        self.assertIn(0, result.by_round)
        self.assertIn("idle", result.by_gate)
        self.assertIn("measure", result.by_gate)

    def test_native_decoder_forward_estimate_uses_native_fast_path(self) -> None:
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
            ],
        )
        observables = (LogicalObservable(id=0, measurement_keys=("m",)),)
        simulator = FaultScopeSimulator(circuit, observables=observables)
        decoder = NativeNoCorrectionDecoder(observable_ids=(0,))

        native_result = simulator.estimate(shots=4096, seed=111, decoder=decoder)
        default_result = simulator.estimate(shots=4096, seed=111)

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(native_result.mean_loss, default_result.mean_loss)
        self.assertEqual(native_result.hotspots, default_result.hotspots)

    def test_native_decoder_invalid_correction_raises_value_error(self) -> None:
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
            ],
        )
        observables = (LogicalObservable(id=0, measurement_keys=("m",)),)
        decoder = NativeNoCorrectionDecoder(observable_ids=(0, 0))

        with self.assertRaisesRegex(ValueError, "duplicate correction observable id 0"):
            FaultScopeSimulator(
                circuit,
                observables=observables,
            ).estimate(shots=128, seed=113, decoder=decoder)

    def test_native_decoder_with_python_loss_uses_slow_compat_path(self) -> None:
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
            ],
        )
        observables = (LogicalObservable(id=0, measurement_keys=("m",)),)
        decoder = NativeNoCorrectionDecoder(observable_ids=(0,))

        result = FaultScopeSimulator(
            circuit,
            observables=observables,
        ).estimate(
            shots=128,
            seed=112,
            decoder=decoder,
            loss_mask_fn=lambda batch, corrections: corrections[0],
        )

        self.assertEqual(decoder.python_decode_call_count, 1)
        self.assertEqual(result.mean_loss, 0.0)

    def test_batch_samples_random_ideal_measurements(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        batch = FaultScopeSimulator(circuit).run_batch(
            shots=128,
            rng=random.Random(15),
        )
        self.assertGreater(batch.measurements["m"].bit_count(), 0)
        self.assertLess(batch.measurements["m"].bit_count(), batch.shots)

    def test_batch_preserves_random_measurement_correlations(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m0", basis="Z"),
                Operation.measure(0, key="m1", basis="Z"),
                Operation.detector(("m0", "m1"), detector_id=0),
            ],
        )
        batch = FaultScopeSimulator(circuit).run_batch(
            shots=128,
            rng=random.Random(20),
        )
        self.assertEqual(batch.measurements["m0"], batch.measurements["m1"])
        self.assertEqual(batch.detectors[0], 0)

    def test_batch_random_measurement_detector_responds_to_noise(self) -> None:
        location = NoiseLocation(
            id="x_between",
            model=BernoulliPauliNoise("X"),
            rate=1.0,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m0", basis="Z"),
                Operation.noise(location),
                Operation.measure(0, key="m1", basis="Z"),
                Operation.detector(("m0", "m1"), detector_id=0),
            ],
        )
        batch = FaultScopeSimulator(circuit).run_batch(
            shots=64,
            rng=random.Random(21),
        )
        self.assertEqual(batch.detectors[0], batch.all_mask)
        self.assertEqual(batch.noise_event_masks["x_between"], batch.all_mask)

    def test_batch_random_reset_prepares_requested_state(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.reset(0, key="r", basis="Z"),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        batch = FaultScopeSimulator(circuit).run_batch(
            shots=128,
            rng=random.Random(22),
        )
        self.assertGreater(batch.measurements["r"].bit_count(), 0)
        self.assertLess(batch.measurements["r"].bit_count(), batch.shots)
        self.assertEqual(batch.measurements["m"], 0)

    def test_batch_detector_and_observable_masks_are_recorded(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.x(0),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=4),
                Operation.observable_include(1, ("m",)),
            ],
        )
        batch = FaultScopeSimulator(circuit).run_batch(
            shots=8,
            rng=random.Random(17),
        )
        self.assertEqual(batch.detectors[4], batch.all_mask)
        self.assertEqual(batch.observables[1], batch.all_mask)
        self.assertEqual(batch.detector_bit(4, 3), 1)
        self.assertEqual(batch.observable_bit(1, 5), 1)

    def test_batch_measurement_bit_flip_masks_are_recorded(self) -> None:
        location = NoiseLocation(
            id="mflip",
            model=MeasurementBitFlip(),
            rate=1.0,
            qubits=(0,),
            tags={"gate": "measure"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.measure(0, key="m", basis="Z", noise=location),
            ],
        )
        batch = FaultScopeSimulator(circuit).run_batch(
            shots=7,
            rng=random.Random(19),
        )

        self.assertEqual(batch.measurements["m"], batch.all_mask)
        self.assertEqual(batch.noise_event_masks["mflip"], batch.all_mask)


def _default_batch_loss(batch, corrections) -> int:
    return logical_residual_loss_mask(
        batch.observables,
        corrections,
        all_mask=batch.all_mask,
    )


def _stim_batch_for_circuit(
    circuit: Circuit,
    *,
    shots: int,
    seed: int,
) -> SampleBatch:
    stim_circuit, key_order = to_stim_circuit(circuit)
    samples = stim_circuit.compile_sampler(seed=seed).sample(shots)
    detector_flips, observable_flips = stim_circuit.compile_m2d_converter().convert(
        measurements=samples,
        separate_observables=True,
    )
    measurement_batch = measurement_batch_from_stim_samples(samples, key_order)
    detectors: list[Detector] = []
    for operation in circuit.operations:
        if operation.kind != "detector":
            continue
        detector_id = operation.metadata.get("detector_id")
        if detector_id is None:
            detector_id = len(detectors)
        detectors.append(Detector(id=int(detector_id), measurement_keys=()))
    observables = tuple(
        LogicalObservable(id=int(operation.observable_id))
        for operation in circuit.operations
        if operation.kind == "observable_include" and operation.observable_id is not None
    )
    dem_batch = dem_batch_from_stim_samples(
        detector_flips,
        observable_flips,
        detectors=tuple(detectors),
        observables=observables,
    )
    return SampleBatch(
        shots=shots,
        all_mask=(1 << shots) - 1,
        x_frame=(),
        z_frame=(),
        measurements=measurement_batch.measurements,
        detectors=dem_batch.detectors,
        observables=dem_batch.observables,
        noise_event_masks={},
    )


class NativePackedSamplerTests(unittest.TestCase):
    def _native_sampler_or_skip(self, circuit: Circuit, *, observables=()):
        try:
            return compile_native_sampler(circuit, observables=observables)
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native extension unavailable: {exc}")

    def _surface_decoder_or_skip(self, example):
        try:
            return example.make_decoder()
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_native_backend_matches_stim_batch_sampler_masks(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=1.0,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
                Operation.observable_include(0, ("m",)),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        native_batch = sampler.sample(shots=9, seed=123)
        stim_batch = _stim_batch_for_circuit(
            circuit,
            shots=9,
            seed=456,
        )

        self.assertEqual(native_batch.measurements, stim_batch.measurements)
        self.assertEqual(native_batch.detectors, stim_batch.detectors)
        self.assertEqual(native_batch.observables, stim_batch.observables)
        self.assertEqual(native_batch.noise_event_masks["x0"], native_batch.all_mask)

    def test_backend_keyword_is_not_accepted(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[Operation.measure(0, key="m", basis="Z")],
        )
        dem = DetectorErrorModel(detectors=(), observables=(), edges=())

        with self.assertRaises(TypeError):
            compile_native_sampler(circuit, backend="native")
        with self.assertRaises(TypeError):
            generate_native_dem(circuit, backend="native")
        with self.assertRaises(TypeError):
            compile_native_dem_sampler(dem, backend="native")
        with self.assertRaises(TypeError):
            compile_native_sampler(circuit, strict=True)

    def test_native_backend_reports_missing_or_unsupported_extension(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[Operation.measure(0, key="m", basis="Z")],
        )
        try:
            __import__("faultscope._native")
        except ImportError:
            with self.assertRaises(UnsupportedNativeCircuitError):
                compile_native_sampler(circuit)
        else:
            sampler = compile_native_sampler(circuit)
            self.assertEqual(sampler.sample(shots=4, seed=1).shots, 4)

    def test_native_circuit_repeated_compilation_preserves_observables(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        first = self._native_sampler_or_skip(circuit)
        second = self._native_sampler_or_skip(circuit)

        self.assertEqual(
            first.sample(shots=129, seed=7).measurements,
            second.sample(shots=129, seed=7).measurements,
        )

        observable_sampler = self._native_sampler_or_skip(
            circuit,
            observables=(LogicalObservable(id=7, measurement_keys=("m",)),),
        )
        self.assertIn(7, observable_sampler.sample(shots=129, seed=7).observables)

    def test_native_backend_matches_deterministic_stim_masks(self) -> None:
        x_location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=1.0,
            qubits=(0,),
        )
        m_location = NoiseLocation(
            id="mflip",
            model=MeasurementBitFlip(),
            rate=1.0,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.reset(0, key="r", basis="Z"),
                Operation.noise(x_location),
                Operation.measure(0, key="m", basis="Z", noise=m_location),
                Operation.detector(("r", "m"), detector_id=3),
                Operation.observable_include(0, ("m",)),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        native_batch = sampler.sample(shots=17, seed=11)
        stim_batch = _stim_batch_for_circuit(
            circuit,
            shots=17,
            seed=12,
        )

        self.assertEqual(native_batch.x_frame, (native_batch.all_mask,))
        self.assertEqual(native_batch.z_frame, (0,))
        self.assertEqual(native_batch.measurements, stim_batch.measurements)
        self.assertEqual(native_batch.detectors, stim_batch.detectors)
        self.assertEqual(native_batch.observables, stim_batch.observables)
        self.assertEqual(native_batch.noise_event_masks["x0"], native_batch.all_mask)
        self.assertEqual(native_batch.noise_event_masks["mflip"], native_batch.all_mask)

    def test_native_batch_single_mask_accessors_match_bulk_properties(self) -> None:
        x_location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=1.0,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(x_location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.measure(0, key="unused", basis="Z"),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        native_batch = sampler.run_native_batch(17, 11)

        self.assertEqual(int(native_batch.x_mask(0)), native_batch.x_frame[0])
        self.assertEqual(int(native_batch.z_mask(0)), native_batch.z_frame[0])
        self.assertEqual(
            int(native_batch.measurement_mask("m")),
            native_batch.measurements["m"],
        )
        self.assertEqual(
            native_batch.measurement_masks(("m",)),
            {"m": native_batch.measurements["m"]},
        )
        with self.assertRaisesRegex(TypeError, "iterable of strings, not a string"):
            native_batch.measurement_masks("m")
        with self.assertRaisesRegex(TypeError, "iterable of strings"):
            native_batch.measurement_masks((1,))
        with self.assertRaisesRegex(ValueError, "unknown measurement key.*missing"):
            native_batch.measurement_masks(("missing",))

    def test_batch_forward_simulator_is_native_and_exposes_metadata(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.1,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        engine = FaultScopeSimulator(circuit)

        self.assertIs(engine.circuit, circuit)
        self.assertEqual(engine.observables, ())
        self.assertEqual(engine.locations["x0"].id, "x0")

        with self.assertRaises(ValueError):
            engine.run_batch(shots=4, rng=random.Random(1), seed=1)

    def test_surface_code_decoder_loss_matches_native_batch_accessors(self) -> None:
        example = make_large_rotated_surface_code_memory_example(
            distance=5,
            rounds=1,
        )
        decoder = self._surface_decoder_or_skip(example)
        sampler = self._native_sampler_or_skip(
            example.circuit,
            observables=example.observables,
        )
        native_batch = sampler.run_native_batch(64, 23)
        converted_batch = sampler.sample(shots=64, seed=23)
        native_corrections = decoder.decode_batch_masks(native_batch)
        converted_corrections = decoder.decode_batch_masks(converted_batch)

        self.assertEqual(
            _default_batch_loss(native_batch, native_corrections),
            _default_batch_loss(converted_batch, converted_corrections),
        )

    def test_surface_code_estimate_uses_default_logical_loss(self) -> None:
        example = make_large_rotated_surface_code_memory_example(
            distance=5,
            rounds=1,
        )
        decoder = self._surface_decoder_or_skip(example)
        sampler = self._native_sampler_or_skip(
            example.circuit,
            observables=example.observables,
        )
        batch = sampler.run_native_batch(96, 37)
        corrections = decoder.decode_batch_masks(batch)
        loss_mask = _default_batch_loss(batch, corrections)

        result = sampler.estimate(
            shots=96,
            seed=37,
            decoder=decoder,
        )

        self.assertEqual(result.mean_loss, loss_mask.bit_count() / 96)

    def test_repetition_code_estimate_uses_default_logical_loss(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(0, 1): 0.12, (1, 3): 0.09},
            measurement_error_rate={(0, 0): 0.04, (2, 2): 0.05},
        )
        sampler = self._native_sampler_or_skip(
            experiment.circuit,
            observables=experiment.observables,
        )

        result = sampler.estimate(
            shots=256,
            seed=49,
            decoder=experiment.decoder,
        )

        native_batch = sampler.run_native_batch(256, 49)
        expected_loss = _default_batch_loss(
            native_batch,
            experiment.decoder.decode_batch_masks(native_batch),
        )

        self.assertEqual(result.mean_loss, expected_loss.bit_count() / 256)

    def test_external_pauli_observable_matches_final_measurement(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(1, 2): 0.17, (2, 0): 0.09},
            measurement_error_rate=0.035,
        )
        circuit = Circuit(
            n_qubits=experiment.circuit.n_qubits,
            operations=[
                *experiment.circuit.operations,
                Operation.measure(
                    experiment.data_qubits[0],
                    key="final_d_0",
                    basis="Z",
                ),
            ],
        )
        sampler = self._native_sampler_or_skip(
            circuit,
            observables=experiment.observables,
        )
        batch = sampler.sample(shots=512, seed=12345)

        self.assertIn(0, batch.observables)
        self.assertEqual(batch.observables[0], batch.measurements["final_d_0"])

    def test_forward_estimate_uses_pymatching_batch_decoder(self) -> None:
        try:
            decoder = PyMatchingDecoder.from_dem(
                DetectorErrorModel(
                    detectors=(Detector(id=0, measurement_keys=("m",)),),
                    observables=(LogicalObservable(id=0, measurement_keys=("m",)),),
                    edges=(
                        DetectorErrorEdge(
                            probability=0.1,
                            detectors=(0,),
                            observables=(0,),
                            location_id="x0",
                            event="X",
                        ),
                    ),
                )
            )
        except ImportError as exc:
            self.skipTest(f"optional PyMatching dependencies are not installed: {exc}")

        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=1.0,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
                Operation.observable_include(0, ("m",)),
            ],
        )

        raw = FaultScopeSimulator(circuit).estimate(
            shots=16,
            seed=53,
        )
        decoded = FaultScopeSimulator(circuit).estimate(
            shots=16,
            seed=53,
            decoder=decoder,
        )

        self.assertEqual(raw.mean_loss, 1.0)
        self.assertEqual(decoded.mean_loss, 0.0)
        self.assertEqual(decoded.hotspots["x0"], 0.0)

    def test_native_backend_samples_random_measurement_masks(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        batch = sampler.sample(shots=1024, seed=12)
        ones = batch.measurements["m"].bit_count()

        self.assertGreater(ones, 350)
        self.assertLess(ones, 674)

    def test_native_backend_matches_auto_measurement_keys(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.measure(0, basis="Z"),
                Operation.measure(0, basis="Z"),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        batch = sampler.sample(shots=5, seed=13)

        self.assertEqual(set(batch.measurements), {"m0", "m1"})
        self.assertEqual(batch.measurements["m0"], 0)
        self.assertEqual(batch.measurements["m1"], 0)

    def test_measurement_only_sampling_matches_batch_measurements(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.x(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        sampler = compile_native_sampler(circuit)
        measurements = sampler.sample_measurements(shots=6, seed=14)
        batch = sampler.sample(shots=6, seed=14)

        self.assertEqual(measurements, batch.measurements)

    def test_measurement_only_sampling_applies_native_noise_fast_path(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=1.0,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        measurements = sampler.sample_measurements(shots=13, seed=22)

        self.assertEqual(measurements["m"], (1 << 13) - 1)

    def test_native_noise_event_counts_are_statistical(self) -> None:
        cases = (
            ("x_low_rate", BernoulliPauliNoise("X"), 0.001),
            ("depol_mid_rate", SingleQubitDepolarizing(), 0.025),
            ("channel_high_rate", PauliChannel({"X": 1.0, "Y": 2.0, "Z": 1.0}), 0.14),
        )
        shots = 50_000
        for case_index, (location_id, model, rate) in enumerate(cases):
            with self.subTest(location_id=location_id):
                location = NoiseLocation(
                    id=location_id,
                    model=model,
                    rate=rate,
                    qubits=(0,),
                )
                circuit = Circuit(
                    n_qubits=1,
                    operations=[
                        Operation.noise(location),
                        Operation.measure(0, key="m", basis="Z"),
                    ],
                )
                sampler = self._native_sampler_or_skip(circuit)
                batch = sampler.sample(shots=shots, seed=90 + case_index)

                observed = batch.noise_event_masks[location_id].bit_count()
                _assert_binomial_count_close(
                    self,
                    observed,
                    shots=shots,
                    probability=rate,
                )

    def test_native_hotspot_estimate_returns_aggregates_and_top_cache(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.2,
            qubits=(0,),
            tags={"round": 2, "gate": "idle", "operation": "noise"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        result = sampler.estimate(
            shots=40_000,
            seed=91,
            loss_mask_fn=lambda batch: batch.measurements["m"],
            top_k=1,
        )

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.sensitivities["x0"], 1.0, delta=0.08)
        self.assertAlmostEqual(result.hotspots["x0"], 1.0, delta=0.08)
        self.assertAlmostEqual(result.by_qubit[0], result.hotspots["x0"])
        self.assertAlmostEqual(result.by_round[2], result.hotspots["x0"])
        self.assertAlmostEqual(result.by_gate["idle"], result.hotspots["x0"])
        self.assertAlmostEqual(result.by_operation["noise"], result.hotspots["x0"])
        self.assertEqual(result.top_hotspots(1)[0].location_id, "x0")

    def test_native_rejects_nonprimitive_tags_and_batch_does_not_fallback(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.2,
            qubits=(0,),
            tags={"round": (1, 2)},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )

        with self.assertRaises(UnsupportedNativeCircuitError):
            compile_native_sampler(circuit)

        with self.assertRaises(ValueError):
            FaultScopeSimulator(circuit).estimate(
                shots=2_000,
                seed=92,
                loss_mask_fn=lambda batch: batch.measurements["m"],
            )


class NativeDetectorErrorModelTests(unittest.TestCase):
    def _require_native_dem(self) -> None:
        try:
            __import__("faultscope._native")
        except ImportError as exc:
            self.skipTest(f"native extension unavailable: {exc}")

    def test_native_dem_generator_matches_repetition_stim_dem(self) -> None:
        self._require_native_dem()
        experiment = make_repetition_code_experiment(
            distance=3,
            rounds=1,
            data_error_rate=0.1,
            measurement_error_rate=0.0,
        )
        observable = LogicalObservable(id=0, measurement_keys=("final_d_0",))
        circuit = final_data_measurement_circuit(
            experiment.circuit,
            (experiment.data_qubits[0],),
            basis="Z",
            prefix="final_d",
        )
        native = generate_native_dem(
            circuit,
            detectors=experiment.detectors,
            observables=(observable,),
        )
        stim_circuit, _ = to_stim_circuit(
            with_dem_declarations(
                circuit,
                detectors=experiment.detectors,
                observables=(observable,),
            )
        )
        stim_dem = stim_circuit.detector_error_model(decompose_errors=False)

        self.assertEqual(faultscope_dem_error_edges(native), stim_dem_error_edges(stim_dem))

    def test_native_dem_generator_splits_pauli_channel_edges(self) -> None:
        self._require_native_dem()
        location = NoiseLocation(
            id="pc",
            model=PauliChannel({"X": 1.0, "Y": 3.0}),
            rate=0.4,
            qubits=(0,),
            tags={"gate": "channel"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )

        native = generate_native_dem(circuit)
        by_event = {edge.event: edge for edge in native.edges}

        self.assertEqual(set(by_event), {"X", "Y"})
        self.assertAlmostEqual(by_event["X"].probability, 0.1)
        self.assertAlmostEqual(by_event["Y"].probability, 0.3)
        self.assertEqual(by_event["Y"].detectors, (0,))
        self.assertEqual(dict(by_event["Y"].tags), {"gate": "channel"})

    def test_native_dem_reports_missing_extension(self) -> None:
        dem = DetectorErrorModel(detectors=(), observables=(), edges=())
        circuit = Circuit(n_qubits=0, operations=[])
        with mock.patch("importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaises(UnsupportedNativeCircuitError):
                generate_native_dem(circuit)
            with self.assertRaises(UnsupportedNativeCircuitError):
                compile_native_dem_sampler(dem)

    def test_native_dem_sampler_samples_packed_masks(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=1.0,
                    detectors=(0,),
                    observables=(0,),
                    location_id="certain",
                    event="X",
                ),
                DetectorErrorEdge(
                    probability=0.0,
                    detectors=(0,),
                    observables=(),
                    location_id="never",
                    event="Z",
                ),
            ),
        )
        sampler = compile_native_dem_sampler(dem)
        batch = sampler.run_batch(shots=9, seed=123)

        all_mask = (1 << 9) - 1
        self.assertEqual(batch.all_mask, all_mask)
        self.assertEqual(batch.edge_event_masks[0], all_mask)
        self.assertEqual(batch.edge_event_masks[1], 0)
        self.assertEqual(batch.detectors[0], all_mask)
        self.assertEqual(batch.observables[0], all_mask)

        compact_batch = sampler.run_batch(
            shots=9,
            seed=123,
            return_edge_events=False,
        )

        self.assertEqual(compact_batch.all_mask, all_mask)
        self.assertEqual(compact_batch.edge_event_masks, {})
        self.assertEqual(compact_batch.detectors[0], all_mask)
        self.assertEqual(compact_batch.observables[0], all_mask)

    def test_native_dem_sampler_edge_counts_are_statistical(self) -> None:
        self._require_native_dem()
        probabilities = (0.001, 0.025, 0.14)
        dem = DetectorErrorModel(
            detectors=(),
            observables=(),
            edges=tuple(
                DetectorErrorEdge(
                    probability=probability,
                    detectors=(),
                    observables=(),
                    location_id=f"edge_{index}",
                    event="X",
                )
                for index, probability in enumerate(probabilities)
            ),
        )
        sampler = compile_native_dem_sampler(dem)
        shots = 50_000
        batch = sampler.run_batch(shots=shots, seed=57)

        for edge_index, probability in enumerate(probabilities):
            observed = batch.edge_event_masks[edge_index].bit_count()
            _assert_binomial_count_close(
                self,
                observed,
                shots=shots,
                probability=probability,
            )

    def test_native_dem_default_estimate_matches_logical_edge_gradient(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(),
                    observables=(0,),
                    location_id="logical_edge",
                    event="L",
                    tags={"round": 1},
                ),
            ),
        )
        sampler = compile_native_dem_sampler(dem)
        result = sampler.estimate_default(shots=40_000, seed=54)

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.edge_sensitivities[0], 1.0, delta=0.08)
        self.assertAlmostEqual(result.sensitivities["logical_edge"], 1.0, delta=0.08)
        self.assertEqual(result.by_round[1], result.hotspots["logical_edge"])
        self.assertEqual(result.top_edges(1)[0].edge_index, 0)
        self.assertEqual(result.top_hotspots(1)[0].location_id, "logical_edge")

    def test_native_dem_estimate_accepts_native_decoder_fast_path(self) -> None:
        self._require_native_dem()
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
        decoder = NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(0,))
        sampler = compile_native_dem_sampler(dem)

        native_result = sampler.estimate(shots=20_000, seed=55, decoder=decoder)
        default_result = sampler.estimate(shots=20_000, seed=55)

        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(native_result.mean_loss, default_result.mean_loss)
        self.assertEqual(native_result.edge_hotspots, default_result.edge_hotspots)

    def test_native_dem_decoder_with_python_loss_uses_slow_compat_path(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(DetectorErrorEdge(0.2, (0,), (0,), "edge0", "X"),),
        )
        decoder = NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(0,))

        result = compile_native_dem_sampler(dem).estimate(
            shots=128,
            seed=56,
            decoder=decoder,
            loss_mask_fn=lambda batch, corrections: corrections[0],
        )

        self.assertEqual(decoder.python_decode_call_count, 1)
        self.assertEqual(result.mean_loss, 0.0)

    def test_graphlike_detector_copy_decoder_from_dem_uses_native_fast_path(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(DetectorErrorEdge(0.2, (0,), (0,), "edge0", "X"),),
        )
        decoder = NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

        result = compile_native_dem_sampler(dem).estimate(
            shots=4096,
            seed=59,
            decoder=decoder,
        )

        self.assertEqual(decoder.name, "graphlike-detector-copy")
        self.assertEqual(decoder.detector_ids, (0,))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertEqual(decoder.python_decode_call_count, 0)
        self.assertEqual(result.mean_loss, 0.0)

    def test_graphlike_detector_copy_decoder_from_circuit_matches_from_dem(self) -> None:
        self._require_native_dem()
        location = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.2, (0,))
        circuit = Circuit(
            1,
            (
                Operation.noise(location),
                Operation.measure(0, key="m0", basis="Z"),
                Operation.detector(("m0",), detector_id=0),
                Operation.observable_include(0, ("m0",)),
            ),
        )
        dem = generate_native_dem(circuit)
        from_dem = NativeGraphlikeDetectorCopyDecoder.from_dem(dem)
        from_circuit = NativeGraphlikeDetectorCopyDecoder.from_circuit(circuit)

        self.assertEqual(from_circuit.name, from_dem.name)
        self.assertEqual(from_circuit.detector_ids, from_dem.detector_ids)
        self.assertEqual(from_circuit.observable_ids, from_dem.observable_ids)

    def test_graphlike_detector_copy_decoder_with_python_loss_uses_slow_path(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(DetectorErrorEdge(0.2, (0,), (0,), "edge0", "X"),),
        )
        decoder = NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

        result = compile_native_dem_sampler(dem).estimate(
            shots=128,
            seed=60,
            decoder=decoder,
            loss_mask_fn=lambda batch, corrections: corrections[0],
        )

        self.assertEqual(decoder.python_decode_call_count, 1)
        self.assertGreaterEqual(result.mean_loss, 0.0)

    def test_graphlike_detector_copy_decoder_rejects_ambiguous_mapping(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=0, measurement_keys=()),
                Detector(id=1, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(0.1, (0,), (0,), "edge0", "X"),
                DetectorErrorEdge(0.1, (1,), (0,), "edge1", "X"),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "multiple single-detector candidate edges for observable id 0",
        ):
            NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

    def test_native_dem_custom_loss_uses_rust_hotspot_aggregation(self) -> None:
        self._require_native_dem()
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
                    tags={"gate": "idle"},
                ),
            ),
        )
        sampler = compile_native_dem_sampler(dem)
        result = sampler.estimate(
            shots=40_000,
            seed=58,
            loss_mask_fn=lambda batch, corrections: batch.detectors[0],
            top_k=1,
        )

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.edge_sensitivities[0], 1.0, delta=0.08)
        self.assertAlmostEqual(result.sensitivities["edge0"], 1.0, delta=0.08)
        self.assertAlmostEqual(result.by_gate["idle"], result.hotspots["edge0"])
        self.assertEqual(result.top_edges(1)[0].edge_index, 0)

    def test_native_dem_problem_views_are_available(self) -> None:
        self._require_native_dem()
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=5, measurement_keys=(), coords=(5.0, 0.0)),
                Detector(id=2, measurement_keys=(), coords=(2.0, 0.0)),
            ),
            observables=(LogicalObservable(id=7),),
            edges=(
                DetectorErrorEdge(0.1, (2, 9), (7,), "a", "X"),
                DetectorErrorEdge(0.2, (5,), (), "b", "Z"),
            ),
        )

        indexed = dem.compile_indexed()
        graphlike = dem.compile_graphlike_problem()
        binary = dem.compile_binary_linear_problem()

        self.assertIsInstance(indexed, IndexedDem)
        self.assertIsInstance(graphlike, GraphlikeDecodingProblem)
        self.assertIsInstance(binary, BinaryLinearDecodingProblem)
        self.assertEqual(indexed.detector_ids, (5, 2, 9))
        self.assertEqual(indexed.detector_coords, ((5.0, 0.0), (2.0, 0.0), ()))
        self.assertEqual(indexed.observable_ids, (7,))
        self.assertEqual(indexed.detector_count, 3)
        self.assertEqual(indexed.observable_count, 1)
        self.assertEqual(indexed.edge_count, 2)
        self.assertEqual(indexed.edges[0].detectors, (1, 2))
        self.assertEqual(indexed.edge_summary[0]["dem_edge_index"], 0)
        self.assertEqual(indexed.edge_summary[0]["detectors"], (1, 2))
        self.assertEqual(indexed.edge_summary[0]["observables"], (0,))
        self.assertIn("IndexedDem(detector_count=3", repr(indexed))
        self.assertEqual(graphlike.edge_count, 2)
        self.assertEqual(graphlike.detector_coords, indexed.detector_coords)
        self.assertEqual(graphlike.detector_count, 3)
        self.assertEqual(graphlike.observable_count, 1)
        self.assertEqual(graphlike.edges[0].fault_observables, (0,))
        self.assertEqual(graphlike.edge_summary[0]["fault_observables"], (0,))
        self.assertIn("GraphlikeDecodingProblem(detector_count=3", repr(graphlike))
        self.assertIsInstance(binary.h, SparseBinaryMatrix)
        self.assertEqual(binary.detector_coords, indexed.detector_coords)
        self.assertEqual(binary.detector_count, 3)
        self.assertEqual(binary.observable_count, 1)
        self.assertEqual(binary.h.entries, ((1, 0), (2, 0), (0, 1)))
        self.assertEqual(binary.h.entry_count, 3)
        self.assertIn("SparseBinaryMatrix(row_count=3", repr(binary.h))
        self.assertEqual(binary.f.entries, ((0, 0),))
        self.assertEqual(binary.edge_summary[0]["dem_edge_index"], 0)
        self.assertEqual(binary.edge_summary[0]["edge_index"], 0)
        self.assertIn("BinaryLinearDecodingProblem(detector_count=3", repr(binary))
        self.assertTrue(dem.is_graphlike())

        bad = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(),
            edges=(DetectorErrorEdge(0.1, (0, 1, 2), (), "bad", "X"),),
        )
        self.assertFalse(bad.is_graphlike())
        with self.assertRaises(ValueError):
            bad.compile_graphlike_problem()

    def test_native_dem_sampler_compiles_directly_from_circuit(self) -> None:
        self._require_native_dem()
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.25,
            qubits=(0,),
            tags={"round": 1},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
                Operation.observable_include(0, ("m",)),
            ],
        )
        dem = generate_native_dem(circuit)
        regular_sampler = compile_native_dem_sampler(dem)
        generator = compile_native_dem_generator(circuit)
        direct_sampler = compile_native_dem_sampler_from_circuit(circuit)
        direct_light_sampler = compile_native_dem_sampler_from_circuit(
            circuit,
            materialize_dem=False,
        )

        def edge_rows(model: DetectorErrorModel) -> list[tuple[object, ...]]:
            return [
                (
                    edge.probability,
                    edge.detectors,
                    edge.observables,
                    edge.location_id,
                    edge.event,
                    dict(edge.tags),
                )
                for edge in model.edges
            ]

        self.assertIsInstance(direct_sampler.dem, DetectorErrorModel)
        self.assertEqual(edge_rows(direct_sampler.dem), edge_rows(dem))
        self.assertEqual(edge_rows(generator.generate_dem()), edge_rows(dem))
        self.assertIsNone(direct_light_sampler.dem)

        regular_batch = regular_sampler.run_batch(shots=256, seed=123)
        direct_batch = direct_sampler.run_batch(shots=256, seed=123)
        self.assertEqual(direct_batch.detectors, regular_batch.detectors)
        self.assertEqual(direct_batch.observables, regular_batch.observables)
        self.assertEqual(direct_batch.edge_event_masks, regular_batch.edge_event_masks)

        direct_light_batch = direct_light_sampler.run_batch(shots=256, seed=123)
        self.assertEqual(direct_light_batch.detectors, regular_batch.detectors)
        self.assertEqual(direct_light_batch.observables, regular_batch.observables)
        self.assertEqual(direct_light_batch.edge_event_masks, regular_batch.edge_event_masks)
        generator_light_batch = generator.compile_sampler(
            materialize_dem=False,
        ).run_batch(shots=256, seed=123)
        self.assertEqual(generator_light_batch.detectors, regular_batch.detectors)
        self.assertEqual(generator_light_batch.observables, regular_batch.observables)
        self.assertEqual(generator_light_batch.edge_event_masks, regular_batch.edge_event_masks)

        direct_result = direct_sampler.estimate_default(shots=256, seed=123)
        self.assertEqual(direct_result.dem.edges[0].location_id, "x0")
        self.assertEqual(dict(direct_result.dem.edges[0].tags), {"round": 1})
        with self.assertRaises(ValueError):
            direct_light_sampler.estimate_default(shots=256, seed=123)
        direct_light_native_batch = direct_light_sampler.run_native_batch(
            shots=256,
            seed=123,
        )
        with self.assertRaises(ValueError):
            direct_light_sampler.estimate_hotspots(
                direct_light_native_batch,
                direct_light_native_batch.detectors[0],
            )

        direct_edge = direct_result.top_edges(1)[0]
        direct_graph_edge = direct_result.detector_graph_hotspots.edge_hotspots[0]
        self.assertEqual(direct_edge.edge_index, direct_graph_edge.edge_index)

    def test_dem_faultscope_simulator_compiles_from_circuit_with_faultscope_shape(self) -> None:
        self._require_native_dem()
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.25,
            qubits=(0,),
            tags={"round": 1},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
                Operation.observable_include(0, ("m",)),
            ],
        )

        simulator = DemFaultScopeSimulator(circuit)
        direct_sampler = compile_native_dem_sampler_from_circuit(circuit)
        direct_batch = direct_sampler.run_batch(shots=256, seed=123)

        batch = simulator.run_batch(shots=256, seed=123)
        sample_batch = simulator.sample(256, seed=123)
        result = simulator.estimate(shots=4096, seed=123)

        self.assertIs(simulator.circuit, circuit)
        self.assertIsInstance(simulator.dem, DetectorErrorModel)
        self.assertEqual(simulator.edge_count, direct_sampler.edge_count)
        self.assertIsInstance(batch, DemSampleBatch)
        self.assertEqual(batch.detectors, direct_batch.detectors)
        self.assertEqual(batch.observables, direct_batch.observables)
        self.assertEqual(batch.edge_event_masks, direct_batch.edge_event_masks)
        self.assertEqual(sample_batch.detectors, direct_batch.detectors)
        self.assertAlmostEqual(result.mean_loss, 0.25, delta=0.05)
        self.assertEqual(result.dem.edges[0].location_id, "x0")
        self.assertIn("DemFaultScopeSimulator", repr(simulator))

    def test_native_dem_reparses_mutated_duck_typed_operation(self) -> None:
        self._require_native_dem()
        first = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.25, (0,))
        second = NoiseLocation("x0", BernoulliPauliNoise("X"), 0.5, (0,))
        duck_noise_op = SimpleNamespace(
            kind="noise",
            qubits=(0,),
            noise_location=first,
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                duck_noise_op,
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )

        self.assertEqual(generate_native_dem(circuit).edges[0].probability, 0.25)
        duck_noise_op.noise_location = second
        self.assertEqual(generate_native_dem(circuit).edges[0].probability, 0.5)

    def test_native_operation_with_duck_typed_noise_location_is_reparsed(self) -> None:
        self._require_native_dem()
        duck_location = SimpleNamespace(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.25,
            qubits=(0,),
            tags={},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(duck_location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )

        self.assertEqual(generate_native_dem(circuit).edges[0].probability, 0.25)
        duck_location.rate = 0.5
        self.assertEqual(generate_native_dem(circuit).edges[0].probability, 0.5)

    def test_native_noise_location_tags_getter_returns_copy(self) -> None:
        self._require_native_dem()
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.25,
            qubits=(0,),
            tags={"round": 1},
        )
        tags = location.tags
        tags["round"] = 2
        self.assertEqual(dict(location.tags), {"round": 1})

        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
            ],
        )
        self.assertEqual(dict(generate_native_dem(circuit).edges[0].tags), {"round": 1})

    def test_native_dem_generator_rejects_random_ideal_measurement(self) -> None:
        self._require_native_dem()
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        with self.assertRaises(UnsupportedNativeCircuitError):
            generate_native_dem(
                circuit,
                detectors=(Detector(id=0, measurement_keys=("m",)),),
            )


class DetectorErrorModelTests(unittest.TestCase):
    def test_single_x_error_generates_detector_and_logical_edge(self) -> None:
        location = NoiseLocation(
            id="x0",
            model=BernoulliPauliNoise("X"),
            rate=0.125,
            qubits=(0,),
            tags={"qubit": 0, "round": 0, "gate": "idle"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        dem = DetectorErrorModelGenerator(
            circuit,
            detectors=(Detector(id=0, measurement_keys=("m",), coords=(0.0,)),),
            observables=(LogicalObservable(id=0, pauli_qubits=(0,), pauli="Z"),),
        ).generate()

        self.assertEqual(len(dem.edges), 1)
        edge = dem.edges[0]
        self.assertEqual(edge.location_id, "x0")
        self.assertEqual(edge.event, "X")
        self.assertAlmostEqual(edge.probability, 0.125)
        self.assertEqual(edge.detectors, (0,))
        self.assertEqual(edge.observables, (0,))
        self.assertIn("error(0.125) D0 L0", dem.to_dem_text())

    def test_repetition_code_generates_dem_edges(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=3,
            rounds=1,
            data_error_rate=0.1,
            measurement_error_rate=0.01,
        )
        dem = DetectorErrorModelGenerator(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=experiment.observables,
        ).generate()

        by_location = dem.edges_by_location()
        self.assertIn("data_r0_q0", by_location)
        self.assertIn("data_r0_q1", by_location)
        self.assertIn("meas_r0_c0", by_location)
        self.assertEqual(by_location["data_r0_q0"][0].detectors, (0,))
        self.assertEqual(by_location["data_r0_q0"][0].observables, (0,))
        self.assertEqual(by_location["data_r0_q1"][0].detectors, (0, 1))
        self.assertEqual(by_location["meas_r0_c0"][0].detectors, (0,))

    def test_dem_rejects_random_ideal_measurement(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        with self.assertRaises(ValueError):
            DetectorErrorModelGenerator(
                circuit,
                detectors=(Detector(id=0, measurement_keys=("m",)),),
            ).generate()

    def test_projects_location_sensitivities_to_detector_graph(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=3,
            rounds=1,
            data_error_rate=0.1,
            measurement_error_rate=0.01,
        )
        dem = DetectorErrorModelGenerator(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=experiment.observables,
        ).generate()

        graph = dem.project_sensitivities_to_detector_graph(
            {
                "data_r0_q0": 3.0,
                "data_r0_q1": -2.0,
                "meas_r0_c0": 1.0,
            }
        )

        self.assertAlmostEqual(graph.by_detector_edge[((0,), (0,))], 3.0)
        self.assertAlmostEqual(graph.signed_by_detector_edge[((0,), (0,))], 3.0)
        self.assertAlmostEqual(graph.by_detector_edge[((0, 1), ())], 2.0)
        self.assertAlmostEqual(graph.signed_by_detector_edge[((0, 1), ())], -2.0)
        self.assertAlmostEqual(graph.by_detector_edge[((0,), ())], 1.0)
        self.assertAlmostEqual(graph.by_detector[0], 5.0)
        self.assertAlmostEqual(graph.signed_by_detector[0], 3.0)
        self.assertAlmostEqual(graph.by_detector[1], 1.0)
        self.assertAlmostEqual(graph.signed_by_detector[1], -1.0)
        self.assertAlmostEqual(graph.by_observable[0], 3.0)
        self.assertEqual(graph.top_edges(1)[0].location_id, "data_r0_q0")

    def test_splits_location_sensitivity_across_pauli_channel_edges(self) -> None:
        location = NoiseLocation(
            id="pc",
            model=PauliChannel({"X": 1.0, "Y": 3.0}),
            rate=0.4,
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
        dem = DetectorErrorModelGenerator(circuit).generate()
        graph = dem.project_sensitivities_to_detector_graph({"pc": 8.0})
        by_event = {edge.event: edge for edge in graph.edge_hotspots}

        self.assertAlmostEqual(by_event["X"].sensitivity, 2.0)
        self.assertAlmostEqual(by_event["Y"].sensitivity, 6.0)
        self.assertAlmostEqual(graph.by_detector_edge[((0,), ())], 8.0)

    def test_circuit_detector_operations_generate_measurement_noise_dem(self) -> None:
        location = NoiseLocation(
            id="mflip",
            model=MeasurementBitFlip(),
            rate=0.2,
            qubits=(0,),
            tags={"gate": "measure"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.measure(0, key="m", basis="Z", noise=location),
                Operation.detector(("m",), detector_id=5, coords=(1.0, 2.0)),
                Operation.observable_include(2, ("m",)),
            ],
        )
        generator = DetectorErrorModelGenerator(circuit)
        self.assertIsInstance(generator, DetectorErrorModelGenerator)
        self.assertEqual(generator.detectors[0].id, 5)
        self.assertEqual(generator.observables[0].id, 2)
        dem = generator.generate()

        self.assertEqual(len(dem.detectors), 1)
        self.assertEqual(dem.detectors[0].id, 5)
        self.assertEqual(dem.detectors[0].coords, (1.0, 2.0))
        self.assertEqual(len(dem.observables), 1)
        self.assertEqual(dem.observables[0].id, 2)
        self.assertEqual(len(dem.edges), 1)
        edge = dem.edges[0]
        self.assertEqual(edge.location_id, "mflip")
        self.assertEqual(edge.event, True)
        self.assertAlmostEqual(edge.probability, 0.2)
        self.assertEqual(edge.detectors, (5,))
        self.assertEqual(edge.observables, (2,))
        self.assertIn("detector(1, 2) D5", dem.to_dem_text())


class DemHotspotEstimatorTests(unittest.TestCase):
    def test_dem_edge_hotspot_estimates_logical_edge_gradient(self) -> None:
        dem = DetectorErrorModel(
            detectors=(),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(),
                    observables=(0,),
                    location_id="logical_edge",
                    event="L",
                    tags={"round": 1, "operation": "dem_error"},
                ),
            ),
        )

        result = DemHotspotEstimator(dem).estimate(shots=40_000, seed=51)

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.edge_sensitivities[0], 1.0, delta=0.08)
        self.assertAlmostEqual(result.sensitivities["logical_edge"], 1.0, delta=0.08)
        self.assertAlmostEqual(result.hotspots["logical_edge"], 1.0, delta=0.08)
        self.assertEqual(result.by_round[1], result.hotspots["logical_edge"])
        self.assertEqual(result.by_operation["dem_error"], result.hotspots["logical_edge"])
        self.assertEqual(result.top_edges(1)[0].location_id, "logical_edge")

    def test_native_dem_simulator_run_batch_accepts_python_rng(self) -> None:
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(),
            edges=(
                DetectorErrorEdge(
                    probability=0.5,
                    detectors=(0,),
                    observables=(),
                    location_id="detector_edge",
                    event="X",
                ),
            ),
        )
        simulator = DemHotspotEstimator(dem)

        batch = simulator.run_batch(
            shots=16,
            rng=random.Random(123),
            return_edge_events=False,
        )

        self.assertIs(simulator.dem, dem)
        self.assertEqual(batch.shots, 16)
        self.assertEqual(batch.edge_event_masks, {})
        self.assertIn(0, batch.detectors)
        with self.assertRaises(ValueError):
            simulator.run_batch(shots=16, rng=random.Random(123), seed=1)

    def test_dem_decoder_correction_can_remove_logical_failure(self) -> None:
        class CopyDetectorDecoder:
            @staticmethod
            def decode_batch_masks(batch):
                return {0: batch.detectors[0]}

        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.35,
                    detectors=(0,),
                    observables=(0,),
                    location_id="correctable_edge",
                    event="X",
                ),
            ),
        )

        result = DemHotspotEstimator(dem).estimate(
            shots=10_000,
            seed=52,
            decoder=CopyDetectorDecoder(),
        )

        self.assertEqual(result.mean_loss, 0.0)
        self.assertEqual(result.edge_sensitivities[0], 0.0)
        self.assertEqual(result.hotspots["correctable_edge"], 0.0)

    def test_dem_location_sensitivity_uses_edge_probability_weights(self) -> None:
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(),
                    observables=(0,),
                    location_id="multi_event_location",
                    event="logical",
                    tags={"gate": "idle"},
                ),
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(0,),
                    observables=(),
                    location_id="multi_event_location",
                    event="detector_only",
                    tags={"gate": "idle"},
                ),
            ),
        )

        result = DemHotspotEstimator(dem).estimate(shots=40_000, seed=53)

        self.assertAlmostEqual(result.edge_sensitivities[0], 1.0, delta=0.08)
        self.assertAlmostEqual(result.edge_sensitivities[1], 0.0, delta=0.08)
        self.assertAlmostEqual(
            result.sensitivities["multi_event_location"],
            0.5,
            delta=0.08,
        )
        self.assertAlmostEqual(
            result.detector_graph_hotspots.by_observable[0],
            abs(result.edge_sensitivities[0]),
        )
        self.assertAlmostEqual(result.by_gate["idle"], result.hotspots["multi_event_location"])


class _FakeMatrix:
    def __init__(self, args, shape=None, dtype=None) -> None:
        self.args = args
        self.shape = shape
        self.dtype = dtype


class _FakeSparse:
    @staticmethod
    def csc_matrix(args, shape=None, dtype=None) -> _FakeMatrix:
        return _FakeMatrix(args, shape=shape, dtype=dtype)


class _FakeNumpy:
    uint8 = int

    @staticmethod
    def array(values, dtype=None):
        return tuple(values)


class _FakeMatching:
    def decode(self, syndrome):
        return [int(syndrome[0]) if syndrome else 0]

    def decode_batch(self, syndromes):
        return [[int(row[0]) if len(row) else 0] for row in syndromes]


class _FakePyMatching:
    calls = []

    class Matching:
        @staticmethod
        def from_check_matrix(h, **kwargs):
            _FakePyMatching.calls.append((h, kwargs))
            return _FakeMatching()


class PyMatchingDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakePyMatching.calls.clear()

    def _build_dem(self) -> DetectorErrorModel:
        return DetectorErrorModel(
            detectors=(
                Detector(id=0, measurement_keys=()),
                Detector(id=1, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.1,
                    detectors=(0,),
                    observables=(0,),
                    location_id="x0",
                    event="X",
                ),
                DetectorErrorEdge(
                    probability=0.2,
                    detectors=(0, 1),
                    observables=(),
                    location_id="x1",
                    event="X",
                ),
            ),
        )

    def test_builds_pymatching_decoder_from_graphlike_dem(self) -> None:
        decoder = PyMatchingDecoder.from_dem(
            self._build_dem(),
            pymatching_module=_FakePyMatching,
            numpy_module=_FakeNumpy,
            scipy_sparse_module=_FakeSparse,
        )

        self.assertEqual(decoder.detector_ids, (0, 1))
        self.assertEqual(decoder.observable_ids, (0,))
        self.assertEqual(decoder.edge_count, 2)
        self.assertEqual(len(_FakePyMatching.calls), 1)

        h, kwargs = _FakePyMatching.calls[0]
        self.assertEqual(h.shape, (2, 2))
        self.assertEqual(h.args, ([1, 1, 1], ([0, 0, 1], [0, 1, 1])))
        self.assertEqual(kwargs["faults_matrix"].shape, (1, 2))
        self.assertEqual(kwargs["faults_matrix"].args, ([1], ([0], [0])))
        self.assertEqual(kwargs["weights"], (2.1972245773362196, 1.3862943611198906))
        self.assertEqual(kwargs["error_probabilities"], (0.1, 0.2))

    def test_decodes_single_and_batch_records(self) -> None:
        decoder = PyMatchingDecoder.from_dem(
            self._build_dem(),
            pymatching_module=_FakePyMatching,
            numpy_module=_FakeNumpy,
            scipy_sparse_module=_FakeSparse,
        )

        self.assertEqual(decoder.decode_detector_record({0: 1, 1: 0}), {0: 1})
        self.assertEqual(
            decoder.decode_batch_detector_records([{0: 0, 1: 1}, {0: 1, 1: 1}]),
            [{0: 0}, {0: 1}],
        )

    def test_decodes_bit_packed_detector_masks(self) -> None:
        decoder = PyMatchingDecoder.from_dem(
            self._build_dem(),
            pymatching_module=_FakePyMatching,
            numpy_module=_FakeNumpy,
            scipy_sparse_module=_FakeSparse,
        )

        self.assertEqual(decoder.decode_batch_masks({0: 0b1010, 1: 0}, shots=4), {0: 0b1010})

    def test_rejects_dem_hyperedge(self) -> None:
        dem = DetectorErrorModel(
            detectors=(
                Detector(id=0, measurement_keys=()),
                Detector(id=1, measurement_keys=()),
                Detector(id=2, measurement_keys=()),
            ),
            observables=(),
            edges=(
                DetectorErrorEdge(
                    probability=0.1,
                    detectors=(0, 1, 2),
                    observables=(),
                    location_id="bad",
                    event="X",
                ),
            ),
        )

        with self.assertRaises(UnsupportedPyMatchingDemError):
            PyMatchingDecoder.from_dem(
                dem,
                pymatching_module=_FakePyMatching,
                numpy_module=_FakeNumpy,
                scipy_sparse_module=_FakeSparse,
            )

    def test_real_pymatching_decodes_boundary_logical_edge_when_installed(self) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "faultscope-matplotlib-cache"),
        )
        try:
            import pymatching  # noqa: F401
            import numpy  # noqa: F401
            from scipy import sparse  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional PyMatching dependencies are not installed: {exc}")

        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=0.1,
                    detectors=(0,),
                    observables=(0,),
                    location_id="x0",
                    event="X",
                ),
            ),
        )

        decoder = PyMatchingDecoder.from_dem(dem)
        self.assertEqual(decoder.decode_detector_record({0: 1}), {0: 1})
        self.assertEqual(
            decoder.decode_batch_detector_records([{0: 0}, {0: 1}]),
            [{0: 0}, {0: 1}],
        )
        self.assertEqual(decoder.decode_batch_masks({0: 0b1010}, shots=4), {0: 0b1010})

    def test_real_pymatching_decodes_bit_packed_masks_from_batch_objects(self) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "faultscope-matplotlib-cache"),
        )
        try:
            import pymatching  # noqa: F401
            import numpy  # noqa: F401
            from scipy import sparse  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"optional PyMatching dependencies are not installed: {exc}")

        dem = DetectorErrorModel(
            detectors=(
                Detector(id=0, measurement_keys=()),
                Detector(id=1, measurement_keys=()),
            ),
            observables=(LogicalObservable(id=0), LogicalObservable(id=1)),
            edges=(
                DetectorErrorEdge(
                    probability=0.1,
                    detectors=(0,),
                    observables=(0,),
                    location_id="x0",
                    event="X",
                ),
                DetectorErrorEdge(
                    probability=0.1,
                    detectors=(1,),
                    observables=(1,),
                    location_id="x1",
                    event="X",
                ),
            ),
        )
        decoder = PyMatchingDecoder.from_dem(dem)
        detector_masks = {0: 0b1010, 1: 0b1100}
        expected = {0: 0b1010, 1: 0b1100}

        self.assertEqual(
            decoder.decode_batch_masks(detector_masks, shots=4),
            expected,
        )
        self.assertEqual(
            decoder.decode_batch_masks(
                DemSampleBatch(
                    shots=4,
                    all_mask=0b1111,
                    detectors=detector_masks,
                    observables={},
                    edge_event_masks={},
                )
            ),
            expected,
        )
        self.assertEqual(
            decoder.decode_batch_masks(
                SampleBatch(
                    shots=4,
                    all_mask=0b1111,
                    x_frame=(),
                    z_frame=(),
                    measurements={},
                    detectors=detector_masks,
                    observables={},
                    noise_event_masks={},
                )
            ),
            expected,
        )


class HotspotVisualizationTests(unittest.TestCase):
    def test_generates_repetition_hotspot_heatmap_png(self) -> None:
        distance = 3
        rounds = 3
        hot_data = (1, 1)
        hot_measurement = (2, 0)
        data_rates = {
            (round_idx, data_idx): (0.18 if (round_idx, data_idx) == hot_data else 0.04)
            for round_idx in range(rounds)
            for data_idx in range(distance)
        }
        measurement_rates = {
            (round_idx, check_idx): (0.16 if (round_idx, check_idx) == hot_measurement else 0.03)
            for round_idx in range(rounds)
            for check_idx in range(distance - 1)
        }
        experiment = make_repetition_code_experiment(
            distance=distance,
            rounds=rounds,
            data_error_rate=data_rates,
            measurement_error_rate=measurement_rates,
        )
        result = FaultScopeSimulator(
            experiment.circuit,
            observables=experiment.observables,
        ).estimate(
            shots=5_000,
            seed=31,
            decoder=experiment.decoder,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "hotspot_heatmap.png")
            try:
                written = write_repetition_hotspot_heatmap(
                    result,
                    path,
                    distance=distance,
                    rounds=rounds,
                    highlighted_data=hot_data,
                    highlighted_measurement=hot_measurement,
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            self._assert_png_nonblank(path)
        self.assertGreater(max(result.hotspots.values()), 0.0)

    def test_generates_gate_structure_hotspot_map_png(self) -> None:
        distance = 3
        rounds = 2
        hot_cx = (1, 0, "right")
        experiment = make_repetition_code_experiment(
            distance=distance,
            rounds=rounds,
            data_error_rate=0.03,
            measurement_error_rate=0.03,
        )
        circuit = self._add_cx_noise_to_repetition_circuit(
            experiment.circuit,
            distance=distance,
            hot_cx=hot_cx,
        )
        result = FaultScopeSimulator(
            circuit,
            observables=experiment.observables,
        ).estimate(
            shots=5_000,
            seed=32,
            decoder=experiment.decoder,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "gate_structure_hotspots.png")
            try:
                written = write_repetition_gate_structure_hotspot_map(
                    result,
                    path,
                    distance=distance,
                    rounds=rounds,
                    highlighted_data=(0, 1),
                    highlighted_measurement=(1, 0),
                    highlighted_cx=hot_cx,
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            self._assert_png_nonblank(path)
        self.assertIn("cx_right_r1_c0", result.hotspots)

    def test_generates_d5_rotated_surface_code_spatial_hotspot_map_png(self) -> None:
        distance = 5
        result = self._make_synthetic_rotated_surface_code_result(distance)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "rotated_surface_code_d5_hotspots.png")
            try:
                written = write_rotated_surface_code_spatial_hotspot_map(
                    result,
                    path,
                    distance=distance,
                    highlighted_data=(2, 2),
                    highlighted_check_ids=("z_check_2_2",),
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            self._assert_png_nonblank(path)
        self.assertEqual(len(result.locations), 49)
        self.assertIn("data_2_2", result.hotspots)
        self.assertIn("z_check_2_2", result.hotspots)

    def test_d5_rotated_surface_code_integration_generates_spatial_hotspot_map(
        self,
    ) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "faultscope-matplotlib-cache"),
        )
        try:
            import numpy as np
            import pymatching
            from scipy import sparse
        except ImportError as exc:
            self.skipTest(f"optional PyMatching dependencies are not installed: {exc}")

        distance = 5
        rounds = 3
        z_checks = self._rotated_surface_code_z_checks(distance)
        circuit = self._make_rotated_surface_code_bitflip_circuit(
            distance=distance,
            rounds=rounds,
            z_checks=z_checks,
            hot_data=(2, 2),
            hot_check_id="z_check_2_2",
        )
        matching = self._make_surface_code_matching(
            distance=distance,
            z_checks=z_checks,
            np=np,
            pymatching=pymatching,
            sparse=sparse,
        )
        loss_mask_fn = self._make_surface_code_loss_mask_fn(
            distance=distance,
            rounds=rounds,
            z_checks=z_checks,
            matching=matching,
        )
        result = FaultScopeSimulator(circuit).estimate(
            shots=7_000,
            seed=33,
            loss_mask_fn=loss_mask_fn,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "d5_surface_code_integration_hotspots.png")
            try:
                written = write_rotated_surface_code_spatial_hotspot_map(
                    result,
                    path,
                    distance=distance,
                    highlighted_data=(2, 2),
                    highlighted_check_ids=("meas_r2_z_check_2_2",),
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            self._assert_png_nonblank(path)

        self.assertGreater(result.logical_failure_rate, 0.0)
        self.assertIn("data_r1_2_2", result.hotspots)
        self.assertIn("meas_r2_z_check_2_2", result.hotspots)
        self.assertGreater(max(result.hotspots.values()), 0.0)

    def _make_synthetic_rotated_surface_code_result(
        self,
        distance: int,
    ) -> FailureEstimate:
        locations: dict[str, NoiseLocation] = {}
        hotspots: dict[str, float] = {}
        sensitivities: dict[str, float] = {}
        qubit_index = 0

        for row in range(distance):
            for col in range(distance):
                location_id = f"data_{row}_{col}"
                hotspot = 0.02 + 0.01 * ((row + 2 * col) % 5)
                if (row, col) == (2, 2):
                    hotspot = 0.42
                locations[location_id] = NoiseLocation(
                    id=location_id,
                    model=BernoulliPauliNoise("X"),
                    rate=0.04,
                    qubits=(qubit_index,),
                    tags={
                        "layout": "rotated_surface_code",
                        "role": "data",
                        "row": row,
                        "col": col,
                        "operation": "data_noise",
                    },
                )
                hotspots[location_id] = hotspot
                sensitivities[location_id] = hotspot
                qubit_index += 1

        x_checks, z_checks = _rotated_surface_code_checks(distance)
        for basis, checks in (("x", x_checks), ("z", z_checks)):
            for check in checks:
                location_id = str(check["id"])
                x_coord = float(check["x"])
                y_coord = float(check["y"])
                hotspot = 0.015 + 0.012 * ((len(location_id) + int(10 * x_coord)) % 4)
                if location_id == "z_check_2_2":
                    hotspot = 0.36
                locations[location_id] = NoiseLocation(
                    id=location_id,
                    model=BernoulliPauliNoise("X"),
                    rate=0.03,
                    qubits=(),
                    tags={
                        "layout": "rotated_surface_code",
                        "role": f"{basis}_check",
                        "x": x_coord,
                        "y": y_coord,
                        "operation": "check_noise",
                    },
                )
                hotspots[location_id] = hotspot
                sensitivities[location_id] = hotspot

        return FailureEstimate(
            shots=12_000,
            mean_loss=0.071,
            baseline=0.071,
            sensitivities=sensitivities,
            hotspots=hotspots,
            by_qubit={},
            by_round={},
            by_gate={},
            by_operation={},
            locations=locations,
            losses=[],
        )

    def _rotated_surface_code_z_checks(
        self,
        distance: int,
    ) -> list[dict[str, object]]:
        _, z_checks = _rotated_surface_code_checks(distance)
        return [dict(check) for check in z_checks]

    def _make_rotated_surface_code_bitflip_circuit(
        self,
        *,
        distance: int,
        rounds: int,
        z_checks: list[dict[str, object]],
        hot_data: tuple[int, int],
        hot_check_id: str,
    ) -> Circuit:
        operations = []
        for round_idx in range(rounds):
            for row in range(distance):
                for col in range(distance):
                    location = NoiseLocation(
                        id=f"data_r{round_idx}_{row}_{col}",
                        model=BernoulliPauliNoise("X"),
                        rate=0.13 if (row, col) == hot_data else 0.035,
                        qubits=(_surface_data_index(distance, row, col),),
                        tags={
                            "layout": "rotated_surface_code",
                            "role": "data",
                            "row": row,
                            "col": col,
                            "round": round_idx,
                            "operation": "data_noise",
                        },
                    )
                    operations.append(Operation.noise(location))

            for check in z_checks:
                check_id = str(check["id"])
                data = tuple(check["data"])
                qubits = tuple(_surface_data_index(distance, row, col) for row, col in data)
                location = NoiseLocation(
                    id=f"meas_r{round_idx}_{check_id}",
                    model=MeasurementBitFlip(),
                    rate=0.11 if check_id == hot_check_id else 0.02,
                    qubits=qubits[:1],
                    tags={
                        "layout": "rotated_surface_code",
                        "role": "z_check",
                        "x": float(check["x"]),
                        "y": float(check["y"]),
                        "round": round_idx,
                        "check": check_id,
                        "operation": "measurement_noise",
                    },
                )
                operations.append(
                    Operation.measure_pauli(
                        qubits,
                        "Z" * len(qubits),
                        key=f"r{round_idx}_{check_id}",
                        noise=location,
                    )
                )
        return Circuit(n_qubits=distance * distance, operations=operations)

    def _make_surface_code_matching(
        self,
        *,
        distance: int,
        z_checks: list[dict[str, object]],
        np,
        pymatching,
        sparse,
    ):
        rows = []
        cols = []
        data = []
        for check_index, check in enumerate(z_checks):
            for row, col in check["data"]:
                rows.append(check_index)
                cols.append(_surface_data_index(distance, row, col))
                data.append(1)
        h = sparse.csc_matrix(
            (data, (rows, cols)),
            shape=(len(z_checks), distance * distance),
            dtype=np.uint8,
        )
        faults_matrix = sparse.eye(distance * distance, format="csc", dtype=np.uint8)
        return pymatching.Matching.from_check_matrix(
            h,
            faults_matrix=faults_matrix,
            weights=np.ones(distance * distance),
            merge_strategy="independent",
            use_virtual_boundary_node=True,
        )

    def _make_surface_code_loss_mask_fn(
        self,
        *,
        distance: int,
        rounds: int,
        z_checks: list[dict[str, object]],
        matching,
    ):
        logical_path = tuple(_surface_data_index(distance, row, 0) for row in range(distance))

        def loss_mask_fn(batch):
            final_round = rounds - 1
            syndromes = [
                [batch.measurement_bit(f"r{final_round}_{check['id']}", shot) for check in z_checks]
                for shot in range(batch.shots)
            ]
            predictions = matching.decode_batch(syndromes)
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            loss_mask = 0
            for shot, correction in enumerate(predictions):
                residual_logical = 0
                for qubit in logical_path:
                    residual_logical ^= batch.x_bit(qubit, shot) ^ int(correction[qubit])
                if residual_logical:
                    loss_mask |= 1 << shot
            return loss_mask

        return loss_mask_fn

    def _add_cx_noise_to_repetition_circuit(
        self,
        circuit: Circuit,
        *,
        distance: int,
        hot_cx: tuple[int, int, str],
    ) -> Circuit:
        operations = []
        cx_seen = 0
        local_cx_per_round = 2 * (distance - 1)
        for operation in circuit.operations:
            operations.append(operation)
            if operation.kind != "cx":
                continue
            round_idx = cx_seen // local_cx_per_round
            local_idx = cx_seen % local_cx_per_round
            check_idx = local_idx // 2
            side = "left" if local_idx % 2 == 0 else "right"
            rate = 0.14 if (round_idx, check_idx, side) == hot_cx else 0.025
            control, target = operation.qubits
            location = NoiseLocation(
                id=f"cx_{side}_r{round_idx}_c{check_idx}",
                model=BernoulliPauliNoise("XI"),
                rate=rate,
                qubits=(control, target),
                tags={
                    "round": round_idx,
                    "qubit": control,
                    "check": check_idx,
                    "gate": "cx",
                    "operation": "cx_noise",
                    "side": side,
                    "control": control,
                    "target": target,
                },
            )
            operations.append(Operation.noise(location))
            cx_seen += 1
        return Circuit(n_qubits=circuit.n_qubits, operations=operations)

    def _assert_png_nonblank(self, path: str) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            self.skipTest(f"Pillow is not installed: {exc}")
        self.assertGreater(os.path.getsize(path), 1_000)
        with Image.open(path) as image:
            self.assertGreaterEqual(image.size[0], 500)
            self.assertGreaterEqual(image.size[1], 300)
            colors = image.convert("RGB").resize((32, 32)).getcolors(maxcolors=1024)
        self.assertIsNotNone(colors)
        self.assertGreater(len(colors), 8)


def _surface_data_index(distance: int, row: int, col: int) -> int:
    return row * distance + col


class StimImportTests(unittest.TestCase):
    def test_imports_stim_subset_and_generates_dem(self) -> None:
        imported = parse_stim_circuit(
            """
            R 0 1 2
            X_ERROR(0.1) 0
            CX 0 1 2 1
            M(0.01) 1
            DETECTOR(0.5, 0) rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )

        self.assertEqual(imported.circuit.n_qubits, 3)
        self.assertEqual(imported.measurement_keys, ("m0",))
        self.assertEqual(len(imported.detectors), 1)
        self.assertEqual(imported.detectors[0].measurement_keys, ("m0",))
        self.assertEqual(imported.detectors[0].coords, (0.5, 0.0))
        self.assertEqual(len(imported.observables), 1)
        self.assertEqual(imported.observables[0].measurement_keys, ("m0",))

        locations = imported.circuit.noise_locations()
        self.assertEqual(len(locations), 2)
        dem = DetectorErrorModelGenerator(
            imported.circuit,
            detectors=imported.detectors,
            observables=imported.observables,
        ).generate()
        dem_from_circuit = DetectorErrorModelGenerator(imported.circuit).generate()
        dem_text = dem.to_dem_text()
        self.assertIn("detector(0.5, 0) D0", dem_text)
        self.assertIn("D0 L0", dem_text)
        self.assertEqual(dem.to_dem_text(), dem_from_circuit.to_dem_text())
        operation_kinds = [operation.kind for operation in imported.circuit.operations]
        self.assertIn("detector_rec", operation_kinds)
        self.assertIn(
            "observable_include_rec",
            operation_kinds,
        )
        measurement = next(
            operation for operation in imported.circuit.operations if operation.kind == "measure"
        )
        self.assertIsNone(measurement.key)

    def test_imports_pauli_channels_and_mpp(self) -> None:
        imported = parse_stim_circuit(
            """
            R 0 1
            PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0
            PAULI_CHANNEL_2(0.001,0,0,0,0,0,0,0,0,0,0,0,0,0,0.002) 0 1
            MPP X0*X1
            DETECTOR rec[-1]
            """
        )
        self.assertEqual(imported.circuit.n_qubits, 2)
        self.assertEqual(imported.measurement_keys, ("m0",))
        self.assertEqual(len(imported.circuit.noise_locations()), 2)
        self.assertEqual(imported.detectors[0].measurement_keys, ("m0",))

    def test_imports_relative_measurement_record_references(self) -> None:
        imported = parse_stim_circuit(
            """
            M 0
            M 1
            DETECTOR rec[-1] rec[-2]
            OBSERVABLE_INCLUDE(3) rec[-2]
            """
        )

        self.assertEqual(imported.measurement_keys, ("m0", "m1"))
        self.assertEqual(imported.detectors[0].measurement_keys, ("m1", "m0"))
        self.assertEqual(imported.observables[0].id, 3)
        self.assertEqual(imported.observables[0].measurement_keys, ("m0",))

    def test_imports_generated_rotated_surface_code_with_measure_resets(self) -> None:
        try:
            import stim
        except ImportError as exc:
            self.skipTest(f"Stim is not installed: {exc}")

        stim_circuit = stim.Circuit.generated(
            code_task="surface_code:rotated_memory_x",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.001,
        )
        imported = parse_stim_circuit(str(stim_circuit.flattened()))

        self.assertEqual(len(imported.measurement_keys), stim_circuit.num_measurements)
        self.assertEqual(len(imported.detectors), stim_circuit.num_detectors)
        self.assertEqual(len(imported.observables), stim_circuit.num_observables)
        self.assertIn("reset", [operation.kind for operation in imported.circuit.operations])

    def test_imports_repeat_blocks(self) -> None:
        imported = parse_stim_circuit(
            """
            REPEAT 3 {
                M 0
                DETECTOR rec[-1]
            }
            """
        )
        self.assertEqual(imported.measurement_keys, ("m0", "m1", "m2"))
        self.assertEqual(len(imported.detectors), 3)
        self.assertEqual(imported.circuit.operations[0].kind, "repeat")

    def test_nested_repeat_rec_and_shift_coords(self) -> None:
        imported = parse_stim_circuit(
            """
            M 0
            REPEAT 2 {
                REPEAT 2 {
                    M 0
                    SHIFT_COORDS(0, 0, 0.5)
                    DETECTOR(1, 2, 0) rec[-1] rec[-2]
                }
            }
            """
        )
        self.assertEqual(imported.measurement_keys, ("m0", "m1", "m2", "m3", "m4"))
        self.assertEqual(len(imported.detectors), 4)
        self.assertEqual(
            tuple(detector.coords for detector in imported.detectors),
            (
                (1.0, 2.0, 0.5),
                (1.0, 2.0, 1.0),
                (1.0, 2.0, 1.5),
                (1.0, 2.0, 2.0),
            ),
        )
        self.assertEqual(imported.detectors[-1].measurement_keys, ("m4", "m3"))

    def test_compact_and_flattened_surface_code_compile_identically(self) -> None:
        try:
            import stim
        except ImportError as exc:
            self.skipTest(f"Stim is not installed: {exc}")
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
        )
        compact = parse_stim_circuit(str(circuit))
        flattened = parse_stim_circuit(str(circuit.flattened()))
        compact_sampler = compile_native_sampler(compact.circuit)
        flattened_sampler = compile_native_sampler(flattened.circuit)
        self.assertLess(compact_sampler.stored_operation_count, compact_sampler.operation_count)
        self.assertLess(flattened_sampler.stored_operation_count, flattened_sampler.operation_count)
        self.assertEqual(compact_sampler.operation_count, flattened_sampler.operation_count)
        self.assertEqual(
            compact_sampler.sample_measurements(shots=257, seed=9123),
            flattened_sampler.sample_measurements(shots=257, seed=9123),
        )

    def test_long_flattened_surface_code_keeps_warmup_outside_periodic_loop(self) -> None:
        try:
            import stim
        except ImportError as exc:
            self.skipTest(f"Stim is not installed: {exc}")
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=10,
            rounds=10,
        )
        compact = parse_stim_circuit(str(circuit))
        flattened = parse_stim_circuit(str(circuit.flattened()))
        recovered_repeats = [
            operation for operation in flattened.circuit.operations if operation.kind == "repeat"
        ]

        self.assertEqual(len(recovered_repeats), 1)
        self.assertEqual(recovered_repeats[0].repeat_count, 9)
        compact_sampler = compile_native_sampler(compact.circuit)
        flattened_sampler = compile_native_sampler(flattened.circuit)
        self.assertEqual(compact_sampler.loop_kernel_count, 1)
        self.assertEqual(flattened_sampler.loop_kernel_count, 1)
        self.assertEqual(
            compact_sampler.sample_measurements(shots=65, seed=441),
            flattened_sampler.sample_measurements(shots=65, seed=441),
        )

    def test_repeat_noise_ids_and_dem_match_flattened_reference(self) -> None:
        compact = parse_stim_circuit(
            """
            R 0
            REPEAT 3 {
                X_ERROR(0.125) 0
                M 0
                DETECTOR rec[-1]
                R 0
            }
            """
        )
        flattened = parse_stim_circuit(
            """
            R 0
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
            R 0
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
            R 0
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
            R 0
            """
        )
        self.assertEqual(len(compact.circuit.noise_locations()), 3)
        compact_dem = DetectorErrorModelGenerator(compact.circuit).generate()
        flattened_dem = DetectorErrorModelGenerator(flattened.circuit).generate()

        def canonical(dem):
            return sorted(
                (
                    edge.probability,
                    tuple(edge.detectors),
                    tuple(edge.observables),
                    edge.event,
                )
                for edge in dem.edges
            )

        self.assertEqual(canonical(compact_dem), canonical(flattened_dem))

    def test_repeat_parser_validation(self) -> None:
        for source in (
            "REPEAT 0 {\nM 0\n}",
            "REPEAT 2 {\nM 0",
            "}",
            "REPEAT 2 {\nDETECTOR rec[-1]\n}",
        ):
            with self.subTest(source=source):
                with self.assertRaises(StimImportError):
                    parse_stim_circuit(source)


if __name__ == "__main__":
    unittest.main()
