import os
import random
import tempfile
import unittest

from npsim.batch import BatchForwardNoiseAwareSimulator
from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.dem import (
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    DetectorErrorModelGenerator,
    LogicalObservable,
    UnsupportedDemCircuitError,
)
from npsim.dem_sampler import DemBatchHotspotSimulator
from npsim.noise import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
)
from npsim.native import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_sampler,
    generate_native_dem,
)
from npsim.pymatching_decoder import (
    PyMatchingBatchDecoder,
    UnsupportedPyMatchingDemError,
)
from npsim.repetition import make_repetition_code_experiment
from npsim.simulator import ForwardNoiseAwareSimulator, SimulationResult
from npsim.stabilizer import StabilizerState
from npsim.stim_import import StimImportError, parse_stim_circuit
from npsim.visualization import (
    VisualizationUnavailableError,
    write_rotated_surface_code_spatial_hotspot_map,
    write_repetition_gate_structure_hotspot_map,
    write_repetition_hotspot_heatmap,
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


class StabilizerStateTests(unittest.TestCase):
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
    def test_ideal_pauli_gate_does_not_create_frame_error(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.x(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        trajectory = ForwardNoiseAwareSimulator(circuit).run_shot(rng=random.Random(8))
        self.assertEqual(trajectory.measurement_by_key["m"].bit, 1)
        self.assertEqual(trajectory.frame.pauli_on((0,)), "I")

    def test_basis_resets_prepare_requested_eigenstates(self) -> None:
        circuit = Circuit(
            n_qubits=2,
            operations=[
                Operation.x(0),
                Operation.reset(0, basis="X"),
                Operation.measure(0, key="mx", basis="X"),
                Operation.reset(1, basis="Y"),
                Operation.measure(1, key="my", basis="Y"),
            ],
        )
        trajectory = ForwardNoiseAwareSimulator(circuit).run_shot(rng=random.Random(9))
        self.assertEqual(trajectory.measurement_by_key["mx"].bit, 0)
        self.assertEqual(trajectory.measurement_by_key["my"].bit, 0)

    def test_common_clifford_gates_run_through_circuit_api(self) -> None:
        circuit = Circuit(
            n_qubits=2,
            operations=[
                Operation.h(0),
                Operation.h(1),
                Operation.cz(0, 1),
                Operation.measure_pauli((0, 1), "XZ", key="k0"),
                Operation.measure_pauli((0, 1), "ZX", key="k1"),
                Operation.swap(0, 1),
                Operation.s(0),
                Operation.s_dag(0),
            ],
        )
        trajectory = ForwardNoiseAwareSimulator(circuit).run_shot(rng=random.Random(10))
        self.assertEqual(trajectory.measurement_by_key["k0"].bit, 0)
        self.assertEqual(trajectory.measurement_by_key["k1"].bit, 0)

    def test_pauli_channel_estimates_total_error_rate_gradient(self) -> None:
        location = NoiseLocation(
            id="pc",
            model=PauliChannel({"X": 0.7, "Y": 0.3}),
            rate=0.25,
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
        result = ForwardNoiseAwareSimulator(circuit).estimate(
            shots=30_000,
            seed=12,
            loss_fn=lambda trajectory, decoded: trajectory.measurement_by_key["m"].bit,
        )
        self.assertAlmostEqual(result.mean_loss, 0.25, delta=0.025)
        self.assertAlmostEqual(result.sensitivities["pc"], 1.0, delta=0.1)

    def test_score_function_estimates_single_x_noise_gradient(self) -> None:
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
        result = ForwardNoiseAwareSimulator(circuit).estimate(
            shots=30_000,
            seed=5,
            loss_fn=lambda trajectory, decoded: trajectory.measurement_by_key["m"].bit,
        )

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.sensitivities["x0"], 1.0, delta=0.08)
        self.assertAlmostEqual(result.hotspots["x0"], 1.0, delta=0.08)
        self.assertIn(0, result.by_qubit)
        self.assertIn(0, result.by_round)
        self.assertIn("idle", result.by_gate)

    def test_zero_loss_has_zero_hotspot(self) -> None:
        location = NoiseLocation(
            id="irrelevant",
            model=BernoulliPauliNoise("Z"),
            rate=0.4,
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
        result = ForwardNoiseAwareSimulator(circuit).estimate(
            shots=2_000,
            seed=6,
            loss_fn=lambda trajectory, decoded: 0.0,
        )
        self.assertEqual(result.mean_loss, 0.0)
        self.assertEqual(result.hotspots["irrelevant"], 0.0)

    def test_measurement_bit_flip_estimates_gradient(self) -> None:
        location = NoiseLocation(
            id="mflip",
            model=MeasurementBitFlip(),
            rate=0.3,
            qubits=(0,),
            tags={"qubit": 0, "round": 0, "gate": "measure"},
        )
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.measure(0, key="m", basis="Z", noise=location),
            ],
        )
        result = ForwardNoiseAwareSimulator(circuit).estimate(
            shots=30_000,
            seed=18,
            loss_fn=lambda trajectory, decoded: trajectory.measurement_by_key["m"].bit,
        )

        self.assertAlmostEqual(result.mean_loss, 0.3, delta=0.025)
        self.assertAlmostEqual(result.sensitivities["mflip"], 1.0, delta=0.1)
        self.assertIn("measure", result.by_gate)

    def test_detector_and_observable_operations_are_recorded(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.x(0),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=2, coords=(1.5, 2.0)),
                Operation.observable_include(0, ("m",)),
            ],
        )
        trajectory = ForwardNoiseAwareSimulator(circuit).run_shot(rng=random.Random(16))
        self.assertEqual(trajectory.detectors, {2: 1})
        self.assertEqual(trajectory.observables, {0: 1})
        self.assertEqual(trajectory.detector_record, {2: 1})

    def test_repetition_code_experiment_runs_and_aggregates(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=3,
            rounds=1,
            data_error_rate={(0, 0): 0.15, (0, 1): 0.15, (0, 2): 0.01},
            measurement_error_rate=0.02,
        )
        result = ForwardNoiseAwareSimulator(experiment.circuit).estimate(
            shots=10_000,
            seed=7,
            detector_fn=experiment.detector_fn,
            decoder=experiment.decoder,
            loss_fn=experiment.loss_fn,
        )

        self.assertGreaterEqual(result.mean_loss, 0.0)
        self.assertLessEqual(result.mean_loss, 1.0)
        self.assertTrue(result.top_hotspots(top_k=3))
        self.assertIn(0, result.by_round)
        self.assertIn("idle", result.by_gate)
        self.assertIn("measure", result.by_gate)


class BatchNoiseAwareSimulatorTests(unittest.TestCase):
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
        result = BatchForwardNoiseAwareSimulator(circuit).estimate(
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
        result = BatchForwardNoiseAwareSimulator(experiment.circuit).estimate(
            shots=10_000,
            seed=14,
            loss_mask_fn=experiment.batch_loss_mask_fn,
        )

        self.assertGreaterEqual(result.mean_loss, 0.0)
        self.assertLessEqual(result.mean_loss, 1.0)
        self.assertTrue(result.top_hotspots(top_k=3))
        self.assertIn(0, result.by_round)
        self.assertIn("idle", result.by_gate)
        self.assertIn("measure", result.by_gate)

    def test_batch_samples_random_ideal_measurements(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        batch = BatchForwardNoiseAwareSimulator(circuit).run_batch(
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
        batch = BatchForwardNoiseAwareSimulator(circuit).run_batch(
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
        batch = BatchForwardNoiseAwareSimulator(circuit).run_batch(
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
        batch = BatchForwardNoiseAwareSimulator(circuit).run_batch(
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
        batch = BatchForwardNoiseAwareSimulator(circuit).run_batch(
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
        batch = BatchForwardNoiseAwareSimulator(circuit).run_batch(
            shots=7,
            rng=random.Random(19),
        )

        self.assertEqual(batch.measurements["m"], batch.all_mask)
        self.assertEqual(batch.noise_event_masks["mflip"], batch.all_mask)


class NativePackedSamplerTests(unittest.TestCase):
    def _native_sampler_or_skip(self, circuit: Circuit):
        try:
            return compile_native_sampler(circuit, backend="native")
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native extension unavailable: {exc}")

    def test_python_backend_matches_batch_sampler_masks(self) -> None:
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
        sampler = compile_native_sampler(circuit, backend="python")
        native_batch = sampler.sample(shots=9, seed=123)
        reference = BatchForwardNoiseAwareSimulator(circuit).run_batch(
            shots=9,
            rng=random.Random(123),
        )

        self.assertEqual(sampler.backend_name, "python")
        self.assertEqual(native_batch.measurements, reference.measurements)
        self.assertEqual(native_batch.detectors, reference.detectors)
        self.assertEqual(native_batch.observables, reference.observables)
        self.assertEqual(native_batch.noise_event_masks, reference.noise_event_masks)

    def test_auto_backend_falls_back_without_native_extension(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[Operation.measure(0, key="m", basis="Z")],
        )
        sampler = compile_native_sampler(circuit, backend="auto")
        batch = sampler.sample(shots=4, seed=1)

        self.assertIn(sampler.backend_name, {"native", "python"})
        self.assertEqual(batch.shots, 4)
        self.assertIn("m", batch.measurements)

    def test_native_backend_reports_missing_or_unsupported_extension(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[Operation.measure(0, key="m", basis="Z")],
        )
        try:
            __import__("npsim._npsim_native")
        except ImportError:
            with self.assertRaises(UnsupportedNativeCircuitError):
                compile_native_sampler(circuit, backend="native")
        else:
            sampler = compile_native_sampler(circuit, backend="native")
            self.assertTrue(sampler.is_native)

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            compile_native_sampler(Circuit(n_qubits=0, operations=[]), backend="gpu")

    def test_native_backend_matches_deterministic_reference_masks(self) -> None:
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
                Operation.x(0),
                Operation.reset(0, key="r", basis="Z"),
                Operation.noise(x_location),
                Operation.measure(0, key="m", basis="Z", noise=m_location),
                Operation.detector(("r", "m"), detector_id=3),
                Operation.observable_include(0, ("m",)),
            ],
        )
        sampler = self._native_sampler_or_skip(circuit)
        native_batch = sampler.sample(shots=17, seed=11)
        reference = BatchForwardNoiseAwareSimulator(circuit).run_batch(
            shots=17,
            rng=random.Random(11),
        )

        self.assertEqual(native_batch.x_frame, reference.x_frame)
        self.assertEqual(native_batch.z_frame, reference.z_frame)
        self.assertEqual(native_batch.measurements, reference.measurements)
        self.assertEqual(native_batch.detectors, reference.detectors)
        self.assertEqual(native_batch.observables, reference.observables)
        self.assertEqual(native_batch.noise_event_masks, reference.noise_event_masks)

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
        sampler = compile_native_sampler(circuit, backend="auto")
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


class NativeDetectorErrorModelTests(unittest.TestCase):
    def _require_native_dem(self) -> None:
        try:
            __import__("npsim._npsim_native")
        except ImportError as exc:
            self.skipTest(f"native extension unavailable: {exc}")

    def test_native_dem_generator_matches_repetition_reference(self) -> None:
        self._require_native_dem()
        experiment = make_repetition_code_experiment(
            distance=3,
            rounds=1,
            data_error_rate=0.1,
            measurement_error_rate=0.01,
        )
        generator = DetectorErrorModelGenerator(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=experiment.observables,
        )
        reference = generator._generate_python()
        native = generate_native_dem(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=experiment.observables,
            backend="native",
        )

        self.assertEqual(native.to_dem_text(), reference.to_dem_text())
        self.assertEqual(
            [
                (
                    edge.location_id,
                    edge.event,
                    edge.probability,
                    edge.detectors,
                    edge.observables,
                    dict(edge.tags),
                )
                for edge in native.edges
            ],
            [
                (
                    edge.location_id,
                    edge.event,
                    edge.probability,
                    edge.detectors,
                    edge.observables,
                    dict(edge.tags),
                )
                for edge in reference.edges
            ],
        )

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

        native = generate_native_dem(circuit, backend="native")
        by_event = {edge.event: edge for edge in native.edges}

        self.assertEqual(set(by_event), {"X", "Y"})
        self.assertAlmostEqual(by_event["X"].probability, 0.1)
        self.assertAlmostEqual(by_event["Y"].probability, 0.3)
        self.assertEqual(by_event["Y"].detectors, (0,))
        self.assertEqual(dict(by_event["Y"].tags), {"gate": "channel"})

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
        sampler = compile_native_dem_sampler(dem, backend="native")
        batch = sampler.run_batch(shots=9, seed=123)

        all_mask = (1 << 9) - 1
        self.assertEqual(batch.all_mask, all_mask)
        self.assertEqual(batch.edge_event_masks[0], all_mask)
        self.assertEqual(batch.edge_event_masks[1], 0)
        self.assertEqual(batch.detectors[0], all_mask)
        self.assertEqual(batch.observables[0], all_mask)

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
        sampler = compile_native_dem_sampler(dem, backend="native")
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
        sampler = compile_native_dem_sampler(dem, backend="native")
        result = sampler.estimate_default(shots=40_000, seed=54)

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.edge_sensitivities[0], 1.0, delta=0.08)
        self.assertAlmostEqual(result.sensitivities["logical_edge"], 1.0, delta=0.08)
        self.assertEqual(result.by_round[1], result.hotspots["logical_edge"])

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
                backend="native",
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
        with self.assertRaises(UnsupportedDemCircuitError):
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
        dem = DetectorErrorModelGenerator(circuit).generate()

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


class DemBatchHotspotSimulatorTests(unittest.TestCase):
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

        result = DemBatchHotspotSimulator(dem).estimate(shots=40_000, seed=51)

        self.assertAlmostEqual(result.mean_loss, 0.2, delta=0.02)
        self.assertAlmostEqual(result.edge_sensitivities[0], 1.0, delta=0.08)
        self.assertAlmostEqual(result.sensitivities["logical_edge"], 1.0, delta=0.08)
        self.assertAlmostEqual(result.hotspots["logical_edge"], 1.0, delta=0.08)
        self.assertEqual(result.by_round[1], result.hotspots["logical_edge"])
        self.assertEqual(result.by_operation["dem_error"], result.hotspots["logical_edge"])
        self.assertEqual(result.top_edges(1)[0].location_id, "logical_edge")

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

        result = DemBatchHotspotSimulator(dem).estimate(
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

        result = DemBatchHotspotSimulator(dem).estimate(shots=40_000, seed=53)

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
        return [[int(row[0]) if row else 0] for row in syndromes]


class _FakePyMatching:
    calls = []

    class Matching:
        @staticmethod
        def from_check_matrix(h, **kwargs):
            _FakePyMatching.calls.append((h, kwargs))
            return _FakeMatching()


class PyMatchingBatchDecoderTests(unittest.TestCase):
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
        decoder = PyMatchingBatchDecoder.from_dem(
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
        decoder = PyMatchingBatchDecoder.from_dem(
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
        decoder = PyMatchingBatchDecoder.from_dem(
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
            PyMatchingBatchDecoder.from_dem(
                dem,
                pymatching_module=_FakePyMatching,
                numpy_module=_FakeNumpy,
                scipy_sparse_module=_FakeSparse,
            )

    def test_real_pymatching_decodes_boundary_logical_edge_when_installed(self) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "npsim-matplotlib-cache"),
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

        decoder = PyMatchingBatchDecoder.from_dem(dem)
        self.assertEqual(decoder.decode_detector_record({0: 1}), {0: 1})
        self.assertEqual(
            decoder.decode_batch_detector_records([{0: 0}, {0: 1}]),
            [{0: 0}, {0: 1}],
        )
        self.assertEqual(decoder.decode_batch_masks({0: 0b1010}, shots=4), {0: 0b1010})


class HotspotVisualizationTests(unittest.TestCase):
    def test_generates_repetition_hotspot_heatmap_png(self) -> None:
        distance = 3
        rounds = 3
        hot_data = (1, 1)
        hot_measurement = (2, 0)
        data_rates = {
            (round_idx, data_idx): (
                0.18 if (round_idx, data_idx) == hot_data else 0.04
            )
            for round_idx in range(rounds)
            for data_idx in range(distance)
        }
        measurement_rates = {
            (round_idx, check_idx): (
                0.16 if (round_idx, check_idx) == hot_measurement else 0.03
            )
            for round_idx in range(rounds)
            for check_idx in range(distance - 1)
        }
        experiment = make_repetition_code_experiment(
            distance=distance,
            rounds=rounds,
            data_error_rate=data_rates,
            measurement_error_rate=measurement_rates,
        )
        result = BatchForwardNoiseAwareSimulator(experiment.circuit).estimate(
            shots=5_000,
            seed=31,
            loss_mask_fn=experiment.batch_loss_mask_fn,
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
        result = BatchForwardNoiseAwareSimulator(circuit).estimate(
            shots=5_000,
            seed=32,
            loss_mask_fn=experiment.batch_loss_mask_fn,
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
                    highlighted_check_ids=("x_check_2_2",),
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            self._assert_png_nonblank(path)
        self.assertEqual(len(result.locations), 49)
        self.assertIn("data_2_2", result.hotspots)
        self.assertIn("x_check_2_2", result.hotspots)

    def test_d5_rotated_surface_code_integration_generates_spatial_hotspot_map(
        self,
    ) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "npsim-matplotlib-cache"),
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
        result = BatchForwardNoiseAwareSimulator(circuit).estimate(
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
    ) -> SimulationResult:
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

        check_specs: list[tuple[str, str, float, float]] = []
        for row in range(distance - 1):
            for col in range(distance - 1):
                basis = "x" if (row + col) % 2 == 0 else "z"
                check_specs.append((basis, f"{row}_{col}", col + 0.5, row + 0.5))
        check_specs.extend(
            (
                ("x", "top_0", 0.5, -0.35),
                ("z", "top_1", 2.5, -0.35),
                ("x", "bottom_0", 1.5, distance - 0.65),
                ("z", "bottom_1", 3.5, distance - 0.65),
                ("z", "left_0", -0.35, 0.5),
                ("x", "left_1", -0.35, 2.5),
                ("z", "right_0", distance - 0.65, 1.5),
                ("x", "right_1", distance - 0.65, 3.5),
            )
        )
        for basis, suffix, x_coord, y_coord in check_specs:
            location_id = f"{basis}_check_{suffix}"
            hotspot = 0.015 + 0.012 * ((len(suffix) + int(10 * x_coord)) % 4)
            if location_id == "x_check_2_2":
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

        return SimulationResult(
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
        checks: list[dict[str, object]] = []
        for row in range(distance - 1):
            for col in range(distance - 1):
                if (row + col) % 2:
                    continue
                checks.append(
                    {
                        "id": f"z_check_{row}_{col}",
                        "data": (
                            (row, col),
                            (row + 1, col),
                            (row, col + 1),
                            (row + 1, col + 1),
                        ),
                        "x": col + 0.5,
                        "y": row + 0.5,
                    }
                )
        checks.extend(
            (
                {
                    "id": "z_check_top_right",
                    "data": ((0, distance - 2), (0, distance - 1)),
                    "x": distance - 1.5,
                    "y": -0.35,
                },
                {
                    "id": "z_check_bottom_left",
                    "data": ((distance - 1, 0), (distance - 1, 1)),
                    "x": 0.5,
                    "y": distance - 0.65,
                },
            )
        )
        return checks

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
                qubits = tuple(
                    _surface_data_index(distance, row, col)
                    for row, col in data
                )
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
        logical_path = tuple(
            _surface_data_index(distance, row, 0)
            for row in range(distance)
        )

        def loss_mask_fn(batch):
            final_round = rounds - 1
            syndromes = [
                [
                    batch.measurement_bit(f"r{final_round}_{check['id']}", shot)
                    for check in z_checks
                ]
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
        self.assertIn("detector", [operation.kind for operation in imported.circuit.operations])
        self.assertIn(
            "observable_include",
            [operation.kind for operation in imported.circuit.operations],
        )

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

    def test_rejects_repeat_blocks(self) -> None:
        with self.assertRaises(StimImportError):
            parse_stim_circuit(
                """
                REPEAT 3 {
                    M 0
                }
                """
            )


if __name__ == "__main__":
    unittest.main()
