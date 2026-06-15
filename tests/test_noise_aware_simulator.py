import random
import unittest

from npsim.batch import BatchForwardNoiseAwareSimulator, UnsupportedBatchCircuitError
from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.noise import BernoulliPauliNoise, PauliChannel
from npsim.repetition import make_repetition_code_experiment
from npsim.simulator import ForwardNoiseAwareSimulator
from npsim.stabilizer import StabilizerState


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

    def test_batch_rejects_random_ideal_measurements(self) -> None:
        circuit = Circuit(
            n_qubits=1,
            operations=[
                Operation.h(0),
                Operation.measure(0, key="m", basis="Z"),
            ],
        )
        with self.assertRaises(UnsupportedBatchCircuitError):
            BatchForwardNoiseAwareSimulator(circuit).estimate(
                shots=100,
                seed=15,
                loss_mask_fn=lambda batch: batch.measurements["m"],
            )


if __name__ == "__main__":
    unittest.main()
