import random
import unittest

from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.noise import BernoulliPauliNoise
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


class NoiseAwareSimulatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
