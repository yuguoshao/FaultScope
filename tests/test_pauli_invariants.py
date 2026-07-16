import unittest

import faultscope.core as core
import faultscope.core.pauli as pauli_module
from faultscope.core import BernoulliPauliNoise, PauliFrame, StabilizerState


class _CountingRng:
    def __init__(self, value: int = 0) -> None:
        self.calls = 0
        self.value = value

    def randrange(self, stop: int) -> int:
        self.calls += 1
        return self.value % stop


class _FailingRng:
    def __init__(self) -> None:
        self.calls = 0

    def randrange(self, stop: int) -> int:
        self.calls += 1
        raise RuntimeError("rng failed")


class PublicApiInvariantTests(unittest.TestCase):
    def test_low_level_pauli_helpers_are_not_public_or_defined(self) -> None:
        for name in (
            "multiply_pauli_rows",
            "sparse_pauli_to_xz",
            "symplectic_product",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, core.__all__)
                self.assertFalse(hasattr(core, name))
                self.assertFalse(hasattr(pauli_module, name))

    def test_stabilizer_state_has_no_raw_tableau_constructor(self) -> None:
        with self.assertRaises(TypeError):
            StabilizerState()
        with self.assertRaises(TypeError):
            StabilizerState([], [], [])
        self.assertEqual(StabilizerState.zero(0).n_qubits, 0)


class NoisePauliInvariantTests(unittest.TestCase):
    def test_apply_rejects_state_frame_width_mismatch_before_mutation(self) -> None:
        state = StabilizerState.zero(2)
        frame = PauliFrame.zero(1)
        state_before = (state.x, state.z, state.sign)
        frame_before = (frame.x, frame.z)

        with self.assertRaisesRegex(ValueError, "must have the same width"):
            BernoulliPauliNoise("X").apply("X", state, frame, (1,))

        self.assertEqual((state.x, state.z, state.sign), state_before)
        self.assertEqual((frame.x, frame.z), frame_before)

    def test_apply_validates_sparse_event_once_before_committing_both_objects(self) -> None:
        state = StabilizerState.zero(2)
        frame = PauliFrame.zero(2)
        invalid_calls = (
            lambda: BernoulliPauliNoise("XZ").apply("XZ", state, frame, (0, 0)),
            lambda: BernoulliPauliNoise("X").apply("X", state, frame, (2,)),
            lambda: BernoulliPauliNoise("A").apply("A", state, frame, (0,)),
        )

        for call in invalid_calls:
            state_before = (state.x, state.z, state.sign)
            frame_before = (frame.x, frame.z)
            with self.assertRaises(ValueError):
                call()
            self.assertEqual((state.x, state.z, state.sign), state_before)
            self.assertEqual((frame.x, frame.z), frame_before)

        BernoulliPauliNoise("XZ").apply("XZ", state, frame, (0, 1))
        self.assertEqual(frame.pauli_on((0, 1)), "XZ")
        self.assertEqual(state.sign, [1, 0])


class PauliFrameInvariantTests(unittest.TestCase):
    def test_constructor_rejects_mismatched_and_non_binary_support(self) -> None:
        for x, z in (([], [0]), ([0], []), ([0], [0, 0])):
            with self.subTest(x=x, z=z):
                with self.assertRaises(ValueError):
                    PauliFrame(x, z)

        for value in (-1, 2, 256, 0.5, "1"):
            for x, z in (([value], [0]), ([0], [value])):
                with self.subTest(value=value, x=x, z=z):
                    with self.assertRaises(ValueError):
                        PauliFrame(x, z)

        self.assertEqual(PauliFrame([], []).n_qubits, 0)
        with self.assertRaises(ValueError):
            PauliFrame.zero(-1)

    def test_sparse_and_gate_boundaries_reject_before_mutation(self) -> None:
        frame = PauliFrame([1, 0], [0, 1])
        invalid_calls = (
            ("apply_h_negative", lambda: frame.apply_h(-1)),
            ("apply_s_range", lambda: frame.apply_s(2)),
            ("apply_cx_duplicate", lambda: frame.apply_cx(0, 0)),
            ("apply_cz_negative", lambda: frame.apply_cz(-1, 1)),
            ("apply_swap_range", lambda: frame.apply_swap(0, 2)),
            ("apply_pauli_negative", lambda: frame.apply_pauli(-1, "X")),
            ("apply_pauli_range", lambda: frame.apply_pauli(2, "X")),
            ("apply_pauli_bad", lambda: frame.apply_pauli(0, "A")),
            ("sparse_negative", lambda: frame.apply_pauli_string([-1], "X")),
            ("sparse_range", lambda: frame.apply_pauli_string([2], "X")),
            ("sparse_duplicate", lambda: frame.apply_pauli_string([0, 0], "XZ")),
            ("sparse_short_qubits", lambda: frame.apply_pauli_string([0], "XZ")),
            ("sparse_short_pauli", lambda: frame.apply_pauli_string([0, 1], "X")),
            ("flip_duplicate", lambda: frame.measurement_flip([0, 0], "ZZ")),
            ("pauli_on_negative", lambda: frame.pauli_on([-1])),
            ("pauli_on_range", lambda: frame.pauli_on([2])),
            ("pauli_on_duplicate", lambda: frame.pauli_on([0, 0])),
            ("reset_range", lambda: frame.reset(2)),
        )

        for name, call in invalid_calls:
            with self.subTest(name=name):
                before = (frame.x, frame.z)
                with self.assertRaises(ValueError):
                    call()
                self.assertEqual((frame.x, frame.z), before)

    def test_identity_and_zero_qubit_inputs_remain_valid(self) -> None:
        frame = PauliFrame.zero(2)
        frame.apply_pauli_string([0, 1], "II")
        self.assertEqual((frame.x, frame.z), ([0, 0], [0, 0]))
        self.assertEqual(frame.measurement_flip([0, 1], "II"), 0)

        empty = PauliFrame.zero(0)
        empty.apply_pauli_string([], "")
        self.assertEqual(empty.measurement_flip([], ""), 0)
        self.assertEqual(empty.pauli_on([]), "")


class StabilizerStateInvariantTests(unittest.TestCase):
    def test_dense_methods_reject_all_width_and_value_errors_without_mutation(self) -> None:
        invalid_supports = (
            ([], []),
            ([0], [0]),
            ([0], [0, 0]),
            ([0, 0], [0]),
            ([0, 0, 0], [0, 0]),
            ([0, 0], [0, 0, 0]),
            ([-1, 0], [0, 0]),
            ([2, 0], [0, 0]),
            ([256, 0], [0, 0]),
            ([0, 0], [-1, 0]),
            ([0, 0], [2, 0]),
            ([0, 0], [256, 0]),
            ([0.5, 0], [0, 0]),
            ([0, 0], ["1", 0]),
        )
        for x, z in invalid_supports:
            state = StabilizerState.zero(2)
            rng = _CountingRng()
            before = (state.x, state.z, state.sign)
            invalid_calls = (
                ("apply_pauli_string", lambda: state.apply_pauli_string(x, z)),
                ("measure_pauli", lambda: state.measure_pauli(x, z, rng)),
                ("is_deterministic_pauli", lambda: state.is_deterministic_pauli(x, z)),
                (
                    "deterministic_measurement_bit",
                    lambda: state.deterministic_measurement_bit(x, z),
                ),
            )
            for name, call in invalid_calls:
                with self.subTest(x=x, z=z, method=name):
                    with self.assertRaises(ValueError):
                        call()
                    self.assertEqual((state.x, state.z, state.sign), before)
                    self.assertEqual(rng.calls, 0)
                    self.assertTrue(all(len(row) == 2 for row in state.x))
                    self.assertTrue(all(len(row) == 2 for row in state.z))

    def test_target_errors_do_not_mutate_or_consume_rng(self) -> None:
        state = StabilizerState.zero(2)
        rng = _CountingRng()
        invalid_calls = (
            ("apply_h_negative", lambda: state.apply_h(-1)),
            ("apply_s_range", lambda: state.apply_s(2)),
            ("apply_cx_duplicate", lambda: state.apply_cx(0, 0)),
            ("apply_cz_negative", lambda: state.apply_cz(-1, 1)),
            ("apply_swap_range", lambda: state.apply_swap(0, 2)),
            ("apply_pauli_negative", lambda: state.apply_pauli(-1, "X")),
            ("apply_pauli_range", lambda: state.apply_pauli(2, "X")),
            ("apply_pauli_bad", lambda: state.apply_pauli(0, "A")),
            ("measure_z_negative", lambda: state.measure_z(-1, rng)),
            ("measure_x_range", lambda: state.measure_x(2, rng)),
            ("reset_y_negative", lambda: state.reset_y(-1, rng)),
            ("reset_z_range", lambda: state.reset_z(2, rng)),
        )
        for name, call in invalid_calls:
            with self.subTest(name=name):
                before = (state.x, state.z, state.sign)
                with self.assertRaises(ValueError):
                    call()
                self.assertEqual((state.x, state.z, state.sign), before)
                self.assertEqual(rng.calls, 0)

    def test_measurement_rng_is_lazy_and_callback_failure_is_atomic(self) -> None:
        state = StabilizerState.zero(1)
        deterministic_rng = _CountingRng(1)
        before = (state.x, state.z, state.sign)
        self.assertEqual(state.measure_pauli([0], [1], deterministic_rng), 0)
        self.assertEqual(deterministic_rng.calls, 0)
        self.assertEqual((state.x, state.z, state.sign), before)

        with self.assertRaisesRegex(ValueError, "measurement is random"):
            state.deterministic_measurement_bit([1], [0])
        self.assertEqual((state.x, state.z, state.sign), before)

        failing_rng = _FailingRng()
        before = (state.x, state.z, state.sign)
        with self.assertRaisesRegex(RuntimeError, "rng failed"):
            state.measure_pauli([1], [0], failing_rng)
        self.assertEqual(failing_rng.calls, 1)
        self.assertEqual((state.x, state.z, state.sign), before)

        random_rng = _CountingRng(1)
        self.assertEqual(state.measure_pauli([1], [0], random_rng), 1)
        self.assertEqual(random_rng.calls, 1)

    def test_identity_and_zero_qubit_inputs_remain_valid(self) -> None:
        state = StabilizerState.zero(2)
        state.apply_pauli_string([0, 0], [0, 0])
        self.assertTrue(state.is_deterministic_pauli([0, 0], [0, 0]))
        self.assertEqual(state.deterministic_measurement_bit([0, 0], [0, 0]), 0)

        empty = StabilizerState.zero(0)
        rng = _CountingRng(1)
        self.assertEqual(empty.measure_pauli([], [], rng), 0)
        self.assertEqual(rng.calls, 0)
        with self.assertRaises(ValueError):
            StabilizerState.zero(-1)


if __name__ == "__main__":
    unittest.main()
