"""Cirq CliffordSimulator worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fsbench.ir import load_case
from fsbench.protocol import EngineWorker, IterationResult, batch_ranges, derive_seed, worker_main


class CirqWorker(EngineWorker):
    engine_key = "cirq"
    engine_label = "Cirq"

    def load_engine(self) -> None:
        import numpy as np
        import cirq

        self.np = np
        self.cirq = cirq
        # Cirq 1.6.1's stabilizer mixture fallback handles one-qubit matrices
        # only. Use its act_on protocol to sample the exact two-qubit Pauli
        # mixture, then let Cirq apply each selected Pauli.
        class TwoQubitDepolarizing(cirq.Gate):
            def __init__(self, probability):
                self.probability = probability

            def _num_qubits_(self):
                return 2

            def _act_on_(self, sim_state, qubits):
                if sim_state.prng.random_sample() < self.probability:
                    code = int(sim_state.prng.randint(1, 16))
                    for q, digit in zip(qubits, (code & 3, code >> 2)):
                        if digit:
                            cirq.act_on((cirq.X, cirq.Y, cirq.Z)[digit - 1], sim_state, [q])
                return True

            def _has_stabilizer_effect_(self):
                return True

        self.two_qubit_depolarizing = TwoQubitDepolarizing
        self.native_reset = True
        self.reset_lowering = "cirq.ResetChannel"

    def _simulator(self, seed: int) -> Any:
        return self.cirq.CliffordSimulator(seed=int(seed % (2**31 - 1)), split_untangled_states=False)

    def _fallback_reset_ops(self, qubit: Any, index: int) -> list[Any]:
        key = f"_fsbench_reset_{index}"
        measurement = self.cirq.measure(qubit, key=key)
        correction = self.cirq.X(qubit).with_classical_controls(key)
        return [measurement, correction]

    def warmup(self) -> None:
        qubit, second = self.cirq.LineQubit.range(2)
        native = self.cirq.Circuit(
            self.cirq.H(qubit),
            self.cirq.S(qubit),
            self.cirq.CNOT(qubit, second),
            self.cirq.bit_flip(0.001)(qubit),
            self.cirq.depolarize(0.001)(qubit),
            self.two_qubit_depolarizing(0.001)(qubit, second),
            self.cirq.measure(qubit, key="m"),
            self.cirq.measure(second, key="m"),
            self.cirq.reset(qubit),
            self.cirq.reset(second),
        )
        try:
            self._simulator(1).run(native, repetitions=2)
        except Exception:
            fallback = self.cirq.Circuit(
                self.cirq.H(qubit),
                self.cirq.S(qubit),
                self.cirq.CNOT(qubit, second),
                self.cirq.measure(qubit, key="m"),
                *self._fallback_reset_ops(qubit, 0),
                self.cirq.measure(second, key="m"),
                *self._fallback_reset_ops(second, 1),
            )
            self._simulator(1).run(fallback, repetitions=2)
            self.native_reset = False
            self.reset_lowering = "measurement plus classically controlled X"

    def probe_details(self) -> dict[str, Any]:
        version = getattr(self.cirq, "__version__", "unknown")
        return {
            "engine": self.engine_key,
            "label": self.engine_label,
            "available": True,
            "version": version,
            "expected_version": "1.6.1",
            "version_ok": version == "1.6.1",
            "adapter": "neutral IR -> Cirq circuit -> CliffordSimulator",
            "reset_lowering": self.reset_lowering,
            "two_qubit_noise_lowering": "exact Pauli mixture via Cirq act_on protocol",
            "single_threaded": True,
            "capabilities": {
                "operations": ["H", "S", "CX", "M", "R", "X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"],
                "ordered_measurement_records": True,
                "count_only": True,
                "deterministic_seed": True,
                "max_batch_shots_honored": True,
            },
        }

    def run_iteration(
        self,
        *,
        case_path: Path,
        shots: int,
        seed: int,
        max_batch_shots: int,
        return_records: bool,
    ) -> IterationResult:
        case = load_case(case_path)
        qubits = self.cirq.LineQubit.range(int(case.metadata["num_qubits"]))
        cirq_operations: list[Any] = []
        reset_index = 0
        for operation in case.operations:
            if operation.probability is not None:
                if operation.name == "X_ERROR":
                    gate = self.cirq.bit_flip(operation.probability)
                elif operation.name == "DEPOLARIZE2":
                    gate = self.two_qubit_depolarizing(operation.probability)
                else:
                    gate = self.cirq.depolarize(operation.probability)
                cirq_operations.append(gate.on(*(qubits[q] for q in operation.targets)))
            elif operation.name == "H":
                cirq_operations.append(self.cirq.H(qubits[operation.targets[0]]))
            elif operation.name == "S":
                cirq_operations.append(self.cirq.S(qubits[operation.targets[0]]))
            elif operation.name == "CX":
                cirq_operations.append(
                    self.cirq.CNOT(qubits[operation.targets[0]], qubits[operation.targets[1]])
                )
            elif operation.name == "M":
                cirq_operations.append(
                    self.cirq.measure(qubits[operation.targets[0]], key="m")
                )
            elif operation.name == "R":
                qubit = qubits[operation.targets[0]]
                if self.native_reset:
                    cirq_operations.append(self.cirq.reset(qubit))
                else:
                    cirq_operations.extend(self._fallback_reset_ops(qubit, reset_index))
                    reset_index += 1
            else:
                raise ValueError(f"unsupported operation {operation.name}")
        circuit = self.cirq.Circuit(cirq_operations)
        measurements = int(case.metadata["num_measurements"])
        total_ones = 0
        record_batches: list[Any] = []
        for batch_index, (_, batch_shots) in enumerate(batch_ranges(shots, max_batch_shots)):
            result = self._simulator(derive_seed(seed, batch_index)).run(
                circuit, repetitions=batch_shots
            )
            records = self.np.asarray(result.records["m"], dtype=self.np.bool_).reshape(
                batch_shots, measurements
            )
            total_ones += int(self.np.count_nonzero(records))
            if return_records:
                record_batches.append(records)
        full_records = self.np.concatenate(record_batches, axis=0) if return_records else None
        return IterationResult(
            one_bits_total=total_ones,
            shots_completed=shots,
            num_measurements=measurements,
            records=full_records,
            details={
                "max_resident_record_shots": min(shots, max_batch_shots),
                "retained_records_across_batches": return_records,
                "reset_lowering": self.reset_lowering,
            },
        )


if __name__ == "__main__":
    worker_main(CirqWorker())
