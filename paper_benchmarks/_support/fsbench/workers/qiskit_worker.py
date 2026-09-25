"""Qiskit Aer stabilizer worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fsbench.ir import load_case
from fsbench.protocol import EngineWorker, IterationResult, batch_ranges, derive_seed, worker_main


class QiskitAerWorker(EngineWorker):
    engine_key = "qiskit-aer"
    engine_label = "Qiskit Aer"

    def load_engine(self) -> None:
        import numpy as np
        import qiskit
        import qiskit_aer
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import depolarizing_error, pauli_error

        self.np = np
        self.qiskit = qiskit
        self.qiskit_aer = qiskit_aer
        self.QuantumCircuit = QuantumCircuit
        self.transpile = transpile
        self.AerSimulator = AerSimulator
        self.depolarizing_error = depolarizing_error
        self.pauli_error = pauli_error

    def _backend(self) -> Any:
        return self.AerSimulator(
            method="stabilizer",
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )

    def warmup(self) -> None:
        backend = self._backend()
        circuit = self.QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.s(0)
        circuit.cx(0, 1)
        circuit.append(self.pauli_error([("I", 0.999), ("X", 0.001)]).to_instruction(), [0])
        circuit.append(self.depolarizing_error(4 * 0.001 / 3, 1).to_instruction(), [0])
        circuit.append(self.depolarizing_error(16 * 0.001 / 15, 2).to_instruction(), [0, 1])
        circuit.measure([0, 1], [0, 1])
        circuit.reset([0, 1])
        prepared = self.transpile(circuit, backend, optimization_level=0, seed_transpiler=1)
        backend.run(prepared, shots=2, seed_simulator=1).result().get_counts()

    def probe_details(self) -> dict[str, Any]:
        qiskit_version = getattr(self.qiskit, "__version__", "unknown")
        aer_version = getattr(self.qiskit_aer, "__version__", "unknown")
        return {
            "engine": self.engine_key,
            "label": self.engine_label,
            "available": True,
            "version": qiskit_version,
            "aer_version": aer_version,
            "expected_version": "2.5.0",
            "expected_aer_version": "0.17.2",
            "version_ok": qiskit_version == "2.5.0" and aer_version == "0.17.2",
            "adapter": "neutral IR -> QuantumCircuit -> transpile(level=0) -> Aer stabilizer",
            "single_threaded": True,
            "capabilities": {
                "operations": ["H", "S", "CX", "M", "R", "X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"],
                "ordered_measurement_records": True,
                "count_only": True,
                "deterministic_seed": True,
                "max_batch_shots_honored": True,
            },
        }

    @staticmethod
    def _clean_bits(value: str) -> str:
        return value.replace(" ", "")

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
        measurements = int(case.metadata["num_measurements"])
        circuit = self.QuantumCircuit(int(case.metadata["num_qubits"]), measurements)
        noise_instructions = {}
        for operation in case.operations:
            if operation.probability is not None:
                key = (operation.name, operation.probability)
                if key not in noise_instructions:
                    p = operation.probability
                    if operation.name == "X_ERROR":
                        error = self.pauli_error([("I", 1 - p), ("X", p)])
                    else:
                        n = len(operation.targets)
                        # Aer uses the mixing probability; the neutral format
                        # uses total probability of a nonidentity Pauli.
                        error = self.depolarizing_error(p * 4**n / (4**n - 1), n)
                    noise_instructions[key] = error.to_instruction()
                circuit.append(noise_instructions[key], list(operation.targets))
            elif operation.name == "H":
                circuit.h(operation.targets[0])
            elif operation.name == "S":
                circuit.s(operation.targets[0])
            elif operation.name == "CX":
                circuit.cx(*operation.targets)
            elif operation.name == "M":
                circuit.measure(operation.targets[0], int(operation.measurement_index))
            elif operation.name == "R":
                circuit.reset(operation.targets[0])
            else:
                raise ValueError(f"unsupported operation {operation.name}")
        backend = self._backend()
        prepared = self.transpile(
            circuit,
            backend,
            optimization_level=0,
            seed_transpiler=int(seed % (2**31 - 1)),
        )

        total_ones = 0
        record_batches: list[Any] = []
        for batch_index, (_, batch_shots) in enumerate(batch_ranges(shots, max_batch_shots)):
            run = backend.run(
                prepared,
                shots=batch_shots,
                seed_simulator=int(derive_seed(seed, batch_index) % (2**31 - 1)),
                memory=return_records,
            )
            result = run.result()
            if return_records:
                memory = [self._clean_bits(item) for item in result.get_memory()]
                records = self.np.zeros((batch_shots, measurements), dtype=self.np.bool_)
                for shot, bit_string in enumerate(memory):
                    if len(bit_string) != measurements:
                        raise RuntimeError(f"unexpected Qiskit memory width {len(bit_string)}")
                    records[shot, :] = self.np.fromiter(
                        (bit == "1" for bit in reversed(bit_string)),
                        dtype=self.np.bool_,
                        count=measurements,
                    )
                total_ones += int(self.np.count_nonzero(records))
                record_batches.append(records)
            else:
                counts = result.get_counts()
                for bit_string, count in counts.items():
                    clean = self._clean_bits(bit_string)
                    if len(clean) != measurements:
                        raise RuntimeError(
                            f"Qiskit counts width {len(clean)} does not cover {measurements} measurements"
                        )
                    total_ones += clean.count("1") * int(count)
        full_records = self.np.concatenate(record_batches, axis=0) if return_records else None
        return IterationResult(
            one_bits_total=total_ones,
            shots_completed=shots,
            num_measurements=measurements,
            records=full_records,
            details={
                "max_resident_record_shots": min(shots, max_batch_shots),
                "retained_records_across_batches": return_records,
                "count_only_uses_full_width_counts": not return_records,
            },
        )


if __name__ == "__main__":
    worker_main(QiskitAerWorker())
