"""Stim 1.16 worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fsbench.ir import load_case
from fsbench.protocol import EngineWorker, IterationResult, batch_ranges, worker_main


class StimWorker(EngineWorker):
    engine_key = "stim"
    engine_label = "Stim"

    def load_engine(self) -> None:
        import numpy as np
        import stim

        self.np = np
        self.stim = stim

    def warmup(self) -> None:
        circuit = self.stim.Circuit("H 0\nS 0\nCX 0 1\nX_ERROR(0.001) 0\nDEPOLARIZE1(0.001) 0\nDEPOLARIZE2(0.001) 0 1\nM 0 1\nR 0 1")
        circuit.compile_sampler(seed=1).sample_bit_packed(2)

    def probe_details(self) -> dict[str, Any]:
        return {
            "engine": self.engine_key,
            "label": self.engine_label,
            "available": True,
            "version": self.stim.__version__,
            "expected_version": "1.16.0",
            "version_ok": self.stim.__version__ == "1.16.0",
            "adapter": (
                "neutral IR -> Stim text -> compile_sampler -> sample_bit_packed"
            ),
            "single_threaded": True,
            "capabilities": {
                "operations": ["H", "S", "CX", "M", "R", "X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"],
                "ordered_measurement_records": True,
                "count_only": True,
                "deterministic_seed": True,
                "max_batch_shots_honored": True,
                "bit_packed_records": True,
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
        lines: list[str] = []
        for operation in case.operations:
            if operation.probability is not None:
                targets = " ".join(str(q) for q in operation.targets)
                lines.append(f"{operation.name}({operation.probability!r}) {targets}")
            elif operation.name == "CX":
                lines.append(f"CX {operation.targets[0]} {operation.targets[1]}")
            else:
                lines.append(f"{operation.name} {operation.targets[0]}")
        circuit = self.stim.Circuit("\n".join(lines))
        sampler = circuit.compile_sampler(seed=seed)

        measurements = int(case.metadata["num_measurements"])
        row_bytes = (measurements + 7) // 8
        trailing_bits = measurements % 8
        trailing_mask = (1 << trailing_bits) - 1 if trailing_bits else 0xFF
        total_ones = 0
        record_batches: list[Any] = []
        for batch_index, (_, batch_shots) in enumerate(batch_ranges(shots, max_batch_shots)):
            # A single compiled sampler is intentionally reused within one E2E
            # invocation; its RNG stream advances across fixed-size batches.
            del batch_index
            packed = self.np.asarray(sampler.sample_bit_packed(batch_shots))
            if packed.dtype != self.np.uint8:
                raise RuntimeError(f"unexpected Stim packed dtype {packed.dtype}")
            if packed.shape != (batch_shots, row_bytes):
                raise RuntimeError(f"unexpected Stim packed shape {packed.shape}")
            if row_bytes:
                if trailing_bits:
                    if row_bytes > 1:
                        total_ones += int(
                            self.np.bitwise_count(packed[:, :-1]).sum(
                                dtype=self.np.uint64
                            )
                        )
                    total_ones += int(
                        self.np.bitwise_count(packed[:, -1] & trailing_mask).sum(
                            dtype=self.np.uint64
                        )
                    )
                else:
                    total_ones += int(
                        self.np.bitwise_count(packed).sum(dtype=self.np.uint64)
                    )
            if return_records:
                record_batches.append(packed)
        full_packed_records = (
            self.np.concatenate(record_batches, axis=0) if return_records else None
        )
        return IterationResult(
            one_bits_total=total_ones,
            shots_completed=shots,
            num_measurements=measurements,
            packed_records=full_packed_records,
            details={
                "max_resident_record_shots": min(shots, max_batch_shots),
                "retained_records_across_batches": return_records,
                "stim_sampling_method": "sample_bit_packed",
                "packed_row_bytes": row_bytes,
                "unpacked_records_materialized": False,
            },
        )


if __name__ == "__main__":
    worker_main(StimWorker())
