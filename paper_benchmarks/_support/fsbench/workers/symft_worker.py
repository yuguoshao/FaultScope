"""SymFT 0.1.1 CPU worker using ordered, bit-packed measurement records."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

from fsbench.ir import load_case
from fsbench.protocol import EngineWorker, IterationResult, batch_ranges, derive_seed, worker_main


class SymFTWorker(EngineWorker):
    engine_key = "symft"
    engine_label = "SymFT"

    def load_engine(self) -> None:
        import numpy as np
        import symft

        self.np = np
        self.symft = symft

    def warmup(self) -> None:
        circuit = self.symft.Circuit("H 0\nS 0\nCX 0 1\nX_ERROR(0.001) 0\nDEPOLARIZE1(0.001) 0\nDEPOLARIZE2(0.001) 0 1\nM 0 1\nR 0 1")
        circuit.compile_sampler(batch=True, batch_size=2, sample_chunk_shots=2).sample(
            shots=2, seed=1, bit_packed=True
        )

    def probe_details(self) -> dict[str, Any]:
        distribution = importlib.metadata.distribution("symft")
        native_hashes = {
            str(item): hashlib.sha256(distribution.locate_file(item).read_bytes()).hexdigest()
            for item in distribution.files or ()
            if str(item).endswith((".so", ".pyd", ".dylib"))
        }
        return {
            "engine": self.engine_key,
            "label": self.engine_label,
            "available": True,
            "version": self.symft.__version__,
            "expected_version": "0.1.1",
            "version_ok": self.symft.__version__ == "0.1.1",
            "adapter": (
                "neutral IR -> SymFT text -> compile_sampler(batch=True) -> sample(bit_packed=True)"
            ),
            "single_threaded": True,
            "simd_backend": self.symft.simd_backend(),
            "cuda_enabled": self.symft.cuda_enabled(),
            "device": "cpu",
            "native_modules_sha256": native_hashes,
            "batch_seed_policy": "derive_seed(invocation_seed, batch_index)",
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
        circuit = self.symft.Circuit("\n".join(lines))
        # Compilation/planning belongs to each timed E2E invocation. Both
        # internal batch sizes and Python-visible buffers obey the same cap.
        batch_limit = min(shots, max_batch_shots)
        sampler = circuit.compile_sampler(
            batch=True, batch_size=batch_limit, sample_chunk_shots=batch_limit
        )

        measurements = int(case.metadata["num_measurements"])
        row_bytes = (measurements + 7) // 8
        trailing_bits = measurements % 8
        trailing_mask = (1 << trailing_bits) - 1 if trailing_bits else 0xFF
        total_ones = 0
        record_batches: list[Any] = []
        for batch_index, (_, batch_shots) in enumerate(batch_ranges(shots, max_batch_shots)):
            # SymFT restarts its RNG on each sample() call. Distinct derived
            # seeds prevent repeated batches while preserving reproducibility.
            packed = self.np.asarray(sampler.sample(
                shots=batch_shots, seed=derive_seed(seed, batch_index), bit_packed=True
            ))
            if packed.dtype != self.np.uint8:
                raise RuntimeError(f"unexpected SymFT packed dtype {packed.dtype}")
            if packed.shape != (batch_shots, row_bytes):
                raise RuntimeError(f"unexpected SymFT packed shape {packed.shape}")
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
                "symft_sampling_method": "sample(bit_packed=True)",
                "symft_batch_size": batch_limit,
                "symft_sample_chunk_shots": batch_limit,
                "packed_row_bytes": row_bytes,
                "unpacked_records_materialized": False,
            },
        )


if __name__ == "__main__":
    worker_main(SymFTWorker())
