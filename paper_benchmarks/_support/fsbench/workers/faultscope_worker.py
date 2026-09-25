"""FaultScope native packed-sampler worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from fsbench.ir import load_case
from fsbench.protocol import EngineWorker, IterationResult, batch_ranges, derive_seed, worker_main


class FaultScopeWorker(EngineWorker):
    engine_key = "faultscope"
    engine_label = "FaultScope"

    def load_engine(self) -> None:
        import numpy as np
        import faultscope
        from faultscope import Circuit, Operation
        from faultscope.runtime import compile_native_sampler

        self.np = np
        self.faultscope = faultscope
        self.Circuit = Circuit
        self.Operation = Operation
        self.compile_native_sampler = compile_native_sampler
        self.noise_models = {
            "X_ERROR": faultscope.BernoulliPauliNoise("X"),
            "DEPOLARIZE1": faultscope.SingleQubitDepolarizing(),
            "DEPOLARIZE2": faultscope.TwoQubitDepolarizing(),
        }

    def _build(self, case: Any) -> Any:
        operations: list[Any] = []
        for operation in case.operations:
            if operation.name in self.noise_models:
                location = self.faultscope.NoiseLocation(
                    id=f"noise_{len(operations)}", model=self.noise_models[operation.name],
                    rate=operation.probability, qubits=operation.targets,
                )
                operations.append(self.Operation.noise(location))
            elif operation.name == "H":
                operations.append(self.Operation.h(operation.targets[0]))
            elif operation.name == "S":
                operations.append(self.Operation.s(operation.targets[0]))
            elif operation.name == "CX":
                control, target = operation.targets
                operations.append(self.Operation.cx(control, target))
            elif operation.name == "M":
                operations.append(
                    self.Operation.measure(
                        operation.targets[0],
                        key=f"m{operation.measurement_index}",
                        basis="Z",
                    )
                )
            elif operation.name == "R":
                operations.append(
                    self.Operation.reset(operation.targets[0], key=None, basis="Z")
                )
            else:  # pragma: no cover - IR validation prevents this
                raise ValueError(operation.name)
        return self.Circuit(n_qubits=int(case.metadata["num_qubits"]), operations=operations)

    def warmup(self) -> None:
        circuit = self.Circuit(
            n_qubits=2,
            operations=[
                self.Operation.h(0),
                self.Operation.s(0),
                self.Operation.cx(0, 1),
                *(self.Operation.noise(self.faultscope.NoiseLocation(
                    id=f"warmup_{name}", model=model, rate=0.001,
                    qubits=(0, 1) if name == "DEPOLARIZE2" else (0,),
                )) for name, model in self.noise_models.items()),
                self.Operation.measure(0, key="m0", basis="Z"),
                self.Operation.measure(1, key="m1", basis="Z"),
                self.Operation.reset(0, key=None, basis="Z"),
                self.Operation.reset(1, key=None, basis="Z"),
            ],
        )
        self.compile_native_sampler(circuit).sample_measurements(2, seed=1)

    def probe_details(self) -> dict[str, Any]:
        version = getattr(self.faultscope, "__version__", "unknown")
        package_file = Path(self.faultscope.__file__).resolve()
        native_file = package_file.parent / "_native.abi3.so"
        if not native_file.exists():
            candidates = sorted(package_file.parent.glob("_native*.so")) + sorted(
                package_file.parent.glob("_native*.pyd")
            )
            native_file = candidates[0] if candidates else package_file
        digest = hashlib.sha256(native_file.read_bytes()).hexdigest()
        distribution = importlib.metadata.distribution("faultscope")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
        vcs_info = (direct_url or {}).get("vcs_info", {})
        editable = bool((direct_url or {}).get("dir_info", {}).get("editable", False))
        wheel_metadata = distribution.read_text("WHEEL")
        installation_mode = "editable" if editable else ("wheel" if wheel_metadata else "unknown")
        return {
            "engine": self.engine_key,
            "label": self.engine_label,
            "available": True,
            "version": version,
            "package_file": str(package_file),
            "native_module_file": str(native_file),
            "native_module_sha256": digest,
            "installation_mode": installation_mode,
            "direct_url": direct_url,
            "source_repository": (direct_url or {}).get("url"),
            "source_commit": vcs_info.get("commit_id"),
            "source_requested_revision": vcs_info.get("requested_revision"),
            "adapter": "neutral IR -> Circuit/Operation -> compile_native_sampler",
            "cx_lowering": "native CX(control,target)",
            "single_threaded": True,
            "capabilities": {
                "operations": ["H", "S", "CX", "M", "R", "X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"],
                "ordered_measurement_records": True,
                "count_only": True,
                "deterministic_seed": True,
                "max_batch_shots_honored": True,
            },
        }

    def _masks_to_records(self, masks: dict[str, int], shots: int, measurements: int) -> Any:
        records = self.np.zeros((shots, measurements), dtype=self.np.bool_)
        for measurement_index in range(measurements):
            mask = int(masks[f"m{measurement_index}"])
            records[:, measurement_index] = self.np.fromiter(
                ((mask >> shot) & 1 for shot in range(shots)),
                dtype=self.np.uint8,
                count=shots,
            )
        return records

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
        circuit = self._build(case)
        sampler = self.compile_native_sampler(circuit)
        measurements = int(case.metadata["num_measurements"])
        total_ones = 0
        record_batches: list[Any] = []
        for batch_index, (_, batch_shots) in enumerate(batch_ranges(shots, max_batch_shots)):
            masks = dict(
                sampler.sample_measurements(
                    batch_shots, seed=derive_seed(seed, batch_index)
                )
            )
            expected_keys = {f"m{index}" for index in range(measurements)}
            if set(masks) != expected_keys:
                raise RuntimeError(
                    "FaultScope measurement keys do not cover the canonical record"
                )
            total_ones += sum(int(mask).bit_count() for mask in masks.values())
            if return_records:
                record_batches.append(self._masks_to_records(masks, batch_shots, measurements))
        full_records = self.np.concatenate(record_batches, axis=0) if return_records else None
        return IterationResult(
            one_bits_total=total_ones,
            shots_completed=shots,
            num_measurements=measurements,
            records=full_records,
            details={
                "max_resident_record_shots": min(shots, max_batch_shots),
                "retained_records_across_batches": return_records,
            },
        )


if __name__ == "__main__":
    worker_main(FaultScopeWorker())
