"""Shared JSON-lines worker protocol and record encoding helpers."""

from __future__ import annotations

import base64
import gc
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


PROTOCOL_VERSION = 1
_SEED_STEP = 0x1E37_79B9_7F4A_7C15
_SEED_MASK = (1 << 63) - 1


def derive_seed(base_seed: int, ordinal: int) -> int:
    """Derive a deterministic positive 63-bit seed."""

    return (int(base_seed) + _SEED_STEP * (int(ordinal) + 1)) & _SEED_MASK


def batch_ranges(shots: int, max_batch_shots: int) -> list[tuple[int, int]]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    if max_batch_shots <= 0:
        raise ValueError("max_batch_shots must be positive")
    return [
        (start, min(max_batch_shots, shots - start))
        for start in range(0, shots, max_batch_shots)
    ]


def encode_records(records: Any) -> tuple[str, list[int]]:
    """Encode a shot-major Boolean matrix using little-endian bits per row."""

    import numpy as np

    array = np.asarray(records, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"records must be two-dimensional, found {array.shape}")
    packed = np.packbits(array, axis=1, bitorder="little")
    return base64.b64encode(packed.tobytes(order="C")).decode("ascii"), list(array.shape)


def encode_bit_packed_records(
    packed_records: Any, *, shots: int, measurements: int
) -> tuple[str, list[int]]:
    """Encode already-packed little-endian shot-major records without unpacking."""

    import numpy as np

    shots = int(shots)
    measurements = int(measurements)
    if shots < 0 or measurements < 0:
        raise ValueError("record dimensions must be non-negative")
    array = np.asarray(packed_records)
    if array.dtype != np.uint8:
        raise ValueError(f"packed records must have dtype uint8, found {array.dtype}")
    expected_shape = (shots, (measurements + 7) // 8)
    if array.shape != expected_shape:
        raise ValueError(
            f"packed records have shape {array.shape}, expected {expected_shape}"
        )
    return (
        base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        [shots, measurements],
    )


def decode_records(encoded: str, shape: list[int] | tuple[int, int]) -> Any:
    import numpy as np

    shots, measurements = (int(shape[0]), int(shape[1]))
    row_bytes = (measurements + 7) // 8
    raw = base64.b64decode(encoded)
    expected = shots * row_bytes
    if len(raw) != expected:
        raise ValueError(f"packed record length {len(raw)} does not match {expected}")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(shots, row_bytes)
    return np.unpackbits(packed, axis=1, count=measurements, bitorder="little").astype(
        np.bool_, copy=False
    )


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def write_message(stream: TextIO, message: dict[str, Any]) -> None:
    stream.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def _installed_distribution_lock() -> dict[str, str]:
    lock: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            lock[str(name).lower()] = str(distribution.version)
    return dict(sorted(lock.items()))


@dataclass(slots=True)
class IterationResult:
    one_bits_total: int
    shots_completed: int
    num_measurements: int
    records: Any | None = None
    packed_records: Any | None = None
    details: dict[str, Any] | None = None


class EngineWorker(ABC):
    """Base class for a persistent, pre-warmed benchmark worker."""

    engine_key: str
    engine_label: str

    @abstractmethod
    def load_engine(self) -> None:
        """Import the simulator and cache module references."""

    @abstractmethod
    def warmup(self) -> None:
        """Exercise every code path whose JIT/startup cost is excluded."""

    @abstractmethod
    def probe_details(self) -> dict[str, Any]:
        """Return version and capability details."""

    @abstractmethod
    def run_iteration(
        self,
        *,
        case_path: Path,
        shots: int,
        seed: int,
        max_batch_shots: int,
        return_records: bool,
    ) -> IterationResult:
        """Read, translate, prepare, execute, and reduce one E2E invocation."""


def serve_worker(worker: EngineWorker) -> None:
    available = True
    unavailable_reason: str | None = None
    try:
        worker.load_engine()
        worker.warmup()
        gc.collect()
        probe = worker.probe_details()
        probe.setdefault(
            "runtime",
            {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "installed_distributions": _installed_distribution_lock(),
            },
        )
        probe.setdefault(
            "capabilities",
            {
                "operations": ["H", "S", "CX", "M", "R"],
                "ordered_measurement_records": True,
                "count_only": True,
                "deterministic_seed": True,
                "max_batch_shots_honored": True,
            },
        )
    except Exception as exc:  # pragma: no cover - depends on optional environments
        available = False
        unavailable_reason = f"{type(exc).__name__}: {exc}"
        probe = {
            "engine": worker.engine_key,
            "label": worker.engine_label,
            "available": False,
            "reason": unavailable_reason,
        }

    write_message(
        sys.stdout,
        {
            "type": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "engine": worker.engine_key,
            "available": available,
            "probe": probe,
        },
    )

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw_line)
            command = request.get("command")
            request_id = request.get("request_id")
            if command == "shutdown":
                write_message(sys.stdout, {"type": "bye", "request_id": request_id})
                return
            if command == "probe":
                write_message(
                    sys.stdout,
                    {
                        "type": "probe",
                        "request_id": request_id,
                        "status": "ok" if available else "unsupported",
                        "details": probe,
                    },
                )
                continue
            if command != "run":
                raise ValueError(f"unknown worker command {command!r}")
            if not available:
                write_message(
                    sys.stdout,
                    {
                        "type": "result",
                        "request_id": request_id,
                        "status": "unsupported",
                        "error": unavailable_reason,
                    },
                )
                continue

            shots = int(request["shots"])
            iterations = int(request.get("iterations", 1))
            max_batch_shots = int(request.get("max_batch_shots", 1_000))
            return_records = bool(request.get("return_records", False))
            if iterations <= 0:
                raise ValueError("iterations must be positive")
            if return_records and iterations != 1:
                raise ValueError("return_records requires iterations=1")

            started_ns = time.perf_counter_ns()
            one_bits_total = 0
            shots_completed = 0
            num_measurements: int | None = None
            encoded_records: str | None = None
            records_shape: list[int] | None = None
            iteration_details: list[dict[str, Any]] = []
            for iteration in range(iterations):
                result = worker.run_iteration(
                    case_path=Path(request["case_path"]),
                    shots=shots,
                    seed=derive_seed(int(request["seed"]), iteration),
                    max_batch_shots=max_batch_shots,
                    return_records=return_records,
                )
                if num_measurements is None:
                    num_measurements = result.num_measurements
                elif num_measurements != result.num_measurements:
                    raise RuntimeError("measurement count changed across iterations")
                one_bits_total += int(result.one_bits_total)
                shots_completed += int(result.shots_completed)
                if result.details:
                    iteration_details.append(result.details)
                if return_records:
                    has_records = result.records is not None
                    has_packed_records = result.packed_records is not None
                    if has_records == has_packed_records:
                        raise RuntimeError(
                            "return_records requires exactly one unpacked or packed record payload"
                        )
                    if has_packed_records:
                        encoded_records, records_shape = encode_bit_packed_records(
                            result.packed_records,
                            shots=result.shots_completed,
                            measurements=result.num_measurements,
                        )
                    else:
                        encoded_records, records_shape = encode_records(result.records)
            wall_time_ns = time.perf_counter_ns() - started_ns
            write_message(
                sys.stdout,
                {
                    "type": "result",
                    "request_id": request_id,
                    "status": "ok",
                    "engine": worker.engine_key,
                    "wall_time_ns": wall_time_ns,
                    "wall_seconds": wall_time_ns / 1e9,
                    "seconds_per_e2e": wall_time_ns / 1e9 / iterations,
                    "iterations": iterations,
                    "shots_requested": shots,
                    "shots_completed": shots_completed,
                    "num_measurements": num_measurements,
                    "one_bits_total": one_bits_total,
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "records_b64": encoded_records,
                    "records_shape": records_shape,
                    "records_encoding": "shot-major-packbits-little" if return_records else None,
                    "details": iteration_details,
                },
            )
        except MemoryError as exc:
            write_message(
                sys.stdout,
                {
                    "type": "result",
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "status": "oom",
                    "error": str(exc) or "MemoryError",
                },
            )
        except Exception as exc:
            write_message(
                sys.stdout,
                {
                    "type": "result",
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                },
            )


def worker_main(worker: EngineWorker) -> None:
    defaults = {
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "QISKIT_PARALLEL": "FALSE",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    serve_worker(worker)
