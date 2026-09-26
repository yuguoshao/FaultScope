"""Worker orchestration, resource enforcement, and adaptive repetition policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import selectors
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil

from .constants import MASTER_SEED, PROFILES, Profile
from .engines import EngineSpec, engine_spec, version_matches, worker_environment
from .host import host_manifest, memory_cap_bytes, source_tree_sha256
from .ir import case_metadata_paths, load_case_metadata, sha256_file
from .protocol import PROTOCOL_VERSION, derive_seed
from .summary import read_jsonl, summarize_results


MANIFEST_SCHEMA = "fsbench-run-manifest-v1"
MANIFEST_SCHEMA_V2 = "fsbench-run-manifest-v2"
SUPPORTED_MANIFEST_SCHEMAS = (MANIFEST_SCHEMA, MANIFEST_SCHEMA_V2)
MAX_BATCH_SHOTS = 1_000
STARTUP_TIMEOUT_SECONDS = 300.0
FAST_SAMPLE_TARGET_SECONDS = 1.0
MAX_INVOCATIONS_PER_SAMPLE = 1_000_000
_RAW_APPEND_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with _RAW_APPEND_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _stable_ordinal(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 31) - 1)


def _metadata_case_id(metadata: dict[str, Any]) -> str:
    if metadata.get("case_id"):
        return str(metadata["case_id"])
    if "distance" in metadata:
        distance = int(metadata["distance"])
        rounds = int(metadata.get("rounds", distance))
        return f"rotated_memory_z_d{distance}_r{rounds}"
    raise ValueError("case metadata has no stable identity")


def _row_case_id(row: dict[str, Any]) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    if "distance" in row:
        distance = int(row["distance"])
        rounds = int(row.get("rounds", distance))
        return f"rotated_memory_z_d{distance}_r{rounds}"
    return "<missing-case-id>"


def _case_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    parameters = metadata.get("parameters", {})
    fields: dict[str, Any] = {
        "case_id": _metadata_case_id(metadata),
        "family": str(metadata.get("family", "unknown")),
        "width": parameters.get("width"),
        "depth": parameters.get("depth"),
        "instance": parameters.get("instance"),
        "circuit_seed": parameters.get("circuit_seed"),
    }
    if "distance" in metadata:
        fields["distance"] = int(metadata["distance"])
    if "rounds" in metadata:
        fields["rounds"] = int(metadata["rounds"])
    return fields


def _select_cpu(cpu_id: int | None) -> int | None:
    if cpu_id is not None:
        return int(cpu_id)
    if not sys_platform_linux():
        return None
    try:
        allowed = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    return allowed[0] if allowed else None


def _parse_cpu_list(value: str) -> frozenset[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return frozenset(cpus)


def _physical_cpu_candidates(allowed: list[int], base_cpu: int) -> list[int]:
    ordered = [base_cpu, *(cpu for cpu in allowed if cpu != base_cpu)]
    candidates: list[int] = []
    seen_siblings: set[frozenset[int]] = set()
    for cpu in ordered:
        sibling_path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        try:
            siblings = _parse_cpu_list(sibling_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot identify physical core for allowed CPU {cpu}") from exc
        sibling_key = frozenset(value for value in siblings if value in allowed)
        if not sibling_key:
            sibling_key = frozenset({cpu})
        if sibling_key in seen_siblings:
            continue
        seen_siblings.add(sibling_key)
        candidates.append(cpu)
    return candidates


def _worker_cpu_slots(
    *, base_cpu: int | None, requested: int, require_exact: bool
) -> tuple[int | None, ...]:
    if requested <= 0:
        raise ValueError("parallel worker count must be positive")
    if not sys_platform_linux():
        return tuple(None for _ in range(requested))
    try:
        allowed = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        allowed = []
    if base_cpu is None or base_cpu not in allowed:
        raise RuntimeError(f"base CPU {base_cpu!r} is outside affinity {allowed}")
    candidates = _physical_cpu_candidates(allowed, base_cpu)
    if require_exact and len(candidates) < requested:
        raise RuntimeError(
            f"profile requires {requested} physical CPUs, found {len(candidates)}"
        )
    selected = candidates[:requested]
    if not selected:
        raise RuntimeError("no worker CPUs are available")
    return tuple(selected)


def sys_platform_linux() -> bool:
    return platform.system() == "Linux"


class WorkerFailure(RuntimeError):
    def __init__(self, status: str, message: str, *, peak_rss_bytes: int = 0):
        super().__init__(message)
        self.status = status
        self.peak_rss_bytes = peak_rss_bytes


class WorkerSession:
    """One fresh worker process with bounded message waits and RSS monitoring."""

    def __init__(
        self,
        spec: EngineSpec,
        *,
        cpu_id: int | None,
        memory_limit_bytes: int,
        startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self.spec = spec
        self.cpu_id = cpu_id
        self.memory_limit_bytes = int(memory_limit_bytes)
        self.startup_timeout = float(startup_timeout)
        self.process: subprocess.Popen[str] | None = None
        self._stderr: Any | None = None
        self._selector: selectors.BaseSelector | None = None
        self.peak_rss_bytes = 0
        self.ready: dict[str, Any] | None = None

    def __enter__(self) -> "WorkerSession":
        try:
            self.start()
        except Exception:
            self._terminate()
            if self._selector is not None:
                self._selector.close()
                self._selector = None
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def start(self) -> dict[str, Any]:
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.spec.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
                env=worker_environment(),
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            self._stderr.close()
            self._stderr = None
            raise WorkerFailure("unsupported", f"cannot start worker: {exc}") from exc

        assert self.process.stdout is not None
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        if self.cpu_id is not None:
            try:
                psutil.Process(self.process.pid).cpu_affinity([self.cpu_id])
            except (AttributeError, psutil.Error, ValueError) as exc:
                self._terminate()
                raise WorkerFailure(
                    "error", f"failed to pin worker to CPU {self.cpu_id}: {exc}"
                ) from exc

        ready = self._wait_message(timeout=self.startup_timeout, expected_type="ready")
        if int(ready.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise WorkerFailure(
                "error",
                f"worker protocol {ready.get('protocol_version')!r} != {PROTOCOL_VERSION}",
                peak_rss_bytes=self.peak_rss_bytes,
            )
        if ready.get("engine") != self.spec.key:
            raise WorkerFailure(
                "error",
                f"worker identified as {ready.get('engine')!r}, expected {self.spec.key!r}",
                peak_rss_bytes=self.peak_rss_bytes,
            )
        self.ready = ready
        return ready

    def _stderr_text(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read()[-8_000:].strip()
        except OSError:
            return ""

    def _sample_rss(self) -> int:
        if self.process is None:
            return 0
        total = 0
        try:
            root = psutil.Process(self.process.pid)
            try:
                processes = [root, *root.children(recursive=True)]
            except (psutil.Error, OSError, PermissionError):
                # Some managed macOS environments deny the global process-list
                # query needed by psutil.children(). The worker itself is still
                # monitored; Linux paper hosts are expected to expose children.
                processes = [root]
            for process in processes:
                try:
                    total += int(process.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, PermissionError):
            return self.peak_rss_bytes
        self.peak_rss_bytes = max(self.peak_rss_bytes, total)
        return total

    def _wait_message(self, *, timeout: float, expected_type: str) -> dict[str, Any]:
        if self.process is None or self._selector is None:
            raise WorkerFailure("error", "worker is not running")
        deadline = time.monotonic() + timeout
        while True:
            rss = self._sample_rss()
            if rss > self.memory_limit_bytes:
                self._terminate()
                raise WorkerFailure(
                    "oom",
                    f"process-tree RSS {rss} exceeded limit {self.memory_limit_bytes}",
                    peak_rss_bytes=self.peak_rss_bytes,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise WorkerFailure(
                    "timeout",
                    f"worker exceeded {timeout:.3f} s timeout",
                    peak_rss_bytes=self.peak_rss_bytes,
                )
            events = self._selector.select(min(0.05, remaining))
            if not events:
                if self.process.poll() is not None:
                    raise WorkerFailure(
                        "error",
                        f"worker exited with code {self.process.returncode}: {self._stderr_text()}",
                        peak_rss_bytes=self.peak_rss_bytes,
                    )
                continue
            assert self.process.stdout is not None
            line = self.process.stdout.readline()
            if not line:
                raise WorkerFailure(
                    "error",
                    f"worker closed stdout: {self._stderr_text()}",
                    peak_rss_bytes=self.peak_rss_bytes,
                )
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkerFailure(
                    "error", f"worker emitted invalid JSON: {line[:500]!r}"
                ) from exc
            if message.get("type") != expected_type:
                raise WorkerFailure(
                    "error",
                    f"expected worker message {expected_type!r}, got {message.get('type')!r}",
                    peak_rss_bytes=self.peak_rss_bytes,
                )
            return message

    def request(self, message: dict[str, Any], *, timeout: float, expected_type: str) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise WorkerFailure("error", "worker stdin is unavailable")
        try:
            self.process.stdin.write(
                json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerFailure(
                "error", f"failed to send request: {exc}", peak_rss_bytes=self.peak_rss_bytes
            ) from exc
        response = self._wait_message(timeout=timeout, expected_type=expected_type)
        if response.get("request_id") != message.get("request_id"):
            raise WorkerFailure(
                "error",
                "worker response request_id mismatch",
                peak_rss_bytes=self.peak_rss_bytes,
            )
        response["peak_rss_bytes"] = max(
            int(response.get("peak_rss_bytes") or 0), self.peak_rss_bytes
        )
        return response

    def _terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            try:
                self.process.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.request(
                    {"command": "shutdown", "request_id": uuid.uuid4().hex},
                    timeout=5.0,
                    expected_type="bye",
                )
            except WorkerFailure:
                self._terminate()
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


def _prefetch_case(metadata_path: Path) -> None:
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    for field in ("operations_file", "sidecar_file"):
        path = metadata_path.parent / metadata[field]
        with path.open("rb") as stream:
            while stream.read(1024 * 1024):
                pass


def probe_engine(
    engine: str,
    *,
    cpu_id: int | None = None,
    memory_limit_bytes: int | None = None,
) -> dict[str, Any]:
    spec = engine_spec(engine)
    try:
        with WorkerSession(
            spec,
            cpu_id=_select_cpu(cpu_id),
            memory_limit_bytes=memory_limit_bytes or memory_cap_bytes(),
        ) as worker:
            assert worker.ready is not None
            if not worker.ready.get("available", False):
                return {
                    "engine": engine,
                    "status": "unsupported",
                    "command": list(spec.command),
                    "details": worker.ready.get("probe", {}),
                }
            response = worker.request(
                {"command": "probe", "request_id": uuid.uuid4().hex},
                timeout=30.0,
                expected_type="probe",
            )
            details = dict(response.get("details", {}))
            version_ok, errors = version_matches(spec, details)
            return {
                "engine": engine,
                "status": response.get("status", "error"),
                "command": list(spec.command),
                "version_ok": version_ok,
                "version_errors": errors,
                "details": details,
                "peak_rss_bytes": response.get("peak_rss_bytes", 0),
            }
    except WorkerFailure as exc:
        return {
            "engine": engine,
            "status": exc.status,
            "command": list(spec.command),
            "version_ok": False,
            "version_errors": [str(exc)],
            "details": {"reason": str(exc)},
            "peak_rss_bytes": exc.peak_rss_bytes,
        }


def probe_engines(
    engines: Iterable[str],
    *,
    cpu_id: int | None = None,
    memory_limit_bytes: int | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        engine: probe_engine(
            engine, cpu_id=cpu_id, memory_limit_bytes=memory_limit_bytes
        )
        for engine in engines
    }


def run_worker_invocation(
    *,
    engine: str,
    case_path: Path,
    shots: int,
    seed: int,
    iterations: int,
    timeout_seconds: float,
    cpu_id: int | None,
    memory_limit_bytes: int,
    return_records: bool = False,
) -> dict[str, Any]:
    spec = engine_spec(engine)
    try:
        with WorkerSession(
            spec,
            cpu_id=_select_cpu(cpu_id),
            memory_limit_bytes=memory_limit_bytes,
        ) as worker:
            assert worker.ready is not None
            if not worker.ready.get("available", False):
                return {
                    "status": "unsupported",
                    "error": worker.ready.get("probe", {}).get("reason", "unavailable"),
                    "peak_rss_bytes": worker.peak_rss_bytes,
                }
            # This read happens after import/JIT warmup but before the worker's
            # run timer. The worker independently reopens every file in the
            # timed region.
            _prefetch_case(case_path)
            response = worker.request(
                {
                    "command": "run",
                    "request_id": uuid.uuid4().hex,
                    "case_path": str(case_path.resolve()),
                    "shots": int(shots),
                    "seed": int(seed),
                    "iterations": int(iterations),
                    "max_batch_shots": MAX_BATCH_SHOTS,
                    "return_records": bool(return_records),
                },
                timeout=timeout_seconds,
                expected_type="result",
            )
            return response
    except WorkerFailure as exc:
        return {
            "status": exc.status,
            "error": str(exc),
            "peak_rss_bytes": exc.peak_rss_bytes,
        }


def _engine_version(probe: dict[str, Any]) -> str:
    details = probe.get("details", {})
    if probe.get("engine") == "qiskit-aer":
        return f"Qiskit {details.get('version', 'unknown')} / Aer {details.get('aer_version', 'unknown')}"
    return str(details.get("version", "unknown"))


def _sample_row(
    *,
    run_id: str,
    profile: Profile,
    engine: str,
    probe: dict[str, Any],
    case: Any,
    shots: int,
    sample_index: int,
    phase: str,
    included: bool,
    iterations: int,
    seed: int,
    worker_cpu: int | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    metadata = case.metadata
    return {
        "schema": (
            MANIFEST_SCHEMA_V2
            if str(metadata.get("schema")) == "fsbench-v2"
            else MANIFEST_SCHEMA
        ),
        "record_type": "sample",
        "timestamp_utc": utc_now(),
        "run_id": run_id,
        "profile": profile.name,
        "engine": engine,
        "engine_label": engine_spec(engine).label,
        "engine_version": _engine_version(probe),
        **_case_fields(metadata),
        "shots": int(shots),
        "sample_index": int(sample_index),
        "phase": phase,
        "included_in_summary": bool(included),
        "iterations": int(iterations),
        "seed": int(seed),
        "worker_cpu": worker_cpu,
        "max_batch_shots": MAX_BATCH_SHOTS,
        "case_path": str(case.metadata_path),
        "case_operations_sha256": metadata["operations_sha256"],
        "num_qubits": int(metadata["num_qubits"]),
        "num_measurements": int(metadata["num_measurements"]),
        "two_qubit_operations": int(metadata["two_qubit_operations"]),
        **result,
    }


def _terminal_row(
    *,
    run_id: str,
    profile: Profile,
    engine: str,
    case: Any,
    shots: int,
    status: str,
    repetitions: int,
    low_sample: bool,
    worker_cpu: int | None = None,
    error: str = "",
    sampling_cap_reached: bool = False,
) -> dict[str, Any]:
    return {
        "schema": (
            MANIFEST_SCHEMA_V2
            if str(case.metadata.get("schema")) == "fsbench-v2"
            else MANIFEST_SCHEMA
        ),
        "record_type": "config_status",
        "timestamp_utc": utc_now(),
        "run_id": run_id,
        "profile": profile.name,
        "engine": engine,
        "engine_label": engine_spec(engine).label,
        **_case_fields(case.metadata),
        "shots": int(shots),
        "status": status,
        "repetitions": int(repetitions),
        "low_sample": bool(low_sample),
        "worker_cpu": worker_cpu,
        "sampling_cap_reached": bool(sampling_cap_reached),
        "error": error,
    }


def _run_configuration(
    *,
    raw_path: Path,
    run_id: str,
    profile: Profile,
    engine: str,
    probe: dict[str, Any],
    case: Any,
    shots: int,
    timeout_seconds: float,
    cpu_id: int | None,
    memory_limit_bytes: int,
    master_seed: int,
    existing_samples: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    case_id = _metadata_case_id(case.metadata)
    seed_identity: object = (
        int(case.metadata["distance"])
        if str(case.metadata.get("schema")) == "fsbench-v1"
        and "distance" in case.metadata
        else case_id
    )
    seed_base = derive_seed(master_seed, _stable_ordinal(engine, seed_identity, shots))
    existing_samples = list(existing_samples)
    sample_counter = max(
        (int(row.get("sample_index", -1)) for row in existing_samples), default=-1
    ) + 1
    included_rows: list[dict[str, Any]] = [
        row
        for row in existing_samples
        if row.get("included_in_summary") is True and row.get("status") == "ok"
    ]

    def invoke(iterations: int, phase: str, included: bool) -> dict[str, Any]:
        nonlocal sample_counter
        seed = derive_seed(seed_base, sample_counter)
        result = run_worker_invocation(
            engine=engine,
            case_path=case.metadata_path,
            shots=shots,
            seed=seed,
            iterations=iterations,
            timeout_seconds=timeout_seconds,
            cpu_id=cpu_id,
            memory_limit_bytes=memory_limit_bytes,
        )
        row = _sample_row(
            run_id=run_id,
            profile=profile,
            engine=engine,
            probe=probe,
            case=case,
            shots=shots,
            sample_index=sample_counter,
            phase=phase,
            included=included,
            iterations=iterations,
            seed=seed,
            worker_cpu=cpu_id,
            result=result,
        )
        sample_counter += 1
        return row

    calibration = invoke(1, "calibration", False)
    if calibration.get("status") != "ok":
        append_jsonl(raw_path, calibration)
        terminal = _terminal_row(
            run_id=run_id,
            profile=profile,
            engine=engine,
            case=case,
            shots=shots,
            status=str(calibration.get("status", "error")),
            repetitions=0,
            low_sample=False,
            worker_cpu=cpu_id,
            error=str(calibration.get("error", "calibration failed")),
        )
        append_jsonl(raw_path, terminal)
        return terminal

    calibration_seconds = float(calibration["seconds_per_e2e"])
    sampling_cap_reached = False

    if calibration_seconds <= 60.0:
        iterations = (
            min(
                MAX_INVOCATIONS_PER_SAMPLE,
                max(1, math.ceil(FAST_SAMPLE_TARGET_SECONDS / max(calibration_seconds, 1e-12))),
            )
            if calibration_seconds < FAST_SAMPLE_TARGET_SECONDS
            else 1
        )
        if iterations == 1 and len(included_rows) < 15:
            calibration["included_in_summary"] = True
            included_rows.append(calibration)
        append_jsonl(raw_path, calibration)
        cumulative = sum(float(row["wall_seconds"]) for row in included_rows)
        while (len(included_rows) < 7 or cumulative < 10.0) and len(included_rows) < 15:
            row = invoke(iterations, "measurement", True)
            append_jsonl(raw_path, row)
            if row.get("status") != "ok":
                terminal = _terminal_row(
                    run_id=run_id,
                    profile=profile,
                    engine=engine,
                    case=case,
                    shots=shots,
                    status=str(row.get("status", "error")),
                    repetitions=len(included_rows),
                    low_sample=False,
                    worker_cpu=cpu_id,
                    error=str(row.get("error", "measurement failed")),
                )
                append_jsonl(raw_path, terminal)
                return terminal
            included_rows.append(row)
            cumulative += float(row["wall_seconds"])
            # The single-invocation calibration can overestimate steady-state
            # cost for extremely small fixtures. Resize the next composed
            # sample from the just-observed per-invocation time so subsequent
            # records reach the one-second target without changing the metric.
            if float(row["wall_seconds"]) < FAST_SAMPLE_TARGET_SECONDS:
                iterations = min(
                    MAX_INVOCATIONS_PER_SAMPLE,
                    max(
                        1,
                        math.ceil(
                            FAST_SAMPLE_TARGET_SECONDS
                            / max(float(row["seconds_per_e2e"]), 1e-12)
                        ),
                    ),
                )
        sampling_cap_reached = len(included_rows) >= 15 and cumulative < 10.0
        low_sample = False
    elif calibration_seconds <= 300.0:
        calibration["included_in_summary"] = len(included_rows) < 3
        append_jsonl(raw_path, calibration)
        if calibration["included_in_summary"]:
            included_rows.append(calibration)
        for _ in range(max(0, 3 - len(included_rows))):
            row = invoke(1, "measurement", True)
            append_jsonl(raw_path, row)
            if row.get("status") != "ok":
                terminal = _terminal_row(
                    run_id=run_id,
                    profile=profile,
                    engine=engine,
                    case=case,
                    shots=shots,
                    status=str(row.get("status", "error")),
                    repetitions=len(included_rows),
                    low_sample=False,
                    worker_cpu=cpu_id,
                    error=str(row.get("error", "measurement failed")),
                )
                append_jsonl(raw_path, terminal)
                return terminal
            included_rows.append(row)
        low_sample = False
    else:
        calibration["included_in_summary"] = not included_rows
        append_jsonl(raw_path, calibration)
        if calibration["included_in_summary"]:
            included_rows.append(calibration)
        low_sample = True

    terminal = _terminal_row(
        run_id=run_id,
        profile=profile,
        engine=engine,
        case=case,
        shots=shots,
        status="ok",
        repetitions=len(included_rows),
        low_sample=low_sample,
        worker_cpu=cpu_id,
        sampling_cap_reached=sampling_cap_reached,
    )
    append_jsonl(raw_path, terminal)
    return terminal


def _dependency_hashes(root: Path) -> dict[str, str]:
    paths = [
        root.parent / "requirements.txt",
        root.parent / "requirements-lock.txt",
        root / "pyproject.toml",
        root / "julia" / "Project.toml",
        root / "julia" / "Manifest.toml",
    ]
    return {str(path.relative_to(root.parent)): sha256_file(path) for path in paths if path.exists()}


def _manifest_cases(cases: Iterable[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        item = {
            "case_id": case.metadata["case_id"],
            "metadata_path": str(case.metadata_path),
            "operations_sha256": case.metadata["operations_sha256"],
            "sidecar_sha256": case.metadata["sidecar_sha256"],
            "source_sha256": case.metadata["source_sha256"],
        }
        if "distance" in case.metadata:
            item["distance"] = case.metadata["distance"]
        if str(case.metadata.get("schema")) in {"fsbench-v2", "fsbench-v3"}:
            item["schema"] = case.metadata["schema"]
            item["family"] = case.metadata.get("family")
            item["parameters"] = case.metadata.get("parameters", {})
        output.append(item)
    return output


def _probe_fingerprint(probe: dict[str, Any]) -> dict[str, Any]:
    details = probe.get("details", {})
    return {
        "command": probe.get("command"),
        "status": probe.get("status"),
        "version_ok": probe.get("version_ok"),
        "version": details.get("version"),
        "aer_version": details.get("aer_version"),
        "julia_version": details.get("julia_version"),
        "native_module_sha256": details.get("native_module_sha256"),
        "installation_mode": details.get("installation_mode"),
        "source_repository": details.get("source_repository"),
        "source_commit": details.get("source_commit"),
        "source_requested_revision": details.get("source_requested_revision"),
        "direct_url": details.get("direct_url"),
        "reset_lowering": details.get("reset_lowering"),
        "installed_distributions": details.get("runtime", {}).get(
            "installed_distributions"
        ),
        "julia_dependency_lock": details.get("dependency_lock"),
    }


def _host_fingerprint(host: dict[str, Any]) -> dict[str, Any]:
    install_request = host.get("faultscope_install_request", {})
    return {
        "system": host.get("system"),
        "release": host.get("release"),
        "machine": host.get("machine"),
        "cpu_model": host.get("cpu_model"),
        "total_memory_bytes": host.get("total_memory_bytes"),
        "faultscope_repository": install_request.get("repository"),
        "faultscope_revision": install_request.get("revision"),
    }


def run_benchmark(
    *,
    profile_name: str,
    cases_dir: Path,
    results_dir: Path,
    engines: Iterable[str] | None = None,
    distances: Iterable[int] | None = None,
    widths: Iterable[int] | None = None,
    instances: Iterable[int] | None = None,
    shots_values: Iterable[int] | None = None,
    cpu_id: int | None = None,
    master_seed: int = MASTER_SEED,
    resume: bool = True,
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    if profile.publishable and (
        not sys_platform_linux() or platform.machine().lower() not in {"x86_64", "amd64"}
    ):
        raise RuntimeError(
            f"{profile.name} requires Linux x86_64"
        )
    selected_engines = tuple(engines or profile.engines)
    selected_shots = tuple(int(value) for value in (shots_values or profile.shots))
    unknown = set(selected_engines) - set(profile.engines)
    if unknown:
        raise ValueError(f"engines not in {profile.name}: {sorted(unknown)}")

    all_cases = [
        load_case_metadata(path, verify_hash=True) for path in case_metadata_paths(cases_dir)
    ]
    family_cases = [
        case for case in all_cases if str(case.metadata.get("family")) == profile.family
    ]
    if profile.family == "random_clifford":
        selected_widths = tuple(int(value) for value in (widths or profile.widths))
        selected_instances = tuple(int(value) for value in (instances or profile.instances))
        selected_distances: tuple[int, ...] = ()
        selected_depths = profile.depth_by_width(selected_widths)
        cases_by_parameters = {
            (
                int(case.metadata.get("parameters", {}).get("width", -1)),
                int(case.metadata.get("parameters", {}).get("depth", -1)),
                int(case.metadata.get("parameters", {}).get("instance", -1)),
            ): case
            for case in family_cases
        }
        missing_parameters = {
            (width, selected_depths[width], instance)
            for width in selected_widths
            for instance in selected_instances
        } - set(cases_by_parameters)
        if missing_parameters:
            raise ValueError(
                f"missing generated random-Clifford cases {sorted(missing_parameters)}"
            )
        selected_cases = [
            cases_by_parameters[(width, selected_depths[width], instance)]
            for width in selected_widths
            for instance in selected_instances
        ]
    else:
        selected_distances = tuple(int(value) for value in (distances or profile.distances))
        selected_widths = ()
        selected_instances = ()
        cases_by_distance = {
            int(case.metadata["distance"]): case for case in family_cases
        }
        missing = set(selected_distances) - set(cases_by_distance)
        if missing:
            raise ValueError(f"missing generated cases for distances {sorted(missing)}")
        selected_cases = [cases_by_distance[distance] for distance in selected_distances]

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "raw_runs.jsonl"
    manifest_path = results_dir / "run_manifest.json"
    if (raw_path.exists() or manifest_path.exists()) and not resume:
        raise FileExistsError(
            f"{raw_path} exists; a new paper run requires an empty results directory"
        )

    fixed_cpu = _select_cpu(cpu_id)
    worker_cpus = _worker_cpu_slots(
        base_cpu=fixed_cpu,
        requested=profile.parallel_workers,
        require_exact=False,
    )
    memory_limit = memory_cap_bytes()
    print(f"Physical worker CPUs ({len(worker_cpus)}): {list(worker_cpus)}", flush=True)
    probes = probe_engines(
        selected_engines, cpu_id=fixed_cpu, memory_limit_bytes=memory_limit
    )
    invalid_probes = {engine: probe.get("version_errors", probe.get("error")) for engine, probe in probes.items() if probe.get("status") != "ok" or probe.get("version_ok") is not True}
    if profile.publishable and invalid_probes:
        _atomic_json(results_dir / "engine_probe_failure.json", probes)
        raise RuntimeError(f"Engine preflight failed before the benchmark grid: {invalid_probes}")
    root = Path(__file__).resolve().parent.parent
    dependency_hashes = _dependency_hashes(root)
    from paper_runtime import source_sha256
    benchmark_hash = source_sha256()
    current_host = host_manifest()
    if profile.family == "random_clifford":
        selection = {
            "engines": list(selected_engines),
            "family": profile.family,
            "case_ids": [_metadata_case_id(case.metadata) for case in selected_cases],
            "widths": list(selected_widths),
            "depth_rule": profile.depth_rule,
            "depth_by_width": {
                str(width): selected_depths[width] for width in selected_widths
            },
            "instances": list(selected_instances),
            "shots": list(selected_shots),
        }
    else:
        selection = {
            "engines": list(selected_engines),
            "distances": list(selected_distances),
            "shots": list(selected_shots),
        }
    current_cases = _manifest_cases(selected_cases)
    fresh_manifest: dict[str, Any] = {
        "schema": (
            MANIFEST_SCHEMA_V2
            if profile.family == "random_clifford"
            else MANIFEST_SCHEMA
        ),
        "benchmark_schema": (
            "fsbench-v3" if any(c.metadata.get("schema") == "fsbench-v3" for c in selected_cases)
            else "fsbench-v2" if profile.family == "random_clifford" else "fsbench-v1"
        ),
        "run_id": uuid.uuid4().hex,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        # Normalize tuples to their JSON representation before comparing a
        # fresh profile with a manifest loaded during resume.
        "profile": json.loads(json.dumps(asdict(profile))),
        "selection": selection,
        "master_seed": int(master_seed),
        "resource_control": {
            "worker_cpu": fixed_cpu,
            "worker_cpus": list(worker_cpus),
            "configuration_parallelism": len(worker_cpus),
            "physical_cpu_candidates": _physical_cpu_candidates(sorted(os.sched_getaffinity(0)), fixed_cpu) if sys_platform_linux() else list(worker_cpus),
            "rss_poll_interval_seconds": 0.05,
            "parallel_scope": (
                "sequential-grid"
                if profile.parallel_workers == 1
                else "across-random-grid"
                if profile.family == "random_clifford"
                else "across-surface-grid"
            ),
            "timeout_seconds": profile.timeout_seconds,
            "memory_limit_bytes": memory_limit,
            "max_batch_shots": MAX_BATCH_SHOTS,
            "fresh_worker_per_sample": True,
            "page_cache_prefetch_after_ready": True,
            "thread_environment": {
                key: worker_environment()[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OMP_DYNAMIC",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "MKL_DYNAMIC",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "RAYON_NUM_THREADS",
                    "JULIA_NUM_THREADS",
                    "QISKIT_PARALLEL",
                    "PYTHONHASHSEED",
                )
            },
        },
        "host": current_host,
        "engine_probes": probes,
        "dependency_file_sha256": dependency_hashes,
        "benchmark_source_sha256": benchmark_hash,
        "cases": current_cases,
        "configuration_order": [],
        "raw_runs": str(raw_path),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_errors: list[str] = []
        if manifest.get("schema") != fresh_manifest["schema"]:
            resume_errors.append("manifest schema changed")
        if manifest.get("profile") != fresh_manifest["profile"]:
            resume_errors.append("profile fingerprint changed")
        if manifest.get("selection") != selection:
            resume_errors.append("grid selection changed")
        if int(manifest.get("master_seed", -1)) != int(master_seed):
            resume_errors.append("master seed changed")
        if manifest.get("benchmark_source_sha256") != benchmark_hash:
            resume_errors.append("benchmark source changed")
        if manifest.get("dependency_file_sha256") != dependency_hashes:
            resume_errors.append("dependency lock files changed")
        if manifest.get("cases") != current_cases:
            resume_errors.append("generated case files changed")
        if _host_fingerprint(manifest.get("host", {})) != _host_fingerprint(current_host):
            resume_errors.append("host or FaultScope install request changed")
        old_resources = manifest.get("resource_control", {})
        if old_resources.get("worker_cpu") != fixed_cpu:
            resume_errors.append("worker CPU changed")
        if old_resources.get("worker_cpus") != list(worker_cpus):
            resume_errors.append("worker CPU set changed")
        if int(old_resources.get("configuration_parallelism", -1)) != len(
            worker_cpus
        ):
            resume_errors.append("configuration parallelism changed")
        if old_resources.get("parallel_scope") != fresh_manifest[
            "resource_control"
        ]["parallel_scope"]:
            resume_errors.append("parallel scope changed")
        if float(old_resources.get("timeout_seconds", -1)) != profile.timeout_seconds:
            resume_errors.append("timeout changed")
        if int(old_resources.get("memory_limit_bytes", -1)) != int(memory_limit):
            resume_errors.append("memory limit changed")
        old_probes = manifest.get("engine_probes", {})
        for engine in selected_engines:
            if _probe_fingerprint(old_probes.get(engine, {})) != _probe_fingerprint(probes[engine]):
                resume_errors.append(f"{engine} worker environment changed")
        if resume_errors:
            raise RuntimeError(
                "cannot resume a heterogeneous run: " + "; ".join(resume_errors)
            )
        manifest.setdefault("resume_events", []).append(
            {"timestamp_utc": utc_now(), "host": current_host, "engine_probes": probes}
        )
    else:
        if raw_path.exists():
            raise RuntimeError(f"{raw_path} exists without {manifest_path}")
        manifest = fresh_manifest
    run_id = str(manifest["run_id"])
    _atomic_json(manifest_path, manifest)

    existing_raw_rows = read_jsonl(raw_path)
    completed = {
        (str(row["engine"]), _row_case_id(row), int(row["shots"]))
        for row in existing_raw_rows
        if row.get("record_type") == "config_status"
    }
    partial_samples: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in existing_raw_rows:
        if row.get("record_type") != "sample":
            continue
        key = (str(row["engine"]), _row_case_id(row), int(row["shots"]))
        partial_samples.setdefault(key, []).append(row)
    recorded_orders = {
        (
            int(item.get("width", item.get("distance", -1))),
            int(item["shots"]),
        )
        for item in manifest.get("configuration_order", [])
    }

    if profile.family == "random_clifford":
        cases_by_width = {
            width: [
                case
                for case in selected_cases
                if int(case.metadata["parameters"]["width"]) == width
            ]
            for width in selected_widths
        }
        blocks = [(width, cases_by_width[width]) for width in selected_widths]
    else:
        blocks = [
            (distance, [case])
            for distance, case in zip(selected_distances, selected_cases, strict=True)
        ]

    pending: list[tuple[Any, str, int, tuple[str, str, int]]] = []
    grid_changed = False
    for scale_value, block_cases in blocks:
        for shots in selected_shots:
            configurations = [
                (case, engine) for case in block_cases for engine in selected_engines
            ]
            order_ordinal = (
                _stable_ordinal("order", profile.family, scale_value, shots)
                if profile.family == "random_clifford"
                else _stable_ordinal("order", scale_value, shots)
            )
            random.Random(derive_seed(master_seed, order_ordinal)).shuffle(configurations)
            if (scale_value, shots) not in recorded_orders:
                if profile.family == "random_clifford":
                    order_item = {
                        "width": scale_value,
                        "shots": shots,
                        "configurations": [
                            {
                                "case_id": _metadata_case_id(case.metadata),
                                "engine": engine,
                            }
                            for case, engine in configurations
                        ],
                    }
                else:
                    order_item = {
                        "distance": scale_value,
                        "shots": shots,
                        "engines": [engine for _case, engine in configurations],
                    }
                manifest["configuration_order"].append(order_item)
                recorded_orders.add((scale_value, shots))
                _atomic_json(manifest_path, manifest)
            for case, engine in configurations:
                case_id = _metadata_case_id(case.metadata)
                key = (engine, case_id, shots)
                if key in completed:
                    continue
                probe = probes[engine]
                usable = probe.get("status") == "ok" and probe.get("version_ok") is True
                if not usable:
                    error = "; ".join(probe.get("version_errors", [])) or str(
                        probe.get("details", {}).get("reason", "engine unavailable")
                    )
                    terminal = _terminal_row(
                        run_id=run_id,
                        profile=profile,
                        engine=engine,
                        case=case,
                        shots=shots,
                        status="unsupported",
                        repetitions=0,
                        low_sample=False,
                        error=error,
                    )
                    append_jsonl(raw_path, terminal)
                    completed.add(key)
                    grid_changed = True
                else:
                    pending.append((case, engine, shots, key))

    if pending:
        pending_iterator = iter(enumerate(pending))

        def execute_configuration(
            case: Any,
            engine: str,
            configuration_shots: int,
            key: tuple[str, str, int],
            worker_cpu: int | None,
        ) -> tuple[str, str, int]:
            terminal = _run_configuration(
                raw_path=raw_path,
                run_id=run_id,
                profile=profile,
                engine=engine,
                probe=probes[engine],
                case=case,
                shots=configuration_shots,
                timeout_seconds=profile.timeout_seconds,
                cpu_id=worker_cpu,
                memory_limit_bytes=memory_limit,
                master_seed=master_seed,
                existing_samples=partial_samples.get(key, ()),
            )
            print(f"{engine} {case.metadata['case_id']}: {terminal['status']} ({terminal['repetitions']} timing samples)", flush=True)
            return key

        futures: dict[Future[tuple[str, str, int]], tuple[int | None, int]] = {}
        with ThreadPoolExecutor(
            max_workers=min(len(worker_cpus), len(pending)),
            thread_name_prefix="fsbench-config",
        ) as executor:

            def submit_next(worker_cpu: int | None) -> bool:
                try:
                    ordinal, (case, engine, configuration_shots, key) = next(
                        pending_iterator
                    )
                except StopIteration:
                    return False
                future = executor.submit(
                    execute_configuration,
                    case,
                    engine,
                    configuration_shots,
                    key,
                    worker_cpu,
                )
                futures[future] = (worker_cpu, ordinal)
                return True

            for worker_cpu in worker_cpus:
                if not submit_next(worker_cpu):
                    break
            while futures:
                finished, _unfinished = wait(futures, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda item: futures[item][1]):
                    worker_cpu, _ordinal = futures.pop(future)
                    completed.add(future.result())
                    submit_next(worker_cpu)
        grid_changed = True

    if grid_changed:
        summarize_results(results_dir)
        manifest["updated_utc"] = utc_now()
        _atomic_json(manifest_path, manifest)

    summary_rows, status_rows = summarize_results(results_dir)
    manifest["updated_utc"] = utc_now()
    manifest["terminal_counts"] = {
        status: sum(row["status"] == status for row in status_rows)
        for status in ("ok", "timeout", "oom", "unsupported", "error")
    }
    manifest["complete_grid"] = len(status_rows) == (
        len(selected_engines) * len(selected_cases) * len(selected_shots)
    )
    manifest["summary_rows"] = len(summary_rows)
    _atomic_json(manifest_path, manifest)
    return manifest
