"""Engine registry and reproducible worker command construction."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

from .constants import ENGINE_LABELS, PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class EngineSpec:
    key: str
    label: str
    command: tuple[str, ...]
    expected_versions: dict[str, str]


_PYTHON_MODULES = {
    "faultscope": "fsbench.workers.faultscope_worker",
    "stim": "fsbench.workers.stim_worker",
    "qiskit-aer": "fsbench.workers.qiskit_worker",
    "cirq": "fsbench.workers.cirq_worker",
    "symft": "fsbench.workers.symft_worker",
}


def engine_spec(engine: str) -> EngineSpec:
    if engine not in ENGINE_LABELS:
        raise KeyError(f"unknown engine {engine!r}")
    if engine in _PYTHON_MODULES:
        command = (sys.executable, "-m", _PYTHON_MODULES[engine])
    elif engine == "quantumclifford":
        bundled = PROJECT_ROOT.parent / "runtime" / "julia-1.12.6" / "bin" / "julia"
        julia = str(bundled) if bundled.is_file() else (shutil.which("julia") or "julia")
        command = (
            julia,
            f"--project={PROJECT_ROOT / 'julia'}",
            "--threads=1",
            "--startup-file=no",
            "--history-file=no",
            str(PROJECT_ROOT / "fsbench" / "workers" / "quantumclifford_worker.jl"),
        )
    else:  # pragma: no cover - registry is exhaustive
        raise KeyError(engine)

    expected = {
        "faultscope": {"version": "0.2.11"},
        "stim": {"version": "1.16.0"},
        "qiskit-aer": {"version": "2.5.0", "aer_version": "0.17.2"},
        "cirq": {"version": "1.6.1"},
        "symft": {"version": "0.1.1"},
        "quantumclifford": {"version": "0.11.5", "julia_version": "1.12.6"},
    }[engine]
    return EngineSpec(
        key=engine,
        label=ENGINE_LABELS[engine],
        command=tuple(str(item) for item in command),
        expected_versions=expected,
    )


def worker_environment() -> dict[str, str]:
    from paper_runtime import thread_environment
    return thread_environment()


def version_matches(spec: EngineSpec, probe: dict[str, object]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field, expected in spec.expected_versions.items():
        if field == "julia_version_prefix":
            actual = str(probe.get("julia_version", ""))
            if not actual.startswith(expected):
                errors.append(f"julia_version={actual!r}, expected prefix {expected!r}")
        elif str(probe.get(field, "")) != expected:
            errors.append(f"{field}={probe.get(field)!r}, expected {expected!r}")
    return not errors, errors
