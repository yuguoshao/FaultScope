"""Local paths, fixed dependencies, and lifecycle for the four paper scripts.

Only the public entry points call run_entry. Private helpers are also usable by
small tests without launching the full paper workloads.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "_support"
RESULTS = ROOT / "results"
FAULTSCOPE_COMMIT = "90fa318ff9c95a5fa413f3b952c3970103e80d4c"
VERSION_PINS = {
    "faultscope": "0.2.11", "stim": "1.16.0", "qiskit": "2.5.0",
    "qiskit-aer": "0.17.2", "cirq-core": "1.6.1", "symft": "0.1.1",
    "numpy": "2.5.1", "psutil": "7.0.0", "matplotlib": "3.11.0", "pillow": "11.3.0",
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_DYNAMIC": "FALSE",
    "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1", "QISKIT_PARALLEL": "FALSE", "PYTHONHASHSEED": "0",
}
_ENGINE_DISTRIBUTIONS = {
    "faultscope": ("faultscope",), "stim": ("stim",),
    "qiskit-aer": ("qiskit", "qiskit-aer"), "cirq": ("cirq-core",),
    "symft": ("symft",), "quantumclifford": (),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_sha256() -> str:
    """Fingerprint distributable code/config, independently of its location."""
    digest = hashlib.sha256()
    paths = []
    for directory, directories, files in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if not d.startswith(".") and d not in {"results", "runtime", "__pycache__", "tests"})
        paths.extend(Path(directory) / name for name in files)
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        if not path.name.startswith(".") and path.suffix in {".py", ".jl", ".json", ".toml", ".txt", ".sh"}:
            digest.update(rel.as_posix().encode() + b"\0")
            digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def physical_cpus(limit: int = 64) -> tuple[int, ...]:
    """Choose one allowed logical CPU per physical core, never assuming CPU 0."""
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("The full paper benchmarks require Linux CPU affinity.")
    cores: dict[tuple[int, int], int] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot identify physical topology for allowed CPU {cpu}") from exc
        cores.setdefault(key, cpu)
    selected = tuple(sorted(cores.values())[:limit])
    if not selected:
        raise RuntimeError("No available physical CPU in the current affinity set")
    return selected


def pin_to_cpu(cpu: int) -> None:
    if cpu not in os.sched_getaffinity(0):
        raise RuntimeError(f"CPU {cpu} is outside the current affinity set")
    os.sched_setaffinity(0, {cpu})


def thread_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # No inherited suite/interpreter overrides may change the fixed experiment.
    for key in list(environment):
        if key.startswith("FSBENCH_"):
            del environment[key]
    environment.update(THREAD_ENV)
    environment.update({
        "PYTHONPATH": str(SUPPORT), "PYTHONNOUSERSITE": "1",
        "JULIA_DEPOT_PATH": str(ROOT / "runtime" / "julia-depot"),
        "MPLBACKEND": "Agg",
    })
    return environment


def julia_executable() -> Path:
    local = ROOT / "runtime" / "julia-1.12.6" / "bin" / "julia"
    candidate = str(local) if local.is_file() else shutil.which("julia")
    if not candidate:
        raise RuntimeError("Julia 1.12.6 is missing. Run bash setup.sh first.")
    check = subprocess.run([candidate, "--startup-file=no", "--version"],
                           capture_output=True, text=True, timeout=30, check=True)
    if check.stdout.strip() != "julia version 1.12.6":
        raise RuntimeError(f"Expected Julia 1.12.6, found {check.stdout.strip()!r}. Run bash setup.sh.")
    return Path(candidate).resolve()


def _faultscope_identity() -> dict:
    distribution = metadata.distribution("faultscope")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    native = importlib.import_module("faultscope._native")
    path = Path(native.__file__).resolve()
    if commit != FAULTSCOPE_COMMIT:
        raise RuntimeError(f"FaultScope must be installed from commit {FAULTSCOPE_COMMIT}; run bash setup.sh.")
    return {"version": distribution.version, "commit": commit,
            "native_module": str(path), "native_module_sha256": file_sha256(path)}


def require_dependencies(engines: tuple[str, ...] | list[str]) -> None:
    errors = []
    pins = dict(VERSION_PINS)
    for line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.partition("#")[0].strip()
        if "==" in requirement:
            name, version = requirement.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = version.strip()
    packages = set(pins)
    for engine in engines:
        packages.update(_ENGINE_DISTRIBUTIONS[engine])
    for name in sorted(packages):
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != pins[name]:
            errors.append(f"{name}: expected {pins[name]}, found {actual}")
    if "faultscope" in engines:
        try:
            _faultscope_identity()
        except (RuntimeError, ImportError, metadata.PackageNotFoundError) as exc:
            errors.append(str(exc))
    if "quantumclifford" in engines:
        try:
            julia_executable()
        except (RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("Dependency preflight failed:\n  " + "\n  ".join(errors) + "\nRun bash setup.sh.")


def runtime_manifest() -> dict:
    from fsbench.host import host_manifest

    versions = {}
    for name in VERSION_PINS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        identity = _faultscope_identity() if versions["faultscope"] else None
    except (RuntimeError, ImportError, OSError, ValueError) as exc:
        identity = {"error": str(exc)}
    return {
        "hostname": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(), "python": sys.version,
        "executable": sys.executable, "versions": versions,
        "faultscope": identity, "thread_environment": dict(THREAD_ENV),
        "allowed_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "host": host_manifest(),
        "installed_distributions": dict(sorted(
            (dist.metadata["Name"], dist.version) for dist in metadata.distributions()
            if dist.metadata["Name"]
        )),
        "benchmark_source_sha256": source_sha256(),
    }


def finish_run(run_dir: Path, manifest: dict) -> None:
    """Commit a completed run's metadata after writing its experiment artifacts."""
    result = dict(manifest)
    result.update({"complete": True, "completed_at": datetime.now(timezone.utc).isoformat(),
                   "paper_source_sha256": source_sha256()})
    result.setdefault("benchmark_source_sha256", source_sha256())
    result.setdefault("run_id", Path(run_dir).name)
    result["output_sha256"] = {
        path.relative_to(run_dir).as_posix(): file_sha256(path)
        for path in sorted(Path(run_dir).rglob("*"))
        if path.is_file() and path.name not in {"run_manifest.json", "run.log", "run_status.json"}
        and not path.name.endswith(".tmp")
    }
    write_json(Path(run_dir) / "run_manifest.json", result)


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def run_entry(name: str, callback: Callable[[Path], None]) -> None:
    if len(sys.argv) != 1:
        raise SystemExit("This paper benchmark has fixed settings and accepts no arguments.")
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("Full paper benchmarks require Linux x86_64. No experiment was started.")
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"Benchmark environment is missing. Run: bash {ROOT / 'setup.sh'}")
    if (Path(sys.prefix).resolve() != (ROOT / ".venv").resolve()
            or any(os.environ.get(key) != value for key, value in THREAD_ENV.items())
            or os.environ.get("PYTHONNOUSERSITE") != "1"):
        os.execve(str(python), [str(python), str(Path(sys.argv[0]).resolve())], thread_environment())
    if sys.version_info[:2] != (3, 12) or platform.python_implementation() != "CPython":
        raise SystemExit("The benchmark environment must use CPython 3.12; rerun setup.sh.")
    os.environ.update(thread_environment())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = RESULTS / name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run_status.json", {"status": "running", "entry": name})
    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                print(f"{name}: {run_dir}", flush=True)
                try:
                    # Preserve host/software evidence even when a later trial
                    # fails and the callback cannot finish its run manifest.
                    write_json(run_dir / "environment.json", runtime_manifest())
                    callback(run_dir)
                    manifest = json.loads((run_dir / "run_manifest.json").read_text())
                    if manifest.get("complete") is not True:
                        raise RuntimeError("Benchmark did not finish its run manifest")
                except BaseException as exc:
                    write_json(run_dir / "run_status.json", {"status": "failed", "error": str(exc)})
                    traceback.print_exc()
                    raise
                write_json(run_dir / "run_status.json", {"status": "complete", "entry": name})
                print(f"Completed: {run_dir}", flush=True)
    finally:
        # Write after the log is closed so its checksum includes the final
        # status line or traceback. This index deliberately excludes itself.
        write_json(run_dir / "file_hashes.json", {
            path.relative_to(run_dir).as_posix(): file_sha256(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "file_hashes.json"
            and not path.name.endswith(".tmp")
        })
