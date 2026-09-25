"""Host, source, and environment provenance capture."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def total_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:  # pragma: no cover
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)


def memory_cap_bytes() -> int:
    return 36 * 1024**3


def _cpu_model() -> str:
    if sys.platform == "darwin":
        value = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _linux_cpu_controls() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        return {"governor": None, "turbo": None, "thread_siblings": None}
    governors = sorted(
        {
            value
            for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
            if (value := _read_optional(path))
        }
    )
    no_turbo = _read_optional(Path("/sys/devices/system/cpu/intel_pstate/no_turbo"))
    boost = _read_optional(Path("/sys/devices/system/cpu/cpufreq/boost"))
    turbo: str | None
    if no_turbo is not None:
        turbo = "disabled" if no_turbo == "1" else "enabled"
    elif boost is not None:
        turbo = "enabled" if boost == "1" else "disabled"
    else:
        turbo = None
    siblings = {
        path.parts[-3]: value
        for path in Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/topology/thread_siblings_list"
        )
        if (value := _read_optional(path))
    }
    return {
        "governor": governors or None,
        "turbo": turbo,
        "thread_siblings": siblings or None,
    }


def _git_root(path: Path) -> Path | None:
    output = _command_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    return Path(output) if output else None


def git_identity(path: Path, *, scope: Path | None = None) -> dict[str, Any]:
    root = _git_root(path)
    if root is None:
        return {"available": False, "path": str(path.resolve())}
    status_command = ["git", "-C", str(root), "status", "--porcelain"]
    if scope is not None:
        try:
            relative_scope = scope.resolve().relative_to(root.resolve())
            status_command.extend(["--", str(relative_scope)])
        except ValueError:
            pass
    dirty_paths = (_command_output(status_command) or "").splitlines()
    return {
        "available": True,
        "root": str(root),
        "commit": _command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "branch": _command_output(["git", "-C", str(root), "branch", "--show-current"]),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def source_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    generated_directories = {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "deployment-backups",
        "dist",
        "logs",
        "sources",
        "target",
    }
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and not generated_directories.intersection(item.parts)
        and not any(part.endswith(".egg-info") for part in item.parts)
        and item.suffix not in {".pyc", ".png", ".pdf", ".so", ".dylib", ".pyd", ".whl"}
        and item.relative_to(path).as_posix() != "julia/Manifest.toml"
        and "results" not in item.parts
        and "cases" not in item.parts
    )
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def host_manifest() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = None
    from paper_runtime import FAULTSCOPE_COMMIT
    from .engines import worker_environment
    return {
        "hostname": platform.node(),
        "machine_id": _read_optional(Path("/etc/machine-id")),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_controls": _linux_cpu_controls(),
        "total_memory_bytes": total_memory_bytes(),
        "memory_cap_bytes": memory_cap_bytes(),
        "python": sys.version,
        "python_executable": sys.executable,
        "thread_environment": {
            key: worker_environment().get(key)
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
            )
        },
        "benchmark_source": git_identity(PROJECT_ROOT, scope=PROJECT_ROOT),
        "benchmark_source_sha256": source_tree_sha256(PROJECT_ROOT),
        "faultscope_install_request": {
            "revision": FAULTSCOPE_COMMIT,
        },
        "compiler": {
            "cc": _command_output([os.environ.get("CC", "cc"), "--version"]),
            "rustc": _command_output(["rustc", "--version"]),
            "cargo": _command_output(["cargo", "--version"]),
        },
    }
