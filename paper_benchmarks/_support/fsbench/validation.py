"""Validate the fixed paper grid, raw measurements, and execution provenance."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import ALL_ENGINES, DEFAULT_DISTANCES, MASTER_SEED, RANDOM_CLIFFORD_WIDTHS
from .engines import engine_spec, version_matches
from .ir import sha256_file
from .protocol import derive_seed
from .runner import _parse_cpu_list, _stable_ordinal
from .summary import aggregate_rows, read_jsonl

ALLOWED_TERMINAL_STATUSES = {"ok", "timeout", "oom", "unsupported", "error"}
INPUT_FILES = ("raw_runs.jsonl", "summary.csv", "status.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def expected_case_ids(family: str) -> tuple[str, ...]:
    if family == "random_clifford":
        return tuple(f"random_clifford_n{n}_d64_i{i}" for n in RANDOM_CLIFFORD_WIDTHS for i in (0, 1, 2))
    if family == "surface_code:rotated_memory_z":
        return tuple(f"rotated_memory_z_d{d}_r{d}_p0.001" for d in DEFAULT_DISTANCES)
    raise ValueError(f"unknown paper family {family!r}")


def resource_errors(manifest: dict[str, Any]) -> list[str]:
    resources, host = manifest.get("resource_control", {}), manifest.get("host", {})
    errors: list[str] = []
    cpus = resources.get("worker_cpus", [])
    allowed = host.get("cpu_affinity") or []
    sibling_map = host.get("cpu_controls", {}).get("thread_siblings") or {}
    physical: list[int] = []
    seen: set[frozenset[int]] = set()
    for cpu in allowed:
        siblings = _parse_cpu_list(sibling_map.get(f"cpu{cpu}", str(cpu)))
        group = frozenset(siblings.intersection(allowed)) or frozenset({cpu})
        if group not in seen:
            seen.add(group)
            physical.append(cpu)
    if not cpus or cpus != physical[:64] or len(cpus) != len(set(cpus)):
        errors.append("worker CPUs must be the first min(64, allowed physical cores), one thread per core")
    if resources.get("worker_cpu") != (cpus[0] if cpus else None) or resources.get("configuration_parallelism") != len(cpus):
        errors.append("base CPU and parallelism must match the actual worker CPU set")
    if host.get("system") != "Linux" or str(host.get("machine", "")).lower() not in {"x86_64", "amd64"}:
        errors.append("paper execution requires Linux x86_64")
    for field, value in {"max_batch_shots": 1000, "memory_limit_bytes": 36 * 1024**3, "timeout_seconds": 3600.0, "rss_poll_interval_seconds": 0.05}.items():
        if resources.get(field) != value:
            errors.append(f"resource policy differs: {field}")
    env = resources.get("thread_environment", {})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "JULIA_NUM_THREADS"):
        if env.get(name) != "1":
            errors.append(f"worker thread limit is not one: {name}")
    for name in ("OMP_DYNAMIC", "MKL_DYNAMIC", "QISKIT_PARALLEL"):
        if env.get(name) != "FALSE":
            errors.append(f"worker dynamic parallelism must be disabled: {name}")
    if env.get("PYTHONHASHSEED") != "0":
        errors.append("Python hash seed must be zero")
    return errors


def software_errors(manifest: dict[str, Any]) -> list[str]:
    from paper_runtime import FAULTSCOPE_COMMIT
    probes = manifest.get("engine_probes", {})
    errors: list[str] = []
    if set(probes) != set(ALL_ENGINES):
        errors.append("exactly six engine probes are required")
    for engine in ALL_ENGINES:
        probe = probes.get(engine, {})
        valid, details = version_matches(engine_spec(engine), probe.get("details", {}))
        if not valid or probe.get("status") != "ok" or probe.get("version_ok") is not True:
            errors.append(f"{engine} probe/version mismatch: {details}")
        if engine != "quantumclifford":
            runtime = probe.get("details", {}).get("runtime", {})
            if not str(runtime.get("python", "")).startswith("3.12."):
                errors.append(f"{engine} worker must use Python 3.12")
            if runtime.get("python_executable") != manifest.get("host", {}).get("python_executable"):
                errors.append(f"{engine} worker uses a different Python interpreter")
    fs = probes.get("faultscope", {}).get("details", {})
    if fs.get("source_commit") != FAULTSCOPE_COMMIT or len(str(fs.get("native_module_sha256", ""))) != 64:
        errors.append("FaultScope must record the pinned commit and native binary hash")
    qc = probes.get("quantumclifford", {}).get("details", {})
    if len(str(qc.get("dependency_lock", {}).get("manifest_sha256", ""))) != 64:
        errors.append("QuantumClifford dependency lock is missing")
    return errors


def _csv_equal(expected: list[dict[str, Any]], actual: list[dict[str, str]]) -> bool:
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual, strict=True):
        if set(left) != set(right):
            return False
        for field, value in left.items():
            if isinstance(value, float):
                try:
                    if not math.isclose(value, float(right[field]), rel_tol=1e-12, abs_tol=1e-12):
                        return False
                except ValueError:
                    return False
            elif ("" if value is None else str(value)) != right[field]:
                return False
    return True


def validate_paper_run(results_dir: Path, *, check_runtime: bool = True) -> dict[str, Any]:
    """Check terminal completeness without treating censored points as timings."""
    from paper_inputs import HASH_FIELDS, expected_case_hashes
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    selection = manifest["selection"]
    family = manifest["profile"]["family"]
    case_ids = expected_case_ids(family)
    expected_hashes = expected_case_hashes()
    errors: list[str] = []
    if tuple(selection.get("engines", ())) != ALL_ENGINES or selection.get("shots") != [1_000_000] or manifest.get("master_seed") != MASTER_SEED:
        errors.append("engine grid, shot count, or master seed differs from the paper protocol")
    if family == "random_clifford":
        if tuple(selection.get("widths", ())) != RANDOM_CLIFFORD_WIDTHS or selection.get("instances") != [0, 1, 2] or selection.get("depth_rule") != "fixed" or selection.get("depth_by_width") != {str(n): 64 for n in RANDOM_CLIFFORD_WIDTHS}:
            errors.append("random grid must contain all seven widths, three instances, and depth 64")
    elif tuple(selection.get("distances", ())) != DEFAULT_DISTANCES:
        errors.append("surface grid must contain all twelve distances")
    cases = manifest.get("cases", [])
    if Counter(c.get("case_id") for c in cases) != Counter(case_ids):
        errors.append("manifest case grid is incomplete or duplicated")
    for case in cases:
        if any(case.get(field) != expected_hashes.get(case.get("case_id"), {}).get(field) for field in HASH_FIELDS):
            errors.append(f"paper input identity differs: {case.get('case_id')}")
    expected_grid = {(engine, case_id, 1_000_000) for engine in ALL_ENGINES for case_id in case_ids}
    raw = read_jsonl(results_dir / "raw_runs.jsonl")
    terminals = [row for row in raw if row.get("record_type") == "config_status"]
    key = lambda row: (row.get("engine"), row.get("case_id"), row.get("shots"))
    if Counter(key(row) for row in terminals) != Counter(expected_grid):
        errors.append("raw data must have exactly one terminal row for each fixed grid point")
    if not manifest.get("complete_grid"):
        errors.append("manifest does not mark the grid complete")
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    sample_keys: set[tuple] = set()
    calibrations: dict[tuple, float] = {}
    for row in raw:
        if row.get("record_type") not in {"sample", "config_status"}:
            errors.append("unknown raw record type")
        if key(row) not in expected_grid or row.get("run_id") != manifest["run_id"]:
            errors.append("raw row has an unknown case, engine, shots, or run identity")
        if row.get("status") not in ALLOWED_TERMINAL_STATUSES:
            errors.append("unknown terminal/sample status")
        if check_runtime and row.get("status") != "unsupported" and row.get("worker_cpu") not in manifest["resource_control"]["worker_cpus"]:
            errors.append("raw row used a CPU outside the recorded worker set")
        if row.get("record_type") != "sample":
            continue
        identity = (*key(row), row.get("sample_index"))
        if identity in sample_keys:
            errors.append("duplicate sample identity")
        sample_keys.add(identity)
        expected_seed = derive_seed(derive_seed(MASTER_SEED, _stable_ordinal(row["engine"], row["case_id"], row["shots"])), row["sample_index"])
        if row.get("seed") != expected_seed or row.get("case_operations_sha256") != expected_hashes.get(row["case_id"], {}).get("operations_sha256"):
            errors.append("sample seed or operation hash differs")
        if row.get("status") != "ok":
            continue
        case_id = row["case_id"]
        if family == "random_clifford":
            width = int(case_id.split("_n", 1)[1].split("_", 1)[0])
            if (row.get("width"), row.get("depth"), row.get("instance")) != (width, 64, int(case_id[-1])):
                errors.append("random sample size/depth/instance differs from its case identity")
        else:
            distance = int(case_id.split("_d", 1)[1].split("_", 1)[0])
            if (row.get("distance"), row.get("rounds")) != (distance, distance):
                errors.append("surface sample distance/rounds differs from its case identity")
        if row.get("phase") == "calibration":
            calibrations[key(row)] = float(row["seconds_per_e2e"])
        iterations = int(row.get("iterations", 0))
        if check_runtime and row["engine"] == "quantumclifford":
            details = row.get("details", [])
            if len(details) != iterations or any(
                invocation.get("reference_evaluations") != 1
                or invocation.get("reference_reused_across_batches") is not True
                or invocation.get("reference_seed_policy") != "sha256-invocation"
                for invocation in details
            ):
                errors.append("QuantumClifford must construct one noiseless reference per timed invocation and reuse it across batches")
        wall = float(row.get("wall_seconds", -1))
        seconds = float(row.get("seconds_per_e2e", -1))
        if iterations <= 0 or not math.isfinite(wall) or wall <= 0 or not math.isfinite(seconds) or seconds <= 0 or not math.isclose(wall, row.get("wall_time_ns", 0) / 1e9, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(seconds * iterations, wall, rel_tol=1e-12, abs_tol=1e-12):
            errors.append("sample timing units or iteration count are invalid")
        if row.get("shots_completed") != 1_000_000 * iterations or row.get("max_batch_shots") != 1000 or row.get("records_b64") is not None:
            errors.append("sample shot/batch/count-only contract differs")
        if row.get("included_in_summary") is True:
            grouped[key(row)].append(row)
    for row in terminals:
        samples = grouped[key(row)]
        if row.get("status") == "ok":
            n, cumulative = len(samples), sum(s["wall_seconds"] for s in samples)
            calibration = calibrations.get(key(row), -1)
            valid = (
                (calibration > 300 and n == 1 and row.get("low_sample") is True)
                or (60 < calibration <= 300 and n == 3 and not row.get("low_sample"))
                or (0 < calibration <= 60 and 7 <= n <= 15 and not row.get("low_sample")
                    and (cumulative >= 10 or (n == 15 and row.get("sampling_cap_reached"))))
            )
            if row.get("repetitions") != n or not valid:
                errors.append("successful grid point violates the adaptive repetition policy")
    summaries, statuses = aggregate_rows(raw)
    if not _csv_equal(summaries, read_csv(results_dir / "summary.csv")) or not _csv_equal(statuses, read_csv(results_dir / "status.csv")):
        errors.append("summary/status CSV does not reproduce from the raw measurements")
    if check_runtime:
        errors.extend(resource_errors(manifest))
        errors.extend(software_errors(manifest))
    return {"schema": "paper-run-validation-v1", "passed": not errors, "errors": sorted(set(errors)), "expected_grid_points": len(expected_grid), "terminal_counts": dict(Counter(row["status"] for row in terminals)), "validated_inputs_sha256": {name: sha256_file(results_dir / name) for name in INPUT_FILES}}
