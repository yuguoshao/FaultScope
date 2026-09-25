"""Fixed Table 3 sampling throughput for FaultScope and Stim.

Each measurement uses the installed paper environment through a fresh,
CPU-pinned worker process.
The sampling clock surrounds only public sampler calls; record validation and
checksumming happen outside that clock but inside the end-to-end clock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engines import EngineSpec
from .ir import load_case, load_case_metadata, sha256_file
from .protocol import PROTOCOL_VERSION, batch_ranges, derive_seed, write_message


ENGINES = ("faultscope", "stim")
BATCH_CHOICES = (1_000, 10_000, 100_000, 1_000_000)
RANDOM_WIDTHS = (16, 128, 1024)
SURFACE_DISTANCES = (3, 10, 20)
SHOTS = 1_000_000
PILOT_REPEATS = 2
FORMAL_REPEATS = 6
MEMORY_LIMIT_BYTES = 36 * 1024**3
TIMEOUT_SECONDS = 3_600.0
EXPECTED_VERSIONS = {"faultscope": "0.2.11", "stim": "1.16.0"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def _stim_text(case: Any) -> str:
    # Keep gate and noise lowering identical to the shared Stim worker.
    lines: list[str] = []
    for operation in case.operations:
        if operation.probability is not None:
            targets = " ".join(str(q) for q in operation.targets)
            lines.append(f"{operation.name}({operation.probability!r}) {targets}")
        elif operation.name == "CX":
            lines.append(f"CX {operation.targets[0]} {operation.targets[1]}")
        else:
            lines.append(f"{operation.name} {operation.targets[0]}")
    return "\n".join(lines)


def _worker_for(engine: str) -> Any:
    if engine == "faultscope":
        from .workers.faultscope_worker import FaultScopeWorker

        return FaultScopeWorker()
    if engine == "stim":
        from .workers.stim_worker import StimWorker

        return StimWorker()
    raise ValueError(f"unknown throughput engine {engine!r}")


def _measure(worker: Any, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    shots = int(request["shots"])
    batch_shots = int(request["batch_shots"])
    seed = int(request["seed"])
    symft_mode = str(request.get("symft_mode", "none"))
    if shots <= 0 or batch_shots not in BATCH_CHOICES:
        raise ValueError("shots must be positive and batch_shots must be a declared candidate")
    if worker.engine_key not in ENGINES or symft_mode != "none":
        raise ValueError("Table 3 measures only FaultScope and Stim")

    started = time.perf_counter()
    case = load_case(Path(request["case_path"]))
    measurements = int(case.metadata["num_measurements"])
    if worker.engine_key == "faultscope":
        circuit = worker._build(case)
        sampler = worker.compile_native_sampler(circuit)
        output_layout = "measurement-major Python integer masks"
        internal_batch_size = None
        sample_chunk_shots = None
    elif worker.engine_key == "stim":
        circuit = worker.stim.Circuit(_stim_text(case))
        sampler = circuit.compile_sampler(seed=seed)
        output_layout = "shot-major little-endian packed uint8 rows"
        internal_batch_size = None
        sample_chunk_shots = None
    setup_seconds = time.perf_counter() - started

    digest = hashlib.sha256()
    digest.update(measurements.to_bytes(8, "little"))
    sample_call_seconds = 0.0
    one_bits_total = 0
    row_bytes = (measurements + 7) // 8
    trailing_bits = measurements % 8
    trailing_mask = (1 << trailing_bits) - 1 if trailing_bits else 0xFF
    expected_keys = {f"m{index}" for index in range(measurements)} if worker.engine_key == "faultscope" else None
    batches = batch_ranges(shots, batch_shots)
    for batch_index, (_, this_batch) in enumerate(batches):
        call_started = time.perf_counter()
        if worker.engine_key == "faultscope":
            output = sampler.sample_measurements(
                this_batch, seed=derive_seed(seed, batch_index)
            )
        elif worker.engine_key == "stim":
            output = sampler.sample_bit_packed(this_batch)
        sample_call_seconds += time.perf_counter() - call_started

        digest.update(this_batch.to_bytes(8, "little"))
        if worker.engine_key == "faultscope":
            if not isinstance(output, dict) or set(output) != expected_keys:
                raise RuntimeError("FaultScope returned an incomplete measurement record")
            mask_bytes = (this_batch + 7) // 8
            for measurement_index in range(measurements):
                mask = int(output[f"m{measurement_index}"])
                if mask < 0 or mask.bit_length() > this_batch:
                    raise RuntimeError("FaultScope mask has out-of-range shot bits")
                one_bits_total += mask.bit_count()
                digest.update(mask.to_bytes(mask_bytes, "little"))
        else:
            packed = np.asarray(output)
            if packed.dtype != np.uint8 or packed.shape != (this_batch, row_bytes):
                raise RuntimeError(
                    f"{worker.engine_key} returned {packed.dtype} {packed.shape}, "
                    f"expected uint8 {(this_batch, row_bytes)}"
                )
            if not packed.flags.c_contiguous:
                raise RuntimeError("packed sampler output is not contiguous")
            # Count in bounded blocks. A whole-array bitwise_count would
            # allocate a second output-sized array and skew the RSS gate for
            # large, otherwise feasible native batches.
            for first in range(0, this_batch, 8_192):
                block = packed[first:first + 8_192]
                if trailing_bits:
                    # Padding is outside the record; the existing paper
                    # adapters mask it rather than requiring a fixed value.
                    if row_bytes > 1:
                        one_bits_total += int(np.bitwise_count(block[:, :-1]).sum(dtype=np.uint64))
                    one_bits_total += int(
                        np.bitwise_count(block[:, -1] & trailing_mask).sum(dtype=np.uint64)
                    )
                else:
                    one_bits_total += int(np.bitwise_count(block).sum(dtype=np.uint64))
            digest.update(memoryview(packed))
            del packed
        del output
    end_to_end_seconds = time.perf_counter() - started
    return {
        "status": "ok",
        "shots_completed": shots,
        "num_measurements": measurements,
        "one_bits_total": one_bits_total,
        "record_sha256_native_layout": digest.hexdigest(),
        "output_layout": output_layout,
        "batches": len(batches),
        "batch_shots": batch_shots,
        "symft_mode": symft_mode,
        "symft_batch_size": internal_batch_size,
        "symft_sample_chunk_shots": sample_chunk_shots,
        "setup_seconds": setup_seconds,
        "sample_call_seconds": sample_call_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "shots_per_second": shots / sample_call_seconds,
        "worker_high_water_rss_bytes": _peak_rss_bytes(),
    }


def serve(engine: str) -> None:
    worker = _worker_for(engine)
    try:
        worker.load_engine()
        worker.warmup()
        probe = worker.probe_details()
        if engine == "stim":
            package_dir = Path(worker.stim.__file__).resolve().parent
            native_files = sorted(package_dir.glob("*.so"))
            if not native_files:
                raise RuntimeError("Stim native modules were not found")
            probe["native_modules_sha256"] = {
                path.name: sha256_file(path) for path in native_files
            }
            selected = next(
                (path for path in native_files if "_stim_sse2" in path.name),
                native_files[0],
            )
            probe["native_module_file"] = str(selected)
            probe["native_module_sha256"] = sha256_file(selected)
        probe["python_executable"] = sys.executable
        probe["python_version"] = sys.version
        probe["numpy_version"] = worker.np.__version__
        available = True
        error = None
    except Exception as exc:
        probe = {}
        available = False
        error = f"{type(exc).__name__}: {exc}"
    write_message(sys.stdout, {
        "type": "ready", "protocol_version": PROTOCOL_VERSION,
        "engine": engine, "available": available, "probe": probe,
        "error": error,
    })
    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            if request.get("command") == "shutdown":
                write_message(sys.stdout, {"type": "bye", "request_id": request_id})
                return
            if request.get("command") != "measure":
                raise ValueError("expected measure or shutdown command")
            if not available:
                raise RuntimeError(error or "simulator unavailable")
            result = _measure(worker, request)
            write_message(sys.stdout, {
                "type": "throughput", "request_id": request_id,
                "engine": engine, **result,
            })
        except MemoryError as exc:
            write_message(sys.stdout, {
                "type": "throughput", "request_id": request.get("request_id"),
                "engine": engine, "status": "oom", "error": str(exc),
                "worker_high_water_rss_bytes": _peak_rss_bytes(),
            })
        except Exception as exc:
            write_message(sys.stdout, {
                "type": "throughput", "request_id": request.get("request_id"),
                "engine": engine, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
                "worker_high_water_rss_bytes": _peak_rss_bytes(),
            })


def _case_groups(random_dir: Path, surface_dir: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for width in RANDOM_WIDTHS:
        name = f"random_n{width}"
        groups[name] = []
        for instance in range(3):
            path = random_dir / f"random_clifford_n{width}_d64_i{instance}.json"
            case = load_case_metadata(path, verify_hash=True)
            groups[name].append({
                "path": str(path.resolve()), "case_id": case.metadata["case_id"],
                "family": "random_clifford", "size": width, "instance": instance,
                "metadata_sha256": sha256_file(path),
                "operations_sha256": case.metadata["operations_sha256"],
                "sidecar_sha256": case.metadata["sidecar_sha256"],
                "num_measurements": case.metadata["num_measurements"],
            })
    for distance in SURFACE_DISTANCES:
        name = f"surface_d{distance}"
        path = surface_dir / f"rotated_memory_z_d{distance}_r{distance}_p0.001.json"
        case = load_case_metadata(path, verify_hash=True)
        groups[name] = [{
            "path": str(path.resolve()), "case_id": case.metadata["case_id"],
            "family": "surface_code", "size": distance, "instance": None,
            "metadata_sha256": sha256_file(path),
            "operations_sha256": case.metadata["operations_sha256"],
            "sidecar_sha256": case.metadata["sidecar_sha256"],
            "num_measurements": case.metadata["num_measurements"],
        }]
    return groups


def _candidates(engine: str) -> list[tuple[int, str]]:
    if engine not in ENGINES:
        raise ValueError(f"unknown Table 3 engine {engine!r}")
    return [(size, "none") for size in BATCH_CHOICES]


def _worker_spec(engine: str) -> EngineSpec:
    return EngineSpec(
        key=engine, label=engine,
        command=(sys.executable, "-m", "fsbench.throughput", "worker", engine),
        expected_versions={"version": EXPECTED_VERSIONS[engine]},
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _seed(phase: str, case_id: str, repeat: int) -> int:
    payload = f"fsbench-throughput-v1\0{phase}\0{case_id}\0{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _run_trial(
    *, engine: str, case: dict[str, Any], phase: str, repeat: int,
    batch_shots: int, symft_mode: str, cpu: int, shots: int,
    probe: dict[str, Any], source_sha256: str,
) -> dict[str, Any]:
    from .runner import WorkerFailure, WorkerSession

    seed = _seed(phase, case["case_id"], repeat)
    row = {
        "phase": phase, "engine": engine, "case_id": case["case_id"],
        "family": case["family"], "size": case["size"],
        "instance": case["instance"], "repeat": repeat,
        "batch_shots": batch_shots, "symft_mode": symft_mode,
        "shots_requested": shots, "seed": seed,
        "cpu": cpu, "started_at": utc_now(),
        "engine_version": probe.get("version"),
        "native_module_sha256": probe.get("native_module_sha256") or probe.get("native_modules_sha256"),
        "case_metadata_sha256": case["metadata_sha256"],
        "operations_sha256": case["operations_sha256"],
        "sidecar_sha256": case["sidecar_sha256"],
        "throughput_source_sha256": source_sha256,
    }
    request = {
        "command": "measure", "request_id": uuid.uuid4().hex,
        "case_path": case["path"], "shots": shots, "seed": seed,
        "batch_shots": batch_shots, "symft_mode": symft_mode,
    }
    try:
        with WorkerSession(
            _worker_spec(engine), cpu_id=cpu,
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
        ) as session:
            ready = session.ready or {}
            if not ready.get("available"):
                raise WorkerFailure("unsupported", str(ready.get("error")))
            # serve() already executes the paper worker's representative
            # warmup before publishing ready. Keeping each case invocation
            # fresh avoids an extra untimed full-size compilation per trial.
            result = session.request(
                request, timeout=TIMEOUT_SECONDS, expected_type="throughput"
            )
            row.update(result)
            row["worker_startup_warmup"] = True
            row["peak_rss_bytes"] = max(
                int(result.get("peak_rss_bytes") or 0),
                int(result.get("worker_high_water_rss_bytes") or 0),
            )
    except WorkerFailure as exc:
        row.update({
            "status": exc.status, "error": str(exc),
            "peak_rss_bytes": exc.peak_rss_bytes,
        })
    row.pop("type", None)
    row.pop("request_id", None)
    row["finished_at"] = utc_now()
    return row


def _candidate_score(
    rows: list[dict[str, Any]], cases: list[dict[str, Any]],
    engine: str, batch_shots: int, mode: str,
    pilot_repeats: int,
) -> float | None:
    values: list[float] = []
    for case in cases:
        mine = [
            row for row in rows
            if row["phase"] == "pilot" and row["engine"] == engine
            and row["case_id"] == case["case_id"]
            and row["batch_shots"] == batch_shots
            and row["symft_mode"] == mode
            and row["status"] == "ok"
        ]
        if (len(mine) != pilot_repeats
                or {row["repeat"] for row in mine} != set(range(pilot_repeats))):
            return None
        values.append(statistics.median(float(row["shots_per_second"]) for row in mine))
    return statistics.median(values)


def _selected_candidates(
    rows: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]],
    pilot_repeats: int,
) -> dict[str, dict[str, dict[str, Any] | None]]:
    selected: dict[str, dict[str, dict[str, Any] | None]] = {}
    for group, cases in groups.items():
        selected[group] = {}
        for engine in ENGINES:
            scores = [
                (score, batch, mode)
                for batch, mode in _candidates(engine)
                if (score := _candidate_score(rows, cases, engine, batch, mode, pilot_repeats)) is not None
            ]
            if not scores:
                selected[group][engine] = None
                continue
            # A smaller batch breaks an exact score tie, reducing peak memory.
            score, batch, mode = max(scores, key=lambda item: (item[0], -item[1], item[2] == "default"))
            selected[group][engine] = {
                "batch_shots": batch, "symft_mode": mode,
                "pilot_median_shots_per_second": score,
            }
    return selected


def run(results_dir: Path, random_dir: Path, surface_dir: Path) -> dict[str, Any]:
    """Run the complete fixed grid; paths are supplied by the public entry point."""
    from .runner import WorkerSession
    from paper_runtime import FAULTSCOPE_COMMIT, physical_cpus, write_json

    groups = _case_groups(random_dir, surface_dir)
    raw_path = results_dir / "raw_runs.jsonl"
    source_hash = sha256_file(Path(__file__))
    cpu = physical_cpus(limit=1)[0]
    probes: dict[str, dict[str, Any]] = {}
    for engine in ENGINES:
        with WorkerSession(_worker_spec(engine), cpu_id=cpu,
                           memory_limit_bytes=MEMORY_LIMIT_BYTES) as session:
            ready = session.ready or {}
            if not ready.get("available"):
                raise RuntimeError(f"{engine} unavailable: {ready.get('error')}")
            probe = dict(ready["probe"])
            if probe.get("version") != EXPECTED_VERSIONS[engine]:
                raise RuntimeError(f"unexpected {engine} version: {probe.get('version')}")
            probes[engine] = probe
    manifest = {
        "schema": "paper-table3-throughput-v1", "created_at": utc_now(),
        "run_id": results_dir.name, "host": platform.node(),
        "platform": platform.platform(), "cpu": cpu,
        "thread_limit": 1, "concurrent_workers": 1,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_monitor_interval_seconds": 0.05,
        "timeout_seconds": TIMEOUT_SECONDS,
        "shots": SHOTS, "pilot_repeats": PILOT_REPEATS,
        "formal_repeats": FORMAL_REPEATS,
        "expected_pilot_trials": 192, "expected_formal_trials": 144,
        "batch_choices": list(BATCH_CHOICES), "engines": list(ENGINES),
        "sampling_metric": "shots divided by sum of public sampling-call wall times",
        "output_format_policy": "each engine's complete native measurement output",
        "source_sha256": source_hash, "probes": probes,
        "faultscope_source_commit": FAULTSCOPE_COMMIT, "groups": groups,
    }
    write_json(results_dir / "throughput_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    for group, cases in groups.items():
        for engine in ENGINES:
            for batch_shots, mode in _candidates(engine):
                for case in cases:
                    for repeat in range(PILOT_REPEATS):
                        row = _run_trial(
                            engine=engine, case=case, phase="pilot", repeat=repeat,
                            batch_shots=batch_shots, symft_mode=mode, cpu=cpu,
                            shots=SHOTS, probe=probes[engine], source_sha256=source_hash,
                        )
                        _append_jsonl(raw_path, row)
                        rows.append(row)
                        print(f"pilot {group} {engine} {case['case_id']} B={batch_shots} r={repeat}: {row['status']}", flush=True)
    selected = _selected_candidates(rows, groups, PILOT_REPEATS)
    write_json(results_dir / "selected_batches.json", selected)
    for repeat in range(FORMAL_REPEATS):
        order = ENGINES[repeat % len(ENGINES):] + ENGINES[:repeat % len(ENGINES)]
        for group, cases in groups.items():
            for engine in order:
                setting = selected[group][engine]
                if setting is None:
                    continue
                for case in cases:
                    row = _run_trial(
                        engine=engine, case=case, phase="formal", repeat=repeat,
                        batch_shots=int(setting["batch_shots"]), symft_mode="none", cpu=cpu,
                        shots=SHOTS, probe=probes[engine], source_sha256=source_hash,
                    )
                    _append_jsonl(raw_path, row)
                    rows.append(row)
                    print(f"formal {group} {engine} {case['case_id']} r={repeat}: {row['status']}", flush=True)
    summary = report(results_dir)
    write_json(results_dir / "trial_status.json", {
        "complete": summary["complete"],
        "pilot_trials": sum(row["phase"] == "pilot" for row in rows),
        "formal_trials": sum(row["phase"] == "formal" for row in rows),
        "failures": [row for row in rows if row["status"] != "ok"],
    })
    if not summary["complete"]:
        raise RuntimeError("Table 3 is incomplete; see trial_status.json and raw_runs.jsonl")
    return manifest


def _quartiles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if len(ordered) < 2:
        return ordered[0], ordered[0]
    cuts = statistics.quantiles(ordered, n=4, method="inclusive")
    return cuts[0], cuts[2]


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_candidates(rows, manifest["groups"], int(manifest["pilot_repeats"]))
    output: list[dict[str, Any]] = []
    for group, cases in manifest["groups"].items():
        for engine in ENGINES:
            setting = selected[group][engine]
            item: dict[str, Any] = {
                "group": group, "engine": engine, "family": cases[0]["family"],
                "size": cases[0]["size"], "instances_expected": len(cases),
                "repeats_expected": manifest["formal_repeats"],
                "batch_shots": setting["batch_shots"] if setting else None,
                "symft_mode": setting["symft_mode"] if setting else None,
                "status": "unavailable" if setting is None else "incomplete",
            }
            if setting is None:
                failures = [row for row in rows if row["phase"] == "pilot"
                            and row["engine"] == engine and row["status"] != "ok"
                            and row["case_id"] in {case["case_id"] for case in cases}]
                item["error"] = "no complete pilot candidate"
                if failures:
                    item["error"] += "; " + "; ".join(dict.fromkeys(
                        f"{row['status']}: {row.get('error', 'trial failed')}" for row in failures))
            if setting is not None:
                per_case = []
                for case in cases:
                    mine = [
                        row for row in rows if row["phase"] == "formal"
                        and row["engine"] == engine and row["case_id"] == case["case_id"]
                        and row["batch_shots"] == setting["batch_shots"]
                        and row["symft_mode"] == setting["symft_mode"]
                    ]
                    if (len(mine) != manifest["formal_repeats"]
                            or {row["repeat"] for row in mine} != set(range(manifest["formal_repeats"]))
                            or any(row["status"] != "ok" for row in mine)):
                        failures = [row for row in mine if row["status"] != "ok"]
                        item["error"] = f"{case['case_id']}: missing, duplicate or failed formal repeats"
                        if failures:
                            item["error"] += "; " + "; ".join(dict.fromkeys(
                                f"{row['status']}: {row.get('error', 'trial failed')}" for row in failures))
                        per_case = []
                        break
                    per_case.append(mine)
                if per_case:
                    samples = [
                        statistics.median(float(row["shots_per_second"]) for row in mine)
                        for mine in per_case
                    ]
                    setup = [
                        statistics.median(float(row["setup_seconds"]) for row in mine)
                        for mine in per_case
                    ]
                    e2e = [
                        statistics.median(float(row["end_to_end_seconds"]) for row in mine)
                        for mine in per_case
                    ]
                    item.update({
                        "status": "ok", "median_shots_per_second": statistics.median(samples),
                        "q1_shots_per_second": _quartiles(samples if len(cases) > 1 else [float(row["shots_per_second"]) for row in per_case[0]])[0],
                        "q3_shots_per_second": _quartiles(samples if len(cases) > 1 else [float(row["shots_per_second"]) for row in per_case[0]])[1],
                        "instance_medians_shots_per_second": samples,
                        "median_setup_seconds": statistics.median(setup),
                        "median_end_to_end_seconds": statistics.median(e2e),
                        "max_peak_rss_bytes": max(int(row["peak_rss_bytes"]) for mine in per_case for row in mine),
                    })
            output.append(item)
    report_data = {
        "schema": "fsbench-native-throughput-summary-v1",
        "run_id": manifest["run_id"], "rows": output,
        "complete": all(item["status"] == "ok" for item in output),
    }
    return report_data


def report(results_dir: Path) -> dict[str, Any]:
    from table_reports import write_table3

    manifest = json.loads((results_dir / "throughput_manifest.json").read_text())
    report_data = summarize(_read_jsonl(results_dir / "raw_runs.jsonl"), manifest)
    output = report_data["rows"]
    (results_dir / "summary.json").write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n")
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "group", "engine", "family", "size", "status", "error", "batch_shots", "symft_mode",
            "median_shots_per_second", "q1_shots_per_second", "q3_shots_per_second",
            "median_setup_seconds", "median_end_to_end_seconds", "max_peak_rss_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)

    write_table3(results_dir / "table3.tex", output)
    return report_data


def main() -> None:
    # Private worker protocol, not a configurable benchmark entry point.
    if len(sys.argv) != 3 or sys.argv[1] != "worker" or sys.argv[2] not in ENGINES:
        raise SystemExit("Use table3_sampling_throughput.py without arguments")
    serve(sys.argv[2])


if __name__ == "__main__":
    main()
