"""Fixed Table 4 repetition-code timing using the public native batch API."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import signal
import sys
import time
from functools import lru_cache
from pathlib import Path

from table_reports import METHODS, summarize_attribution, write_csv, write_table4


DISTANCES = (7, 31, 127, 511, 2047)
SHOTS = 100_000
ERROR_PROBABILITY = 0.5
BASELINE = 0.0
WARMUPS = 2
REPEATS = 7
SEED_BASE = 290001
MEMORY_LIMIT_BYTES = 36 * 1024**3
TRIAL_TIMEOUT_SECONDS = 3600


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def _api():
    import faultscope as fs
    from faultscope.decoders import RepetitionCodeDecoder
    from faultscope.runtime.loss import logical_residual_loss_mask

    return fs, RepetitionCodeDecoder, logical_residual_loss_mask


def make_circuit(distance: int):
    """One X-error layer, ideal adjacent Z checks, then ideal Z readout."""
    fs = _api()[0]
    require(distance >= 3 and distance % 2 == 1, "distance must be odd and at least 3")
    operations = [fs.Operation.noise(fs.NoiseLocation(
        id=f"x{qubit}", model=fs.BernoulliPauliNoise("X"), rate=ERROR_PROBABILITY,
        qubits=(qubit,), tags={"qubit": qubit, "round": 0, "gate": "idle"},
    )) for qubit in range(distance)]
    keys = tuple(f"check{qubit}" for qubit in range(distance - 1))
    for qubit, key in enumerate(keys):
        ancilla = distance + qubit
        operations.extend((
            fs.Operation.cx(qubit, ancilla), fs.Operation.cx(qubit + 1, ancilla),
            fs.Operation.measure(ancilla, key=key, basis="Z"),
            fs.Operation.detector((key,), detector_id=qubit),
        ))
    operations.extend(fs.Operation.measure(q, key=f"data{q}", basis="Z")
                      for q in range(distance))
    circuit = fs.Circuit(n_qubits=2 * distance - 1, operations=tuple(operations))
    observables = (fs.LogicalObservable(id=0, measurement_keys=("data0",)),)
    return circuit, observables, keys


def make_problem(distance: int):
    fs, decoder_class, _ = _api()
    circuit, observables, keys = make_circuit(distance)
    sampler = fs.compile_native_sampler(circuit, observables=observables)
    decoder = decoder_class(distance, measurement_keys=keys, observable_id=0)
    return sampler, decoder


def describe_circuit(circuit, observables, measurement_keys) -> dict:
    """Serialize the actual FaultScope objects, outside every timed trial."""
    operations = []
    for operation in circuit.operations:
        item = {name: getattr(operation, name) for name in (
            "kind", "qubits", "key", "basis", "pauli", "measurement_keys",
            "observable_id", "metadata",
        )}
        location = operation.noise_location
        item["noise_location"] = None if location is None else {
            "id": location.id, "rate": location.rate, "qubits": location.qubits,
            "tags": location.tags,
            "model": {"type": type(location.model).__name__, "pauli": location.model.pauli},
        }
        operations.append(item)
    return {
        "schema": "paper-attribution-input-v1", "initial_state": "all-zero",
        "n_qubits": circuit.n_qubits, "operations": operations,
        "logical_observables": [{name: getattr(observable, name) for name in (
            "id", "measurement_keys", "pauli_qubits", "pauli",
        )} for observable in observables],
        "decoder": {"type": "RepetitionCodeDecoder", "distance": (circuit.n_qubits + 1) // 2,
                    "measurement_keys": measurement_keys, "observable_id": 0},
    }


def canonical_input_bytes(description: dict) -> bytes:
    return (json.dumps(description, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def write_verified_inputs(output_dir: Path) -> dict[str, str]:
    expected = json.loads(Path(__file__).with_name("expected_attribution_cases.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for distance in DISTANCES:
        case_id = f"repetition_d{distance}_p05"
        payload = canonical_input_bytes(describe_circuit(*make_circuit(distance)))
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == expected["cases"].get(case_id), f"fixed attribution input changed: {case_id}")
        (output_dir / f"{case_id}.json").write_bytes(payload)
        hashes[case_id] = digest
    return hashes


def decoded_loss(batch, decoder) -> int:
    correction = decoder.decode_batch_masks(batch)
    return _api()[2](batch.observables, correction, observable_ids=(0,),
                     all_mask=batch.all_mask)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def trial(method: str, distance: int, shots: int, seed: int) -> dict:
    """Arguments are private test hooks; the public run uses only fixed constants."""
    require(method in METHODS, "unknown attribution method")

    def timeout(_signal, _frame):
        raise TimeoutError(f"{method} d={distance} exceeded {TRIAL_TIMEOUT_SECONDS}s")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(TRIAL_TIMEOUT_SECONDS)
    try:
        start = time.perf_counter_ns()
        sampler, decoder = make_problem(distance)
        prepared = time.perf_counter_ns()
        batch = sampler.run_native_batch(shots=shots, seed=seed,
                                         record_events=method != "no_events")
        sampled = time.perf_counter_ns()
        loss_mask = decoded_loss(batch, decoder)
        loss_count = loss_mask.bit_count()
        decoded = time.perf_counter_ns()
        estimate = None
        if method == "score":
            estimate = sampler.estimate_hotspots(batch, loss_mask, baseline=BASELINE,
                                                 top_k=distance)
        finished = time.perf_counter_ns()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    # Result checks, RSS inspection and serialization are outside the clock.
    require(batch.records_events == (method != "no_events"), "event-recording flag mismatch")
    if estimate is not None:
        require(len(estimate.sensitivities) == distance, "missing sensitivities")
        require(all(math.isfinite(value) for value in estimate.sensitivities.values()),
                "non-finite sensitivity")
        require(math.isclose(estimate.mean_loss, loss_count / shots, abs_tol=1e-15),
                "attribution loss differs from decoded loss")
    return {
        "method": method, "distance": distance, "physical_qubits": 2 * distance - 1,
        "error_locations": distance, "shots": shots,
        "error_probability": ERROR_PROBABILITY, "record_events": method != "no_events",
        "status": "completed", "seed": seed, "loss_count": loss_count,
        "preparation_seconds": (prepared - start) / 1e9,
        "sampling_seconds": (sampled - prepared) / 1e9,
        "decoding_seconds": (decoded - sampled) / 1e9,
        "scoring_seconds": (finished - decoded) / 1e9,
        "total_seconds": (finished - start) / 1e9,
        "peak_process_rss_bytes": _peak_rss_bytes(),
    }


def validate_paths(distances: tuple[int, ...] = DISTANCES) -> list[dict]:
    checks = []
    for distance in distances:
        sampler, decoder = make_problem(distance)
        shots, seed = 257, 760000 + distance
        no_events = sampler.run_native_batch(shots=shots, seed=seed, record_events=False)
        with_events = sampler.run_native_batch(shots=shots, seed=seed, record_events=True)
        default = sampler.run_native_batch(shots=shots, seed=seed)
        require(not no_events.records_events and with_events.records_events and default.records_events,
                "record_events API does not have the required behavior")
        require(no_events.noise_event_masks == {}, "no-event batch contains event masks")
        require(with_events.noise_event_masks == default.noise_event_masks,
                "default recording differs from explicit recording")
        require(len(with_events.noise_event_masks) == distance, "missing event masks")
        require(no_events.measurement_masks(decoder.measurement_keys)
                == with_events.measurement_masks(decoder.measurement_keys), "parity masks differ")
        require(no_events.observables == with_events.observables, "observables differ")
        loss_no, loss_yes = decoded_loss(no_events, decoder), decoded_loss(with_events, decoder)
        require(loss_no == loss_yes, "decoded losses differ")
        result = sampler.estimate_hotspots(with_events, loss_yes, baseline=BASELINE, top_k=distance)
        require(len(result.sensitivities) == distance, "missing sensitivities")
        require(result.mean_loss == loss_yes.bit_count() / shots, "mean loss mismatch")
        # Independently recover each score from joint event/loss counts.
        for location, event_mask in with_events.noise_event_masks.items():
            joint = (event_mask & loss_yes).bit_count()
            expected = (joint / ERROR_PROBABILITY
                        - (loss_yes.bit_count() - joint) / (1 - ERROR_PROBABILITY)) / shots
            require(math.isclose(result.sensitivities[location], expected, abs_tol=1e-14),
                    f"sensitivity count check failed for {location}")
        try:
            sampler.estimate_hotspots(no_events, loss_no, baseline=BASELINE, top_k=distance)
        except ValueError as error:
            require("requires recorded event masks" in str(error), "unexpected no-event error")
        else:
            raise RuntimeError("attribution accepted a batch without event records")
        checks.append({
            "distance": distance, "shots": shots, "seed": seed,
            "error_probability": ERROR_PROBABILITY, "loss_count": loss_yes.bit_count(),
            "measurements_equal": True, "observables_equal": True,
            "decoded_loss_equal": True, "default_records_events": True,
            "no_events_records_events": False, "with_events_records_events": True,
            "no_events_masks_empty": True, "no_event_attribution_rejected": True,
            "score_locations": distance, "score_count_formula_checked": True,
        })
    return checks


def run(run_dir: Path) -> dict:
    from paper_runtime import FAULTSCOPE_COMMIT, physical_cpus, pin_to_cpu, write_json

    _api()  # Imports are excluded from the benchmark.
    input_hashes = write_verified_inputs(run_dir / "inputs")
    cpu = physical_cpus(limit=1)[0]
    pin_to_cpu(cpu)
    settings = {
        "faultscope_commit": FAULTSCOPE_COMMIT, "faultscope_version": "0.2.11",
        "distances": DISTANCES, "shots_per_evaluation": SHOTS,
        "error_probability": ERROR_PROBABILITY, "baseline": BASELINE,
        "methods": METHODS, "warmups": WARMUPS, "repeats": REPEATS,
        "seed_base": SEED_BASE, "cpu": cpu, "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "trial_timeout_seconds": TRIAL_TIMEOUT_SECONDS,
        "memory_policy": "cumulative process high-water RSS checked after each trial",
        "timeout_policy": "SIGALRM; Python handles the signal when native execution returns",
        "timer": "time.perf_counter_ns",
        "timing_boundary": "construction, compilation, sampling, decoding and requested estimation",
        "excluded": "imports, warmups, validation, checks and serialization",
        "input_sha256": input_hashes,
    }
    write_json(run_dir / "settings.json", settings)
    checks, validation_failures = [], {}
    for distance in DISTANCES:
        try:
            checks.extend(validate_paths((distance,)))
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            validation_failures[distance] = reason
            checks.append({"distance": distance, "status": "validation_error", "error": reason})
    write_json(run_dir / "path_checks.json", checks)
    records = []
    try:
        with (run_dir / "timing_trials.jsonl").open("w", encoding="utf-8") as stream:
            for distance in DISTANCES:
                for repeat in range(-WARMUPS, REPEATS):
                    offset = (repeat + WARMUPS) % len(METHODS)
                    order = METHODS[offset:] + METHODS[:offset]
                    seed = SEED_BASE + distance * 10000 + (repeat + WARMUPS) * 100
                    for method in order:
                        try:
                            if distance in validation_failures:
                                raise RuntimeError(validation_failures[distance])
                            row = trial(method, distance, SHOTS, seed)
                            if row["peak_process_rss_bytes"] > MEMORY_LIMIT_BYTES:
                                row.update(status="oom", error=f"RSS limit exceeded at d={distance}")
                        except Exception as error:
                            status = ("validation_error" if distance in validation_failures else
                                      "timeout" if isinstance(error, TimeoutError) else
                                      "oom" if isinstance(error, MemoryError) else "error")
                            row = {
                                "method": method, "distance": distance, "shots": SHOTS,
                                "seed": seed, "error_probability": ERROR_PROBABILITY,
                                "record_events": method != "no_events", "status": status,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        row.update(repeat=repeat, warmup=repeat < 0, method_order=order)
                        records.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                        if row["status"] != "completed":
                            print(f"d={distance} {method} r={repeat}: {row['status']}: {row['error']}", flush=True)
                print(f"completed d={distance}", flush=True)
        report = summarize_attribution(records, DISTANCES, REPEATS)
        write_json(run_dir / "summary.json", report)
        write_csv(run_dir / "timing_summary.csv", report["rows"])
        write_csv(run_dir / "comparison_summary.csv", report["comparisons"])
        write_table4(run_dir / "table4.tex", report["rows"], report["comparisons"])
    except Exception as error:
        write_json(run_dir / "trial_status.json", {"complete": False, "trials": len(records),
                                                   "error": f"{type(error).__name__}: {error}"})
        raise
    failures = [row for row in records if row["status"] != "completed"]
    complete = report["complete"] and not failures
    write_json(run_dir / "trial_status.json", {
        "complete": complete, "trials": len(records),
        "formal_trials": sum(not row["warmup"] for row in records),
        "warmup_trials": sum(row["warmup"] for row in records), "failures": failures,
    })
    if not complete:
        raise RuntimeError("Table 4 is incomplete; all other trials were attempted; see trial_status.json")
    return settings
