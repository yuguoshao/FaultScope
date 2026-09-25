"""Fixed Fig. 2 execution, reporting, and compatible-run combination."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fsbench.constants import (
    ALL_ENGINES, DEFAULT_DISTANCES, MASTER_SEED,
    RANDOM_CLIFFORD_GENERATOR_SEED, RANDOM_CLIFFORD_WIDTHS,
)

ENTRIES = {"random_clifford": "fig2a_random_clifford", "surface_code": "fig2b_surface_code"}
PROFILES = {"random_clifford": "random-clifford-depth64-symft-1m-paper-linux", "surface_code": "surface-symft-1m-paper-linux"}
PAPER_PROTOCOL = {
    "schema": "paper-e2e-protocol-v1", "engines": list(ALL_ENGINES),
    "shots": 1_000_000, "max_batch_shots": 1000, "master_seed": MASTER_SEED,
    "random_generator_seed": RANDOM_CLIFFORD_GENERATOR_SEED,
    "random_widths": list(RANDOM_CLIFFORD_WIDTHS), "random_instances": [0, 1, 2], "random_depth": 64,
    "surface_distances": list(DEFAULT_DISTANCES), "surface_rounds": "distance",
    "surface_noise": {name: 0.001 for name in ("after_clifford_depolarization", "after_reset_flip_probability", "before_measure_flip_probability", "before_round_data_depolarization")},
    "timeout_seconds": 3600, "rss_limit_bytes": 36 * 1024**3, "rss_poll_seconds": 0.05,
    "parallelism": "min(64, allowed physical cores)", "threads_per_worker": 1,
    "adaptive_repetitions": {"calibration": "one full E2E invocation", "fast_seconds_max": 60, "fast_samples_min": 7, "fast_samples_max": 15, "fast_total_seconds_min": 10, "fast_sample_target_seconds": 1, "medium_seconds_max": 300, "medium_samples": 3, "slow_samples": 1},
}


def latest_completed_run(results: Path, entry: str, *, completing_run: Path | None = None) -> Path | None:
    parent = results / entry
    if not parent.is_dir():
        return None
    for directory in sorted((p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()), reverse=True):
        try:
            manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
            status_path = directory / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        except (OSError, ValueError):
            continue
        # The caller has already validated the current run; other processes
        # must finish their lifecycle before their output can be selected.
        finished = status.get("status") == "complete"
        current = directory == completing_run and status.get("status") == "running"
        if manifest.get("complete") is True and (finished or current):
            return directory
    return None


def combine_latest(run_dir: Path) -> dict[str, Any]:
    from paper_runtime import RESULTS, write_json
    from fsbench.combined_report import build_combined_report
    selected = {family: latest_completed_run(RESULTS, entry, completing_run=run_dir) for family, entry in ENTRIES.items()}
    if any(directory is None for directory in selected.values()):
        result = {"status": "waiting", "reason": "Both Fig. 2 workloads must have a completed run before combination."}
    else:
        try:
            output_dir = run_dir / "combined"
            build_combined_report(surface_results_dir=selected["surface_code"], random_results_dir=selected["random_clifford"], output_dir=output_dir)
            result = {"status": "complete", "output_dir": str(output_dir), "sources": {key: str(path) for key, path in selected.items()}}
            print(f"Combined Fig. 2 and Table 2: {output_dir}")
        except ValueError as exc:
            result = {"status": "incompatible", "reason": str(exc), "sources": {key: str(path) for key, path in selected.items()}}
            print(f"Fig. 2 combination skipped: {exc}")
    write_json(run_dir / "combination_status.json", result)
    return result


def run_end_to_end(run_dir: Path, family: str) -> None:
    from paper_runtime import finish_run, require_dependencies, runtime_manifest, write_json
    from paper_inputs import generate_verified_cases
    from fsbench.runner import run_benchmark
    from fsbench.validation import read_csv, validate_paper_run

    if family not in ENTRIES:
        raise ValueError(f"unknown paper family {family!r}")
    require_dependencies(ALL_ENGINES)
    runtime = runtime_manifest()
    random_dir, surface_dir = generate_verified_cases(
        run_dir / "inputs",
        random_widths=RANDOM_CLIFFORD_WIDTHS if family == "random_clifford" else (),
        surface_distances=DEFAULT_DISTANCES if family == "surface_code" else (),
    )
    manifest = run_benchmark(profile_name=PROFILES[family], cases_dir=random_dir if family == "random_clifford" else surface_dir, results_dir=run_dir, resume=False)
    validation = validate_paper_run(run_dir)
    write_json(run_dir / "validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError("paper run validation failed: " + "; ".join(validation["errors"]))
    summary, status = read_csv(run_dir / "summary.csv"), read_csv(run_dir / "status.csv")
    if family == "random_clifford":
        from fsbench.random_report import aggregate_random_instances, generate_random_scaling_figure
        aggregate = aggregate_random_instances(manifest=manifest, summary_rows=summary, status_rows=status)
        write_json(run_dir / "random_instance_aggregate.json", aggregate)
        figure = generate_random_scaling_figure(results_dir=run_dir, manifest=manifest, aggregate_rows=aggregate)
    else:
        from fsbench.report import generate_scaling_figure
        figure = generate_scaling_figure(results_dir=run_dir, manifest=manifest, summary_rows=summary, status_rows=status)
    manifest.update(entrypoint=ENTRIES[family], runtime=runtime, paper_protocol=PAPER_PROTOCOL, paper_validation=validation, panel=figure)
    finish_run(run_dir, manifest)
    combine_latest(run_dir)
    # Include combination status and all generated artifacts in this run's
    # checksums. The combined report keeps immutable input-manifest snapshots.
    finish_run(run_dir, manifest)
