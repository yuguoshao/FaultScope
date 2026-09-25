#!/usr/bin/env python3
"""Reproduce Table 3 with the complete fixed sampling-throughput experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_support"))
from paper_runtime import run_entry


def benchmark(run_dir: Path) -> None:
    from paper_inputs import generate_verified_cases
    from paper_runtime import finish_run, require_dependencies, runtime_manifest
    from fsbench.throughput import RANDOM_WIDTHS, SURFACE_DISTANCES, run

    require_dependencies(("faultscope", "stim"))
    manifest = runtime_manifest()
    random_dir, surface_dir = generate_verified_cases(
        run_dir / "inputs", random_widths=RANDOM_WIDTHS, surface_distances=SURFACE_DISTANCES,
    )
    manifest["benchmark"] = run(run_dir, random_dir, surface_dir)
    finish_run(run_dir, manifest)


if __name__ == "__main__":
    run_entry("table3_sampling_throughput", benchmark)
