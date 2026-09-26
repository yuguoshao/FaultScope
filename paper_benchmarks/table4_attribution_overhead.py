#!/usr/bin/env python3
"""Reproduce Table 4 with the fixed five-size attribution-overhead experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_support"))
from paper_runtime import run_entry


def benchmark(run_dir: Path) -> None:
    from paper_runtime import finish_run, require_dependencies, runtime_manifest
    from attribution import run

    require_dependencies(("faultscope",))
    manifest = runtime_manifest()
    manifest["benchmark"] = run(run_dir)
    finish_run(run_dir, manifest)


if __name__ == "__main__":
    run_entry("table4_attribution_overhead", benchmark)
