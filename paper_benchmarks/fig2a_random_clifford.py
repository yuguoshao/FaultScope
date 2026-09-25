#!/usr/bin/env python3
"""Reproduce Fig. 2(a) with all six simulators and the fixed random circuits."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_support"))
from paper_runtime import run_entry


def benchmark(run_dir: Path) -> None:
    from end_to_end import run_end_to_end
    run_end_to_end(run_dir, "random_clifford")


if __name__ == "__main__":
    run_entry("fig2a_random_clifford", benchmark)
