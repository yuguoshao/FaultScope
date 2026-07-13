"""Forward-sampler compiler benchmark on flattened Stim surface-code circuits.

Run from the repository root after building the native extension in release mode:

    .venv/bin/python benchmarks/compiler_throughput.py --distances 5 10 15 20

The ``mr`` representation keeps Stim's combined measure-reset instructions.
The ``m-plus-r`` representation expands each one into a measurement block and
an equivalent reset block, matching the Figure-1 comparison circuit shape.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faultscope
from faultscope import compile_native_sampler, parse_stim_circuit


T = TypeVar("T")
_MEASURE_RESET_RE = re.compile(r"^(\s*)MR([XY]?)(\([^)]*\))?(\s+.+)$")
_SKIP_INSTRUCTIONS = {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    stim = _load_stim()

    print(
        "distance\trepresentation\tqubits\tmeasurements\toperations\t"
        "runtime_operations\tparse_s\tcompile_s\tsample_s\tpeak_rss_kib",
        flush=True,
    )
    records = [_run_case(stim, distance, args) for distance in args.distances]
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--sample-repeats", type=int, default=11)
    parser.add_argument("--representation", choices=("mr", "m-plus-r"), default="m-plus-r")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if any(distance < 2 for distance in args.distances):
        raise SystemExit("distances must be at least 2")
    for name in ("shots", "compile_repeats", "sample_repeats"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")


def _run_case(stim: Any, distance: int, args: argparse.Namespace) -> dict[str, object]:
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
    ).flattened()
    source = _normalized_source(stim_circuit, args.representation)

    parse_seconds, imported = _measure_repeated(
        lambda: parse_stim_circuit(source),
        max(args.compile_repeats, 1),
    )
    compile_seconds, sampler = _measure_repeated(
        lambda: compile_native_sampler(imported.circuit),
        args.compile_repeats,
    )
    sample_seconds: list[float] = []
    for repetition in range(args.sample_repeats):
        started = time.perf_counter()
        sampler.sample_measurements(shots=args.shots, seed=args.seed + repetition)
        sample_seconds.append(time.perf_counter() - started)

    gate_counts = Counter(operation.kind for operation in imported.circuit.operations)
    record: dict[str, object] = {
        "distance": distance,
        "rounds": distance,
        "representation": args.representation,
        "shots": args.shots,
        "qubits": int(imported.circuit.n_qubits),
        "measurements": int(stim_circuit.num_measurements),
        "operations": len(imported.circuit.operations),
        "runtime_operations": int(sampler.operation_count),
        "gate_counts": dict(sorted(gate_counts.items())),
        "parse_seconds": parse_seconds,
        "parse_median_seconds": statistics.median(parse_seconds),
        "compile_seconds": compile_seconds,
        "compile_median_seconds": statistics.median(compile_seconds),
        "sample_seconds": sample_seconds,
        "sample_median_seconds": statistics.median(sample_seconds),
        "peak_rss_kib": _peak_rss_kib(),
        "faultscope_version": faultscope.__version__,
        "stim_version": stim.__version__,
    }
    print(
        f"{distance}\t{args.representation}\t{record['qubits']}\t"
        f"{record['measurements']}\t{record['operations']}\t"
        f"{record['runtime_operations']}\t{record['parse_median_seconds']:.9f}\t"
        f"{record['compile_median_seconds']:.9f}\t{record['sample_median_seconds']:.9f}\t"
        f"{record['peak_rss_kib']}",
        flush=True,
    )
    return record


def _measure_repeated(
    function: Callable[[], T],
    repeats: int,
) -> tuple[list[float], T]:
    values: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - started)
    assert result is not None
    return values, result


def _expand_measure_resets(source: str) -> str:
    expanded: list[str] = []
    for line in source.splitlines():
        match = _MEASURE_RESET_RE.match(line)
        if match is None:
            expanded.append(line)
            continue
        indent, suffix, args, targets = match.groups()
        expanded.append(f"{indent}M{suffix}{args or ''}{targets}")
        expanded.append(f"{indent}R{suffix}{targets}")
    return "\n".join(expanded) + "\n"


def _normalized_source(stim_circuit: Any, representation: str) -> str:
    used_qubits = sorted(
        {
            target.value
            for instruction in stim_circuit
            if instruction.name not in _SKIP_INSTRUCTIONS and instruction.name != "TICK"
            for target in instruction.targets_copy()
            if target.is_qubit_target
        }
    )
    remap = {old: new for new, old in enumerate(used_qubits)}
    lines: list[str] = []
    for instruction in stim_circuit:
        if instruction.name in _SKIP_INSTRUCTIONS:
            continue
        if instruction.name == "TICK":
            lines.append("TICK")
            continue
        targets = [remap[target.value] for target in instruction.targets_copy()]
        if not targets:
            continue
        args = instruction.gate_args_copy()
        arg_text = "" if not args else f"({','.join(format(value, '.17g') for value in args)})"
        lines.append(
            f"{instruction.name}{arg_text} " + " ".join(str(target) for target in targets)
        )
    source = "\n".join(lines) + "\n"
    return _expand_measure_resets(source) if representation == "m-plus-r" else source


def _peak_rss_kib() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss // 1024 if sys.platform == "darwin" else rss


def _load_stim() -> Any:
    try:
        import stim
    except ImportError as exc:
        raise SystemExit("Stim is required for this benchmark") from exc
    return stim


if __name__ == "__main__":
    main()
