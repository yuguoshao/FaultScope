"""Forward-sampler compiler benchmark on compact and flattened Stim circuits.

Run from the repository root after building the native extension in release mode:

    .venv/bin/python benchmarks/compiler_throughput.py --distances 5 10 15 20

The default preserves Stim ``REPEAT`` blocks so the Rust compiler can use its
native loop path. Pass ``--input-form flattened`` only when intentionally
measuring already-expanded input, or ``--input-form both`` to compare them.

The ``mr`` representation keeps Stim's combined measure-reset instructions.
The ``m-plus-r`` representation expands each one into a measurement block and
an equivalent reset block, matching the Figure-1 comparison circuit shape.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
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
    _load_stim()

    print(
        "distance\tinput_form\trepresentation\tqubits\tmeasurements\t"
        "stored_operations\tlogical_operations\truntime_operations\tloop_kernels\t"
        "loop_recovered\tparse_s\tcompile_s\tsample_s\tpeak_rss_kib",
        flush=True,
    )
    input_forms = ("repeat", "flattened") if args.input_form == "both" else (args.input_form,)
    records = []
    for distance in args.distances:
        for input_form in input_forms:
            record = _run_case_isolated(distance, input_form, args)
            _print_record(record)
            records.append(record)
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
    parser.add_argument(
        "--input-form",
        choices=("repeat", "flattened", "both"),
        default="repeat",
        help=(
            "preserve Stim REPEAT blocks (default), flatten them for an expanded-input "
            "stress test, or benchmark both"
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if any(distance < 2 for distance in args.distances):
        raise SystemExit("distances must be at least 2")
    for name in ("shots", "compile_repeats", "sample_repeats"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")


def _run_case(
    stim: Any,
    distance: int,
    input_form: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
    )
    source_circuit = stim_circuit if input_form == "repeat" else stim_circuit.flattened()
    source = _normalized_source(source_circuit, args.representation)

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

    gate_counts = Counter(_iter_operation_kinds(imported.circuit.operations))
    loop_recovered = input_form == "flattened" and any(
        operation.kind == "repeat" for operation in imported.circuit.operations
    )
    record: dict[str, object] = {
        "distance": distance,
        "rounds": distance,
        "representation": args.representation,
        "input_form": input_form,
        "shots": args.shots,
        "qubits": int(imported.circuit.n_qubits),
        "measurements": int(stim_circuit.num_measurements),
        "operations": len(imported.circuit.operations),
        "stored_operations": int(sampler.stored_operation_count),
        "logical_operations": int(sampler.operation_count),
        "runtime_operations": int(sampler.operation_count),
        "loop_kernels": int(sampler.loop_kernel_count),
        "loop_recovered": loop_recovered,
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
    return record


def _run_case_isolated(
    distance: int,
    input_form: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=1) as pool:
        return pool.apply(
            _run_case_worker,
            ((distance, input_form, vars(args)),),
        )


def _run_case_worker(
    payload: tuple[int, str, dict[str, object]],
) -> dict[str, object]:
    distance, input_form, raw_args = payload
    return _run_case(
        _load_stim(),
        distance,
        input_form,
        argparse.Namespace(**raw_args),
    )


def _print_record(record: dict[str, object]) -> None:
    print(
        f"{record['distance']}\t{record['input_form']}\t{record['representation']}\t"
        f"{record['qubits']}\t"
        f"{record['measurements']}\t{record['stored_operations']}\t"
        f"{record['logical_operations']}\t{record['runtime_operations']}\t"
        f"{record['loop_kernels']}\t{int(bool(record['loop_recovered']))}\t"
        f"{record['parse_median_seconds']:.9f}\t"
        f"{record['compile_median_seconds']:.9f}\t{record['sample_median_seconds']:.9f}\t"
        f"{record['peak_rss_kib']}",
        flush=True,
    )


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
    flattened = stim_circuit.flattened()
    used_qubits = sorted(
        {
            target.value
            for instruction in flattened
            if instruction.name not in _SKIP_INSTRUCTIONS and instruction.name != "TICK"
            for target in instruction.targets_copy()
            if target.is_qubit_target
        }
    )
    remap = {old: new for new, old in enumerate(used_qubits)}
    lines = _normalized_lines(stim_circuit, remap)
    source = "\n".join(lines) + "\n"
    return _expand_measure_resets(source) if representation == "m-plus-r" else source


def _normalized_lines(stim_circuit: Any, remap: dict[int, int], indent: str = "") -> list[str]:
    lines: list[str] = []
    for instruction in stim_circuit:
        if instruction.name == "REPEAT":
            lines.append(f"{indent}REPEAT {instruction.repeat_count} {{")
            lines.extend(_normalized_lines(instruction.body_copy(), remap, indent + "    "))
            lines.append(f"{indent}}}")
            continue
        if instruction.name in _SKIP_INSTRUCTIONS:
            continue
        if instruction.name == "TICK":
            lines.append(f"{indent}TICK")
            continue
        targets = [remap[target.value] for target in instruction.targets_copy()]
        if not targets:
            continue
        args = instruction.gate_args_copy()
        arg_text = "" if not args else f"({','.join(format(value, '.17g') for value in args)})"
        lines.append(
            f"{indent}{instruction.name}{arg_text} " + " ".join(str(target) for target in targets)
        )
    return lines


def _iter_operation_kinds(operations: Any) -> list[str]:
    kinds: list[str] = []
    for operation in operations:
        kinds.append(operation.kind)
        if operation.kind == "repeat":
            kinds.extend(_iter_operation_kinds(operation.body))
    return kinds


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
