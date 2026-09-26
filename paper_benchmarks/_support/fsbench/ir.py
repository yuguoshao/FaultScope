"""Version-neutral stabilizer-circuit IR and surface-code fixture generation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .constants import (
    DEFAULT_DISTANCES,
    RANDOM_CLIFFORD_GENERATOR_SEED,
    RANDOM_CLIFFORD_INSTANCES,
    RANDOM_CLIFFORD_WIDTHS,
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
)


V1_SUPPORTED_OPS = frozenset({"H", "CX", "M", "R"})
SUPPORTED_OPS = frozenset({"H", "S", "CX", "M", "R"})
NOISE_OPS = frozenset({"X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"})


def allowed_operations(schema: str) -> frozenset[str]:
    if schema == SCHEMA_V1:
        return V1_SUPPORTED_OPS
    return SUPPORTED_OPS | NOISE_OPS if schema == SCHEMA_V3 else SUPPORTED_OPS


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    targets: tuple[int, ...]
    measurement_index: int | None = None
    probability: float | None = None

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_OPS | NOISE_OPS:
            raise ValueError(f"unsupported fsbench operation: {self.name}")
        expected_targets = 2 if self.name in {"CX", "DEPOLARIZE2"} else 1
        if len(self.targets) != expected_targets:
            raise ValueError(f"{self.name} expects {expected_targets} target(s)")
        if min(self.targets) < 0:
            raise ValueError("qubit targets must be non-negative")
        if self.name == "M" and self.measurement_index is None:
            raise ValueError("M requires a measurement index")
        if self.name != "M" and self.measurement_index is not None:
            raise ValueError(f"{self.name} must not have a measurement index")
        if self.name in NOISE_OPS:
            if self.probability is None or not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
                raise ValueError("noise probability must be finite and in [0, 1]")
        elif self.probability is not None:
            raise ValueError("only noise operations accept a probability")

    def to_line(self) -> str:
        if self.name == "M":
            return f"M\t{self.targets[0]}\t{self.measurement_index}"
        if self.name in NOISE_OPS:
            return "\t".join((self.name, *(str(q) for q in self.targets), repr(self.probability)))
        return "\t".join((self.name, *(str(target) for target in self.targets)))


@dataclass(frozen=True, slots=True)
class Case:
    metadata_path: Path
    metadata: dict[str, Any]
    operations: tuple[Operation, ...]

    @property
    def sidecar_path(self) -> Path:
        return self.metadata_path.parent / self.metadata["sidecar_file"]


@dataclass(frozen=True, slots=True)
class CaseMetadata:
    """Lightweight case descriptor that does not materialize the operation stream."""

    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def sidecar_path(self) -> Path:
        return self.metadata_path.parent / self.metadata["sidecar_file"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_operation_line(line: str, *, line_number: int) -> Operation:
    parts = line.rstrip("\n").split("\t")
    name = parts[0]
    try:
        if name in NOISE_OPS and len(parts) == (4 if name == "DEPOLARIZE2" else 3):
            return Operation(name, tuple(int(q) for q in parts[1:-1]), probability=float(parts[-1]))
        if name == "CX" and len(parts) == 3:
            return Operation(name, (int(parts[1]), int(parts[2])))
        if name == "M" and len(parts) == 3:
            return Operation(name, (int(parts[1]),), int(parts[2]))
        if name in {"H", "S", "R"} and len(parts) == 2:
            return Operation(name, (int(parts[1]),))
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid integer target") from exc
    raise ValueError(f"line {line_number}: invalid fsbench operation {line!r}")


def read_operations(path: Path) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            operations.append(parse_operation_line(line, line_number=line_number))
    return tuple(operations)


def load_case(metadata_path: Path, *, verify_hash: bool = False) -> Case:
    metadata_path = metadata_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{metadata_path}: expected one of {SUPPORTED_SCHEMA_VERSIONS!r}, "
            f"found {metadata.get('schema')!r}"
        )
    ops_path = metadata_path.parent / metadata["operations_file"]
    if verify_hash and sha256_file(ops_path) != metadata["operations_sha256"]:
        raise ValueError(f"operation hash mismatch for {ops_path}")
    operations = read_operations(ops_path)
    validate_case_contents(metadata, operations)
    return Case(metadata_path=metadata_path, metadata=metadata, operations=operations)


def load_case_metadata(metadata_path: Path, *, verify_hash: bool = False) -> CaseMetadata:
    """Load and optionally hash-check a case without parsing millions of operations."""

    metadata_path = metadata_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{metadata_path}: expected one of {SUPPORTED_SCHEMA_VERSIONS!r}, "
            f"found {metadata.get('schema')!r}"
        )
    operations_path = metadata_path.parent / metadata["operations_file"]
    sidecar_path = metadata_path.parent / metadata["sidecar_file"]
    if verify_hash:
        if sha256_file(operations_path) != metadata["operations_sha256"]:
            raise ValueError(f"operation hash mismatch for {operations_path}")
        if sha256_file(sidecar_path) != metadata["sidecar_sha256"]:
            raise ValueError(f"sidecar hash mismatch for {sidecar_path}")
    return CaseMetadata(metadata_path=metadata_path, metadata=metadata)


def validate_case_contents(metadata: dict[str, Any], operations: Sequence[Operation]) -> None:
    schema = str(metadata.get("schema"))
    allowed_ops = allowed_operations(schema)
    counts = {name: 0 for name in sorted(allowed_ops)}
    measurement_indices: list[int] = []
    max_qubit = -1
    for operation in operations:
        if operation.name not in allowed_ops:
            raise ValueError(f"{schema} does not support operation {operation.name}")
        counts[operation.name] += 1
        max_qubit = max(max_qubit, *operation.targets)
        if operation.name == "M":
            assert operation.measurement_index is not None
            measurement_indices.append(operation.measurement_index)
    if measurement_indices != list(range(len(measurement_indices))):
        raise ValueError("measurement indices are not contiguous and ordered")
    if max_qubit + 1 != int(metadata["num_qubits"]):
        raise ValueError("qubit count does not match operation stream")
    if len(measurement_indices) != int(metadata["num_measurements"]):
        raise ValueError("measurement count does not match operation stream")
    expected_counts = {key: int(value) for key, value in metadata["operation_counts"].items()}
    if counts != expected_counts:
        raise ValueError(f"operation counts differ: expected {expected_counts}, got {counts}")


def case_metadata_paths(cases_dir: Path) -> list[Path]:
    index_path = cases_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"invalid case index schema in {index_path}")
    return [cases_dir / item for item in index["cases"]]


def _stim_quantum_instructions(circuit: Any) -> Iterator[Any]:
    for instruction in circuit.flattened():
        if instruction.name not in {
            "QUBIT_COORDS",
            "SHIFT_COORDS",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "TICK",
        }:
            yield instruction


def _used_qubits(circuit: Any) -> list[int]:
    used: set[int] = set()
    for instruction in _stim_quantum_instructions(circuit):
        for target in instruction.targets_copy():
            if target.is_qubit_target:
                used.add(target.value)
    return sorted(used)


def _absolute_records(instruction: Any, measurements_seen: int) -> list[int]:
    records: list[int] = []
    for target in instruction.targets_copy():
        if not target.is_measurement_record_target:
            raise ValueError(f"non-record target in {instruction.name}")
        absolute = measurements_seen + target.value
        if absolute < 0 or absolute >= measurements_seen:
            raise ValueError(f"invalid record lookback {target.value} in {instruction}")
        records.append(absolute)
    return records


def normalize_stim_surface_case(source: Any, *, distance: int) -> tuple[list[Operation], dict[str, Any]]:
    flattened = source.flattened()
    old_ids = _used_qubits(flattened)
    remap = {old: new for new, old in enumerate(old_ids)}
    operations: list[Operation] = []
    detectors: list[dict[str, Any]] = []
    observables: dict[str, list[int]] = {}
    measurement_index = 0

    for instruction in flattened:
        name = instruction.name
        if name in {"QUBIT_COORDS", "SHIFT_COORDS", "TICK"}:
            continue
        if name == "DETECTOR":
            detectors.append(
                {
                    "id": len(detectors),
                    "measurements": _absolute_records(instruction, measurement_index),
                    "coords": instruction.gate_args_copy(),
                }
            )
            continue
        if name == "OBSERVABLE_INCLUDE":
            observable_id = str(int(instruction.gate_args_copy()[0]))
            observables.setdefault(observable_id, []).extend(
                _absolute_records(instruction, measurement_index)
            )
            continue

        targets = [remap[target.value] for target in instruction.targets_copy() if target.is_qubit_target]
        if name == "H":
            operations.extend(Operation("H", (target,)) for target in targets)
        elif name in {"CX", "CNOT"}:
            if len(targets) % 2:
                raise ValueError("CX target list has odd length")
            operations.extend(
                Operation("CX", (targets[index], targets[index + 1]))
                for index in range(0, len(targets), 2)
            )
        elif name in NOISE_OPS:
            width = 2 if name == "DEPOLARIZE2" else 1
            probability = float(instruction.gate_args_copy()[0])
            if len(targets) % width:
                raise ValueError(f"invalid target count for {name}")
            operations.extend(
                Operation(name, tuple(targets[i:i + width]), probability=probability)
                for i in range(0, len(targets), width)
            )
        elif name == "M":
            for target in targets:
                operations.append(Operation("M", (target,), measurement_index))
                measurement_index += 1
        elif name == "R":
            operations.extend(Operation("R", (target,)) for target in targets)
        elif name == "MR":
            # Interleaving M/R preserves measurement order and semantics because
            # targets are distinct. It also lets backends with an MR primitive
            # perform a documented lowering without changing the neutral IR.
            for target in targets:
                operations.append(Operation("M", (target,), measurement_index))
                measurement_index += 1
                operations.append(Operation("R", (target,)))
        else:
            raise ValueError(f"surface-code generator emitted unsupported gate {name}")

    sidecar = {
        "schema": SCHEMA_VERSION,
        "case_id": f"rotated_memory_z_d{distance}_r{distance}",
        "detectors": detectors,
        "observables": [
            {"id": int(observable_id), "measurements": measurements}
            for observable_id, measurements in sorted(observables.items(), key=lambda item: int(item[0]))
        ],
        "qubit_remap": {str(old): new for old, new in remap.items()},
    }
    return operations, sidecar


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_case(
    out_dir: Path,
    *,
    case_id: str,
    family: str,
    distance: int | None,
    rounds: int | None,
    operations: Sequence[Operation],
    sidecar: dict[str, Any],
    generator: dict[str, Any],
    source_sha256: str,
    schema: str = SCHEMA_V1,
    parameters: dict[str, Any] | None = None,
    logical_operation_counts: dict[str, int] | None = None,
) -> Path:
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported case schema {schema!r}")
    if schema == SCHEMA_V1 and (distance is None or rounds is None):
        raise ValueError("fsbench-v1 cases require distance and rounds")
    out_dir.mkdir(parents=True, exist_ok=True)
    operations_path = out_dir / f"{case_id}.ops"
    sidecar_path = out_dir / f"{case_id}.sidecar.json"
    metadata_path = out_dir / f"{case_id}.json"
    grammar = (
        "H q | CX control target | M q measurement_index | R q"
        if schema == SCHEMA_V1
        else "H q | S q | CX control target | M q measurement_index | R q"
    )
    if schema == SCHEMA_V3:
        grammar += " | X_ERROR q p | DEPOLARIZE1 q p | DEPOLARIZE2 q1 q2 p"
    operations_path.write_text(
        f"# {schema}: {grammar}\n"
        + "".join(operation.to_line() + "\n" for operation in operations),
        encoding="utf-8",
    )
    _write_json(sidecar_path, sidecar)

    allowed_ops = allowed_operations(schema)
    counts = {name: 0 for name in sorted(allowed_ops)}
    max_qubit = -1
    for operation in operations:
        if operation.name not in allowed_ops:
            raise ValueError(f"{schema} does not support operation {operation.name}")
        counts[operation.name] += 1
        max_qubit = max(max_qubit, *operation.targets)
    metadata = {
        "schema": schema,
        "case_id": case_id,
        "family": family,
        "num_qubits": max_qubit + 1,
        "num_measurements": counts["M"],
        "two_qubit_operations": counts["CX"],
        "operation_counts": counts,
        "operations_file": operations_path.name,
        "operations_sha256": sha256_file(operations_path),
        "sidecar_file": sidecar_path.name,
        "sidecar_sha256": sha256_file(sidecar_path),
        "source_sha256": source_sha256,
        "generator": generator,
    }
    if distance is not None:
        metadata["distance"] = int(distance)
    if rounds is not None:
        metadata["rounds"] = int(rounds)
    if parameters is not None:
        metadata["parameters"] = parameters
    if logical_operation_counts is not None:
        metadata["logical_operation_counts"] = {
            str(name): int(value)
            for name, value in sorted(logical_operation_counts.items())
        }
    validate_case_contents(metadata, operations)
    _write_json(metadata_path, metadata)
    return metadata_path


def generate_surface_cases(
    out_dir: Path,
    *,
    distances: Iterable[int] = DEFAULT_DISTANCES,
    noise_probability: float = 0.0,
) -> list[Path]:
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Stim 1.16.0 is required to generate benchmark cases") from exc
    if stim.__version__ != "1.16.0":
        raise RuntimeError(f"case generation requires Stim 1.16.0, found {stim.__version__}")

    if not math.isfinite(noise_probability) or not 0 <= noise_probability <= 1:
        raise ValueError("noise_probability must be finite and in [0, 1]")
    schema = SCHEMA_V3 if noise_probability else SCHEMA_V1
    noise_settings = {
        "after_clifford_depolarization": noise_probability,
        "after_reset_flip_probability": noise_probability,
        "before_measure_flip_probability": noise_probability,
        "before_round_data_depolarization": noise_probability,
    }
    metadata_paths: list[Path] = []
    for distance in distances:
        if distance < 2:
            raise ValueError("surface-code distance must be at least 2")
        source = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=distance,
            **noise_settings,
        )
        source_text = str(source.flattened()).encode("utf-8")
        operations, sidecar = normalize_stim_surface_case(source, distance=distance)
        case_id = f"rotated_memory_z_d{distance}_r{distance}"
        if noise_probability:
            case_id += f"_p{noise_probability:g}"
            sidecar.update(schema=schema, case_id=case_id)
        metadata_path = write_case(
            out_dir,
            case_id=case_id,
            family="surface_code:rotated_memory_z",
            distance=distance,
            rounds=distance,
            operations=operations,
            sidecar=sidecar,
            generator={"name": "stim", "version": stim.__version__},
            source_sha256=hashlib.sha256(source_text).hexdigest(),
            schema=schema,
            parameters={"noise_model": noise_settings} if noise_probability else None,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "num_qubits": 2 * distance**2 - 1,
            "num_measurements": distance**3 + distance**2 - distance,
            "two_qubit_operations": 4 * distance**2 * (distance - 1),
        }
        for field, value in expected.items():
            if metadata[field] != value:
                raise AssertionError(
                    f"{case_id}: {field}={metadata[field]}, expected {value}"
                )
        metadata_paths.append(metadata_path)

    index = {
        "schema": schema,
        "generator": {"name": "stim", "version": "1.16.0"},
        "cases": [path.name for path in metadata_paths],
    }
    _write_json(out_dir / "index.json", index)
    return metadata_paths


def random_clifford_case_id(*, width: int, depth: int, instance: int) -> str:
    return f"random_clifford_n{width}_d{depth}_i{instance}"


def random_clifford_seed(*, width: int, depth: int, instance: int) -> int:
    payload = (
        f"{RANDOM_CLIFFORD_GENERATOR_SEED}\0{width}\0{depth}\0{instance}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _lower_random_clifford_gate(
    operations: list[Operation], name: str, targets: tuple[int, ...]
) -> None:
    if name == "I" or name == "I2":
        return
    if name == "H":
        operations.append(Operation("H", targets))
    elif name == "S":
        operations.append(Operation("S", targets))
    elif name == "S_DAG":
        operations.extend(Operation("S", targets) for _ in range(3))
    elif name == "CX":
        operations.append(Operation("CX", targets))
    elif name == "CZ":
        control, target = targets
        operations.extend(
            (
                Operation("H", (target,)),
                Operation("CX", (control, target)),
                Operation("H", (target,)),
            )
        )
    elif name == "SWAP":
        left, right = targets
        operations.extend(
            (
                Operation("CX", (left, right)),
                Operation("CX", (right, left)),
                Operation("CX", (left, right)),
            )
        )
    else:  # pragma: no cover - generator choices are closed above
        raise ValueError(f"unsupported logical random-Clifford gate {name!r}")


def make_random_clifford_operations(
    *, width: int, depth: int, instance: int
) -> tuple[list[Operation], list[tuple[str, tuple[int, ...]]], dict[str, int], int]:
    if width <= 0:
        raise ValueError("random-Clifford width must be positive")
    if depth <= 0:
        raise ValueError("random-Clifford depth must be positive")
    if instance < 0:
        raise ValueError("random-Clifford instance must be non-negative")
    seed = random_clifford_seed(width=width, depth=depth, instance=instance)
    rng = random.Random(seed)
    logical: list[tuple[str, tuple[int, ...]]] = []
    logical_counts = {
        name: 0
        for name in ("H", "S", "S_DAG", "I", "CX", "CZ", "SWAP", "I2", "M")
    }
    operations: list[Operation] = []
    single_qubit_gates = ("H", "S", "S_DAG", "I")
    two_qubit_gates = ("CX", "CZ", "SWAP", "I2")

    for _layer in range(depth):
        for qubit in range(width):
            name = rng.choice(single_qubit_gates)
            targets = (qubit,)
            logical.append((name, targets))
            logical_counts[name] += 1
            _lower_random_clifford_gate(operations, name, targets)

        pairing = list(range(width))
        rng.shuffle(pairing)
        for left, right in zip(pairing[0::2], pairing[1::2]):
            name = rng.choice(two_qubit_gates)
            targets = (left, right)
            logical.append((name, targets))
            logical_counts[name] += 1
            _lower_random_clifford_gate(operations, name, targets)

    for measurement_index in range(width):
        logical.append(("M", (measurement_index,)))
        logical_counts["M"] += 1
        operations.append(Operation("M", (measurement_index,), measurement_index))
    return operations, logical, logical_counts, seed


def invert_unitary_operations(operations: Sequence[Operation]) -> list[Operation]:
    inverse: list[Operation] = []
    for operation in reversed(operations):
        if operation.name == "M":
            continue
        if operation.name in {"H", "CX"}:
            inverse.append(operation)
        elif operation.name == "S":
            inverse.extend(Operation("S", operation.targets) for _ in range(3))
        elif operation.name == "R":
            raise ValueError("reset has no unitary inverse")
        else:  # pragma: no cover - Operation validation is exhaustive
            raise ValueError(operation.name)
    return inverse


def _gf2_nullspace(equations: Sequence[int], *, variables: int) -> list[int]:
    pivots: dict[int, int] = {}
    for equation in equations:
        row = int(equation)
        while row:
            pivot = (row & -row).bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    free = [column for column in range(variables) if column not in pivots]
    basis: list[int] = []
    for free_column in free:
        vector = 1 << free_column
        for pivot in sorted(pivots, reverse=True):
            if (pivots[pivot] & vector).bit_count() & 1:
                vector |= 1 << pivot
        basis.append(vector)
    return basis


def measurement_parities_from_operations(
    operations: Sequence[Operation], *, num_qubits: int
) -> list[dict[str, Any]]:
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Stim 1.16.0 is required to generate parity constraints") from exc
    if stim.__version__ != "1.16.0":
        raise RuntimeError(f"parity generation requires Stim 1.16.0, found {stim.__version__}")
    lines: list[str] = []
    for operation in operations:
        if operation.name == "M":
            continue
        if operation.name in {"H", "S"}:
            lines.append(f"{operation.name} {operation.targets[0]}")
        elif operation.name == "CX":
            lines.append(f"CX {operation.targets[0]} {operation.targets[1]}")
        else:
            raise ValueError(f"random-Clifford parity generation found {operation.name}")
    tableau = stim.Tableau.from_circuit(stim.Circuit("\n".join(lines)))
    stabilizers = list(tableau.to_stabilizers(canonicalize=False))
    x_equations = [
        sum(
            1 << stabilizer_index
            for stabilizer_index, stabilizer in enumerate(stabilizers)
            if int(stabilizer[qubit]) in {1, 2}
        )
        for qubit in range(num_qubits)
    ]
    coefficient_basis = _gf2_nullspace(x_equations, variables=len(stabilizers))
    constraints: list[dict[str, Any]] = []
    for coefficients in coefficient_basis:
        product = stim.PauliString(num_qubits)
        for index, stabilizer in enumerate(stabilizers):
            if coefficients & (1 << index):
                product *= stabilizer
        paulis = [int(product[qubit]) for qubit in range(num_qubits)]
        if any(pauli not in {0, 3} for pauli in paulis):
            raise AssertionError("nullspace product is not Z-only")
        sign = complex(product.sign)
        if abs(sign - 1) < 1e-12:
            expected = 0
        elif abs(sign + 1) < 1e-12:
            expected = 1
        else:
            raise AssertionError(f"Z-only stabilizer has non-real sign {sign}")
        constraints.append(
            {
                "measurements": [
                    qubit for qubit, pauli in enumerate(paulis) if pauli == 3
                ],
                "expected": expected,
            }
        )
    return constraints


def generate_random_clifford_cases(
    out_dir: Path,
    *,
    widths: Iterable[int] = RANDOM_CLIFFORD_WIDTHS,
    depth: int | None = None,
    instances: Iterable[int] = RANDOM_CLIFFORD_INSTANCES,
) -> list[Path]:
    metadata_paths: list[Path] = []
    for width in widths:
        resolved_depth = int(width) if depth is None else int(depth)
        for instance in instances:
            operations, logical, logical_counts, seed = make_random_clifford_operations(
                width=int(width), depth=resolved_depth, instance=int(instance)
            )
            case_id = random_clifford_case_id(
                width=int(width), depth=resolved_depth, instance=int(instance)
            )
            logical_source = json.dumps(
                [[name, *targets] for name, targets in logical],
                separators=(",", ":"),
            ).encode("utf-8")
            constraints = measurement_parities_from_operations(
                operations, num_qubits=int(width)
            )
            sidecar = {
                "schema": SCHEMA_V2,
                "case_id": case_id,
                "measurement_parities": constraints,
                "support_dimension": int(width) - len(constraints),
            }
            metadata_paths.append(
                write_case(
                    out_dir,
                    case_id=case_id,
                    family="random_clifford",
                    distance=None,
                    rounds=None,
                    operations=operations,
                    sidecar=sidecar,
                    generator={
                        "name": "fsbench-random-clifford",
                        "version": "1",
                        "distribution": "layered-local-clifford",
                        "master_seed": RANDOM_CLIFFORD_GENERATOR_SEED,
                    },
                    source_sha256=hashlib.sha256(logical_source).hexdigest(),
                    schema=SCHEMA_V2,
                    parameters={
                        "width": int(width),
                        "depth": resolved_depth,
                        "instance": int(instance),
                        "circuit_seed": int(seed),
                    },
                    logical_operation_counts=logical_counts,
                )
            )
    index = {
        "schema": SCHEMA_V2,
        "generator": {
            "name": "fsbench-random-clifford",
            "version": "1",
            "master_seed": RANDOM_CLIFFORD_GENERATOR_SEED,
        },
        "cases": [path.name for path in metadata_paths],
    }
    _write_json(out_dir / "index.json", index)
    return metadata_paths
