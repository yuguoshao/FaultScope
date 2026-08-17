"""Import a practical subset of Stim text circuits."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from faultscope.core import Circuit, NoiseLocation, Operation
from faultscope.dem import Detector, LogicalObservable
from faultscope.core import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)


class StimImportError(ValueError):
    """Raised when a Stim statement is outside the supported import subset."""


@dataclass(frozen=True)
class StimImportResult:
    circuit: Circuit
    detectors: tuple[Detector, ...]
    observables: tuple[LogicalObservable, ...]
    measurement_keys: tuple[str, ...]


_INSTRUCTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?(?:\s+(.*))?$")
_REC_RE = re.compile(r"^rec\[(-[0-9]+)\]$")
_MPP_TARGET_RE = re.compile(r"^([XYZ])([0-9]+)$")
_REPEAT_RE = re.compile(r"^REPEAT\s+([0-9]+)\s*\{$", re.IGNORECASE)

_MAX_RECOVERED_TICK_PERIOD = 64
_RECOVERABLE_TICK_INSTRUCTIONS = frozenset(
    {
        "TICK",
        "H",
        "S",
        "S_DAG",
        "SQRT_Z_DAG",
        "X",
        "Y",
        "Z",
        "CX",
        "CNOT",
        "CZ",
        "SWAP",
        "R",
        "RX",
        "RY",
        "MR",
        "MRX",
        "MRY",
        "M",
        "MX",
        "MY",
        "MPP",
    }
)
_RECOVERED_CLIFFORD_INSTRUCTIONS = frozenset(
    {"H", "S", "S_DAG", "SQRT_Z_DAG", "CX", "CNOT", "CZ", "SWAP"}
)


def load_stim_file(path: str | Path) -> StimImportResult:
    """Parse a supported Stim circuit from a UTF-8 text file."""

    return parse_stim_circuit(Path(path).read_text())


def parse_stim_circuit(text: str) -> StimImportResult:
    """Parse supported Stim syntax into a FaultScope circuit and declarations."""

    importer = _StructuredStimImporter()
    try:
        return importer.parse(text)
    except StimImportError:
        raise
    except (TypeError, ValueError) as exc:
        raise StimImportError(f"invalid Stim circuit: {exc}") from exc


@dataclass(frozen=True)
class _InstructionNode:
    name: str
    args: tuple[float, ...]
    targets: tuple[str, ...]
    line_no: int


@dataclass(frozen=True)
class _RepeatNode:
    count: int
    body: tuple["_StimNode", ...]
    line_no: int


_StimNode = _InstructionNode | _RepeatNode
_InstructionSignature = tuple[str, tuple[float, ...], tuple[str, ...]]


@dataclass(frozen=True)
class _TickRepeatCandidate:
    start_segment: int
    period: int
    repetitions: int
    saved_node_count: int


class _StructuredStimImporter:
    """Recursive Stim importer with compact repeat preservation."""

    def __init__(self) -> None:
        self.max_qubit = -1
        self.noise_index = 0
        self.measurement_keys: list[str] = []
        self.detectors: list[Detector] = []
        self.observables_by_id: dict[int, list[str]] = {}
        self.coordinate_shift: list[float] = []

    def parse(self, text: str) -> StimImportResult:
        nodes = _parse_stim_nodes(text)
        nodes = _recover_exact_tick_repeat(nodes)
        operations = self._build_nodes(nodes)
        self._scan_nodes(nodes)
        observables = tuple(
            LogicalObservable(id=observable_id, measurement_keys=tuple(keys))
            for observable_id, keys in sorted(self.observables_by_id.items())
        )
        return StimImportResult(
            circuit=Circuit(
                n_qubits=self.max_qubit + 1 if self.max_qubit >= 0 else 0,
                operations=operations,
            ),
            detectors=tuple(self.detectors),
            observables=observables,
            measurement_keys=tuple(self.measurement_keys),
        )

    def _build_nodes(self, nodes: Sequence[_StimNode]) -> list[Operation]:
        operations: list[Operation] = []
        for node in nodes:
            if isinstance(node, _RepeatNode):
                operations.append(Operation.repeat(node.count, self._build_nodes(node.body)))
            else:
                operations.extend(self._build_instruction(node))
        return operations

    def _build_instruction(self, node: _InstructionNode) -> list[Operation]:
        name = node.name
        args = node.args
        targets = node.targets
        line_no = node.line_no
        if name == "TICK":
            _require_no_args(args, name, line_no)
            _require_no_targets(targets, name, line_no)
            return [Operation.tick()]
        if name == "SHIFT_COORDS":
            _require_no_targets(targets, name, line_no)
            return [Operation.shift_coords(args)]
        if name == "QUBIT_COORDS":
            self._record_qubits(_parse_qubit_targets(targets, line_no))
            return []
        if name in {"H", "S", "S_DAG", "SQRT_Z_DAG", "X", "Y", "Z"}:
            _require_no_args(args, name, line_no)
            out: list[Operation] = []
            for qubit in _parse_qubit_targets(targets, line_no):
                if name == "H":
                    out.append(Operation.h(qubit))
                elif name == "S":
                    out.append(Operation.s(qubit))
                elif name in {"S_DAG", "SQRT_Z_DAG"}:
                    out.append(Operation.s_dag(qubit))
                elif name == "X":
                    out.append(Operation.x(qubit))
                elif name == "Y":
                    out.append(Operation.y(qubit))
                else:
                    out.append(Operation.z(qubit))
                self._record_qubits((qubit,))
            return out
        if name in {"CX", "CNOT", "CZ", "SWAP"}:
            _require_no_args(args, name, line_no)
            qubits = _parse_qubit_targets(targets, line_no)
            if len(qubits) % 2:
                raise StimImportError(
                    f"{name} requires an even number of targets on line {line_no}"
                )
            out = []
            for left, right in zip(qubits[::2], qubits[1::2]):
                if name in {"CX", "CNOT"}:
                    out.append(Operation.cx(left, right))
                elif name == "CZ":
                    out.append(Operation.cz(left, right))
                else:
                    out.append(Operation.swap(left, right))
                self._record_qubits((left, right))
            return out
        if name in {"R", "RX", "RY"}:
            _require_no_args(args, name, line_no)
            basis = {"R": "Z", "RX": "X", "RY": "Y"}[name]
            qubits = _parse_qubit_targets(targets, line_no)
            self._record_qubits(qubits)
            return [Operation.reset(qubit, basis=basis) for qubit in qubits]
        if name in {"MR", "MRX", "MRY"}:
            rate = _optional_single_arg(args, name, line_no)
            if rate not in {None, 0.0}:
                raise StimImportError(
                    f"{name} measurement noise is not supported on line {line_no}"
                )
            basis = {"MR": "Z", "MRX": "X", "MRY": "Y"}[name]
            qubits = _parse_qubit_targets(targets, line_no)
            self._record_qubits(qubits)
            return [Operation.measure_reset(qubit, basis=basis) for qubit in qubits]
        if name in {"M", "MX", "MY"}:
            basis = {"M": "Z", "MX": "X", "MY": "Y"}[name]
            rate = _optional_single_arg(args, name, line_no)
            out = []
            for qubit in _parse_qubit_targets(targets, line_no):
                noise = None
                if rate is not None and rate > 0:
                    noise = self._make_noise_location(
                        name=name,
                        line_no=line_no,
                        model=MeasurementBitFlip(),
                        rate=rate,
                        qubits=(qubit,),
                        operation="measurement_noise",
                    )
                out.append(Operation.measure(qubit, basis=basis, noise=noise))
                self._record_qubits((qubit,))
            return out
        if name == "MPP":
            rate = _optional_single_arg(args, name, line_no)
            out = []
            for product in _split_mpp_products(targets, line_no):
                product_qubits: list[int] = []
                paulis: list[str] = []
                for factor in product:
                    match = _MPP_TARGET_RE.match(factor)
                    if not match:
                        raise StimImportError(
                            f"unsupported MPP target {factor!r} on line {line_no}"
                        )
                    paulis.append(match.group(1))
                    product_qubits.append(int(match.group(2)))
                noise = None
                if rate is not None and rate > 0:
                    noise = self._make_noise_location(
                        name=name,
                        line_no=line_no,
                        model=MeasurementBitFlip(),
                        rate=rate,
                        qubits=tuple(product_qubits[:1]),
                        operation="measurement_noise",
                    )
                out.append(
                    Operation.measure_pauli(tuple(product_qubits), "".join(paulis), noise=noise)
                )
                self._record_qubits(product_qubits)
            return out
        if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
            rate = _required_single_arg(args, name, line_no)
            out = []
            for qubit in _parse_qubit_targets(targets, line_no):
                out.append(
                    Operation.noise(
                        self._make_noise_location(
                            name=name,
                            line_no=line_no,
                            model=BernoulliPauliNoise(name[0]),
                            rate=rate,
                            qubits=(qubit,),
                            operation="pauli_noise",
                        )
                    )
                )
                self._record_qubits((qubit,))
            return out
        if name == "DEPOLARIZE1":
            return self._build_single_noise(
                node,
                SingleQubitDepolarizing(),
                "depolarize1",
            )
        if name == "DEPOLARIZE2":
            return self._build_pair_noise(
                node,
                TwoQubitDepolarizing(),
                "depolarize2",
            )
        if name in {"PAULI_CHANNEL_1", "PAULI_CHANNEL_2"}:
            return self._build_pauli_channel(node)
        if name == "DETECTOR":
            return [
                Operation.detector_rec(
                    _parse_rec_lookbacks(targets, line_no),
                    coords=args,
                )
            ]
        if name == "OBSERVABLE_INCLUDE":
            observable_id = _observable_id(args, line_no)
            return [
                Operation.observable_include_rec(
                    observable_id,
                    _parse_rec_lookbacks(targets, line_no),
                )
            ]
        raise StimImportError(f"unsupported Stim instruction {name!r} on line {line_no}")

    def _build_single_noise(
        self,
        node: _InstructionNode,
        model: object,
        operation: str,
    ) -> list[Operation]:
        rate = _required_single_arg(node.args, node.name, node.line_no)
        out = []
        for qubit in _parse_qubit_targets(node.targets, node.line_no):
            out.append(
                Operation.noise(
                    self._make_noise_location(
                        name=node.name,
                        line_no=node.line_no,
                        model=model,
                        rate=rate,
                        qubits=(qubit,),
                        operation=operation,
                    )
                )
            )
            self._record_qubits((qubit,))
        return out

    def _build_pair_noise(
        self,
        node: _InstructionNode,
        model: object,
        operation: str,
    ) -> list[Operation]:
        rate = _required_single_arg(node.args, node.name, node.line_no)
        qubits = _parse_qubit_targets(node.targets, node.line_no)
        if len(qubits) % 2:
            raise StimImportError(f"{node.name} requires target pairs on line {node.line_no}")
        out = []
        for left, right in zip(qubits[::2], qubits[1::2]):
            out.append(
                Operation.noise(
                    self._make_noise_location(
                        name=node.name,
                        line_no=node.line_no,
                        model=model,
                        rate=rate,
                        qubits=(left, right),
                        operation=operation,
                    )
                )
            )
            self._record_qubits((left, right))
        return out

    def _build_pauli_channel(self, node: _InstructionNode) -> list[Operation]:
        events = (
            ("X", "Y", "Z")
            if node.name == "PAULI_CHANNEL_1"
            else (
                "IX",
                "IY",
                "IZ",
                "XI",
                "XX",
                "XY",
                "XZ",
                "YI",
                "YX",
                "YY",
                "YZ",
                "ZI",
                "ZX",
                "ZY",
                "ZZ",
            )
        )
        if len(node.args) != len(events):
            raise StimImportError(f"{node.name} requires {len(events)} args on line {node.line_no}")
        for probability in node.args:
            _validate_probability(probability, node.name, node.line_no)
        if sum(node.args) > 1.0:
            raise StimImportError(
                f"{node.name} probabilities must sum to at most 1 on line {node.line_no}"
            )
        weights = {
            event: probability for event, probability in zip(events, node.args) if probability > 0
        }
        rate = sum(node.args)
        qubits = _parse_qubit_targets(node.targets, node.line_no)
        group = 1 if node.name == "PAULI_CHANNEL_1" else 2
        if len(qubits) % group:
            raise StimImportError(
                f"{node.name} requires target groups of {group} on line {node.line_no}"
            )
        self._record_qubits(qubits)
        if rate <= 0:
            return []
        out = []
        for start in range(0, len(qubits), group):
            local = tuple(qubits[start : start + group])
            out.append(
                Operation.noise(
                    self._make_noise_location(
                        name=node.name,
                        line_no=node.line_no,
                        model=PauliChannel(weights),
                        rate=rate,
                        qubits=local,
                        operation=node.name.lower(),
                    )
                )
            )
        return out

    def _scan_nodes(self, nodes: Sequence[_StimNode]) -> None:
        for node in nodes:
            if isinstance(node, _RepeatNode):
                for _ in range(node.count):
                    self._scan_nodes(node.body)
                continue
            name = node.name
            if name == "SHIFT_COORDS":
                _add_coordinate_shift(self.coordinate_shift, node.args)
            elif name in {"M", "MX", "MY", "MR", "MRX", "MRY"}:
                for _ in node.targets:
                    self._next_measurement_key()
            elif name == "MPP":
                for _ in _split_mpp_products(node.targets, node.line_no):
                    self._next_measurement_key()
            elif name == "DETECTOR":
                keys = self._keys_for_lookbacks(_parse_rec_lookbacks(node.targets, node.line_no))
                detector = Detector(
                    id=len(self.detectors),
                    measurement_keys=tuple(keys),
                    coords=tuple(_shifted_coordinates(node.args, self.coordinate_shift)),
                )
                self.detectors.append(detector)
            elif name == "OBSERVABLE_INCLUDE":
                observable_id = _observable_id(node.args, node.line_no)
                keys = self._keys_for_lookbacks(_parse_rec_lookbacks(node.targets, node.line_no))
                self.observables_by_id.setdefault(observable_id, []).extend(keys)

    def _next_measurement_key(self) -> str:
        key = f"m{len(self.measurement_keys)}"
        self.measurement_keys.append(key)
        return key

    def _keys_for_lookbacks(self, lookbacks: Sequence[int]) -> list[str]:
        keys: list[str] = []
        for lookback in lookbacks:
            if lookback > len(self.measurement_keys):
                raise StimImportError(f"measurement record target rec[-{lookback}] out of range")
            keys.append(self.measurement_keys[-lookback])
        return keys

    def _make_noise_location(
        self,
        *,
        name: str,
        line_no: int,
        model: object,
        rate: float,
        qubits: tuple[int, ...],
        operation: str,
    ) -> NoiseLocation:
        location = NoiseLocation(
            id=f"stim_l{line_no}_{self.noise_index}_{name.lower()}",
            model=model,
            rate=rate,
            qubits=qubits,
            tags={
                "source": "stim",
                "stim_gate": name,
                "line": line_no,
                "gate": name.lower(),
                "operation": operation,
            },
        )
        self.noise_index += 1
        return location

    def _record_qubits(self, qubits: Sequence[int]) -> None:
        if qubits:
            self.max_qubit = max(self.max_qubit, max(qubits))


def _parse_stim_nodes(text: str) -> tuple[_StimNode, ...]:
    root: list[_StimNode] = []
    stack: list[tuple[list[_StimNode], int, int]] = []
    current = root
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line == "}":
            if not stack:
                raise StimImportError(f"unexpected closing brace on line {line_no}")
            body = tuple(current)
            parent, count, repeat_line = stack.pop()
            parent.append(_RepeatNode(count=count, body=body, line_no=repeat_line))
            current = parent
            continue
        repeat_match = _REPEAT_RE.match(line)
        if repeat_match:
            count = int(repeat_match.group(1))
            if count <= 0:
                raise StimImportError(f"REPEAT count must be positive on line {line_no}")
            stack.append((current, count, line_no))
            current = []
            continue
        if "{" in line or "}" in line:
            raise StimImportError(f"could not parse Stim block syntax on line {line_no}: {line!r}")
        match = _INSTRUCTION_RE.match(line)
        if not match:
            raise StimImportError(f"could not parse Stim line {line_no}: {line!r}")
        current.append(
            _InstructionNode(
                name=match.group(1).upper(),
                args=_parse_args(match.group(2), line_no),
                targets=tuple(match.group(3).split()) if match.group(3) else (),
                line_no=line_no,
            )
        )
    if stack:
        _, _, repeat_line = stack[-1]
        raise StimImportError(f"unterminated REPEAT block starting on line {repeat_line}")
    return tuple(root)


def _recover_exact_tick_repeat(nodes: tuple[_StimNode, ...]) -> tuple[_StimNode, ...]:
    """Recover one exact, side-effect-free top-level TICK cycle."""

    if any(isinstance(node, _RepeatNode) for node in nodes):
        return nodes
    instructions = tuple(node for node in nodes if isinstance(node, _InstructionNode))
    if len(instructions) != len(nodes):
        return nodes

    tick_indexes = [index for index, node in enumerate(instructions) if node.name == "TICK"]
    if len(tick_indexes) < 2:
        return nodes
    boundaries = tick_indexes + [len(instructions)]
    segment_signatures = [
        tuple(
            _instruction_signature(node)
            for node in instructions[boundaries[index] : boundaries[index + 1]]
        )
        for index in range(len(tick_indexes))
    ]
    signature_ids: dict[tuple[_InstructionSignature, ...], int] = {}
    segment_ids = [
        signature_ids.setdefault(signature, len(signature_ids)) for signature in segment_signatures
    ]

    candidates: list[_TickRepeatCandidate] = []
    maximum_period = min(_MAX_RECOVERED_TICK_PERIOD, len(segment_ids) // 2)
    for period in range(1, maximum_period + 1):
        run_start: int | None = None
        for index in range(period, len(segment_ids) + 1):
            matches_previous = (
                index < len(segment_ids) and segment_ids[index] == segment_ids[index - period]
            )
            if matches_previous:
                if run_start is None:
                    run_start = index
                continue
            if run_start is None:
                continue
            matched_segments = index - run_start
            repetitions = (matched_segments + period) // period
            start_segment = run_start - period
            candidate = _make_tick_repeat_candidate(
                instructions,
                boundaries,
                start_segment=start_segment,
                period=period,
                repetitions=repetitions,
            )
            if candidate is not None:
                candidates.append(candidate)
            run_start = None

    candidates.sort(
        key=lambda candidate: (
            -candidate.saved_node_count,
            candidate.period,
            candidate.start_segment,
        )
    )
    candidate = next(
        (
            candidate
            for candidate in candidates
            if _tick_repeat_iterations_match(instructions, boundaries, candidate)
        ),
        None,
    )
    if candidate is None:
        return nodes

    body_start = boundaries[candidate.start_segment]
    body_end = boundaries[candidate.start_segment + candidate.period]
    repeated_end = boundaries[candidate.start_segment + candidate.period * candidate.repetitions]
    body = tuple(instructions[body_start:body_end])
    recovered = _RepeatNode(
        count=candidate.repetitions - 1,
        body=body,
        line_no=instructions[body_start].line_no,
    )
    return tuple(instructions[:body_end]) + (recovered,) + tuple(instructions[repeated_end:])


def _make_tick_repeat_candidate(
    instructions: tuple[_InstructionNode, ...],
    boundaries: list[int],
    *,
    start_segment: int,
    period: int,
    repetitions: int,
) -> _TickRepeatCandidate | None:
    if repetitions < 2:
        return None
    body_start = boundaries[start_segment]
    body_end = boundaries[start_segment + period]
    body = instructions[body_start:body_end]
    if not _is_recoverable_tick_body(body):
        return None

    recovered_repetitions = repetitions - 1
    has_clifford = any(node.name in _RECOVERED_CLIFFORD_INSTRUCTIONS for node in body)
    minimum_repetitions = 8 if has_clifford else 4
    if recovered_repetitions < minimum_repetitions:
        return None

    saved_node_count = (repetitions - 2) * len(body) - 1
    if saved_node_count <= 0:
        return None
    return _TickRepeatCandidate(
        start_segment=start_segment,
        period=period,
        repetitions=repetitions,
        saved_node_count=saved_node_count,
    )


def _is_recoverable_tick_body(body: Sequence[_InstructionNode]) -> bool:
    if not body or body[0].name != "TICK":
        return False
    has_quantum_instruction = False
    for node in body:
        if node.name not in _RECOVERABLE_TICK_INSTRUCTIONS or node.args:
            return False
        if node.name == "TICK":
            if node.targets:
                return False
        else:
            has_quantum_instruction = True
    return has_quantum_instruction


def _tick_repeat_iterations_match(
    instructions: tuple[_InstructionNode, ...],
    boundaries: list[int],
    candidate: _TickRepeatCandidate,
) -> bool:
    body_start = boundaries[candidate.start_segment]
    body_end = boundaries[candidate.start_segment + candidate.period]
    reference = instructions[body_start:body_end]
    for repetition in range(1, candidate.repetitions):
        start_segment = candidate.start_segment + repetition * candidate.period
        iteration = instructions[
            boundaries[start_segment] : boundaries[start_segment + candidate.period]
        ]
        if len(iteration) != len(reference) or any(
            _instruction_signature(left) != _instruction_signature(right)
            for left, right in zip(reference, iteration)
        ):
            return False
    return True


def _instruction_signature(node: _InstructionNode) -> _InstructionSignature:
    return (node.name, node.args, node.targets)


def _parse_rec_lookbacks(targets: Sequence[str], line_no: int) -> tuple[int, ...]:
    lookbacks: list[int] = []
    for target in targets:
        match = _REC_RE.match(target)
        if not match:
            raise StimImportError(
                f"only rec[-k] targets are supported here, got {target!r} on line {line_no}"
            )
        lookbacks.append(-int(match.group(1)))
    return tuple(lookbacks)


def _observable_id(args: tuple[float, ...], line_no: int) -> int:
    if len(args) != 1 or int(args[0]) != args[0] or args[0] < 0:
        raise StimImportError(
            f"OBSERVABLE_INCLUDE requires one non-negative integer arg on line {line_no}"
        )
    return int(args[0])


def _add_coordinate_shift(target: list[float], offsets: Sequence[float]) -> None:
    if len(target) < len(offsets):
        target.extend(0.0 for _ in range(len(offsets) - len(target)))
    for index, offset in enumerate(offsets):
        target[index] += offset


def _shifted_coordinates(coords: Sequence[float], shift: Sequence[float]) -> list[float]:
    result = list(coords)
    if len(result) < len(shift):
        result.extend(0.0 for _ in range(len(shift) - len(result)))
    for index, offset in enumerate(shift):
        result[index] += offset
    return result


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _parse_args(raw: str | None, line_no: int) -> tuple[float, ...]:
    if raw is None:
        return ()
    parsed: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            parsed.append(0.0)
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise StimImportError(f"invalid numeric argument {token!r} on line {line_no}") from exc
        if not math.isfinite(value):
            raise StimImportError(f"numeric arguments must be finite on line {line_no}")
        parsed.append(value)
    return tuple(parsed)


def _required_single_arg(args: tuple[float, ...], name: str, line_no: int) -> float:
    if len(args) != 1:
        raise StimImportError(f"{name} requires exactly one arg on line {line_no}")
    return _validate_probability(args[0], name, line_no)


def _require_no_args(args: Sequence[float], name: str, line_no: int) -> None:
    if args:
        raise StimImportError(f"{name} does not accept arguments on line {line_no}")


def _require_no_targets(targets: Sequence[str], name: str, line_no: int) -> None:
    if targets:
        raise StimImportError(f"{name} does not accept targets on line {line_no}")


def _optional_single_arg(
    args: tuple[float, ...],
    name: str,
    line_no: int,
) -> float | None:
    if not args:
        return None
    if len(args) != 1:
        raise StimImportError(f"{name} accepts at most one arg on line {line_no}")
    return _validate_probability(args[0], name, line_no)


def _validate_probability(value: float, name: str, line_no: int) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise StimImportError(f"{name} probability must be in [0, 1] on line {line_no}")
    return value


def _parse_qubit_targets(targets: Sequence[str], line_no: int) -> tuple[int, ...]:
    qubits: list[int] = []
    for target in targets:
        try:
            qubit = int(target)
        except ValueError as exc:
            raise StimImportError(
                f"only integer qubit targets are supported, got {target!r} on line {line_no}"
            ) from exc
        if qubit < 0:
            raise StimImportError(f"negative qubit target {qubit} on line {line_no}")
        qubits.append(qubit)
    return tuple(qubits)


def _split_mpp_products(targets: Sequence[str], line_no: int) -> list[list[str]]:
    pieces: list[str] = []
    for target in targets:
        start = 0
        for index, char in enumerate(target):
            if char != "*":
                continue
            if index > start:
                pieces.append(target[start:index])
            pieces.append("*")
            start = index + 1
        if start < len(target):
            pieces.append(target[start:])

    products: list[list[str]] = []
    current: list[str] = []
    after_combiner = False
    for piece in pieces:
        if piece == "*":
            if not current or after_combiner:
                raise StimImportError(f"invalid MPP combiner on line {line_no}")
            after_combiner = True
            continue
        if not _MPP_TARGET_RE.match(piece):
            raise StimImportError(f"unsupported MPP target {piece!r} on line {line_no}")
        if current and not after_combiner:
            products.append(current)
            current = []
        current.append(piece)
        after_combiner = False
    if after_combiner:
        raise StimImportError(f"invalid trailing MPP combiner on line {line_no}")
    if current:
        products.append(current)
    if not products:
        raise StimImportError(f"MPP requires at least one Pauli product on line {line_no}")
    return products
