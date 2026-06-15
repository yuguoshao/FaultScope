"""Import a practical subset of Stim text circuits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.dem import Detector, LogicalObservable
from npsim.noise import (
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


def load_stim_file(path: str | Path) -> StimImportResult:
    return parse_stim_circuit(Path(path).read_text())


def parse_stim_circuit(text: str) -> StimImportResult:
    importer = _StimImporter()
    return importer.parse(text)


class _StimImporter:
    def __init__(self) -> None:
        self.operations: list[Operation] = []
        self.detectors: list[Detector] = []
        self.observables_by_id: dict[int, list[str]] = {}
        self.measurement_keys: list[str] = []
        self.max_qubit = -1
        self.noise_index = 0

    def parse(self, text: str) -> StimImportResult:
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = _strip_comment(raw_line).strip()
            if not line:
                continue
            self._parse_line(line, line_no)

        observables = tuple(
            LogicalObservable(
                id=observable_id,
                measurement_keys=tuple(keys),
            )
            for observable_id, keys in sorted(self.observables_by_id.items())
        )
        return StimImportResult(
            circuit=Circuit(
                n_qubits=self.max_qubit + 1 if self.max_qubit >= 0 else 0,
                operations=self.operations,
            ),
            detectors=tuple(self.detectors),
            observables=observables,
            measurement_keys=tuple(self.measurement_keys),
        )

    def _parse_line(self, line: str, line_no: int) -> None:
        if "{" in line or "}" in line:
            raise StimImportError("REPEAT blocks are not supported by the subset importer")

        match = _INSTRUCTION_RE.match(line)
        if not match:
            raise StimImportError(f"could not parse Stim line {line_no}: {line!r}")
        name = match.group(1).upper()
        args = _parse_args(match.group(2))
        targets = match.group(3).split() if match.group(3) else []

        if name in {"TICK", "SHIFT_COORDS"}:
            return
        if name == "QUBIT_COORDS":
            self._record_qubits(_parse_qubit_targets(targets, line_no))
            return
        if name in {"H", "S", "S_DAG", "SQRT_Z_DAG", "X", "Y", "Z"}:
            self._parse_single_qubit_gate(name, targets, line_no)
            return
        if name in {"CX", "CNOT", "CZ", "SWAP"}:
            self._parse_two_qubit_gate(name, targets, line_no)
            return
        if name in {"R", "RX", "RY"}:
            self._parse_reset(name, targets, line_no)
            return
        if name in {"M", "MX", "MY"}:
            self._parse_measurement(name, args, targets, line_no)
            return
        if name == "MPP":
            self._parse_mpp(args, targets, line_no)
            return
        if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
            self._parse_bernoulli_pauli_noise(name, args, targets, line_no)
            return
        if name == "DEPOLARIZE1":
            self._parse_depolarize1(args, targets, line_no)
            return
        if name == "DEPOLARIZE2":
            self._parse_depolarize2(args, targets, line_no)
            return
        if name == "PAULI_CHANNEL_1":
            self._parse_pauli_channel1(args, targets, line_no)
            return
        if name == "PAULI_CHANNEL_2":
            self._parse_pauli_channel2(args, targets, line_no)
            return
        if name == "DETECTOR":
            self._parse_detector(args, targets, line_no)
            return
        if name == "OBSERVABLE_INCLUDE":
            self._parse_observable(args, targets, line_no)
            return

        raise StimImportError(f"unsupported Stim instruction {name!r} on line {line_no}")

    def _parse_single_qubit_gate(
        self,
        name: str,
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        for qubit in _parse_qubit_targets(targets, line_no):
            if name == "H":
                self.operations.append(Operation.h(qubit))
            elif name == "S":
                self.operations.append(Operation.s(qubit))
            elif name in {"S_DAG", "SQRT_Z_DAG"}:
                self.operations.append(Operation.s_dag(qubit))
            elif name == "X":
                self.operations.append(Operation.x(qubit))
            elif name == "Y":
                self.operations.append(Operation.y(qubit))
            elif name == "Z":
                self.operations.append(Operation.z(qubit))
            self._record_qubits((qubit,))

    def _parse_two_qubit_gate(
        self,
        name: str,
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        qubits = _parse_qubit_targets(targets, line_no)
        if len(qubits) % 2:
            raise StimImportError(f"{name} requires an even number of targets on line {line_no}")
        for left, right in zip(qubits[::2], qubits[1::2]):
            if name in {"CX", "CNOT"}:
                self.operations.append(Operation.cx(left, right))
            elif name == "CZ":
                self.operations.append(Operation.cz(left, right))
            elif name == "SWAP":
                self.operations.append(Operation.swap(left, right))
            self._record_qubits((left, right))

    def _parse_reset(self, name: str, targets: Sequence[str], line_no: int) -> None:
        basis = {"R": "Z", "RX": "X", "RY": "Y"}[name]
        for qubit in _parse_qubit_targets(targets, line_no):
            self.operations.append(Operation.reset(qubit, basis=basis))
            self._record_qubits((qubit,))

    def _parse_measurement(
        self,
        name: str,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        basis = {"M": "Z", "MX": "X", "MY": "Y"}[name]
        rate = _optional_single_arg(args, name, line_no)
        for qubit in _parse_qubit_targets(targets, line_no):
            key = self._next_measurement_key()
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
            self.operations.append(Operation.measure(qubit, key=key, basis=basis, noise=noise))
            self._record_qubits((qubit,))

    def _parse_mpp(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        rate = _optional_single_arg(args, "MPP", line_no)
        for product in _split_mpp_products(targets):
            qubits: list[int] = []
            paulis: list[str] = []
            for factor in product:
                match = _MPP_TARGET_RE.match(factor)
                if not match:
                    raise StimImportError(
                        f"unsupported MPP target {factor!r} on line {line_no}"
                    )
                paulis.append(match.group(1))
                qubits.append(int(match.group(2)))
            key = self._next_measurement_key()
            noise = None
            if rate is not None and rate > 0:
                noise = self._make_noise_location(
                    name="MPP",
                    line_no=line_no,
                    model=MeasurementBitFlip(),
                    rate=rate,
                    qubits=tuple(qubits[:1]),
                    operation="measurement_noise",
                )
            self.operations.append(
                Operation.measure_pauli(tuple(qubits), "".join(paulis), key=key, noise=noise)
            )
            self._record_qubits(tuple(qubits))

    def _parse_bernoulli_pauli_noise(
        self,
        name: str,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        rate = _required_single_arg(args, name, line_no)
        pauli = name[0]
        for qubit in _parse_qubit_targets(targets, line_no):
            location = self._make_noise_location(
                name=name,
                line_no=line_no,
                model=BernoulliPauliNoise(pauli),
                rate=rate,
                qubits=(qubit,),
                operation="pauli_noise",
            )
            self.operations.append(Operation.noise(location))
            self._record_qubits((qubit,))

    def _parse_depolarize1(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        rate = _required_single_arg(args, "DEPOLARIZE1", line_no)
        for qubit in _parse_qubit_targets(targets, line_no):
            location = self._make_noise_location(
                name="DEPOLARIZE1",
                line_no=line_no,
                model=SingleQubitDepolarizing(),
                rate=rate,
                qubits=(qubit,),
                operation="depolarize1",
            )
            self.operations.append(Operation.noise(location))
            self._record_qubits((qubit,))

    def _parse_depolarize2(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        rate = _required_single_arg(args, "DEPOLARIZE2", line_no)
        qubits = _parse_qubit_targets(targets, line_no)
        if len(qubits) % 2:
            raise StimImportError(f"DEPOLARIZE2 requires target pairs on line {line_no}")
        for left, right in zip(qubits[::2], qubits[1::2]):
            location = self._make_noise_location(
                name="DEPOLARIZE2",
                line_no=line_no,
                model=TwoQubitDepolarizing(),
                rate=rate,
                qubits=(left, right),
                operation="depolarize2",
            )
            self.operations.append(Operation.noise(location))
            self._record_qubits((left, right))

    def _parse_pauli_channel1(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        if len(args) != 3:
            raise StimImportError(f"PAULI_CHANNEL_1 requires 3 args on line {line_no}")
        rate = sum(args)
        weights = {
            pauli: probability
            for pauli, probability in zip(("X", "Y", "Z"), args)
            if probability > 0
        }
        for qubit in _parse_qubit_targets(targets, line_no):
            if rate <= 0:
                continue
            location = self._make_noise_location(
                name="PAULI_CHANNEL_1",
                line_no=line_no,
                model=PauliChannel(weights),
                rate=rate,
                qubits=(qubit,),
                operation="pauli_channel_1",
            )
            self.operations.append(Operation.noise(location))
            self._record_qubits((qubit,))

    def _parse_pauli_channel2(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        if len(args) != 15:
            raise StimImportError(f"PAULI_CHANNEL_2 requires 15 args on line {line_no}")
        events = (
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
        weights = {
            event: probability
            for event, probability in zip(events, args)
            if probability > 0
        }
        rate = sum(args)
        qubits = _parse_qubit_targets(targets, line_no)
        if len(qubits) % 2:
            raise StimImportError(f"PAULI_CHANNEL_2 requires target pairs on line {line_no}")
        if rate <= 0:
            self._record_qubits(qubits)
            return
        for left, right in zip(qubits[::2], qubits[1::2]):
            location = self._make_noise_location(
                name="PAULI_CHANNEL_2",
                line_no=line_no,
                model=PauliChannel(weights),
                rate=rate,
                qubits=(left, right),
                operation="pauli_channel_2",
            )
            self.operations.append(Operation.noise(location))
            self._record_qubits((left, right))

    def _parse_detector(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        keys = self._parse_rec_targets(targets, line_no)
        detector = Detector(
            id=len(self.detectors),
            measurement_keys=tuple(keys),
            coords=tuple(args),
        )
        self.detectors.append(detector)
        self.operations.append(
            Operation.detector(
                keys,
                detector_id=detector.id,
                coords=detector.coords,
            )
        )

    def _parse_observable(
        self,
        args: tuple[float, ...],
        targets: Sequence[str],
        line_no: int,
    ) -> None:
        if len(args) != 1 or int(args[0]) != args[0] or args[0] < 0:
            raise StimImportError(
                f"OBSERVABLE_INCLUDE requires one non-negative integer arg on line {line_no}"
            )
        observable_id = int(args[0])
        keys = self._parse_rec_targets(targets, line_no)
        self.observables_by_id.setdefault(observable_id, []).extend(keys)
        self.operations.append(Operation.observable_include(observable_id, keys))

    def _parse_rec_targets(self, targets: Sequence[str], line_no: int) -> list[str]:
        keys: list[str] = []
        for target in targets:
            match = _REC_RE.match(target)
            if not match:
                raise StimImportError(
                    f"only rec[-k] targets are supported here, got {target!r} on line {line_no}"
                )
            offset = int(match.group(1))
            index = len(self.measurement_keys) + offset
            if index < 0 or index >= len(self.measurement_keys):
                raise StimImportError(f"measurement record target {target!r} out of range")
            keys.append(self.measurement_keys[index])
        return keys

    def _next_measurement_key(self) -> str:
        key = f"m{len(self.measurement_keys)}"
        self.measurement_keys.append(key)
        return key

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


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _parse_args(raw: str | None) -> tuple[float, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _required_single_arg(args: tuple[float, ...], name: str, line_no: int) -> float:
    if len(args) != 1:
        raise StimImportError(f"{name} requires exactly one arg on line {line_no}")
    return args[0]


def _optional_single_arg(
    args: tuple[float, ...],
    name: str,
    line_no: int,
) -> float | None:
    if not args:
        return None
    if len(args) != 1:
        raise StimImportError(f"{name} accepts at most one arg on line {line_no}")
    return args[0]


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


def _split_mpp_products(targets: Sequence[str]) -> list[list[str]]:
    products: list[list[str]] = []
    for target in targets:
        factors = [factor for factor in target.split("*") if factor]
        if not factors:
            continue
        products.append(factors)
    return products
