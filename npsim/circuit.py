"""Circuit data structures for forward noise-aware stabilizer simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from npsim.noise import StochasticNoise


@dataclass(frozen=True)
class NoiseLocation:
    """A differentiable local noise-rate parameter."""

    id: str
    model: StochasticNoise
    rate: float
    qubits: tuple[int, ...]
    tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Operation:
    """One forward circuit operation."""

    kind: str
    qubits: tuple[int, ...] = ()
    key: str | None = None
    basis: str = "Z"
    pauli: str | None = None
    measurement_keys: tuple[str, ...] = ()
    observable_id: int | None = None
    noise_location: NoiseLocation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def h(qubit: int, **metadata: Any) -> "Operation":
        return Operation("h", (qubit,), metadata=metadata)

    @staticmethod
    def s(qubit: int, **metadata: Any) -> "Operation":
        return Operation("s", (qubit,), metadata=metadata)

    @staticmethod
    def s_dag(qubit: int, **metadata: Any) -> "Operation":
        return Operation("s_dag", (qubit,), metadata=metadata)

    @staticmethod
    def x(qubit: int, **metadata: Any) -> "Operation":
        return Operation.pauli_gate((qubit,), "X", **metadata)

    @staticmethod
    def y(qubit: int, **metadata: Any) -> "Operation":
        return Operation.pauli_gate((qubit,), "Y", **metadata)

    @staticmethod
    def z(qubit: int, **metadata: Any) -> "Operation":
        return Operation.pauli_gate((qubit,), "Z", **metadata)

    @staticmethod
    def cx(control: int, target: int, **metadata: Any) -> "Operation":
        return Operation("cx", (control, target), metadata=metadata)

    @staticmethod
    def cz(left: int, right: int, **metadata: Any) -> "Operation":
        return Operation("cz", (left, right), metadata=metadata)

    @staticmethod
    def swap(left: int, right: int, **metadata: Any) -> "Operation":
        return Operation("swap", (left, right), metadata=metadata)

    @staticmethod
    def pauli_gate(qubits: Sequence[int], pauli: str, **metadata: Any) -> "Operation":
        return Operation("pauli", tuple(qubits), pauli=pauli, metadata=metadata)

    @staticmethod
    def noise(location: NoiseLocation, **metadata: Any) -> "Operation":
        return Operation(
            "noise",
            tuple(location.qubits),
            noise_location=location,
            metadata=metadata,
        )

    @staticmethod
    def measure(
        qubit: int,
        *,
        key: str | None = None,
        basis: str = "Z",
        noise: NoiseLocation | None = None,
        **metadata: Any,
    ) -> "Operation":
        return Operation(
            "measure",
            (qubit,),
            key=key,
            basis=basis,
            noise_location=noise,
            metadata=metadata,
        )

    @staticmethod
    def measure_pauli(
        qubits: Sequence[int],
        pauli: str,
        *,
        key: str | None = None,
        noise: NoiseLocation | None = None,
        **metadata: Any,
    ) -> "Operation":
        return Operation(
            "measure_pauli",
            tuple(qubits),
            key=key,
            pauli=pauli,
            noise_location=noise,
            metadata=metadata,
        )

    @staticmethod
    def reset(
        qubit: int,
        *,
        key: str | None = None,
        basis: str = "Z",
        **metadata: Any,
    ) -> "Operation":
        return Operation("reset", (qubit,), key=key, basis=basis, metadata=metadata)

    @staticmethod
    def detector(
        measurement_keys: Sequence[str],
        *,
        detector_id: int | None = None,
        coords: Sequence[float] = (),
        **metadata: Any,
    ) -> "Operation":
        metadata = {
            **metadata,
            "detector_id": detector_id,
            "coords": tuple(float(coord) for coord in coords),
        }
        return Operation(
            "detector",
            measurement_keys=tuple(measurement_keys),
            metadata=metadata,
        )

    @staticmethod
    def observable_include(
        observable_id: int,
        measurement_keys: Sequence[str],
        **metadata: Any,
    ) -> "Operation":
        return Operation(
            "observable_include",
            observable_id=observable_id,
            measurement_keys=tuple(measurement_keys),
            metadata=metadata,
        )


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    operations: tuple[Operation, ...] | list[Operation]

    def noise_locations(self) -> dict[str, NoiseLocation]:
        locations: dict[str, NoiseLocation] = {}
        for operation in self.operations:
            if operation.kind == "noise" and operation.noise_location is not None:
                locations[operation.noise_location.id] = operation.noise_location
            elif (
                operation.kind in {"measure", "measure_pauli"}
                and operation.noise_location is not None
            ):
                locations[operation.noise_location.id] = operation.noise_location
        return locations
