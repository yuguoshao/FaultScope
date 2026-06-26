from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faultscope.core import Circuit, NoiseLocation, Operation
from faultscope.core import BernoulliPauliNoise, MeasurementBitFlip
from faultscope.dem import LogicalObservable


@dataclass(frozen=True)
class RotatedSurfaceCodeMemoryExample:
    distance: int
    rounds: int
    circuit: Circuit
    x_checks: tuple[dict[str, object], ...]
    z_checks: tuple[dict[str, object], ...]
    hot_x_data: tuple[int, int]
    hot_z_data: tuple[int, int]
    hot_x_check: str
    hot_z_check: str

    @property
    def data_qubits(self) -> int:
        return self.distance * self.distance

    @property
    def x_logical_qubits(self) -> tuple[int, ...]:
        return tuple(
            _data_index(self.distance, row, 0)
            for row in range(self.distance)
        )

    @property
    def z_logical_qubits(self) -> tuple[int, ...]:
        return tuple(
            _data_index(self.distance, 0, col)
            for col in range(self.distance)
        )

    @property
    def terminal_measurement_pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            (f"r{self.rounds}_{self.hot_x_check}", f"r0_{self.hot_x_check}"),
            (f"r{self.rounds}_{self.hot_z_check}", f"r0_{self.hot_z_check}"),
        )

    @property
    def observables(self) -> tuple[LogicalObservable, ...]:
        return (
            LogicalObservable(
                id=0,
                pauli_qubits=self.x_logical_qubits,
                pauli="Z" * self.distance,
            ),
            LogicalObservable(
                id=1,
                pauli_qubits=self.z_logical_qubits,
                pauli="X" * self.distance,
            ),
        )

    def make_decoder(self) -> "RotatedSurfaceCodeMemoryDecoder":
        np, pymatching, sparse = _load_matching_modules()
        return RotatedSurfaceCodeMemoryDecoder(
            distance=self.distance,
            rounds=self.rounds,
            x_checks=self.x_checks,
            z_checks=self.z_checks,
            x_logical_qubits=self.x_logical_qubits,
            z_logical_qubits=self.z_logical_qubits,
            matching_x=_make_matching(
                self.distance,
                self.x_checks,
                np,
                pymatching,
                sparse,
            ),
            matching_z=_make_matching(
                self.distance,
                self.z_checks,
                np,
                pymatching,
                sparse,
            ),
        )


@dataclass(frozen=True)
class RotatedSurfaceCodeMemoryDecoder:
    distance: int
    rounds: int
    x_checks: tuple[dict[str, object], ...]
    z_checks: tuple[dict[str, object], ...]
    x_logical_qubits: tuple[int, ...]
    z_logical_qubits: tuple[int, ...]
    matching_x: Any
    matching_z: Any

    def decode_batch_masks(self, batch: Any) -> dict[int, int]:
        x_syndromes = [
            [
                batch.measurement_bit(f"r{self.rounds}_{check['id']}", shot)
                ^ batch.measurement_bit(f"r0_{check['id']}", shot)
                for check in self.x_checks
            ]
            for shot in range(batch.shots)
        ]
        z_syndromes = [
            [
                batch.measurement_bit(f"r{self.rounds}_{check['id']}", shot)
                ^ batch.measurement_bit(f"r0_{check['id']}", shot)
                for check in self.z_checks
            ]
            for shot in range(batch.shots)
        ]
        z_corrections = self.matching_x.decode_batch(x_syndromes)
        x_corrections = self.matching_z.decode_batch(z_syndromes)

        x_correction_masks = [0] * (self.distance * self.distance)
        for shot, row in enumerate(_rows(x_corrections)):
            for qubit, bit in enumerate(row):
                if int(bit):
                    x_correction_masks[qubit] |= 1 << shot
        z_correction_masks = [0] * (self.distance * self.distance)
        for shot, row in enumerate(_rows(z_corrections)):
            for qubit, bit in enumerate(row):
                if int(bit):
                    z_correction_masks[qubit] |= 1 << shot

        x_prediction = 0
        for qubit in self.x_logical_qubits:
            x_prediction ^= x_correction_masks[qubit]
        z_prediction = 0
        for qubit in self.z_logical_qubits:
            z_prediction ^= z_correction_masks[qubit]
        return {0: x_prediction, 1: z_prediction}


def make_large_rotated_surface_code_memory_example(
    *,
    distance: int,
    rounds: int = 2,
) -> RotatedSurfaceCodeMemoryExample:
    if distance < 3 or distance % 2 != 1:
        raise ValueError("distance must be an odd integer >= 3")
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    x_checks, z_checks = _rotated_surface_code_checks(distance)
    hot_x_data = (distance // 2, 0)
    hot_z_data = (0, distance // 2)
    hot_x_check = _central_check_id(x_checks, distance)
    hot_z_check = _central_check_id(z_checks, distance)
    circuit = _make_memory_circuit(
        distance=distance,
        rounds=rounds,
        x_checks=x_checks,
        z_checks=z_checks,
        hot_x_data=hot_x_data,
        hot_z_data=hot_z_data,
        hot_x_check=hot_x_check,
        hot_z_check=hot_z_check,
    )
    return RotatedSurfaceCodeMemoryExample(
        distance=distance,
        rounds=rounds,
        circuit=circuit,
        x_checks=tuple(x_checks),
        z_checks=tuple(z_checks),
        hot_x_data=hot_x_data,
        hot_z_data=hot_z_data,
        hot_x_check=hot_x_check,
        hot_z_check=hot_z_check,
    )


def _make_memory_circuit(
    *,
    distance: int,
    rounds: int,
    x_checks: list[dict[str, object]],
    z_checks: list[dict[str, object]],
    hot_x_data: tuple[int, int],
    hot_z_data: tuple[int, int],
    hot_x_check: str,
    hot_z_check: str,
) -> Circuit:
    operations: list[Operation] = []
    _append_check_measurements(
        operations,
        distance=distance,
        round_idx=0,
        checks=x_checks,
        basis="X",
        noise=False,
        hot_check_id=hot_x_check,
    )
    _append_check_measurements(
        operations,
        distance=distance,
        round_idx=0,
        checks=z_checks,
        basis="Z",
        noise=False,
        hot_check_id=hot_z_check,
    )

    for round_idx in range(1, rounds + 1):
        for row in range(distance):
            for col in range(distance):
                qubit = _data_index(distance, row, col)
                operations.append(
                    Operation.noise(
                        NoiseLocation(
                            id=f"d{distance}_data_x_r{round_idx}_{row}_{col}",
                            model=BernoulliPauliNoise("X"),
                            rate=0.14 if (row, col) == hot_x_data else 0.025,
                            qubits=(qubit,),
                            tags={
                                "layout": "rotated_surface_code",
                                "role": "data",
                                "row": row,
                                "col": col,
                                "round": round_idx,
                                "basis": "X",
                                "operation": "data_x_noise",
                            },
                        )
                    )
                )
                operations.append(
                    Operation.noise(
                        NoiseLocation(
                            id=f"d{distance}_data_z_r{round_idx}_{row}_{col}",
                            model=BernoulliPauliNoise("Z"),
                            rate=0.13 if (row, col) == hot_z_data else 0.025,
                            qubits=(qubit,),
                            tags={
                                "layout": "rotated_surface_code",
                                "role": "data",
                                "row": row,
                                "col": col,
                                "round": round_idx,
                                "basis": "Z",
                                "operation": "data_z_noise",
                            },
                        )
                    )
                )
        _append_check_measurements(
            operations,
            distance=distance,
            round_idx=round_idx,
            checks=x_checks,
            basis="X",
            noise=True,
            hot_check_id=hot_x_check,
        )
        _append_check_measurements(
            operations,
            distance=distance,
            round_idx=round_idx,
            checks=z_checks,
            basis="Z",
            noise=True,
            hot_check_id=hot_z_check,
        )

    return Circuit(n_qubits=distance * distance, operations=operations)


def _append_check_measurements(
    operations: list[Operation],
    *,
    distance: int,
    round_idx: int,
    checks: list[dict[str, object]],
    basis: str,
    noise: bool,
    hot_check_id: str,
) -> None:
    for check in checks:
        check_id = str(check["id"])
        qubits = tuple(_data_index(distance, row, col) for row, col in check["data"])
        location = None
        if noise:
            role = "x_check" if basis == "X" else "z_check"
            location = NoiseLocation(
                id=f"d{distance}_meas_r{round_idx}_{check_id}",
                model=MeasurementBitFlip(),
                rate=0.10 if check_id == hot_check_id else 0.015,
                qubits=qubits[:1],
                tags={
                    "layout": "rotated_surface_code",
                    "role": role,
                    "x": float(check["x"]),
                    "y": float(check["y"]),
                    "round": round_idx,
                    "check": check_id,
                    "basis": basis,
                    "operation": "measurement_noise",
                },
            )
        operations.append(
            Operation.measure_pauli(
                qubits,
                basis * len(qubits),
                key=f"r{round_idx}_{check_id}",
                noise=location,
            )
        )


def _rotated_surface_code_checks(
    distance: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x_checks: list[dict[str, object]] = []
    z_checks: list[dict[str, object]] = []
    for row in range(distance - 1):
        for col in range(distance - 1):
            check = {
                "data": (
                    (row, col),
                    (row + 1, col),
                    (row, col + 1),
                    (row + 1, col + 1),
                ),
                "x": col + 0.5,
                "y": row + 0.5,
            }
            if (row + col) % 2:
                x_checks.append({"id": f"x_check_{row}_{col}", **check})
            else:
                z_checks.append({"id": f"z_check_{row}_{col}", **check})
    for col in range(0, distance - 1, 2):
        x_checks.append(
            {
                "id": f"x_check_top_{col}",
                "data": ((0, col), (0, col + 1)),
                "x": col + 0.5,
                "y": -0.5,
            }
        )
    for col in range(1, distance - 1, 2):
        x_checks.append(
            {
                "id": f"x_check_bottom_{col}",
                "data": (
                    (distance - 1, col),
                    (distance - 1, col + 1),
                ),
                "x": col + 0.5,
                "y": distance - 0.5,
            }
        )
    for row in range(1, distance - 1, 2):
        z_checks.append(
            {
                "id": f"z_check_left_{row}",
                "data": ((row, 0), (row + 1, 0)),
                "x": -0.5,
                "y": row + 0.5,
            }
        )
    for row in range(0, distance - 1, 2):
        z_checks.append(
            {
                "id": f"z_check_right_{row}",
                "data": (
                    (row, distance - 1),
                    (row + 1, distance - 1),
                ),
                "x": distance - 0.5,
                "y": row + 0.5,
            }
        )
    return x_checks, z_checks


def _central_check_id(checks: list[dict[str, object]], distance: int) -> str:
    center = (distance - 1) / 2
    check = min(
        checks,
        key=lambda item: (float(item["x"]) - center) ** 2
        + (float(item["y"]) - center) ** 2,
    )
    return str(check["id"])


def _data_index(distance: int, row: int, col: int) -> int:
    return row * distance + col


def _load_matching_modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pymatching
        from scipy import sparse
    except ImportError as exc:
        raise ImportError(
            "PyMatching, NumPy, and SciPy are required for the surface-code decoder"
        ) from exc
    return np, pymatching, sparse


def _make_matching(distance: int, checks, np, pymatching, sparse):
    rows = []
    cols = []
    data = []
    for check_index, check in enumerate(checks):
        for row, col in check["data"]:
            rows.append(check_index)
            cols.append(_data_index(distance, row, col))
            data.append(1)
    h = sparse.csc_matrix(
        (data, (rows, cols)),
        shape=(len(checks), distance * distance),
        dtype=np.uint8,
    )
    faults_matrix = sparse.eye(distance * distance, format="csc", dtype=np.uint8)
    return pymatching.Matching.from_check_matrix(
        h,
        faults_matrix=faults_matrix,
        weights=np.ones(distance * distance),
        merge_strategy="independent",
        use_virtual_boundary_node=True,
    )


def _rows(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
