from __future__ import annotations

from dataclasses import dataclass

from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.noise import BernoulliPauliNoise, MeasurementBitFlip


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

    def native_loss_spec(self, kind: str) -> dict[str, object] | None:
        if kind != "surface_diagnostic":
            return None
        return {
            "kind": kind,
            "x_qubits": tuple(
                _data_index(self.distance, row, 0)
                for row in range(self.distance)
            ),
            "z_qubits": tuple(
                _data_index(self.distance, 0, col)
                for col in range(self.distance)
            ),
            "measurement_pairs": (
                (f"r{self.rounds}_{self.hot_x_check}", f"r0_{self.hot_x_check}"),
                (f"r{self.rounds}_{self.hot_z_check}", f"r0_{self.hot_z_check}"),
            ),
        }

    def diagnostic_loss_mask(self, batch) -> int:
        """A cheap smoke-test loss involving logical paths and hot detectors."""

        all_mask = int(batch.all_mask)
        x_logical = 0
        x_mask = getattr(batch, "x_mask", None)
        if callable(x_mask):
            for row in range(self.distance):
                x_logical ^= int(x_mask(_data_index(self.distance, row, 0)))
        else:
            x_frame = batch.x_frame
            for row in range(self.distance):
                x_logical ^= x_frame[_data_index(self.distance, row, 0)]

        z_logical = 0
        z_mask = getattr(batch, "z_mask", None)
        if callable(z_mask):
            for col in range(self.distance):
                z_logical ^= int(z_mask(_data_index(self.distance, 0, col)))
        else:
            z_frame = batch.z_frame
            for col in range(self.distance):
                z_logical ^= z_frame[_data_index(self.distance, 0, col)]

        detector_mask = 0
        measurement_mask = getattr(batch, "measurement_mask", None)
        if callable(measurement_mask):
            for check_id in (self.hot_x_check, self.hot_z_check):
                detector_mask |= (
                    int(measurement_mask(f"r{self.rounds}_{check_id}"))
                    ^ int(measurement_mask(f"r0_{check_id}"))
                )
        else:
            measurements = batch.measurements
            for check_id in (self.hot_x_check, self.hot_z_check):
                detector_mask |= (
                    measurements[f"r{self.rounds}_{check_id}"]
                    ^ measurements[f"r0_{check_id}"]
                )
        return (x_logical | z_logical | detector_mask) & all_mask


RotatedSurfaceCodeMemoryExample.diagnostic_loss_mask._npsim_native_loss = (  # type: ignore[attr-defined]
    "surface_diagnostic"
)


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
