import os
import tempfile
import unittest

from npsim.core import Circuit, NoiseLocation, Operation
from npsim.runtime import BatchForwardNoiseAwareSimulator
from npsim.core import BernoulliPauliNoise, MeasurementBitFlip
from npsim.dem import LogicalObservable
from npsim.viz import (
    VisualizationUnavailableError,
    write_rotated_surface_code_spatial_hotspot_map,
)
from tests.surface_code_examples import _rotated_surface_code_checks


class RotatedSurfaceCodeXZIntegrationTests(unittest.TestCase):
    def test_d5_full_xz_memory_generates_spatial_hotspot_map(self) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "npsim-matplotlib-cache"),
        )
        try:
            import numpy as np
            import pymatching
            from scipy import sparse
        except ImportError as exc:
            self.skipTest(f"optional PyMatching dependencies are not installed: {exc}")

        distance = 5
        rounds = 3
        x_checks, z_checks = _rotated_surface_code_checks(distance)
        circuit = _make_full_xz_memory_circuit(
            distance=distance,
            rounds=rounds,
            x_checks=x_checks,
            z_checks=z_checks,
            hot_x_data=(2, 2),
            hot_z_data=(1, 3),
            hot_x_check="x_check_1_2",
            hot_z_check="z_check_2_2",
        )
        decoder = _SurfaceCodeXZDecoder(
            distance=distance,
            x_checks=x_checks,
            z_checks=z_checks,
            matching_x=_make_matching(distance, x_checks, np, pymatching, sparse),
            matching_z=_make_matching(distance, z_checks, np, pymatching, sparse),
            rounds=rounds,
        )

        result = BatchForwardNoiseAwareSimulator(
            circuit,
            observables=_surface_observables(distance),
        ).estimate(
            shots=1_000,
            seed=41,
            decoder=decoder,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "d5_full_xz_surface_code_hotspots.png")
            try:
                written = write_rotated_surface_code_spatial_hotspot_map(
                    result,
                    path,
                    distance=distance,
                    highlighted_data=(2, 2),
                    highlighted_check_ids=("meas_r3_z_check_2_2",),
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            _assert_png_nonblank(path)

        self.assertGreater(result.logical_failure_rate, 0.0)
        self.assertIn("data_x_r1_2_2", result.hotspots)
        self.assertIn("data_z_r1_1_3", result.hotspots)
        self.assertIn("meas_r3_x_check_1_2", result.hotspots)
        self.assertIn("meas_r3_z_check_2_2", result.hotspots)
        self.assertGreater(
            max(
                hotspot
                for location_id, hotspot in result.hotspots.items()
                if location_id.startswith("data_x_")
            ),
            0.0,
        )
        self.assertGreater(
            max(
                hotspot
                for location_id, hotspot in result.hotspots.items()
                if location_id.startswith("data_z_")
            ),
            0.0,
        )


class _SurfaceCodeXZDecoder:
    def __init__(
        self,
        *,
        distance: int,
        x_checks: list[dict[str, object]],
        z_checks: list[dict[str, object]],
        matching_x,
        matching_z,
        rounds: int,
    ) -> None:
        self.distance = distance
        self.x_checks = x_checks
        self.z_checks = z_checks
        self.matching_x = matching_x
        self.matching_z = matching_z
        self.rounds = rounds

    def detector_record(self, trajectory) -> dict[str, tuple[int, ...]]:
        return {
            "x": tuple(
                trajectory.measurement_by_key[f"r{self.rounds}_{check['id']}"].bit
                ^ trajectory.measurement_by_key[f"r0_{check['id']}"].bit
                for check in self.x_checks
            ),
            "z": tuple(
                trajectory.measurement_by_key[f"r{self.rounds}_{check['id']}"].bit
                ^ trajectory.measurement_by_key[f"r0_{check['id']}"].bit
                for check in self.z_checks
            ),
        }

    def decode(self, detector_record, measurements, trajectory):
        del measurements, trajectory
        z_correction = self.matching_x.decode(detector_record["x"])
        x_correction = self.matching_z.decode(detector_record["z"])
        if hasattr(z_correction, "tolist"):
            z_correction = z_correction.tolist()
        if hasattr(x_correction, "tolist"):
            x_correction = x_correction.tolist()
        return {
            "x_correction": tuple(int(bit) for bit in x_correction),
            "z_correction": tuple(int(bit) for bit in z_correction),
        }

    def loss(self, trajectory, correction) -> float:
        x_logical = 0
        z_logical = 0
        for row in range(self.distance):
            qubit = _data_index(self.distance, row, 0)
            x_logical ^= trajectory.frame.x[qubit] ^ correction["x_correction"][qubit]
        for col in range(self.distance):
            qubit = _data_index(self.distance, 0, col)
            z_logical ^= trajectory.frame.z[qubit] ^ correction["z_correction"][qubit]
        return float(bool(x_logical or z_logical))

    def decode_batch_masks(self, batch):
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
        if hasattr(z_corrections, "tolist"):
            z_corrections = z_corrections.tolist()
        if hasattr(x_corrections, "tolist"):
            x_corrections = x_corrections.tolist()

        x_correction_masks = [0] * (self.distance * self.distance)
        for shot, x_correction in enumerate(x_corrections):
            for qubit, bit in enumerate(x_correction):
                if int(bit):
                    x_correction_masks[qubit] |= 1 << shot
        z_correction_masks = [0] * (self.distance * self.distance)
        for shot, z_correction in enumerate(z_corrections):
            for qubit, bit in enumerate(z_correction):
                if int(bit):
                    z_correction_masks[qubit] |= 1 << shot

        x_prediction = 0
        for row in range(self.distance):
            x_prediction ^= x_correction_masks[_data_index(self.distance, row, 0)]
        z_prediction = 0
        for col in range(self.distance):
            z_prediction ^= z_correction_masks[_data_index(self.distance, 0, col)]
        return {0: x_prediction, 1: z_prediction}


def _make_full_xz_memory_circuit(
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
    operations = []
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
                            id=f"data_x_r{round_idx}_{row}_{col}",
                            model=BernoulliPauliNoise("X"),
                            rate=0.11 if (row, col) == hot_x_data else 0.025,
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
                            id=f"data_z_r{round_idx}_{row}_{col}",
                            model=BernoulliPauliNoise("Z"),
                            rate=0.10 if (row, col) == hot_z_data else 0.025,
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
                id=f"meas_r{round_idx}_{check_id}",
                model=MeasurementBitFlip(),
                rate=0.08 if check_id == hot_check_id else 0.015,
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


def _data_index(distance: int, row: int, col: int) -> int:
    return row * distance + col


def _surface_observables(distance: int) -> tuple[LogicalObservable, ...]:
    return (
        LogicalObservable(
            id=0,
            pauli_qubits=tuple(_data_index(distance, row, 0) for row in range(distance)),
            pauli="Z" * distance,
        ),
        LogicalObservable(
            id=1,
            pauli_qubits=tuple(_data_index(distance, 0, col) for col in range(distance)),
            pauli="X" * distance,
        ),
    )


def _assert_png_nonblank(path: str) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise unittest.SkipTest(f"Pillow is not installed: {exc}") from exc
    with Image.open(path) as image:
        colors = image.convert("RGB").resize((32, 32)).getcolors(maxcolors=1024)
        width, height = image.size
    if os.path.getsize(path) <= 1_000:
        raise AssertionError("PNG output is unexpectedly small")
    if width < 500 or height < 300:
        raise AssertionError("PNG output dimensions are unexpectedly small")
    if colors is None or len(colors) <= 8:
        raise AssertionError("PNG output appears blank")
