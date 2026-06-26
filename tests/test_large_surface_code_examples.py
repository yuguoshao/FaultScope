import os
import tempfile
import unittest

from faultscope.runtime import FaultScopeSimulator
from faultscope.viz import (
    VisualizationUnavailableError,
    write_rotated_surface_code_spatial_hotspot_map,
)

from tests.surface_code_examples import (
    _rotated_surface_code_checks,
    make_large_rotated_surface_code_memory_example,
)


class LargeRotatedSurfaceCodeExampleTests(unittest.TestCase):
    def test_d9_d11_d13_examples_have_expected_data_qubit_counts(self) -> None:
        expected_data_qubits = {
            9: 81,
            11: 121,
            13: 169,
        }
        for distance, expected in expected_data_qubits.items():
            with self.subTest(distance=distance):
                example = make_large_rotated_surface_code_memory_example(
                    distance=distance,
                    rounds=2,
                )
                self.assertEqual(example.data_qubits, expected)
                self.assertEqual(example.circuit.n_qubits, expected)
                self.assertEqual(
                    len(example.x_checks) + len(example.z_checks),
                    distance * distance - 1,
                )
                self.assertEqual(
                    len(example.x_checks),
                    (distance * distance - 1) // 2,
                )
                self.assertEqual(
                    len(example.z_checks),
                    (distance * distance - 1) // 2,
                )
                self.assertEqual(
                    sum(
                        1
                        for check in (*example.x_checks, *example.z_checks)
                        if len(check["data"]) == 2
                    ),
                    2 * (distance - 1),
                )

                locations = example.circuit.noise_locations()
                self.assertEqual(
                    sum(
                        1
                        for location in locations.values()
                        if location.tags.get("role") == "data"
                    ),
                    2 * example.rounds * expected,
                )
                self.assertIn(
                    f"d{distance}_data_x_r1_{example.hot_x_data[0]}_{example.hot_x_data[1]}",
                    locations,
                )
                self.assertIn(
                    f"d{distance}_data_z_r1_{example.hot_z_data[0]}_{example.hot_z_data[1]}",
                    locations,
                )
                self.assertIn(f"d{distance}_meas_r1_{example.hot_x_check}", locations)
                self.assertIn(f"d{distance}_meas_r1_{example.hot_z_check}", locations)

    def test_d7_check_coordinates_match_stim_generated_rotated_layout(self) -> None:
        try:
            import stim
        except ImportError as exc:
            self.skipTest(f"optional Stim dependency is not installed: {exc}")

        distance = 7
        x_checks, z_checks = _rotated_surface_code_checks(distance)
        local_coords = {
            (int(2 * (float(check["x"]) + 0.5)), int(2 * (float(check["y"]) + 0.5)))
            for check in (*x_checks, *z_checks)
        }

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=1,
        )
        qubit_coords = {}
        stim_check_qubits = None
        for instruction in circuit.flattened():
            if instruction.name == "QUBIT_COORDS":
                coords = tuple(int(coord) for coord in instruction.gate_args_copy()[:2])
                for target in instruction.targets_copy():
                    qubit_coords[int(target.value)] = coords
            elif instruction.name == "MR" and stim_check_qubits is None:
                stim_check_qubits = tuple(
                    int(target.value) for target in instruction.targets_copy()
                )

        self.assertIsNotNone(stim_check_qubits)
        assert stim_check_qubits is not None
        stim_coords = {qubit_coords[qubit] for qubit in stim_check_qubits}

        self.assertEqual(len(local_coords), distance * distance - 1)
        self.assertEqual(local_coords, stim_coords)

    def test_d9_example_runs_batch_hotspot_estimation(self) -> None:
        example = make_large_rotated_surface_code_memory_example(
            distance=9,
            rounds=2,
        )
        decoder = self._make_decoder_or_skip(example)
        result = FaultScopeSimulator(
            example.circuit,
            observables=example.observables,
        ).estimate(
            shots=256,
            seed=91,
            decoder=decoder,
        )

        self.assertEqual(result.shots, 256)
        self.assertGreaterEqual(result.logical_failure_rate, 0.0)
        self.assertLessEqual(result.logical_failure_rate, 1.0)
        self.assertGreater(max(result.hotspots.values()), 0.0)
        self.assertIn("measurement_noise", result.by_operation)
        self.assertIn("data_x_noise", result.by_operation)
        self.assertIn("data_z_noise", result.by_operation)

    def test_d11_and_d13_examples_run_short_batch_smoke_tests(self) -> None:
        for distance in (11, 13):
            with self.subTest(distance=distance):
                example = make_large_rotated_surface_code_memory_example(
                    distance=distance,
                    rounds=1,
                )
                decoder = self._make_decoder_or_skip(example)
                result = FaultScopeSimulator(
                    example.circuit,
                    observables=example.observables,
                ).estimate(
                    shots=32,
                    seed=100 + distance,
                    decoder=decoder,
                )

                self.assertEqual(result.shots, 32)
                self.assertEqual(
                    result.locations.keys(),
                    example.circuit.noise_locations().keys(),
                )
                self.assertGreaterEqual(result.logical_failure_rate, 0.0)
                self.assertLessEqual(result.logical_failure_rate, 1.0)
                self.assertTrue(result.top_hotspots(top_k=5))

    def test_d13_example_writes_spatial_hotspot_map(self) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            os.path.join(tempfile.gettempdir(), "faultscope-matplotlib-cache"),
        )
        example = make_large_rotated_surface_code_memory_example(
            distance=13,
            rounds=1,
        )
        decoder = self._make_decoder_or_skip(example)
        result = FaultScopeSimulator(
            example.circuit,
            observables=example.observables,
        ).estimate(
            shots=32,
            seed=113,
            decoder=decoder,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "d13_surface_code_hotspots.png")
            try:
                written = write_rotated_surface_code_spatial_hotspot_map(
                    result,
                    path,
                    distance=example.distance,
                    highlighted_data=example.hot_x_data,
                    highlighted_check_ids=(
                        f"d13_meas_r1_{example.hot_x_check}",
                        f"d13_meas_r1_{example.hot_z_check}",
                    ),
                )
            except VisualizationUnavailableError as exc:
                self.skipTest(str(exc))
            self.assertEqual(str(written), path)
            self._assert_png_nonblank(path)

    def _assert_png_nonblank(self, path: str) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            self.skipTest(f"Pillow is not installed: {exc}")
        self.assertGreater(os.path.getsize(path), 1_000)
        with Image.open(path) as image:
            self.assertGreaterEqual(image.size[0], 1_700)
            self.assertGreaterEqual(image.size[1], 1_400)
            colors = image.convert("RGB").resize((32, 32)).getcolors(maxcolors=1024)
        self.assertIsNotNone(colors)
        self.assertGreater(len(colors), 8)

    def _make_decoder_or_skip(self, example):
        try:
            return example.make_decoder()
        except ImportError as exc:
            self.skipTest(str(exc))


if __name__ == "__main__":
    unittest.main()
