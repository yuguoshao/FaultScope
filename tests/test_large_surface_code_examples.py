import os
import tempfile
import unittest

from npsim.batch import BatchForwardNoiseAwareSimulator
from npsim.visualization import (
    VisualizationUnavailableError,
    write_rotated_surface_code_spatial_hotspot_map,
)

from tests.surface_code_examples import make_large_rotated_surface_code_memory_example


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

    def test_d9_example_runs_batch_hotspot_estimation(self) -> None:
        example = make_large_rotated_surface_code_memory_example(
            distance=9,
            rounds=2,
        )
        result = BatchForwardNoiseAwareSimulator(example.circuit).estimate(
            shots=256,
            seed=91,
            loss_mask_fn=example.diagnostic_loss_mask,
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
                result = BatchForwardNoiseAwareSimulator(example.circuit).estimate(
                    shots=32,
                    seed=100 + distance,
                    loss_mask_fn=example.diagnostic_loss_mask,
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
            os.path.join(tempfile.gettempdir(), "npsim-matplotlib-cache"),
        )
        example = make_large_rotated_surface_code_memory_example(
            distance=13,
            rounds=1,
        )
        result = BatchForwardNoiseAwareSimulator(example.circuit).estimate(
            shots=32,
            seed=113,
            loss_mask_fn=example.diagnostic_loss_mask,
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


if __name__ == "__main__":
    unittest.main()
