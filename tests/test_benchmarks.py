import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class BenchmarkSmokeTests(unittest.TestCase):
    def test_surface_code_decoder_performance_smoke(self) -> None:
        for module_name in ("stim", "pymatching", "numpy"):
            if not _has_module(module_name):
                self.skipTest(f"{module_name} is required for this benchmark")

        script = ROOT / "benchmarks" / "surface_code_decoder_performance.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--distances",
                "3",
                "--rates",
                "0.01",
                "--shots",
                "128",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        rows = [
            line.split("\t")
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(rows), 2)
        header = rows[0]
        self.assertEqual(header[5], "path")
        self.assertEqual(header[-2:], ["python_decode_calls", "status"])

        body = [dict(zip(header, row)) for row in rows[1:]]
        rows_by_path = {row["path"]: row for row in body}
        self.assertIn("stim-dem-pymatching", rows_by_path)
        self.assertIn("stim-dem-pymatching-bitpacked", rows_by_path)
        self.assertIn("faultscope-dem-pymatching", rows_by_path)
        self.assertIn("faultscope-dem-pymatching-native", rows_by_path)
        self.assertIn("faultscope-dem-fusion-blossom", rows_by_path)

        native_pymatching_row = rows_by_path["faultscope-dem-pymatching-native"]
        if native_pymatching_row["status"] == "ok":
            self.assertEqual(native_pymatching_row["python_decode_calls"], "0")
        else:
            self.assertTrue(native_pymatching_row["status"].startswith("skip:"))

        fusion_row = rows_by_path["faultscope-dem-fusion-blossom"]
        if fusion_row["status"] == "ok":
            self.assertEqual(fusion_row["python_decode_calls"], "0")
        else:
            self.assertTrue(fusion_row["status"].startswith("skip:"))


if __name__ == "__main__":
    unittest.main()
