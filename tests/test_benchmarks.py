import importlib.util
import io
import json
import runpy
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class BenchmarkSmokeTests(unittest.TestCase):
    def test_collection_throughput_writes_machine_readable_json(self) -> None:
        script = ROOT / "benchmarks" / "collection_throughput.py"
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "collection.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shots",
                    "16",
                    "--batch-size",
                    "4",
                    "--small-batch-size",
                    "2",
                    "--adaptive-start-batch-size",
                    "1",
                    "--adaptive-max-batch-size",
                    "4",
                    "--workers",
                    "1",
                    "--tasks",
                    "1",
                    "--repeats",
                    "1",
                    "--json-out",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = json.loads(output.read_text())
            self.assertEqual(len(records), 5)
            self.assertEqual(records[0]["mode"], "single")
            self.assertEqual(records[0]["status"], "ok")
            self.assertGreater(records[0]["shots_per_second"], 0)

    def test_collection_throughput_includes_adaptive_scenarios(self) -> None:
        script = ROOT / "benchmarks" / "collection_throughput.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--shots",
                "64",
                "--batch-size",
                "16",
                "--small-batch-size",
                "4",
                "--adaptive-start-batch-size",
                "1",
                "--adaptive-max-batch-size",
                "8",
                "--max-batch-seconds",
                "0.001",
                "--workers",
                "1",
                "2",
                "--tasks",
                "2",
                "--repeats",
                "1",
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
        modes = {
            line.split("\t", 1)[0] for line in completed.stdout.splitlines()[1:] if line.strip()
        }
        self.assertEqual(
            modes,
            {"single", "multi", "multi-small", "adaptive-single", "adaptive-multi"},
        )

    def test_collection_throughput_rejects_zero_adaptive_max_batch_size(self) -> None:
        script = ROOT / "benchmarks" / "collection_throughput.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--adaptive-max-batch-size",
                "0",
                "--repeats",
                "1",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("adaptive-max-batch-size must be positive", completed.stderr)

    def test_surface_code_threshold_uses_public_threshold_analysis(self) -> None:
        script = ROOT / "benchmarks" / "surface_code_threshold.py"
        source = script.read_text(encoding="utf-8")

        self.assertNotIn("_crossing_rate", source)
        self.assertIn("from faultscope.collection import (", source)
        self.assertIn("    TaskStats,", source)
        self.assertIn("    analyze_thresholds,", source)
        self.assertIn("TaskStats(", source)
        self.assertIn("analyze_thresholds(", source)
        self.assertIn('series_keys=("path", "basis")', source)

    def test_surface_code_threshold_analysis_smoke(self) -> None:
        for module_name in ("stim", "pymatching", "numpy", "scipy"):
            if not _has_module(module_name):
                self.skipTest(f"{module_name} is required for this benchmark")

        script = ROOT / "benchmarks" / "surface_code_threshold.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--distances",
                "3",
                "5",
                "7",
                "--rates",
                "0.005",
                "0.01",
                "--shots",
                "16",
                "--bootstrap-samples",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        lines = completed.stdout.splitlines()
        pairwise = [line for line in lines if line.startswith("threshold-pairwise\t")]
        scaling = [line for line in lines if line.startswith("threshold-scaling\t")]
        self.assertTrue(pairwise, msg=completed.stdout)
        self.assertTrue(scaling, msg=completed.stdout)
        pairwise_paths = {line.split("\t")[2] for line in pairwise}
        self.assertIn("stim", pairwise_paths)
        self.assertIn("stim-dem", pairwise_paths)
        self.assertTrue(any(line.split("\t")[3] in {"ok", "insufficient_data"} for line in scaling))

    def test_surface_code_threshold_formats_estimates_and_intervals(self) -> None:
        from faultscope.collection import (
            FiniteSizeScalingFit,
            PairwiseCrossing,
            ThresholdAnalysisResult,
            ThresholdEstimate,
        )

        estimate = ThresholdEstimate(0.01, 0.009, 0.011, 0.95, 100, 91)
        exponent = ThresholdEstimate(1.5, 1.2, 1.8, 0.95, 100, 88)
        result = ThresholdAnalysisResult(
            series={"path": "stim", "basis": "x"},
            points=(),
            crossings=(PairwiseCrossing(3.0, 5.0, "ok", (0.01,), estimate),),
            pairwise_threshold=estimate,
            scaling_fit=FiniteSizeScalingFit(
                status="ok",
                threshold=estimate,
                critical_exponent=exponent,
                coefficients=(0.1, 0.2),
                reduced_chi_squared=1.25,
                message=None,
            ),
        )
        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results"]((result,))

        pairwise, summary, scaling = output.getvalue().splitlines()
        self.assertEqual(
            pairwise.split("\t"),
            [
                "threshold-pairwise",
                "x",
                "stim",
                "3-5",
                "ok",
                "0.01",
                "0.01",
                "0.0089999999999999993",
                "0.010999999999999999",
                "91",
            ],
        )
        self.assertEqual(summary.split("\t")[3:], pairwise.split("\t")[6:])
        self.assertEqual(
            scaling.split("\t")[3:],
            [
                "ok",
                "0.01",
                "0.0089999999999999993",
                "0.010999999999999999",
                "91",
                "1.5",
                "1.2",
                "1.8",
                "88",
                "1.25",
                "NA",
            ],
        )

    def test_surface_code_threshold_formats_missing_path_diagnostics(self) -> None:
        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results_for_paths"](
                (),
                basis="x",
                paths=("missing",),
                distances=(3, 5, 7),
            )

        lines = output.getvalue().splitlines()
        pairwise = [line.split("\t") for line in lines if line.startswith("threshold-pairwise\t")]
        scaling = [line.split("\t") for line in lines if line.startswith("threshold-scaling\t")]
        self.assertEqual([row[3] for row in pairwise], ["3-5", "5-7"])
        self.assertTrue(all(row[2] == "missing" for row in pairwise))
        self.assertTrue(all(row[4] == "no_crossing" for row in pairwise))
        self.assertEqual(len(scaling), 1)
        self.assertEqual(scaling[0][2:4], ["missing", "insufficient_data"])

    def test_surface_code_threshold_formats_missing_distance_diagnostics(self) -> None:
        from faultscope.collection import (
            FiniteSizeScalingFit,
            PairwiseCrossing,
            ThresholdAnalysisResult,
        )

        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        partial = ThresholdAnalysisResult(
            series={"basis": "x", "path": "partial"},
            points=(),
            crossings=(PairwiseCrossing(3.0, 7.0, "ok", (0.01,), None),),
            pairwise_threshold=None,
            scaling_fit=FiniteSizeScalingFit(
                status="insufficient_data",
                threshold=None,
                critical_exponent=None,
                coefficients=(),
                reduced_chi_squared=None,
                message="missing distance",
            ),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results_for_paths"](
                (partial,),
                basis="x",
                paths=("partial",),
                distances=(3, 5, 7),
            )

        pairwise = [
            line.split("\t")
            for line in output.getvalue().splitlines()
            if line.startswith("threshold-pairwise\t")
        ]
        self.assertEqual([row[3] for row in pairwise], ["3-5", "5-7"])
        self.assertTrue(all(row[4] == "no_crossing" for row in pairwise))

    def test_surface_code_threshold_keeps_valid_summary_with_missing_edge_distance(self) -> None:
        from faultscope.collection import (
            FiniteSizeScalingFit,
            PairwiseCrossing,
            ThresholdAnalysisResult,
            ThresholdEstimate,
            ThresholdPoint,
        )

        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        estimate = ThresholdEstimate(0.01, 0.009, 0.011, 0.95, 100, 80)
        partial = ThresholdAnalysisResult(
            series={"basis": "x", "path": "partial"},
            points=(
                ThresholdPoint(0.005, 5.0, 100, 1, 0.01, 0.01),
                ThresholdPoint(0.015, 5.0, 100, 2, 0.02, 0.01),
                ThresholdPoint(0.005, 7.0, 100, 2, 0.02, 0.01),
                ThresholdPoint(0.015, 7.0, 100, 1, 0.01, 0.01),
            ),
            crossings=(PairwiseCrossing(5.0, 7.0, "ok", (0.01,), estimate),),
            pairwise_threshold=estimate,
            scaling_fit=FiniteSizeScalingFit(
                status="insufficient_data",
                threshold=None,
                critical_exponent=None,
                coefficients=(),
                reduced_chi_squared=None,
                message="missing distance",
            ),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results_for_paths"](
                (partial,),
                basis="x",
                paths=("partial",),
                distances=(3, 5, 7),
            )

        lines = output.getvalue().splitlines()
        summary = next(
            line.split("\t") for line in lines if line.startswith("threshold-pairwise-summary\t")
        )
        self.assertEqual(
            summary[3:], ["0.01", "0.0089999999999999993", "0.010999999999999999", "80"]
        )

    def test_surface_code_threshold_drops_ci_from_discarded_duplicate_candidate(self) -> None:
        from faultscope.collection import (
            FiniteSizeScalingFit,
            PairwiseCrossing,
            ThresholdAnalysisResult,
            ThresholdEstimate,
            ThresholdPoint,
        )

        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        contaminated = ThresholdEstimate(0.01, 0.001, 0.019, 0.95, 100, 99)
        retained = ThresholdEstimate(0.01, 0.009, 0.011, 0.95, 100, 80)
        partial = ThresholdAnalysisResult(
            series={"basis": "x", "path": "partial"},
            points=tuple(
                ThresholdPoint(x, distance, 100, 1, 0.01, 0.01)
                for distance in (3.0, 7.0, 9.0)
                for x in (0.005, 0.015)
            ),
            crossings=(
                PairwiseCrossing(3.0, 7.0, "ok", (0.01,), contaminated),
                PairwiseCrossing(7.0, 9.0, "ok", (0.01,), retained),
            ),
            pairwise_threshold=contaminated,
            scaling_fit=FiniteSizeScalingFit(
                status="insufficient_data",
                threshold=None,
                critical_exponent=None,
                coefficients=(),
                reduced_chi_squared=None,
                message="missing distance",
            ),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results_for_paths"](
                (partial,),
                basis="x",
                paths=("partial",),
                distances=(3, 5, 7, 9),
            )

        summary = next(
            line.split("\t")
            for line in output.getvalue().splitlines()
            if line.startswith("threshold-pairwise-summary\t")
        )
        self.assertEqual(summary[4:6], ["0.0089999999999999993", "0.010999999999999999"])
        self.assertEqual(summary[6], "80")

    def test_surface_code_threshold_drops_ci_from_discarded_candidate_free_pair(self) -> None:
        from faultscope.collection import (
            FiniteSizeScalingFit,
            PairwiseCrossing,
            ThresholdAnalysisResult,
            ThresholdEstimate,
            ThresholdPoint,
        )

        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        contaminated = ThresholdEstimate(0.01, 0.001, 0.019, 0.95, 100, 99)
        retained = ThresholdEstimate(0.01, 0.009, 0.011, 0.95, 100, 80)
        partial = ThresholdAnalysisResult(
            series={"basis": "x", "path": "partial"},
            points=tuple(
                ThresholdPoint(x, distance, 100, 1, 0.01, 0.01)
                for distance in (3.0, 7.0, 9.0)
                for x in (0.005, 0.015)
            ),
            crossings=(
                PairwiseCrossing(3.0, 7.0, "no_crossing", (), None),
                PairwiseCrossing(7.0, 9.0, "ok", (0.01,), retained),
            ),
            pairwise_threshold=contaminated,
            scaling_fit=FiniteSizeScalingFit(
                status="insufficient_data",
                threshold=None,
                critical_exponent=None,
                coefficients=(),
                reduced_chi_squared=None,
                message="missing distance",
            ),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results_for_paths"](
                (partial,),
                basis="x",
                paths=("partial",),
                distances=(3, 5, 7, 9),
            )

        summary = next(
            line.split("\t")
            for line in output.getvalue().splitlines()
            if line.startswith("threshold-pairwise-summary\t")
        )
        self.assertEqual(summary[4:6], ["0.0089999999999999993", "0.010999999999999999"])
        self.assertEqual(summary[6], "80")

    def test_surface_code_threshold_does_not_use_one_pair_ci_for_two_retained_pairs(self) -> None:
        from faultscope.collection import (
            FiniteSizeScalingFit,
            PairwiseCrossing,
            ThresholdAnalysisResult,
            ThresholdEstimate,
            ThresholdPoint,
        )

        namespace = runpy.run_path(str(ROOT / "benchmarks" / "surface_code_threshold.py"))
        first = ThresholdEstimate(0.01, 0.008, 0.012, 0.95, 100, 70)
        second = ThresholdEstimate(0.01, 0.009, 0.011, 0.95, 100, 80)
        contaminated = ThresholdEstimate(0.01, 0.001, 0.019, 0.95, 100, 99)
        partial = ThresholdAnalysisResult(
            series={"basis": "x", "path": "partial"},
            points=tuple(
                ThresholdPoint(x, distance, 100, 1, 0.01, 0.01)
                for distance in (3.0, 7.0, 9.0, 11.0)
                for x in (0.005, 0.015)
            ),
            crossings=(
                PairwiseCrossing(3.0, 7.0, "no_crossing", (), None),
                PairwiseCrossing(7.0, 9.0, "ok", (0.01,), first),
                PairwiseCrossing(9.0, 11.0, "ok", (0.01,), second),
            ),
            pairwise_threshold=contaminated,
            scaling_fit=FiniteSizeScalingFit(
                status="insufficient_data",
                threshold=None,
                critical_exponent=None,
                coefficients=(),
                reduced_chi_squared=None,
                message="missing distance",
            ),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_threshold_results_for_paths"](
                (partial,),
                basis="x",
                paths=("partial",),
                distances=(3, 5, 7, 9, 11),
            )

        summary = next(
            line.split("\t")
            for line in output.getvalue().splitlines()
            if line.startswith("threshold-pairwise-summary\t")
        )
        self.assertEqual(summary[3:], ["0.01", "NA", "NA", "0"])

    def test_test_extra_installs_pytest_for_threshold_suite(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as f:
            test_extra = tomllib.load(f)["project"]["optional-dependencies"]["test"]
        self.assertTrue(any(dependency.startswith("pytest") for dependency in test_extra))

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
        rows = [line.split("\t") for line in completed.stdout.splitlines() if line.strip()]
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
        self.assertIn("faultscope-dem-mwpm", rows_by_path)
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

        mwpm_row = rows_by_path["faultscope-dem-mwpm"]
        if mwpm_row["status"] == "ok":
            self.assertEqual(mwpm_row["python_decode_calls"], "0")
        else:
            self.assertTrue(mwpm_row["status"].startswith("skip:"))


if __name__ == "__main__":
    unittest.main()
