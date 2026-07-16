from dataclasses import fields
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest

import faultscope
import faultscope.backends
import faultscope.collection
import faultscope.core
import faultscope.decoders
import faultscope.dem
import faultscope.experiments
import faultscope.io
import faultscope.runtime
import faultscope.viz

from faultscope import DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.collection import (
    COLLECTION_CSV_HEADER,
    Collector,
    CollectionData,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    FiniteSizeScalingFit,
    HotspotCollectionResult,
    PairwiseCrossing,
    Progress,
    TaskStats,
    ThresholdAnalysisResult,
    ThresholdEstimate,
    ThresholdPoint,
    analyze_thresholds,
    collect,
    collect_hotspots,
    iter_collect,
    iter_progress,
    plot_threshold_analysis,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SNAPSHOT = REPO_ROOT / "tests" / "public_api_contract.json"


class ReleaseContractTests(unittest.TestCase):
    def test_python_distribution_version_is_v0_2(self) -> None:
        self.assertEqual(faultscope.__version__, "0.2.4")
        self.assertEqual(importlib.metadata.version("faultscope"), "0.2.4")

    def test_package_contains_pep561_marker_and_native_stub(self) -> None:
        package_root = Path(faultscope.__file__).resolve().parent
        self.assertTrue((package_root / "py.typed").is_file())
        native_stub = package_root / "_native.pyi"
        self.assertTrue(native_stub.is_file())
        stub = native_stub.read_text()
        self.assertIn("class NativeDemSampler", stub)
        self.assertIn("from typing_extensions import Never, Self", stub)
        stabilizer_block = stub.split("class StabilizerState:", 1)[1].split("\n@final", 1)[0]
        self.assertIn(
            "def __new__(cls, _no_direct_construction: Never, /) -> Self",
            stabilizer_block,
        )

    def test_public_exports_match_contract(self) -> None:
        modules = {
            "faultscope": faultscope,
            "faultscope.backends": faultscope.backends,
            "faultscope.collection": faultscope.collection,
            "faultscope.core": faultscope.core,
            "faultscope.decoders": faultscope.decoders,
            "faultscope.dem": faultscope.dem,
            "faultscope.experiments": faultscope.experiments,
            "faultscope.io": faultscope.io,
            "faultscope.runtime": faultscope.runtime,
            "faultscope.viz": faultscope.viz,
        }
        expected = json.loads(API_SNAPSHOT.read_text())
        actual = {name: sorted(module.__all__) for name, module in modules.items()}
        self.assertEqual(actual, expected)

    def test_collection_csv_header_matches_contract(self) -> None:
        self.assertEqual(
            COLLECTION_CSV_HEADER,
            "shots,errors,discards,seconds,decoder,strong_id,json_metadata,custom_counts",
        )

    def test_public_python_function_signatures_match_contract(self) -> None:
        expected_parameters = {
            collect: ("tasks", "options", "run_options"),
            collect_hotspots: ("tasks", "options", "run_options"),
            iter_collect: ("tasks", "options", "run_options"),
            iter_progress: ("tasks", "options", "run_options"),
            analyze_thresholds: (
                "stats",
                "x_key",
                "distance_key",
                "series_keys",
                "count_key",
                "bootstrap_samples",
                "confidence_level",
                "seed",
                "scaling_order",
            ),
            plot_threshold_analysis: ("results", "output", "axes", "log_y"),
            read_stats_from_csv_files: ("filepaths",),
            write_stats_to_csv_file: ("filepath", "stats", "append"),
            faultscope.create_native_decoder: (
                "name",
                "dem",
                "circuit",
                "detectors",
                "observables",
                "options",
            ),
            faultscope.load_stim_file: ("path",),
            faultscope.parse_stim_circuit: ("text",),
        }
        actual = {
            function: tuple(inspect.signature(function).parameters)
            for function in expected_parameters
        }
        self.assertEqual(actual, expected_parameters)

    def test_public_dataclass_fields_and_class_methods_match_contract(self) -> None:
        expected_fields = {
            CollectionOptions: (
                "max_shots",
                "max_errors",
                "batch_size",
                "start_batch_size",
                "max_batch_size",
                "max_batch_seconds",
                "min_shots",
            ),
            CollectionRunOptions: (
                "seed",
                "num_workers",
                "existing_data_filepaths",
                "save_resume_filepath",
                "count_observable_error_combos",
                "count_detection_events",
                "custom_error_count_key",
                "decoders",
            ),
            Collector: ("options", "run_options"),
            CollectionTask: (
                "circuit",
                "dem",
                "detectors",
                "observables",
                "decoder",
                "decoder_options",
                "metadata",
                "collection_options",
                "task_id",
                "postselection_mask",
                "postselected_observables_mask",
            ),
            TaskStats: (
                "task_id",
                "shots",
                "errors",
                "discards",
                "seconds",
                "decoder",
                "metadata",
                "strong_id",
                "custom_counts",
            ),
            HotspotCollectionResult: (
                "stats",
                "batch_stats",
                "edge_sensitivities",
            ),
            Progress: ("new_stats", "status_message"),
            ThresholdPoint: ("x", "distance", "shots", "errors", "rate", "stderr"),
            ThresholdEstimate: (
                "value",
                "ci_low",
                "ci_high",
                "confidence_level",
                "bootstrap_samples",
                "bootstrap_successes",
            ),
            PairwiseCrossing: (
                "lower_distance",
                "upper_distance",
                "status",
                "candidates",
                "estimate",
            ),
            FiniteSizeScalingFit: (
                "status",
                "threshold",
                "critical_exponent",
                "coefficients",
                "reduced_chi_squared",
                "message",
            ),
            ThresholdAnalysisResult: (
                "series",
                "points",
                "crossings",
                "pairwise_threshold",
                "scaling_fit",
            ),
        }
        self.assertEqual(
            {cls: tuple(field.name for field in fields(cls)) for cls in expected_fields},
            expected_fields,
        )
        self.assertEqual(
            sorted(name for name in CollectionData.__dict__ if not name.startswith("_")),
            ["add_sample", "values"],
        )
        self.assertEqual(
            sorted(name for name in TaskStats.__dict__ if not name.startswith("_")),
            [
                "accepted_error_rate",
                "accepted_error_rate_stderr",
                "accepted_shots",
                "from_csv_row",
                "logical_error_rate",
                "logical_error_rate_stderr",
                "raw_error_rate",
                "strong_id",
                "to_csv_line",
                "to_csv_row",
                "with_edits",
            ],
        )
        self.assertEqual(
            sorted(name for name in Collector.__dict__ if not name.startswith("_")),
            ["collect", "collect_hotspots", "iter_collect", "iter_progress"],
        )
        self.assertEqual(
            sorted(name for name in ThresholdAnalysisResult.__dict__ if not name.startswith("_")),
            ["to_dict"],
        )

    def test_collection_cli_subcommands_match_contract(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "faultscope.collection", "--help"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for command in ("collect", "summarize", "merge", "plot", "fit", "threshold"):
            self.assertIn(command, result.stdout)

    def test_top_level_callables_have_runtime_documentation(self) -> None:
        undocumented = [
            name
            for name in faultscope.__all__
            if callable(getattr(faultscope, name)) and not inspect.getdoc(getattr(faultscope, name))
        ]
        self.assertEqual(undocumented, [])

    def test_collection_strong_id_schema_v2_is_frozen(self) -> None:
        dem = DetectorErrorModel(
            detectors=(),
            observables=(LogicalObservable(id=0),),
            edges=(
                DetectorErrorEdge(
                    probability=1.0,
                    detectors=(),
                    observables=(0,),
                    location_id="logical",
                    event="L",
                ),
            ),
        )
        (stats,) = collect(
            [CollectionTask(dem=dem, metadata={"d": 3, "p": 0.01}, task_id="golden")],
            options=CollectionOptions(max_shots=1),
            run_options=CollectionRunOptions(seed=7),
        )
        self.assertEqual(
            stats.strong_id,
            "372f4320007288494205f1f808fbbc70d1743fd90c65ea06799350c52e2feba3",
        )


if __name__ == "__main__":
    unittest.main()
