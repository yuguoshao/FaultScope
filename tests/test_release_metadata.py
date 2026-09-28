from pathlib import Path
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _toml(path: str) -> dict[str, object]:
    with (ROOT / path).open("rb") as f:
        return tomllib.load(f)


class ReleaseMetadataTests(unittest.TestCase):
    def test_ci_and_release_workflows_define_required_gates(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        wheels = (ROOT / ".github/workflows/wheels.yml").read_text()
        release = (ROOT / ".github/workflows/release.yml").read_text()
        benchmarks = (ROOT / ".github/workflows/benchmarks.yml").read_text()

        self.assertIn(
            'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]',
            ci,
        )
        self.assertIn("toolchain: 1.85", ci)
        self.assertIn("cargo clippy --workspace --all-targets --all-features -- -D warnings", ci)
        self.assertIn("python -m mypy.stubtest faultscope._native", ci)
        self.assertGreaterEqual(ci.count("python -m venv .venv"), 3)
        self.assertGreaterEqual(
            ci.count('.venv/bin/python -m pip install --upgrade pip "maturin>=1.7,<2"'),
            3,
        )
        self.assertGreaterEqual(
            ci.count(".venv/bin/maturin develop --extras test --locked"),
            2,
        )
        self.assertIn(".venv/bin/maturin develop --locked", ci)
        self.assertIn(".venv/bin/python benchmarks/collection_throughput.py", ci)
        self.assertIn("workflow_call:", wheels)
        self.assertIn("manylinux", wheels)
        self.assertIn("windows", wheels.lower())
        self.assertIn("macos", wheels.lower())
        self.assertIn("faultscope-pymatching-sdist", wheels)
        self.assertIn("scripts/wheel_smoke.py --pymatching", wheels)
        self.assertIn("tags:", release)
        self.assertIn('- "v*.*.*"', release)
        self.assertIn("environment: release", release)
        self.assertIn("needs: [publish-crates, wheels]", release)
        self.assertLess(
            release.index("cargo publish -p faultscope-core"),
            release.index("cargo publish -p faultscope-collection"),
        )
        self.assertLess(
            release.index("publish-main"),
            release.index("publish-pymatching"),
        )
        self.assertIn("schedule:", benchmarks)
        self.assertIn("--json-out collection-throughput.json", benchmarks)
        self.assertNotIn("fail-under", benchmarks)

    def test_workspace_owns_version_and_msrv(self) -> None:
        workspace = _toml("Cargo.toml")["workspace"]
        package = workspace["package"]
        self.assertEqual(package["version"], "0.2.11")
        self.assertEqual(package["rust-version"], "1.85")

    def test_faultscope_workspace_packages_and_internal_dependencies_are_v0_2(self) -> None:
        manifests = (
            "crates/faultscope-core/Cargo.toml",
            "crates/faultscope-collection/Cargo.toml",
            "crates/faultscope-python/Cargo.toml",
            "backends/faultscope-pymatching/Cargo.toml",
            "backends/faultscope-fusion-blossom/Cargo.toml",
        )
        internal_packages = {
            "faultscope-core",
            "faultscope-collection",
            "faultscope-python",
            "faultscope-pymatching-python",
            "faultscope-fusion-blossom-python",
        }
        for path in manifests:
            manifest = _toml(path)
            self.assertIs(manifest["package"]["version"]["workspace"], True, path)
            for dependency_name, dependency in manifest.get("dependencies", {}).items():
                if dependency_name.startswith("faultscope-"):
                    self.assertEqual(dependency["version"], "0.2.11", path)

        locked = _toml("Cargo.lock")["package"]
        locked_versions = {
            package["name"]: package["version"]
            for package in locked
            if package["name"] in internal_packages
        }
        self.assertEqual(set(locked_versions), internal_packages)
        self.assertEqual(set(locked_versions.values()), {"0.2.11"})

    def test_python_metadata_uses_maturin_dynamic_version(self) -> None:
        project = _toml("pyproject.toml")["project"]
        self.assertNotIn("version", project)
        self.assertIn("version", project["dynamic"])
        self.assertEqual(project["requires-python"], ">=3.10")

    def test_test_extra_supports_toml_on_python_3_10(self) -> None:
        project = _toml("pyproject.toml")["project"]
        self.assertIn(
            "tomli>=2; python_version < '3.11'",
            project["optional-dependencies"]["test"],
        )
        for path in ("tests/test_benchmarks.py", "tests/test_release_metadata.py"):
            source = (ROOT / path).read_text()
            self.assertIn("import tomli as tomllib", source, path)

    def test_test_extra_installs_pillow_for_visualization_suite(self) -> None:
        project = _toml("pyproject.toml")["project"]
        test_dependencies = project["optional-dependencies"]["test"]
        self.assertTrue(any(dependency.startswith("pillow") for dependency in test_dependencies))

    def test_mypy_skips_optional_dependency_stubs(self) -> None:
        config = _toml("pyproject.toml")["tool"]["mypy"]
        self.assertIs(config["follow_imports_for_stubs"], True)
        overrides = config["overrides"]
        optional_override = next(
            override for override in overrides if "numpy" in override["module"]
        )
        self.assertEqual(optional_override["follow_imports"], "skip")
        for module in ("matplotlib", "numpy", "pymatching", "scipy"):
            self.assertIn(module, optional_override["module"])
            self.assertIn(f"{module}.*", optional_override["module"])

    def test_runtime_versions_come_from_package_metadata(self) -> None:
        native_lib = (ROOT / "crates/faultscope-python/src/lib.rs").read_text()
        self.assertIn('env!("CARGO_PKG_VERSION")', native_lib)
        for path in (
            "backends/faultscope-pymatching/src/faultscope_pymatching/__init__.py",
            "backends/faultscope-fusion-blossom/src/faultscope_fusion_blossom/__init__.py",
        ):
            source = (ROOT / path).read_text()
            self.assertIn("_distribution_version", source, path)
            self.assertNotIn('__version__ = "0.1.0"', source, path)

    def test_only_library_crates_are_publishable(self) -> None:
        core = _toml("crates/faultscope-core/Cargo.toml")["package"]
        collection = _toml("crates/faultscope-collection/Cargo.toml")["package"]
        self.assertIsNot(core.get("publish", True), False)
        self.assertIsNot(collection.get("publish", True), False)
        for path in (
            "crates/faultscope-python/Cargo.toml",
            "backends/faultscope-pymatching/Cargo.toml",
            "backends/faultscope-fusion-blossom/Cargo.toml",
        ):
            package = _toml(path)["package"]
            self.assertIs(package["publish"], False, path)

    def test_release_documents_and_published_crate_metadata_exist(self) -> None:
        for path in ("LICENSE", "README.md", "CHANGELOG.md", "docs/release.md"):
            self.assertTrue((ROOT / path).is_file(), path)
        for crate in ("faultscope-core", "faultscope-collection"):
            package = _toml(f"crates/{crate}/Cargo.toml")["package"]
            self.assertTrue(package["description"], crate)
            self.assertEqual(package["readme"], "README.md")
            self.assertEqual(package["documentation"], f"https://docs.rs/{crate}")
            self.assertTrue((ROOT / "crates" / crate / "LICENSE").is_file())

    def test_published_path_dependencies_have_registry_versions(self) -> None:
        collection = _toml("crates/faultscope-collection/Cargo.toml")
        dependency = collection["dependencies"]["faultscope-core"]
        self.assertEqual(dependency["version"], "0.2.11")
        self.assertEqual(dependency["path"], "../faultscope-core")

    def test_all_python_extensions_use_abi3_py310(self) -> None:
        manifests = (
            "crates/faultscope-python/Cargo.toml",
            "backends/faultscope-pymatching/Cargo.toml",
            "backends/faultscope-fusion-blossom/Cargo.toml",
        )
        for path in manifests:
            pyo3 = _toml(path)["dependencies"]["pyo3"]
            self.assertIn("abi3-py310", pyo3["features"], path)

    def test_backend_python_package_requires_compatible_faultscope(self) -> None:
        for path in (
            "backends/faultscope-pymatching/pyproject.toml",
            "backends/faultscope-fusion-blossom/pyproject.toml",
        ):
            project = _toml(path)["project"]
            self.assertIn("faultscope>=0.2.11,<0.3", project["dependencies"], path)

    def test_native_decoder_public_constants_are_coherent_for_abi_v4(self) -> None:
        from faultscope import _native
        from faultscope.backends import NATIVE_DECODER_PLUGIN_ABI
        from faultscope.backends.registry import NATIVE_DECODER_ENTRY_POINT_GROUP

        self.assertEqual(NATIVE_DECODER_PLUGIN_ABI, "faultscope.native_decoder_plugin.v4")
        self.assertEqual(_native.NATIVE_DECODER_PLUGIN_ABI_VERSION, 4)
        self.assertEqual(_native.NATIVE_DECODER_PLUGIN_ABI, NATIVE_DECODER_PLUGIN_ABI)
        self.assertEqual(
            _native.NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
            "faultscope.native_decoders",
        )
        self.assertEqual(
            NATIVE_DECODER_ENTRY_POINT_GROUP,
            _native.NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
        )

    def test_native_decoder_docs_define_only_the_v4_runtime_contract(self) -> None:
        abi = (ROOT / "docs/native_decoder_abi.md").read_text()
        development = (ROOT / "docs/decoder_development.md").read_text()
        decoding = (ROOT / "docs/guides/decoding.md").read_text()
        contract = " ".join(abi.split())

        for required in (
            "faultscope.native_decoder_plugin.v4",
            "faultscope.native_decoders",
            "__faultscope_native_decoder_capsule__",
            "FaultScopeNativeDecoderFactoryV4",
            "FaultScopeNativeDecoderWorkerV4",
            "FaultScopeNativeDetectorBatchViewV4",
            "FaultScopeNativeCorrectionBatchMutViewV4",
            "FaultScopeNativeDecoderU32SliceV1",
            "| `FaultScopeNativeDecoderFactoryV4` | 80 | 8 |",
            "| `FaultScopeNativeDecoderWorkerV4` | 32 | 8 |",
            "thread-safe factory flag",
            "output descriptor and its capacity",
            "drop_factory_state",
            "drop_worker_state",
            "must not unwind across the C boundary",
            "exclusive mutable objects",
            "batch_formats",
            "MASKS",
            "PACKED",
            "EVENTS",
            "GF(2)",
            "reserved",
            "padding",
            "DemAttributionTrace",
            "V3 to V4 migration",
        ):
            with self.subTest(required=required):
                self.assertTrue(
                    required in contract,
                    f"docs/native_decoder_abi.md is missing {required!r}",
                )

        docs = abi + "\n" + development + "\n" + decoding
        for obsolete in (
            "FaultScopeNativeDecoderFactoryV3 {",
            "FaultScopeNativeDecoderWorkerV3 {",
            "fn supports_packed_batch",
            "fn decode_packed_batch(",
            "fn decode_detector_event_batch(",
        ):
            self.assertNotIn(obsolete, docs, obsolete)

    def test_native_decoder_development_uses_factory_worker_contract(self) -> None:
        development = (ROOT / "docs/decoder_development.md").read_text()
        prose = " ".join(development.split())

        # The guide explains integration; exact signatures live in the linked source and ABI.
        for required in (
            "`NativeBatchDecoder`",
            "NativeDecoderFactory: Send + Sync",
            "NativeDecoderWorker: Send",
            "`&mut self`",
            "`batch_formats`",
            "`decode_batch`",
            "`__faultscope_native_decoder_capsule__()`",
            "`strong_id_payload()`",
            "(native_decoder_abi.md)",
            "native_decoder_abi.md#callback-contract",
            "native_decoder_abi.md#ownership-and-lifetime",
            "crates/faultscope-core/src/decoder.rs",
            "crates/faultscope-python/src/decoder_api.rs",
        ):
            with self.subTest(required=required):
                self.assertTrue(
                    required in prose,
                    f"docs/decoder_development.md is missing {required!r}",
                )

        for obsolete in (
            "faultscope_core::NativeBatchDecoder",
            "impl NativeBatchDecoder for",
            "Arc<dyn NativeBatchDecoder",
            "fn decode_batch(\n        &self,",
            "fn decode_packed_batch(\n    &self,",
            "fn decode_packed_batch(\n        &mut self,",
            "fn decode_detector_event_batch(\n        &mut self,",
        ):
            self.assertNotIn(obsolete, development, obsolete)

    def test_native_decoder_guides_document_backend_availability(self) -> None:
        decoding = (ROOT / "docs/guides/decoding.md").read_text()
        for required in (
            "NativePyMatchingDecoder",
            "NativeFusionBlossomDecoder",
            "`mwpm`",
            "`bpdecoder`",
            "`bposd`",
        ):
            with self.subTest(required=required):
                self.assertTrue(
                    required in decoding,
                    f"docs/guides/decoding.md is missing {required!r}",
                )

        for path in ("docs/native_decoder_abi.md", "docs/decoder_development.md"):
            with self.subTest(path=path):
                self.assertTrue(
                    "guides/decoding.md" in (ROOT / path).read_text(),
                    f"{path} must link to the decoder installation and availability guide",
                )


if __name__ == "__main__":
    unittest.main()
