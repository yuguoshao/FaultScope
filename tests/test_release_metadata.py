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
        self.assertGreaterEqual(ci.count("python -m venv .venv"), 2)
        self.assertGreaterEqual(
            ci.count('.venv/bin/python -m pip install --upgrade pip "maturin>=1.7,<2"'),
            2,
        )
        self.assertGreaterEqual(
            ci.count(".venv/bin/maturin develop --extras test --locked"),
            2,
        )
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
        self.assertEqual(package["version"], "0.1.0")
        self.assertEqual(package["rust-version"], "1.85")

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
        self.assertEqual(dependency["version"], "0.1.0")
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
        project = _toml("backends/faultscope-pymatching/pyproject.toml")["project"]
        self.assertIn("faultscope>=0.1,<0.2", project["dependencies"])


if __name__ == "__main__":
    unittest.main()
