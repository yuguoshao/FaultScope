# Development and Benchmarks

Use this page when changing FaultScope, measuring performance, or editing the
documentation. To run your first simulation, start with
[Getting Started](getting_started.md).

## Set up a development environment

Use Python 3.10+ and Rust 1.85+. Download or clone the
[FaultScope source](https://github.com/yuguoshao/FaultScope), then run these
commands from the repository root; on Windows, replace the activation command with
`.venv\Scripts\Activate.ps1`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "maturin>=1.7,<2"
maturin develop --extras test --locked
```

`maturin develop` builds the extension for the active environment and makes the
source package importable. Run it again after changing Rust code. Python source
edits take effect without rebuilding the extension.

For compiler, lint, and test dependencies, see
[the project configuration](https://github.com/yuguoshao/FaultScope/blob/main/pyproject.toml).
Optional native decoder packages have their own
[source build steps](guides/decoding.md#build-a-backend-from-source-for-development).

## Run checks

Run the checks relevant to your change. The
[CI workflow](https://github.com/yuguoshao/FaultScope/blob/main/.github/workflows/ci.yml)
defines the complete test matrix, minimum Rust version, and package smoke checks.

Rust:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Python tests and types:

```bash
python -m unittest discover -s tests -q
python -m pytest -q
python -m mypy
```

Python style and generated bindings:

```bash
python -m ruff check faultscope tests benchmarks scripts backends/faultscope-pymatching/src backends/faultscope-fusion-blossom/src
python -m ruff format --check faultscope tests benchmarks scripts backends/faultscope-pymatching/src backends/faultscope-fusion-blossom/src
python scripts/generate_native_stub.py --check
python -m mypy.stubtest faultscope._native --ignore-missing-stub
```

Public API changes also need updates to the reference and deliberate review of
the export contract. Native decoder changes must satisfy the
[ABI contract](native_decoder_abi.md); see the
[decoder development checklist](decoder_development.md#development-checklist).

## Measure performance

Build in release mode before measuring:

```bash
maturin develop --release --extras test --locked
```

FaultScope processes batches using packed bit arrays in Rust. Reusing compiled
simulators avoids paying compilation costs for every batch. Measure compilation,
sampling, decoding, and sensitivity estimation separately when the distinction
matters to your application.

Start with a small run, then increase the workload:

```bash
python benchmarks/sampling_throughput.py --distances 3 --rounds 3 --shots 1024 --repeats 2
python benchmarks/surface_code_decoder_performance.py --distances 3 --rounds 3 --shots 1024
```

The sampling script compares packed measurement sampling with Stim. The decoder
script samples a canonical DEM and compares decoding paths; it does not measure
the Forward circuit workflow from the getting-started tutorial. Each Stim
`error` instruction remains one sampled event. Separator groups guide decoder
construction using an uncorrelated graphlike approximation.

Optional native decoder paths are reported as skipped when their packages are
unavailable. Use `--help` for settings and report package versions, build mode,
hardware, circuit parameters, shot counts, and enabled paths with any timings.
For native decoder comparisons, `--same-seed-across-paths` shares sampling seeds;
`--split-native-baseline` separates the no-decoder baseline from added decoder
work. Equal seeds do not imply identical samples from different samplers.

Other benchmarks cover specific stages:

| Stage | Script in `benchmarks/` |
| --- | --- |
| Circuit compilation | `compiler_throughput.py` |
| DEM generation and sampling | `dem_throughput.py` |
| Noise sensitivity estimation | `hotspot_throughput.py` |
| Parallel collection | `collection_throughput.py` |
| Native decoder integration | `native_decoder_fast_path.py` |
| Surface-code threshold experiments | `surface_code_threshold.py` |

Browse the [benchmark sources](https://github.com/yuguoshao/FaultScope/tree/main/benchmarks)
for exact workloads. Benchmark target labels and individual runs are not general
speed guarantees.

## Build the documentation

Install the same documentation dependency used in CI:

```bash
python -m pip install mkdocs-material
python -m mkdocs serve
```

Open the local URL printed by MkDocs. To check all published pages without
starting a server:

```bash
python -m mkdocs build --strict
```

Keep guides focused on complete tasks and put exact contracts in the API or ABI
reference. Run changed examples, check relative links and section anchors, and
inspect both code blocks and equations in the preview. Archived design plans in
`docs/superpowers/` stay in the repository but are excluded from the published
site and its search index.
