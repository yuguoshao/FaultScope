# Release And Compatibility

FaultScope uses one workspace version for the Python distribution and the
published Rust libraries. The current workspace version is `0.2.9`.

## Compatibility Contract

- FaultScope is pre-1.0. Python and Rust API compatibility is not promised
  between releases, including patch releases; removals and renames may ship
  without deprecated aliases.
- Public Python modules and Rust crate-root exports remain the intended import
  surfaces. Private modules and `_`-prefixed names are implementation details.
- `tests/public_api_contract.json` records the current Python export surface for
  deliberate review; it is not a historical compatibility guarantee.
- The native decoder plugin ABI is independently versioned. ABI V4 is the only
  runtime contract in `0.2.9`; V1–V3 plugins are rejected and must be rebuilt.
- Collection CSV and strong-id regression tests detect changes, but pre-1.0
  releases may deliberately revise those formats with release notes.

## Supported Toolchains

| Component | Supported versions |
| --- | --- |
| Python | CPython 3.10–3.14 |
| Rust | 1.85 or newer |
| Linux wheels | manylinux x86_64 and aarch64 |
| macOS wheels | macOS 11+, x86_64 and arm64 |
| Windows wheels | x86_64 |

The main extension and official backend extensions use the CPython stable ABI
with an `abi3-py310` wheel tag.

## Published Packages

- PyPI: `faultscope` and `faultscope-pymatching`.
- crates.io: `faultscope-core`, then `faultscope-collection`.
- `faultscope-fusion-blossom` remains a source-install beta for the `0.2.x`
  release line.

Official backend packages require `faultscope>=0.2.9,<0.3` so they are not
installed against an older main package with a different pre-1.0 API surface.

## Release Process

1. Update `CHANGELOG.md` and ensure the `vX.Y.Z` tag matches the workspace
   version.
2. Run Rust formatting, clippy, tests, docs, MSRV, crate packaging, Python
   typing, tests, and wheel smoke checks.
3. Build and test release artifacts in GitHub Actions and create a draft GitHub
   release.
4. Approve the protected `release` environment.
5. Publish `faultscope-core`, wait for the crates.io index, then publish
   `faultscope-collection`.
6. Publish the main PyPI distribution before `faultscope-pymatching`.

The repository workflow does not publish from ordinary branches or pull
requests. Registry trusted-publisher configuration and the crates.io token are
maintainer-owned external prerequisites.
