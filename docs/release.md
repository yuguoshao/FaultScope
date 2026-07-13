# Release And Compatibility

FaultScope uses one workspace version for the Python distribution and the
published Rust libraries. The current release line is `0.2.x`.

## Compatibility Contract

- Python names listed by public-module `__all__` values are stable within
  `0.2.x`. Private modules and `_`-prefixed names are not public API.
- The root exports of `faultscope-core` and `faultscope-collection` are the Rust
  compatibility boundary. Their source modules are private.
- Compatible additions may ship in a patch release. Removal or renaming is
  deprecated first and deferred to a later minor release.
- The collection CSV header, strong-id input encoding, and native decoder plugin
  ABI are covered by golden compatibility tests.

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
- `faultscope-fusion-blossom` remains a source-install beta for the `0.2.0`
  release line.

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
