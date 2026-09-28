# Release and Compatibility

The Python distribution and Rust libraries share workspace version `0.2.11`.
Install from PyPI with `pip install faultscope`; the
[installation guide](getting_started.md#installation) includes optional dependencies
and a complete example.
The [changelog](https://github.com/yuguoshao/FaultScope/blob/main/CHANGELOG.md)
records release-specific changes.

## Compatibility Contract

- FaultScope is pre-1.0. Python and Rust APIs may change between releases,
  including patch releases. Renames and removals may ship without deprecated
  aliases.
- Use public Python modules and Rust crate-root exports. Private modules and
  names beginning with `_` are implementation details.
- `tests/public_api_contract.json` records the current Python exports for review;
  it does not guarantee compatibility with older releases.
- The native decoder plugin ABI is versioned independently. Version 0.2.11
  accepts **ABI V4 only**. Plugins built for V1–V3 must be rebuilt; see the
  [migration contract](native_decoder_abi.md#v3-to-v4-migration).
- Collection CSV formats and task identities can also change. Check the upgrade
  notes before resuming an older experiment.

## Supported Toolchains

| Component | Supported versions or build targets |
| --- | --- |
| Python | 64-bit CPython 3.10–3.14 |
| Rust, for source builds only | 1.85 or newer; not needed to install wheels |
| Linux wheels | glibc 2.17+, x86_64 and aarch64 |
| macOS wheels | macOS 11+, x86_64 and arm64 |
| Windows wheels | x86_64 |

The main extension and native PyMatching backend use the CPython stable ABI,
with `cp310-abi3` in their wheel filenames. Pip selects the wheel for your Python
interpreter and platform. Installing these wheels does not compile Rust or C++
code.

<span id="published-packages"></span>

## Distribution Packages

| Package | Role |
| --- | --- |
| `faultscope` | Main Python package and Rust extension |
| `faultscope-pymatching` | Optional native PyMatching backend |
| `faultscope-fusion-blossom` | Optional native fusion-blossom backend; source-install beta for 0.2.x |
| `faultscope-core` | Rust circuit, sampling, DEM, and sensitivity library |
| `faultscope-collection` | Rust logical error-rate collection library |

Install the main package with `pip install faultscope` and the optional native
PyMatching backend with `pip install faultscope_pymatching`.
Official backend packages require `faultscope>=0.2.11,<0.3`. See
[Decoding](guides/decoding.md#native-backends) for backend installation.
The fusion-blossom backend remains a source-install beta for 0.2.x.

<span id="upgrading-stored-results"></span>

## Upgrading Collection Runs

Collection currently uses CSV schema v3 and Python-generated strong-id schema
v5. A strong id identifies a resumable task. CSV v3 cannot read or append CSV v2
files. Start a new resume file when upgrading from those older formats.

Version 0.2.11 uses the `explicit-v1` seed policy: an explicit task seed takes
precedence over the run seed; metadata and decoder identities no longer salt it.
Strong-id v5 includes the task seed and seed policy, so it does not reuse task
identities from the previous policy. Rust callers that supply their own strong
ids must distinguish the old and new policies themselves.

Old identity-derived seed sequences are not reproduced. Reproducibility also
depends on the sampling input, batch settings, implementation, and dependencies.
See [seeds and resume](guides/collection.md#seeds-and-resume) for current behavior.

## Release Process

Before the first PyPI release, register a pending trusted publisher for each
Python package. Both use the GitHub repository `yuguoshao/FaultScope` and workflow
filename `release.yml`, with these environment names:

| PyPI project | GitHub environment |
| --- | --- |
| `faultscope` | `release` |
| `faultscope-pymatching` | `release-pymatching` |

PyPI does not allow two pending publishers with the same repository, workflow,
and environment. Use the explicit environment names above instead of leaving
the field as **Any**, and create the matching environments in the GitHub
repository settings. The workflow at the release tag must use these same names;
editing a branch does not update the workflow stored in an existing tag.

Maintainers follow the
[release workflow](https://github.com/yuguoshao/FaultScope/blob/main/.github/workflows/release.yml):

1. Update the changelog and make the `vX.Y.Z` tag match the workspace version.
2. Run the Rust, Python, documentation, minimum-version, and packaging checks.
3. Push the tag to build and test artifacts and create a draft GitHub release.
4. To publish registries, manually dispatch the workflow for that tag with
   `publish_registries` enabled. Approve the `release` and `release-pymatching`
   deployments if their environments require approval.
5. Publish `faultscope-core`, wait for the crates.io index, then publish
   `faultscope-collection`. Publish the main Python distribution before
   `faultscope-pymatching`.

Ordinary branches, pull requests, and tag pushes do not publish to registries.
Trusted-publisher configuration and the crates.io token are maintainer-managed
prerequisites.
