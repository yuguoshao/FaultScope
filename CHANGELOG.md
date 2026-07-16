# Changelog

All notable changes to FaultScope are documented here. FaultScope is pre-1.0:
unless a contract is explicitly versioned (such as the native decoder ABI),
patch releases may make breaking API changes.

## [0.2.2] - Unreleased

### Added

- Rust-native circuit, detector-error-model, hotspot, and decoder primitives.
- Native logical error-rate collection with deterministic multi-worker scheduling.
- Immutable Python `Collector` configuration, CSV/resume, typed streaming progress,
  explicit raw/accepted logical rates, decoder fanout, and threshold analysis.
- Pure factory/worker native decoder plugin ABI v2 and optional PyMatching and
  fusion-blossom backends.

### Changed

- Pauli and stabilizer validation now occurs once at Python and Rust public
  boundaries. Validated owning state and borrowed sparse-Pauli views are used
  by internal hot paths, avoiding repeated full-width scans and allocations for
  parsed targets.
- Rust now exposes owning `PauliFrame` and zero-state-only
  `ConcreteStabilizer`; public state mutations are fallible, and lazy Pauli
  measurement requests randomness only for nondeterministic outcomes.
- Removed the low-level Rust Pauli/frame free-function exports and the Python
  `multiply_pauli_rows`, `symplectic_product`, and `sparse_pauli_to_xz`
  helpers. Python `StabilizerState` no longer accepts raw tableau matrices.
- The Rust forward API now compiles and executes `SamplerProgram` directly.
  The redundant `CompiledCircuit`, `compile_runtime_operations*`, `RunOperation`,
  and legacy string-keyed packed executor have been removed.
- Forward and detector-error-model compilation now share one integer-indexed
  expanded program; DEM event and measurement plans no longer rebuild or clone
  string-key maps internally.
- Noise-location labels and hotspot tags are interned once into dense IDs;
  forward/DEM sampling and hotspot aggregation materialize strings only in
  public result objects.
- `DemHotspotEstimator` is now the single compiled DEM sampling path; obsolete
  standalone string-edge grouping/sampling helpers and `PackedBatch` were
  removed from the pre-1.0 Rust API.
- DEM propagation now indexes fault-event ranges by noise ID and measurements
  by measurement ID, shares one flip-mask assembly layer across product and
  fallback kernels, and keeps sampling-only edges free of location metadata.
- DEM sampling and materialized detector-error models now use only
  `DetectorErrorEdge`; the redundant generated/sampler edge aliases and public
  lazy edge-reference type were removed.
- Packed logical residual counting now has one core implementation with a
  reusable linear-time observable-ID alignment table for reordered decoders.
- Forward/DEM compilation share the same integer-indexed observable type.
- Stim text is always imported through the compact structured IR; the legacy
  line-by-line importer and its eager string-key operations were removed.
- Added `generate_dem_edges_from_event_plan`; the obsolete compatibility
  overload with an unused operations argument was removed.
- Python forward decoders can bulk-select only their required packed
  measurement masks; the repetition-code decoder now reuses that selection and
  evaluates all shots with bit-parallel integer operations.

### Fixed

- Collection strong ids now use a versioned canonical source encoding and
  configuration-aware decoder fingerprints, preventing resume and aggregation
  from merging statistics produced by different same-named decoders.
- Pauli-frame and stabilizer-state APIs now reject mismatched, non-binary,
  negative, out-of-range, or duplicate targets before computation, RNG use, or
  tableau mutation.
- Python noise-model Pauli events validate state/frame widths and sparse targets
  before atomically updating both objects.
- Noise models now reject empty, identity-only, invalid, or non-finite Pauli
  configurations before sampling. Python and native weighted channels share
  boundary semantics, and two-qubit depolarizing noise is fixed to its
  canonical 15-event distribution.
- Generated type stubs now mark factory-only native classes, including
  `StabilizerState`, as unavailable for direct construction.

[0.2.2]: https://github.com/yuguoshao/FaultScope/releases/tag/v0.2.2
