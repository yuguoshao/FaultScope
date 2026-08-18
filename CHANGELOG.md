# Changelog

All notable changes to FaultScope are documented here. FaultScope is pre-1.0:
unless a contract is explicitly versioned (such as the native decoder ABI),
patch releases may make breaking API changes.

## [0.2.9] - Unreleased

### Added

- Rust-native circuit, detector-error-model, hotspot, and decoder primitives.
- Native logical error-rate collection with deterministic multi-worker scheduling.
- Public Rust Forward collection tasks, logical-error functions, hotspot
  collection, and mode-neutral options/stats/counter aliases.
- Immutable Python `Collector` configuration, CSV/resume, typed streaming progress,
  explicit raw/accepted logical rates, decoder fanout, and threshold analysis.
- Factory/worker native decoder plugin ABI V4 and optional PyMatching and
  fusion-blossom backends.
- Optional graphlike decomposition hints for compiling canonical DEM hyperedges
  without changing the sampled detector/observable joint distribution.
- Model-bound `GraphlikeDecompositionHints` and `GeneratedDetectorErrorModel`
  package the canonical sampling DEM with optional decoder-only structure;
  native generators expose the forward-compatible `generate_artifact()` API.

### Changed

- `faultscope.collection` is now Forward-only: `CollectionTask(circuit=...)`
  executes the packed circuit runtime directly, and `CollectionTask(dem=...)`
  raises a migration error. Explicit DEM sampling remains available only through
  the library-only `faultscope.collection.dem` module and is not exposed by the
  collection CLI or top-level exports.
- Circuit-to-DEM generation now follows Stim's disjoint-error policy. Uniform
  one- and two-qubit depolarizing channels are reparameterized exactly into
  independent DEM mechanisms, and one-qubit `PauliChannel` first attempts
  Stim's independent X/Y/Z conversion. A multi-component channel that cannot
  take that exact path is rejected by default and requires explicit
  `approximate_disjoint_errors=True` (or a component-probability threshold).
  Matching propagated effects are first combined by disjoint probability sum;
  distinct effect classes are then treated as independent edges.
- The surface-code decoder benchmark now samples one canonical FaultScope DEM
  edge per Stim `error` instruction and passes separator groups only to decoder
  construction through a typed artifact. Only instructions with actual
  multi-component separators allocate sparse hints. Native backend summaries distinguish canonical DEM,
  pre-merge graphlike, and post-merge solver edge counts.

- Native decoder plugins now use the breaking ABI V4. Factories publish a
  stable ordered Masks/Packed/Events preference list and workers expose one
  tagged `decode_batch` callback. Masks input returns mask-major corrections;
  Packed and Events input return packed rows. V1–V3 capsules/manifests are
  rejected without an adapter.
- DEM syndrome layout is independent of hotspot attribution. Sampling produces
  the decoder's preferred layout directly while FaultScope records an optional
  private edge-event sidecar; ordinary sampling allocates no attribution trace.
  Detailed counting now uses one Logical or Decoder plan and a shared residual,
  discard, combo, and loss-mask implementation across all formats.
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
- The compiler throughput benchmark now preserves explicit Stim `REPEAT`
  blocks by default so it exercises the native Rust loop path; flattened input
  remains available as an explicit stress-test mode.
- Collection decoder fan-out now compiles each shared Forward circuit once and
  reuses its immutable sampler across decoder task views. A string decoder may
  build one cached DEM for its static problem, but every shot remains Forward;
  the isolated legacy DEM collector separately reuses each shared DEM sampler.
- Collection sampling identities now use domain-separated source and decoder
  digests (sampling schema v2, strong-id schema v4). Existing resume rows use
  older ids and are intentionally not reused; task-derived random streams can
  therefore differ after upgrading.
- Native decoder detector layouts must contain unique ids. Built-in
  constructors, composite children, ABI V4 factories, and DEM sampling plans
  now reject duplicate detector ids instead of silently selecting one column.
- Added `generate_dem_edges_from_event_plan`; the obsolete compatibility
  overload with an unused operations argument was removed.
- Python forward decoders can bulk-select only their required packed
  measurement masks; the repetition-code decoder now reuses that selection and
  evaluates all shots with bit-parallel integer operations.

### Fixed

- Circuit-generated DEM sampling now preserves the detector/observable joint
  distribution of supported depolarizing channels instead of independently
  sampling the mutually exclusive Pauli marginals. The surface-code Stim DEM
  converter also preserves detector coordinates declared after error
  instructions.
- Graphlike decoding problems now treat edge probability as the authoritative
  value and derive the ABI/matching weight in core. The compatibility edge DTO
  no longer accepts an independently writable weight that could be non-finite
  or disagree with backend behavior.
- Rust `SamplerProgram` internals and `FaultScopeSimulator::program` are now
  private and exposed through read-only getters. Safe downstream callers can no
  longer mutate compiler-validated qubit, noise, measurement, observable, or
  capacity indices into layouts that panic in the packed runtime.
- Collection completion now uses one predicate across single-task, fixed,
  adaptive, resumed, and hotspot runs. A zero error limit with zero minimum
  shots returns empty statistics without starting workers. Task-level
  `CollectionOptions` now distinguish omitted fields from explicit defaults and
  `None`, and invalid boolean, fractional, or non-finite numeric options fail at
  Python construction instead of being truncated or reaching native code.
- Hotspot estimation now rejects batches without recorded event masks, batches
  from another compiled estimator, zero-shot states, and invalid loss-mask
  widths with `NpError`/`ValueError` instead of panicking or silently
  substituting zero event masks. Invariant-bearing Rust batch/program fields
  are private and exposed through read-only getters, so aggregation validates
  once at its public boundary and uses a trusted internal hot path.
- Collection counter schemas are now explicit and versioned. The v3 strong-id
  and CSV resume contract isolates all count-flag combinations, rejects v2 CSV
  reads/appends, preserves fixed zero counters, and validates custom stop keys
  before sampling or worker startup.
- Collection strong ids now use a versioned canonical source encoding and
  configuration-aware decoder fingerprints, preventing resume and aggregation
  from merging statistics produced by different same-named decoders.
- Collection resume now rebinds validated historical statistics to the current
  task identity before completion checks, so partial and already-complete CSV
  resumes return the caller's current display label instead of the strong hash.
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

[0.2.9]: https://github.com/yuguoshao/FaultScope/releases/tag/v0.2.9
