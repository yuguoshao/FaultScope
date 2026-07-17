# faultscope-core

`faultscope-core` contains FaultScope's Python-independent circuit, detector
error model, bit-packed sampler, hotspot, and native decoder primitives.

The crate is pre-1.0. API compatibility is not guaranteed between releases,
including patch releases. Import the supported high-level types from the crate
root; low-level Pauli, row, word, and validation helpers are private.

`PauliFrame` owns x/z data validated at construction. `ConcreteStabilizer`
supports zero-state construction and invariant-preserving fallible operations;
its lazy measurement entry point requests randomness only for a genuinely
random Pauli measurement. Sparse targets use borrowed validated views, and
`ConcreteStabilizer::apply_pauli_event` atomically updates a matching frame.
