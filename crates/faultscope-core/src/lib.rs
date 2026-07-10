//! Rust core types for FaultScope.
//!
//! This crate is the Python-independent home for FaultScope's circuit, detector
//! error model, and hotspot data model.  The PyO3 crate is responsible for
//! adapting these types to Python and for the remaining Python callback
//! integration while the algorithms are migrated here.

mod compile;
mod decoder;
mod dem;
mod dem_problem;
mod dem_sampling;
mod expr;
mod hotspot;
mod mask;
mod model;
mod packed;
mod pauli;
mod rng;
mod sampling;
mod stabilizer;

pub use compile::{compile_runtime_operations, CompiledCircuit};
#[cfg(feature = "decoder-fusion-blossom")]
pub use decoder::NativeFusionBlossomDecoder;
pub use decoder::{
    logical_residual_loss_mask_native, CorrectionMaskBatch, DetectorEventShotBatchView,
    DetectorMaskBatchView, FaultScopeNativeCorrectionMaskBatchMutViewV1,
    FaultScopeNativeDecoderI64SliceV1, FaultScopeNativeDecoderMaskMutViewV1,
    FaultScopeNativeDecoderMaskViewV1, FaultScopeNativeDecoderStatusV1,
    FaultScopeNativeDecoderStringViewV1, FaultScopeNativeDecoderV1,
    FaultScopeNativeDetectorEventShotBatchViewV1, FaultScopeNativeDetectorMaskBatchViewV1,
    FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeNativePackedObservableShotBatchMutViewV1, NativeBatchDecoder,
    NativeGraphlikeDetectorCopyDecoder, NativeNoCorrectionDecoder, PackedDetectorShotBatchView,
    PackedObservableShotBatch, NATIVE_DECODER_PLUGIN_ABI_NAME, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_CAPSULE_METHOD, NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
    NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP, NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
    NATIVE_DECODER_PLUGIN_STATUS_ERROR, NATIVE_DECODER_PLUGIN_STATUS_OK,
};
pub use dem::{
    collect_dem_event_plan, detectors_from_circuit, generate_dem_edges,
    generate_dem_edges_from_plan, observables_from_circuit, DemEventPlan,
    DetectorErrorModelGenerator, GeneratedDemEdge, GeneratedDemEdgeRef, LazyDetectorErrorModel,
};
pub use dem_problem::{
    log_likelihood_ratio, BinaryLinearDecodingProblem, GraphlikeDecodingProblem, GraphlikeEdge,
    IndexedDem, IndexedDemEdge, SparseBinaryMatrix,
};
pub use dem_sampling::{
    build_dem_location_groups, run_dem_batch, run_dem_detector_event_shot_batch,
    run_dem_packed_shot_batch, DemHotspotEstimator, DetectorEventDemShotBatch, PackedDemShotBatch,
};
pub use expr::Expr;
pub use hotspot::{compute_dem_estimate, compute_packed_estimate};
pub use mask::{word_count, Mask};
pub use model::{
    Circuit, DemBatch, DemEvent, DemHotspotEstimate, DemLocationGroup, DemSamplerEdge, Detector,
    DetectorErrorEdge, DetectorErrorModel, DetectorGraphEstimate, DetectorGraphKey,
    HotspotEstimate, LogicalObservable, NoiseLocation, NoiseModel, NpError, NpResult, Operation,
    PackedBatch, RunOperation, TagValue,
};
pub use packed::{run_packed_sample, FaultScopeSimulator, RuntimeState};
pub use pauli::{
    coeff_bit, highest_bit, multiply_concrete_rows, multiply_symbolic_rows, pauli_product,
    pauli_to_xz, solve_row_span, sparse_pauli_to_xz, support_to_words, symplectic_product,
    xor_words, xz_to_pauli,
};
pub use rng::SmallRng;
pub use sampling::{
    bernoulli_mask, choose_weighted_event, compile_pauli_channel_events, for_each_bernoulli_event,
    positive_unit_f64, random_bit_mask, set_shot_bit, CompiledPauliEvent,
    TWO_QUBIT_DEPOLARIZING_EVENTS,
};
pub use stabilizer::{
    frame_apply_cx, frame_apply_cz, frame_apply_h, frame_apply_pauli_string, frame_apply_s,
    frame_apply_swap, frame_measurement_flip_bits, ConcreteStabilizer, SymbolicStabilizer,
};
