//! Rust core types for FaultScope.
//!
//! This crate is the Python-independent home for FaultScope's circuit, detector
//! error model, and hotspot data model.  The PyO3 crate is responsible for
//! adapting these types to Python and for the remaining Python callback
//! integration while the algorithms are migrated here.

mod compile;
mod decoder;
mod dem;
mod dem_canonical;
mod dem_problem;
mod dem_sampling;
mod expr;
mod hotspot;
mod labels;
mod mask;
mod model;
mod packed;
mod pauli;
mod program;
mod rng;
mod sampling;
mod stabilizer;

pub use compile::compile_sampler_program_ref;
#[cfg(feature = "decoder-fusion-blossom")]
pub use decoder::NativeFusionBlossomDecoder;
pub use decoder::{
    convert_detector_batch, logical_residual_loss_mask_native, mask_corrections_to_packed,
    packed_corrections_to_masks, packed_residual_failure_count, select_detector_batch_format,
    validate_decoder_batch_formats, validate_decoder_detector_ids, validate_decoder_observable_ids,
    CorrectionMaskBatch, DecoderCorrectionBatch, DetectorBatch, DetectorBatchFormat,
    DetectorBatchView, DetectorEventShotBatchView, DetectorMaskBatchView,
    FaultScopeNativeCorrectionBatchMutViewV4, FaultScopeNativeCorrectionBatchPayloadV4,
    FaultScopeNativeCorrectionMaskBatchMutViewV1, FaultScopeNativeDecoderFactoryV4,
    FaultScopeNativeDecoderI64SliceV1, FaultScopeNativeDecoderMaskMutViewV1,
    FaultScopeNativeDecoderMaskViewV1, FaultScopeNativeDecoderStatusV1,
    FaultScopeNativeDecoderStringViewV1, FaultScopeNativeDecoderU32SliceV1,
    FaultScopeNativeDecoderWorkerV4, FaultScopeNativeDetectorBatchPayloadV4,
    FaultScopeNativeDetectorBatchViewV4, FaultScopeNativeDetectorEventShotBatchViewV1,
    FaultScopeNativeDetectorMaskBatchViewV1, FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeNativePackedObservableShotBatchMutViewV1, NativeCompositeDecoder,
    NativeDecoderFactory, NativeDecoderWorker, NativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder, PackedDetectorShotBatchView, PackedObservableShotBatch,
    NATIVE_DECODER_BATCH_FORMAT_EVENTS, NATIVE_DECODER_BATCH_FORMAT_MASKS,
    NATIVE_DECODER_BATCH_FORMAT_PACKED, NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE,
    NATIVE_DECODER_PLUGIN_ABI_NAME, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_CAPSULE_METHOD, NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
    NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP, NATIVE_DECODER_PLUGIN_STATUS_ERROR,
    NATIVE_DECODER_PLUGIN_STATUS_OK,
};
pub use dem::{
    collect_dem_event_plan, collect_dem_event_plan_with_options, detectors_from_circuit,
    generate_dem_edges, generate_dem_edges_from_event_plan, generate_dem_edges_with_options,
    observables_from_circuit, DemEventPlan, DemGenerationOptions, DetectorErrorModelGenerator,
    LazyDetectorErrorModel, ValidatedDemCircuit,
};
pub use dem_problem::{
    log_likelihood_ratio, BinaryLinearDecodingProblem, FaultScopeNativeGraphlikeEdgeV1,
    FaultScopeNativeGraphlikeProblemV1, GraphlikeDecodingProblem, GraphlikeDecompositionComponent,
    GraphlikeEdge, GraphlikeEdgeRef, IndexedDem, IndexedDemEdge, SparseBinaryMatrix,
    NATIVE_GRAPHLIKE_PROBLEM_ABI_NAME, NATIVE_GRAPHLIKE_PROBLEM_ABI_VERSION,
    NATIVE_GRAPHLIKE_PROBLEM_CAPSULE_METHOD, NATIVE_GRAPHLIKE_PROBLEM_CAPSULE_NAME,
};
pub use dem_sampling::{
    CompiledDemLogicalCountPlan, CompiledDemSamplingPlan, DemAttributionTrace, DemHotspotEstimator,
    DemObservableBatch, DemSamplingResult, DetectorEventDemShotBatch, PackedDemShotBatch,
};
pub use expr::Expr;
pub use hotspot::compute_packed_estimate;
pub use labels::{IndexedNoiseLocation, LocationCatalog, LocationId};
pub use mask::{word_count, Mask};
pub use model::{
    Circuit, DemBatch, DemEvent, DemHotspotEstimate, DemLocationGroup, Detector, DetectorErrorEdge,
    DetectorErrorModel, DetectorGraphEstimate, DetectorGraphKey, HotspotEstimate,
    LogicalObservable, NoiseLocation, NoiseModel, NpError, NpResult, Operation, TagValue,
};
pub use packed::{
    run_sampler_program, FaultScopeSimulator, PauliBasis, RuntimeCapacities, RuntimeState,
    SamplerObservable, SamplerOperation, SamplerProgram,
};
pub use program::expand_circuit_operations;
pub use rng::SmallRng;
pub use sampling::{
    bernoulli_mask, choose_weighted_event, compile_pauli_channel_events, for_each_bernoulli_event,
    positive_unit_f64, random_bit_mask, set_shot_bit, CompiledPauliEvent,
    TWO_QUBIT_DEPOLARIZING_EVENTS,
};
pub use stabilizer::{ConcreteStabilizer, PauliFrame};
