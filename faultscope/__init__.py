"""Forward noise-aware stabilizer trajectory simulator."""

from importlib.metadata import PackageNotFoundError, version as _distribution_version

from faultscope.runtime import (
    DemFaultScopeSimulator,
    FaultScopeSimulator,
    SampleBatch,
    UnsupportedBatchCircuitError,
)
from faultscope.core import Circuit, NoiseLocation, Operation
from faultscope.dem import (
    BinaryLinearDecodingProblem,
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    DetectorErrorModelGenerator,
    DetectorGraphEdgeHotspot,
    DetectorGraphHotspots,
    GraphlikeDecodingProblem,
    IndexedDem,
    LogicalObservable,
    SparseBinaryMatrix,
    UnsupportedDemCircuitError,
)
from faultscope.dem import (
    DemHotspotEstimator,
    DemSampleBatch,
    DemEdgeHotspot,
    DemHotspotEstimate,
    DemLocationHotspot,
    DemLocationMetadata,
)
from faultscope.core import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from faultscope.runtime import (
    NativeDemGenerator,
    NativeDemSampler,
    NativePackedSampler,
    UnsupportedNativeCircuitError,
    compile_native_dem_generator,
    compile_native_dem_sampler,
    compile_native_dem_sampler_from_circuit,
    compile_native_sampler,
    generate_native_dem,
)
from faultscope.decoders import (
    NativeBatchDecoder,
    NativeCompositeDecoder,
    NativeBpDecoder,
    NativeBposdDecoder,
    NativeDecoderBackendUnavailable,
    NativeFusionBlossomDecoder,
    NativeGraphlikeDetectorCopyDecoder,
    NativeMwpmDecoder,
    NativeNoCorrectionDecoder,
    NativePyMatchingDecoder,
    PyMatchingDecoder,
    PyMatchingUnavailableError,
    UnsupportedPyMatchingDemError,
    available_native_decoders,
    create_native_decoder,
    get_native_decoder_class,
)
from faultscope.runtime import FaultHotspot, FailureEstimate
from faultscope.io import StimImportError, StimImportResult, load_stim_file, parse_stim_circuit
from faultscope.viz import (
    VisualizationUnavailableError,
    write_rotated_surface_code_spatial_hotspot_map,
    write_repetition_gate_structure_hotspot_map,
    write_repetition_hotspot_heatmap,
)
from faultscope.collection import (
    Collector,
    CollectionCounterSchema,
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    HotspotCollectionResult,
    Progress,
    TaskStats,
    collect,
    collect_hotspots,
    iter_collect,
    iter_progress,
)


def _package_version() -> str:
    try:
        return _distribution_version("faultscope")
    except PackageNotFoundError:  # Source tree without installed distribution metadata.
        from faultscope._native import __version__ as native_version

        return native_version


__version__ = _package_version()
del _package_version

__all__ = [
    "__version__",
    "BernoulliPauliNoise",
    "CollectionOptions",
    "CollectionCounterSchema",
    "CollectionRunOptions",
    "CollectionTask",
    "HotspotCollectionResult",
    "Collector",
    "FaultScopeSimulator",
    "SampleBatch",
    "BinaryLinearDecodingProblem",
    "Circuit",
    "DemHotspotEstimator",
    "DemSampleBatch",
    "DemEdgeHotspot",
    "DemHotspotEstimate",
    "DemLocationHotspot",
    "DemLocationMetadata",
    "DemFaultScopeSimulator",
    "Detector",
    "DetectorErrorEdge",
    "DetectorErrorModel",
    "DetectorErrorModelGenerator",
    "DetectorGraphEdgeHotspot",
    "DetectorGraphHotspots",
    "GraphlikeDecodingProblem",
    "FaultHotspot",
    "IndexedDem",
    "LogicalObservable",
    "MeasurementBitFlip",
    "NativeBatchDecoder",
    "NativeCompositeDecoder",
    "NativeBpDecoder",
    "NativeBposdDecoder",
    "NativeDecoderBackendUnavailable",
    "NativeFusionBlossomDecoder",
    "NativeDemSampler",
    "NativeDemGenerator",
    "NativeGraphlikeDetectorCopyDecoder",
    "NativeMwpmDecoder",
    "NativeNoCorrectionDecoder",
    "NativePackedSampler",
    "NativePyMatchingDecoder",
    "NoiseLocation",
    "Operation",
    "PauliChannel",
    "PyMatchingDecoder",
    "PyMatchingUnavailableError",
    "Progress",
    "FailureEstimate",
    "SingleQubitDepolarizing",
    "SparseBinaryMatrix",
    "StimImportError",
    "StimImportResult",
    "TaskStats",
    "TwoQubitDepolarizing",
    "UnsupportedBatchCircuitError",
    "UnsupportedDemCircuitError",
    "UnsupportedNativeCircuitError",
    "UnsupportedPyMatchingDemError",
    "VisualizationUnavailableError",
    "available_native_decoders",
    "compile_native_dem_generator",
    "compile_native_dem_sampler",
    "compile_native_dem_sampler_from_circuit",
    "compile_native_sampler",
    "collect",
    "collect_hotspots",
    "create_native_decoder",
    "generate_native_dem",
    "get_native_decoder_class",
    "iter_collect",
    "iter_progress",
    "load_stim_file",
    "parse_stim_circuit",
    "write_rotated_surface_code_spatial_hotspot_map",
    "write_repetition_gate_structure_hotspot_map",
    "write_repetition_hotspot_heatmap",
]
