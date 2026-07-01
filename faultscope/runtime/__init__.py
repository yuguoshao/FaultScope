"""Runtime simulators and native wrappers."""

from faultscope.runtime.batch import (
    BatchCorrectionMaskFn,
    FaultScopeSimulator,
    BatchLossMaskFn,
    SampleBatch,
    UnsupportedBatchCircuitError,
)
from faultscope.runtime.native import (
    DemFaultScopeSimulator,
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
from faultscope.runtime.results import FaultHotspot, FailureEstimate

__all__ = [
    "BatchCorrectionMaskFn",
    "DemFaultScopeSimulator",
    "FaultScopeSimulator",
    "BatchLossMaskFn",
    "SampleBatch",
    "FaultHotspot",
    "NativeDemSampler",
    "NativeDemGenerator",
    "NativePackedSampler",
    "FailureEstimate",
    "UnsupportedBatchCircuitError",
    "UnsupportedNativeCircuitError",
    "compile_native_dem_generator",
    "compile_native_dem_sampler",
    "compile_native_dem_sampler_from_circuit",
    "compile_native_sampler",
    "generate_native_dem",
]
