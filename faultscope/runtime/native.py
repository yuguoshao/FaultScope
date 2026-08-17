"""Runtime wrappers around the Rust core Python module.

The product runtime lives in ``faultscope-core`` and is exposed through the PyO3
module ``faultscope._native``.  These wrappers keep the public
``UnsupportedNativeCircuitError`` boundary for compile/generate entry points.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from faultscope.core import Circuit


class UnsupportedNativeCircuitError(ValueError):
    """Raised when a circuit cannot be compiled by the native sampler."""


if TYPE_CHECKING:
    from faultscope._native import (
        DemFaultScopeSimulator,
        NativeDemGenerator,
        NativeDemSampler,
        NativePackedSampler,
    )
else:
    try:
        _native_mod = importlib.import_module("faultscope._native")
        DemFaultScopeSimulator = _native_mod.DemFaultScopeSimulator
        NativePackedSampler = _native_mod.NativePackedSampler
        NativeDemSampler = _native_mod.NativeDemSampler
        NativeDemGenerator = _native_mod.NativeDemGenerator
    except (AttributeError, ImportError):
        DemFaultScopeSimulator = Any
        NativePackedSampler = Any
        NativeDemSampler = Any
        NativeDemGenerator = Any


def compile_native_sampler(
    circuit: Circuit,
    *,
    observables: Any | None = None,
) -> NativePackedSampler:
    """Compile ``circuit`` into a native packed sampler."""

    observables_tuple = tuple(observables) if observables is not None else ()

    try:
        native_mod = importlib.import_module("faultscope._native")
        return native_mod.compile_sampler(circuit, observables_tuple)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def compile_native_collection_sampler(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
) -> NativePackedSampler:
    """Compile a circuit with collection detector/observable override semantics.

    ``None`` keeps declarations embedded in the circuit. An explicit iterable,
    including an empty one, replaces the corresponding embedded declarations.
    """

    detectors_value = None if detectors is None else tuple(detectors)
    observables_value = None if observables is None else tuple(observables)
    try:
        native_mod = importlib.import_module("faultscope._native")
        return native_mod.compile_collection_sampler(
            circuit,
            detectors_value,
            observables_value,
        )
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def generate_native_dem(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
    approximate_disjoint_errors: bool | float = False,
) -> Any:
    """Generate a detector error model through the native extension.

    ``approximate_disjoint_errors`` follows Stim's policy for categorical
    Pauli channels; the default is strict.
    """

    try:
        generator = compile_native_dem_generator(
            circuit,
            detectors=detectors,
            observables=observables,
            approximate_disjoint_errors=approximate_disjoint_errors,
        )
        if hasattr(generator, "generate_dem"):
            return generator.generate_dem()
        return generator.generate()
    except UnsupportedNativeCircuitError:
        raise
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def compile_native_dem_generator(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
    approximate_disjoint_errors: bool | float = False,
) -> NativeDemGenerator:
    """Compile ``circuit`` into a reusable native DEM generator.

    One-qubit ``PauliChannel`` locations first attempt Stim's independent X/Y/Z
    conversion. Multi-component channels that cannot use that path are rejected
    by default because categorical outcomes are not independent DEM
    instructions. Set ``approximate_disjoint_errors=True`` (or a maximum
    component-probability threshold) to opt into Stim-compatible approximation.
    """

    try:
        native_mod = importlib.import_module("faultscope._native")
        compile_generator = getattr(native_mod, "compile_dem_generator", None)
        if compile_generator is None:
            generator = native_mod.DetectorErrorModelGenerator(
                circuit,
                detectors=detectors,
                observables=observables,
                approximate_disjoint_errors=approximate_disjoint_errors,
            )
        else:
            generator = compile_generator(
                circuit,
                detectors,
                observables,
                approximate_disjoint_errors=approximate_disjoint_errors,
            )
        return generator
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def compile_native_dem_sampler(
    dem: Any,
) -> NativeDemSampler:
    """Compile a detector error model into a packed native DEM sampler."""

    try:
        native_mod = importlib.import_module("faultscope._native")
        return native_mod.compile_dem_sampler(dem)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def compile_native_dem_sampler_from_circuit(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
    approximate_disjoint_errors: bool | float = False,
    materialize_dem: bool = True,
) -> NativeDemSampler:
    """Generate and compile a native DEM sampler directly from ``circuit``.

    ``materialize_dem=False`` skips constructing the Python ``DetectorErrorModel``
    object. In that mode, ``sampler.dem`` is ``None`` and APIs requiring full
    DEM metadata raise ``ValueError``. ``approximate_disjoint_errors`` controls
    the same Stim-compatible categorical-channel approximation as
    :func:`generate_native_dem`.
    """

    try:
        generator = compile_native_dem_generator(
            circuit,
            detectors=detectors,
            observables=observables,
            approximate_disjoint_errors=approximate_disjoint_errors,
        )
        if not hasattr(generator, "compile_sampler"):
            native_mod = importlib.import_module("faultscope._native")
            if materialize_dem:
                payload = native_mod.generate_and_compile_dem_sampler(
                    circuit,
                    detectors,
                    observables,
                    approximate_disjoint_errors=approximate_disjoint_errors,
                )
                return payload["sampler"]
            return native_mod.compile_generated_dem_sampler(
                circuit,
                detectors,
                observables,
                approximate_disjoint_errors=approximate_disjoint_errors,
            )
        if materialize_dem:
            return generator.compile_sampler(materialize_dem=True)
        return generator.compile_sampler(materialize_dem=False)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


__all__ = [
    "DemFaultScopeSimulator",
    "NativeDemSampler",
    "NativeDemGenerator",
    "NativePackedSampler",
    "UnsupportedNativeCircuitError",
    "compile_native_dem_generator",
    "compile_native_dem_sampler",
    "compile_native_dem_sampler_from_circuit",
    "compile_native_sampler",
    "generate_native_dem",
]
