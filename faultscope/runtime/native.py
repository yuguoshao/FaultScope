"""Runtime wrappers around the Rust core Python module.

The product runtime lives in ``faultscope-core`` and is exposed through the PyO3
module ``faultscope._native``.  These wrappers keep the public
``UnsupportedNativeCircuitError`` boundary for compile/generate entry points.
"""

from __future__ import annotations

import importlib
from typing import Any

from faultscope.core import Circuit


class UnsupportedNativeCircuitError(ValueError):
    """Raised when a circuit cannot be compiled by the native sampler."""


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


def generate_native_dem(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
) -> Any:
    """Generate a detector error model through the native extension."""

    try:
        generator = compile_native_dem_generator(
            circuit,
            detectors=detectors,
            observables=observables,
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
) -> NativeDemGenerator:
    """Compile ``circuit`` into a reusable native DEM generator."""

    try:
        native_mod = importlib.import_module("faultscope._native")
        compile_generator = getattr(native_mod, "compile_dem_generator", None)
        if compile_generator is None:
            generator = native_mod.DetectorErrorModelGenerator(
                circuit,
                detectors=detectors,
                observables=observables,
            )
        else:
            generator = compile_generator(circuit, detectors, observables)
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
    materialize_dem: bool = True,
) -> NativeDemSampler:
    """Generate and compile a native DEM sampler directly from ``circuit``.

    ``materialize_dem=False`` skips constructing the Python ``DetectorErrorModel``
    object. In that mode, ``sampler.dem`` is ``None`` and APIs requiring full
    DEM metadata raise ``ValueError``.
    """

    try:
        generator = compile_native_dem_generator(
            circuit,
            detectors=detectors,
            observables=observables,
        )
        if not hasattr(generator, "compile_sampler"):
            native_mod = importlib.import_module("faultscope._native")
            if materialize_dem:
                payload = native_mod.generate_and_compile_dem_sampler(
                    circuit,
                    detectors,
                    observables,
                )
                return payload["sampler"]
            return native_mod.compile_generated_dem_sampler(
                circuit,
                detectors,
                observables,
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
