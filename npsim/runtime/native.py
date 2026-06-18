"""Runtime wrappers around the Rust core Python module.

The product runtime lives in ``npsim-core`` and is exposed through the PyO3
module ``npsim._npsim_native``.  These wrappers keep the public
``UnsupportedNativeCircuitError`` boundary for compile/generate entry points.
"""

from __future__ import annotations

import importlib
from typing import Any

from npsim.core import Circuit


class UnsupportedNativeCircuitError(ValueError):
    """Raised when a circuit cannot be compiled by the native sampler."""


try:
    _native_mod = importlib.import_module("npsim._npsim_native")
    NativePackedSampler = _native_mod.NativePackedSampler
    NativeDemSampler = _native_mod.NativeDemSampler
except ImportError:
    NativePackedSampler = Any
    NativeDemSampler = Any


def compile_native_sampler(
    circuit: Circuit,
    *,
    observables: Any | None = None,
) -> NativePackedSampler:
    """Compile ``circuit`` into a native packed sampler."""

    observables_tuple = tuple(observables) if observables is not None else ()

    try:
        native_mod = importlib.import_module("npsim._npsim_native")
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
        native_mod = importlib.import_module("npsim._npsim_native")
        return native_mod.DetectorErrorModelGenerator(
            circuit,
            detectors=detectors,
            observables=observables,
        ).generate()
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def compile_native_dem_sampler(
    dem: Any,
) -> NativeDemSampler:
    """Compile a detector error model into a packed native DEM sampler."""

    try:
        native_mod = importlib.import_module("npsim._npsim_native")
        return native_mod.compile_dem_sampler(dem)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


__all__ = [
    "NativeDemSampler",
    "NativePackedSampler",
    "UnsupportedNativeCircuitError",
    "compile_native_dem_sampler",
    "compile_native_sampler",
    "generate_native_dem",
]
