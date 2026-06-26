"""FaultScope fusion-blossom backend.

This package provides the official post-install backend package shape for a
native fusion-blossom decoder. The current implementation is a minimal serial
beta MWPM adapter: it constructs Rust solver state in this package and exposes
it to FaultScope through the native decoder PyCapsule ABI.
"""

from __future__ import annotations

import math

from faultscope._native import NATIVE_DECODER_PLUGIN_ABI
from faultscope.runtime import generate_native_dem

try:
    from . import _native
except ImportError as exc:  # pragma: no cover - exercised before local build.
    _native = None
    _native_import_error = exc
else:
    _native_import_error = None

__version__ = "0.1.0"
BACKEND_NAME = "fusion-blossom"
DEFAULT_WEIGHT_SCALE = 10_000


class NativeFusionBlossomDecoder:
    """Minimal serial beta fusion-blossom decoder proxy.

    This is not the production parallel or streaming fusion-blossom adapter. It
    owns Rust solver state in this backend package so the post-install PyCapsule
    path can decode graphlike DEM batches without moving hot-path syndrome or
    correction data through Python. The builder safely compresses identical
    boundary and two-detector parallel edges and rejects ambiguous parallel
    logical effects.
    """

    backend_name = BACKEND_NAME

    def __init__(self, inner):
        self._inner = inner
        self._python_decode_call_count = 0

    @staticmethod
    def from_dem(dem, *, options=None):
        _require_native_extension()
        parsed = _parse_options(options)
        problem = dem.compile_graphlike_problem()
        inner = _native.NativeFusionBlossomNativeDecoder.from_graphlike_problem(
            problem,
            weight_scale=parsed["weight_scale"],
        )
        return NativeFusionBlossomDecoder(inner)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        dem = generate_native_dem(
            circuit,
            detectors=detectors,
            observables=observables,
        )
        return NativeFusionBlossomDecoder.from_dem(dem, options=options)

    @property
    def name(self):
        return self._inner.name

    @property
    def detector_ids(self):
        return self._inner.detector_ids

    @property
    def detector_coords(self):
        return self._inner.detector_coords

    @property
    def observable_ids(self):
        return self._inner.observable_ids

    @property
    def edge_count(self):
        return self._inner.edge_count

    @property
    def solver_vertex_count(self):
        return self._inner.solver_vertex_count

    @property
    def solver_edge_count(self):
        return self._inner.solver_edge_count

    @property
    def boundary_vertex_count(self):
        return self._inner.boundary_vertex_count

    @property
    def build_summary(self):
        return self._inner.build_summary

    @property
    def python_decode_call_count(self):
        return self._python_decode_call_count

    def __faultscope_native_decoder_capsule__(self):
        return self._inner.__faultscope_native_decoder_capsule__()

    def decode_batch_masks(self, batch):
        self._python_decode_call_count += 1
        word_count = (batch.shots + 63) // 64
        detector_words = {
            detector_id: _int_to_words(batch.detectors[detector_id], word_count)
            for detector_id in self.detector_ids
        }
        correction_words = self._inner.decode_batch_words(
            batch.shots,
            detector_words,
        )
        return {
            observable_id: _words_to_int(words)
            for observable_id, words in correction_words.items()
        }

    def __repr__(self):
        return (
            "NativeFusionBlossomDecoder("
            f"name={self.name!r}, "
            f"detector_ids={self.detector_ids!r}, "
            f"observable_ids={self.observable_ids!r}, "
            f"solver_vertex_count={self.solver_vertex_count!r}, "
            f"solver_edge_count={self.solver_edge_count!r})"
        )


def backend_manifest():
    """Return the FaultScope native decoder plugin manifest."""

    return {
        "abi_version": NATIVE_DECODER_PLUGIN_ABI,
        "name": BACKEND_NAME,
        "version": __version__,
        "source": "faultscope-fusion-blossom minimal serial beta adapter",
        "decoders": {BACKEND_NAME: NativeFusionBlossomDecoder},
    }


def _parse_options(options):
    parsed = {
        "weight_scale": DEFAULT_WEIGHT_SCALE,
    }
    if options is None:
        return parsed
    if not isinstance(options, dict):
        raise ValueError("fusion-blossom options must be a dict or None")
    unknown = set(options) - {"weight_scale"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown fusion-blossom option(s): {names}")
    weight_scale = options.get("weight_scale", DEFAULT_WEIGHT_SCALE)
    if isinstance(weight_scale, bool) or not isinstance(weight_scale, (int, float)):
        raise ValueError("fusion-blossom weight_scale must be numeric")
    weight_scale = float(weight_scale)
    if not math.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("fusion-blossom weight_scale must be positive and finite")
    parsed["weight_scale"] = weight_scale

    return parsed


def _require_native_extension():
    if _native is None:
        raise ImportError(
            "faultscope-fusion-blossom native extension is not built; install it with "
            "`python -m pip install -e backends/faultscope-fusion-blossom`"
        ) from _native_import_error


def native_extension_available():
    return _native is not None


def _int_to_words(value, word_count):
    if value < 0:
        raise ValueError("packed mask integers must be non-negative")
    return [(value >> (64 * index)) & ((1 << 64) - 1) for index in range(word_count)]


def _words_to_int(words):
    value = 0
    for index, word in enumerate(words):
        value |= int(word) << (64 * index)
    return value


__all__ = [
    "BACKEND_NAME",
    "DEFAULT_WEIGHT_SCALE",
    "NativeFusionBlossomDecoder",
    "__version__",
    "backend_manifest",
    "native_extension_available",
]
