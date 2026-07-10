"""FaultScope PyMatching backend.

This package provides an optional post-install native PyMatching sparse-blossom
decoder. Construction happens in Python from a FaultScope circuit or DEM, but the
solver state lives in this extension package and hot-path batches cross into
FaultScope through the native decoder PyCapsule ABI.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _distribution_version

from faultscope._native import NATIVE_DECODER_PLUGIN_ABI
from faultscope.runtime import generate_native_dem

try:
    from . import _native
except ImportError as exc:  # pragma: no cover - exercised before local build.
    _native = None
    _native_import_error = exc
else:
    _native_import_error = None


def _package_version() -> str:
    try:
        return _distribution_version("faultscope-pymatching")
    except PackageNotFoundError:
        if _native is not None:
            return _native.__version__
        return "0+unknown"


__version__ = _package_version()
del _package_version
BACKEND_NAME = "pymatching"


class NativePyMatchingDecoder:
    """Native PyMatching sparse-blossom decoder proxy."""

    backend_name = BACKEND_NAME

    def __init__(self, inner):
        self._inner = inner
        self._python_decode_call_count = 0

    @staticmethod
    def from_dem(dem, *, options=None):
        _require_native_extension()
        _parse_options(options)
        inner = _native.NativePyMatchingNativeDecoder.from_dem(dem)
        return NativePyMatchingDecoder(inner)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        dem = generate_native_dem(
            circuit,
            detectors=detectors,
            observables=observables,
        )
        return NativePyMatchingDecoder.from_dem(dem, options=options)

    @property
    def name(self):
        return self._inner.name

    @property
    def detector_ids(self):
        return self._inner.detector_ids

    @property
    def observable_ids(self):
        return self._inner.observable_ids

    @property
    def edge_count(self):
        return self._inner.edge_count

    @property
    def solver_edge_count(self):
        return self._inner.solver_edge_count

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
            observable_id: _words_to_int(words) for observable_id, words in correction_words.items()
        }

    def __repr__(self):
        return (
            "NativePyMatchingDecoder("
            f"name={self.name!r}, "
            f"detector_ids={self.detector_ids!r}, "
            f"observable_ids={self.observable_ids!r}, "
            f"solver_edge_count={self.solver_edge_count!r})"
        )


def backend_manifest():
    """Return the FaultScope native decoder plugin manifest."""

    return {
        "abi_version": NATIVE_DECODER_PLUGIN_ABI,
        "name": BACKEND_NAME,
        "version": __version__,
        "source": "faultscope-pymatching native sparse-blossom adapter",
        "decoders": {BACKEND_NAME: NativePyMatchingDecoder},
    }


def _parse_options(options):
    if options is None:
        return
    if not isinstance(options, dict):
        raise ValueError("pymatching options must be a dict or None")
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unknown pymatching option(s): {names}")


def _require_native_extension():
    if _native is None:
        raise ImportError(
            "faultscope-pymatching native extension is not built; install it with "
            "`python -m pip install -e backends/faultscope-pymatching --no-build-isolation`"
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
    "NativePyMatchingDecoder",
    "__version__",
    "backend_manifest",
    "native_extension_available",
]
