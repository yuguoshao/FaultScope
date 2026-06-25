"""NPSim fusion-blossom backend scaffold.

This package reserves the official post-install backend package shape for a
future native fusion-blossom decoder. The current implementation is deliberately
a smoke backend: it constructs a small Rust graphlike detector-copy decoder in
this package and exposes it to NPSim through the native decoder PyCapsule ABI. It
does not run the real fusion-blossom solver.
"""

from __future__ import annotations

from npsim._npsim_native import NATIVE_DECODER_PLUGIN_ABI
from npsim.runtime import generate_native_dem

try:
    from . import _native
except ImportError as exc:  # pragma: no cover - exercised before local build.
    _native = None
    _native_import_error = exc
else:
    _native_import_error = None

__version__ = "0.1.0"
BACKEND_NAME = "fusion-blossom"


class NativeFusionBlossomDecoder:
    """Scaffold fusion-blossom decoder proxy.

    This is not a production MWPM decoder. It owns a Rust graphlike smoke
    decoder in this backend package so the post-install PyCapsule path can be
    validated without moving hot-path syndrome or correction data through
    Python.
    """

    backend_name = BACKEND_NAME

    def __init__(self, inner):
        self._inner = inner
        self._python_decode_call_count = 0

    @staticmethod
    def from_dem(dem, *, options=None):
        _reject_options(options)
        _require_native_extension()
        problem = dem.compile_graphlike_problem()
        inner = _native.NativeFusionBlossomNativeDecoder.from_graphlike_problem(
            problem
        )
        return NativeFusionBlossomDecoder(inner)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        _reject_options(options)
        dem = generate_native_dem(
            circuit,
            detectors=detectors,
            observables=observables,
        )
        return NativeFusionBlossomDecoder.from_dem(dem)

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
    def python_decode_call_count(self):
        return self._python_decode_call_count

    def __npsim_native_decoder_capsule__(self):
        return self._inner.__npsim_native_decoder_capsule__()

    def decode_batch_masks(self, batch):
        self._python_decode_call_count += 1
        corrections = {}
        for observable_id, detector_id in zip(
            self.observable_ids,
            self._inner.observable_detector_ids,
        ):
            corrections[observable_id] = (
                0 if detector_id is None else batch.detectors[detector_id]
            )
        return corrections

    def __repr__(self):
        return (
            "NativeFusionBlossomDecoder("
            f"name={self.name!r}, "
            f"detector_ids={self.detector_ids!r}, "
            f"observable_ids={self.observable_ids!r})"
        )


def backend_manifest():
    """Return the NPSim native decoder plugin manifest."""

    return {
        "abi_version": NATIVE_DECODER_PLUGIN_ABI,
        "name": BACKEND_NAME,
        "version": __version__,
        "source": "npsim-fusion-blossom scaffold",
        "decoders": {BACKEND_NAME: NativeFusionBlossomDecoder},
    }


def _reject_options(options):
    if options is not None:
        raise ValueError(
            "npsim-fusion-blossom scaffold does not accept options yet"
        )


def _require_native_extension():
    if _native is None:
        raise ImportError(
            "npsim-fusion-blossom native extension is not built; install it with "
            "`python -m pip install -e backends/npsim-fusion-blossom`"
        ) from _native_import_error


def native_extension_available():
    return _native is not None


__all__ = [
    "BACKEND_NAME",
    "NativeFusionBlossomDecoder",
    "__version__",
    "backend_manifest",
    "native_extension_available",
]
