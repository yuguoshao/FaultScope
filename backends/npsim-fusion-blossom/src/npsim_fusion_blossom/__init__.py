"""NPSim fusion-blossom backend scaffold.

This package reserves the official post-install backend package shape for a
future native fusion-blossom decoder. The current implementation is deliberately
a smoke backend: it delegates construction to NPSim's in-tree
NativeGraphlikeDetectorCopyDecoder and does not run the real fusion-blossom
solver.
"""

from __future__ import annotations

from npsim._npsim_native import (
    NATIVE_DECODER_PLUGIN_ABI,
    NativeGraphlikeDetectorCopyDecoder,
)

__version__ = "0.1.0"
BACKEND_NAME = "fusion-blossom"


class NativeFusionBlossomDecoder:
    """Scaffold fusion-blossom decoder proxy.

    This is not a production MWPM decoder. It returns an existing in-tree native
    graphlike smoke decoder so the post-install backend path can be validated
    without moving hot-path syndrome or correction data through Python.
    """

    backend_name = BACKEND_NAME

    @staticmethod
    def from_dem(dem, *, options=None):
        _reject_options(options)
        return NativeGraphlikeDetectorCopyDecoder.from_dem(dem)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        _reject_options(options)
        return NativeGraphlikeDetectorCopyDecoder.from_circuit(
            circuit,
            detectors=detectors,
            observables=observables,
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


__all__ = [
    "BACKEND_NAME",
    "NativeFusionBlossomDecoder",
    "__version__",
    "backend_manifest",
]
