"""Classical, PyMatching, and native decoder handles."""

from faultscope._native import (
    NativeBatchDecoder,
    NativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder,
)
from faultscope.backends import (
    NativeDecoderBackendUnavailable,
    available_native_decoders,
    backend_unavailable_message,
    get_native_decoder_class as _get_native_decoder_class,
)
from faultscope.decoders.classical import NoCorrectionDecoder, RepetitionCodeDecoder
from faultscope.decoders.pymatching import (
    PyMatchingDecoder,
    PyMatchingUnavailableError,
    UnsupportedPyMatchingDemError,
)


def get_native_decoder_class(name):
    """Return the installed native decoder class for a backend name."""

    backend = _get_native_decoder_class(name)
    if backend is None:
        raise NativeDecoderBackendUnavailable(backend_unavailable_message(name))
    return backend


def create_native_decoder(
    name,
    *,
    dem=None,
    circuit=None,
    detectors=None,
    observables=None,
    options=None,
):
    """Construct an installed native decoder by backend name."""

    backend = get_native_decoder_class(name)
    if dem is not None and circuit is not None:
        raise ValueError("supply either dem or circuit, not both")
    if dem is not None:
        return backend.from_dem(dem, options=options)
    if circuit is not None:
        return backend.from_circuit(
            circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )
    raise ValueError("supply dem or circuit")


class _NativeDecoderProxy:
    backend_name = ""

    @classmethod
    def from_dem(cls, dem, *, options=None):
        return create_native_decoder(
            cls.backend_name,
            dem=dem,
            options=options,
        )

    @classmethod
    def from_circuit(cls, circuit, *, detectors=None, observables=None, options=None):
        return create_native_decoder(
            cls.backend_name,
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativeFusionBlossomDecoder(_NativeDecoderProxy):
    """Proxy for the optional post-install fusion-blossom native backend."""

    backend_name = "fusion-blossom"

    @staticmethod
    def from_dem(dem, *, options=None):
        return create_native_decoder("fusion-blossom", dem=dem, options=options)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        return create_native_decoder(
            "fusion-blossom",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativeBposdDecoder(_NativeDecoderProxy):
    """Proxy for the reserved post-install BP+OSD native backend."""

    backend_name = "bposd"

    @staticmethod
    def from_dem(dem, *, options=None):
        return create_native_decoder("bposd", dem=dem, options=options)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        return create_native_decoder(
            "bposd",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativePyMatchingDecoder(_NativeDecoderProxy):
    """Proxy for the optional post-install PyMatching native backend."""

    backend_name = "pymatching"

    @staticmethod
    def from_dem(dem, *, options=None):
        return create_native_decoder("pymatching", dem=dem, options=options)

    @staticmethod
    def from_circuit(circuit, *, detectors=None, observables=None, options=None):
        return create_native_decoder(
            "pymatching",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


__all__ = [
    "NativeBatchDecoder",
    "NativeBposdDecoder",
    "NativeDecoderBackendUnavailable",
    "NativeFusionBlossomDecoder",
    "NativeGraphlikeDetectorCopyDecoder",
    "NativeNoCorrectionDecoder",
    "NativePyMatchingDecoder",
    "NoCorrectionDecoder",
    "PyMatchingDecoder",
    "PyMatchingUnavailableError",
    "RepetitionCodeDecoder",
    "UnsupportedPyMatchingDemError",
    "available_native_decoders",
    "create_native_decoder",
    "get_native_decoder_class",
]
