"""Classical, PyMatching, and native decoder handles."""

from typing import Any, Mapping

from faultscope._native import (
    NativeBatchDecoder,
    NativeCompositeDecoder,
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


def get_native_decoder_class(name: str) -> type[Any]:
    """Return the installed native decoder class for a backend name."""

    backend = _get_native_decoder_class(name)
    if backend is None:
        raise NativeDecoderBackendUnavailable(backend_unavailable_message(name))
    return backend


def create_native_decoder(
    name: str,
    *,
    dem: Any = None,
    circuit: Any = None,
    detectors: Any = None,
    observables: Any = None,
    options: Mapping[str, object] | None = None,
) -> Any:
    """Construct an installed native decoder by backend name."""

    if dem is not None and circuit is not None:
        raise ValueError("supply either dem or circuit, not both")
    if name == "no-correction":
        if dem is not None:
            indexed = dem.compile_indexed()
            return NativeNoCorrectionDecoder(
                observable_ids=indexed.observable_ids,
                detector_ids=indexed.detector_ids,
            )
        if circuit is not None:
            return NativeNoCorrectionDecoder(
                observable_ids=tuple(observable.id for observable in observables or ()),
                detector_ids=tuple(detector.id for detector in detectors or ()),
            )
        raise ValueError("supply dem or circuit")
    if name == "graphlike-detector-copy":
        if dem is not None:
            return NativeGraphlikeDetectorCopyDecoder.from_dem(dem)
        if circuit is not None:
            return NativeGraphlikeDetectorCopyDecoder.from_circuit(
                circuit,
                detectors=detectors,
                observables=observables,
            )
        raise ValueError("supply dem or circuit")

    backend = get_native_decoder_class(name)
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
    def from_dem(
        cls,
        dem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder(
            cls.backend_name,
            dem=dem,
            options=options,
        )

    @classmethod
    def from_circuit(
        cls,
        circuit: Any,
        *,
        detectors: Any = None,
        observables: Any = None,
        options: Mapping[str, object] | None = None,
    ) -> Any:
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
    def from_dem(
        dem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        backend = get_native_decoder_class("fusion-blossom")
        factory = getattr(backend, "from_graphlike_problem", None)
        if factory is not None:
            return factory(dem.compile_graphlike_problem(), options=options)
        return backend.from_dem(dem, options=options)

    @staticmethod
    def from_graphlike_problem(
        problem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        backend = get_native_decoder_class("fusion-blossom")
        factory = getattr(backend, "from_graphlike_problem", None)
        if factory is None:
            raise NativeDecoderBackendUnavailable(
                "installed fusion-blossom backend does not support from_graphlike_problem"
            )
        return factory(problem, options=options)

    @staticmethod
    def from_circuit(
        circuit: Any,
        *,
        detectors: Any = None,
        observables: Any = None,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder(
            "fusion-blossom",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativeBpDecoder(_NativeDecoderProxy):
    """Proxy for the optional post-install bpdecoder.rs native backend."""

    backend_name = "bpdecoder"

    @staticmethod
    def from_dem(
        dem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder("bpdecoder", dem=dem, options=options)

    @staticmethod
    def from_circuit(
        circuit: Any,
        *,
        detectors: Any = None,
        observables: Any = None,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder(
            "bpdecoder",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativeBposdDecoder(_NativeDecoderProxy):
    """Proxy for the reserved post-install BP+OSD native backend."""

    backend_name = "bposd"

    @staticmethod
    def from_dem(
        dem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder("bposd", dem=dem, options=options)

    @staticmethod
    def from_circuit(
        circuit: Any,
        *,
        detectors: Any = None,
        observables: Any = None,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder(
            "bposd",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativeMwpmDecoder(_NativeDecoderProxy):
    """Proxy for the optional post-install mwpm.rs native backend."""

    backend_name = "mwpm"

    @staticmethod
    def from_dem(
        dem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder("mwpm", dem=dem, options=options)

    @staticmethod
    def from_circuit(
        circuit: Any,
        *,
        detectors: Any = None,
        observables: Any = None,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder(
            "mwpm",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


class NativePyMatchingDecoder(_NativeDecoderProxy):
    """Proxy for the optional post-install PyMatching native backend."""

    backend_name = "pymatching"

    @staticmethod
    def from_dem(
        dem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        backend = get_native_decoder_class("pymatching")
        factory = getattr(backend, "from_graphlike_problem", None)
        if factory is not None:
            return factory(dem.compile_graphlike_problem(), options=options)
        return backend.from_dem(dem, options=options)

    @staticmethod
    def from_graphlike_problem(
        problem: Any,
        *,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        backend = get_native_decoder_class("pymatching")
        factory = getattr(backend, "from_graphlike_problem", None)
        if factory is None:
            raise NativeDecoderBackendUnavailable(
                "installed pymatching backend does not support from_graphlike_problem"
            )
        return factory(problem, options=options)

    @staticmethod
    def from_circuit(
        circuit: Any,
        *,
        detectors: Any = None,
        observables: Any = None,
        options: Mapping[str, object] | None = None,
    ) -> Any:
        return create_native_decoder(
            "pymatching",
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            options=options,
        )


__all__ = [
    "NativeBatchDecoder",
    "NativeCompositeDecoder",
    "NativeBpDecoder",
    "NativeBposdDecoder",
    "NativeDecoderBackendUnavailable",
    "NativeFusionBlossomDecoder",
    "NativeGraphlikeDetectorCopyDecoder",
    "NativeMwpmDecoder",
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
