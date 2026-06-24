"""Optional native decoder backend discovery and installer helpers."""

from npsim.backends.registry import (
    NATIVE_DECODER_ENTRY_POINT_GROUP,
    NATIVE_DECODER_PLUGIN_ABI,
    NativeDecoderBackendCatalogEntry,
    NativeDecoderBackendStatus,
    NativeDecoderInstallPlan,
    NativeDecoderBackendUnavailable,
    available_native_decoders,
    backend_unavailable_message,
    clear_native_decoder_plugin_cache,
    fusion_blossom_unavailable_message,
    get_native_decoder_backend_catalog_entry,
    get_native_decoder_class,
    native_decoder_install_plan,
    native_decoder_backend_error,
    native_decoder_backend_statuses,
    official_native_decoder_backend_catalog,
)

__all__ = [
    "NATIVE_DECODER_ENTRY_POINT_GROUP",
    "NATIVE_DECODER_PLUGIN_ABI",
    "NativeDecoderBackendCatalogEntry",
    "NativeDecoderBackendStatus",
    "NativeDecoderInstallPlan",
    "NativeDecoderBackendUnavailable",
    "available_native_decoders",
    "backend_unavailable_message",
    "clear_native_decoder_plugin_cache",
    "fusion_blossom_unavailable_message",
    "get_native_decoder_backend_catalog_entry",
    "get_native_decoder_class",
    "native_decoder_install_plan",
    "native_decoder_backend_error",
    "native_decoder_backend_statuses",
    "official_native_decoder_backend_catalog",
]
