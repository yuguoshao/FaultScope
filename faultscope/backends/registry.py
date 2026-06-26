"""Discovery and installation metadata for optional native decoder backends."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Mapping

from faultscope._native import (
    NATIVE_DECODER_PLUGIN_ABI,
    NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP,
    available_native_decoders as _builtin_native_decoders,
)

NATIVE_DECODER_ENTRY_POINT_GROUP = NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP


class NativeDecoderBackendUnavailable(ImportError):
    """Raised when an optional native decoder backend is not installed."""


@dataclass(frozen=True)
class NativeDecoderBackendCatalogEntry:
    """Official post-install native decoder backend metadata."""

    name: str
    package_name: str
    decoder_class_name: str
    problem_kind: str
    repo_url: str | None
    default_rev: str | None
    installable: bool
    description: str


@dataclass(frozen=True)
class NativeDecoderBackendStatus:
    """Load status for one native decoder backend."""

    name: str
    installed: bool
    loadable: bool
    version: str | None = None
    source: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class NativeDecoderInstallPlan:
    """Explicit installation steps for one optional native decoder backend."""

    backend: NativeDecoderBackendCatalogEntry
    checkout: str | None
    commands: tuple[tuple[str, ...], ...]
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class _LoadedBackend:
    name: str
    version: str | None
    source: str | None
    decoders: Mapping[str, type]


_BACKEND_CACHE: dict[str, _LoadedBackend] | None = None
_STATUS_CACHE: dict[str, NativeDecoderBackendStatus] | None = None

OFFICIAL_BACKEND_CATALOG: dict[str, NativeDecoderBackendCatalogEntry] = {
    "fusion-blossom": NativeDecoderBackendCatalogEntry(
        name="fusion-blossom",
        package_name="faultscope-fusion-blossom",
        decoder_class_name="NativeFusionBlossomDecoder",
        problem_kind="graphlike",
        repo_url="https://github.com/yuewuo/fusion-blossom.git",
        default_rev="main",
        installable=True,
        description=(
            "Minimal serial beta fusion-blossom MWPM graphlike decoder backend."
        ),
    ),
    "pymatching": NativeDecoderBackendCatalogEntry(
        name="pymatching",
        package_name="faultscope-pymatching",
        decoder_class_name="NativePyMatchingDecoder",
        problem_kind="graphlike",
        repo_url="https://github.com/oscarhiggott/PyMatching.git",
        default_rev="v2.4.0",
        installable=True,
        description=(
            "Optional native PyMatching sparse-blossom graphlike decoder backend."
        ),
    ),
    "bposd": NativeDecoderBackendCatalogEntry(
        name="bposd",
        package_name="faultscope-bposd",
        decoder_class_name="NativeBposdDecoder",
        problem_kind="binary-linear",
        repo_url=None,
        default_rev=None,
        installable=False,
        description="Reserved BP+OSD/LDPC-style native decoder backend.",
    ),
}


def clear_native_decoder_plugin_cache() -> None:
    """Clear cached entry point discovery results."""

    global _BACKEND_CACHE, _STATUS_CACHE
    _BACKEND_CACHE = None
    _STATUS_CACHE = None


def official_native_decoder_backend_catalog() -> tuple[NativeDecoderBackendCatalogEntry, ...]:
    """Return official post-install backend catalog entries."""

    return tuple(OFFICIAL_BACKEND_CATALOG[name] for name in sorted(OFFICIAL_BACKEND_CATALOG))


def get_native_decoder_backend_catalog_entry(
    name: str,
) -> NativeDecoderBackendCatalogEntry | None:
    """Return one official backend catalog entry by name."""

    return OFFICIAL_BACKEND_CATALOG.get(name)


def available_native_decoders() -> tuple[str, ...]:
    """Return built-in plus loadable post-install native decoder names."""

    names = list(_builtin_native_decoders())
    for backend in _load_backends().values():
        for decoder_name in backend.decoders:
            if decoder_name not in names:
                names.append(decoder_name)
    return tuple(names)


def native_decoder_backend_statuses() -> tuple[NativeDecoderBackendStatus, ...]:
    """Return status rows for built-in and optional native decoder backends."""

    _load_backends()
    statuses = [
        NativeDecoderBackendStatus(
            name=name,
            installed=True,
            loadable=True,
            source="built-in",
        )
        for name in _builtin_native_decoders()
    ]
    optional = dict(_STATUS_CACHE or {})
    for entry in official_native_decoder_backend_catalog():
        optional.setdefault(
            entry.name,
            NativeDecoderBackendStatus(
                name=entry.name,
                installed=False,
                loadable=False,
                source=entry.package_name,
                error=backend_unavailable_message(entry.name),
            ),
        )
    statuses.extend(optional[name] for name in sorted(optional))
    return tuple(statuses)


def get_native_decoder_class(name: str) -> type | None:
    """Return a post-install native decoder class by backend name."""

    for backend in _load_backends().values():
        decoder = backend.decoders.get(name)
        if decoder is not None:
            return decoder
    return None


def native_decoder_backend_error(name: str) -> str | None:
    """Return the most recent loading error for a backend name, if any."""

    _load_backends()
    status = (_STATUS_CACHE or {}).get(name)
    if status is None:
        return None
    return status.error


def backend_unavailable_message(name: str) -> str:
    """Return the user-facing missing-backend message for a backend name."""

    entry = get_native_decoder_backend_catalog_entry(name)
    error = native_decoder_backend_error(name)
    details = f": {error}" if error else ""
    if entry is None:
        return f"Native decoder backend {name!r} is not installed{details}."
    install_hint = f"python -m faultscope.backends install {entry.name}"
    if not entry.installable:
        return (
            f"{entry.decoder_class_name} requires the optional `{entry.package_name}` "
            f"native backend, but `{entry.name}` is reserved and not installable yet. "
            f"Requested install command would be `{install_hint}`{details}."
        )
    return (
        f"{entry.decoder_class_name} requires the optional `{entry.package_name}` "
        f"native backend. Install it with `{install_hint}`{details}."
    )


def fusion_blossom_unavailable_message() -> str:
    """Return the fusion-blossom missing-backend message."""

    return backend_unavailable_message("fusion-blossom")


def native_decoder_install_plan(
    name: str,
    *,
    target_dir: str,
    rev: str | None = None,
    python_executable: str,
) -> NativeDecoderInstallPlan:
    """Return the explicit installation plan for a catalog backend."""

    entry = get_native_decoder_backend_catalog_entry(name)
    if entry is None:
        raise ValueError(f"unknown native decoder backend {name!r}")
    if not entry.installable:
        return NativeDecoderInstallPlan(
            backend=entry,
            checkout=None,
            commands=(),
            unavailable_reason=backend_unavailable_message(name),
        )
    selected_rev = rev or entry.default_rev
    checkout = f"{target_dir.rstrip('/')}/{entry.name}"
    commands: list[tuple[str, ...]] = []
    if entry.repo_url is not None:
        commands.append(("git", "clone", entry.repo_url, checkout))
        if selected_rev is not None:
            commands.append(("git", "-C", checkout, "checkout", selected_rev))
    commands.append(
        (
            python_executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            entry.package_name,
        )
    )
    return NativeDecoderInstallPlan(
        backend=entry,
        checkout=checkout,
        commands=tuple(commands),
    )


def _load_backends() -> dict[str, _LoadedBackend]:
    global _BACKEND_CACHE, _STATUS_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE

    backends: dict[str, _LoadedBackend] = {}
    decoder_sources: dict[str, str] = {
        decoder_name: "built-in" for decoder_name in _builtin_native_decoders()
    }
    statuses: dict[str, NativeDecoderBackendStatus] = {}
    for entry_point in _iter_entry_points():
        entry_name = getattr(entry_point, "name", "<unknown>")
        try:
            manifest = _load_manifest(entry_point)
            backend = _normalize_manifest(entry_name, manifest)
        except Exception as exc:  # noqa: BLE001 - discovery must not break import.
            statuses[entry_name] = NativeDecoderBackendStatus(
                name=entry_name,
                installed=True,
                loadable=False,
                error=str(exc),
            )
            continue
        duplicate = next(
            (
                decoder_name
                for decoder_name in backend.decoders
                if decoder_name in decoder_sources
            ),
            None,
        )
        if duplicate is not None:
            statuses[backend.name] = NativeDecoderBackendStatus(
                name=backend.name,
                installed=True,
                loadable=False,
                version=backend.version,
                source=backend.source,
                error=(
                    f"decoder {duplicate!r} is already provided by "
                    f"{decoder_sources[duplicate]!r}"
                ),
            )
            continue
        backends[backend.name] = backend
        for decoder_name in backend.decoders:
            decoder_sources[decoder_name] = backend.name
        statuses[backend.name] = NativeDecoderBackendStatus(
            name=backend.name,
            installed=True,
            loadable=True,
            version=backend.version,
            source=backend.source,
        )

    _BACKEND_CACHE = backends
    _STATUS_CACHE = statuses
    return backends


def _iter_entry_points() -> tuple[Any, ...]:
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=NATIVE_DECODER_ENTRY_POINT_GROUP))
    if hasattr(entry_points, "get"):
        return tuple(entry_points.get(NATIVE_DECODER_ENTRY_POINT_GROUP, ()))
    return ()


def _load_manifest(entry_point: Any) -> Any:
    loaded = entry_point.load()
    if callable(loaded) and not hasattr(loaded, "decoders"):
        return loaded()
    return loaded


def _normalize_manifest(entry_name: str, manifest: Any) -> _LoadedBackend:
    if isinstance(manifest, Mapping):
        get = manifest.get
    else:
        get = lambda key, default=None: getattr(manifest, key, default)
    abi_version = get("abi_version")
    if abi_version != NATIVE_DECODER_PLUGIN_ABI:
        raise RuntimeError(
            f"native decoder backend {entry_name!r} has ABI {abi_version!r}; "
            f"expected {NATIVE_DECODER_PLUGIN_ABI!r}"
        )
    name = get("name", None)
    version = get("version", None)
    source = get("source", None)
    for field_name, field_value in (
        ("name", name),
        ("version", version),
        ("source", source),
    ):
        if not field_value:
            raise RuntimeError(
                f"native decoder backend {entry_name!r} did not declare {field_name}"
            )
    name = str(name)
    decoders = get("decoders", None)
    if not isinstance(decoders, Mapping) or not decoders:
        raise RuntimeError(f"native decoder backend {name!r} did not declare decoders")
    return _LoadedBackend(
        name=name,
        version=str(version),
        source=str(source),
        decoders=dict(decoders),
    )
