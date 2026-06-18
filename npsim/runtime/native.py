"""Native packed sampler API.

The Rust extension is the only runtime for packed batch sampling and
detector-error-model sampling.
"""

from __future__ import annotations

import importlib
import inspect
import random
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from npsim.runtime.batch import BatchTrajectory
from npsim.runtime.loss import logical_residual_loss_mask
from npsim.core import Circuit, NoiseLocation, Operation
from npsim.core import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)

_NATIVE_OP_KIND = {
    "h": 0,
    "s": 1,
    "s_dag": 2,
    "cx": 3,
    "cz": 4,
    "swap": 5,
    "pauli": 6,
    "noise": 7,
    "measure": 8,
    "measure_pauli": 9,
    "reset": 10,
    "detector": 11,
    "observable_include": 12,
}

_NATIVE_NOISE_MODEL = {
    "bernoulli_pauli": 0,
    "measurement_bit_flip": 1,
    "single_qubit_depolarizing": 2,
    "two_qubit_depolarizing": 3,
    "pauli_channel": 4,
}

_DEM_LIGHT_SAMPLER_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[Circuit],
        dict[
            tuple[int, int, object],
            tuple[tuple[Any, ...], tuple[Any, ...], "NativeDemSampler"],
        ],
    ],
] = {}
_DEM_LIGHT_SAMPLER_OPERATIONS_CACHE: dict[
    int,
    tuple[
        object,
        dict[
            tuple[
                int,
                int,
                int,
                object,
            ],
            tuple[tuple[Any, ...], tuple[Any, ...], "NativeDemSampler"],
        ],
    ],
] = {}
_DEM_LIGHT_SAMPLER_OPERATIONS_CACHE_MAX = 64
_DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE: dict[
    tuple[int, object, int, int],
    tuple[
        tuple[Operation, ...],
        tuple[Any, ...],
        tuple[Any, ...],
        "NativeDemSampler",
    ],
] = {}
_DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE_MAX = 32
_DEM_GENERATOR_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[Circuit],
        dict[tuple[int, int, object], "NativeDemGenerator"],
    ],
] = {}


class UnsupportedNativeCircuitError(ValueError):
    """Raised when a circuit cannot be compiled by the native sampler."""


class _NativeDemEdgeSequence(Sequence):
    """Lazily materialized Python view of native-owned DEM edges."""

    def __init__(self, circuit: Circuit, native_dem: Any):
        self._circuit = circuit
        self._native_dem = native_dem
        self._edges: tuple[Any, ...] | None = None

    def __len__(self) -> int:
        return int(self._native_dem.edge_count)

    def __getitem__(self, index: int | slice) -> Any:
        return self._materialize()[index]

    def __iter__(self):
        return iter(self._materialize())

    def _materialize(self) -> tuple[Any, ...]:
        if self._edges is None:
            self._edges = _payload_to_detector_error_edges(
                self._circuit,
                self._native_dem.to_rows(),
            )
        return self._edges


@dataclass(frozen=True)
class PackedMeasurementBytes:
    """Column-major packed measurement bytes returned by the native sampler.

    ``data[i * bytes_per_mask:(i + 1) * bytes_per_mask]`` is the little-endian
    bit mask for ``keys[i]``.  The least-significant bit is shot 0.
    """

    shots: int
    keys: tuple[str, ...]
    data: bytes
    bytes_per_mask: int

    def __post_init__(self) -> None:
        expected = len(self.keys) * self.bytes_per_mask
        if len(self.data) != expected:
            raise ValueError(
                f"packed measurement data has {len(self.data)} bytes, expected {expected}"
            )
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if self.bytes_per_mask != 8 * ((self.shots + 63) // 64):
            raise ValueError("bytes_per_mask does not match shots")

    def mask(self, key: str) -> int:
        try:
            index = self.keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        start = index * self.bytes_per_mask
        stop = start + self.bytes_per_mask
        return int.from_bytes(self.data[start:stop], "little")

    def masks(self) -> dict[str, int]:
        return {key: self.mask(key) for key in self.keys}

    def as_packed_numpy(self) -> Any:
        import numpy as np

        return np.frombuffer(self.data, dtype=np.uint8).reshape(
            (len(self.keys), self.bytes_per_mask)
        )


@dataclass(frozen=True)
class NativePackedSampler:
    """Compiled native packed sampler wrapper."""

    circuit: Circuit
    _engine: Any
    observables: tuple[Any, ...] = ()

    def sample(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> BatchTrajectory:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            raise ValueError("native sampler accepts seed, not a Python rng")
        payload = self._engine.sample(int(shots), seed)
        batch = _payload_to_batch_trajectory(payload)
        _validate_declared_observables(batch.observables, self.observables)
        return batch

    def sample_measurements(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> dict[str, int]:
        """Return only packed measurement masks for ordinary sampling benchmarks."""

        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            raise ValueError("native sampler accepts seed, not a Python rng")
        if hasattr(self._engine, "sample_measurements"):
            payload = self._engine.sample_measurements(int(shots), seed)
            return {str(key): int(value) for key, value in payload.items()}
        return self.sample(shots=shots, seed=seed).measurements

    def sample_measurements_packed(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> PackedMeasurementBytes:
        """Return measurement masks as one contiguous column-major byte buffer."""

        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            raise ValueError("native sampler accepts seed, not a Python rng")
        if not hasattr(self._engine, "sample_measurements_packed"):
            measurements = self.sample_measurements(shots=shots, seed=seed)
            bytes_per_mask = 8 * ((int(shots) + 63) // 64)
            keys = tuple(measurements)
            data = b"".join(
                int(measurements[key]).to_bytes(bytes_per_mask, "little")
                for key in keys
            )
            return PackedMeasurementBytes(
                shots=int(shots),
                keys=keys,
                data=data,
                bytes_per_mask=bytes_per_mask,
            )
        payload = self._engine.sample_measurements_packed(int(shots), seed)
        return PackedMeasurementBytes(
            shots=int(payload["shots"]),
            keys=tuple(str(key) for key in payload["keys"]),
            data=bytes(payload["data"]),
            bytes_per_mask=int(payload["bytes_per_mask"]),
        )

    def estimate(
        self,
        *,
        shots: int,
        loss_mask_fn: Any | None = None,
        decoder: Any | None = None,
        correction_mask_fn: Any | None = None,
        seed: int | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if decoder is not None and correction_mask_fn is not None:
            raise ValueError("supply either decoder or correction_mask_fn, not both")
        native_baseline = _native_baseline_value(baseline)
        batch = self._engine.run_native_batch(int(shots), seed)
        corrections = _forward_correction_masks(
            batch,
            decoder,
            correction_mask_fn,
        )
        if loss_mask_fn is None:
            loss_mask = _forward_default_loss_mask(
                batch,
                corrections,
                observables=self.observables,
            )
        else:
            loss_mask = _call_forward_loss_mask_fn(loss_mask_fn, batch, corrections)
        loss_mask &= int(batch.all_mask)
        payload = self._engine.estimate_hotspots(
            batch,
            loss_mask,
            native_baseline,
            int(top_k),
        )
        return _payload_to_simulation_result(self.circuit, payload)


@dataclass(frozen=True)
class NativeDemSampler:
    """Compiled native DEM sampler wrapper."""

    dem: Any
    _engine: Any

    def run_batch(
        self,
        *,
        shots: int,
        seed: int | None = None,
        rng: random.Random | None = None,
        return_edge_events: bool = True,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if seed is not None and rng is not None:
            raise ValueError("supply either seed or rng, not both")
        if rng is not None:
            raise ValueError("native DEM sampler accepts seed, not a Python rng")
        payload = self._engine.run_batch(
            int(shots),
            seed,
            bool(return_edge_events),
        )
        return _payload_to_dem_batch(payload)

    def estimate_default(
        self,
        *,
        shots: int,
        seed: int | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        self._require_dem_metadata()
        native_baseline = _native_baseline_value(baseline)
        payload = self._engine.estimate_default(
            int(shots),
            seed,
            native_baseline,
            int(top_k),
        )
        return _payload_to_dem_hotspot_result(self.dem, payload)

    def estimate(
        self,
        *,
        shots: int,
        seed: int | None = None,
        decoder: Any | None = None,
        correction_mask_fn: Any | None = None,
        loss_mask_fn: Any | None = None,
        baseline: str | float = "mean",
        top_k: int = 10,
    ) -> Any:
        if shots <= 0:
            raise ValueError("shots must be positive")
        self._require_dem_metadata()
        if decoder is not None and correction_mask_fn is not None:
            raise ValueError("supply either decoder or correction_mask_fn, not both")
        native_baseline = _native_baseline_value(baseline)
        if decoder is None and correction_mask_fn is None and loss_mask_fn is None:
            payload = self._engine.estimate_default(
                int(shots),
                seed,
                native_baseline,
                int(top_k),
            )
            return _payload_to_dem_hotspot_result(self.dem, payload)

        batch = self._engine.run_native_batch(int(shots), seed)
        if correction_mask_fn is not None:
            corrections = dict(correction_mask_fn(batch))
        elif decoder is not None:
            if not hasattr(decoder, "decode_batch_masks"):
                raise TypeError("DEM decoder must provide decode_batch_masks(batch)")
            corrections = dict(decoder.decode_batch_masks(batch))
        else:
            corrections = {}
        if loss_mask_fn is None:
            loss_mask = logical_residual_loss_mask(
                batch.observables,
                corrections,
                observable_ids=(observable.id for observable in self.dem.observables),
                all_mask=batch.all_mask,
            )
        else:
            loss_mask = loss_mask_fn(batch, corrections)
        loss_mask = int(loss_mask) & int(batch.all_mask)
        payload = self._engine.estimate_hotspots(
            batch,
            loss_mask,
            native_baseline,
            int(top_k),
        )
        return _payload_to_dem_hotspot_result(self.dem, payload)

    def _require_dem_metadata(self) -> None:
        if self.dem is None:
            raise ValueError(
                "this native DEM sampler was compiled without Python DEM metadata"
            )


@dataclass(frozen=True)
class NativeDemGenerator:
    """Compiled native DEM generator wrapper."""

    circuit: Circuit
    detectors: tuple[Any, ...]
    observables: tuple[Any, ...]
    _engine: Any

    def generate_native_dem(self) -> Any:
        """Generate a native-owned detector error model object."""

        return self._engine.generate_native_dem()

    def generate_dem(self) -> Any:
        native_dem = self._engine.generate_native_dem()
        return _native_generated_dem_to_detector_error_model(
            self.circuit,
            self.detectors,
            self.observables,
            native_dem,
        )

    def compile_sampler(self, *, materialize_dem: bool = True) -> NativeDemSampler:
        if not materialize_dem:
            return NativeDemSampler(
                dem=None,
                _engine=self._engine.compile_sampler(),
            )
        native_dem = self._engine.generate_native_dem()
        dem = _native_generated_dem_to_detector_error_model(
            self.circuit,
            self.detectors,
            self.observables,
            native_dem,
        )
        return NativeDemSampler(
            dem=dem,
            _engine=self._engine.compile_sampler(),
        )


def compile_native_sampler(
    circuit: Circuit,
    *,
    observables: Any | None = None,
) -> NativePackedSampler:
    """Compile ``circuit`` into a native packed sampler."""

    observables_tuple = tuple(observables) if observables is not None else ()

    try:
        spec = _serialize_circuit(
            circuit,
            observables=observables_tuple,
        )
        native_mod = importlib.import_module("npsim._npsim_native")
        engine = native_mod.compile_sampler(spec)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc

    return NativePackedSampler(
        circuit=circuit,
        _engine=engine,
        observables=observables_tuple,
    )


def generate_native_dem(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
) -> Any:
    """Generate a detector error model through the native extension."""

    generator = compile_native_dem_generator(
        circuit,
        detectors=detectors,
        observables=observables,
    )
    return generator.generate_dem()


def compile_native_dem_generator(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
) -> NativeDemGenerator:
    """Compile a circuit into a reusable native DEM generator."""

    detectors, observables = _coerce_dem_declarations(circuit, detectors, observables)
    cached = _native_dem_generator_cache_get(circuit, detectors, observables)
    if cached is not None:
        return cached

    try:
        native_mod = importlib.import_module("npsim._npsim_native")
        serialized_detectors = [_serialize_dem_detector(detector) for detector in detectors]
        serialized_observables = [
            _serialize_dem_observable(observable) for observable in observables
        ]
        direct_compile = getattr(native_mod, "compile_dem_generator_from_circuit", None)
        if direct_compile is not None:
            try:
                engine = direct_compile(
                    circuit,
                    serialized_detectors,
                    serialized_observables,
                )
            except Exception:
                spec = _serialize_circuit(circuit)
                engine = native_mod.compile_dem_generator(
                    spec,
                    serialized_detectors,
                    serialized_observables,
                )
        else:
            spec = _serialize_circuit(circuit)
            engine = native_mod.compile_dem_generator(
                spec,
                serialized_detectors,
                serialized_observables,
            )
        generator = NativeDemGenerator(
            circuit=circuit,
            detectors=detectors,
            observables=observables,
            _engine=engine,
        )
        _native_dem_generator_cache_put(circuit, detectors, observables, generator)
        return generator
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc


def compile_native_dem_sampler_from_circuit(
    circuit: Circuit,
    *,
    detectors: Any | None = None,
    observables: Any | None = None,
    materialize_dem: bool = True,
) -> NativeDemSampler:
    """Generate and compile a native DEM sampler in a single native call."""

    if not materialize_dem:
        detectors, observables = _coerce_dem_declarations(circuit, detectors, observables)
        cache_key = _native_dem_light_sampler_cache_key(
            circuit,
            detectors,
            observables,
        )
        cache = _native_dem_light_sampler_cache(circuit) if cache_key is not None else None
        if cache_key is not None:
            cached_entry = cache.get(cache_key) if cache is not None else None
            if (
                cached_entry is not None
                and cached_entry[0] is detectors
                and cached_entry[1] is observables
            ):
                return cached_entry[2]
            cached = _native_dem_light_sampler_operations_cache_get(
                circuit,
                detectors,
                observables,
                cache_key[2],
            )
            if cached is not None:
                if cache is not None:
                    cache[cache_key] = (detectors, observables, cached)
                    _native_dem_light_sampler_operations_cache_put(
                        circuit,
                        detectors,
                        observables,
                        cache_key[2],
                        cached,
                    )
                return cached
            cached = _native_dem_light_sampler_sequence_cache_get(
                circuit,
                detectors,
                observables,
            )
            if cached is not None:
                if cache is not None:
                    cache[cache_key] = (detectors, observables, cached)
                    _native_dem_light_sampler_operations_cache_put(
                        circuit,
                        detectors,
                        observables,
                        cache_key[2],
                        cached,
                    )
                return cached
        try:
            native_mod = importlib.import_module("npsim._npsim_native")
            serialized_detectors = [_serialize_dem_detector(detector) for detector in detectors]
            serialized_observables = [
                _serialize_dem_observable(observable) for observable in observables
            ]
            direct_compile = getattr(
                native_mod, "compile_generated_dem_sampler_from_circuit", None
            )
            if direct_compile is not None:
                try:
                    engine = direct_compile(
                        circuit,
                        serialized_detectors,
                        serialized_observables,
                    )
                except Exception:
                    spec = _serialize_circuit(circuit, include_tags=False)
                    engine = native_mod.compile_generated_dem_sampler(
                        spec,
                        serialized_detectors,
                        serialized_observables,
                    )
            else:
                spec = _serialize_circuit(circuit, include_tags=False)
                engine = native_mod.compile_generated_dem_sampler(
                    spec,
                    serialized_detectors,
                    serialized_observables,
                )
            sampler = NativeDemSampler(dem=None, _engine=engine)
            if cache_key is not None and cache is not None:
                cache[cache_key] = (detectors, observables, sampler)
                _native_dem_light_sampler_operations_cache_put(
                    circuit,
                    detectors,
                    observables,
                    cache_key[2],
                    sampler,
                )
                _native_dem_light_sampler_sequence_cache_put(
                    circuit,
                    detectors,
                    observables,
                    sampler,
                )
            return sampler
        except Exception as exc:
            raise UnsupportedNativeCircuitError(str(exc)) from exc

    generator = compile_native_dem_generator(
        circuit,
        detectors=detectors,
        observables=observables,
    )
    return generator.compile_sampler(materialize_dem=materialize_dem)


def _native_dem_light_sampler_cache_key(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
) -> tuple[int, int, object] | None:
    if isinstance(circuit.operations, tuple):
        operations_key = None
    elif isinstance(circuit.operations, list):
        operations_key = _operation_sequence_identity_key(circuit.operations)
    else:
        return None
    return id(detectors), id(observables), operations_key


def _operation_sequence_identity_key(operations: Any) -> object:
    try:
        native_mod = importlib.import_module("npsim._npsim_native")
        key_fn = getattr(native_mod, "operation_sequence_identity_key", None)
        if key_fn is not None:
            return key_fn(operations)
    except Exception:
        pass
    return tuple(id(operation) for operation in operations)


def _native_dem_generator_cache_get(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
) -> NativeDemGenerator | None:
    cache_key = _native_dem_generator_cache_key(circuit, detectors, observables)
    if cache_key is None:
        return None
    entry = _DEM_GENERATOR_CACHE.get(id(circuit))
    if entry is None:
        return None
    circuit_ref, cache = entry
    if circuit_ref() is not circuit:
        _DEM_GENERATOR_CACHE.pop(id(circuit), None)
        return None
    return cache.get(cache_key)


def _native_dem_generator_cache_put(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    generator: NativeDemGenerator,
) -> None:
    cache_key = _native_dem_generator_cache_key(circuit, detectors, observables)
    if cache_key is None:
        return
    cache_id = id(circuit)
    entry = _DEM_GENERATOR_CACHE.get(cache_id)
    if entry is None or entry[0]() is not circuit:
        def cleanup(
            _ref: weakref.ReferenceType[Circuit],
            *,
            cache_id: int = cache_id,
        ) -> None:
            _DEM_GENERATOR_CACHE.pop(cache_id, None)

        try:
            circuit_ref = weakref.ref(circuit, cleanup)
        except TypeError:
            return
        cache: dict[tuple[int, int, object], NativeDemGenerator] = {}
        _DEM_GENERATOR_CACHE[cache_id] = (circuit_ref, cache)
    else:
        cache = entry[1]
    cache[cache_key] = generator


def _native_dem_generator_cache_key(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
) -> tuple[int, int, object] | None:
    if isinstance(circuit.operations, tuple):
        operations_key = None
    elif isinstance(circuit.operations, list):
        operations_key = _operation_sequence_identity_key(circuit.operations)
    else:
        return None
    return id(detectors), id(observables), operations_key


def _native_dem_light_sampler_cache(
    circuit: Circuit,
) -> dict[
    tuple[int, int, tuple[int, ...] | None],
    tuple[tuple[Any, ...], tuple[Any, ...], NativeDemSampler],
] | None:
    cache_id = id(circuit)
    entry = _DEM_LIGHT_SAMPLER_CACHE.get(cache_id)
    if entry is not None:
        ref, cache = entry
        if ref() is circuit:
            return cache
        _DEM_LIGHT_SAMPLER_CACHE.pop(cache_id, None)

    def cleanup(_ref: weakref.ReferenceType[Circuit], *, cache_id: int = cache_id) -> None:
        _DEM_LIGHT_SAMPLER_CACHE.pop(cache_id, None)

    try:
        ref = weakref.ref(circuit, cleanup)
    except TypeError:
        return None
    cache: dict[
        tuple[int, int, tuple[int, ...] | None],
        tuple[tuple[Any, ...], tuple[Any, ...], NativeDemSampler],
    ] = {}
    _DEM_LIGHT_SAMPLER_CACHE[cache_id] = (ref, cache)
    return cache


def _native_dem_light_sampler_operations_cache_get(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    operations_key: object,
) -> NativeDemSampler | None:
    entry = _DEM_LIGHT_SAMPLER_OPERATIONS_CACHE.get(id(circuit.operations))
    if entry is None:
        return None
    operations, cache = entry
    if operations is not circuit.operations:
        _DEM_LIGHT_SAMPLER_OPERATIONS_CACHE.pop(id(circuit.operations), None)
        return None
    key = (int(circuit.n_qubits), id(detectors), id(observables), operations_key)
    entry = cache.get(key)
    if entry is None or entry[0] is not detectors or entry[1] is not observables:
        return None
    return entry[2]


def _native_dem_light_sampler_operations_cache_put(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    operations_key: object,
    sampler: NativeDemSampler,
) -> None:
    if len(_DEM_LIGHT_SAMPLER_OPERATIONS_CACHE) >= _DEM_LIGHT_SAMPLER_OPERATIONS_CACHE_MAX:
        _DEM_LIGHT_SAMPLER_OPERATIONS_CACHE.pop(
            next(iter(_DEM_LIGHT_SAMPLER_OPERATIONS_CACHE)),
            None,
        )
    entry = _DEM_LIGHT_SAMPLER_OPERATIONS_CACHE.setdefault(
        id(circuit.operations),
        (circuit.operations, {}),
    )
    operations, cache = entry
    if operations is not circuit.operations:
        cache = {}
        _DEM_LIGHT_SAMPLER_OPERATIONS_CACHE[id(circuit.operations)] = (
            circuit.operations,
            cache,
        )
    key = (int(circuit.n_qubits), id(detectors), id(observables), operations_key)
    cache[key] = (detectors, observables, sampler)


def _native_dem_light_sampler_sequence_cache_get(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
) -> NativeDemSampler | None:
    key = (
        int(circuit.n_qubits),
        _operation_sequence_identity_key(circuit.operations),
        id(detectors),
        id(observables),
    )
    entry = _DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE.get(key)
    if entry is None:
        return None
    _cached_operations, cached_detectors, cached_observables, sampler = entry
    if cached_detectors is detectors and cached_observables is observables:
        return sampler
    _DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE.pop(key, None)
    return None


def _native_dem_light_sampler_sequence_cache_put(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    sampler: NativeDemSampler,
) -> None:
    if (
        len(_DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE)
        >= _DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE_MAX
    ):
        _DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE.pop(
            next(iter(_DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE)),
            None,
        )
    operations = tuple(circuit.operations)
    key = (
        int(circuit.n_qubits),
        _operation_sequence_identity_key(operations),
        id(detectors),
        id(observables),
    )
    _DEM_LIGHT_SAMPLER_OPERATION_SEQUENCE_CACHE[key] = (
        operations,
        detectors,
        observables,
        sampler,
    )


def compile_native_dem_sampler(
    dem: Any,
) -> NativeDemSampler:
    """Compile a detector error model into a packed native DEM sampler."""

    try:
        spec = _serialize_dem(dem)
        native_mod = importlib.import_module("npsim._npsim_native")
        engine = native_mod.compile_dem_sampler(spec)
    except Exception as exc:
        raise UnsupportedNativeCircuitError(str(exc)) from exc

    return NativeDemSampler(
        dem=dem,
        _engine=engine,
    )


def _serialize_circuit(
    circuit: Circuit,
    *,
    observables: tuple[Any, ...] = (),
    include_tags: bool = True,
) -> dict[str, Any]:
    return {
        "format": "compact_v1",
        "n_qubits": int(circuit.n_qubits),
        "operations": [
            _serialize_operation_compact(operation, include_tags=include_tags)
            for operation in circuit.operations
        ],
        "observables": [
            _serialize_dem_observable(observable)
            for observable in observables
        ],
    }


def _serialize_operation_compact(
    operation: Operation,
    *,
    include_tags: bool = True,
) -> tuple[Any, ...]:
    try:
        kind = _NATIVE_OP_KIND[operation.kind]
    except KeyError as exc:
        raise UnsupportedNativeCircuitError(
            f"unsupported native operation kind {operation.kind!r}"
        ) from exc
    detector_id = None
    if operation.kind == "detector":
        detector_id = operation.metadata.get("detector_id")
    return (
        kind,
        tuple(int(qubit) for qubit in operation.qubits),
        operation.key,
        operation.basis,
        operation.pauli,
        tuple(operation.measurement_keys),
        operation.observable_id,
        (
            _serialize_noise_location_compact(
                operation.noise_location,
                include_tags=include_tags,
            )
            if operation.noise_location is not None
            else None
        ),
        detector_id,
    )


def _serialize_operation(operation: Operation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": operation.kind,
        "qubits": tuple(int(qubit) for qubit in operation.qubits),
        "basis": operation.basis,
        "metadata": dict(operation.metadata),
        "measurement_keys": tuple(operation.measurement_keys),
    }
    if operation.key is not None:
        out["key"] = operation.key
    if operation.pauli is not None:
        out["pauli"] = operation.pauli
    if operation.observable_id is not None:
        out["observable_id"] = int(operation.observable_id)
    if operation.noise_location is not None:
        out["noise_location"] = _serialize_noise_location(operation.noise_location)
    return out


def _serialize_noise_location_compact(
    location: NoiseLocation,
    *,
    include_tags: bool = True,
) -> tuple[Any, ...]:
    return (
        location.id,
        float(location.rate),
        tuple(int(qubit) for qubit in location.qubits),
        dict(location.tags) if include_tags else {},
        _serialize_noise_model_compact(location.model),
    )


def _serialize_noise_location(location: NoiseLocation) -> dict[str, Any]:
    return {
        "id": location.id,
        "rate": float(location.rate),
        "qubits": tuple(int(qubit) for qubit in location.qubits),
        "tags": dict(location.tags),
        "model": _serialize_noise_model(location.model),
    }


def _serialize_noise_model_compact(model: object) -> tuple[Any, ...]:
    if isinstance(model, BernoulliPauliNoise):
        return (_NATIVE_NOISE_MODEL["bernoulli_pauli"], model.pauli, ())
    if isinstance(model, MeasurementBitFlip):
        return (_NATIVE_NOISE_MODEL["measurement_bit_flip"], None, ())
    if isinstance(model, SingleQubitDepolarizing):
        return (_NATIVE_NOISE_MODEL["single_qubit_depolarizing"], None, ())
    if isinstance(model, TwoQubitDepolarizing):
        return (_NATIVE_NOISE_MODEL["two_qubit_depolarizing"], None, ())
    if isinstance(model, PauliChannel):
        return (
            _NATIVE_NOISE_MODEL["pauli_channel"],
            None,
            tuple((pauli, float(weight)) for pauli, weight in model.weights.items()),
        )
    raise UnsupportedNativeCircuitError(
        f"unsupported native noise model {type(model).__name__}"
    )


def _serialize_noise_model(model: object) -> dict[str, Any]:
    if isinstance(model, BernoulliPauliNoise):
        return {"type": "bernoulli_pauli", "pauli": model.pauli}
    if isinstance(model, MeasurementBitFlip):
        return {"type": "measurement_bit_flip"}
    if isinstance(model, SingleQubitDepolarizing):
        return {"type": "single_qubit_depolarizing"}
    if isinstance(model, TwoQubitDepolarizing):
        return {"type": "two_qubit_depolarizing"}
    if isinstance(model, PauliChannel):
        return {
            "type": "pauli_channel",
            "weights": tuple(
                (pauli, float(weight))
                for pauli, weight in model.weights.items()
            ),
        }
    raise UnsupportedNativeCircuitError(
        f"unsupported native noise model {type(model).__name__}"
    )


def _coerce_dem_declarations(
    circuit: Circuit,
    detectors: Any | None,
    observables: Any | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if detectors is not None and observables is not None:
        return tuple(detectors), tuple(observables)
    from npsim.dem.model import _detectors_from_circuit, _observables_from_circuit

    return (
        tuple(detectors) if detectors is not None else _detectors_from_circuit(circuit),
        tuple(observables) if observables is not None else _observables_from_circuit(circuit),
    )


def _serialize_dem_detector(detector: Any) -> dict[str, Any]:
    return {
        "id": int(detector.id),
        "measurement_keys": tuple(str(key) for key in detector.measurement_keys),
        "coords": tuple(float(coord) for coord in detector.coords),
    }


def _serialize_dem_observable(observable: Any) -> dict[str, Any]:
    return {
        "id": int(observable.id),
        "measurement_keys": tuple(str(key) for key in observable.measurement_keys),
        "pauli_qubits": tuple(int(qubit) for qubit in observable.pauli_qubits),
        "pauli": str(observable.pauli),
    }


def _serialize_dem(dem: Any) -> dict[str, Any]:
    return {
        "detectors": [_serialize_dem_detector(detector) for detector in dem.detectors],
        "observables": [
            _serialize_dem_observable(observable)
            for observable in dem.observables
        ],
        "edges": [
            {
                "probability": float(edge.probability),
                "detectors": tuple(int(detector_id) for detector_id in edge.detectors),
                "observables": tuple(
                    int(observable_id)
                    for observable_id in edge.observables
                ),
                "location_id": str(edge.location_id),
                "tags": dict(edge.tags),
            }
            for edge in dem.edges
        ],
    }


def _payload_to_detector_error_model(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    payload: Any,
) -> Any:
    from npsim.dem.model import DetectorErrorModel

    return DetectorErrorModel(
        detectors=tuple(detectors),
        observables=tuple(observables),
        edges=_payload_to_detector_error_edges(circuit, payload),
    )


def _native_generated_dem_to_detector_error_model(
    circuit: Circuit,
    detectors: tuple[Any, ...],
    observables: tuple[Any, ...],
    native_dem: Any,
) -> Any:
    from npsim.dem.model import DetectorErrorModel

    return DetectorErrorModel(
        detectors=tuple(detectors),
        observables=tuple(observables),
        edges=_NativeDemEdgeSequence(circuit, native_dem),
    )


def _payload_to_detector_error_edges(
    circuit: Circuit,
    payload: Any,
) -> tuple[Any, ...]:
    from npsim.dem.model import DetectorErrorEdge

    locations = circuit.noise_locations()
    edges = []
    for item in payload:
        if hasattr(item, "keys"):
            probability = item["probability"]
            detectors_payload = item["detectors"]
            observables_payload = item["observables"]
            location_id = str(item["location_id"])
            event = item["event"]
        else:
            (
                probability,
                detectors_payload,
                observables_payload,
                location_id,
                event,
            ) = item
            location_id = str(location_id)
        location = locations.get(location_id)
        edges.append(
            DetectorErrorEdge(
                probability=float(probability),
                detectors=tuple(int(value) for value in detectors_payload),
                observables=tuple(int(value) for value in observables_payload),
                location_id=location_id,
                event=event,
                tags=dict(location.tags) if location is not None else {},
            )
        )
    return tuple(edges)


def _payload_to_dem_batch(payload: Mapping[str, Any]) -> Any:
    from npsim.dem.sampler import DemBatchTrajectory

    return DemBatchTrajectory(
        shots=int(payload["shots"]),
        all_mask=int(payload["all_mask"]),
        detectors={int(key): int(value) for key, value in payload["detectors"].items()},
        observables={
            int(key): int(value)
            for key, value in payload["observables"].items()
        },
        edge_event_masks={
            int(key): int(value)
            for key, value in payload["edge_event_masks"].items()
        },
    )


def _payload_to_dem_hotspot_result(dem: Any, payload: Mapping[str, Any]) -> Any:
    from npsim.dem.sampler import (
        DemEdgeHotspotRow,
        DemHotspotResult,
        DemLocationHotspotRow,
        DemLocationMetadata,
    )

    edge_sensitivities = _required_payload_field(payload, "edge_sensitivities")
    edge_hotspots = _required_payload_field(payload, "edge_hotspots")
    sensitivities = _required_payload_field(payload, "sensitivities")
    hotspots = _required_payload_field(payload, "hotspots")
    locations = _dem_location_metadata(dem)
    by_detector = _required_payload_field(payload, "by_detector")
    detector_graph_hotspots = _payload_to_detector_graph_hotspots(
        dem,
        _required_payload_field(payload, "detector_graph_hotspots"),
    )
    top_edges_cache = tuple(
        DemEdgeHotspotRow(
            edge_index=int(row["edge_index"]),
            location_id=dem.edges[int(row["edge_index"])].location_id,
            event=dem.edges[int(row["edge_index"])].event,
            probability=dem.edges[int(row["edge_index"])].probability,
            detectors=dem.edges[int(row["edge_index"])].detectors,
            observables=dem.edges[int(row["edge_index"])].observables,
            sensitivity=float(row["sensitivity"]),
            hotspot=float(row["hotspot"]),
        )
        for row in _required_payload_field(payload, "top_edges")
    )
    top_hotspots_cache = tuple(
        DemLocationHotspotRow(
            location_id=str(row["location_id"]),
            sensitivity=float(row["sensitivity"]),
            hotspot=float(row["hotspot"]),
            qubits=locations[str(row["location_id"])].qubits,
            tags=locations[str(row["location_id"])].tags,
        )
        for row in _required_payload_field(payload, "top_hotspots")
    )
    return DemHotspotResult(
        dem=dem,
        shots=int(payload["shots"]),
        mean_loss=float(payload["mean_loss"]),
        baseline=float(payload["baseline"]),
        edge_sensitivities=edge_sensitivities,
        edge_hotspots=edge_hotspots,
        sensitivities=sensitivities,
        hotspots=hotspots,
        by_detector=by_detector,
        by_round=_required_payload_field(payload, "by_round"),
        by_gate=_required_payload_field(payload, "by_gate"),
        by_operation=_required_payload_field(payload, "by_operation"),
        locations=locations,
        detector_graph_hotspots=detector_graph_hotspots,
        top_edges_cache=top_edges_cache,
        top_hotspots_cache=top_hotspots_cache,
    )


def _required_payload_field(payload: Mapping[str, Any], field: str) -> Any:
    try:
        return payload[field]
    except KeyError as exc:
        raise UnsupportedNativeCircuitError(
            f"native payload is missing {field}"
        ) from exc


def _payload_to_detector_graph_hotspots(dem: Any, payload: Mapping[str, Any]) -> Any:
    from npsim.dem.model import DetectorGraphEdgeHotspot, DetectorGraphHotspots

    edge_hotspots = []
    for row in payload["edge_hotspots"]:
        edge_index = int(row["edge_index"])
        edge = dem.edges[edge_index]
        edge_hotspots.append(
            DetectorGraphEdgeHotspot(
                edge_index=edge_index,
                location_id=edge.location_id,
                event=edge.event,
                probability=edge.probability,
                detectors=edge.detectors,
                observables=edge.observables,
                sensitivity=float(row["sensitivity"]),
                hotspot=float(row["hotspot"]),
                weight=1.0,
            )
        )

    return DetectorGraphHotspots(
        edge_hotspots=tuple(edge_hotspots),
        by_detector_edge=_graph_key_rows_to_dict(payload["by_detector_edge"]),
        signed_by_detector_edge=_graph_key_rows_to_dict(payload["signed_by_detector_edge"]),
        by_detector={int(key): float(value) for key, value in payload["by_detector"].items()},
        signed_by_detector={
            int(key): float(value)
            for key, value in payload["signed_by_detector"].items()
        },
        by_observable={
            int(key): float(value)
            for key, value in payload["by_observable"].items()
        },
        signed_by_observable={
            int(key): float(value)
            for key, value in payload["signed_by_observable"].items()
        },
        by_location={
            str(key): float(value)
            for key, value in payload["by_location"].items()
        },
        signed_by_location={
            str(key): float(value)
            for key, value in payload["signed_by_location"].items()
        },
    )


def _graph_key_rows_to_dict(rows: Any) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    if hasattr(rows, "items"):
        return rows
    return {
        (
            tuple(int(value) for value in row["detectors"]),
            tuple(int(value) for value in row["observables"]),
        ): float(row["value"])
        for row in rows
    }


def _payload_to_simulation_result(circuit: Circuit, payload: Mapping[str, Any]) -> Any:
    from npsim.runtime.results import HotspotRow, SimulationResult

    locations = circuit.noise_locations()
    sensitivities = payload["sensitivities"]
    hotspots = payload["hotspots"]
    top_hotspots_cache = tuple(
        HotspotRow(
            location_id=str(row["location_id"]),
            sensitivity=float(row["sensitivity"]),
            hotspot=float(row["hotspot"]),
            qubits=locations[str(row["location_id"])].qubits,
            tags=locations[str(row["location_id"])].tags,
        )
        for row in _required_payload_field(payload, "top_hotspots")
    )
    return SimulationResult(
        shots=int(payload["shots"]),
        mean_loss=float(payload["mean_loss"]),
        baseline=float(payload["baseline"]),
        sensitivities=sensitivities,
        hotspots=hotspots,
        by_qubit=payload["by_qubit"],
        by_round=payload["by_round"],
        by_gate=payload["by_gate"],
        by_operation=payload["by_operation"],
        locations=locations,
        losses=[],
        top_hotspots_cache=top_hotspots_cache,
    )


def _dem_location_metadata(dem: Any) -> dict[str, Any]:
    from npsim.dem.sampler import DemLocationMetadata

    out: dict[str, DemLocationMetadata] = {}
    for edge in dem.edges:
        if edge.location_id not in out:
            out[edge.location_id] = DemLocationMetadata(
                id=edge.location_id,
                tags=edge.tags,
            )
    return out


def _native_baseline_value(baseline: str | float) -> float | None:
    if baseline == "mean":
        return None
    if isinstance(baseline, (int, float)):
        return float(baseline)
    raise ValueError("baseline must be 'mean' or a numeric value")


def _forward_correction_masks(
    batch: Any,
    decoder: Any | None,
    correction_mask_fn: Any | None,
) -> Mapping[int, int]:
    if correction_mask_fn is not None:
        return dict(correction_mask_fn(batch))
    if decoder is None:
        return {}
    if not hasattr(decoder, "decode_batch_masks"):
        raise TypeError("batch decoder must provide decode_batch_masks(batch)")
    return dict(decoder.decode_batch_masks(batch))


def _forward_default_loss_mask(
    batch: Any,
    corrections: Mapping[Any, int],
    *,
    observables: tuple[Any, ...] = (),
) -> int:
    _validate_declared_observables(batch.observables, observables)
    return logical_residual_loss_mask(
        batch.observables,
        corrections,
        observable_ids=_observable_ids(observables),
        all_mask=batch.all_mask,
    )


def _observable_ids(observables: tuple[Any, ...]) -> tuple[int, ...]:
    return tuple(int(observable.id) for observable in observables)


def _validate_declared_observables(
    observed: Mapping[Any, int],
    declared: tuple[Any, ...],
) -> None:
    missing = [
        observable_id
        for observable_id in _observable_ids(declared)
        if observable_id not in observed
    ]
    if missing:
        raise UnsupportedNativeCircuitError(
            "native batch did not return declared logical observables: "
            + ", ".join(str(observable_id) for observable_id in missing)
        )


def _call_forward_loss_mask_fn(
    loss_mask_fn: Any,
    batch: Any,
    corrections: Mapping[Any, int],
) -> int:
    if _accepts_positional_args(loss_mask_fn, 2):
        return int(loss_mask_fn(batch, corrections))
    return int(loss_mask_fn(batch))


def _accepts_positional_args(fn: Any, count: int) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= count


def _payload_to_batch_trajectory(payload: Mapping[str, Any]) -> BatchTrajectory:
    return BatchTrajectory(
        shots=int(payload["shots"]),
        all_mask=int(payload["all_mask"]),
        x_frame=tuple(int(mask) for mask in payload["x_frame"]),
        z_frame=tuple(int(mask) for mask in payload["z_frame"]),
        measurements={
            str(key): int(value)
            for key, value in payload["measurements"].items()
        },
        detectors={int(key): int(value) for key, value in payload["detectors"].items()},
        observables={int(key): int(value) for key, value in payload["observables"].items()},
        noise_event_masks={
            str(key): int(value) for key, value in payload["noise_event_masks"].items()
        },
    )


__all__ = [
    "NativeDemGenerator",
    "NativeDemSampler",
    "NativePackedSampler",
    "PackedMeasurementBytes",
    "UnsupportedNativeCircuitError",
    "compile_native_dem_generator",
    "compile_native_dem_sampler",
    "compile_native_dem_sampler_from_circuit",
    "compile_native_sampler",
    "generate_native_dem",
]
