import unittest
from typing import Iterable, Sequence

from npsim.batch import BatchTrajectory
from npsim.circuit import Circuit, NoiseLocation, Operation
from npsim.dem import Detector, LogicalObservable
from npsim.native import UnsupportedNativeCircuitError, compile_native_sampler
from npsim.native import compile_native_dem_sampler, generate_native_dem
from npsim.noise import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from npsim.repetition import make_repetition_code_experiment
from tests.surface_code_examples import make_large_rotated_surface_code_memory_example


try:
    import numpy as np
except ImportError:  # pragma: no cover - optional test dependency
    np = None

try:
    import stim
except ImportError:  # pragma: no cover - optional test dependency
    stim = None


class StimSamplingComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        if stim is None:
            self.skipTest("Stim is not installed")
        if np is None:
            self.skipTest("NumPy is not installed")

    def test_basic_clifford_noise_and_reset_sampling_matches_stim(self) -> None:
        mflip = NoiseLocation(
            id="mflip",
            model=MeasurementBitFlip(),
            rate=0.07,
            qubits=(0,),
        )
        circuit = Circuit(
            n_qubits=3,
            operations=[
                Operation.h(0),
                Operation.cx(0, 1),
                Operation.cz(1, 2),
                Operation.swap(1, 2),
                Operation.s(2),
                Operation.s_dag(2),
                Operation.y(2),
                Operation.noise(
                    NoiseLocation(
                        id="pc0",
                        model=PauliChannel({"X": 2.0, "Z": 1.0}),
                        rate=0.19,
                        qubits=(0,),
                    )
                ),
                Operation.noise(
                    NoiseLocation(
                        id="depol1",
                        model=SingleQubitDepolarizing(),
                        rate=0.13,
                        qubits=(1,),
                    )
                ),
                Operation.noise(
                    NoiseLocation(
                        id="depol12",
                        model=TwoQubitDepolarizing(),
                        rate=0.06,
                        qubits=(1, 2),
                    )
                ),
                Operation.measure(0, key="m0", basis="Z", noise=mflip),
                Operation.measure(1, key="m1", basis="X"),
                Operation.measure_pauli((0, 2), "XZ", key="m2"),
                Operation.reset(2, key="r2", basis="X"),
                Operation.measure(2, key="after_reset", basis="X"),
            ],
        )
        batch, stim_samples, key_order = _sample_both(circuit, shots=40_000)

        for key in key_order:
            _assert_key_rate_close(self, batch, stim_samples, key_order, key)
        for keys in (("m0", "m1"), ("m0", "m2"), ("m1", "m2", "r2")):
            _assert_parity_rate_close(self, batch, stim_samples, key_order, keys)

    def test_repetition_code_example_sampling_matches_stim(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(1, 2): 0.17, (2, 0): 0.09},
            measurement_error_rate=0.035,
        )
        circuit = _with_final_data_measurements(
            experiment.circuit,
            experiment.data_qubits,
            basis="Z",
            prefix="final_d",
        )
        batch, stim_samples, key_order = _sample_both(circuit, shots=35_000)

        for key in key_order:
            if key.startswith("r") or key.startswith("final_d"):
                _assert_key_rate_close(self, batch, stim_samples, key_order, key)
        for detector in experiment.detectors:
            _assert_parity_rate_close(
                self,
                batch,
                stim_samples,
                key_order,
                detector.measurement_keys,
            )

        npsim_loss = _repetition_final_data_loss_mask(batch, distance=5, rounds=3)
        stim_loss = _stim_repetition_final_data_loss_rate(
            stim_samples,
            key_order,
            distance=5,
            rounds=3,
        )
        _assert_rates_close(
            self,
            npsim_loss.bit_count() / batch.shots,
            stim_loss,
            batch.shots,
            "repetition final-data logical loss",
        )

    def test_rotated_surface_code_example_measurements_match_stim(self) -> None:
        example = make_large_rotated_surface_code_memory_example(
            distance=5,
            rounds=2,
        )
        batch, stim_samples, key_order = _sample_both(example.circuit, shots=18_000)

        selected_keys = {
            f"r0_{example.hot_x_check}",
            f"r0_{example.hot_z_check}",
            f"r1_{example.hot_x_check}",
            f"r1_{example.hot_z_check}",
            f"r2_{example.hot_x_check}",
            f"r2_{example.hot_z_check}",
        }
        selected_keys.update(key for key in key_order if key.startswith("r2_"))
        for key in sorted(selected_keys):
            _assert_key_rate_close(self, batch, stim_samples, key_order, key)

        spec = example.native_loss_spec("surface_diagnostic")
        self.assertIsNotNone(spec)
        assert spec is not None
        for measurement_pair in spec["measurement_pairs"]:
            _assert_parity_rate_close(
                self,
                batch,
                stim_samples,
                key_order,
                measurement_pair,
            )

    def test_rotated_surface_code_final_path_readouts_match_stim(self) -> None:
        example = make_large_rotated_surface_code_memory_example(
            distance=5,
            rounds=2,
        )
        spec = example.native_loss_spec("surface_diagnostic")
        self.assertIsNotNone(spec)
        assert spec is not None

        for prefix, qubits, basis in (
            ("x_path_z_readout", spec["x_qubits"], "Z"),
            ("z_path_x_readout", spec["z_qubits"], "X"),
        ):
            with self.subTest(prefix=prefix):
                circuit = _with_final_data_measurements(
                    example.circuit,
                    qubits,
                    basis=basis,
                    prefix=prefix,
                )
                batch, stim_samples, key_order = _sample_both(circuit, shots=14_000)
                path_keys = tuple(f"{prefix}_{idx}" for idx in range(len(qubits)))
                for key in path_keys:
                    _assert_key_rate_close(self, batch, stim_samples, key_order, key)
                _assert_parity_rate_close(
                    self,
                    batch,
                    stim_samples,
                    key_order,
                    path_keys,
                )

    def test_native_dem_generator_matches_stim_detector_error_model(self) -> None:
        circuit = Circuit(
            n_qubits=2,
            operations=[
                Operation.noise(
                    NoiseLocation(
                        id="data_x",
                        model=BernoulliPauliNoise("X"),
                        rate=0.125,
                        qubits=(0,),
                    )
                ),
                Operation.measure(0, key="m0", basis="Z"),
                Operation.detector(("m0",), detector_id=0),
                Operation.observable_include(0, ("m0",)),
                Operation.measure(
                    1,
                    key="m1",
                    basis="Z",
                    noise=NoiseLocation(
                        id="meas_flip",
                        model=MeasurementBitFlip(),
                        rate=0.2,
                        qubits=(1,),
                    ),
                ),
                Operation.detector(("m1",), detector_id=1),
                Operation.observable_include(1, ("m1",)),
            ],
        )

        try:
            npsim_dem = generate_native_dem(circuit, backend="native")
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native DEM generator unavailable: {exc}")
        stim_circuit, _ = _to_stim_circuit(circuit)
        stim_dem = stim_circuit.detector_error_model(decompose_errors=False)

        self.assertEqual(_npsim_dem_error_edges(npsim_dem), _stim_dem_error_edges(stim_dem))

    def test_native_dem_sampler_matches_stim_dem_sampler(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(0, 1): 0.12, (1, 3): 0.09, (2, 0): 0.07},
            measurement_error_rate={(0, 0): 0.04, (1, 2): 0.06, (2, 3): 0.05},
        )
        circuit = _with_dem_declarations(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=(),
        )
        try:
            npsim_dem = generate_native_dem(
                experiment.circuit,
                detectors=experiment.detectors,
                observables=(),
                backend="native",
            )
            npsim_batch = compile_native_dem_sampler(
                npsim_dem,
                backend="native",
            ).run_batch(shots=45_000, seed=24680)
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native DEM mode unavailable: {exc}")

        stim_circuit, _ = _to_stim_circuit(circuit)
        stim_dem = stim_circuit.detector_error_model(decompose_errors=False)
        stim_detectors, stim_observables, _ = stim_dem.compile_sampler(seed=13579).sample(
            npsim_batch.shots,
        )

        for detector in experiment.detectors:
            detector_id = detector.id
            npsim_rate = npsim_batch.detectors[detector_id].bit_count() / npsim_batch.shots
            stim_rate = float(stim_detectors[:, detector_id].mean())
            _assert_rates_close(
                self,
                npsim_rate,
                stim_rate,
                npsim_batch.shots,
                f"DEM detector D{detector_id}",
            )

        self.assertEqual(stim_observables.shape[1], 0)


def _sample_both(
    circuit: Circuit,
    *,
    shots: int,
    native_seed: int = 12345,
    stim_seed: int = 67890,
) -> tuple[BatchTrajectory, object, tuple[str, ...]]:
    try:
        sampler = compile_native_sampler(circuit, backend="native")
    except UnsupportedNativeCircuitError as exc:
        raise unittest.SkipTest(f"native forward sampler unavailable: {exc}") from exc
    batch = sampler.sample(shots=shots, seed=native_seed)
    stim_circuit, key_order = _to_stim_circuit(circuit)
    samples = stim_circuit.compile_sampler(seed=stim_seed).sample(shots)
    return batch, samples, key_order


def _to_stim_circuit(circuit: Circuit) -> tuple[object, tuple[str, ...]]:
    assert stim is not None
    out = stim.Circuit()
    measurement_keys: list[str] = []
    measurement_index_by_key: dict[str, int] = {}

    for operation in circuit.operations:
        kind = operation.kind
        if kind in {"h", "s", "s_dag"}:
            gate = {"h": "H", "s": "S", "s_dag": "S_DAG"}[kind]
            out.append(gate, operation.qubits)
        elif kind in {"cx", "cz", "swap"}:
            gate = {"cx": "CX", "cz": "CZ", "swap": "SWAP"}[kind]
            out.append(gate, operation.qubits)
        elif kind == "pauli":
            if operation.pauli is None:
                raise ValueError("pauli operation requires a Pauli string")
            _append_pauli_gate(out, operation.qubits, operation.pauli)
        elif kind == "noise":
            if operation.noise_location is None:
                raise ValueError("noise operation requires a noise location")
            _append_noise(out, operation.noise_location)
        elif kind == "measure":
            basis = operation.basis.upper()
            gate = {"Z": "M", "X": "MX", "Y": "MY"}[basis]
            _append_measurement_gate(out, gate, operation.qubits, operation.noise_location)
            _record_measurement_key(
                operation.key,
                measurement_keys,
                measurement_index_by_key,
            )
        elif kind == "measure_pauli":
            if operation.pauli is None:
                raise ValueError("measure_pauli operation requires a Pauli string")
            _append_measurement_gate(
                out,
                "MPP",
                _mpp_targets(operation.qubits, operation.pauli),
                operation.noise_location,
            )
            _record_measurement_key(
                operation.key,
                measurement_keys,
                measurement_index_by_key,
            )
        elif kind == "reset":
            (qubit,) = operation.qubits
            basis = operation.basis.upper()
            if operation.key is None:
                out.append({"Z": "R", "X": "RX", "Y": "RY"}[basis], [qubit])
            else:
                out.append({"Z": "MR", "X": "MRX", "Y": "MRY"}[basis], [qubit])
                _record_measurement_key(
                    operation.key,
                    measurement_keys,
                    measurement_index_by_key,
                )
        elif kind == "detector":
            out.append(
                "DETECTOR",
                _rec_targets(operation.measurement_keys, measurement_index_by_key),
                operation.metadata.get("coords", ()),
            )
        elif kind == "observable_include":
            if operation.observable_id is None:
                raise ValueError("observable_include requires observable_id")
            out.append(
                "OBSERVABLE_INCLUDE",
                _rec_targets(operation.measurement_keys, measurement_index_by_key),
                operation.observable_id,
            )
        else:
            raise ValueError(f"unsupported operation kind {kind!r}")

    return out, tuple(measurement_keys)


def _append_measurement_gate(
    circuit: object,
    gate: str,
    targets: Sequence[object] | Sequence[int],
    location: NoiseLocation | None,
) -> None:
    if location is None:
        circuit.append(gate, targets)
        return
    if not isinstance(location.model, MeasurementBitFlip):
        raise ValueError("Stim comparison only supports MeasurementBitFlip on measurements")
    circuit.append(gate, targets, location.rate)


def _append_noise(circuit: object, location: NoiseLocation) -> None:
    model = location.model
    if isinstance(model, BernoulliPauliNoise):
        if len(model.pauli) == 1:
            gate = {"X": "X_ERROR", "Y": "Y_ERROR", "Z": "Z_ERROR", "I": None}[
                model.pauli
            ]
            if gate is not None:
                circuit.append(gate, location.qubits, location.rate)
            return
        circuit.append("E", _correlated_error_targets(location.qubits, model.pauli), location.rate)
        return
    if isinstance(model, SingleQubitDepolarizing):
        circuit.append("DEPOLARIZE1", location.qubits, location.rate)
        return
    if isinstance(model, TwoQubitDepolarizing):
        circuit.append("DEPOLARIZE2", location.qubits, location.rate)
        return
    if isinstance(model, PauliChannel):
        probabilities = _pauli_channel_probabilities(model, location.rate)
        if model.event_length == 1:
            circuit.append(
                "PAULI_CHANNEL_1",
                location.qubits,
                [probabilities.get(pauli, 0.0) for pauli in ("X", "Y", "Z")],
            )
            return
        if model.event_length == 2:
            events = (
                "IX",
                "IY",
                "IZ",
                "XI",
                "XX",
                "XY",
                "XZ",
                "YI",
                "YX",
                "YY",
                "YZ",
                "ZI",
                "ZX",
                "ZY",
                "ZZ",
            )
            circuit.append(
                "PAULI_CHANNEL_2",
                location.qubits,
                [probabilities.get(event, 0.0) for event in events],
            )
            return
    raise ValueError(f"unsupported Stim comparison noise model {type(model).__name__}")


def _append_pauli_gate(circuit: object, qubits: Sequence[int], pauli: str) -> None:
    by_gate: dict[str, list[int]] = {"X": [], "Y": [], "Z": []}
    for qubit, local_pauli in zip(qubits, pauli):
        if local_pauli in by_gate:
            by_gate[local_pauli].append(qubit)
        elif local_pauli != "I":
            raise ValueError(f"unsupported Pauli {local_pauli!r}")
    for gate, targets in by_gate.items():
        if targets:
            circuit.append(gate, targets)


def _mpp_targets(qubits: Sequence[int], pauli: str) -> list[object]:
    assert stim is not None
    factors = _pauli_targets(qubits, pauli)
    if not factors:
        raise ValueError("Stim MPP comparison does not support empty Pauli products")
    targets: list[object] = []
    for idx, target in enumerate(factors):
        if idx:
            targets.append(stim.target_combiner())
        targets.append(target)
    return targets


def _correlated_error_targets(qubits: Sequence[int], pauli: str) -> list[object]:
    targets = _pauli_targets(qubits, pauli)
    if not targets:
        return []
    return targets


def _pauli_targets(qubits: Sequence[int], pauli: str) -> list[object]:
    assert stim is not None
    targets: list[object] = []
    for qubit, local_pauli in zip(qubits, pauli):
        if local_pauli == "I":
            continue
        if local_pauli == "X":
            targets.append(stim.target_x(qubit))
        elif local_pauli == "Y":
            targets.append(stim.target_y(qubit))
        elif local_pauli == "Z":
            targets.append(stim.target_z(qubit))
        else:
            raise ValueError(f"unsupported Pauli {local_pauli!r}")
    return targets


def _pauli_channel_probabilities(model: PauliChannel, rate: float) -> dict[str, float]:
    total_weight = model.total_weight
    return {
        event: rate * weight / total_weight
        for event, weight in model.weights.items()
        if weight > 0
    }


def _record_measurement_key(
    key: str | None,
    measurement_keys: list[str],
    measurement_index_by_key: dict[str, int],
) -> None:
    if key is None:
        key = f"m{len(measurement_keys)}"
    if key in measurement_index_by_key:
        raise ValueError(f"duplicate measurement key {key!r}")
    measurement_index_by_key[key] = len(measurement_keys)
    measurement_keys.append(key)


def _rec_targets(
    keys: Sequence[str],
    measurement_index_by_key: dict[str, int],
) -> list[object]:
    assert stim is not None
    current_index = len(measurement_index_by_key)
    return [
        stim.target_rec(measurement_index_by_key[key] - current_index)
        for key in keys
    ]


def _assert_key_rate_close(
    testcase: unittest.TestCase,
    batch: BatchTrajectory,
    stim_samples: object,
    key_order: Sequence[str],
    key: str,
) -> None:
    key_to_col = {name: col for col, name in enumerate(key_order)}
    npsim_rate = batch.measurements[key].bit_count() / batch.shots
    stim_rate = float(stim_samples[:, key_to_col[key]].mean())
    _assert_rates_close(testcase, npsim_rate, stim_rate, batch.shots, f"measurement {key}")


def _assert_parity_rate_close(
    testcase: unittest.TestCase,
    batch: BatchTrajectory,
    stim_samples: object,
    key_order: Sequence[str],
    keys: Sequence[str],
) -> None:
    npsim_mask = 0
    for key in keys:
        npsim_mask ^= batch.measurements[key]
    npsim_rate = npsim_mask.bit_count() / batch.shots
    stim_rate = _stim_parity_rate(stim_samples, key_order, keys)
    _assert_rates_close(testcase, npsim_rate, stim_rate, batch.shots, f"parity {tuple(keys)}")


def _stim_parity_rate(
    stim_samples: object,
    key_order: Sequence[str],
    keys: Sequence[str],
) -> float:
    assert np is not None
    key_to_col = {name: col for col, name in enumerate(key_order)}
    parity = np.zeros(stim_samples.shape[0], dtype=bool)
    for key in keys:
        parity ^= stim_samples[:, key_to_col[key]]
    return float(parity.mean())


def _assert_rates_close(
    testcase: unittest.TestCase,
    npsim_rate: float,
    stim_rate: float,
    shots: int,
    label: str,
) -> None:
    pooled = 0.5 * (npsim_rate + stim_rate)
    sigma = (2.0 * pooled * (1.0 - pooled) / shots) ** 0.5
    tolerance = max(0.025, 7.0 * sigma, 20.0 / shots)
    testcase.assertLessEqual(
        abs(npsim_rate - stim_rate),
        tolerance,
        f"{label}: NPSim={npsim_rate:.6g}, Stim={stim_rate:.6g}, tolerance={tolerance:.6g}",
    )


def _with_final_data_measurements(
    circuit: Circuit,
    qubits: Iterable[int],
    *,
    basis: str,
    prefix: str,
) -> Circuit:
    operations = list(circuit.operations)
    for idx, qubit in enumerate(qubits):
        operations.append(Operation.measure(int(qubit), key=f"{prefix}_{idx}", basis=basis))
    return Circuit(n_qubits=circuit.n_qubits, operations=operations)


def _with_dem_declarations(
    circuit: Circuit,
    *,
    detectors: Sequence[Detector],
    observables: Sequence[LogicalObservable],
) -> Circuit:
    operations = list(circuit.operations)
    for detector in detectors:
        operations.append(
            Operation.detector(
                detector.measurement_keys,
                detector_id=detector.id,
                coords=detector.coords,
            )
        )
    for observable in observables:
        if observable.pauli:
            raise ValueError("Stim comparison needs measurement-only observables")
        operations.append(
            Operation.observable_include(
                observable.id,
                observable.measurement_keys,
            )
        )
    return Circuit(n_qubits=circuit.n_qubits, operations=operations)


def _npsim_dem_error_edges(dem: object) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                round(float(edge.probability), 12),
                tuple(int(detector_id) for detector_id in edge.detectors),
                tuple(int(observable_id) for observable_id in edge.observables),
            )
            for edge in dem.edges
        )
    )


def _stim_dem_error_edges(stim_dem: object) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
    edges: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    for instruction in stim_dem:
        if instruction.type != "error":
            continue
        detectors: list[int] = []
        observables: list[int] = []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(int(target.val))
            elif target.is_logical_observable_id():
                observables.append(int(target.val))
            elif not target.is_separator():
                raise ValueError(f"unsupported Stim DEM target {target!r}")
        edges.append(
            (
                round(float(instruction.args_copy()[0]), 12),
                tuple(detectors),
                tuple(observables),
            )
        )
    return tuple(sorted(edges))


def _repetition_final_data_loss_mask(
    batch: BatchTrajectory,
    *,
    distance: int,
    rounds: int,
) -> int:
    loss_mask = 0
    final_round = rounds - 1
    for shot in range(batch.shots):
        syndrome = [
            batch.measurement_bit(f"r{final_round}_c{check_idx}", shot)
            for check_idx in range(distance - 1)
        ]
        correction = _decode_repetition_shot(syndrome)
        residual_weight = 0
        for data_idx in range(distance):
            residual_weight += (
                batch.measurement_bit(f"final_d_{data_idx}", shot)
                ^ correction[data_idx]
            )
        if residual_weight > distance // 2:
            loss_mask |= 1 << shot
    return loss_mask


def _stim_repetition_final_data_loss_rate(
    stim_samples: object,
    key_order: Sequence[str],
    *,
    distance: int,
    rounds: int,
) -> float:
    key_to_col = {name: col for col, name in enumerate(key_order)}
    final_round = rounds - 1
    failures = 0
    for shot in range(stim_samples.shape[0]):
        syndrome = [
            int(stim_samples[shot, key_to_col[f"r{final_round}_c{check_idx}"]])
            for check_idx in range(distance - 1)
        ]
        correction = _decode_repetition_shot(syndrome)
        residual_weight = 0
        for data_idx in range(distance):
            residual_weight += (
                int(stim_samples[shot, key_to_col[f"final_d_{data_idx}"]])
                ^ correction[data_idx]
            )
        failures += int(residual_weight > distance // 2)
    return failures / stim_samples.shape[0]


def _decode_repetition_shot(syndrome: list[int]) -> list[int]:
    candidate = [0] * (len(syndrome) + 1)
    for idx, bit in enumerate(syndrome):
        candidate[idx + 1] = candidate[idx] ^ int(bit)
    complement = [bit ^ 1 for bit in candidate]
    return candidate if sum(candidate) <= sum(complement) else complement
