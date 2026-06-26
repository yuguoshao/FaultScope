import unittest
from typing import Any, Sequence

from faultscope.runtime import SampleBatch
from faultscope.core import Circuit, NoiseLocation, Operation
from faultscope.decoders import (
    PyMatchingDecoder,
    PyMatchingUnavailableError,
    UnsupportedPyMatchingDemError,
)
from faultscope.dem import Detector, LogicalObservable
from faultscope.io import parse_stim_circuit
from faultscope.runtime import UnsupportedNativeCircuitError, compile_native_sampler
from faultscope.runtime import compile_native_dem_sampler, generate_native_dem
from faultscope.runtime.loss import logical_residual_loss_mask
from faultscope.core import (
    BernoulliPauliNoise,
    MeasurementBitFlip,
    PauliChannel,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
)
from faultscope.experiments import make_repetition_code_experiment
from tests.surface_code_examples import (
    _data_index,
    _rotated_surface_code_checks,
    make_large_rotated_surface_code_memory_example,
)
from tests.stim_helpers import (
    dem_batch_from_stim_samples,
    dense_predictions_to_masks,
    final_data_measurement_circuit,
    masks_to_dense_array,
    measurement_batch_from_stim_samples,
    faultscope_dem_error_edges,
    stim_dem_error_edges,
    stim_observable_masks,
    to_stim_circuit,
    with_dem_declarations,
)


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
            self.fail("Stim is required for core sampling comparison tests")
        if np is None:
            self.fail("NumPy is required for core sampling comparison tests")

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
        circuit = final_data_measurement_circuit(
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

        faultscope_loss = _repetition_final_data_loss_mask(batch, distance=5, rounds=3)
        stim_loss = _stim_repetition_final_data_loss_rate(
            stim_samples,
            key_order,
            distance=5,
            rounds=3,
        )
        _assert_rates_close(
            self,
            faultscope_loss.bit_count() / batch.shots,
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

        for measurement_pair in example.terminal_measurement_pairs:
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

        for prefix, qubits, basis in (
            ("x_path_z_readout", example.x_logical_qubits, "Z"),
            ("z_path_x_readout", example.z_logical_qubits, "X"),
        ):
            with self.subTest(prefix=prefix):
                circuit = final_data_measurement_circuit(
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

    def test_repetition_code_logical_error_rate_matches_stim_raw_samples(self) -> None:
        shots = 15_000
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(1, 2): 0.17, (2, 0): 0.09},
            measurement_error_rate=0.035,
        )
        try:
            faultscope_result = compile_native_sampler(
                experiment.circuit,
                observables=experiment.observables,
            ).estimate(
                shots=shots,
                seed=12345,
                decoder=experiment.decoder,
            )
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native forward sampler unavailable: {exc}")

        observable = LogicalObservable(id=0, measurement_keys=("final_d_0",))
        declared_circuit = with_dem_declarations(
            final_data_measurement_circuit(
                experiment.circuit,
                (experiment.data_qubits[0],),
                basis="Z",
                prefix="final_d",
            ),
            detectors=experiment.detectors,
            observables=(observable,),
        )
        stim_batch, stim_observables = _sample_stim_measurements_and_observables(
            declared_circuit,
            observables=(observable,),
            shots=shots,
            seed=67890,
        )
        stim_corrections = experiment.decoder.decode_batch_masks(stim_batch)
        stim_loss_rate = _residual_rate(
            stim_observables[0],
            stim_corrections.get(0, 0),
            stim_batch.shots,
        )

        _assert_rates_close(
            self,
            faultscope_result.mean_loss,
            stim_loss_rate,
            shots,
            "repetition raw-sample decoded logical error rate",
        )

    def test_surface_code_initialized_memory_logical_error_rates_match_stim_raw_samples(self) -> None:
        shots = 20_000
        for basis in ("x", "z"):
            with self.subTest(basis=basis):
                stim_circuit = stim.Circuit.generated(
                    code_task=f"surface_code:rotated_memory_{basis}",
                    distance=3,
                    rounds=3,
                    after_clifford_depolarization=0.01,
                )
                imported = parse_stim_circuit(str(stim_circuit.flattened()))
                try:
                    stim_dem = stim_circuit.detector_error_model(
                        decompose_errors=True,
                        flatten_loops=True,
                    )
                    pymatching = _load_pymatching_or_skip(self)
                    matcher = pymatching.Matching.from_detector_error_model(stim_dem)
                    decoder = _StimMatcherBatchDecoder(
                        matcher=matcher,
                        detector_ids=tuple(
                            detector.id
                            for detector in imported.detectors
                        ),
                        observable_ids=tuple(
                            observable.id
                            for observable in imported.observables
                        ),
                    )
                    faultscope_result = compile_native_sampler(
                        imported.circuit,
                    ).estimate(
                        shots=shots,
                        seed=22345,
                        decoder=decoder,
                    )
                except UnsupportedNativeCircuitError as exc:
                    self.skipTest(f"native forward sampler unavailable: {exc}")
                except (PyMatchingUnavailableError, UnsupportedPyMatchingDemError) as exc:
                    self.fail(f"PyMatching DEM decoder unavailable: {exc}")

                stim_detectors, stim_observables = stim_circuit.compile_detector_sampler(
                    seed=77890,
                ).sample(
                    shots,
                    separate_observables=True,
                )
                stim_batch = dem_batch_from_stim_samples(
                    stim_detectors,
                    stim_observables,
                    detectors=imported.detectors,
                    observables=imported.observables,
                )
                stim_loss_rate = _dem_decoded_loss_rate(
                    stim_batch,
                    decoder,
                    observable_id=0,
                )

                _assert_rates_close(
                    self,
                    faultscope_result.mean_loss,
                    stim_loss_rate,
                    shots,
                    f"surface initialized memory raw logical error rate {basis}",
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
            faultscope_dem = generate_native_dem(circuit)
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native DEM generator unavailable: {exc}")
        stim_circuit, _ = to_stim_circuit(circuit)
        stim_dem = stim_circuit.detector_error_model(decompose_errors=False)

        self.assertEqual(faultscope_dem_error_edges(faultscope_dem), stim_dem_error_edges(stim_dem))

    def test_native_dem_sampler_matches_stim_dem_sampler(self) -> None:
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(0, 1): 0.12, (1, 3): 0.09, (2, 0): 0.07},
            measurement_error_rate={(0, 0): 0.04, (1, 2): 0.06, (2, 3): 0.05},
        )
        circuit = with_dem_declarations(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=(),
        )
        try:
            faultscope_dem = generate_native_dem(
                experiment.circuit,
                detectors=experiment.detectors,
                observables=(),
            )
            faultscope_batch = compile_native_dem_sampler(faultscope_dem).run_batch(
                shots=45_000,
                seed=24680,
            )
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native DEM mode unavailable: {exc}")

        stim_circuit, _ = to_stim_circuit(circuit)
        stim_dem = stim_circuit.detector_error_model(decompose_errors=False)
        stim_detectors, stim_observables, _ = stim_dem.compile_sampler(seed=13579).sample(
            faultscope_batch.shots,
        )

        for detector in experiment.detectors:
            detector_id = detector.id
            faultscope_rate = faultscope_batch.detectors[detector_id].bit_count() / faultscope_batch.shots
            stim_rate = float(stim_detectors[:, detector_id].mean())
            _assert_rates_close(
                self,
                faultscope_rate,
                stim_rate,
                faultscope_batch.shots,
                f"DEM detector D{detector_id}",
            )

        self.assertEqual(stim_observables.shape[1], 0)

    def test_repetition_code_dem_logical_error_rate_matches_stim_dem_sampler(self) -> None:
        shots = 18_000
        experiment = make_repetition_code_experiment(
            distance=5,
            rounds=3,
            data_error_rate={(0, 1): 0.12, (1, 3): 0.09, (2, 0): 0.07},
            measurement_error_rate={(0, 0): 0.04, (1, 2): 0.06, (2, 3): 0.05},
        )
        observable = LogicalObservable(id=0, measurement_keys=("final_d_0",))
        circuit = final_data_measurement_circuit(
            experiment.circuit,
            (experiment.data_qubits[0],),
            basis="Z",
            prefix="final_d",
        )

        try:
            faultscope_dem = generate_native_dem(
                circuit,
                detectors=experiment.detectors,
                observables=(observable,),
            )
            decoder = PyMatchingDecoder.from_dem(faultscope_dem)
            faultscope_batch = compile_native_dem_sampler(faultscope_dem).run_batch(
                shots=shots,
                seed=24680,
            )
        except UnsupportedNativeCircuitError as exc:
            self.skipTest(f"native DEM mode unavailable: {exc}")
        except (PyMatchingUnavailableError, UnsupportedPyMatchingDemError) as exc:
            self.fail(f"PyMatching DEM decoder unavailable: {exc}")

        stim_circuit, _ = to_stim_circuit(
            with_dem_declarations(
                circuit,
                detectors=experiment.detectors,
                observables=(observable,),
            )
        )
        stim_dem = stim_circuit.detector_error_model(decompose_errors=False)
        stim_detectors, stim_observables, _ = stim_dem.compile_sampler(seed=13579).sample(
            shots,
        )
        stim_batch = dem_batch_from_stim_samples(
            stim_detectors,
            stim_observables,
            detectors=experiment.detectors,
            observables=(observable,),
        )

        _assert_rates_close(
            self,
            _dem_decoded_loss_rate(faultscope_batch, decoder, observable_id=0),
            _dem_decoded_loss_rate(stim_batch, decoder, observable_id=0),
            shots,
            "repetition DEM decoded logical error rate",
        )

    def test_surface_code_dem_logical_error_rates_match_stim_dem_sampler(self) -> None:
        shots = 12_000
        for label, circuit, detectors, observables in _deterministic_surface_memory_cases():
            with self.subTest(memory=label):
                try:
                    faultscope_dem = generate_native_dem(
                        circuit,
                        detectors=detectors,
                        observables=observables,
                    )
                    decoder = PyMatchingDecoder.from_dem(faultscope_dem)
                    faultscope_batch = compile_native_dem_sampler(faultscope_dem).run_batch(
                        shots=shots,
                        seed=34680,
                    )
                except UnsupportedNativeCircuitError as exc:
                    self.skipTest(f"native DEM mode unavailable: {exc}")
                except (PyMatchingUnavailableError, UnsupportedPyMatchingDemError) as exc:
                    self.fail(f"PyMatching DEM decoder unavailable: {exc}")

                stim_circuit, _ = to_stim_circuit(
                    with_dem_declarations(
                        circuit,
                        detectors=detectors,
                        observables=observables,
                    )
                )
                stim_dem = stim_circuit.detector_error_model(decompose_errors=False)

                stim_detectors, stim_observables, _ = stim_dem.compile_sampler(
                    seed=23579,
                ).sample(shots)
                stim_batch = dem_batch_from_stim_samples(
                    stim_detectors,
                    stim_observables,
                    detectors=detectors,
                    observables=observables,
                )

                _assert_rates_close(
                    self,
                    _dem_decoded_loss_rate(faultscope_batch, decoder, observable_id=0),
                    _dem_decoded_loss_rate(stim_batch, decoder, observable_id=0),
                    shots,
                    f"surface DEM decoded logical error rate {label}",
                )


def _sample_both(
    circuit: Circuit,
    *,
    shots: int,
    native_seed: int = 12345,
    stim_seed: int = 67890,
) -> tuple[SampleBatch, object, tuple[str, ...]]:
    try:
        sampler = compile_native_sampler(circuit)
    except UnsupportedNativeCircuitError as exc:
        raise unittest.SkipTest(f"native forward sampler unavailable: {exc}") from exc
    batch = sampler.sample(shots=shots, seed=native_seed)
    stim_circuit, key_order = to_stim_circuit(circuit)
    samples = stim_circuit.compile_sampler(seed=stim_seed).sample(shots)
    return batch, samples, key_order


class _StimMatcherBatchDecoder:
    def __init__(
        self,
        *,
        matcher: Any,
        detector_ids: Sequence[int],
        observable_ids: Sequence[int],
    ) -> None:
        self.matcher = matcher
        self.detector_ids = tuple(int(detector_id) for detector_id in detector_ids)
        self.observable_ids = tuple(
            int(observable_id)
            for observable_id in observable_ids
        )

    def decode_batch_masks(self, batch: SampleBatch) -> dict[int, int]:
        syndromes = masks_to_dense_array(
            batch.detectors,
            self.detector_ids,
            batch.shots,
        )
        predictions = self.matcher.decode_batch(syndromes)
        return dense_predictions_to_masks(
            predictions,
            self.observable_ids,
            batch.shots,
        )


def _load_pymatching_or_skip(testcase: unittest.TestCase) -> Any:
    try:
        import pymatching
    except ImportError as exc:
        testcase.fail(f"PyMatching is required for core sampling comparison tests: {exc}")
    return pymatching


def _sample_stim_measurements_and_observables(
    circuit: Circuit,
    *,
    observables: Sequence[LogicalObservable],
    shots: int,
    seed: int,
) -> tuple[SampleBatch, dict[int, int]]:
    stim_circuit, key_order = to_stim_circuit(circuit)
    samples = stim_circuit.compile_sampler(seed=seed).sample(shots)
    _, observable_flips = stim_circuit.compile_m2d_converter().convert(
        measurements=samples,
        separate_observables=True,
    )
    return (
        measurement_batch_from_stim_samples(samples, key_order),
        stim_observable_masks(observable_flips, observables),
    )


def _residual_rate(observable_mask: int, correction_mask: int, shots: int) -> float:
    all_mask = (1 << shots) - 1
    return (
        logical_residual_loss_mask(
            {0: observable_mask},
            {0: correction_mask},
            observable_ids=(0,),
            all_mask=all_mask,
        ).bit_count()
        / shots
    )


def _dem_decoded_loss_rate(
    batch: SampleBatch,
    decoder: PyMatchingDecoder,
    *,
    observable_id: int,
) -> float:
    corrections = decoder.decode_batch_masks(batch)
    loss_mask = logical_residual_loss_mask(
        batch.observables,
        corrections,
        observable_ids=(int(observable_id),),
        all_mask=batch.all_mask,
    )
    return loss_mask.bit_count() / batch.shots


def _deterministic_surface_memory_cases(
) -> tuple[tuple[str, Circuit, tuple[Detector, ...], tuple[LogicalObservable, ...]], ...]:
    return (
        _deterministic_surface_memory_case("z_memory"),
        _deterministic_surface_memory_case("x_memory"),
    )


def _deterministic_surface_memory_case(
    memory: str,
) -> tuple[str, Circuit, tuple[Detector, ...], tuple[LogicalObservable, ...]]:
    distance = 5
    rounds = 2
    x_checks, z_checks = _rotated_surface_code_checks(distance)
    operations: list[Operation] = []
    if memory == "z_memory":
        checks = z_checks
        basis = "Z"
        data_error = "X"
        logical_qubits = tuple(
            _data_index(distance, row, 0)
            for row in range(distance)
        )
    elif memory == "x_memory":
        checks = x_checks
        basis = "X"
        data_error = "Z"
        logical_qubits = tuple(
            _data_index(distance, 0, col)
            for col in range(distance)
        )
        for qubit in range(distance * distance):
            operations.append(Operation.h(qubit))
    else:
        raise ValueError(f"unknown surface memory case {memory!r}")

    _append_surface_memory_checks(
        operations,
        distance=distance,
        round_idx=0,
        checks=checks,
        basis=basis,
        noise=False,
    )
    for round_idx in range(1, rounds + 1):
        for qubit in range(distance * distance):
            operations.append(
                Operation.noise(
                    NoiseLocation(
                        id=f"{memory}_data_r{round_idx}_q{qubit}",
                        model=BernoulliPauliNoise(data_error),
                        rate=0.04,
                        qubits=(qubit,),
                    )
                )
            )
        _append_surface_memory_checks(
            operations,
            distance=distance,
            round_idx=round_idx,
            checks=checks,
            basis=basis,
            noise=True,
        )
    for idx, qubit in enumerate(logical_qubits):
        operations.append(Operation.measure(qubit, key=f"logical_{idx}", basis=basis))

    detectors = tuple(
        Detector(
            id=idx,
            measurement_keys=(f"r{rounds}_{check['id']}", f"r0_{check['id']}"),
            coords=(float(check["x"]), float(check["y"])),
        )
        for idx, check in enumerate(checks)
    )
    observables = (
        LogicalObservable(
            id=0,
            measurement_keys=_path_keys("logical", len(logical_qubits)),
        ),
    )
    return (
        memory,
        Circuit(n_qubits=distance * distance, operations=operations),
        detectors,
        observables,
    )


def _append_surface_memory_checks(
    operations: list[Operation],
    *,
    distance: int,
    round_idx: int,
    checks: Sequence[dict[str, object]],
    basis: str,
    noise: bool,
) -> None:
    for check in checks:
        check_id = str(check["id"])
        qubits = tuple(
            _data_index(distance, row, col)
            for row, col in check["data"]
        )
        location = None
        if noise:
            location = NoiseLocation(
                id=f"surface_{basis.lower()}_meas_r{round_idx}_{check_id}",
                model=MeasurementBitFlip(),
                rate=0.02,
                qubits=qubits[:1],
            )
        operations.append(
            Operation.measure_pauli(
                qubits,
                basis * len(qubits),
                key=f"r{round_idx}_{check_id}",
                noise=location,
            )
        )


def _path_keys(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{idx}" for idx in range(count))


def _assert_key_rate_close(
    testcase: unittest.TestCase,
    batch: SampleBatch,
    stim_samples: object,
    key_order: Sequence[str],
    key: str,
) -> None:
    key_to_col = {name: col for col, name in enumerate(key_order)}
    faultscope_rate = batch.measurements[key].bit_count() / batch.shots
    stim_rate = float(stim_samples[:, key_to_col[key]].mean())
    _assert_rates_close(testcase, faultscope_rate, stim_rate, batch.shots, f"measurement {key}")


def _assert_parity_rate_close(
    testcase: unittest.TestCase,
    batch: SampleBatch,
    stim_samples: object,
    key_order: Sequence[str],
    keys: Sequence[str],
) -> None:
    faultscope_mask = 0
    for key in keys:
        faultscope_mask ^= batch.measurements[key]
    faultscope_rate = faultscope_mask.bit_count() / batch.shots
    stim_rate = _stim_parity_rate(stim_samples, key_order, keys)
    _assert_rates_close(testcase, faultscope_rate, stim_rate, batch.shots, f"parity {tuple(keys)}")


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
    faultscope_rate: float,
    stim_rate: float,
    shots: int,
    label: str,
) -> None:
    pooled = 0.5 * (faultscope_rate + stim_rate)
    sigma = (2.0 * pooled * (1.0 - pooled) / shots) ** 0.5
    tolerance = max(0.003, 3.0 * sigma, 3.0 / shots)
    testcase.assertLessEqual(
        abs(faultscope_rate - stim_rate),
        tolerance,
        f"{label}: FaultScope={faultscope_rate:.6g}, Stim={stim_rate:.6g}, tolerance={tolerance:.6g}",
    )


def _repetition_final_data_loss_mask(
    batch: SampleBatch,
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
