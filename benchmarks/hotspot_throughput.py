"""Hotspot identification throughput benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21

The benchmark separates sampling, Python loss-callback time, and hotspot
aggregation time.  The target ratio applies to aggregation, which is the stage
implemented in Rust for arbitrary Python loss functions.  Stim raw and DEM
sampling throughput are reported as external baselines.  Stim is required.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim.dem import DetectorErrorModelGenerator
from npsim.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_sampler,
)
from npsim.runtime.loss import logical_residual_loss_mask
from npsim.experiments import make_repetition_code_experiment
from tests.stim_helpers import to_stim_circuit, with_dem_declarations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[9, 13, 21])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    _load_stim()

    print(
        "case\tdistance\trounds\tshots\tlocations\tedges\t"
        "native_batch_full_sps\tnative_batch_agg_sps\t"
        "native_dem_full_sps\tnative_dem_agg_sps\t"
        "stim_raw_sps\tstim_dem_sps\tstatus",
        flush=True,
    )

    for distance in args.distances:
        experiment = make_repetition_code_experiment(
            distance=distance,
            rounds=args.rounds,
            data_error_rate=0.025,
            measurement_error_rate=0.015,
        )
        dem = DetectorErrorModelGenerator(
            experiment.circuit,
            detectors=experiment.detectors,
            observables=experiment.observables,
        ).generate()
        try:
            native_batch_sampler = compile_native_sampler(
                experiment.circuit,
                observables=experiment.observables,
            )
            native_dem_sampler = compile_native_dem_sampler(dem)
            stim_raw_circuit, _ = to_stim_circuit(experiment.circuit)
            stim_raw_sampler = stim_raw_circuit.compile_sampler()
            stim_dem_circuit, _ = to_stim_circuit(
                with_dem_declarations(
                    experiment.circuit,
                    detectors=experiment.detectors,
                    observables=(),
                )
            )
            stim_dem_sampler = stim_dem_circuit.compile_detector_sampler()

            native_batch_full_sps = _median_samples_per_second(
                lambda seed: native_batch_sampler.estimate(
                    shots=args.shots,
                    seed=seed,
                    decoder=experiment.decoder,
                    top_k=args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )

            native_batch = native_batch_sampler._engine.run_native_batch(args.shots, 777)
            native_corrections = experiment.decoder.decode_batch_masks(native_batch)
            native_loss = logical_residual_loss_mask(
                native_batch.observables,
                native_corrections,
                all_mask=native_batch.all_mask,
            )

            native_batch_agg_sps = _median_samples_per_second(
                lambda seed: native_batch_sampler._engine.estimate_hotspots(
                    native_batch,
                    native_loss,
                    None,
                    args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )

            native_dem_full_sps = _median_samples_per_second(
                lambda seed: native_dem_sampler.estimate(
                    shots=args.shots,
                    seed=seed,
                    top_k=args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )

            native_dem_batch = native_dem_sampler._engine.run_native_batch(args.shots, 888)
            native_dem_loss = logical_residual_loss_mask(
                native_dem_batch.observables,
                {},
                observable_ids=(observable.id for observable in dem.observables),
                all_mask=native_dem_batch.all_mask,
            )

            native_dem_agg_sps = _median_samples_per_second(
                lambda seed: native_dem_sampler._engine.estimate_hotspots(
                    native_dem_batch,
                    native_dem_loss,
                    None,
                    args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            stim_raw_sps = _median_samples_per_second(
                lambda seed: _sample_stim_raw(stim_raw_sampler, args.shots, seed),
                shots=args.shots,
                repeats=args.repeats,
            )
            stim_dem_sps = _median_samples_per_second(
                lambda seed: _sample_stim_detectors(stim_dem_sampler, args.shots, seed),
                shots=args.shots,
                repeats=args.repeats,
            )
            print(
                f"repetition-d{distance}\t{distance}\t{args.rounds}\t{args.shots}\t"
                f"{len(experiment.circuit.noise_locations())}\t{len(dem.edges)}\t"
                f"{native_batch_full_sps:.3f}\t{native_batch_agg_sps:.3f}\t"
                f"{native_dem_full_sps:.3f}\t{native_dem_agg_sps:.3f}\t"
                f"{stim_raw_sps:.3f}\t{stim_dem_sps:.3f}\tok",
                flush=True,
            )
        except UnsupportedNativeCircuitError as exc:
            print(
                f"repetition-d{distance}\t{distance}\t{args.rounds}\t{args.shots}\t"
                f"{len(experiment.circuit.noise_locations())}\t{len(dem.edges)}\t"
                f"NA\tNA\tNA\tNA\tNA\tNA\t"
                f"native-skip:{type(exc).__name__}",
                flush=True,
            )


def _median_samples_per_second(fn: Any, *, shots: int, repeats: int) -> float:
    values: list[float] = []
    for repeat in range(repeats):
        seed = 30_000 + repeat
        start = time.perf_counter()
        fn(seed)
        elapsed = time.perf_counter() - start
        values.append(shots / elapsed)
    return statistics.median(values)


def _sample_stim_raw(stim_sampler: Any, shots: int, seed: int) -> Any:
    del seed
    try:
        return stim_sampler.sample(shots=shots, bit_packed=True)
    except TypeError:
        return stim_sampler.sample(shots, bit_packed=True)


def _sample_stim_detectors(stim_sampler: Any, shots: int, seed: int) -> Any:
    del seed
    try:
        return stim_sampler.sample(shots=shots, bit_packed=True)
    except TypeError:
        return stim_sampler.sample(shots, bit_packed=True)


def _load_stim() -> Any:
    try:
        import stim
    except ImportError as exc:
        raise SystemExit("Stim is required for hotspot_throughput.py") from exc
    return stim


if __name__ == "__main__":
    main()
