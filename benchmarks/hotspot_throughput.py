"""Hotspot identification throughput benchmark.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/hotspot_throughput.py --distances 9 13 21

The benchmark separates sampling, Python loss-callback time, and hotspot
aggregation time.  The target ratio applies to aggregation, which is the stage
implemented in Rust for arbitrary Python loss functions.  Python comparisons
use test-only reference implementations, not runtime fallbacks.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim.dem import DetectorErrorModelGenerator
from npsim.dem.sampler import _default_loss_mask
from npsim.runtime import (
    UnsupportedNativeCircuitError,
    compile_native_dem_sampler,
    compile_native_sampler,
)
from npsim.experiments import make_repetition_code_experiment
from tests.reference.batch import (
    BatchForwardNoiseAwareSimulator as ReferenceBatchForwardNoiseAwareSimulator,
)
from tests.reference.dem_sampler import (
    DemBatchHotspotSimulator as ReferenceDemBatchHotspotSimulator,
    _aggregate_by_tag,
    _aggregate_detector_hotspots,
    _edge_sensitivities_to_detector_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=int, default=[9, 13, 21])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print(
        "case\tdistance\trounds\tshots\tlocations\tedges\t"
        "native_batch_full_sps\treference_batch_full_sps\tbatch_full_ratio\t"
        "native_batch_agg_sps\treference_batch_agg_sps\tbatch_agg_ratio\t"
        "native_dem_full_sps\treference_dem_full_sps\tdem_full_ratio\t"
        "native_dem_agg_sps\treference_dem_agg_sps\tdem_agg_ratio\tstatus",
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
                backend="native",
            )
            reference_batch_engine = ReferenceBatchForwardNoiseAwareSimulator(
                experiment.circuit,
            )
            native_dem_sampler = compile_native_dem_sampler(dem, backend="native")
            reference_dem_engine = ReferenceDemBatchHotspotSimulator(dem)

            native_batch_full_sps = _median_samples_per_second(
                lambda seed: native_batch_sampler.estimate(
                    shots=args.shots,
                    seed=seed,
                    loss_mask_fn=experiment.batch_loss_mask_fn,
                    top_k=args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )
            reference_batch_full_sps = _median_samples_per_second(
                lambda seed: _reference_batch_estimate(
                    reference_batch_engine,
                    shots=args.shots,
                    seed=seed,
                    loss_mask_fn=experiment.batch_loss_mask_fn,
                    top_k=args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )

            native_batch = native_batch_sampler._engine.run_native_batch(args.shots, 777)
            native_loss = int(experiment.batch_loss_mask_fn(native_batch)) & int(
                native_batch.all_mask
            )
            reference_batch = reference_batch_engine.run_batch(
                shots=args.shots,
                rng=random.Random(777),
            )
            reference_loss = (
                experiment.batch_loss_mask_fn(reference_batch)
                & reference_batch.all_mask
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
            reference_batch_agg_sps = _median_samples_per_second(
                lambda seed: _reference_batch_aggregate(
                    reference_batch_engine,
                    reference_batch,
                    reference_loss,
                    top_k=args.top_k,
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
            reference_dem_full_sps = _median_samples_per_second(
                lambda seed: reference_dem_engine.estimate(
                    shots=args.shots,
                    seed=seed,
                    baseline="mean",
                ),
                shots=args.shots,
                repeats=args.repeats,
            )

            native_dem_batch = native_dem_sampler._engine.run_native_batch(args.shots, 888)
            native_dem_loss = _default_loss_mask(native_dem_batch, {}, dem) & int(
                native_dem_batch.all_mask
            )
            reference_dem_batch = reference_dem_engine.run_batch(
                shots=args.shots,
                rng=random.Random(888),
            )
            reference_dem_loss = (
                _default_loss_mask(reference_dem_batch, {}, dem)
                & reference_dem_batch.all_mask
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
            reference_dem_agg_sps = _median_samples_per_second(
                lambda seed: _reference_dem_aggregate(
                    reference_dem_engine,
                    reference_dem_batch,
                    reference_dem_loss,
                    top_k=args.top_k,
                ),
                shots=args.shots,
                repeats=args.repeats,
            )

            batch_full_ratio = native_batch_full_sps / reference_batch_full_sps
            batch_agg_ratio = native_batch_agg_sps / reference_batch_agg_sps
            dem_full_ratio = native_dem_full_sps / reference_dem_full_sps
            dem_agg_ratio = native_dem_agg_sps / reference_dem_agg_sps
            status = (
                "pass"
                if batch_agg_ratio >= 5.0 and dem_agg_ratio >= 5.0
                else "below-target"
            )
            print(
                f"repetition-d{distance}\t{distance}\t{args.rounds}\t{args.shots}\t"
                f"{len(experiment.circuit.noise_locations())}\t{len(dem.edges)}\t"
                f"{native_batch_full_sps:.3f}\t{reference_batch_full_sps:.3f}\t{batch_full_ratio:.3f}\t"
                f"{native_batch_agg_sps:.3f}\t{reference_batch_agg_sps:.3f}\t{batch_agg_ratio:.3f}\t"
                f"{native_dem_full_sps:.3f}\t{reference_dem_full_sps:.3f}\t{dem_full_ratio:.3f}\t"
                f"{native_dem_agg_sps:.3f}\t{reference_dem_agg_sps:.3f}\t{dem_agg_ratio:.3f}\t{status}",
                flush=True,
            )
        except UnsupportedNativeCircuitError as exc:
            print(
                f"repetition-d{distance}\t{distance}\t{args.rounds}\t{args.shots}\t"
                f"{len(experiment.circuit.noise_locations())}\t{len(dem.edges)}\t"
                f"NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t"
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


def _reference_batch_aggregate(
    engine: ReferenceBatchForwardNoiseAwareSimulator,
    batch: Any,
    loss_mask: int,
    *,
    top_k: int,
) -> Any:
    loss_mask &= batch.all_mask
    loss_count = loss_mask.bit_count()
    mean_loss = loss_count / batch.shots
    sensitivities: dict[str, float] = {}
    for location_id, location in engine.locations.items():
        event_mask = batch.noise_event_masks.get(location_id, 0) & batch.all_mask
        event_count = event_mask.bit_count()
        loss_event_count = (loss_mask & event_mask).bit_count()
        loss_no_event_count = loss_count - loss_event_count
        no_event_count = batch.shots - event_count
        event_score, no_event_score = _score_pair(location.rate)
        sum_loss_score = (
            loss_event_count * event_score + loss_no_event_count * no_event_score
        )
        sum_score = event_count * event_score + no_event_count * no_event_score
        sensitivities[location_id] = (sum_loss_score - mean_loss * sum_score) / batch.shots
    hotspots = {
        location_id: abs(sensitivity)
        for location_id, sensitivity in sensitivities.items()
    }
    by_qubit: dict[int, float] = {}
    by_round: dict[Any, float] = {}
    by_gate: dict[Any, float] = {}
    by_operation: dict[Any, float] = {}
    for location_id, hotspot in hotspots.items():
        location = engine.locations[location_id]
        for qubit in location.qubits:
            by_qubit[qubit] = by_qubit.get(qubit, 0.0) + hotspot
        for target, tag in (
            (by_round, "round"),
            (by_gate, "gate"),
            (by_operation, "operation"),
        ):
            value = location.tags.get(tag)
            if value is not None:
                target[value] = target.get(value, 0.0) + hotspot
    top_hotspots = sorted(hotspots.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return sensitivities, hotspots, by_qubit, by_round, by_gate, by_operation, top_hotspots


def _reference_dem_aggregate(
    engine: ReferenceDemBatchHotspotSimulator,
    batch: Any,
    loss_mask: int,
    *,
    top_k: int,
) -> Any:
    loss_mask &= batch.all_mask
    loss_count = loss_mask.bit_count()
    mean_loss = loss_count / batch.shots
    edge_sensitivities = engine._estimate_edge_sensitivities(
        batch,
        loss_mask,
        loss_count,
        mean_loss,
    )
    edge_hotspots = {
        edge_index: abs(sensitivity)
        for edge_index, sensitivity in edge_sensitivities.items()
    }
    sensitivities = engine._aggregate_location_sensitivities(edge_sensitivities)
    hotspots = {
        location_id: abs(sensitivity)
        for location_id, sensitivity in sensitivities.items()
    }
    locations = engine._location_metadata()
    top_edges = sorted(edge_hotspots.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_hotspots = sorted(hotspots.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return (
        edge_sensitivities,
        edge_hotspots,
        sensitivities,
        hotspots,
        _aggregate_detector_hotspots(engine.dem, edge_hotspots),
        _aggregate_by_tag(hotspots, locations, "round"),
        _aggregate_by_tag(hotspots, locations, "gate"),
        _aggregate_by_tag(hotspots, locations, "operation"),
        _edge_sensitivities_to_detector_graph(engine.dem, edge_sensitivities),
        top_edges,
        top_hotspots,
    )


def _score_pair(probability: float) -> tuple[float, float]:
    p = min(1.0 - 1e-12, max(1e-12, probability))
    return 1.0 / p, -1.0 / (1.0 - p)


def _reference_batch_estimate(
    engine: ReferenceBatchForwardNoiseAwareSimulator,
    *,
    shots: int,
    seed: int | None,
    loss_mask_fn: Any,
    top_k: int,
) -> Any:
    rng = random.Random(seed)
    batch = engine.run_batch(shots=shots, rng=rng)
    loss_mask = loss_mask_fn(batch) & batch.all_mask
    return _reference_batch_aggregate(engine, batch, loss_mask, top_k=top_k)


if __name__ == "__main__":
    main()
