#!/usr/bin/env python3
"""Smoke-test an installed FaultScope wheel outside the source tree."""

from __future__ import annotations

import argparse

import faultscope
from faultscope import Detector, DetectorErrorEdge, DetectorErrorModel, LogicalObservable
from faultscope.collection import (
    CollectionOptions,
    CollectionRunOptions,
    CollectionTask,
    collect,
)
from faultscope.decoders import available_native_decoders, create_native_decoder
from faultscope.runtime import compile_native_dem_sampler


def _logical_edge_dem() -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=1.0,
                detectors=(),
                observables=(0,),
                location_id="logical",
                event="L",
            ),
        ),
    )


def _pymatching_dem() -> DetectorErrorModel:
    return DetectorErrorModel(
        detectors=(Detector(id=0, measurement_keys=()),),
        observables=(LogicalObservable(id=0),),
        edges=(
            DetectorErrorEdge(
                probability=0.2,
                detectors=(0,),
                observables=(0,),
                location_id="edge0",
                event="X",
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pymatching", action="store_true")
    args = parser.parse_args()

    stats = collect(
        [CollectionTask(dem=_logical_edge_dem(), task_id="wheel-smoke")],
        options=CollectionOptions(max_shots=8),
        run_options=CollectionRunOptions(seed=1),
    )[0]
    assert stats.shots == 8
    assert stats.errors == 8

    if args.pymatching:
        assert "pymatching" in available_native_decoders()
        dem = _pymatching_dem()
        decoder = create_native_decoder("pymatching", dem=dem)
        result = compile_native_dem_sampler(dem).estimate(
            shots=1024,
            seed=11,
            decoder=decoder,
            aggregate_hotspots=False,
        )
        assert result.mean_loss == 0.0
        assert decoder.python_decode_call_count == 0

    print(f"faultscope {faultscope.__version__}: wheel smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
