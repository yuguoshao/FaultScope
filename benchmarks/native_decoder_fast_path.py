"""Smoke benchmark for the native decoder fast path.

Run from the repository root after building the native extension:

    .venv/bin/python benchmarks/native_decoder_fast_path.py

The script verifies that ``estimate(..., decoder=NativeNoCorrectionDecoder(...))``
does not call the Python ``decode_batch_masks`` compatibility method.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npsim import (
    BatchForwardNoiseAwareSimulator,
    BernoulliPauliNoise,
    Circuit,
    LogicalObservable,
    NativeNoCorrectionDecoder,
    NoiseLocation,
    Operation,
)


def main() -> None:
    shots = 200_000
    circuit = Circuit(
        1,
        (
            Operation.noise(
                NoiseLocation("x0", BernoulliPauliNoise("X"), 0.05, (0,))
            ),
            Operation.measure(0, key="m0", basis="Z"),
        ),
    )
    simulator = BatchForwardNoiseAwareSimulator(
        circuit,
        observables=(LogicalObservable(0, measurement_keys=("m0",)),),
    )
    decoder = NativeNoCorrectionDecoder(observable_ids=(0,))

    started = time.perf_counter()
    result = simulator.estimate(shots=shots, seed=7, decoder=decoder)
    elapsed = time.perf_counter() - started

    status = "ok" if decoder.python_decode_call_count == 0 else "python-callback-used"
    print(
        "shots\tseconds\tsamples_per_second\tpython_decode_calls\tstatus",
        flush=True,
    )
    print(
        f"{shots}\t{elapsed:.6f}\t{shots / elapsed:.3f}\t"
        f"{decoder.python_decode_call_count}\t{status}",
        flush=True,
    )
    if status != "ok":
        raise SystemExit(status)
    if not 0.0 <= result.mean_loss <= 1.0:
        raise SystemExit("invalid mean loss")


if __name__ == "__main__":
    main()
