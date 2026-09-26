"""Shared benchmark constants and profile definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SCHEMA_V1 = "fsbench-v1"
SCHEMA_V2 = "fsbench-v2"
SCHEMA_V3 = "fsbench-v3"
SCHEMA_VERSION = SCHEMA_V1
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_V1, SCHEMA_V2, SCHEMA_V3)
DEFAULT_DISTANCES = (2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100)
PAPER_SHOTS = (1_000_000,)
RANDOM_CLIFFORD_WIDTHS = (16, 32, 64, 128, 256, 512, 1_024)
RANDOM_CLIFFORD_INSTANCES = (0, 1, 2)
RANDOM_CLIFFORD_GENERATOR_SEED = 0xC11F_F04D_2026
PAPER_PARALLEL_WORKERS = 64
PAPER_TIMEOUT_SECONDS = 3_600.0
ALL_ENGINES = ("faultscope", "stim", "qiskit-aer", "cirq", "quantumclifford", "symft")
MASTER_SEED = 0xF417_5C0E_2026_0721

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Profile:
    name: str
    family: str
    distances: tuple[int, ...]
    widths: tuple[int, ...]
    depth_rule: Literal["none", "fixed", "width"]
    fixed_depth: int | None
    instances: tuple[int, ...]
    shots: tuple[int, ...]
    engines: tuple[str, ...]
    parallel_workers: int
    timeout_seconds: float
    publishable: bool

    def depth_for_width(self, width: int) -> int:
        """Resolve the concrete logical depth for a random-circuit width."""

        if self.depth_rule == "width":
            return int(width)
        if self.depth_rule == "fixed" and self.fixed_depth is not None:
            return int(self.fixed_depth)
        raise ValueError(f"profile {self.name} has no random-circuit depth")

    def depth_by_width(self, widths: tuple[int, ...] | None = None) -> dict[int, int]:
        selected = self.widths if widths is None else tuple(int(value) for value in widths)
        return {width: self.depth_for_width(width) for width in selected}


def selection_depth_by_width(selection: Mapping[str, Any]) -> dict[int, int]:
    """Read new per-width depths, with legacy scalar-depth compatibility."""

    widths = tuple(int(value) for value in selection.get("widths", ()))
    raw_mapping = selection.get("depth_by_width")
    if isinstance(raw_mapping, Mapping):
        depths: dict[int, int] = {}
        for width in widths:
            value = raw_mapping.get(str(width), raw_mapping.get(width))
            if value is None:
                raise ValueError(f"missing depth for width {width}")
            depths[width] = int(value)
        return depths
    if "depth" in selection:
        depth = int(selection["depth"])
        return {width: depth for width in widths}
    raise ValueError("random-Clifford selection has no depth definition")


PROFILES = {
    "surface-symft-1m-paper-linux": Profile(
        name="surface-symft-1m-paper-linux",
        family="surface_code:rotated_memory_z",
        distances=DEFAULT_DISTANCES,
        widths=(), depth_rule="none", fixed_depth=None, instances=(),
        shots=PAPER_SHOTS, engines=ALL_ENGINES,
        parallel_workers=PAPER_PARALLEL_WORKERS,
        timeout_seconds=PAPER_TIMEOUT_SECONDS, publishable=True,
    ),
    "random-clifford-depth64-symft-1m-paper-linux": Profile(
        name="random-clifford-depth64-symft-1m-paper-linux",
        family="random_clifford",
        distances=(), widths=RANDOM_CLIFFORD_WIDTHS,
        depth_rule="fixed", fixed_depth=64,
        instances=RANDOM_CLIFFORD_INSTANCES,
        shots=PAPER_SHOTS, engines=ALL_ENGINES,
        parallel_workers=PAPER_PARALLEL_WORKERS,
        timeout_seconds=PAPER_TIMEOUT_SECONDS, publishable=True,
    ),
}


ENGINE_LABELS = {
    "faultscope": "FaultScope",
    "stim": "Stim",
    "qiskit-aer": "Qiskit Aer",
    "cirq": "Cirq",
    "quantumclifford": "QuantumClifford",
    "symft": "SymFT",
}
