"""Result containers shared by native runtime entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from npsim.core import NoiseLocation


@dataclass(frozen=True)
class HotspotRow:
    location_id: str
    sensitivity: float
    hotspot: float
    qubits: tuple[int, ...]
    tags: Mapping[str, Any]


@dataclass
class SimulationResult:
    shots: int
    mean_loss: float
    baseline: float
    sensitivities: dict[str, float]
    hotspots: dict[str, float]
    by_qubit: dict[int, float]
    by_round: dict[Any, float]
    by_gate: dict[Any, float]
    by_operation: dict[Any, float]
    locations: dict[str, NoiseLocation]
    losses: list[float]
    top_hotspots_cache: tuple[HotspotRow, ...] = ()

    @property
    def logical_failure_rate(self) -> float:
        return self.mean_loss

    def top_hotspots(self, top_k: int = 10) -> list[HotspotRow]:
        if self.top_hotspots_cache and top_k <= len(self.top_hotspots_cache):
            return list(self.top_hotspots_cache[:top_k])
        rows = [
            HotspotRow(
                location_id=location_id,
                sensitivity=self.sensitivities[location_id],
                hotspot=self.hotspots[location_id],
                qubits=self.locations[location_id].qubits,
                tags=self.locations[location_id].tags,
            )
            for location_id in self.hotspots
        ]
        rows.sort(key=lambda row: row.hotspot, reverse=True)
        return rows[:top_k]

    def hotspot_table(self, top_k: int = 10) -> str:
        lines = ["location_id\tsensitivity\thotspot\tqubits\ttags"]
        for row in self.top_hotspots(top_k):
            lines.append(
                f"{row.location_id}\t{row.sensitivity:.6g}\t"
                f"{row.hotspot:.6g}\t{row.qubits}\t{dict(row.tags)}"
            )
        return "\n".join(lines)
