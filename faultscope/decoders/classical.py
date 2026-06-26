"""Simple decoder interfaces and reference decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NoCorrectionDecoder:
    """Decoder that returns no correction."""

    def decode(
        self,
        detector_record: Any,
        measurements: Mapping[str, Any],
        trajectory: Any,
    ) -> Any:
        del detector_record, measurements, trajectory
        return None


@dataclass(frozen=True)
class RepetitionCodeDecoder:
    """Open-boundary minimum-weight decoder for a final repetition syndrome."""

    distance: int
    measurement_keys: tuple[str, ...] = ()
    observable_id: int = 0

    def decode(
        self,
        detector_record: Any,
        measurements: Mapping[str, Any],
        trajectory: Any,
    ) -> list[int]:
        del measurements, trajectory
        syndrome = self._coerce_syndrome(detector_record)
        if len(syndrome) != self.distance - 1:
            raise ValueError(
                f"expected {self.distance - 1} syndrome bits, got {len(syndrome)}"
            )

        candidate = [0] * self.distance
        for idx, bit in enumerate(syndrome):
            candidate[idx + 1] = candidate[idx] ^ int(bit)

        complement = [bit ^ 1 for bit in candidate]
        return candidate if sum(candidate) <= sum(complement) else complement

    @staticmethod
    def _coerce_syndrome(detector_record: Any) -> list[int]:
        if isinstance(detector_record, Mapping):
            return [int(detector_record[key]) for key in sorted(detector_record)]
        if isinstance(detector_record, Sequence) and not isinstance(detector_record, str):
            return [int(bit) for bit in detector_record]
        raise ValueError("repetition decoder requires a sequence or mapping syndrome")

    def decode_batch_masks(self, batch: Any) -> dict[int, int]:
        if len(self.measurement_keys) != self.distance - 1:
            raise ValueError(
                "repetition batch decoder requires distance-1 measurement keys"
            )
        correction_mask = 0
        for shot in range(int(batch.shots)):
            syndrome = [
                (int(batch.measurements[key]) >> shot) & 1
                for key in self.measurement_keys
            ]
            correction = self.decode(syndrome, {}, batch)
            if correction[0]:
                correction_mask |= 1 << shot
        return {self.observable_id: correction_mask}
