"""Simple decoder interfaces and reference decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _packed_count_greater_than(
    masks: Sequence[int],
    *,
    threshold: int,
    all_mask: int,
) -> int:
    """Returns lanes whose population count is greater than ``threshold``."""

    count_planes: list[int] = []
    for mask in masks:
        carry = int(mask) & all_mask
        plane = 0
        while carry:
            if plane == len(count_planes):
                count_planes.append(carry)
                break
            carry, count_planes[plane] = (
                count_planes[plane] & carry,
                count_planes[plane] ^ carry,
            )
            plane += 1

    greater = 0
    equal = all_mask
    width = max(len(count_planes), threshold.bit_length())
    for plane in range(width - 1, -1, -1):
        count_bit = count_planes[plane] if plane < len(count_planes) else 0
        if (threshold >> plane) & 1:
            equal &= count_bit
        else:
            greater |= equal & count_bit
            equal &= ~count_bit
    return greater & all_mask


@dataclass(frozen=True)
class NoCorrectionDecoder:
    """Decoder that returns no correction."""

    def strong_id_payload(self) -> Mapping[str, object]:
        """Return the stable collection identity for this decoder."""

        return {
            "backend": "faultscope-python",
            "decoder": "no-correction",
            "implementation_version": 1,
            "parameters": {},
        }

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

    def strong_id_payload(self) -> Mapping[str, object]:
        """Return the stable collection identity for this decoder."""

        return {
            "backend": "faultscope-python",
            "decoder": "repetition-code",
            "implementation_version": 1,
            "parameters": {
                "distance": self.distance,
                "measurement_keys": list(self.measurement_keys),
                "observable_id": self.observable_id,
            },
        }

    def decode(
        self,
        detector_record: Any,
        measurements: Mapping[str, Any],
        trajectory: Any,
    ) -> list[int]:
        del measurements, trajectory
        syndrome = self._coerce_syndrome(detector_record)
        if len(syndrome) != self.distance - 1:
            raise ValueError(f"expected {self.distance - 1} syndrome bits, got {len(syndrome)}")

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
            raise ValueError("repetition batch decoder requires distance-1 measurement keys")
        select_measurements = getattr(batch, "measurement_masks", None)
        measurements = (
            select_measurements(self.measurement_keys)
            if callable(select_measurements)
            else batch.measurements
        )
        measurement_masks = tuple(int(measurements[key]) for key in self.measurement_keys)
        shots = int(batch.shots)
        all_mask = (1 << shots) - 1
        candidate = 0
        candidate_masks: list[int] = []
        for syndrome_mask in measurement_masks:
            candidate ^= syndrome_mask
            candidate_masks.append(candidate)
        correction_mask = _packed_count_greater_than(
            candidate_masks,
            threshold=self.distance // 2,
            all_mask=all_mask,
        )
        return {self.observable_id: correction_mask}
