"""Shared loss-mask helpers for bit-packed batch results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def logical_residual_loss_mask(
    observables: Mapping[Any, int],
    corrections: Mapping[Any, int],
    *,
    observable_ids: Iterable[Any] = (),
    all_mask: int | None = None,
) -> int:
    """Return the logical residual mask from observable truth and corrections."""

    ids = set(observable_ids)
    ids.update(observables)
    ids.update(corrections)

    loss_mask = 0
    for observable_id in ids:
        if not isinstance(observable_id, int):
            raise TypeError(
                "default batch loss requires observable-id correction masks; "
                "supply loss_mask_fn for data-qubit corrections"
            )
        loss_mask |= int(observables.get(observable_id, 0)) ^ int(corrections.get(observable_id, 0))

    if all_mask is not None:
        loss_mask &= int(all_mask)
    return loss_mask
