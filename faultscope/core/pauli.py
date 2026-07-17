"""Pauli helpers and a forward Pauli-frame tracker."""

from __future__ import annotations

from faultscope._native import PauliFrame


PAULI_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}

XZ_TO_PAULI = {value: key for key, value in PAULI_TO_XZ.items()}


def xz_to_pauli(x: int, z: int) -> str:
    return XZ_TO_PAULI[(int(bool(x)), int(bool(z)))]


def pauli_to_xz(pauli: str) -> tuple[int, int]:
    try:
        return PAULI_TO_XZ[pauli]
    except KeyError as exc:
        raise ValueError(f"unsupported Pauli {pauli!r}") from exc


def pauli_string_to_xz(
    pauli_string: str, n_qubits: int | None = None
) -> tuple[list[int], list[int]]:
    if n_qubits is None:
        n_qubits = len(pauli_string)
    if len(pauli_string) != n_qubits:
        raise ValueError(
            f"Pauli string length {len(pauli_string)} does not match n_qubits={n_qubits}"
        )
    xs: list[int] = []
    zs: list[int] = []
    for pauli in pauli_string:
        x, z = pauli_to_xz(pauli)
        xs.append(x)
        zs.append(z)
    return xs, zs
