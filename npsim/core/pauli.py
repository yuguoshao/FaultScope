"""Pauli helpers and a forward Pauli-frame tracker."""

from __future__ import annotations

from typing import Sequence

from npsim._npsim_native import PauliFrame


PAULI_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}

XZ_TO_PAULI = {value: key for key, value in PAULI_TO_XZ.items()}

_MUL_TABLE: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}


def xz_to_pauli(x: int, z: int) -> str:
    return XZ_TO_PAULI[(int(bool(x)), int(bool(z)))]


def pauli_to_xz(pauli: str) -> tuple[int, int]:
    try:
        return PAULI_TO_XZ[pauli]
    except KeyError as exc:
        raise ValueError(f"unsupported Pauli {pauli!r}") from exc


def symplectic_product(
    x1: Sequence[int],
    z1: Sequence[int],
    x2: Sequence[int],
    z2: Sequence[int],
) -> int:
    """Return 1 iff the two Pauli strings anticommute."""

    acc = 0
    for a_x, a_z, b_x, b_z in zip(x1, z1, x2, z2):
        acc ^= (int(a_x) & int(b_z)) ^ (int(a_z) & int(b_x))
    return acc & 1


def multiply_pauli_rows(
    left_x: Sequence[int],
    left_z: Sequence[int],
    left_sign: int,
    right_x: Sequence[int],
    right_z: Sequence[int],
    right_sign: int,
) -> tuple[list[int], list[int], int]:
    """Multiply two commuting Hermitian Pauli rows.

    Rows use ``sign=0`` for +P and ``sign=1`` for -P. The result is expected to
    be Hermitian with a real global sign. This is the case when multiplying
    stabilizer generators.
    """

    phase = -1 if left_sign else 1
    phase *= -1 if right_sign else 1
    out_x: list[int] = []
    out_z: list[int] = []

    for lx, lz, rx, rz in zip(left_x, left_z, right_x, right_z):
        lp = xz_to_pauli(lx, lz)
        rp = xz_to_pauli(rx, rz)
        local_phase, product = _MUL_TABLE[(lp, rp)]
        phase *= local_phase
        px, pz = pauli_to_xz(product)
        out_x.append(px)
        out_z.append(pz)

    if phase == 1:
        return out_x, out_z, 0
    if phase == -1:
        return out_x, out_z, 1
    raise ValueError("product of stabilizer rows produced a non-Hermitian phase")


def pauli_string_to_xz(pauli_string: str, n_qubits: int | None = None) -> tuple[list[int], list[int]]:
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


def sparse_pauli_to_xz(
    n_qubits: int,
    qubits: Sequence[int],
    paulis: str | Sequence[str],
) -> tuple[list[int], list[int]]:
    if isinstance(paulis, str):
        pauli_chars = list(paulis)
    else:
        pauli_chars = list(paulis)
    if len(qubits) != len(pauli_chars):
        raise ValueError("qubits and paulis must have the same length")

    xs = [0] * n_qubits
    zs = [0] * n_qubits
    for qubit, pauli in zip(qubits, pauli_chars):
        x, z = pauli_to_xz(pauli)
        xs[qubit] ^= x
        zs[qubit] ^= z
    return xs, zs
