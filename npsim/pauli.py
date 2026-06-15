"""Pauli helpers and a forward Pauli-frame tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


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


@dataclass
class PauliFrame:
    """Forward Pauli-frame tracker for sampled stochastic errors."""

    x: list[int]
    z: list[int]

    @classmethod
    def zero(cls, n_qubits: int) -> "PauliFrame":
        return cls([0] * n_qubits, [0] * n_qubits)

    @property
    def n_qubits(self) -> int:
        return len(self.x)

    def copy(self) -> "PauliFrame":
        return PauliFrame(self.x.copy(), self.z.copy())

    def apply_pauli(self, qubit: int, pauli: str) -> None:
        px, pz = pauli_to_xz(pauli)
        self.x[qubit] ^= px
        self.z[qubit] ^= pz

    def apply_pauli_string(self, qubits: Sequence[int], paulis: str | Sequence[str]) -> None:
        if isinstance(paulis, str):
            pauli_chars = list(paulis)
        else:
            pauli_chars = list(paulis)
        if len(qubits) != len(pauli_chars):
            raise ValueError("qubits and paulis must have the same length")
        for qubit, pauli in zip(qubits, pauli_chars):
            self.apply_pauli(qubit, pauli)

    def apply_h(self, qubit: int) -> None:
        self.x[qubit], self.z[qubit] = self.z[qubit], self.x[qubit]

    def apply_s(self, qubit: int) -> None:
        self.z[qubit] ^= self.x[qubit]

    def apply_s_dag(self, qubit: int) -> None:
        self.apply_s(qubit)

    def apply_cx(self, control: int, target: int) -> None:
        self.x[target] ^= self.x[control]
        self.z[control] ^= self.z[target]

    def apply_cz(self, left: int, right: int) -> None:
        self.apply_h(right)
        self.apply_cx(left, right)
        self.apply_h(right)

    def apply_swap(self, left: int, right: int) -> None:
        if left == right:
            return
        self.apply_cx(left, right)
        self.apply_cx(right, left)
        self.apply_cx(left, right)

    def reset(self, qubit: int) -> None:
        self.x[qubit] = 0
        self.z[qubit] = 0

    def measurement_flip(self, qubits: Sequence[int], paulis: str | Sequence[str]) -> int:
        px, pz = sparse_pauli_to_xz(self.n_qubits, qubits, paulis)
        return symplectic_product(self.x, self.z, px, pz)

    def pauli_on(self, qubits: Iterable[int]) -> str:
        return "".join(xz_to_pauli(self.x[q], self.z[q]) for q in qubits)
