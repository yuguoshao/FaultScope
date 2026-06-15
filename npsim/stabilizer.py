"""Forward stabilizer-state tableau with Pauli measurements."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from npsim.pauli import (
    multiply_pauli_rows,
    sparse_pauli_to_xz,
    symplectic_product,
)


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for idx, bit in enumerate(bits):
        if bit:
            value |= 1 << idx
    return value


def _solve_row_span(rows: Sequence[int], target: int) -> int | None:
    """Return a selected-row bit mask whose xor is target, if one exists."""

    basis: dict[int, tuple[int, int]] = {}
    for row_idx, row in enumerate(rows):
        vec = row
        coeff = 1 << row_idx
        while vec:
            pivot = vec.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (vec, coeff)
                break
            basis_vec, basis_coeff = basis[pivot]
            vec ^= basis_vec
            coeff ^= basis_coeff

    vec = target
    coeff = 0
    while vec:
        pivot = vec.bit_length() - 1
        if pivot not in basis:
            return None
        basis_vec, basis_coeff = basis[pivot]
        vec ^= basis_vec
        coeff ^= basis_coeff
    return coeff


@dataclass
class StabilizerState:
    """Pure stabilizer state represented by n commuting generators."""

    x: list[list[int]]
    z: list[list[int]]
    sign: list[int]

    @classmethod
    def zero(cls, n_qubits: int) -> "StabilizerState":
        x = [[0] * n_qubits for _ in range(n_qubits)]
        z = [[0] * n_qubits for _ in range(n_qubits)]
        sign = [0] * n_qubits
        for qubit in range(n_qubits):
            z[qubit][qubit] = 1
        return cls(x, z, sign)

    @property
    def n_qubits(self) -> int:
        return len(self.x)

    def copy(self) -> "StabilizerState":
        return StabilizerState(
            [row.copy() for row in self.x],
            [row.copy() for row in self.z],
            self.sign.copy(),
        )

    def apply_h(self, qubit: int) -> None:
        for row in range(self.n_qubits):
            old_x = self.x[row][qubit]
            old_z = self.z[row][qubit]
            if old_x and old_z:
                self.sign[row] ^= 1
            self.x[row][qubit] = old_z
            self.z[row][qubit] = old_x

    def apply_s(self, qubit: int) -> None:
        for row in range(self.n_qubits):
            old_x = self.x[row][qubit]
            old_z = self.z[row][qubit]
            if old_x and old_z:
                self.sign[row] ^= 1
            self.z[row][qubit] = old_z ^ old_x

    def apply_s_dag(self, qubit: int) -> None:
        for row in range(self.n_qubits):
            old_x = self.x[row][qubit]
            old_z = self.z[row][qubit]
            if old_x and not old_z:
                self.sign[row] ^= 1
            self.z[row][qubit] = old_z ^ old_x

    def apply_cx(self, control: int, target: int) -> None:
        for row in range(self.n_qubits):
            x_c = self.x[row][control]
            z_c = self.z[row][control]
            x_t = self.x[row][target]
            z_t = self.z[row][target]
            self.sign[row] ^= x_t & z_c & (x_c ^ z_t ^ 1)
            self.x[row][target] ^= x_c
            self.z[row][control] ^= z_t

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

    def apply_pauli(self, qubit: int, pauli: str) -> None:
        px, pz = sparse_pauli_to_xz(self.n_qubits, (qubit,), pauli)
        self.apply_pauli_string(px, pz)

    def apply_pauli_string(
        self,
        x: Sequence[int],
        z: Sequence[int] | None = None,
    ) -> None:
        if z is None:
            raise ValueError("x and z vectors are required")
        for row in range(self.n_qubits):
            if symplectic_product(self.x[row], self.z[row], x, z):
                self.sign[row] ^= 1

    def measure_z(self, qubit: int, rng: random.Random) -> int:
        x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "Z")
        return self.measure_pauli(x, z, rng)

    def measure_x(self, qubit: int, rng: random.Random) -> int:
        x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "X")
        return self.measure_pauli(x, z, rng)

    def measure_y(self, qubit: int, rng: random.Random) -> int:
        x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "Y")
        return self.measure_pauli(x, z, rng)

    def measure_pauli(
        self,
        x: Sequence[int],
        z: Sequence[int],
        rng: random.Random,
    ) -> int:
        anti = [
            row
            for row in range(self.n_qubits)
            if symplectic_product(self.x[row], self.z[row], x, z)
        ]
        if not anti:
            return self._deterministic_measurement_bit(x, z)

        outcome = rng.randrange(2)
        pivot = anti[0]
        old_pivot_x = self.x[pivot].copy()
        old_pivot_z = self.z[pivot].copy()
        old_pivot_sign = self.sign[pivot]

        for row in anti[1:]:
            new_x, new_z, new_sign = multiply_pauli_rows(
                self.x[row],
                self.z[row],
                self.sign[row],
                old_pivot_x,
                old_pivot_z,
                old_pivot_sign,
            )
            self.x[row] = new_x
            self.z[row] = new_z
            self.sign[row] = new_sign

        self.x[pivot] = list(x)
        self.z[pivot] = list(z)
        self.sign[pivot] = outcome
        return outcome

    def is_deterministic_pauli(self, x: Sequence[int], z: Sequence[int]) -> bool:
        return all(
            not symplectic_product(self.x[row], self.z[row], x, z)
            for row in range(self.n_qubits)
        )

    def deterministic_measurement_bit(self, x: Sequence[int], z: Sequence[int]) -> int:
        if not self.is_deterministic_pauli(x, z):
            raise ValueError("Pauli measurement is random for this stabilizer state")
        return self._deterministic_measurement_bit(x, z)

    def reset_z(self, qubit: int, rng: random.Random) -> int:
        outcome = self.measure_z(qubit, rng)
        if outcome:
            self.apply_pauli(qubit, "X")
        return outcome

    def reset_x(self, qubit: int, rng: random.Random) -> int:
        outcome = self.measure_x(qubit, rng)
        if outcome:
            self.apply_pauli(qubit, "Z")
        return outcome

    def reset_y(self, qubit: int, rng: random.Random) -> int:
        outcome = self.measure_y(qubit, rng)
        if outcome:
            self.apply_pauli(qubit, "X")
        return outcome

    def _deterministic_measurement_bit(self, x: Sequence[int], z: Sequence[int]) -> int:
        rows = [_bits_to_int(self.x[row] + self.z[row]) for row in range(self.n_qubits)]
        target = _bits_to_int(list(x) + list(z))
        coeff = _solve_row_span(rows, target)
        if coeff is None:
            raise RuntimeError("commuting Pauli was not in the stabilizer span")

        acc_x = [0] * self.n_qubits
        acc_z = [0] * self.n_qubits
        acc_sign = 0
        selected_any = False
        for row in range(self.n_qubits):
            if not (coeff & (1 << row)):
                continue
            if not selected_any:
                acc_x = self.x[row].copy()
                acc_z = self.z[row].copy()
                acc_sign = self.sign[row]
                selected_any = True
            else:
                acc_x, acc_z, acc_sign = multiply_pauli_rows(
                    acc_x,
                    acc_z,
                    acc_sign,
                    self.x[row],
                    self.z[row],
                    self.sign[row],
                )
        if not selected_any:
            return 0
        return acc_sign


@dataclass
class BatchStabilizerState:
    """Pure stabilizer state shared by many bit-packed shots.

    The Pauli support of each generator is shared by all shots. Generator signs
    are stored as bit masks, where bit ``k`` is the sign for shot ``k``.
    """

    x: list[list[int]]
    z: list[list[int]]
    sign: list[int]
    shots: int
    all_mask: int

    @classmethod
    def zero(
        cls,
        n_qubits: int,
        shots: int,
        all_mask: int | None = None,
    ) -> "BatchStabilizerState":
        if shots <= 0:
            raise ValueError("shots must be positive")
        if all_mask is None:
            all_mask = (1 << shots) - 1
        x = [[0] * n_qubits for _ in range(n_qubits)]
        z = [[0] * n_qubits for _ in range(n_qubits)]
        sign = [0] * n_qubits
        for qubit in range(n_qubits):
            z[qubit][qubit] = 1
        return cls(x, z, sign, shots, all_mask)

    @property
    def n_qubits(self) -> int:
        return len(self.x)

    def copy(self) -> "BatchStabilizerState":
        return BatchStabilizerState(
            [row.copy() for row in self.x],
            [row.copy() for row in self.z],
            self.sign.copy(),
            self.shots,
            self.all_mask,
        )

    def apply_h(self, qubit: int) -> None:
        for row in range(self.n_qubits):
            old_x = self.x[row][qubit]
            old_z = self.z[row][qubit]
            if old_x and old_z:
                self.sign[row] ^= self.all_mask
            self.x[row][qubit] = old_z
            self.z[row][qubit] = old_x

    def apply_s(self, qubit: int) -> None:
        for row in range(self.n_qubits):
            old_x = self.x[row][qubit]
            old_z = self.z[row][qubit]
            if old_x and old_z:
                self.sign[row] ^= self.all_mask
            self.z[row][qubit] = old_z ^ old_x

    def apply_s_dag(self, qubit: int) -> None:
        for row in range(self.n_qubits):
            old_x = self.x[row][qubit]
            old_z = self.z[row][qubit]
            if old_x and not old_z:
                self.sign[row] ^= self.all_mask
            self.z[row][qubit] = old_z ^ old_x

    def apply_cx(self, control: int, target: int) -> None:
        for row in range(self.n_qubits):
            x_c = self.x[row][control]
            z_c = self.z[row][control]
            x_t = self.x[row][target]
            z_t = self.z[row][target]
            if x_t & z_c & (x_c ^ z_t ^ 1):
                self.sign[row] ^= self.all_mask
            self.x[row][target] ^= x_c
            self.z[row][control] ^= z_t

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

    def apply_pauli(self, qubit: int, pauli: str) -> None:
        px, pz = sparse_pauli_to_xz(self.n_qubits, (qubit,), pauli)
        self.apply_pauli_string(px, pz)

    def apply_pauli_string(
        self,
        x: Sequence[int],
        z: Sequence[int] | None = None,
    ) -> None:
        self.apply_pauli_string_masked(x, z, self.all_mask)

    def apply_pauli_string_masked(
        self,
        x: Sequence[int],
        z: Sequence[int] | None,
        mask: int,
    ) -> None:
        if z is None:
            raise ValueError("x and z vectors are required")
        mask &= self.all_mask
        if mask == 0:
            return
        for row in range(self.n_qubits):
            if symplectic_product(self.x[row], self.z[row], x, z):
                self.sign[row] ^= mask

    def measure_z_mask(self, qubit: int, rng: random.Random) -> int:
        x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "Z")
        return self.measure_pauli_mask(x, z, rng)

    def measure_x_mask(self, qubit: int, rng: random.Random) -> int:
        x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "X")
        return self.measure_pauli_mask(x, z, rng)

    def measure_y_mask(self, qubit: int, rng: random.Random) -> int:
        x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "Y")
        return self.measure_pauli_mask(x, z, rng)

    def measure_pauli_mask(
        self,
        x: Sequence[int],
        z: Sequence[int],
        rng: random.Random,
    ) -> int:
        anti = [
            row
            for row in range(self.n_qubits)
            if symplectic_product(self.x[row], self.z[row], x, z)
        ]
        if not anti:
            return self.deterministic_measurement_mask(x, z)

        outcome_mask = _random_bit_mask(rng, self.shots) & self.all_mask
        pivot = anti[0]
        old_pivot_x = self.x[pivot].copy()
        old_pivot_z = self.z[pivot].copy()
        old_pivot_sign = self.sign[pivot]

        for row in anti[1:]:
            new_x, new_z, new_sign = self._multiply_rows(
                self.x[row],
                self.z[row],
                self.sign[row],
                old_pivot_x,
                old_pivot_z,
                old_pivot_sign,
            )
            self.x[row] = new_x
            self.z[row] = new_z
            self.sign[row] = new_sign

        self.x[pivot] = list(x)
        self.z[pivot] = list(z)
        self.sign[pivot] = outcome_mask
        return outcome_mask

    def is_deterministic_pauli(self, x: Sequence[int], z: Sequence[int]) -> bool:
        return all(
            not symplectic_product(self.x[row], self.z[row], x, z)
            for row in range(self.n_qubits)
        )

    def deterministic_measurement_mask(
        self,
        x: Sequence[int],
        z: Sequence[int],
    ) -> int:
        if not self.is_deterministic_pauli(x, z):
            raise ValueError("Pauli measurement is random for this stabilizer state")
        return self._deterministic_measurement_mask(x, z)

    def reset_z_mask(self, qubit: int, rng: random.Random) -> int:
        outcome_mask = self.measure_z_mask(qubit, rng)
        if outcome_mask:
            x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "X")
            self.apply_pauli_string_masked(x, z, outcome_mask)
        return outcome_mask

    def reset_x_mask(self, qubit: int, rng: random.Random) -> int:
        outcome_mask = self.measure_x_mask(qubit, rng)
        if outcome_mask:
            x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "Z")
            self.apply_pauli_string_masked(x, z, outcome_mask)
        return outcome_mask

    def reset_y_mask(self, qubit: int, rng: random.Random) -> int:
        outcome_mask = self.measure_y_mask(qubit, rng)
        if outcome_mask:
            x, z = sparse_pauli_to_xz(self.n_qubits, (qubit,), "X")
            self.apply_pauli_string_masked(x, z, outcome_mask)
        return outcome_mask

    def _deterministic_measurement_mask(
        self,
        x: Sequence[int],
        z: Sequence[int],
    ) -> int:
        rows = [_bits_to_int(self.x[row] + self.z[row]) for row in range(self.n_qubits)]
        target = _bits_to_int(list(x) + list(z))
        coeff = _solve_row_span(rows, target)
        if coeff is None:
            raise RuntimeError("commuting Pauli was not in the stabilizer span")

        acc_x = [0] * self.n_qubits
        acc_z = [0] * self.n_qubits
        acc_sign = 0
        selected_any = False
        for row in range(self.n_qubits):
            if not (coeff & (1 << row)):
                continue
            if not selected_any:
                acc_x = self.x[row].copy()
                acc_z = self.z[row].copy()
                acc_sign = self.sign[row]
                selected_any = True
            else:
                acc_x, acc_z, acc_sign = self._multiply_rows(
                    acc_x,
                    acc_z,
                    acc_sign,
                    self.x[row],
                    self.z[row],
                    self.sign[row],
                )
        if not selected_any:
            return 0
        return acc_sign & self.all_mask

    def _multiply_rows(
        self,
        left_x: Sequence[int],
        left_z: Sequence[int],
        left_sign: int,
        right_x: Sequence[int],
        right_z: Sequence[int],
        right_sign: int,
    ) -> tuple[list[int], list[int], int]:
        out_x, out_z, phase_sign = multiply_pauli_rows(
            left_x,
            left_z,
            0,
            right_x,
            right_z,
            0,
        )
        out_sign = left_sign ^ right_sign
        if phase_sign:
            out_sign ^= self.all_mask
        return out_x, out_z, out_sign & self.all_mask


def _random_bit_mask(rng: random.Random, shots: int) -> int:
    mask = 0
    for shot in range(shots):
        if rng.randrange(2):
            mask |= 1 << shot
    return mask
