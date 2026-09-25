"""Regenerate the fixed paper inputs and verify their archived identities."""

from __future__ import annotations

import json
from pathlib import Path

from fsbench.constants import DEFAULT_DISTANCES, RANDOM_CLIFFORD_WIDTHS
from fsbench.ir import (
    generate_random_clifford_cases,
    generate_surface_cases,
    load_case_metadata,
)

HASH_FIELDS = ("operations_sha256", "sidecar_sha256", "source_sha256")


def expected_case_hashes() -> dict[str, dict[str, str]]:
    path = Path(__file__).with_name("expected_cases.json")
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def verify_generated_cases(paths: list[Path]) -> None:
    expected = expected_case_hashes()
    for path in paths:
        case = load_case_metadata(path, verify_hash=True)
        case_id = case.metadata["case_id"]
        if case_id not in expected:
            raise ValueError(f"input is outside the fixed paper grid: {case_id}")
        for field in HASH_FIELDS:
            if case.metadata.get(field) != expected[case_id][field]:
                raise ValueError(f"{case_id}: regenerated {field} differs from the paper input")


def generate_verified_cases(
    output_dir: Path,
    *,
    random_widths: tuple[int, ...] = (),
    surface_distances: tuple[int, ...] = (),
) -> tuple[Path, Path]:
    """Generate only a caller's fixed subset; reject any protocol change."""
    if not set(random_widths) <= set(RANDOM_CLIFFORD_WIDTHS):
        raise ValueError("random widths are outside the paper grid")
    if not set(surface_distances) <= set(DEFAULT_DISTANCES):
        raise ValueError("surface distances are outside the paper grid")
    if len(set(random_widths)) != len(random_widths) or len(set(surface_distances)) != len(surface_distances):
        raise ValueError("duplicate input sizes")
    random_dir, surface_dir = output_dir / "random", output_dir / "surface"
    paths: list[Path] = []
    if random_widths:
        paths.extend(generate_random_clifford_cases(random_dir, widths=random_widths, depth=64, instances=(0, 1, 2)))
    if surface_distances:
        paths.extend(generate_surface_cases(surface_dir, distances=surface_distances, noise_probability=0.001))
    verify_generated_cases(paths)
    return random_dir, surface_dir
