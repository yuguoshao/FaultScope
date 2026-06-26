"""Command line interface for optional FaultScope native decoder backends."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from faultscope.backends.registry import (
    native_decoder_backend_statuses,
    native_decoder_install_plan,
    official_native_decoder_backend_catalog,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m faultscope.backends",
        description="Manage optional FaultScope native decoder backends.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List native decoder backend status.")
    subparsers.add_parser("status", help="List native decoder backend status.")

    install = subparsers.add_parser(
        "install",
        help="Install an optional native decoder backend.",
    )
    install.add_argument(
        "backend",
        choices=tuple(entry.name for entry in official_native_decoder_backend_catalog()),
    )
    install.add_argument("--dry-run", action="store_true")
    install.add_argument(
        "--target-dir",
        type=Path,
        default=Path.home() / ".cache" / "faultscope" / "backends",
        help="Directory used for backend source checkouts.",
    )
    install.add_argument("--rev")

    args = parser.parse_args(argv)
    if args.command in {"list", "status"}:
        return _print_statuses()
    if args.command == "install":
        return _install_backend(args.backend, args.target_dir, args.rev, args.dry_run)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _print_statuses() -> int:
    for status in native_decoder_backend_statuses():
        state = "loadable" if status.loadable else "unavailable"
        installed = "installed" if status.installed else "not-installed"
        version = f" version={status.version}" if status.version else ""
        source = f" source={status.source}" if status.source else ""
        error = f" error={status.error}" if status.error else ""
        print(f"{status.name}\t{installed}\t{state}{version}{source}{error}")
    return 0


def _install_backend(backend: str, target_dir: Path, rev: str | None, dry_run: bool) -> int:
    plan = native_decoder_install_plan(
        backend,
        target_dir=str(target_dir),
        rev=rev,
        python_executable=sys.executable,
    )
    print(f"FaultScope will install the optional {backend} native backend.")
    print(f"backend package: {plan.backend.package_name}")
    print(f"problem kind: {plan.backend.problem_kind}")
    print(f"description: {plan.backend.description}")
    if plan.backend.repo_url is not None:
        print(f"upstream source: {plan.backend.repo_url}")
    if rev or plan.backend.default_rev:
        print(f"revision: {rev or plan.backend.default_rev}")
    if plan.checkout is not None:
        print(f"checkout: {plan.checkout}")
    if plan.unavailable_reason is not None:
        print(plan.unavailable_reason)
        return 0 if dry_run else 1
    for command in plan.commands:
        print("+ " + " ".join(command))
    if dry_run:
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    checkout = Path(plan.checkout) if plan.checkout is not None else None
    if checkout is not None and checkout.exists():
        _run(["git", "-C", str(checkout), "fetch", "--all", "--tags"])
    elif checkout is not None:
        _run(list(plan.commands[0]))
    first_followup = 1 if checkout is not None else 0
    for command in plan.commands[first_followup:-1]:
        _run(list(command))
    try:
        _run(list(plan.commands[-1]))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"failed to install {plan.backend.package_name}. "
            "The post-install plugin package may not be published yet; "
            f"repo={plan.backend.repo_url} rev={rev or plan.backend.default_rev} "
            f"command={' '.join(plan.commands[-1])}"
        ) from exc
    return 0


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"required command not found while installing backend: {command[0]}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
