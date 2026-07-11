#!/usr/bin/env python3
"""Generate the structural type stub for the PyO3 extension module."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any
from typing_extensions import Self

import faultscope._native as native


HEADER = """\
from typing import Any, final
from typing_extensions import Self

"""


def _typed_signature(signature: inspect.Signature, *, returns: object = Any) -> str:
    parameters = [
        parameter if parameter.name in {"self", "cls"} else parameter.replace(annotation=Any)
        for parameter in signature.parameters.values()
    ]
    rendered = str(signature.replace(parameters=parameters, return_annotation=returns))
    return rendered.replace("typing.Any", "Any")


def _constructor_signature(cls: type[object]) -> str | None:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return None
    cls_parameter = inspect.Parameter(
        "cls",
        inspect.Parameter.POSITIONAL_ONLY,
    )
    parameters = [cls_parameter, *signature.parameters.values()]
    return _typed_signature(
        signature.replace(parameters=parameters),
        returns=Self,
    )


def _render_class(name: str, cls: type[object]) -> list[str]:
    lines = ["@final", f"class {name}:"]
    body: list[str] = []
    constructor = _constructor_signature(cls)
    if constructor is not None:
        body.append(f"    def __new__{constructor}: ...")

    for member_name in sorted(item for item in dir(cls) if not item.startswith("_")):
        raw_member = inspect.getattr_static(cls, member_name)
        if inspect.isgetsetdescriptor(raw_member):
            body.extend(
                [
                    "    @property",
                    f"    def {member_name}(self) -> Any: ...",
                ]
            )
            continue
        member = getattr(cls, member_name)
        if not callable(member):
            continue
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            continue
        parameters = tuple(signature.parameters.values())
        is_static = not parameters or parameters[0].name not in {"self", "cls"}
        if is_static:
            body.append("    @staticmethod")
        body.append(f"    def {member_name}{_typed_signature(signature)}: ...")

    lines.extend(body or ["    ..."])
    lines.append("")
    return lines


def generate_stub() -> str:
    lines = [HEADER.rstrip(), ""]
    names = sorted(
        name for name in dir(native) if not name.startswith("__") or name == "__version__"
    )
    for name in names:
        value = getattr(native, name)
        if inspect.isclass(value):
            lines.extend(_render_class(name, value))
        elif callable(value):
            try:
                signature = inspect.signature(value)
            except (TypeError, ValueError):
                continue
            lines.append(f"def {name}{_typed_signature(signature)}: ...")
        elif isinstance(value, bool):
            lines.append(f"{name}: bool")
        elif isinstance(value, int):
            lines.append(f"{name}: int")
        elif isinstance(value, str):
            lines.append(f"{name}: str")
        else:
            lines.append(f"{name}: Any")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "faultscope" / "_native.pyi",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = generate_stub()
    if args.check:
        if not args.output.exists() or args.output.read_text() != generated:
            parser.error(f"{args.output} is stale; run {Path(__file__).name}")
        return 0
    args.output.write_text(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
