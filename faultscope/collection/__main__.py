"""Command line entry points for `python -m faultscope.collection`."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from faultscope.collection._collect import collect
from faultscope.collection._types import (
    CollectionTask,
    TaskStats,
    read_stats_from_csv_files,
    write_stats_to_csv_file,
)
from faultscope.collection.analysis import (
    error_rate_points,
    fit_log_error_rate_lines,
    plot_error_rates,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m faultscope.collection")
    subcommands = parser.add_subparsers(dest="command", required=True)

    collect_parser = subcommands.add_parser("collect")
    collect_parser.add_argument("--tasks-factory", required=True)
    collect_parser.add_argument("--factory-arg", action="append", default=[])
    collect_parser.add_argument("--max-shots", type=int, required=True)
    collect_parser.add_argument("--max-errors", type=int)
    collect_parser.add_argument("--batch-size", type=int, default=10_000)
    collect_parser.add_argument("--seed", type=int)
    collect_parser.add_argument("--num-workers", type=int, default=1)
    collect_parser.add_argument("--decoder", action="append", dest="decoders")
    collect_parser.add_argument("--existing-data-filepath", action="append", default=[])
    collect_parser.add_argument("--save-resume-filepath")
    collect_parser.add_argument("--out")
    collect_parser.add_argument(
        "--progress-mode", choices=("final", "stream"), default="final"
    )
    collect_parser.set_defaults(func=_cmd_collect)

    summarize_parser = subcommands.add_parser("summarize")
    summarize_parser.add_argument("stats", nargs="+")
    summarize_parser.set_defaults(func=_cmd_summarize)

    merge_parser = subcommands.add_parser("merge")
    merge_parser.add_argument("out")
    merge_parser.add_argument("inputs", nargs="+")
    merge_parser.set_defaults(func=_cmd_merge)

    plot_parser = subcommands.add_parser("plot")
    plot_parser.add_argument("stats")
    plot_parser.add_argument("--x-key", required=True)
    plot_parser.add_argument("--group-key")
    plot_parser.add_argument("--count-key")
    plot_parser.add_argument("--out", required=True)
    plot_parser.set_defaults(func=_cmd_plot)

    fit_parser = subcommands.add_parser("fit")
    fit_parser.add_argument("stats")
    fit_parser.add_argument("--x-key", required=True)
    fit_parser.add_argument("--group-key", required=True)
    fit_parser.add_argument("--count-key")
    fit_parser.set_defaults(func=_cmd_fit)
    return parser


def _cmd_collect(args: argparse.Namespace) -> int:
    factory = _load_factory(args.tasks_factory)
    tasks = factory(**_factory_kwargs(args.factory_arg))
    if isinstance(tasks, TaskStats):
        raise TypeError("tasks factory must return CollectionTask objects, not TaskStats")
    if isinstance(tasks, CollectionTask):
        tasks = [tasks]
    stats = collect(
        tasks,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        decoders=args.decoders,
        existing_data_filepaths=args.existing_data_filepath,
        save_resume_filepath=args.save_resume_filepath,
        progress_mode=args.progress_mode,
    )
    if args.out is not None and args.out != args.save_resume_filepath:
        write_stats_to_csv_file(args.out, stats)
    print(f"collected {len(stats)} task(s)", file=sys.stderr)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    stats = read_stats_from_csv_files(args.stats)
    print("strong_id\ttask_id\tdecoder\tshots\terrors\tdiscards\tlogical_error_rate\tmetadata")
    for stat in stats:
        print(
            "\t".join(
                [
                    stat.strong_id,
                    stat.task_id,
                    stat.decoder or "",
                    str(stat.shots),
                    str(stat.errors),
                    str(stat.discards),
                    f"{stat.logical_error_rate:.12g}",
                    json.dumps(dict(stat.metadata), sort_keys=True),
                ]
            )
        )
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    stats = read_stats_from_csv_files(args.inputs)
    write_stats_to_csv_file(args.out, stats)
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    stats = read_stats_from_csv_files(args.stats)
    points = error_rate_points(
        stats,
        x_key=args.x_key,
        group_key=args.group_key,
        count_key=args.count_key,
    )
    fits = fit_log_error_rate_lines(points, x_key=args.x_key, group_key=args.group_key)
    print("group\tslope\tintercept\tpoints")
    for group, fit in sorted(fits.items(), key=lambda item: str(item[0])):
        print(
            "\t".join(
                [
                    str(group),
                    f"{float(fit['slope']):.12g}",
                    f"{float(fit['intercept']):.12g}",
                    str(int(fit["points"])),
                ]
            )
        )
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    stats = read_stats_from_csv_files(args.stats)
    plot_error_rates(
        stats,
        x_key=args.x_key,
        group_key=args.group_key,
        count_key=args.count_key,
        output=args.out,
    )
    return 0


def _load_factory(spec: str) -> Any:
    module_spec, function_name = spec.rsplit(":", 1)
    if module_spec.endswith(".py") or Path(module_spec).exists():
        path = Path(module_spec).resolve()
        import_spec = importlib.util.spec_from_file_location(path.stem, path)
        if import_spec is None or import_spec.loader is None:
            raise ImportError(f"could not load tasks factory module {path}")
        module = importlib.util.module_from_spec(import_spec)
        sys.modules[path.stem] = module
        import_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    return getattr(module, function_name)


def _factory_kwargs(items: list[str]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--factory-arg values must be key=value")
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
    return kwargs


if __name__ == "__main__":
    raise SystemExit(main())
