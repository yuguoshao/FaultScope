"""Command line entry points for `python -m faultscope.collection`."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from faultscope.collection._collect import Collector
from faultscope.collection._types import (
    CollectionOptions,
    CollectionRunOptions,
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
from faultscope.collection.threshold import analyze_thresholds, plot_threshold_analysis


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
    collect_parser.add_argument("--progress-mode", choices=("final", "stream"), default="final")
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

    threshold_parser = subcommands.add_parser("threshold")
    threshold_parser.add_argument("stats", nargs="+")
    threshold_parser.add_argument("--x-key", required=True)
    threshold_parser.add_argument("--distance-key", required=True)
    threshold_parser.add_argument("--series-key", action="append", default=[])
    threshold_parser.add_argument("--count-key")
    threshold_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    threshold_parser.add_argument("--confidence-level", type=float, default=0.95)
    threshold_parser.add_argument("--seed", type=int, default=0)
    threshold_parser.add_argument("--scaling-order", type=int, choices=(1, 2, 3), default=2)
    threshold_parser.add_argument("--format", choices=("text", "json"), default="text")
    threshold_parser.add_argument("--plot-out")
    threshold_parser.set_defaults(func=_cmd_threshold)
    return parser


def _cmd_collect(args: argparse.Namespace) -> int:
    factory = _load_factory(args.tasks_factory)
    tasks = factory(**_factory_kwargs(args.factory_arg))
    if isinstance(tasks, TaskStats):
        raise TypeError("tasks factory must return CollectionTask objects, not TaskStats")
    if isinstance(tasks, CollectionTask):
        tasks = [tasks]
    collector = Collector(
        options=CollectionOptions(
            max_shots=args.max_shots,
            max_errors=args.max_errors,
            batch_size=args.batch_size,
        ),
        run_options=CollectionRunOptions(
            seed=args.seed,
            num_workers=args.num_workers,
            decoders=tuple(args.decoders or ()),
            existing_data_filepaths=tuple(args.existing_data_filepath),
            save_resume_filepath=args.save_resume_filepath,
        ),
    )
    if args.progress_mode == "stream":
        stats = collector._collect_with_progress(tasks, lambda _progress: None)
    else:
        stats = collector.collect(tasks)
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


def _cmd_threshold(args: argparse.Namespace) -> int:
    stats = read_stats_from_csv_files(args.stats)
    results = analyze_thresholds(
        stats,
        x_key=args.x_key,
        distance_key=args.distance_key,
        series_keys=args.series_key,
        count_key=args.count_key,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        scaling_order=args.scaling_order,
    )
    if args.plot_out is not None:
        plot_threshold_analysis(results, output=args.plot_out)
    if args.format == "json":
        print(json.dumps([result.to_dict() for result in results]))
    else:
        _print_threshold_text(results)
    return 0


def _print_threshold_text(results: tuple[Any, ...]) -> None:
    print(
        "\t".join(
            [
                "series",
                "kind",
                "lower_distance",
                "upper_distance",
                "status",
                "estimate",
                "ci_low",
                "ci_high",
                "bootstrap_samples",
                "bootstrap_successes",
                "nu",
                "nu_ci_low",
                "nu_ci_high",
                "nu_bootstrap_samples",
                "nu_bootstrap_successes",
                "message",
            ]
        )
    )
    for result in results:
        series = _format_series(result.series)
        for crossing in result.crossings:
            estimate = crossing.estimate
            print(
                "\t".join(
                    [
                        series,
                        "pairwise",
                        _format_number(crossing.lower_distance),
                        _format_number(crossing.upper_distance),
                        crossing.status,
                        _format_estimate_value(estimate),
                        _format_estimate_ci_low(estimate),
                        _format_estimate_ci_high(estimate),
                        str(estimate.bootstrap_samples if estimate is not None else 0),
                        str(estimate.bootstrap_successes if estimate is not None else 0),
                        "",
                        "",
                        "",
                        "0",
                        "0",
                        "",
                    ]
                )
            )
        pairwise = result.pairwise_threshold
        print(
            "\t".join(
                [
                    series,
                    "pairwise_global",
                    "",
                    "",
                    "ok" if pairwise is not None else "no_crossing",
                    _format_estimate_value(pairwise),
                    _format_estimate_ci_low(pairwise),
                    _format_estimate_ci_high(pairwise),
                    str(pairwise.bootstrap_samples if pairwise is not None else 0),
                    str(pairwise.bootstrap_successes if pairwise is not None else 0),
                    "",
                    "",
                    "",
                    "0",
                    "0",
                    "",
                ]
            )
        )
        scaling = result.scaling_fit
        threshold = scaling.threshold
        critical_exponent = scaling.critical_exponent
        print(
            "\t".join(
                [
                    series,
                    "scaling_global",
                    "",
                    "",
                    scaling.status,
                    _format_estimate_value(threshold),
                    _format_estimate_ci_low(threshold),
                    _format_estimate_ci_high(threshold),
                    str(threshold.bootstrap_samples if threshold is not None else 0),
                    str(threshold.bootstrap_successes if threshold is not None else 0),
                    _format_estimate_value(critical_exponent),
                    _format_estimate_ci_low(critical_exponent),
                    _format_estimate_ci_high(critical_exponent),
                    str(
                        critical_exponent.bootstrap_samples if critical_exponent is not None else 0
                    ),
                    str(
                        critical_exponent.bootstrap_successes
                        if critical_exponent is not None
                        else 0
                    ),
                    scaling.message or "",
                ]
            )
        )


def _format_series(series: Any) -> str:
    if not series:
        return "all"
    return ",".join(f"{key}={value}" for key, value in series.items())


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _format_estimate_value(estimate: Any | None) -> str:
    return "" if estimate is None else _format_number(estimate.value)


def _format_estimate_ci_low(estimate: Any | None) -> str:
    return "" if estimate is None else _format_number(estimate.ci_low)


def _format_estimate_ci_high(estimate: Any | None) -> str:
    return "" if estimate is None else _format_number(estimate.ci_high)


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
