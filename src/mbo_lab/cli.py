"""Databento requests and local DBN inspection."""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime

from databento.common.error import BentoError

from mbo_lab.batch import SPLIT_SIZE
from mbo_lab.batch import download as batch_download
from mbo_lab.data import (
    Request,
    build_book,
    discover,
    download,
    estimate,
    inspect_file,
    summarize_book,
)


def request_from_args(args):
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be supplied together")
    request = Request.from_toml(args.config)
    if args.start is not None:
        request = replace(request, start=args.start, end=args.end)
    return request


def print_fields(value, indent=0):
    padding = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            label = key.replace("_", " ")
            if isinstance(item, (dict, list)):
                print(f"{padding}{label}:")
                print_fields(item, indent + 2)
            else:
                print(f"{padding}{label}: {item}")
    elif isinstance(value, list):
        if not value:
            print(f"{padding}(none)")
        for item in value:
            print_fields(item, indent)
    else:
        print(f"{padding}{value}")


def print_estimate(result):
    request = result["request"]
    print(f"{request['dataset']} / {request['symbol']}")
    print(f"{request['start']} to {request['end']} (UTC)")
    print()
    print(f"{'Schema':<12} {'Billable bytes':>16} {'USD/GB':>12} {'Estimate USD':>16}")
    for schema in ("definition", "mbo"):
        rate = result["usd_per_gb"][schema]
        rate_text = f"{rate:.6g}" if rate is not None else "unavailable"
        print(
            f"{schema:<12} {result['billable_bytes'][schema]:>16,} "
            f"{rate_text:>12} {result['usd'][schema]:>16.9f}"
        )
    print(f"Total estimate: ${result['total_usd']:.9f}")
    print(f"Bytes are uncompressed binary; rates are {result['mode']} USD/GB.")
    print("Provider cost estimates include applicable plan discounts.")
    for schema, seconds, label in (("mbo", 600, "10-minute"), ("definition", 86400, "24-hour")):
        if any(
            datetime.fromisoformat(request[edge]).timestamp() % seconds for edge in ("start", "end")
        ):
            print(f"Note: {schema} estimates may overstate partial {label} intervals.")


def print_download(result):
    if "receipt" in result:
        print(f"Downloaded {result['downloaded_files']} files; reused {result['reused_files']}.")
        print(f"Verified: {result['verification']}")
        print(f"Stored size: {result['stored_bytes'] / 1e9:.3f} GB")
        print(f"Data: {result['directory']}")
        print(f"Resume receipt: {result['receipt']}")
        return
    print("Using cached data." if result["cached"] else "Download complete.")
    for chunk in result["chunks"]:
        print("  Cached:" if chunk["cached"] else "  Saved:")
        for schema, path in chunk["paths"].items():
            print(f"    {schema}: {path.resolve()}")
    print(f"Manifest: {result['range_manifest_path'].resolve()}")


def download_progress(index, total, request):
    print(
        f"[{index}/{total}] {request.symbol}: {request.start} to {request.end}",
        file=sys.stderr,
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mbo",
        description="Estimate and download Databento MBO; inspect saved books.",
        epilog="Examples: mbo estimate | mbo download | mbo inspect --file FILE",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover", help="Show available datasets and schemas")
    discovery.add_argument("--dataset", default="GLBX.MDP3")
    parsers = [discovery]
    for name, help_text in (
        ("estimate", "Show cost, billable bytes and unit prices without downloading"),
        ("download", "Download data; spending controls are managed in Databento"),
    ):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument(
            "--config", default="es.toml", help="Request profile (default: es.toml)"
        )
        command.add_argument("--start", help="UTC start at midnight; requires --end")
        command.add_argument("--end", help="Exclusive UTC end; requires --start")
        command.add_argument(
            "--stream", action="store_true", help="Use streaming for a small smoke request"
        )
        if name == "download":
            command.add_argument(
                "--directory", default="data/raw", help="Output folder (default: data/raw)"
            )
            command.add_argument(
                "--dry-run", action="store_true", help="Show request without contacting Databento"
            )
            command.add_argument("--mbo-job", help="Recover an existing MBO batch job ID")
            command.add_argument(
                "--definition-job", help="Recover an existing definitions batch job ID"
            )
        parsers.append(command)
    inspection = commands.add_parser("inspect", help="Inspect a local DBN file")
    inspection.add_argument("--file", required=True)
    book = commands.add_parser("book", help="Build an L3 book from local MBO and definitions")
    book.add_argument("--mbo", required=True)
    book.add_argument("--definitions", required=True)
    book.add_argument("--depth", type=int, default=10)
    for command in [*parsers, inspection, book]:
        command.add_argument("--json", action="store_true", help="Print JSON for scripts")
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "discover":
                result = discover(args.dataset)
            case "estimate":
                request = request_from_args(args)
                if not args.json:
                    print("Fetching estimate from Databento...", file=sys.stderr, flush=True)
                result = estimate(
                    request, mode="historical-streaming" if args.stream else "historical"
                )
            case "download":
                request = request_from_args(args)
                if args.stream and (args.mbo_job or args.definition_job):
                    raise ValueError("Job IDs apply to batch downloads; remove --stream")
                if args.dry_run:
                    result = {
                        "request": vars(request),
                        "method": "stream" if args.stream else "batch",
                        "schemas": ["definition", "mbo"],
                        "encoding": "dbn",
                        "compression": "zstd",
                        "directory": args.directory,
                        "paid_requests_submitted": 0,
                    }
                    if not args.stream:
                        result.update(
                            split_duration="day", split_symbols=True, split_size=SPLIT_SIZE
                        )
                elif args.stream:
                    result = download(
                        request, args.directory, progress=None if args.json else download_progress
                    )
                else:
                    result = batch_download(
                        request,
                        args.directory,
                        progress=None
                        if args.json
                        else lambda message: print(message, file=sys.stderr, flush=True),
                        jobs={
                            schema: job
                            for schema, job in (
                                ("mbo", args.mbo_job),
                                ("definition", args.definition_job),
                            )
                            if job
                        },
                    )
            case "inspect":
                result = inspect_file(args.file)
            case "book":
                if args.depth < 1:
                    raise ValueError("--depth must be positive")
                result = summarize_book(build_book(args.mbo, args.definitions), args.depth)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        elif args.command == "estimate":
            print_estimate(result)
        elif args.command == "download":
            if args.dry_run:
                print_fields(result)
            else:
                print_download(result)
        else:
            print_fields(result)
        return 0
    except (ValueError, RuntimeError, OSError, BentoError) as exc:
        message = str(exc)
        if message == "invalid API key, was None":
            message = "Set DATABENTO_API_KEY in this terminal, then retry."
        print(f"mbo: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("mbo: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
