"""Local commands; only discover, estimate, and download can contact Databento."""

import argparse
import json
import sys
from dataclasses import replace

from databento.common.error import BentoError

from mbo_lab.budget import initialize, status
from mbo_lab.data import (
    Request,
    build_book,
    discover,
    download,
    estimate,
    inspect_file,
    summarize_book,
)


def confirm_download(details):
    print(json.dumps(details, indent=2, default=str))
    print("Estimated cost, not a billing guarantee. Failed transfers retain the reservation.")
    if not sys.stdin.isatty():
        raise ValueError("Download confirmation requires an interactive terminal; no --yes bypass")
    try:
        return input("Type DOWNLOAD to reserve this estimate and request the data: ") == "DOWNLOAD"
    except (EOFError, KeyboardInterrupt):
        return False


def request_from_args(args):
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be supplied together")
    request = Request.from_toml(args.config)
    if args.start is not None:
        request = replace(request, start=args.start, end=args.end)
    return request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mbo")
    commands = parser.add_subparsers(dest="command", required=True)
    budget = commands.add_parser("budget", help="Inspect a local cumulative budget")
    budget.add_argument("--ledger")
    init = commands.add_parser("budget-init", help="Create a ledger; never overwrite one")
    init.add_argument("--ledger", required=True)
    init.add_argument("--available-usd", required=True)
    init.add_argument("--expires-on", required=True)
    discovery = commands.add_parser(
        "discover", help="List datasets, schemas, ranges and publishers"
    )
    discovery.add_argument("--dataset", default="GLBX.MDP3")
    for name in ("estimate", "download"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="es.toml")
        command.add_argument(
            "--start",
            help="Override profile start in UTC ISO-8601 format (requires --end)",
        )
        command.add_argument(
            "--end",
            help="Override profile end in UTC ISO-8601 format (requires --start)",
        )
        if name == "download":
            command.add_argument("--directory", default="data/raw")
            command.add_argument("--max-cost", type=float, required=True)
            command.add_argument(
                "--ledger",
                help="Override MBO_BUDGET_LEDGER for this command",
            )
    inspection = commands.add_parser("inspect", help="Inspect a local DBN file, without an API key")
    inspection.add_argument("--file", required=True)
    book = commands.add_parser("book", help="Build a book from two local DBN files")
    book.add_argument("--mbo", required=True)
    book.add_argument("--definitions", required=True)
    book.add_argument("--depth", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "budget":
                result = status(args.ledger)
            case "budget-init":
                result = initialize(args.ledger, args.available_usd, args.expires_on)
            case "discover":
                result = discover(args.dataset)
            case "estimate":
                result = estimate(request_from_args(args))
            case "download":
                result = download(
                    request_from_args(args),
                    args.directory,
                    max_cost=args.max_cost,
                    budget_path=args.ledger,
                    confirm=confirm_download,
                )
            case "inspect":
                result = inspect_file(args.file)
            case "book":
                result = summarize_book(build_book(args.mbo, args.definitions), args.depth)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ValueError, RuntimeError, OSError, BentoError) as exc:
        print(f"mbo: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
