#!/usr/bin/env python
"""
Parallel Databento daily replay builder for BookSpace.

Discovers daily GLBX MDP3 MBO + definition DBN files, validates that each
day begins with an initialization snapshot, then runs independent daily
exports in parallel worker processes.

Designed for Windows spawn semantics: worker entry points are module-level
and guarded by if __name__ == "__main__".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Job:
    day: str
    mbo_paths: tuple[str, ...]
    definitions: str
    output: str
    timeline_id: str
    stop_ns: int
    block_records: int
    chunk_rows: int


def _next_midnight_ns(day_text: str) -> int:
    day = date.fromisoformat(day_text)
    next_day = day + timedelta(days=1)
    stamp = datetime.combine(next_day, dt_time.min, tzinfo=timezone.utc)
    return int(stamp.timestamp()) * 1_000_000_000


def _extract_date_and_part(path: Path, kind: str, symbol_tag: str) -> tuple[str, int] | None:
    # Example:
    # glbx-mdp3-20250102.mbo.ES.v.0.0000.dbn.zst
    pattern = re.compile(
        rf"^glbx-mdp3-(\d{{8}})\.{re.escape(kind)}\.{re.escape(symbol_tag)}\.(\d+)\.dbn\.zst$",
        re.IGNORECASE,
    )
    match = pattern.match(path.name)
    if not match:
        return None
    raw_day, raw_part = match.groups()
    day = f"{raw_day[:4]}-{raw_day[4:6]}-{raw_day[6:8]}"
    return day, int(raw_part)


def discover_jobs(
    raw_root: Path,
    output_root: Path,
    *,
    symbol_tag: str,
    block_records: int,
    chunk_rows: int,
) -> tuple[list[Job], list[str]]:
    mbo_by_day: dict[str, list[tuple[int, Path]]] = {}
    defs_by_day: dict[str, list[tuple[int, Path]]] = {}

    for path in raw_root.rglob("*.dbn.zst"):
        parsed = _extract_date_and_part(path, "mbo", symbol_tag)
        if parsed is not None:
            day, part = parsed
            mbo_by_day.setdefault(day, []).append((part, path.resolve()))
            continue

        parsed = _extract_date_and_part(path, "definition", symbol_tag)
        if parsed is not None:
            day, part = parsed
            defs_by_day.setdefault(day, []).append((part, path.resolve()))

    errors: list[str] = []
    jobs: list[Job] = []

    all_days = sorted(set(mbo_by_day) | set(defs_by_day))
    for day in all_days:
        mbos = sorted(mbo_by_day.get(day, []))
        defs = sorted(defs_by_day.get(day, []))

        if not mbos:
            errors.append(f"{day}: definition exists but no MBO file")
            continue
        if not defs:
            errors.append(f"{day}: MBO exists but no definition file")
            continue
        if len(defs) != 1:
            errors.append(
                f"{day}: expected exactly one definition file, found {len(defs)}: "
                + ", ".join(str(p) for _, p in defs)
            )
            continue

        out = (output_root / day.replace("-", "")).resolve()
        jobs.append(
            Job(
                day=day,
                mbo_paths=tuple(str(path) for _, path in mbos),
                definitions=str(defs[0][1]),
                output=str(out),
                timeline_id=f"{symbol_tag}-{day.replace('-', '')}",
                stop_ns=_next_midnight_ns(day),
                block_records=block_records,
                chunk_rows=chunk_rows,
            )
        )

    return jobs, errors


def _completed_output(output: Path) -> bool:
    observations = output / "observations.json"
    metrics = output / "metrics.json"
    if not observations.is_file() or not metrics.is_file():
        return False

    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        return int(payload.get("rows", 0)) > 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _preflight_initial_snapshot(job: Job) -> tuple[str, bool, str]:
    """
    Validate that a daily file can be reconstructed independently.

    Weekdays normally begin with Databento's synthetic 00:00 UTC snapshot:
        R(SNAPSHOT), A(SNAPSHOT), ..., F_LAST

    CME weekly-session starts (typically Sunday) can instead begin with an
    unflagged channel-reset R, followed immediately by the CME MBO snapshot.
    That is also safe, provided no live state-changing event appears before the
    first complete F_SNAPSHOT event.
    """
    try:
        import databento_dbn as dbn
        from mbo_lab.dbn_replay import raw_records

        metrics: dict[str, Any] = {
            "read_decompress_seconds": 0.0,
            "dbn_decode_seconds": 0.0,
            "decompressed_bytes": 0,
        }
        records = raw_records(job.mbo_paths[0], metrics)

        try:
            metadata = next(records)
            first = next(records)

            if not isinstance(metadata, dbn.Metadata):
                return job.day, False, (
                    f"first decoded item is {type(metadata).__name__}, not Metadata"
                )
            if not isinstance(first, dbn.MBOMsg):
                return job.day, False, (
                    f"first record is {type(first).__name__}, not MBOMsg"
                )

            if str(first.action) != "R":
                return job.day, False, (
                    f"first record is action={str(first.action)!r}, expected initial clear R"
                )

            # Ordinary weekday synthetic snapshot.
            if first.flags & dbn.F_SNAPSHOT:
                return job.day, True, "synthetic daily snapshot"

            # Weekly-session cold start: allow exactly the initial unflagged
            # clear event, then require a complete CME snapshot before any
            # ordinary live state-changing event.
            initial_clear_complete = bool(first.flags & dbn.F_LAST)
            saw_snapshot = False

            for record in records:
                if not isinstance(record, dbn.MBOMsg):
                    return job.day, False, (
                        f"unexpected record type {type(record).__name__}"
                    )

                is_snapshot = bool(record.flags & dbn.F_SNAPSHOT)

                if is_snapshot:
                    saw_snapshot = True
                    if record.flags & dbn.F_LAST:
                        return job.day, True, "weekly-session CME snapshot"
                    continue

                if not initial_clear_complete:
                    if record.flags & dbn.F_LAST:
                        initial_clear_complete = True
                    continue

                # Once the initial clear event has ended, the next book-building
                # event must be the weekly snapshot. Reject A/M/C live traffic
                # before it, because that would make an independent cold start
                # ambiguous/incomplete.
                if str(record.action) in {"A", "M", "C"}:
                    return job.day, False, (
                        "live state-changing MBO event arrived before the "
                        "weekly-session snapshot"
                    )

                # Permit non-book T/F/N records while waiting; they don't alter
                # book state. Keep scanning until F_SNAPSHOT arrives.

            if saw_snapshot:
                return job.day, False, "snapshot started but never reached F_LAST"
            return job.day, False, "no initialization snapshot found after initial clear"

        finally:
            records.close()

    except Exception as exc:
        return job.day, False, f"{type(exc).__name__}: {exc}"


def _run_job(job: Job) -> dict[str, Any]:
    # Import inside the child process so Windows spawn does not execute a replay
    # during module import and each worker owns its own Nautilus state.
    started = time.perf_counter()
    try:
        from mbo_lab.dbn_replay import export_replay

        _, metrics = export_replay(
            list(job.mbo_paths),
            job.definitions,
            job.output,
            timeline_id=job.timeline_id,
            stop_ns=job.stop_ns,
            block_records=job.block_records,
            chunk_rows=job.chunk_rows,
        )

        return {
            "day": job.day,
            "status": "ok",
            "rows": int(metrics["rows"]),
            "elapsed_seconds": float(metrics["elapsed_seconds"]),
            "output": job.output,
            "worker_wall_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "day": job.day,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "output": job.output,
            "worker_wall_seconds": time.perf_counter() - started,
        }


def _write_summary(
    output_root: Path,
    *,
    args: argparse.Namespace,
    discovered: int,
    skipped: list[str],
    results: list[dict[str, Any]],
    wall_seconds: float,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "parallel-build-summary.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "symbol_tag": args.symbol_tag,
        "raw_root": str(args.raw_root.resolve()),
        "output_root": str(output_root.resolve()),
        "chunk_rows": args.chunk_rows,
        "block_records": args.block_records,
        "discovered_jobs": discovered,
        "skipped_completed": skipped,
        "wall_seconds": wall_seconds,
        "ok": sorted((r for r in results if r["status"] == "ok"), key=lambda r: r["day"]),
        "failed": sorted((r for r in results if r["status"] == "failed"), key=lambda r: r["day"]),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily BookSpace Databento observation stores in parallel."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Root recursively containing Databento .dbn.zst files (default: data/raw)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/databento/daily"),
        help="Parent directory for YYYYMMDD daily outputs",
    )
    parser.add_argument(
        "--symbol-tag",
        default="ES.v.0",
        help="Filename symbol tag (default: ES.v.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Worker PROCESS count (default: 6)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=65536,
        help="Observation rows per compressed NPZ chunk (default: 65536)",
    )
    parser.add_argument(
        "--block-records",
        type=int,
        default=4096,
        help="DBN->Nautilus conversion block size (default: 4096)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip checking that every daily source starts with a snapshot CLEAR",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover, validate and print jobs without running exports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be >= 1")
    if not args.raw_root.is_dir():
        raise SystemExit(f"Raw root does not exist: {args.raw_root}")

    jobs, discovery_errors = discover_jobs(
        args.raw_root,
        args.output_root,
        symbol_tag=args.symbol_tag,
        block_records=args.block_records,
        chunk_rows=args.chunk_rows,
    )

    if discovery_errors:
        print("Discovery errors:", file=sys.stderr)
        for error in discovery_errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    if not jobs:
        print("No matching MBO/definition daily pairs found.", file=sys.stderr)
        return 2

    print(f"Discovered {len(jobs)} daily job(s) for {args.symbol_tag}.")

    skipped: list[str] = []
    pending: list[Job] = []
    incomplete: list[Job] = []

    for job in jobs:
        output = Path(job.output)
        if _completed_output(output):
            skipped.append(job.day)
        elif output.exists():
            incomplete.append(job)
        else:
            pending.append(job)

    if skipped:
        print(f"Resume: {len(skipped)} completed day(s) will be skipped.")

    # Deliberately refuse to delete/overwrite incomplete outputs automatically.
    if incomplete:
        print(
            "\nIncomplete output directories already exist. "
            "export_replay intentionally refuses to reuse them.",
            file=sys.stderr,
        )
        for job in incomplete:
            print(f"  {job.day}: {job.output}", file=sys.stderr)
        print(
            "\nDelete/rename those directories after inspecting them, then rerun.",
            file=sys.stderr,
        )
        return 3

    if not args.skip_preflight:
        print(f"Preflight: checking initialization snapshot on {len(pending)} pending day(s)...")
        failures: list[tuple[str, str]] = []
        for index, job in enumerate(pending, 1):
            day, ok, message = _preflight_initial_snapshot(job)
            if not ok:
                failures.append((day, message))
            if index % 20 == 0 or index == len(pending):
                print(f"  preflight {index}/{len(pending)}")

        if failures:
            print("\nPreflight failed; parallel build NOT started:", file=sys.stderr)
            for day, message in failures:
                print(f"  {day}: {message}", file=sys.stderr)
            return 4

        print("Preflight passed.")

    if args.dry_run:
        print("\nDry run only. Pending jobs:")
        for job in pending:
            print(
                f"  {job.day}: {len(job.mbo_paths)} MBO part(s) -> {job.output}"
            )
        return 0

    if not pending:
        print("Everything is already complete.")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"\nStarting {len(pending)} day(s) with "
        f"{min(args.workers, len(pending))} worker process(es)."
    )
    print(f"chunk_rows={args.chunk_rows:,}, block_records={args.block_records:,}")
    print("Child replay progress may interleave in this console.\n")

    wall_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_job = {pool.submit(_run_job, job): job for job in pending}

        completed = 0
        total = len(future_to_job)
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            try:
                result = future.result()
            except BaseException as exc:
                # Covers catastrophic worker failures not caught inside _run_job.
                result = {
                    "day": job.day,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "output": job.output,
                }

            results.append(result)

            if result["status"] == "ok":
                elapsed = result["elapsed_seconds"]
                rows = result["rows"]
                rate = rows / elapsed if elapsed > 0 else float("inf")
                print(
                    f"[{completed:>3}/{total}] OK   {job.day}  "
                    f"{rows:,} rows  {elapsed/60:.1f} min  {rate:,.0f} rows/s"
                )
            else:
                print(
                    f"[{completed:>3}/{total}] FAIL {job.day}  {result['error']}",
                    file=sys.stderr,
                )

    wall_seconds = time.perf_counter() - wall_start
    summary = _write_summary(
        args.output_root,
        args=args,
        discovered=len(jobs),
        skipped=skipped,
        results=results,
        wall_seconds=wall_seconds,
    )

    ok_count = sum(r["status"] == "ok" for r in results)
    failed_count = sum(r["status"] == "failed" for r in results)

    print(
        f"\nFinished: {ok_count} succeeded, {failed_count} failed, "
        f"{len(skipped)} previously complete."
    )
    print(f"Wall time: {wall_seconds/3600:.2f} h")
    print(f"Summary: {summary}")

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
