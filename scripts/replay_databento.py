"""Export a bounded, single-contract saved-data pilot. No network calls."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from mbo_lab.dbn_replay import export_replay
from mbo_lab.measurements import process_peak_bytes
from mbo_lab.paths import require_data_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mbo",
        type=Path,
        action="append",
        required=True,
        help="Repeat in reconstruction order for storage parts",
    )
    p.add_argument("--definitions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeline-id", required=True)
    p.add_argument("--until", required=True, help="Exclusive UTC timestamp, including Z or +00:00")
    p.add_argument("--block-records", type=int, default=4096)
    p.add_argument("--chunk-rows", type=int, default=65536)
    args = p.parse_args()
    end = datetime.fromisoformat(args.until.replace("Z", "+00:00"))
    if end.tzinfo is None or end.utcoffset() != timezone.utc.utcoffset(end):
        p.error("--until must specify UTC")
    output = require_data_path(args.output)
    _, metrics = export_replay(
        args.mbo,
        args.definitions,
        output,
        timeline_id=args.timeline_id,
        stop_ns=int(end.timestamp() * 1e9),
        block_records=args.block_records,
        chunk_rows=args.chunk_rows,
        memory_sample=process_peak_bytes,
    )
    print(f"Saved {metrics['rows']:,} rows; measurements: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
