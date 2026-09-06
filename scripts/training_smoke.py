"""Command-line wrapper for the reusable ABIDES training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mbo_lab.pipeline import run_training_pipeline


def run(
    output=None,
    seed=0,
    end_time="10:00:00",
    observations_dir=None,
    history=256,
    horizon=128,
    anchor_count=64,
    split_row=None,
):
    """Run the default smoke configuration and return its report."""
    result = run_training_pipeline(
        output=output,
        seed=seed,
        end_time=end_time,
        observations_dir=observations_dir,
        history=history,
        horizon=horizon,
        anchor_count=anchor_count,
        split_row=split_row,
    )
    return result.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Override processed output directory")
    parser.add_argument("--observations-dir", type=Path, help="Override ABIDES source directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end-time", default="10:00:00")
    parser.add_argument("--history", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--anchor-count", type=int, default=64)
    parser.add_argument("--split-row", type=int)
    args = parser.parse_args(argv)
    report = run(
        output=args.output,
        seed=args.seed,
        end_time=args.end_time,
        observations_dir=args.observations_dir,
        history=args.history,
        horizon=args.horizon,
        anchor_count=args.anchor_count,
        split_row=args.split_row,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "feature_names"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
