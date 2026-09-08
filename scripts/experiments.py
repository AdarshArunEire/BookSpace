#!/usr/bin/env python3
"""Run the reusable BookSpace experiment pipeline."""

from mbo_lab.experiment import ExperimentError, main

if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, ValueError) as exc:
        raise SystemExit(f"experiments: {exc}") from exc
