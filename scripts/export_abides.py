"""Run with .local/abides-env39/Scripts/python.exe; export the full shared schema."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".local/vendor/abides"
sys.path[:0] = [str(ROOT / "src"), str(VENDOR / "abides-core"), str(VENDOR / "abides-markets")]

import numpy as np  # noqa: E402
from abides_core import abides  # noqa: E402
from abides_markets.agents import ExchangeAgent  # noqa: E402
from abides_markets.configs import rmsc04  # noqa: E402
from abides_markets.messages.order import OrderMsg  # noqa: E402

from mbo_lab.observations import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    TIME_OF_DAY_TIMEZONE,
    Observations,
    feature_names,
    feature_row,
    save_observations,
)
from mbo_lab.paths import abides_paths, require_data_path  # noqa: E402


def run(output=None, seed=0, end_time="10:00:00"):
    output = require_data_path(output or abides_paths(seed, end_time)[0])
    output.mkdir(parents=True, exist_ok=True)
    names = feature_names()
    parameters = dict(
        seed=seed,
        end_time=end_time,
        log_orders=False,
        exchange_log_orders=False,
        book_logging=False,
        stdout_log_level="WARNING",
    )
    config = rmsc04.build_config(**parameters)
    exchange = next(a for a in config["agents"] if isinstance(a, ExchangeAgent))
    original = exchange.receive_message
    rows, mids, times, segments, offsets = [], [], [], [], []
    counts = Counter()
    previous_mid = previous_time = None
    segment = 0

    def record(current_time, sender_id, message):
        nonlocal previous_mid, previous_time, segment
        original(current_time, sender_id, message)
        if (
            not isinstance(message, OrderMsg)
            or not exchange.mkt_open <= current_time <= exchange.mkt_close
        ):
            return
        counts["order_requests"] += 1
        counts[type(message).__name__] += 1
        book = exchange.order_books["ABM"]

        # Sample after the whole request, including all its partial fills/replacements.
        def levels(side):
            result = []
            for level in side:
                quantities = [
                    order.quantity for order, _ in level.visible_orders if order.quantity > 0
                ]
                if quantities:
                    result.append((level.price, sum(quantities), len(quantities)))
                if len(result) == 10:
                    break
            return result

        bids, asks = levels(book.bids), levels(book.asks)
        if not bids or not asks or bids[0][0] > asks[0][0] or bids[0][0] <= 0:
            counts["invalid_states"] += 1
            segment += 1
            previous_mid = previous_time = None
            return
        mid = (bids[0][0] + asks[0][0]) / 2
        rows.append(
            feature_row(
                bids,
                asks,
                mid,
                1,
                current_time,
                previous_mid,
                previous_time,
                clock="simulation",
            )
        )
        mids.append(mid)
        times.append(current_time)
        segments.append(segment)
        offsets.append(counts["order_requests"] - 1)
        previous_mid, previous_time = mid, current_time

    exchange.receive_message = record
    started = time.perf_counter()
    with patch("socket.socket.connect", side_effect=RuntimeError("Simulation must stay offline")):
        abides.run(config, log_dir=str((output / "logs").resolve()), kernel_seed=seed)
    observations = Observations(
        np.asarray(rows, dtype=float).reshape(-1, len(names)),
        np.asarray(mids),
        np.asarray(segments),
        np.asarray(times, dtype=np.uint64),
        np.asarray(times, dtype=np.uint64),
        np.asarray(offsets),
        names,
        "ABIDES_ABM",
        True,
        "simulation",
    )
    path = output / "observations.npz"
    save_observations(path, observations)
    source_files = [
        "abides-core/abides_core/utils.py",
        "abides-markets/abides_markets/configs/rmsc04.py",
        "abides-markets/abides_markets/utils/__init__.py",
    ]
    provenance = {
        "source": "ABIDES RMSC04",
        "synthetic": True,
        "parameters": parameters,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True
        ).strip(),
        "compatibility_patch": "64-bit seed draws and timedelta nanoseconds on Windows",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "runtime": {
            "python": sys.version,
            **{
                package: version(package) for package in ("numpy", "pandas", "pomegranate", "scipy")
            },
        },
        "patched_source_sha256": {
            p: hashlib.sha256((VENDOR / p).read_bytes()).hexdigest() for p in source_files
        },
        "observations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "clock": "simulation nanoseconds; ts_recv and ts_event both store this clock",
        "sampling": "after each complete exchange OrderMsg during market hours; includes no-ops",
        "price_unit": "ABIDES integer cents; one cent tick; not calibrated ES futures",
        "features": list(names),
        "time_features": {
            "clock": "simulation",
            "timezone": TIME_OF_DAY_TIMEZONE,
            "period_seconds": 86_400,
            "encoding": ["tod_sin", "tod_cos"],
        },
        "rows": len(rows),
        "counts": dict(counts),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "features": len(names),
                "counts": dict(counts),
                "elapsed_seconds": provenance["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Override simulated observation directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end-time", default="10:00:00")
    args = parser.parse_args()
    run(args.output, args.seed, args.end_time)
