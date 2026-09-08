"""Profile window loading on a bounded saved pilot; no model or fitted ES claim."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from mbo_lab.corpus import fingerprint, write_json
from mbo_lab.measurements import process_peak_bytes
from mbo_lab.observation_store import ObservationReader, materialize_pairs
from mbo_lab.pair_index import PairIndex, PairTraversal


def profile(directory, cache_sizes=(2, 32), pairs=512, seed=31):
    directory = Path(directory)
    reader = ObservationReader(directory)
    if reader.manifest["rows"] > 131072:
        raise ValueError("Use a bounded pilot (at most 131072 rows) for this diagnostic")
    record = reader.pair_record()
    corpus = {
        "history": 256,
        "horizon": 128,
        "sessions": [record],
        "fingerprint": fingerprint(record),
    }
    index = PairIndex(corpus)
    pair_ids = PairTraversal(index, seed=seed).take(min(pairs, index.total))
    # Only exercise the normalization interface here; fit real training units separately.
    scale = {
        "corpus": corpus["fingerprint"],
        "features": {
            "names": reader.manifest["names"],
            "mean": [0.0] * 86,
            "scale": [1.0] * 86,
            "time_delta_tau": 1.0,
        },
        "target": {"value": 1.0},
    }
    results = []
    for mib in cache_sizes:
        source = ObservationReader(directory, cache_bytes=mib * 1024**2)
        for phase in ("cold", "warm"):
            before, latencies, total_bytes = dict(source.metrics), [], 0
            start = time.perf_counter()
            for a in range(0, len(pair_ids), 32):
                batch_start = time.perf_counter()
                batch = materialize_pairs(
                    index, pair_ids[a : a + 32], {record["id"]: source}, scale
                )
                latencies.append(time.perf_counter() - batch_start)
                total_bytes += batch.X.nbytes + batch.Y.nbytes
            elapsed = time.perf_counter() - start
            result = {
                "cache_mib": mib,
                "phase": phase,
                "pairs": len(pair_ids),
                "seconds": elapsed,
                "pairs_per_second": len(pair_ids) / elapsed,
                "batch_latency_p50_seconds": float(np.median(latencies)),
                "batch_latency_p95_seconds": float(np.quantile(latencies, 0.95)),
                "peak_cache_bytes": source.peak_bytes,
                "peak_process_bytes": process_peak_bytes(),
                "delivered_XY_bytes": total_bytes,
                **{k: source.metrics[k] - before[k] for k in before},
            }
            results.append(result)
            print(f"{mib} MiB {phase}: {result['pairs_per_second']:.1f} pairs/sec", flush=True)
    data = reader.read_rows(np.arange(reader.manifest["rows"]), ("ts_recv", "mid"))
    anchors = np.concatenate([np.arange(a, b) for a, b in reader.anchor_ranges()])
    seconds = (data["ts_recv"][anchors + 128] - data["ts_recv"][anchors]) / 1e9
    changes = np.r_[0, np.cumsum(np.diff(data["mid"]) != 0)]
    report = {
        "purpose": "Bounded pilot only; identity-like constants, not fitted ES statistics",
        "source": reader.manifest["fingerprint"],
        "rows": reader.manifest["rows"],
        "eligible_anchors": record["anchors"],
        "eligible_pairs": index.total,
        "seed": seed,
        "unique_pair_ids": pair_ids.tolist(),
        "sampling": "Same unique-pair workload repeated solely for cache benchmarking",
        "future_duration_seconds_quantiles": dict(
            zip(("min", "p50", "p95", "max"), np.quantile(seconds, [0, 0.5, 0.95, 1]).tolist())
        ),
        "zero_future_path_fraction": float(np.mean(changes[anchors + 128] == changes[anchors])),
        "window_benchmarks": results,
    }
    write_json(directory / "window-profile.json", report)
    with (directory / "window-profile.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--cache-mib", type=int, nargs="+", default=[2, 32])
    args = parser.parse_args()
    if args.pairs < 1 or min(args.cache_mib) < 1:
        parser.error("pairs and cache sizes must be positive")
    profile(args.directory, args.cache_mib, args.pairs, args.seed)
