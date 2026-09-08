"""Audit, fit, or benchmark saved-session data preparation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mbo_lab.corpus import (
    audit_corpus,
    audit_databento_corpus,
    fingerprint,
    read_corpus,
    write_json,
)
from mbo_lab.measurements import process_peak_bytes
from mbo_lab.pair_index import PairIndex, PairTraversal
from mbo_lab.paths import DATA, require_data_path
from mbo_lab.preprocessing import (
    fit_databento_preprocessing,
    fit_preprocessing,
    read_preprocessing,
)
from mbo_lab.stream import open_pair_stream


def fixed_validation(corpus, path, count=8192, seed=29):
    index = PairIndex(corpus, "validation")
    ids = PairTraversal(index, seed).take(min(count, index.total)).tolist()
    body = {
        "version": 1,
        "corpus": corpus["fingerprint"],
        "split": "validation",
        "seed": seed,
        "traversal_version": PairTraversal.version,
        "pair_count": len(ids),
        "pair_ids": ids,
    }
    body["fingerprint"] = fingerprint(body)
    if path.exists():
        if json.loads(path.read_text()) != body:
            raise ValueError("Existing validation sample differs; choose a new artifact")
    else:
        write_json(path, body)
    return body


def benchmark(corpus, root, preprocessing, output, *, workers, queue_pairs, memory_gib, queues):
    started = time.perf_counter()
    seen, queue_timings = set(), []
    with open_pair_stream(
        corpus,
        root,
        preprocessing,
        workers=workers,
        queue_pairs=queue_pairs,
        batch_pairs=32,
        memory_bytes=int(memory_gib * 1024**3),
    ) as stream:
        for queue_number in range(queues):
            queue_start = time.perf_counter()
            delivered = 0
            while delivered < queue_pairs:
                result = next(stream)
                current = set(result.pair_ids.tolist())
                if current & seen or len(current) != len(result.pair_ids):
                    raise AssertionError("Repeated pair in benchmark stream")
                seen.update(current)
                delivered += len(result.pair_ids)
                if result.X.shape[1:] != (corpus["history"], 86) or not np.isfinite(result.D).all():
                    raise AssertionError("Invalid model-ready batch")
            elapsed = time.perf_counter() - queue_start
            queue_timings.append(
                {
                    "queue": queue_number + 1,
                    "seconds": elapsed,
                    "pairs": delivered,
                    "pairs_per_second": delivered / elapsed,
                }
            )
            print(
                f"workers={workers}, queue={queue_number + 1}: {delivered / elapsed:.1f} pairs/s, "
                f"peak RAM={process_peak_bytes() / 1024**3:.2f} GiB",
                flush=True,
            )
        report = stream.report()
        stream.checkpoint(output.with_suffix(".checkpoint.json"))
    report.update(
        {
            "workers": workers,
            "queue_pairs": queue_pairs,
            "queue_timings": queue_timings,
            "seconds": time.perf_counter() - started,
            "peak_process_bytes": process_peak_bytes(),
            "corpus": corpus["fingerprint"],
            "preprocessing": preprocessing["fingerprint"],
            "purpose": "Data-loader benchmark only; no optimizer or model",
            "cold_definition": "first queue with empty application cache; OS cache uncontrolled",
        }
    )
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "fit", "validation", "benchmark"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=DATA / "processed/abides/training-v1")
    parser.add_argument("--workers", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--queue-pairs", type=int, default=16384)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--queues", type=int, default=2)
    args = parser.parse_args()
    output = require_data_path(args.output)
    if args.action == "audit":
        if args.source is None:
            parser.error("audit requires --source")
        source = args.source.resolve()
        chunked = any((path / "observations.json").is_file() for path in source.iterdir())
        if chunked:
            audit_databento_corpus(source, output / "corpus.json")
        else:
            audit_corpus(source, output / "corpus.json", workers=args.workers)
        return
    corpus, root = read_corpus(output / "corpus.json")
    if args.action == "fit":
        if corpus.get("source_format") == "observation-chunks-v1":
            fit_databento_preprocessing(corpus, root, output / "preprocessing.json")
        else:
            fit_preprocessing(corpus, root, output / "preprocessing.json")
    elif args.action == "validation":
        fixed_validation(corpus, output / "validation_pairs.json")
    else:
        scale = read_preprocessing(output / "preprocessing.json", corpus)
        benchmark(
            corpus,
            root,
            scale,
            output / f"benchmark-{args.workers}-workers-{args.queue_pairs}-pairs.json",
            workers=args.workers,
            queue_pairs=args.queue_pairs,
            memory_gib=args.memory_gib,
            queues=args.queues,
        )


if __name__ == "__main__":
    main()
