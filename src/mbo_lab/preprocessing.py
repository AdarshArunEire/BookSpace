"""Shared training-only statistics and bounded exact scalar median selection."""

import json
import tempfile
from pathlib import Path

import numpy as np

from mbo_lab.corpus import fingerprint, history_ranges, verified_source, write_json
from mbo_lab.metrics.fixed import TargetScale
from mbo_lab.pair_index import PairIndex, PairTraversal
from mbo_lab.samples import FeatureScale


def exact_disk_median(path, count, chunk=262144):
    """Select the middle positive float64 values with eight bounded radix passes."""
    if count < 1 or Path(path).stat().st_size != count * 8:
        raise ValueError("Need a nonempty float64 scalar file")
    # Positive finite IEEE float bits have the same ordering as their numerical values.
    states = [(0, (count - 1) // 2), (0, count // 2)]
    for shift in range(56, -1, -8):
        prefixes = {prefix for prefix, _ in states}
        histograms = {prefix: np.zeros(256, dtype=np.int64) for prefix in prefixes}
        with Path(path).open("rb") as source:
            while len(values := np.fromfile(source, dtype=np.float64, count=chunk)):
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError("Median source must be finite and strictly positive")
                bits = values.view(np.uint64)
                for prefix in prefixes:
                    selected = bits if shift == 56 else bits[(bits >> (shift + 8)) == prefix]
                    byte = ((selected >> shift) & 255).astype(np.int64)
                    histograms[prefix] += np.bincount(byte, minlength=256)
        next_states = []
        for prefix, rank in states:
            cumulative = np.cumsum(histograms[prefix])
            byte = int(np.searchsorted(cumulative, rank, side="right"))
            before = int(cumulative[byte - 1]) if byte else 0
            next_states.append(((prefix << 8) | byte, rank - before))
        states = next_states
    middle = np.array([p for p, _ in states], dtype=np.uint64).view(np.float64)
    return float(middle[0] / 2 + middle[1] / 2)


def transform_rows(x, names, scale):
    if tuple(names) != tuple(scale["names"]):
        raise ValueError("Preprocessing feature schema mismatch")
    values, valid, _ = FeatureScale._prepare(x, tuple(names), scale["time_delta_tau"])
    return np.where(valid, (values - scale["mean"]) / scale["scale"], 0)


def fit_preprocessing(corpus, root, output, *, target_pairs=100000, seed=17, progress=print):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preprocessing already exists; use a new output")
    if not 1 <= target_pairs <= 200000:
        raise ValueError("Target fitting budget must be between 1 and 200000 pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [r for r in corpus["sessions"] if r["split"] == "train"]
    names = tuple(corpus["names"])
    history = corpus["history"]
    delta_col = names.index("time_delta_seconds")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="scalar-fit-") as temporary:
        path = Path(temporary) / "positive-gaps.f64"
        count = 0
        with path.open("wb") as stream:
            for i, record in enumerate(records, 1):
                obs = verified_source(root, record)
                for a, b in history_ranges(record, history):
                    gaps = obs.x[a:b, delta_col]
                    gaps = np.asarray(gaps[gaps > 0], dtype=np.float64)
                    gaps.tofile(stream)
                    count += len(gaps)
                del obs
                if progress and (i % 50 == 0 or i == len(records)):
                    progress(f"Training gap pass {i}/{len(records)}", flush=True)
        tau = exact_disk_median(path, count)
    counts = np.zeros(len(names), dtype=np.int64)
    means, m2 = np.zeros(len(names)), np.zeros(len(names))
    covered_rows = 0
    for i, record in enumerate(records, 1):
        obs = verified_source(root, record)
        for a, b in history_ranges(record, history):
            covered_rows += b - a
            for start in range(a, b, 8192):
                values, valid, protected = FeatureScale._prepare(
                    obs.x[start : min(start + 8192, b)], names, tau
                )
                for col in range(len(names)):
                    present = values[valid[:, col], col]
                    n = len(present)
                    if not n:
                        continue
                    mean = float(present.mean())
                    centered = present - mean
                    variance_sum = float(np.sum(centered * centered))
                    total = int(counts[col]) + n
                    delta = mean - means[col]
                    m2[col] += variance_sum + delta * delta * counts[col] * n / total
                    means[col] += delta * n / total
                    counts[col] = total
        del obs
        if progress and (i % 50 == 0 or i == len(records)):
            progress(f"Training statistics pass {i}/{len(records)}", flush=True)
    std = np.sqrt(np.maximum(0, m2 / np.maximum(counts, 1)))
    constant = (std == 0) & ~protected
    means[protected] = 0
    std[(std == 0) | protected] = 1
    scale = {
        "names": list(names),
        "mean": means.tolist(),
        "scale": std.tolist(),
        "constant": constant.tolist(),
        "counts": counts.tolist(),
        "time_delta_tau": tau,
        "training_history_rows": covered_rows,
        "positive_training_gaps": count,
    }
    index = PairIndex(corpus, "train")
    pair_ids = PairTraversal(index, seed).take(min(target_pairs, index.total))
    ids, inverse = np.unique(
        np.array([index.unrank(p) for p in pair_ids], dtype=np.int64).reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    futures = np.empty((len(ids), corpus["horizon"]), dtype=np.float64)
    for s in np.unique(ids[:, 0]):
        obs = verified_source(root, index.records[s])
        positions = np.flatnonzero(ids[:, 0] == s)
        for offset in range(0, len(positions), 4096):
            take = positions[offset : offset + 4096]
            anchors = ids[take, 1]
            futures[take] = np.log(
                obs.mid[anchors[:, None] + np.arange(1, corpus["horizon"] + 1)]
                / obs.mid[anchors, None]
            )
        del obs
    pairs = inverse.reshape(-1, 2)
    raw = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), 4096):
        selected = pairs[start : start + 4096]
        raw[start : start + len(selected)] = np.sqrt(
            np.mean((futures[selected[:, 0]] - futures[selected[:, 1]]) ** 2, axis=1)
        )
    target = TargetScale.fit(raw, seed)
    # Separate halves diagnose scale sensitivity without fitting on held-out data.
    halves = [
        float(np.median(part[part > 0])) if np.any(part > 0) else None
        for part in np.array_split(raw, 2)
    ]
    body = {
        "version": 1,
        "corpus": corpus["fingerprint"],
        "features": scale,
        "target": {
            "value": target.value,
            "pair_count": target.pair_count,
            "zero_fraction": target.zero_fraction,
            "seed": seed,
            "raw_quantiles": np.quantile(raw, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
            "half_positive_medians": halves,
        },
        "target_pair_ids": pair_ids.tolist(),
        "traversal_version": PairTraversal.version,
        "train_sources": fingerprint(records),
    }
    body["fingerprint"] = fingerprint(body)
    write_json(output, body)
    return body


def read_preprocessing(path, corpus):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if fingerprint({k: v for k, v in data.items() if k != "fingerprint"}) != data["fingerprint"]:
        raise ValueError("Corrupt preprocessing artifact")
    if data["corpus"] != corpus["fingerprint"]:
        raise ValueError("Preprocessing belongs to another corpus")
    return data
