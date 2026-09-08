"""Frozen, audited session metadata. Observation archives remain in place."""

import hashlib
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mbo_lab.datasets import session_dates
from mbo_lab.observations import FEATURE_SCHEMA_VERSION, feature_names
from mbo_lab.pipeline import load_abides_source
from mbo_lab.samples import valid_anchors


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded_map(pool, function, items, workers):
    items = iter(items)
    pending = deque()
    for _ in range(workers):
        item = next(items, None)
        if item is not None:
            pending.append(pool.submit(function, item))
    while pending:
        yield pending.popleft().result()
        item = next(items, None)
        if item is not None:
            pending.append(pool.submit(function, item))


def anchor_ranges(segment, history, horizon):
    edges = np.r_[0, np.flatnonzero(np.diff(segment) != 0) + 1, len(segment)]
    return [
        [int(a + history - 1), int(b - horizon)]
        for a, b in zip(edges[:-1], edges[1:])
        if b - a >= history + horizon
    ]


def expand_ranges(ranges):
    if not ranges:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in ranges])


def history_ranges(record, history):
    return [(a - history + 1, b) for a, b in record["anchor_ranges"]]


def verified_source(root, record):
    source = (Path(root) / record["path"]).resolve()
    if not source.is_relative_to(Path(root).resolve()):
        raise ValueError("Source escapes corpus root")
    if file_hash(source / "provenance.json") != record["provenance_sha256"]:
        raise ValueError(f"Changed provenance: {source}")
    obs, provenance, _, _ = load_abides_source(source)
    if provenance["observations_sha256"] != record["observations_sha256"]:
        raise ValueError(f"Changed observation source: {source}")
    return obs


def audit_session(source, history=256, horizon=128):
    source = Path(source)
    obs, prov, observation_path, provenance_path = load_abides_source(source)
    n = len(obs.mid)
    if n == 0 or prov["rows"] != n:
        raise ValueError("Empty source or provenance row mismatch")
    if not obs.synthetic or obs.clock != "simulation" or prov.get("source") != "ABIDES RMSC04":
        raise ValueError("Expected independent ABIDES simulation sessions")
    for values in (obs.ts_recv, obs.ts_event, obs.source_row, obs.segment):
        if values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError("Expected nonnegative integer metadata")
    if np.any(obs.ts_recv[1:] < obs.ts_recv[:-1]) or not np.array_equal(obs.ts_recv, obs.ts_event):
        raise ValueError("Invalid simulation clock order")
    if np.any(obs.source_row[1:] <= obs.source_row[:-1]):
        raise ValueError("Source offsets must increase strictly")
    day = int(np.datetime64(prov["parameters"]["date"], "D").astype(np.int64))
    if np.any(obs.ts_recv // 86_400_000_000_000 != day):
        raise ValueError("Simulation timestamps disagree with session date")
    starts = np.r_[True, np.diff(obs.segment) != 0]
    names = obs.names
    if not np.array_equal(obs.x[:, names.index("initial")], starts):
        raise ValueError("Initialization flags disagree with segment transitions")
    gaps = obs.x[:, names.index("time_delta_seconds")]
    expected_gaps = np.r_[0, np.diff(obs.ts_recv) / 1e9]
    expected_gaps[starts] = 0
    if not np.allclose(gaps, expected_gaps, rtol=1e-12, atol=1e-12):
        raise ValueError("Time-gap features disagree with timestamps")
    if not np.allclose(obs.x[:, names.index("mid_return")], obs.y, atol=1e-14, rtol=1e-10):
        raise ValueError("Return features disagree with mid-prices/segments")
    for side in ("bid", "ask"):
        previous = None
        for level in range(1, 11):
            col = names.index(f"{side}_{level}_price_ticks")
            fields = obs.x[:, col : col + 4]
            present = fields[:, 3]
            if not np.isin(present, [0, 1]).all():
                raise ValueError("Nonbinary presence masks")
            if level == 1 and not np.all(present == 1):
                raise ValueError("Missing best quotes")
            if np.any(fields[present == 0, :3] != 0):
                raise ValueError("Absent levels must be zero")
            if np.any(fields[present == 1, 1:3] <= 0):
                raise ValueError("Nonpositive present quantity/order count")
            if np.any(fields[:, 2] != np.floor(fields[:, 2])):
                raise ValueError("Fractional order counts")
            if previous is not None:
                if np.any(present > previous[:, 3]):
                    raise ValueError("Holes in level presence")
                delta = fields[present == 1, 0] - previous[present == 1, 0]
                if np.any(delta >= 0 if side == "bid" else delta <= 0):
                    raise ValueError("Unordered book levels")
            previous = fields
    bid = obs.x[:, names.index("bid_1_price_ticks")]
    ask = obs.x[:, names.index("ask_1_price_ticks")]
    if np.any(bid > 0) or np.any(ask < 0) or not np.allclose(bid, -ask):
        raise ValueError("Best quotes inconsistent with midpoint")
    if not np.allclose(obs.x[:, names.index("spread_ticks")], ask - bid):
        raise ValueError("Spread inconsistent with best quotes")
    seconds = (obs.ts_recv % 86_400_000_000_000) / 1e9
    theta = seconds * (2 * np.pi / 86400)
    if not np.allclose(obs.x[:, -2:], np.column_stack((np.sin(theta), np.cos(theta))), atol=1e-12):
        raise ValueError("Invalid time-of-day features")
    with np.load(observation_path, allow_pickle=False) as archive:
        if int(archive["feature_schema_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError("Archive schema mismatch")
        if not np.array_equal(archive["y"], obs.y):
            raise ValueError("Stored one-step returns disagree with mid-prices")
    ranges = anchor_ranges(obs.segment, history, horizon)
    anchors = expand_ranges(ranges)
    if not np.array_equal(anchors, valid_anchors(obs, history, horizon)):
        raise ValueError("Anchor range disagreement")
    q = len(anchors) - np.searchsorted(anchors, anchors + history + horizon)
    durations = (obs.ts_recv[anchors + horizon] - obs.ts_recv[anchors]) / 1e9
    changes = np.r_[0, np.cumsum(np.diff(obs.mid) != 0)]
    zero_paths = changes[anchors + horizon] == changes[anchors]
    lengths = np.diff(np.r_[0, np.flatnonzero(starts)[1:], n])
    signature = {
        key: prov[key]
        for key in (
            "source",
            "synthetic",
            "upstream_commit",
            "patched_source_sha256",
            "runtime",
            "feature_schema_version",
            "sampling",
            "price_unit",
        )
    }
    signature["parameters"] = {
        k: v for k, v in prov["parameters"].items() if k not in ("seed", "date")
    }
    return {
        "id": source.name,
        "path": source.name,
        "rows": n,
        "seed": prov["parameters"]["seed"],
        "date": prov["parameters"]["date"],
        "observations_sha256": prov["observations_sha256"],
        "provenance_sha256": file_hash(provenance_path),
        "signature": signature,
        "compressed_bytes": observation_path.stat().st_size,
        "array_bytes": sum(x.nbytes for x in vars(obs).values() if isinstance(x, np.ndarray)),
        "x_dtype": str(obs.x.dtype),
        "instrument": obs.instrument,
        "anchor_ranges": ranges,
        "anchors": len(anchors),
        "within_pairs": int(q.sum()),
        "segments": int(starts.sum()),
        "segment_length_quantiles": np.quantile(lengths, [0, 0.5, 0.95, 1]).tolist(),
        "future_seconds_quantiles": np.quantile(durations, [0, 0.5, 0.95, 1]).tolist()
        if len(anchors)
        else [],
        "zero_path_fraction": float(zero_paths.mean()) if len(anchors) else None,
        "constant_features": [names[i] for i in np.flatnonzero(np.ptp(obs.x, axis=0) == 0)],
        "absent_fraction": {
            name: float(np.mean(obs.x[:, i] == 0))
            for i, name in enumerate(names)
            if name.endswith("_present")
        },
        "counts": prov["counts"],
    }


def audit_corpus(root, output, *, history=256, horizon=128, workers=2, progress=print):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Frozen corpus already exists; use a new output")
    if history < 1 or horizon < 1 or workers not in (1, 2, 4):
        raise ValueError("Invalid window lengths or worker count (1, 2, 4)")
    original = json.loads((root / "dataset.json").read_text())
    expected_dates = {d.isoformat() for d in session_dates(original["requested_sessions"])}
    listed = {s["date"]: s for s in original["sessions"]}
    sources, excluded = [], []
    # Resume resets dataset.json before rechecking prior exports; recover only this run's dates.
    for source in sorted(p for p in root.iterdir() if p.is_dir()):
        if source.name not in expected_dates:
            raise ValueError(f"Unexpected directory in dataset: {source}")
        if not all((source / name).is_file() for name in ("observations.npz", "provenance.json")):
            if source.name in listed:
                raise ValueError(f"Manifest-listed session is incomplete: {source}")
            excluded.append({"id": source.name, "reason": "incomplete export"})
        else:
            sources.append(source)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = bounded_map(pool, lambda p: audit_session(p, history, horizon), sources, workers)
        for i, record in enumerate(results, 1):
            expected_seed = int.from_bytes(
                hashlib.sha256(f"{original['seed']}:{record['id']}".encode()).digest()[:4], "big"
            ) % (2**32 - 1)
            if record["seed"] != expected_seed or record["date"] != record["id"]:
                raise ValueError("Session identity/configuration mismatch")
            previous = listed.get(record["id"])
            if previous and any(
                previous[k] != record[k] for k in ("rows", "seed", "observations_sha256")
            ):
                raise ValueError("Generation manifest disagrees with source")
            if record["anchors"] == 0:
                excluded.append({"id": record["id"], "reason": "no complete episodes"})
            else:
                records.append(record)
            if progress and (i % 25 == 0 or i == len(sources)):
                progress(f"Audited {i}/{len(sources)} sessions", flush=True)
    if len(records) < 7:
        raise ValueError("Need at least seven usable sessions for 70/15/15 splits")
    if len({r["observations_sha256"] for r in records}) != len(records):
        raise ValueError("Duplicate observation exports")
    if len({fingerprint(r["signature"]) for r in records}) != 1:
        raise ValueError("Mixed simulation configurations/source revisions")
    if len({r["seed"] for r in records}) != len(records):
        raise ValueError("Duplicate session seeds")
    train, validation = int(len(records) * 0.70), int(len(records) * 0.15)
    for i, record in enumerate(records):
        record["split"] = (
            "train" if i < train else "validation" if i < train + validation else "test"
        )
    body = {
        "version": 1,
        "root": os.path.relpath(root, output.parent),
        "source_manifest_sha256": fingerprint(original),
        "generation_manifest_sessions": len(listed),
        "history": history,
        "horizon": horizon,
        "names": list(feature_names()),
        "schema_version": FEATURE_SCHEMA_VERSION,
        "split_policy": "whole sessions in date-label order; floor 70%, floor 15%, remainder",
        "sessions": records,
        "excluded": excluded,
    }
    body["fingerprint"] = fingerprint(body)
    write_json(output, body)
    return body


def read_corpus(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if fingerprint({k: v for k, v in data.items() if k != "fingerprint"}) != data["fingerprint"]:
        raise ValueError("Corrupt frozen corpus manifest")
    return data, (path.parent / data["root"]).resolve()
