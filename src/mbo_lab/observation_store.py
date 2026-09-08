"""Bounded observation chunks indexed on a logical timeline, not by filenames."""

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from mbo_lab.corpus import file_hash, fingerprint, write_json
from mbo_lab.observations import FEATURE_SCHEMA_VERSION, feature_names

COLUMNS = ("x", "mid", "segment", "ts_recv", "ts_event", "source_row")


class ObservationWriter:
    def __init__(self, directory, timeline_id, instrument, *, chunk_rows=65536):
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError("Use a new export directory; incomplete exports are not reusable")
        if not 384 <= chunk_rows <= 131072:
            raise ValueError("chunk_rows must be between 384 and 131072")
        self.directory.mkdir(parents=True)
        self.timeline_id, self.instrument = timeline_id, instrument
        self.chunk_rows = chunk_rows
        self.buffer = []
        self.rows = 0
        self.chunks = []
        self.segments = []

    def append(self, x, mid, segment, ts_recv, ts_event, source_row):
        if not self.segments or self.segments[-1]["segment"] != segment:
            self.segments.append({"segment": segment, "start": self.rows, "stop": self.rows})
        self.buffer.append((x, mid, segment, ts_recv, ts_event, source_row))
        self.rows += 1
        self.segments[-1]["stop"] = self.rows
        if len(self.buffer) == self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        values = list(zip(*self.buffer))
        arrays = {
            key: np.asarray(value, dtype=np.float64 if key in ("x", "mid") else np.int64)
            for key, value in zip(COLUMNS, values)
        }
        path = self.directory / f"rows-{len(self.chunks):06d}.npz"
        np.savez_compressed(path, **arrays)
        self.chunks.append(
            {
                "path": path.name,
                "start": self.rows - len(self.buffer),
                "stop": self.rows,
                "sha256": file_hash(path),
                "array_bytes": sum(v.nbytes for v in arrays.values()),
            }
        )
        self.buffer.clear()

    def finish(self, provenance):
        self.flush()
        if not self.rows:
            raise ValueError("No valid real-event observations were exported")
        manifest = {
            "format": "observation-chunks-v1",
            "timeline_id": self.timeline_id,
            "instrument": self.instrument,
            "synthetic": False,
            "clock": "receive",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "names": list(feature_names()),
            "rows": self.rows,
            "chunks": self.chunks,
            "segments": self.segments,
            "provenance": provenance,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(self.directory / "observations.json", manifest)
        return manifest


class ObservationReader:
    def __init__(self, directory, *, cache_bytes=128 * 1024**2):
        import json

        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "observations.json").read_text())
        m = self.manifest
        if fingerprint({k: v for k, v in m.items() if k != "fingerprint"}) != m["fingerprint"]:
            raise ValueError("Observation manifest checksum mismatch")
        if (
            m["format"] != "observation-chunks-v1"
            or m["names"] != list(feature_names())
            or m["feature_schema_version"] != FEATURE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported observation format/schema")
        previous = 0
        for chunk in m["chunks"]:
            if chunk["start"] != previous or chunk["stop"] <= previous:
                raise ValueError("Noncontiguous observation chunks")
            previous = chunk["stop"]
        if previous != m["rows"]:
            raise ValueError("Chunk row total mismatch")
        if cache_bytes < max(c["array_bytes"] for c in m["chunks"]):
            raise ValueError("Cache must hold at least one chunk")
        self.stops = np.array([c["stop"] for c in m["chunks"]])
        self.cache_bytes, self.resident_bytes, self.peak_bytes = cache_bytes, 0, 0
        self.cache = OrderedDict()
        self.verified_chunks = {}
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evicted_bytes": 0,
            "loaded_array_bytes": 0,
            "checksum_seconds": 0.0,
            "npz_load_seconds": 0.0,
            "requested_rows": 0,
        }

    def _load(self, index, columns=COLUMNS):
        key = (index, tuple(columns))
        if key in self.cache:
            self.metrics["cache_hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        chunk = self.manifest["chunks"][index]
        path = (self.directory / chunk["path"]).resolve()
        started = time.perf_counter()
        if not path.is_relative_to(self.directory):
            raise ValueError("Chunk path escapes observation directory")
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError("Chunk path/checksum mismatch") from exc
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.verified_chunks.get(index) != signature:
            if file_hash(path) != chunk["sha256"]:
                raise ValueError("Chunk path/checksum mismatch")
            self.verified_chunks[index] = signature
        self.metrics["checksum_seconds"] += time.perf_counter() - started
        self.metrics["cache_misses"] += 1
        while self.cache and self.resident_bytes + chunk["array_bytes"] > self.cache_bytes:
            _, evicted = self.cache.popitem(last=False)
            self.resident_bytes -= sum(v.nbytes for v in evicted.values())
            self.metrics["evicted_bytes"] += sum(v.nbytes for v in evicted.values())
            del evicted
        started = time.perf_counter()
        with np.load(path, allow_pickle=False) as archive:
            arrays = {column: archive[column] for column in columns}
        self.metrics["npz_load_seconds"] += time.perf_counter() - started
        loaded_bytes = sum(array.nbytes for array in arrays.values())
        self.metrics["loaded_array_bytes"] += loaded_bytes
        n = chunk["stop"] - chunk["start"]
        for column, array in arrays.items():
            expected = (n, 86) if column == "x" else (n,)
            if array.shape != expected:
                raise ValueError("Chunk column shape mismatch")
        if tuple(columns) == COLUMNS and loaded_bytes != chunk["array_bytes"]:
            raise ValueError("Chunk byte accounting mismatch")
        self.resident_bytes += loaded_bytes
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        self.cache[key] = arrays
        return arrays

    def read_rows(self, rows, columns=COLUMNS):
        rows = np.asarray(rows, dtype=np.int64)
        if (
            rows.ndim != 1
            or len(rows) > 131072
            or np.any(rows < 0)
            or np.any(rows >= self.manifest["rows"])
        ):
            raise ValueError("Request must contain at most 131072 valid row IDs")
        if not set(columns) <= set(COLUMNS):
            raise ValueError("Unknown observation column")
        self.metrics["requested_rows"] += len(rows)
        result = {
            k: np.empty(
                (len(rows), 86) if k == "x" else (len(rows),),
                dtype=np.float64 if k in ("x", "mid") else np.int64,
            )
            for k in columns
        }
        indices = np.searchsorted(self.stops, rows, side="right")
        for index in np.unique(indices):
            positions = np.flatnonzero(indices == index)
            source = self._load(int(index), columns)
            local = rows[positions] - self.manifest["chunks"][index]["start"]
            for k in columns:
                result[k][positions] = source[k][local]
            del source
        return result

    def iter_ranges(self, ranges, columns=("x",), *, batch_rows=65536):
        """Yield contiguous row batches from half-open logical ranges."""
        if not isinstance(batch_rows, int) or not 1 <= batch_rows <= 131072:
            raise ValueError("batch_rows must be between 1 and 131072")
        for start, stop in ranges:
            if not 0 <= start <= stop <= self.manifest["rows"]:
                raise ValueError("Invalid logical row range")
            for offset in range(start, stop, batch_rows):
                rows = np.arange(offset, min(offset + batch_rows, stop), dtype=np.int64)
                yield self.read_rows(rows, columns)

    def anchor_ranges(self, history=256, horizon=128):
        return [
            [s["start"] + history - 1, s["stop"] - horizon]
            for s in self.manifest["segments"]
            if s["stop"] - s["start"] >= history + horizon
        ]

    def histories(self, anchors, history=256, horizon=128):
        anchors = np.asarray(anchors, dtype=np.int64)
        ranges = self.anchor_ranges(history, horizon)
        max_anchors = 131072 // history
        if anchors.ndim != 1 or len(anchors) > max_anchors or history < 1 or horizon < 1:
            raise ValueError(f"Need at most {max_anchors} anchors and positive window lengths")
        if not all(any(a <= t < b for a, b in ranges) for t in anchors):
            raise ValueError("Episode crosses logical segment/timeline boundary")
        past = anchors[:, None] + np.arange(1 - history, 1)
        return self.read_rows(past.ravel(), ("x",))["x"].reshape(len(anchors), history, 86)

    def episodes(self, anchors, history=256, horizon=128):
        anchors = np.asarray(anchors, dtype=np.int64)
        X = self.histories(anchors, history, horizon)
        future = anchors[:, None] + np.arange(horizon + 1)
        mid = self.read_rows(future.ravel(), ("mid",))["mid"].reshape(len(anchors), horizon + 1)
        return X, np.log(mid[:, 1:] / mid[:, :1])

    def pair_record(self, *, split="train", history=256, horizon=128):
        from mbo_lab.pair_index import count_range_pairs

        ranges = self.anchor_ranges(history, horizon)
        count = count_range_pairs(ranges, history + horizon)
        return {
            "id": self.manifest["timeline_id"],
            "split": split,
            "anchor_ranges": ranges,
            "anchors": sum(b - a for a, b in ranges),
            "within_pairs": count,
            "source_fingerprint": self.manifest["fingerprint"],
        }


def materialize_pairs(index, pair_ids, readers, preprocessing):
    """Common pair-batch shape from readers exposing episodes() and feature names.

    Readers are keyed by logical timeline ID; storage chunks never enter pair IDs.
    Caller owns reader cache budgets. This primitive has no model/optimizer.
    """
    from mbo_lab.preprocessing import transform_rows
    from mbo_lab.stream import PairBatch

    if preprocessing.get("corpus") != index.corpus["fingerprint"]:
        raise ValueError("Preprocessing belongs to another corpus")
    pair_ids = np.asarray(pair_ids, dtype=np.int64)
    if pair_ids.ndim != 1 or not 1 <= len(pair_ids) <= 128:
        raise ValueError("Materialize between 1 and 128 pairs per call")
    if len(np.unique(pair_ids)) != len(pair_ids):
        raise ValueError("Repeated pair IDs in batch")
    ids, inverse = np.unique(
        np.array([index.unrank(p) for p in pair_ids]).reshape(-1, 2), axis=0, return_inverse=True
    )
    history, horizon = index.corpus["history"], index.corpus["horizon"]
    X = np.empty((len(ids), history, 86), dtype=np.float32)
    Y = np.empty((len(ids), horizon), dtype=np.float64)
    for session in np.unique(ids[:, 0]):
        positions = np.flatnonzero(ids[:, 0] == session)
        reader = readers[index.records[session]["id"]]
        if index.records[session].get("source_fingerprint") != reader.manifest["fingerprint"]:
            raise ValueError("Reader belongs to another source version")
        raw, future = reader.episodes(ids[positions, 1], history, horizon)
        X[positions] = transform_rows(
            raw.reshape(-1, 86), reader.manifest["names"], preprocessing["features"]
        ).reshape(-1, history, 86)
        Y[positions] = future
    pairs = inverse.reshape(-1, 2)
    raw = np.sqrt(np.mean((Y[pairs[:, 0]] - Y[pairs[:, 1]]) ** 2, axis=1))
    target_scale = preprocessing["target"]["value"]
    if not np.isfinite(target_scale) or target_scale <= 0 or not np.isfinite(X).all():
        raise ValueError("Invalid normalization/target scale")
    return PairBatch(
        X,
        Y,
        pairs,
        raw / target_scale,
        raw,
        tuple((index.records[s]["id"], int(t)) for s, t in ids),
        pair_ids.copy(),
    )
