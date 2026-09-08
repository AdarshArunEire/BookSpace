"""RAM-bounded gathering of globally sampled pairs into model-ready minibatches."""

import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mbo_lab.corpus import bounded_map, fingerprint, verified_source, write_json
from mbo_lab.pair_index import PairIndex, PairTraversal
from mbo_lab.preprocessing import transform_rows


class SessionCache:
    def __init__(self, root, records, capacity):
        self.root, self.records, self.capacity = root, records, capacity
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.loads = self.hits = 0
        self.load_seconds = 0.0
        self.lock = threading.Lock()

    def cached(self, session):
        with self.lock:
            return session in self.entries

    def get(self, session):
        with self.lock:
            if session in self.entries:
                self.hits += 1
                self.entries.move_to_end(session)
                return self.entries[session]
        started = time.perf_counter()
        obs = verified_source(self.root, self.records[session])
        size = sum(a.nbytes for a in vars(obs).values() if isinstance(a, np.ndarray))
        with self.lock:
            self.loads += 1
            self.load_seconds += time.perf_counter() - started
            while self.entries and self.bytes + size > self.capacity:
                _, old = self.entries.popitem(last=False)
                self.bytes -= sum(a.nbytes for a in vars(old).values() if isinstance(a, np.ndarray))
            if size <= self.capacity:
                self.entries[session] = obs
                self.bytes += size
                self.peak_bytes = max(self.peak_bytes, self.bytes)
        return obs


@dataclass
class PairBatch:
    X: np.ndarray
    Y: np.ndarray
    pairs: np.ndarray
    D: np.ndarray
    D_raw: np.ndarray
    sample_ids: tuple
    pair_ids: np.ndarray


class PairStream:
    """Sample globally, gather by session, then emit small deduplicated batches.

    The budget covers owned data buffers and a conservative loading/transform
    reserve, not the Python interpreter or batches retained by the caller.
    Checkpoints record the delivered position; an unfinished RAM queue is
    deterministically rebuilt, without skipping or redelivering completed batches.
    """

    def __init__(
        self,
        corpus,
        root,
        preprocessing,
        *,
        split="train",
        seed=0,
        queue_pairs=16384,
        batch_pairs=32,
        memory_bytes=6 * 1024**3,
        workers=4,
        progress=None,
    ):
        if preprocessing["corpus"] != corpus["fingerprint"]:
            raise ValueError("Preprocessing/corpus mismatch")
        if (
            not isinstance(queue_pairs, int)
            or not isinstance(batch_pairs, int)
            or not 1 <= batch_pairs <= queue_pairs
            or workers not in (1, 2, 4)
        ):
            raise ValueError("Invalid queue, batch or workers")
        self.index = PairIndex(corpus, split)
        self.traversal = PairTraversal(self.index, seed)
        self.preprocessing = preprocessing
        self.history, self.horizon = corpus["history"], corpus["horizon"]
        self.features = tuple(corpus["names"])
        self.queue_pairs, self.batch_pairs = queue_pairs, batch_pairs
        self.memory_bytes, self.workers = memory_bytes, workers
        self.progress = progress
        largest = max(r["array_bytes"] for r in self.index.records)
        self.queue_budget = (
            2 * queue_pairs * (self.history * len(self.features) * 4 + self.horizon * 8 + 64)
        )
        batch_budget = (
            2 * batch_pairs * (self.history * len(self.features) * 4 + self.horizon * 8 + 64)
        )
        # At most workers live sources (including evicted references); archive loading,
        # validation copies, 16-history transforms, plus pair-index/Python metadata reserve.
        self.reserve = workers * (largest * 5 + 64 * 1024**2) + batch_budget + 128 * 1024**2
        capacity = memory_bytes - self.queue_budget - self.reserve
        if capacity < largest:
            raise ValueError("Memory budget too small for queue/loading reserve and one session")
        self.cache = SessionCache(root, self.index.records, capacity)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.queue = None
        self.delivered = 0
        self.metrics = {
            "pairs": 0,
            "same_session_pairs": 0,
            "queue_unique_histories": 0,
            "queues": 0,
            "gather_seconds": 0.0,
            "raw_distance_sum": 0.0,
        }
        self.session_pairs = np.zeros(len(self.index.records), dtype=np.int64)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.pool.shutdown(wait=True)
        self.queue = None
        self.cache.entries.clear()
        self.cache.bytes = 0

    def _fill(self):
        self.queue = None
        start = self.traversal.counter
        pair_ids = self.traversal.take(self.queue_pairs)
        self.traversal.counter = start
        if not len(pair_ids):
            raise StopIteration
        endpoints = np.array([self.index.unrank(p) for p in pair_ids], dtype=np.int64)
        ids, inverse = np.unique(endpoints.reshape(-1, 2), axis=0, return_inverse=True)
        X = np.empty((len(ids), self.history, len(self.features)), dtype=np.float32)
        Y = np.empty((len(ids), self.horizon), dtype=np.float64)
        groups = [(int(s), np.flatnonzero(ids[:, 0] == s)) for s in np.unique(ids[:, 0])]
        groups.sort(key=lambda item: (not self.cache.cached(item[0]), item[0]))
        started = time.perf_counter()

        def gather(group):
            session, positions = group
            obs = self.cache.get(session)
            for offset in range(0, len(positions), 16):
                take = positions[offset : offset + 16]
                anchors = ids[take, 1]
                past = anchors[:, None] + np.arange(1 - self.history, 1)
                raw = obs.x[past]
                normalized = transform_rows(
                    raw.reshape(-1, len(self.features)), obs.names, self.preprocessing["features"]
                )
                X[take] = normalized.reshape(len(take), self.history, len(self.features))
                Y[take] = np.log(
                    obs.mid[anchors[:, None] + np.arange(1, self.horizon + 1)]
                    / obs.mid[anchors, None]
                )

        try:
            for completed, _ in enumerate(
                bounded_map(self.pool, gather, groups, self.workers), start=1
            ):
                if self.progress:
                    self.progress(completed, len(groups))
        except Exception:
            # No batch was delivered. A retry/checkpoint must begin at the same pair.
            self.pool.shutdown(wait=True)
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
            self.traversal.counter = start
            raise
        for offset in range(0, len(ids), 16):
            if (
                not np.isfinite(X[offset : offset + 16]).all()
                or not np.isfinite(Y[offset : offset + 16]).all()
            ):
                self.traversal.counter = start
                raise ValueError("Nonfinite model input/target after float32 conversion")
        self.queue = {
            "start": start,
            "cursor": 0,
            "pair_ids": pair_ids,
            "ids": ids,
            "pairs": inverse.reshape(-1, 2),
            "X": X,
            "Y": Y,
        }
        self.traversal.counter = start + len(pair_ids)
        self.metrics["queues"] += 1
        self.metrics["queue_unique_histories"] += len(ids)
        self.metrics["gather_seconds"] += time.perf_counter() - started

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue is None or self.queue["cursor"] == len(self.queue["pair_ids"]):
            self._fill()
        q = self.queue
        a, b = q["cursor"], min(q["cursor"] + self.batch_pairs, len(q["pair_ids"]))
        unique, inverse = np.unique(q["pairs"][a:b], return_inverse=True)
        pairs = inverse.reshape(-1, 2)
        Y = q["Y"][unique]
        raw = np.sqrt(np.mean((Y[pairs[:, 0]] - Y[pairs[:, 1]]) ** 2, axis=1))
        ids = q["ids"][unique]
        pair_ids = q["pair_ids"][a:b].copy()
        result = PairBatch(
            q["X"][unique],
            Y,
            pairs,
            raw / self.preprocessing["target"]["value"],
            raw,
            tuple((self.index.records[s]["id"], int(t)) for s, t in ids),
            pair_ids,
        )
        q["cursor"] = b
        self.delivered = q["start"] + b
        self.metrics["pairs"] += b - a
        self.metrics["raw_distance_sum"] += float(raw.sum())
        self.metrics["same_session_pairs"] += int(
            np.sum(ids[pairs[:, 0], 0] == ids[pairs[:, 1], 0])
        )
        np.add.at(self.session_pairs, ids[pairs.ravel(), 0], 1)
        return result

    def report(self):
        return {
            **self.metrics,
            "delivered_counter": self.delivered,
            "pair_universe": self.index.total,
            "sessions_seen": int(np.count_nonzero(self.session_pairs)),
            "session_endpoint_counts": self.session_pairs.tolist(),
            "cache_loads": self.cache.loads,
            "cache_hits": self.cache.hits,
            "cache_peak_bytes": self.cache.peak_bytes,
            "cache_capacity": self.cache.capacity,
            "queue_budget_bytes": self.queue_budget,
            "reserve_bytes": self.reserve,
            "memory_budget_bytes": self.memory_bytes,
            "load_worker_seconds": self.cache.load_seconds,
        }

    def checkpoint(self, path):
        state = self.traversal.state()
        state["counter"] = self.delivered
        body = {
            "version": 1,
            "traversal": state,
            "preprocessing": self.preprocessing["fingerprint"],
            "queue_pairs": self.queue_pairs,
            "batch_pairs": self.batch_pairs,
            "metrics": self.metrics,
            "session_endpoint_counts": self.session_pairs.tolist(),
        }
        body["fingerprint"] = fingerprint(body)
        write_json(path, body)

    def restore(self, path):
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            fingerprint({k: v for k, v in body.items() if k != "fingerprint"})
            != body["fingerprint"]
        ):
            raise ValueError("Corrupt stream checkpoint")
        if (
            body["version"] != 1
            or body["preprocessing"] != self.preprocessing["fingerprint"]
            or body["batch_pairs"] != self.batch_pairs
            or body["queue_pairs"] != self.queue_pairs
        ):
            raise ValueError("Incompatible stream checkpoint")
        self.traversal = PairTraversal.restore(self.index, body["traversal"])
        self.delivered = self.traversal.counter
        # Rebuild the original queue so minibatch boundaries and deduplication are identical.
        offset = self.delivered % self.queue_pairs
        if offset:
            self.traversal.counter -= offset
            self._fill()
            self.queue["cursor"] = offset
        else:
            self.queue = None
        self.metrics = body["metrics"]
        self.session_pairs = np.asarray(body["session_endpoint_counts"], dtype=np.int64)


class ChunkPairStream:
    """RAM-bounded pair queues backed by daily ObservationReader stores."""

    def __init__(
        self,
        corpus,
        root,
        preprocessing,
        *,
        split="train",
        seed=0,
        queue_pairs=16384,
        batch_pairs=32,
        memory_bytes=6 * 1024**3,
        workers=4,
        progress=None,
    ):
        from mbo_lab.observation_store import ObservationReader

        if preprocessing["corpus"] != corpus["fingerprint"]:
            raise ValueError("Preprocessing/corpus mismatch")
        if (
            not isinstance(queue_pairs, int)
            or not isinstance(batch_pairs, int)
            or not 1 <= batch_pairs <= queue_pairs
            or workers not in (1, 2, 4)
        ):
            raise ValueError("Invalid queue, batch or workers")
        self.reader_type = ObservationReader
        self.root = Path(root).resolve()
        self.index = PairIndex(corpus, split)
        self.traversal = PairTraversal(self.index, seed)
        self.preprocessing = preprocessing
        self.history, self.horizon = corpus["history"], corpus["horizon"]
        self.features = tuple(corpus["names"])
        self.queue_pairs, self.batch_pairs = queue_pairs, batch_pairs
        self.memory_bytes, self.workers = memory_bytes, workers
        self.progress = progress
        largest = max(int(record["max_chunk_bytes"]) for record in self.index.records)
        self.queue_budget = (
            2 * queue_pairs * (self.history * len(self.features) * 4 + self.horizon * 8 + 64)
        )
        batch_budget = (
            2 * batch_pairs * (self.history * len(self.features) * 4 + self.horizon * 8 + 64)
        )
        self.reserve = workers * (largest * 4 + 64 * 1024**2) + batch_budget + 128 * 1024**2
        if self.queue_budget + self.reserve > memory_bytes:
            raise ValueError("Memory budget too small for chunk queue and worker reserves")
        self.reader_cache_bytes = largest
        self.readers = {}
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.queue = None
        self.delivered = 0
        self.metrics = {
            "pairs": 0,
            "same_session_pairs": 0,
            "queue_unique_histories": 0,
            "queues": 0,
            "gather_seconds": 0.0,
            "raw_distance_sum": 0.0,
            "source_format": "observation-chunks-v1",
        }
        self.session_pairs = np.zeros(len(self.index.records), dtype=np.int64)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.pool.shutdown(wait=True)
        self.queue = None
        self.readers.clear()

    def _reader(self, session):
        if session in self.readers:
            return self.readers[session]
        record = self.index.records[session]
        source = (self.root / record["path"]).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError("Source escapes corpus root")
        reader = self.reader_type(source, cache_bytes=self.reader_cache_bytes)
        if reader.manifest["fingerprint"] != record["source_fingerprint"]:
            raise ValueError(f"Changed observation source: {source}")
        self.readers[session] = reader
        return reader

    def _fill(self):
        self.queue = None
        start = self.traversal.counter
        pair_ids = self.traversal.take(self.queue_pairs)
        self.traversal.counter = start
        if not len(pair_ids):
            raise StopIteration
        endpoints = np.array([self.index.unrank(pair) for pair in pair_ids], dtype=np.int64)
        ids, inverse = np.unique(endpoints.reshape(-1, 2), axis=0, return_inverse=True)
        X = np.empty((len(ids), self.history, len(self.features)), dtype=np.float32)
        Y = np.empty((len(ids), self.horizon), dtype=np.float64)
        groups = [
            (int(session), np.flatnonzero(ids[:, 0] == session)) for session in np.unique(ids[:, 0])
        ]
        started = time.perf_counter()

        def gather(group):
            session, positions = group
            reader = self._reader(session)
            try:
                for offset in range(0, len(positions), 256):
                    take = positions[offset : offset + 256]
                    raw, future = reader.episodes(
                        ids[take, 1], history=self.history, horizon=self.horizon
                    )
                    X[take] = transform_rows(
                        raw.reshape(-1, len(self.features)),
                        reader.manifest["names"],
                        self.preprocessing["features"],
                    ).reshape(len(take), self.history, len(self.features))
                    Y[take] = future
            finally:
                reader.cache.clear()
                reader.resident_bytes = 0

        try:
            for completed, _ in enumerate(
                bounded_map(self.pool, gather, groups, self.workers), start=1
            ):
                if self.progress:
                    self.progress(completed, len(groups))
        except Exception:
            self.pool.shutdown(wait=True)
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
            self.traversal.counter = start
            raise
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            self.traversal.counter = start
            raise ValueError("Nonfinite model input/target after chunk gathering")
        self.queue = {
            "start": start,
            "cursor": 0,
            "pair_ids": pair_ids,
            "ids": ids,
            "pairs": inverse.reshape(-1, 2),
            "X": X,
            "Y": Y,
        }
        self.traversal.counter = start + len(pair_ids)
        self.metrics["queues"] += 1
        self.metrics["queue_unique_histories"] += len(ids)
        self.metrics["gather_seconds"] += time.perf_counter() - started

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue is None or self.queue["cursor"] == len(self.queue["pair_ids"]):
            self._fill()
        q = self.queue
        a, b = q["cursor"], min(q["cursor"] + self.batch_pairs, len(q["pair_ids"]))
        unique, inverse = np.unique(q["pairs"][a:b], return_inverse=True)
        pairs = inverse.reshape(-1, 2)
        Y = q["Y"][unique]
        raw = np.sqrt(np.mean((Y[pairs[:, 0]] - Y[pairs[:, 1]]) ** 2, axis=1))
        ids = q["ids"][unique]
        pair_ids = q["pair_ids"][a:b].copy()
        result = PairBatch(
            q["X"][unique],
            Y,
            pairs,
            raw / self.preprocessing["target"]["value"],
            raw,
            tuple((self.index.records[s]["id"], int(t)) for s, t in ids),
            pair_ids,
        )
        q["cursor"] = b
        self.delivered = q["start"] + b
        self.metrics["pairs"] += b - a
        self.metrics["raw_distance_sum"] += float(raw.sum())
        self.metrics["same_session_pairs"] += int(
            np.sum(ids[pairs[:, 0], 0] == ids[pairs[:, 1], 0])
        )
        np.add.at(self.session_pairs, ids[pairs.ravel(), 0], 1)
        return result

    def report(self):
        reader_metrics = [reader.metrics for reader in self.readers.values()]
        return {
            **self.metrics,
            "delivered_counter": self.delivered,
            "pair_universe": self.index.total,
            "sessions_seen": int(np.count_nonzero(self.session_pairs)),
            "session_endpoint_counts": self.session_pairs.tolist(),
            "reader_cache_bytes_per_worker": self.reader_cache_bytes,
            "queue_budget_bytes": self.queue_budget,
            "reserve_bytes": self.reserve,
            "memory_budget_bytes": self.memory_bytes,
            "reader_checksum_seconds": sum(
                metrics["checksum_seconds"] for metrics in reader_metrics
            ),
            "reader_npz_load_seconds": sum(
                metrics["npz_load_seconds"] for metrics in reader_metrics
            ),
            "reader_cache_misses": sum(metrics["cache_misses"] for metrics in reader_metrics),
            "verified_chunks": sum(len(reader.verified_chunks) for reader in self.readers.values()),
        }

    def checkpoint(self, path):
        state = self.traversal.state()
        state["counter"] = self.delivered
        body = {
            "version": 1,
            "traversal": state,
            "preprocessing": self.preprocessing["fingerprint"],
            "queue_pairs": self.queue_pairs,
            "batch_pairs": self.batch_pairs,
            "metrics": self.metrics,
            "session_endpoint_counts": self.session_pairs.tolist(),
        }
        body["fingerprint"] = fingerprint(body)
        write_json(path, body)

    def restore(self, path):
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            fingerprint({key: value for key, value in body.items() if key != "fingerprint"})
            != body["fingerprint"]
        ):
            raise ValueError("Corrupt stream checkpoint")
        if (
            body["version"] != 1
            or body["preprocessing"] != self.preprocessing["fingerprint"]
            or body["batch_pairs"] != self.batch_pairs
            or body["queue_pairs"] != self.queue_pairs
        ):
            raise ValueError("Incompatible stream checkpoint")
        self.traversal = PairTraversal.restore(self.index, body["traversal"])
        self.delivered = self.traversal.counter
        offset = self.delivered % self.queue_pairs
        if offset:
            self.traversal.counter -= offset
            self._fill()
            self.queue["cursor"] = offset
        else:
            self.queue = None
        self.metrics = body["metrics"]
        self.session_pairs = np.asarray(body["session_endpoint_counts"], dtype=np.int64)


def open_pair_stream(corpus, root, preprocessing, **kwargs):
    """Select the bounded stream implementation declared by the corpus."""
    stream_type = (
        ChunkPairStream if corpus.get("source_format") == "observation-chunks-v1" else PairStream
    )
    return stream_type(corpus, root, preprocessing, **kwargs)
