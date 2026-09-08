"""Implicit unordered pair indexing and a constant-state, nonrepeating traversal."""

import bisect
import hashlib
from collections import OrderedDict

import numpy as np

from mbo_lab.corpus import expand_ranges


def count_range_pairs(ranges, gap):
    """Count separated unordered pairs without expanding a logical timeline."""
    total = 0
    for a, b in ranges:
        for c, d in ranges:
            full = max(0, min(b, c - gap + 1) - a)
            total += full * (d - c)
            start, stop = max(a, c - gap + 1), min(b, d - gap)
            n = max(0, stop - start)
            total += n * (d - gap) - n * (start + stop - 1) // 2
    return total


class PairIndex:
    def __init__(self, corpus, split="train"):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        self.corpus = corpus
        self.split = split
        self.records = [s for s in corpus["sessions"] if s["split"] == split]
        self.gap = corpus["history"] + corpus["horizon"]
        self.anchor_prefix = [0]
        for record in self.records:
            self.anchor_prefix.append(self.anchor_prefix[-1] + record["anchors"])
        self.prefix = [0]
        for i, record in enumerate(self.records):
            suffix = self.anchor_prefix[-1] - self.anchor_prefix[i + 1]
            self.prefix.append(
                self.prefix[-1] + record["within_pairs"] + record["anchors"] * suffix
            )
        self.total = self.prefix[-1]
        if self.total < 1 or self.total > np.iinfo(np.int64).max:
            raise ValueError("Pair universe must fit positive signed int64")
        self._cache = OrderedDict()

    def _arrays(self, session):
        if session not in self._cache:
            anchors = expand_ranges(self.records[session]["anchor_ranges"])
            first = np.searchsorted(anchors, anchors + self.gap)
            prefix = np.r_[0, np.cumsum(len(anchors) - first)]
            self._cache[session] = anchors, first, prefix
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(session)
        return self._cache[session]

    def _anchor(self, session, rank):
        for start, stop in self.records[session]["anchor_ranges"]:
            if rank < stop - start:
                return start + rank
            rank -= stop - start
        raise IndexError("Anchor rank outside session")

    def unrank(self, rank):
        rank = int(rank)
        if not 0 <= rank < self.total:
            raise IndexError("Pair rank outside universe")
        session = bisect.bisect_right(self.prefix, rank) - 1
        local = rank - self.prefix[session]
        within = self.records[session]["within_pairs"]
        if local < within:
            anchors, first, prefix = self._arrays(session)
            left = int(np.searchsorted(prefix, local, side="right") - 1)
            right = int(first[left] + local - prefix[left])
            return (session, int(anchors[left])), (session, int(anchors[right]))
        suffix = self.anchor_prefix[-1] - self.anchor_prefix[session + 1]
        left, later = divmod(local - within, suffix)
        global_anchor = self.anchor_prefix[session + 1] + later
        other = bisect.bisect_right(self.anchor_prefix, global_anchor) - 1
        return (session, self._anchor(session, left)), (
            other,
            self._anchor(other, global_anchor - self.anchor_prefix[other]),
        )

    def rank(self, left, right):
        left, right = sorted((tuple(left), tuple(right)))
        s, a = left
        r, b = right
        if not (0 <= s <= r < len(self.records)):
            raise ValueError("Invalid session indices")
        anchors, first, prefix = self._arrays(s)
        i = int(np.searchsorted(anchors, a))
        other = self._arrays(r)[0]
        j = int(np.searchsorted(other, b))
        if i == len(anchors) or anchors[i] != a or j == len(other) or other[j] != b:
            raise ValueError("Ineligible anchor")
        if s == r:
            if b - a < self.gap:
                raise ValueError("Overlapping episode pair")
            return int(self.prefix[s] + prefix[i] + j - first[i])
        suffix = self.anchor_prefix[-1] - self.anchor_prefix[s + 1]
        return int(
            self.prefix[s]
            + self.records[s]["within_pairs"]
            + i * suffix
            + self.anchor_prefix[r]
            - self.anchor_prefix[s + 1]
            + j
        )


class PairTraversal:
    """Eight-round Feistel permutation with cycle walking over [0, total).

    This is a seeded pseudorandom order, not a uniform draw from all P! orders.
    Bijection, rather than a seen-pair set, guarantees no repetition. Version is
    persisted because changing the permutation would invalidate resume.
    """

    version = "feistel8-blake2b-v1"

    def __init__(self, index, seed=0, counter=0):
        if not isinstance(seed, int) or seed < 0 or not 0 <= counter <= index.total:
            raise ValueError("Invalid traversal seed/counter")
        self.index, self.seed, self.counter = index, seed, counter
        self.half = max(1, ((index.total - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.half) - 1
        identity = f"{self.version}:{index.corpus['fingerprint']}:{index.split}:{seed}"
        self.keys = [hashlib.sha256(f"{identity}:{r}".encode()).digest() for r in range(8)]

    def _permute(self, value):
        left, right = value >> self.half, value & self.mask
        for key in self.keys:
            hashed = int.from_bytes(
                hashlib.blake2b(right.to_bytes(8, "little"), key=key, digest_size=8).digest(),
                "little",
            )
            left, right = right, left ^ (hashed & self.mask)
        return (left << self.half) | right

    def at(self, counter):
        if not 0 <= counter < self.index.total:
            raise IndexError("Traversal exhausted")
        value = self._permute(counter)
        while value >= self.index.total:
            value = self._permute(value)
        return value

    def take(self, count):
        if not isinstance(count, int) or count < 0:
            raise ValueError("Count must be nonnegative")
        end = min(self.counter + count, self.index.total)
        result = np.fromiter((self.at(i) for i in range(self.counter, end)), dtype=np.int64)
        self.counter = end
        return result

    def state(self):
        return {
            "version": self.version,
            "corpus": self.index.corpus["fingerprint"],
            "split": self.index.split,
            "total": self.index.total,
            "seed": self.seed,
            "counter": self.counter,
        }

    @classmethod
    def restore(cls, index, state):
        expected = (cls.version, index.corpus["fingerprint"], index.split, index.total)
        if tuple(state[k] for k in ("version", "corpus", "split", "total")) != expected:
            raise ValueError("Incompatible traversal checkpoint")
        return cls(index, state["seed"], state["counter"])
