"""Scalar path distances and explicitly fitted training-target units."""

from dataclasses import dataclass

import numpy as np


def eligible_pairs(anchors, history=256, horizon=128):
    """Unique disjoint episode pairs within a batch from one observation table."""
    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.ndim != 1 or history < 1 or horizon < 1:
        raise ValueError("Invalid anchors/window lengths")
    i, j = np.triu_indices(len(anchors), k=1)
    keep = np.abs(anchors[i] - anchors[j]) >= history + horizon
    return np.column_stack((i[keep], j[keep]))


def rms_distance(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim < 1 or left.shape[-1] == 0:
        raise ValueError("Distances require matching, nonempty vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Distance inputs must be finite")
    return np.sqrt(np.mean((left - right) ** 2, axis=-1))


@dataclass(frozen=True)
class TargetScale:
    value: float
    pair_count: int
    zero_fraction: float
    seed: int

    @classmethod
    def fit(cls, training_distances, seed=0):
        values = np.asarray(training_distances, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Need finite distances from eligible training pairs")
        if np.any(values < 0) or not np.any(values > 0):
            raise ValueError("Training distances must be nonnegative and not all zero")
        return cls(
            float(np.median(values[values > 0])), len(values), float(np.mean(values == 0)), seed
        )

    def transform(self, distances):
        values = np.asarray(distances, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Distances must be finite and nonnegative")
        return values / self.value


def pair_targets(futures, pairs, scale=None):
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("No eligible pairs or invalid pair shape")
    if np.any(pairs < 0) or np.any(pairs >= len(futures)):
        raise ValueError("Pair index out of range")
    distances = rms_distance(futures[pairs[:, 0]], futures[pairs[:, 1]])
    return distances if scale is None else scale.transform(distances)
