"""Causal windows, cumulative futures and train-only feature transforms."""

from dataclasses import dataclass

import numpy as np


def valid_anchors(observations, history=256, horizon=128, start=0, stop=None):
    """Restrict complete episodes to the half-open split [start, stop)."""
    stop = len(observations.mid) if stop is None else stop
    if history < 1 or horizon < 1 or not 0 <= start <= stop <= len(observations.mid):
        raise ValueError("Invalid window lengths or split bounds")
    anchors = np.arange(start + history - 1, stop - horizon, dtype=np.int64)
    # Count all transitions, including a segment label that later repeats.
    breaks = np.r_[0, np.cumsum(np.diff(observations.segment) != 0)]
    return anchors[breaks[anchors + horizon] == breaks[anchors - history + 1]]


def batch(observations, anchors, history=256, horizon=128, start=0, stop=None):
    anchors = np.asarray(anchors, dtype=np.int64)
    eligible = valid_anchors(observations, history, horizon, start, stop)
    if anchors.ndim != 1 or not np.isin(anchors, eligible).all():
        raise ValueError("Anchor crosses a segment/split or lacks a complete episode")
    past = anchors[:, None] + np.arange(1 - history, 1)
    future = anchors[:, None] + np.arange(1, horizon + 1)
    return observations.x[past].copy(), np.log(
        observations.mid[future] / observations.mid[anchors, None]
    )


@dataclass
class FeatureScale:
    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray
    training_rows: np.ndarray

    @staticmethod
    def _prepare(x, names):
        values = np.asarray(x, dtype=float).copy()
        valid = np.ones_like(values, dtype=bool)
        binary = np.zeros(len(names), dtype=bool)
        for col, name in enumerate(names):
            binary[col] = name.endswith("_present") or name == "initial"
            if name.endswith(("_quantity", "_orders")) or name == "elapsed_seconds":
                if np.any(values[:, col] < 0):
                    raise ValueError("Counts, quantities and elapsed times must be nonnegative")
                values[:, col] = np.log1p(values[:, col])
            if name.endswith(("_price_ticks", "_quantity", "_orders")):
                prefix = (
                    name.rsplit("_", 2)[0]
                    if name.endswith("_price_ticks")
                    else name.rsplit("_", 1)[0]
                )
                valid[:, col] = values[:, names.index(prefix + "_present")] == 1
        return values, valid, binary

    @classmethod
    def fit(cls, observations, training_rows):
        rows = np.unique(np.asarray(training_rows, dtype=np.int64))
        if not len(rows) or rows.min() < 0 or rows.max() >= len(observations.mid):
            raise ValueError("Need valid training row indices")
        values, valid, binary = cls._prepare(observations.x[rows], observations.names)
        mean, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])
        constant = np.zeros(values.shape[1], dtype=bool)
        for col in range(values.shape[1]):
            if binary[col]:
                continue
            present = values[valid[:, col], col]
            if len(present):
                mean[col] = present.mean()
                std = present.std()
                if std > 0:
                    scale[col] = std
                else:
                    constant[col] = True
            else:
                constant[col] = True
        return cls(observations.names, mean, scale, constant, rows)

    def transform(self, observations):
        if observations.names != self.names:
            raise ValueError("Feature schema mismatch")
        values, valid, _ = self._prepare(observations.x, self.names)
        return np.where(valid, (values - self.mean) / self.scale, 0)
