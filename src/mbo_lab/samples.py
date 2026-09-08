"""Causal windows, cumulative futures and train-only feature transforms."""

from dataclasses import dataclass

import numpy as np


def eligible_pairs(anchors, history=256, horizon=128):
    """Return unique, non-overlapping unordered pairs of anchor positions."""
    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.ndim != 1 or history < 1 or horizon < 1:
        raise ValueError("Invalid anchors/window lengths")
    i, j = np.triu_indices(len(anchors), k=1)
    keep = np.abs(anchors[i] - anchors[j]) >= history + horizon
    return np.column_stack((i[keep], j[keep]))


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
    time_delta_tau: float

    @staticmethod
    def _prepare(x, names, time_delta_tau):
        values = np.asarray(x, dtype=float).copy()
        valid = np.ones_like(values, dtype=bool)
        protected = np.zeros(len(names), dtype=bool)
        if not np.isfinite(time_delta_tau) or time_delta_tau <= 0:
            raise ValueError("time_delta_tau must be finite and positive")
        for col, name in enumerate(names):
            protected[col] = (
                name.endswith("_present")
                or name == "initial"
                or name in ("tod_sin", "tod_cos")
            )
            if name.endswith(("_quantity", "_orders")) or name == "time_delta_seconds":
                if np.any(values[:, col] < 0):
                    raise ValueError("Counts, quantities and time deltas must be nonnegative")
                if name == "time_delta_seconds":
                    values[:, col] = np.log1p(values[:, col] / time_delta_tau)
                else:
                    values[:, col] = np.log1p(values[:, col])
            if name.endswith(("_price_ticks", "_quantity", "_orders")):
                prefix = (
                    name.rsplit("_", 2)[0]
                    if name.endswith("_price_ticks")
                    else name.rsplit("_", 1)[0]
                )
                valid[:, col] = values[:, names.index(prefix + "_present")] == 1
        return values, valid, protected

    @staticmethod
    def _fit_time_delta_tau(x, names):
        try:
            column = names.index("time_delta_seconds")
        except ValueError as exc:
            raise ValueError("Feature schema must contain time_delta_seconds") from exc
        deltas = np.asarray(x[:, column], dtype=float)
        if not np.isfinite(deltas).all() or np.any(deltas < 0):
            raise ValueError("Time deltas must be finite and nonnegative")
        positive = deltas[deltas > 0]
        if not len(positive):
            raise ValueError("Training rows must contain a positive time delta")
        return float(np.median(positive))

    @classmethod
    def fit(cls, observations, training_rows):
        rows = np.unique(np.asarray(training_rows, dtype=np.int64))
        if not len(rows) or rows.min() < 0 or rows.max() >= len(observations.mid):
            raise ValueError("Need valid training row indices")
        time_delta_tau = cls._fit_time_delta_tau(observations.x[rows], observations.names)
        values, valid, protected = cls._prepare(
            observations.x[rows], observations.names, time_delta_tau
        )
        mean, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])
        constant = np.zeros(values.shape[1], dtype=bool)
        for col in range(values.shape[1]):
            if protected[col]:
                continue
            present = values[valid[:, col], col]
            if len(present):
                mean[col] = present.mean()
                std = present.std()
                if std > 0:
                    scale[col] = std
                else:
                    constant[col] = True
        return cls(observations.names, mean, scale, constant, rows, time_delta_tau)

    def transform(self, observations):
        if observations.names != self.names:
            raise ValueError("Feature schema mismatch")
        values, valid, _ = self._prepare(observations.x, self.names, self.time_delta_tau)
        return np.where(valid, (values - self.mean) / self.scale, 0)
