"""Shared book observation schema; usable in both Python runtimes."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Observations:
    x: np.ndarray
    mid: np.ndarray
    segment: np.ndarray
    ts_recv: np.ndarray
    ts_event: np.ndarray
    source_row: np.ndarray
    names: tuple[str, ...]
    instrument: str
    synthetic: bool
    clock: str = "receive"

    @property
    def y(self):
        """One-step log returns; segment starts have no predecessor and use zero."""
        values = np.zeros(len(self.mid))
        if len(values) > 1:
            values[1:] = np.where(
                self.segment[1:] == self.segment[:-1],
                np.log(self.mid[1:] / self.mid[:-1]),
                0,
            )
        return values

    def __post_init__(self):
        n = len(self.mid)
        if self.x.shape != (n, len(self.names)):
            raise ValueError("Feature shape does not match rows/schema")
        for values in (self.mid, self.segment, self.ts_recv, self.ts_event, self.source_row):
            if values.shape != (n,):
                raise ValueError("Observation metadata must have one value per row")
        if not np.isfinite(self.x).all() or not np.isfinite(self.mid).all():
            raise ValueError("Observations must be finite")
        if np.any(self.mid <= 0):
            raise ValueError("Mid-prices must be positive")


def feature_names(depth=10):
    return tuple(
        f"{side}_{level}_{field}"
        for side in ("bid", "ask")
        for level in range(1, depth + 1)
        for field in ("price_ticks", "quantity", "orders", "present")
    ) + ("spread_ticks", "elapsed_seconds", "mid_return", "initial")


def feature_row(bids, asks, mid, tick, time, previous_mid, previous_time, depth=10):
    """Each level is (price, visible quantity, visible order count)."""
    if not bids or not asks or tick <= 0 or mid <= 0 or bids[0][0] > asks[0][0]:
        raise ValueError("Need a valid two-sided book and positive price units")
    if previous_time is not None and time < previous_time:
        raise ValueError("Observation clock moved backwards")
    row = []
    for side in (bids, asks):
        for index in range(depth):
            if index < len(side):
                price, quantity, count = side[index]
                if quantity <= 0 or count <= 0:
                    raise ValueError("Present levels require positive volume and order count")
                row.extend(((price - mid) / tick, quantity, count, 1))
            else:
                row.extend((0, 0, 0, 0))
    row.extend(
        (
            (asks[0][0] - bids[0][0]) / tick,
            0 if previous_time is None else (time - previous_time) / 1e9,
            0 if previous_mid is None else np.log(mid / previous_mid),
            int(previous_mid is None),
        )
    )
    return row


def save_observations(path, observations):
    np.savez_compressed(
        path,
        x=observations.x,
        mid=observations.mid,
        y=observations.y,
        segment=observations.segment,
        ts_recv=observations.ts_recv,
        ts_event=observations.ts_event,
        source_row=observations.source_row,
        names=observations.names,
        instrument=observations.instrument,
        synthetic=observations.synthetic,
        clock=observations.clock,
    )


def load_observations(path):
    with np.load(path, allow_pickle=False) as data:
        return Observations(
            data["x"],
            data["mid"],
            data["segment"],
            data["ts_recv"],
            data["ts_event"],
            data["source_row"],
            tuple(data["names"].tolist()),
            str(data["instrument"]),
            bool(data["synthetic"]),
            str(data["clock"]),
        )
