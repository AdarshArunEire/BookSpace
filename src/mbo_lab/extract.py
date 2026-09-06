"""Completed-book observations. No network access or synthetic fallback."""

import databento_dbn as dbn
import numpy as np
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.enums import BookType

from mbo_lab.data import DataError, _records, load_deltas, summarize_book
from mbo_lab.observations import Observations, feature_names, feature_row


def extract_observations(mbo_path, definitions_path, depth=10):
    """Copy top-level features at raw F_LAST boundaries; resets break segments.

    One file/instrument per table. Invalid completed states break continuity.
    Explicit session boundaries must be supplied separately to sample indexing.
    """
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    instrument, deltas = load_deltas(mbo_path, definitions_path)
    book = OrderBook(instrument.id, BookType.L3_MBO)
    tick = float(instrument.price_increment)
    names = feature_names(depth)
    rows, mids, segments, recv, event, offsets = [], [], [], [], [], []
    delta_iter = iter(deltas)
    segment, previous_mid, previous_time = 0, None, None
    for offset, record in enumerate(_records(mbo_path)):
        if not isinstance(record, dbn.MBOMsg):
            continue
        action = str(record.action)
        if action == "R":
            segment += 1
            previous_mid = previous_time = None
        if action in "ACMR":
            delta = next(delta_iter, None)
            if delta is None or delta.sequence != record.sequence:
                raise DataError("Raw record and Nautilus delta alignment failed")
            book.apply_delta(delta)
        if not record.flags & dbn.F_LAST:
            continue
        try:
            state = summarize_book(book, depth)
        except (ValueError, RuntimeError):
            segment += 1
            previous_mid = previous_time = None
            continue
        bid, ask = state["best_bid"], state["best_ask"]
        if bid is None or ask is None or float(bid) > float(ask):
            segment += 1
            previous_mid = previous_time = None
            continue
        mid = (float(bid) + float(ask)) / 2
        if mid <= 0:
            raise DataError("Log returns require positive mid-prices")
        if previous_time is not None and record.ts_recv < previous_time:
            raise DataError("Receive time moved backwards within a segment")
        sides = [
            [(float(v["price"]), float(v["size"]), v["orders"]) for v in state[side]]
            for side in ("bids", "asks")
        ]
        row = feature_row(*sides, mid, tick, record.ts_recv, previous_mid, previous_time, depth)
        rows.append(row)
        mids.append(mid)
        segments.append(segment)
        recv.append(record.ts_recv)
        event.append(record.ts_event)
        offsets.append(offset)
        previous_mid, previous_time = mid, record.ts_recv
    if next(delta_iter, None) is not None:
        raise DataError("Unconsumed book deltas")
    return Observations(
        np.asarray(rows, dtype=float).reshape(-1, len(names)),
        np.asarray(mids),
        np.asarray(segments),
        np.asarray(recv, dtype=np.uint64),
        np.asarray(event, dtype=np.uint64),
        np.asarray(offsets),
        names,
        str(instrument.id),
        str(instrument.id).startswith("SYNTH_"),
    )
