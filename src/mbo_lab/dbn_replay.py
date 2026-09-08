"""Single-contract bounded replay primitives; no acquisition or corpus-wide assumptions."""

import csv
import tempfile
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from bisect import bisect_left, insort

import databento_dbn as dbn
import zstandard as zstd
from databento import Compression
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.enums import BookType

from mbo_lab.corpus import file_hash, write_json
from mbo_lab.data import DataError, inspect_file, load_instrument, summarize_book
from mbo_lab.observation_store import ObservationWriter
from mbo_lab.observations import feature_row


def raw_records(path, metrics, read_bytes=262144):
    decoder = dbn.DBNDecoder(compression=Compression.NONE) # type: ignore
    with Path(path).open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as stream:
        while True:
            started = time.perf_counter()
            chunk = stream.read(read_bytes)
            metrics["read_decompress_seconds"] += time.perf_counter() - started
            if not chunk:
                break
            metrics["decompressed_bytes"] += len(chunk)
            started = time.perf_counter()
            decoder.write(chunk)
            records = decoder.decode()
            metrics["dbn_decode_seconds"] += time.perf_counter() - started
            yield from records
        if decoder.buffer():
            raise DataError("Incomplete DBN record at source end")


def decoded_blocks(
    paths,
    instrument,
    identity,
    metrics,
    scratch,
    *,
    block_records=4096,
    stop_ns=None,
    prefix_file=None,
):
    """Preserve each raw record alongside its SDK delta, including records with no delta.

    The installed SDK exposes file-to-list conversion. Small temporary compressed
    DBN blocks bound that list without implementing a second MBO action decoder.
    This bridge is deliberately measurable and replaceable after the pilot.
    """
    if not 1 <= block_records <= 65536:
        raise ValueError("block_records must be in [1,65536]")
    loader = DatabentoDataLoader()
    global_offset = 0
    for part, path in enumerate(paths):
        iterator = raw_records(path, metrics)
        metadata = next(iterator)
        if (
            not isinstance(metadata, dbn.Metadata)
            or str(metadata.schema) != "mbo"
            or metadata.dataset != "GLBX.MDP3"
        ):
            raise DataError("Expected MBO metadata")
        header, pending, finished = bytes(metadata), [], False
        part_start = global_offset
        reference = None
        if prefix_file is not None:
            if len(paths) != 1:
                raise ValueError("Reference prefix is a single-file pilot check")
            reference = zstd.ZstdCompressor(level=1).stream_writer(Path(prefix_file).open("wb"))
            reference.write(header)

        def convert(records):
            started = time.perf_counter()
            temporary = Path(scratch) / "decode.dbn.zst"
            temporary.write_bytes(
                zstd.ZstdCompressor(level=1).compress(
                    header + b"".join(bytes(record) for _, record in records)
                )
            )
            deltas = loader.from_dbn_file(
                temporary,
                instrument_id=instrument.id,
                price_precision=instrument.price_precision,
                include_trades=False,
            )
            result, position = [], 0
            for offset, record in records:
                delta = None
                if str(record.action) in "ACMR":
                    if position == len(deltas) or deltas[position].sequence != record.sequence:
                        raise DataError("Bounded SDK delta alignment failed")
                    delta = deltas[position]
                    position += 1
                result.append((offset, record, delta))
            if position != len(deltas):
                raise DataError("Unexpected SDK delta count")
            metrics["native_conversion_seconds"] += time.perf_counter() - started
            return result

        try:
            for record in iterator:
                if not isinstance(record, dbn.MBOMsg):
                    raise DataError(f"Unexpected record type {type(record).__name__}")
                if (record.publisher_id, record.instrument_id) != identity:
                    raise DataError("Instrument changed; schedule a separate logical timeline")
                if record.flags & (dbn.F_MBP | dbn.F_TOB):
                    raise DataError("Aggregated records cannot reconstruct L3")
                if reference is not None:
                    reference.write(bytes(record))
                pending.append((global_offset, record))
                global_offset += 1
                metrics["raw_records"] += 1
                finished = bool(
                    stop_ns is not None and record.ts_recv >= stop_ns and record.flags & dbn.F_LAST
                )
                if len(pending) == block_records or finished:
                    yield from convert(pending)
                    pending = []
                if finished:
                    break
            if pending:
                yield from convert(pending)
        finally:
            iterator.close()
            if reference is not None:
                reference.close()
            metrics.setdefault("source_parts", []).append(
                {"part": part, "start_record": part_start, "stop_record": global_offset}
            )
        if finished:
            break

# @profile # type: ignore --dev line-profiler
def export_replay(
    paths,
    definitions,
    output,
    *,
    timeline_id,
    stop_ns,
    block_records=4096,
    chunk_rows=65536,
    memory_sample=None,
    reference_prefix=None,
):
    paths = [Path(p).resolve() for p in paths]
    definitions, output = Path(definitions).resolve(), Path(output).resolve()
    if not paths or stop_ns <= 0:
        raise ValueError("Need ordered paths and an explicit pilot stop timestamp")
    started = time.perf_counter()
    instrument = load_instrument(definitions)
    info = inspect_file(definitions)
    identity = tuple(info["identities"][0])
    source_inventory = [
        {"path": str(p), "sha256": file_hash(p), "bytes": p.stat().st_size} for p in paths
    ]
    writer = ObservationWriter(output, timeline_id, str(instrument.id), chunk_rows=chunk_rows)
    book = OrderBook(instrument.id, BookType.L3_MBO)
    metrics = {
        k: 0.0
        for k in (
            "read_decompress_seconds",
            "dbn_decode_seconds",
            "native_conversion_seconds",
            "book_apply_seconds",
            "feature_seconds",
            "chunk_write_seconds",
            "shadow_check_seconds",
        )
    }
    metrics.update(
        {
            "raw_records": 0,
            "decompressed_bytes": 0,
            "snapshot_records": 0,
            "bad_clock_events": 0,
            "invalid_book_events": 0,
            "shadow_checks": 0,
            "max_shadow_orders": 0,
        }
    )
    actions, shadow, memory = Counter(), {}, []

    # price -> (total_quantity, order_count)
    levels = {
        "BUY": {},
        "SELL": {},
    }

    level_prices = {
    "BUY": [],
    "SELL": [],
    }

    def add_level(side, price, quantity):
        side_levels = levels[side]

        current = side_levels.get(price)

        if current is None:
            side_levels[price] = (quantity, 1)
            insort(level_prices[side], price)
        else:
            amount, count = current
            side_levels[price] = (
                amount + quantity,
                count + 1,
            )


    def remove_level(side, price, quantity):
        side_levels = levels[side]
        current = side_levels.get(price)

        if current is None:
            raise DataError(
                f"Missing aggregated level while removing {side} {price}"
            )

        amount, count = current

        if count == 1:
            del side_levels[price]

            prices = level_prices[side]
            index = bisect_left(prices, price)

            if index == len(prices) or prices[index] != price:
                raise DataError("Price missing from ordered level index")

            prices.pop(index)

        elif count > 1:
            side_levels[price] = (
                amount - quantity,
                count - 1,
            )

        else:
            raise DataError("Aggregated level order count became invalid")

    initialized, snapshot, first, segment = False, False, True, 0
    awaiting_initial_snapshot = False
    initial_clear_event_open = False
    previous_mid = previous_time = None
    previous_receive = None
    last_flags = 0

    def check_shadow():
        t = time.perf_counter()
        expected_levels  = {"BUY": {}, "SELL": {}}
        for side, price, quantity in shadow.values():
            amount, count = expected_levels [side].get(price, (0, 0))
            expected_levels [side][price] = amount + quantity, count + 1
        state = summarize_book(book, 10)
        for side, name in (("BUY", "bids"), ("SELL", "asks")):
            expected = [
                (p, *expected_levels [side][p]) for p in sorted(expected_levels [side], reverse=side == "BUY")[:10]
            ]
            actual = [(float(v["price"]), float(v["size"]), v["orders"]) for v in state[name]]
            if actual != expected:
                raise DataError("Order-ID shadow book disagrees with Nautilus")
        metrics["shadow_checks"] += 1
        metrics["shadow_check_seconds"] += time.perf_counter() - t

    with tempfile.TemporaryDirectory(dir=output.parent, prefix="dbn-block-") as temporary:
        records = decoded_blocks(
            paths,
            instrument,
            identity,
            metrics,
            temporary,
            block_records=block_records,
            stop_ns=stop_ns,
            prefix_file=reference_prefix,
        )
        #####################################################################
        for offset, record, delta in records:######### HOT LOOP #############
            action = str(record.action)
            actions[action] += 1
            last_flags = record.flags
            if first:
                if action != "R":
                    raise DataError("Replay must begin with a book CLEAR")

                if not record.flags & dbn.F_SNAPSHOT:
                    # CME weekly-session start: channel reset first,
                    # CME MBO snapshot follows.
                    awaiting_initial_snapshot = True
                    initial_clear_event_open = True

                first = False
            if record.ts_recv != 2**64 - 1:
                if previous_receive is not None and record.ts_recv < previous_receive:
                    raise DataError("Receive clock moved backwards across source parts")
                previous_receive = record.ts_recv
            if action == "R":
                segment += 1
                previous_mid = previous_time = None
            if record.flags & dbn.F_SNAPSHOT:
                snapshot = True
                initialized = False
                metrics["snapshot_records"] += 1
            if delta is not None:
                t = time.perf_counter()
                book.apply_delta(delta)
                metrics["book_apply_seconds"] += time.perf_counter() - t
                name = delta.action.name
                order = delta.order
                order_id = order.order_id

                if name == "CLEAR":
                    shadow.clear()

                    levels["BUY"].clear()
                    levels["SELL"].clear()

                    level_prices["BUY"].clear()
                    level_prices["SELL"].clear()

                elif name == "DELETE":
                    old = shadow.pop(order_id, None)

                    if old is not None:
                        old_side, old_price, old_quantity = old
                        remove_level(old_side, old_price, old_quantity)

                else:
                    # UPDATE/MODIFY may replace an existing order,
                    # so remove its previous contribution first.
                    old = shadow.get(order_id)

                    if old is not None:
                        old_side, old_price, old_quantity = old
                        remove_level(old_side, old_price, old_quantity)

                    new_side = order.side.name
                    new_price = float(order.price)
                    new_quantity = float(order.size)

                    shadow[order_id] = (
                        new_side,
                        new_price,
                        new_quantity,
                    )

                    add_level(
                        new_side,
                        new_price,
                        new_quantity,
                    )

                metrics["max_shadow_orders"] = max(metrics["max_shadow_orders"], len(shadow))
            if not record.flags & dbn.F_LAST:
                continue

            if snapshot:
                check_shadow()
                snapshot, initialized = False, True
                awaiting_initial_snapshot = False
                initial_clear_event_open = False
                continue

            if awaiting_initial_snapshot:
                # Permit the initial unflagged Sunday channel-reset event.
                if initial_clear_event_open:
                    initial_clear_event_open = False
                    continue

                # We must not build state from ordinary live order traffic
                # before receiving the CME weekly-session snapshot.
                if action in {"A", "M", "C"}:
                    raise DataError(
                        "Live MBO event arrived before initial weekly-session snapshot"
                    )

                # T/F/N do not alter book state; keep waiting for snapshot.
                continue

            if not initialized:
                raise DataError("No complete initialization snapshot")
            if record.ts_recv >= stop_ns:
                break
            if record.flags & dbn.F_BAD_TS_RECV or record.ts_recv == 2**64 - 1:
                metrics["bad_clock_events"] += 1
                segment += 1
                previous_mid = previous_time = None
                continue

            t = time.perf_counter()
            try:
                bid_prices = reversed(level_prices["BUY"][-10:])
                ask_prices = level_prices["SELL"][:10]

                bids = [
                    (
                        price,
                        levels["BUY"][price][0],
                        levels["BUY"][price][1],
                    )
                    for price in bid_prices
                ]

                asks = [
                    (
                        price,
                        levels["SELL"][price][0],
                        levels["SELL"][price][1],
                    )
                    for price in ask_prices
                ]
            except (ValueError, RuntimeError):
                metrics["invalid_book_events"] += 1
                segment += 1
                previous_mid = previous_time = None
                continue

            if not bids or not asks or bids[0][0] > asks[0][0]:
                metrics["invalid_book_events"] += 1
                segment += 1
                previous_mid = previous_time = None
                continue

            mid = (bids[0][0] + asks[0][0]) / 2

            row = feature_row(
                bids,
                asks,
                mid,
                float(instrument.price_increment),
                record.ts_recv,
                previous_mid,
                previous_time,
                clock="receive",
            )

            metrics["feature_seconds"] += (
                time.perf_counter() - t
            )
            t = time.perf_counter()
            writer.append(row, mid, segment, record.ts_recv, record.ts_event, offset)
            metrics["chunk_write_seconds"] += time.perf_counter() - t
            previous_mid, previous_time = mid, record.ts_recv
            if writer.rows % 69420 == 0:
                check_shadow()
            if writer.rows % chunk_rows == 0:
                sample = {
                    "rows": writer.rows,
                    "raw_records": metrics["raw_records"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_process_bytes": memory_sample() if memory_sample else None,
                }
                memory.append(sample)
                print(f"Replay: {writer.rows:,} rows, {sample['elapsed_seconds']:.1f}s", flush=True)
        records.close()
    if not last_flags & dbn.F_LAST:
        raise DataError("Source ended mid-event")
    check_shadow()
    t = time.perf_counter()
    writer.flush()
    metrics["chunk_write_seconds"] += time.perf_counter() - t
    manifest = writer.finish(
        {
            "source": "Databento GLBX.MDP3 MBO",
            "sources": source_inventory,
            "definitions": {"path": str(definitions), "sha256": file_hash(definitions)},
            "stop_ns_exclusive": stop_ns,
            "source_row_unit": "zero-based MBO ordinal across ordered parts",
            "snapshot_policy": "initialize only; do not emit synthetic snapshot observations",
            "bad_receive_policy": "break sample continuity and omit bad event",
            "tick_size": str(instrument.price_increment),
            "runtime": {p: version(p) for p in ("databento-dbn", "nautilus_trader", "numpy")},
        }
    )
    metrics.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "rows": writer.rows,
            "actions": dict(actions),
            "block_records": block_records,
            "chunk_rows": chunk_rows,
            "output_bytes": sum(p.stat().st_size for p in output.iterdir()),
            "peak_process_bytes": memory_sample() if memory_sample else None,
            "chunk_progress": memory,
            "read_timing_scope": (
                "filesystem read plus Zstandard decompression; DBN decoding separate"
            ),
        } # type: ignore
    ) # type: ignore
    write_json(output / "metrics.json", metrics)
    if memory:
        with (output / "memory-progress.csv").open("w", newline="") as stream:
            table = csv.DictWriter(stream, fieldnames=list(memory[0]))
            table.writeheader()
            table.writerows(memory)
    return manifest, metrics
