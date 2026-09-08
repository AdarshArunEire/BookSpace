"""Historical MBO acquisition and local book reconstruction. Network access is explicit."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

import databento as db
import databento_dbn as dbn
import zstandard as zstd
from databento import Compression
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.enums import BookType


class DataError(ValueError):
    """Invalid, incomplete, or incompatible input; no book should be used."""


def _utc(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataError(f"Invalid UTC timestamp: {value!r}") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise DataError("Request times must include UTC: use a trailing Z or +00:00")
    return result.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolution_end_date(end: datetime):
    """Return the exclusive date needed to resolve every UTC date touched by [start, end)."""
    return (end - timedelta(microseconds=1)).date() + timedelta(days=1)


@dataclass(frozen=True)
class Request:
    symbol: str
    start: str
    end: str
    dataset: str = "GLBX.MDP3"
    stype_in: str = "raw_symbol"

    def __post_init__(self):
        start, end = _utc(self.start), _utc(self.end)
        if (
            not self.symbol
            or self.symbol == "ALL_SYMBOLS"
            or "," in self.symbol
            or self.dataset != "GLBX.MDP3"
        ):
            raise DataError("This initial pipeline requires one GLBX.MDP3 futures symbol")
        if self.stype_in not in {"raw_symbol", "instrument_id", "continuous"}:
            raise DataError("stype_in must be raw_symbol, instrument_id, or continuous")
        if self.stype_in == "instrument_id" and not self.symbol.isdigit():
            raise DataError("instrument_id requests must use a numeric symbol")
        if end <= start:
            raise DataError("Choose a positive interval")
        if start.time().isoformat() != "00:00:00":
            raise DataError(
                "Start at 00:00:00Z so the range can be partitioned at book reset boundaries"
            )
        object.__setattr__(self, "start", _iso_utc(start))
        object.__setattr__(self, "end", _iso_utc(end))

    @classmethod
    def from_toml(cls, path: str | Path) -> Request:
        try:
            with Path(path).open("rb") as stream:
                return cls(**tomllib.load(stream))
        except (OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
            raise DataError(f"Invalid request config: {path}") from exc


def _parameters(request: Request, schema: str) -> dict:
    values = asdict(request)
    values["symbols"] = [values.pop("symbol")]
    return {**values, "schema": schema}


def discover(dataset: str = "GLBX.MDP3", *, client=None) -> dict:
    client = client if client is not None else db.Historical()
    return {
        "datasets": client.metadata.list_datasets(),
        "schemas": client.metadata.list_schemas(dataset=dataset),
        "range": client.metadata.get_dataset_range(dataset=dataset),
        "publishers": client.metadata.list_publishers(),
    }


def _mapping_intervals(request: Request, *, client) -> list[tuple[datetime, datetime, str]]:
    """Resolve a continuous symbol over the whole requested range, including roll boundaries."""
    start, end = _utc(request.start), _utc(request.end)
    response = client.symbology.resolve(
        dataset=request.dataset,
        symbols=[request.symbol],
        stype_in=request.stype_in,
        stype_out="instrument_id",
        start_date=start.date().isoformat(),
        end_date=_resolution_end_date(end).isoformat(),
    )
    try:
        items = response["result"][request.symbol]
    except (KeyError, TypeError) as exc:
        raise DataError(f"Databento returned no usable mapping for {request.symbol}") from exc

    intervals = []
    for item in items:
        try:
            left = datetime.fromisoformat(str(item["d0"])).replace(tzinfo=timezone.utc)
            right = datetime.fromisoformat(str(item["d1"])).replace(tzinfo=timezone.utc)
            symbol = str(item["s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"Invalid symbology interval for {request.symbol}: {item!r}") from exc
        left, right = max(left, start), min(right, end)
        if left < right:
            intervals.append((left, right, symbol))

    intervals.sort(key=lambda item: item[0])
    cursor = start
    for left, right, _ in intervals:
        if left > cursor:
            raise DataError(f"Databento symbology has a gap beginning at {_iso_utc(cursor)}")
        cursor = max(cursor, right)
    if cursor < end:
        raise DataError(f"Databento symbology does not cover the range through {_iso_utc(end)}")
    return intervals


def _weekday_midnights(start: datetime, end: datetime) -> list[datetime]:
    """UTC weekday midnights inside (start, end); CME historical MBO snapshots occur here."""
    point = datetime.combine(start.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
    result = []
    while point < end:
        if point.weekday() <= 4:
            result.append(point)
        point += timedelta(days=1)
    return result


def plan(request: Request, *, client=None) -> list[Request]:
    """Partition a range into independently reconstructible, single-instrument cache chunks."""
    client = client if client is not None else db.Historical()
    start, end = _utc(request.start), _utc(request.end)

    if request.stype_in == "continuous":
        mappings = _mapping_intervals(request, client=client)
    else:
        mappings = [(start, end, request.symbol)]

    chunks = []
    for left, right, symbol in mappings:
        cuts = [left, *_weekday_midnights(left, right), right]
        for chunk_start, chunk_end in zip(cuts, cuts[1:]):
            if chunk_start >= chunk_end:
                continue
            chunks.append(
                Request(
                    symbol=symbol,
                    start=_iso_utc(chunk_start),
                    end=_iso_utc(chunk_end),
                    dataset=request.dataset,
                    stype_in="instrument_id"
                    if request.stype_in == "continuous"
                    else request.stype_in,
                )
            )
    if not chunks:
        raise DataError("The requested range produced no downloadable chunks")
    return chunks


def resolve(request: Request, *, client) -> Request:
    """Resolve only when the full request maps to exactly one chunk; kept for callers/tests."""
    chunks = plan(request, client=client)
    if len(chunks) != 1:
        raise DataError("Request spans multiple book/cache chunks; use plan() or estimate()")
    return chunks[0]


def estimate(request: Request, *, client=None, mode="historical") -> dict:
    """Quote the whole range; Databento resolves continuous symbols server-side."""
    client = client if client is not None else db.Historical()
    costs = {
        schema: float(client.metadata.get_cost(**_parameters(request, schema)))
        for schema in ("definition", "mbo")
    }
    if any(not math.isfinite(cost) or cost < 0 for cost in costs.values()):
        raise DataError("Provider returned an invalid cost estimate")
    sizes = {
        schema: client.metadata.get_billable_size(**_parameters(request, schema))
        for schema in ("definition", "mbo")
    }
    if any(not isinstance(size, int) or size < 0 for size in sizes.values()):
        raise DataError("Provider returned an invalid billable size")
    prices = client.metadata.list_unit_prices(dataset=request.dataset)
    rates = next(
        (item["unit_prices"] for item in prices if item["mode"] == mode),
        {},
    )
    return {
        "request": asdict(request),
        "mode": mode,
        "usd": costs,
        "billable_bytes": sizes,
        "usd_per_gb": {schema: rates.get(schema) for schema in costs},
        "total_usd": sum(costs.values()),
    }


def _records(path: str | Path) -> Iterator:
    """Use SDK decoding, but also require complete Zstandard frames and DBN records."""
    path = Path(path)
    decoder = dbn.DBNDecoder(compression=Compression.NONE)  # type: ignore # dbn.Compression.from_int(0) is the same but pylance won't be angry
    decompressor = None
    try:
        with path.open("rb") as stream:
            compressed = stream.read(4) == b"\x28\xb5\x2f\xfd"
            stream.seek(0)
            if compressed:
                decompressor = zstd.ZstdDecompressor().decompressobj()
            while chunk := stream.read(65536):
                while chunk:
                    if compressed:
                        assert decompressor is not None
                        if decompressor.eof:
                            decompressor = zstd.ZstdDecompressor().decompressobj()
                        decoded = decompressor.decompress(chunk)
                        chunk = decompressor.unused_data
                    else:
                        decoded, chunk = chunk, b""
                    decoder.write(decoded)
                    yield from decoder.decode()
            if compressed and decompressor is not None and not decompressor.eof:
                raise DataError(f"Truncated Zstandard frame: {path}")
            if decoder.buffer():
                raise DataError(f"Truncated DBN metadata or record: {path}")
    except DataError:
        raise
    except (OSError, ValueError, RuntimeError, dbn.DBNError, zstd.ZstdError) as exc:
        raise DataError(f"Cannot read complete DBN file {path}: {exc}") from exc


def inspect_file(path: str | Path) -> dict:
    metadata = None
    identities = set()
    actions = Counter()
    count = 0
    first = last = None
    snapshot_count = 0
    symbols = set()
    for record in _records(path):
        if isinstance(record, dbn.Metadata):
            if metadata is not None:
                raise DataError("Multiple DBN metadata headers are not supported")
            metadata = record
            continue
        if not isinstance(record, (dbn.MBOMsg, dbn.InstrumentDefMsg)):
            raise DataError(f"Unexpected record type: {type(record).__name__}")
        identities.add((record.publisher_id, record.instrument_id))
        count += 1
        if isinstance(record, dbn.MBOMsg):
            if metadata is None or metadata.schema != "mbo":
                raise DataError("MBO record conflicts with metadata schema")
            first = first if first is not None else record
            last = record
            actions[str(record.action)] += 1
            snapshot_count += bool(record.flags & dbn.F_SNAPSHOT)
            if record.flags & (dbn.F_MBP | dbn.F_TOB):
                raise DataError("Aggregated/top-of-book records cannot initialise this L3 book")
        else:
            if metadata is None or metadata.schema != "definition":
                raise DataError("Definition record conflicts with metadata schema")
            symbols.add(record.raw_symbol)
    if metadata is None or count == 0:
        raise DataError("DBN file has no metadata or no records")
    result = {
        "dataset": metadata.dataset,
        "schema": str(metadata.schema),
        "start": metadata.start,
        "end": metadata.end,
        "records": count,
        "identities": sorted(identities),
        "symbols": sorted(symbols or set(metadata.symbols)),
        "actions": dict(actions),
        "snapshot_records": snapshot_count,
        "initialised": bool(first and first.action == "R" and first.flags & dbn.F_SNAPSHOT),
        "complete_event": bool(last and last.flags & dbn.F_LAST),
        "last_ts_recv": last.ts_recv if last else None,
    }
    return result


def _validate_pair(mbo_path: str | Path, definitions_path: str | Path) -> tuple[dict, dict]:
    mbo, definitions = inspect_file(mbo_path), inspect_file(definitions_path)
    if mbo["schema"] != "mbo" or definitions["schema"] != "definition":
        raise DataError("Expected MBO and instrument-definition files")
    if mbo["dataset"] != "GLBX.MDP3" or definitions["dataset"] != mbo["dataset"]:
        raise DataError("Mismatched dataset; expected GLBX.MDP3")
    if len(mbo["identities"]) != 1 or mbo["identities"] != definitions["identities"]:
        raise DataError("Mismatched instruments/publishers; supply exactly one contract")
    if len(definitions["symbols"]) != 1:
        raise DataError("Definitions must describe one raw symbol")
    if not mbo["initialised"]:
        raise DataError("Missing initialisation: first MBO record must be snapshot CLEAR")
    if not mbo["complete_event"]:
        raise DataError("Incomplete ending event: final raw MBO record lacks F_LAST")
    return mbo, definitions


def load_instrument(definitions_path: str | Path, *, loader=None):
    info = inspect_file(definitions_path)
    if info["schema"] != "definition" or len(info["identities"]) != 1 or len(info["symbols"]) != 1:
        raise DataError("Expected definitions for one instrument")
    loader = loader if loader is not None else DatabentoDataLoader()
    try:
        instruments = loader.from_dbn_file(Path(definitions_path))
    except (OSError, ValueError, RuntimeError, dbn.DBNError) as exc:
        raise DataError(f"Nautilus definition decoding failed: {exc}") from exc
    if not instruments or len({str(x.id) for x in instruments}) != 1:
        raise DataError("Definitions do not resolve to one Nautilus instrument")
    if len({(str(x.price_increment), str(x.multiplier)) for x in instruments}) != 1:
        raise DataError("Instrument tick size or multiplier changes within the file")
    return instruments[-1]


def load_deltas(mbo_path: str | Path, definitions_path: str | Path) -> tuple:
    info, _ = _validate_pair(mbo_path, definitions_path)
    loader = DatabentoDataLoader()
    instrument = load_instrument(definitions_path, loader=loader)
    try:
        deltas = loader.from_dbn_file(
            Path(mbo_path),
            instrument_id=instrument.id,
            price_precision=instrument.price_precision,
            include_trades=False,
        )
    except (OSError, ValueError, RuntimeError, dbn.DBNError) as exc:
        raise DataError(f"Nautilus MBO decoding failed: {exc}") from exc
    if not deltas or any(
        not isinstance(delta, OrderBookDelta) or delta.instrument_id != instrument.id
        for delta in deltas
    ):
        raise DataError("Mismatched instruments in Nautilus deltas and definitions")
    expected = sum(info["actions"].get(action, 0) for action in "ACMR")
    if len(deltas) != expected:
        raise DataError("Unexpected delta count: only A/C/M/R should mutate the book")
    return instrument, deltas


def build_book(mbo_path: str | Path, definitions_path: str | Path) -> OrderBook:
    instrument, deltas = load_deltas(mbo_path, definitions_path)
    book = OrderBook(instrument.id, BookType.L3_MBO)
    try:
        for delta in deltas:
            book.apply_delta(delta)
        book.check_integrity()
    except (ValueError, RuntimeError) as exc:
        raise DataError(f"Nautilus book integrity/application failed: {exc}") from exc
    return book


def summarize_book(book: OrderBook, depth: int = 10) -> dict:
    if depth < 1:
        raise DataError("Depth must be positive")
    book.check_integrity()

    def levels(side):
        return [
            {
                "price": str(level.price),
                "size": str(sum(order.size.as_decimal() for order in level.orders())),
                "orders": len(level.orders()),
            }
            for level in side[:depth]
        ]

    bid, ask = book.best_bid_price(), book.best_ask_price()
    return {
        "instrument": str(book.instrument_id),
        "synthetic": str(book.instrument_id).startswith("SYNTH_"),
        "book_type": "L3_MBO",
        "delta_count": book.update_count,
        "integrity": "passed",
        "best_bid": str(bid) if bid is not None else None,
        "best_ask": str(ask) if ask is not None else None,
        "spread": (
            str(ask.as_decimal() - bid.as_decimal())
            if bid is not None and ask is not None
            else None
        ),
        "bids": levels(book.bids()),
        "asks": levels(book.asks()),
    }

def top_levels_fast(book: OrderBook, depth: int = 10):
    if depth < 1:
        raise DataError("Depth must be positive")

    bids = [
        (
            level.price.as_double(),
            level.size(),
            level.len(),
        )
        for level in book.bids(depth)
    ]

    asks = [
        (
            level.price.as_double(),
            level.size(),
            level.len(),
        )
        for level in book.asks(depth)
    ]

    return bids, asks


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cache_locations(request: Request, directory: Path):
    key = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()[:16]
    manifest_path = directory / f"{key}.json"
    paths = {schema: directory / f"{key}.{schema}.dbn.zst" for schema in ("definition", "mbo")}
    return manifest_path, paths


def _cached_chunk(request: Request, directory: Path):
    manifest_path, paths = _cache_locations(request, directory)
    if not manifest_path.exists():
        if any(
            path.exists() or path.with_suffix(path.suffix + ".part").exists()
            for path in paths.values()
        ):
            raise DataError(
                "Unfinished download exists; inspect/remove it explicitly before retrying"
            )
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        valid = manifest["request"] == asdict(request) and all(
            path.is_file() and _sha256(path) == manifest["sha256"][schema]
            for schema, path in paths.items()
        )
    except (OSError, ValueError, KeyError) as exc:
        raise DataError(f"Invalid cache manifest: {manifest_path}") from exc
    if not valid:
        raise DataError("Cached files are missing or changed; inspect them before redownloading")
    _validate_pair(paths["mbo"], paths["definition"])
    return {"cached": True, "paths": paths, "manifest_path": manifest_path, "manifest": manifest}


def _download_chunk(
    request: Request,
    directory: Path,
    *,
    client,
) -> dict:
    cached = _cached_chunk(request, directory)
    if cached is not None:
        return cached

    manifest_path, paths = _cache_locations(request, directory)
    directory.mkdir(parents=True, exist_ok=True)
    partials = {schema: path.with_suffix(path.suffix + ".part") for schema, path in paths.items()}

    for schema, path in partials.items():
        client.timeseries.get_range(**_parameters(request, schema), path=path)
        inspect_file(path)

    _, definitions = _validate_pair(partials["mbo"], partials["definition"])
    resolved_identity = definitions["identities"][0][1]
    if request.stype_in == "instrument_id":
        if resolved_identity != int(request.symbol):
            raise DataError("Downloaded definition instrument does not match the planned request")
    elif definitions["symbols"] != [request.symbol]:
        raise DataError("Downloaded definition symbol does not match the planned request")

    manifest = {
        "request": asdict(request),
        "synthetic": any(symbol.startswith("SYNTH_") for symbol in definitions["symbols"]),
        "sha256": {schema: _sha256(path) for schema, path in partials.items()},
        "versions": {name: version(name) for name in ("databento", "nautilus_trader")},
    }
    for schema, path in paths.items():
        partials[schema].replace(path)
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return {"cached": False, "paths": paths, "manifest_path": manifest_path, "manifest": manifest}


def _range_manifest_path(request: Request, directory: Path) -> Path:
    key = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()[:16]
    return directory / f"{key}.range.json"


def iter_range_batches(range_manifest_path: str | Path):
    """Yield verified (request, MBO path, definition path) chunks in chronological order."""
    range_manifest_path = Path(range_manifest_path)
    try:
        manifest = json.loads(range_manifest_path.read_text())
        items = manifest["chunks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"Invalid range manifest: {range_manifest_path}") from exc

    directory = range_manifest_path.parent
    previous_end = None
    for item in items:
        try:
            request = Request(**item["request"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"Invalid chunk request in {range_manifest_path}") from exc
        if previous_end is not None and _utc(request.start) != previous_end:
            raise DataError("Range manifest has a gap or overlap between chunks")
        cached = _cached_chunk(request, directory)
        if cached is None:
            raise DataError(f"Range cache is incomplete; missing chunk {asdict(request)}")
        previous_end = _utc(request.end)
        yield request, cached["paths"]["mbo"], cached["paths"]["definition"]


def _load_range_cache(request: Request, directory: Path):
    path = _range_manifest_path(request, directory)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
        if manifest["request"] != asdict(request):
            raise DataError(f"Range manifest does not match request: {path}")
        chunks = []
        for item in manifest["chunks"]:
            chunk = Request(**item["request"])
            cached = _cached_chunk(chunk, directory)
            if cached is None:
                raise DataError(f"Range cache is incomplete; missing chunk {item['request']}")
            chunks.append(cached)
    except DataError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError(f"Invalid range manifest: {path}") from exc
    return {
        "cached": True,
        "range_manifest_path": path,
        "range_manifest": manifest,
        "chunks": chunks,
    }


def download(
    request: Request,
    directory: str | Path,
    *,
    client=None,
    progress=None,
) -> dict:
    """Download a range safely; multi-day requests fan out into cached single-instrument chunks."""
    directory = Path(directory)

    cached_range = _load_range_cache(request, directory)
    if cached_range is not None:
        return cached_range

    client = client if client is not None else db.Historical()

    chunks = plan(request, client=client)
    results = []
    for index, chunk in enumerate(chunks, 1):
        if progress is not None:
            progress(index, len(chunks), chunk)
        results.append(
            _download_chunk(
                chunk,
                directory,
                client=client,
            )
        )

    range_manifest = {
        "request": asdict(request),
        "chunks": [
            {
                "request": asdict(chunk),
                "manifest": result["manifest_path"].name,
            }
            for chunk, result in zip(chunks, results)
        ],
    }
    range_path = _range_manifest_path(request, directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = range_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(range_manifest, indent=2) + "\n")
    temporary.replace(range_path)
    return {
        "cached": all(result["cached"] for result in results),
        "range_manifest_path": range_path,
        "range_manifest": range_manifest,
        "chunks": results,
    }
