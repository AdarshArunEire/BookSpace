"""Small-file historical MBO loading. Network access is explicit and optional."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

import databento as db
import databento_dbn as dbn
import zstandard as zstd
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.enums import BookType

from mbo_lab.budget import complete, reserve, status

MAX_DECODED_BYTES = 128 * 1024 * 1024


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
        if end <= start or start.date() != (end - timedelta(microseconds=1)).date():
            raise DataError("Choose a positive interval within one UTC day")
        if start.time().isoformat() != "00:00:00" or start.weekday() > 4:
            raise DataError("Start at weekday 00:00:00Z to include the CME MBO snapshot")

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


def resolve(request: Request, *, client) -> Request:
    if request.stype_in in {"raw_symbol", "instrument_id"}:
        return request
    day = _utc(request.start).date()
    response = client.symbology.resolve(
        dataset=request.dataset,
        symbols=[request.symbol],
        stype_in=request.stype_in,
        stype_out="instrument_id",
        start_date=day.isoformat(),
        end_date=(day + timedelta(days=1)).isoformat(),
    )
    try:
        intervals = response["result"][request.symbol]
        symbols = {
            str(item["s"]) for item in intervals if item["d0"] <= day.isoformat() < item["d1"]
        }
    except (KeyError, TypeError) as exc:
        raise DataError(f"Databento returned no usable mapping for {request.symbol}") from exc
    if len(symbols) != 1:
        raise DataError(f"Expected one dated contract for {request.symbol}: {response}")
    return replace(request, symbol=symbols.pop(), stype_in="instrument_id")


def estimate(request: Request, *, client=None) -> dict:
    client = client if client is not None else db.Historical()
    resolved = resolve(request, client=client)
    costs = {
        schema: float(client.metadata.get_cost(**_parameters(resolved, schema)))
        for schema in ("definition", "mbo")
    }
    if any(not math.isfinite(cost) or cost < 0 for cost in costs.values()):
        raise DataError("Provider returned an invalid cost estimate")
    return {"request": asdict(resolved), "usd": costs, "total_usd": sum(costs.values())}


def _records(path: str | Path) -> Iterator:
    """Use SDK decoding, but also require complete Zstandard frames and DBN records."""
    path = Path(path)
    decoder = dbn.DBNDecoder(compression=dbn.Compression.NONE)
    total = 0
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
                        if decompressor.eof:
                            decompressor = zstd.ZstdDecompressor().decompressobj()
                        decoded = decompressor.decompress(chunk)
                        chunk = decompressor.unused_data
                    else:
                        decoded, chunk = chunk, b""
                    total += len(decoded)
                    if total > MAX_DECODED_BYTES:
                        raise DataError(
                            "File exceeds 128 MiB decoded limit; request a shorter range"
                        )
                    decoder.write(decoded)
                    yield from decoder.decode()
            if compressed and not decompressor.eof:
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


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(
    request: Request, directory: str | Path, *, max_cost: float, client=None, budget_path=None,
    confirm=None,
) -> dict:
    """Download only on an explicit call. A verified cache hit needs no client or API key."""
    if not math.isfinite(max_cost) or max_cost < 0:
        raise DataError("max_cost must be a finite, nonnegative USD amount")
    directory = Path(directory)
    key = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()[:16]
    manifest_path = directory / f"{key}.json"
    paths = {schema: directory / f"{key}.{schema}.dbn.zst" for schema in ("definition", "mbo")}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            valid = manifest["request"] == asdict(request) and all(
                path.is_file() and _sha256(path) == manifest["sha256"][schema]
                for schema, path in paths.items()
            )
        except (OSError, ValueError, KeyError) as exc:
            raise DataError(f"Invalid cache manifest: {manifest_path}") from exc
        if not valid:
            raise DataError(
                "Cached files are missing or changed; inspect them before redownloading"
            )
        _validate_pair(paths["mbo"], paths["definition"])
        return {"cached": True, "paths": paths, "manifest": manifest}
    if any(
        path.exists() or path.with_suffix(path.suffix + ".part").exists() for path in paths.values()
    ):
        raise DataError("Unfinished download exists; inspect/remove it explicitly before retrying")
    budget = status(budget_path)
    if budget["expired"]:
        raise DataError("Local credit budget expired; verify your Databento billing page")
    client = client if client is not None else db.Historical()
    quote = estimate(request, client=client)
    if quote["total_usd"] > max_cost:
        raise DataError(f"Estimated ${quote['total_usd']:.6f} exceeds maximum ${max_cost:.6f}")
    if quote["total_usd"] > float(budget["remaining_usd"]):
        raise DataError("Estimate exceeds cumulative remaining budget")
    if confirm is None or confirm({"quote": quote, "budget": budget}) is not True:
        raise DataError("Download not confirmed; no data requested or budget reserved")
    resolved = Request(**quote["request"])
    directory.mkdir(parents=True, exist_ok=True)
    partials = {schema: path.with_suffix(path.suffix + ".part") for schema, path in paths.items()}
    reservation = reserve(quote["total_usd"], asdict(request), budget_path)
    for schema, path in partials.items():
        client.timeseries.get_range(**_parameters(resolved, schema), path=path)
        inspect_file(path)
    _, definitions = _validate_pair(partials["mbo"], partials["definition"])
    resolved_identity = definitions["identities"][0][1]
    if resolved.stype_in == "instrument_id":
        if resolved_identity != int(resolved.symbol):
            raise DataError("Downloaded definition instrument does not match the resolved request")
    elif definitions["symbols"] != [resolved.symbol]:
        raise DataError("Downloaded definition symbol does not match the resolved request")
    manifest = {
        "request": asdict(request),
        "resolved_request": asdict(resolved),
        "synthetic": any(symbol.startswith("SYNTH_") for symbol in definitions["symbols"]),
        "estimate": quote,
        "budget_reservation": reservation,
        "sha256": {schema: _sha256(path) for schema, path in partials.items()},
        "versions": {name: version(name) for name in ("databento", "nautilus_trader")},
    }
    for schema, path in paths.items():
        partials[schema].replace(path)
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    complete(reservation)
    return {"cached": False, "paths": paths, "manifest": manifest}
