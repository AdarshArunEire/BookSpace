"""Local cumulative estimate reservations; not a provider billing guarantee."""

import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4


def ledger_path(path=None):
    if path is None:
        path = os.environ.get("MBO_BUDGET_LEDGER")
    if not path:
        raise ValueError("Set MBO_BUDGET_LEDGER to an initialized local budget ledger")
    return Path(path).expanduser().resolve()


def amount(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid budget amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Budget amounts must be finite and nonnegative")
    return result


@contextmanager
def locked(path):
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"Budget locked: {lock}. Another download may be running. "
            "After a crash, inspect the ledger before manually removing this lock."
        ) from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def append(path, record, mode="a"):
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def initialize(path, available_usd, expires_on):
    path = ledger_path(path)
    available = amount(available_usd)
    if available > Decimal("125"):
        raise ValueError("This project's lifetime budget cannot exceed $125")
    expiry = date.fromisoformat(expires_on)
    if expiry <= datetime.now(timezone.utc).date():
        raise ValueError("Credit expiry must be a future date")
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path):
        append(path, {
            "type": "budget", "version": 1, "available_usd": str(available),
            "expires_on": expiry.isoformat(),
        }, mode="x")
    return status(path)


def status(path=None):
    path = ledger_path(path)
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.endswith("\n"):
            raise ValueError("Incomplete ledger write")
        records = [json.loads(line) for line in raw.splitlines()]
        header = records[0]
        if header["type"] != "budget" or header["version"] != 1:
            raise ValueError("Invalid ledger header")
        limit = amount(header["available_usd"])
        if limit > Decimal("125"):
            raise ValueError("Ledger limit exceeds $125")
        expiry = date.fromisoformat(header["expires_on"])
        reserved = Decimal("0")
        ids = set()
        completed = set()
        for record in records[1:]:
            if record["type"] == "complete":
                if record["id"] not in ids or record["id"] in completed:
                    raise ValueError("Invalid completion record")
                completed.add(record["id"])
                continue
            if record["type"] != "reserve" or record["id"] in ids:
                raise ValueError("Invalid or duplicate reservation")
            ids.add(record["id"])
            reserved += amount(record["usd"])
        if reserved > limit:
            raise ValueError("Ledger reservations exceed its limit")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Budget ledger unavailable or corrupt: {path}: {exc}") from exc
    return {
        "ledger": str(path), "limit_usd": str(limit),
        "reserved_usd": str(reserved), "remaining_usd": str(limit - reserved),
        "expires_on": expiry.isoformat(),
        "expired": expiry <= datetime.now(timezone.utc).date(),
    }


def reserve(cost, request, path=None):
    path = ledger_path(path)
    cost = amount(cost)
    with locked(path):
        current = status(path)
        if current["expired"]:
            raise ValueError("Local credit budget expired; verify your Databento billing page")
        if cost > amount(current["remaining_usd"]):
            raise ValueError(
                f"Estimated ${cost} exceeds cumulative remaining budget "
                f"${current['remaining_usd']}"
            )
        record = {
            "type": "reserve", "id": str(uuid4()), "usd": str(cost),
            "at": datetime.now(timezone.utc).isoformat(), "request": request,
        }
        append(path, record)
    return {"ledger": str(path), **record}


def complete(reservation):
    path = ledger_path(reservation["ledger"])
    with locked(path):
        status(path)
        append(path, {
            "type": "complete", "id": reservation["id"],
            "at": datetime.now(timezone.utc).isoformat(),
        })
