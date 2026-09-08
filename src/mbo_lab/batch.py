"""Resumable acquisition of provider-prepared DBN files, without a spending ledger."""

import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import databento as db
from databento.common.error import BentoClientError

from mbo_lab.data import DataError, Request, _parameters, _sha256, _utc

SPLIT_SIZE = 5_000_000_000


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _lock(path):
    with path.open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise DataError("This request is already running in another process") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise DataError("This request is already running in another process") from exc
        yield


def _job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9-]+", value):
        raise DataError("Invalid Databento job ID")
    return value


def _check_job(details, request, schema):
    symbols = details.get("symbols")
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    expected = {
        "dataset": request.dataset,
        "stype_in": request.stype_in,
        "schema": schema,
        "encoding": "dbn",
        "compression": "zstd",
        "split_duration": "day",
        "split_symbols": True,
        "split_size": SPLIT_SIZE,
        "stype_out": "instrument_id",
    }
    if (
        any(details.get(key) != value for key, value in expected.items())
        or symbols != [request.symbol]
        or details.get("limit") is not None
        or _utc(details["start"]) != _utc(request.start)
        or _utc(details["end"]) != _utc(request.end)
    ):
        raise DataError(f"Batch job does not match the requested {schema} data")


def _inventory(items):
    files = []
    for item in items:
        name, digest, size = item["filename"], item["hash"], item["size"]
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or any(char in name for char in '/\\:<>"|?*')
            or name.endswith((".", " "))
        ):
            raise DataError("Unsafe filename in batch manifest")
        if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
            raise DataError(f"Missing SHA256 checksum for {name}")
        if not isinstance(size, int) or size < 0:
            raise DataError(f"Invalid size for {name}")
        files.append({"filename": name, "size": size, "sha256": digest[7:].lower()})
    if not files or len({item["filename"].lower() for item in files}) != len(files):
        raise DataError("Empty or duplicate batch file inventory")
    return files


def _valid(path, item):
    return (
        path.is_file() and path.stat().st_size == item["size"] and _sha256(path) == item["sha256"]
    )


def download(request: Request, directory, *, client=None, progress=None, jobs=None):
    """Submit two jobs once, then retrieve individual files with provider checksum checks."""
    import hashlib

    key = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(directory).resolve() / key
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "batch.json"
    report = progress or (lambda message: None)
    with _lock(folder / ".download.lock"):
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["request"] != asdict(request) or not isinstance(receipt["jobs"], dict):
                    raise ValueError("request mismatch")
                if set(receipt["jobs"]) - {"definition", "mbo"}:
                    raise ValueError("unexpected schema")
                for saved in receipt["jobs"].values():
                    if "id" in saved:
                        _job_id(saved["id"])
                    if "files" in saved:
                        saved["files"] = _inventory(
                            [
                                {**item, "hash": "sha256:" + item["sha256"]}
                                for item in saved["files"]
                            ]
                        )
            except (ValueError, KeyError, TypeError) as exc:
                raise DataError(
                    f"Invalid batch receipt: {receipt_path}; restore it before retrying"
                ) from exc
        else:
            receipt = {"request": asdict(request), "jobs": {}}
            _write(receipt_path, receipt)

        def api():
            nonlocal client
            if client is None:
                client = db.Historical()
            return client

        for schema, job_id in (jobs or {}).items():
            _job_id(job_id)
            existing = receipt["jobs"].get(schema, {})
            if existing.get("id") not in (None, job_id):
                raise DataError(f"A different {schema} job is already saved")
            details = api().batch.get_job_details(job_id=job_id)
            _check_job(details, request, schema)
            receipt["jobs"][schema] = {"id": job_id}
            _write(receipt_path, receipt)

        # An interrupted submission has an unknown outcome; never retry the purchase blindly.
        for schema in ("definition", "mbo"):
            saved = receipt["jobs"].get(schema)
            if saved is not None and not saved.get("id"):
                raise DataError(
                    f"{schema} submission outcome is unknown. Check Databento Download center; "
                    f"rerun with --{schema}-job JOB_ID to reuse that job. Receipt: {receipt_path}"
                )
        for schema in ("definition", "mbo"):
            if schema in receipt["jobs"]:
                continue
            provider = api()
            receipt["jobs"][schema] = {"submitting": True}
            _write(receipt_path, receipt)
            report(f"Submitting {schema}: {request.symbol}, {request.start} to {request.end}")
            try:
                details = provider.batch.submit_job(
                    **_parameters(request, schema),
                    encoding="dbn",
                    compression="zstd",
                    split_duration="day",
                    split_symbols=True,
                    split_size=SPLIT_SIZE,
                    stype_out="instrument_id",
                    delivery="download",
                )
            except BentoClientError as exc:
                if exc.http_status in (400, 401, 402, 403, 404, 422, 429):
                    del receipt["jobs"][schema]
                    _write(receipt_path, receipt)
                raise
            job_id = _job_id(details["id"])
            receipt["jobs"][schema] = {"id": job_id}
            _write(receipt_path, receipt)
            report(f"Saved {schema} job: {job_id}")

        downloaded = reused = 0
        for schema in ("definition", "mbo"):
            saved = receipt["jobs"][schema]
            job_id = _job_id(saved["id"])
            if "files" not in saved:
                while True:
                    details = api().batch.get_job_details(job_id=job_id)
                    _check_job(details, request, schema)
                    state = details["state"]
                    if state == "done":
                        break
                    if state not in ("queued", "processing", "received"):
                        raise DataError(f"{schema} job {job_id}: {state}; no new job submitted")
                    report(f"{schema} job {job_id}: {state}; checking again in 15 seconds")
                    time.sleep(15)
                saved["files"] = _inventory(api().batch.list_files(job_id=job_id))
                _write(receipt_path, receipt)
            target = folder / job_id
            target.mkdir(exist_ok=True)
            size = sum(item["size"] for item in saved["files"])
            report(f"{schema}: {len(saved['files'])} prepared files, {size / 1e9:.3f} GB on disk")
            for index, item in enumerate(saved["files"], 1):
                path = target / item["filename"]
                report(
                    f"{schema} [{index}/{len(saved['files'])}] {item['filename']} "
                    f"({item['size'] / 1e6:.1f} MB)"
                )
                if _valid(path, item):
                    reused += 1
                    continue
                details = api().batch.get_job_details(job_id=job_id)
                if details["state"] != "done":
                    raise DataError(f"Job {job_id} is {details['state']}; no new job submitted")
                if shutil.disk_usage(folder).free < item["size"]:
                    raise DataError(
                        "Not enough free disk space for the next file; free space and rerun"
                    )
                api().batch.download(
                    job_id=job_id,
                    output_dir=folder / ".partial",
                    filename_to_download=item["filename"],
                )
                staged = folder / ".partial" / job_id / item["filename"]
                if not _valid(staged, item):
                    raise DataError(
                        f"Checksum/size mismatch: {staged}. Rerun to retry the same job."
                    )
                staged.replace(path)
                downloaded += 1
        return {
            "directory": folder,
            "receipt": receipt_path,
            "jobs": {schema: item["id"] for schema, item in receipt["jobs"].items()},
            "downloaded_files": downloaded,
            "reused_files": reused,
            "stored_bytes": sum(
                file["size"] for job in receipt["jobs"].values() for file in job["files"]
            ),
            "verification": "provider SHA256 and file sizes",
        }
