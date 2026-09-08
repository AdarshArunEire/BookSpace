"""Generation of independently seeded ABIDES sessions."""

import hashlib
import json
from datetime import date, timedelta

from mbo_lab.abides import run_abides_export
from mbo_lab.observations import FEATURE_SCHEMA_VERSION
from mbo_lab.paths import DATA, require_data_path
from mbo_lab.pipeline import load_abides_source


def session_dates(sessions):
    """Assign internal timestamp dates to the requested number of sessions."""
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise ValueError("sessions must be a positive integer")
    days = []
    day = date(2021, 2, 1)
    while len(days) < sessions:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def generate_abides_dataset(
    sessions=20, seed=0, *, output=None, end_time="16:00:00"
):
    """Save all observation rows, one session at a time; resume verified exports."""
    days = session_dates(sessions)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    root = require_data_path(
        output if output is not None else DATA / "simulated/abides" / (
            f"{sessions}-sessions-seed-{seed}-until-{end_time.replace(':', '')}"
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "calendar": "internal timestamp labels only; no historical market conditions",
        "sessions_independent": True,
        "requested_sessions": len(days),
        "sessions": [],
        "rows": 0,
        "complete": False,
        "output": str(root),
    }
    manifest = root / "dataset.json"

    def checkpoint():
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(manifest)

    checkpoint()
    for index, day in enumerate(days, 1):
        day_text = day.isoformat()
        # Stable session seeds when the requested session count changes.
        daily_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{day_text}".encode()).digest()[:4], "big"
        ) % (2**32 - 1)
        source = root / day_text
        reused = (source / "provenance.json").exists()
        action = "checking saved session" if reused else "simulating"
        print(f"[{index}/{len(days)}] {day_text}: {action}")
        if not reused:
            run_abides_export(source, seed=daily_seed, end_time=end_time, date=day_text)
        observations, provenance, _, _ = load_abides_source(source)
        expected = {"seed": daily_seed, "date": day_text, "end_time": end_time}
        if any(provenance["parameters"].get(key) != value for key, value in expected.items()):
            raise ValueError(f"Saved session configuration mismatch: {source}")
        rows = len(observations.mid)
        report["sessions"].append({
            "date": day_text,
            "seed": daily_seed,
            "rows": rows,
            "observations": str(source / "observations.npz"),
            "observations_sha256": provenance["observations_sha256"],
        })
        report["rows"] += rows
        del observations
        checkpoint()
        print(f"  {rows:,} rows saved; total {report['rows']:,}")
    report["complete"] = True
    checkpoint()
    return report
