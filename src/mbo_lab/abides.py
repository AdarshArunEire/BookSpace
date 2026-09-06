"""Reusable boundary for the pinned ABIDES runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mbo_lab.paths import REPO_ROOT, abides_paths, require_data_path

ABIDES_INTERPRETER = REPO_ROOT / ".local/abides-env39/Scripts/python.exe"
EXPORT_SCRIPT = REPO_ROOT / "scripts/export_abides.py"


def run_abides_export(output=None, seed=0, end_time="10:00:00", interpreter=None):
    """Run the pinned ABIDES exporter and return its observation directory."""
    target = Path(output) if output is not None else abides_paths(seed, end_time)[0]
    target = require_data_path(target)
    executable = Path(interpreter) if interpreter is not None else ABIDES_INTERPRETER
    executable = executable.resolve()
    if not executable.exists():
        raise RuntimeError("Run scripts/setup_abides.py first")
    if not EXPORT_SCRIPT.exists():
        raise RuntimeError(f"ABIDES exporter is missing: {EXPORT_SCRIPT}")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(executable),
            str(EXPORT_SCRIPT),
            "--output",
            str(target),
            "--seed",
            str(seed),
            "--end-time",
            end_time,
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return target
