"""Default data locations shared by BookSpace and the isolated ABIDES runtime."""

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


def require_data_path(path):
    """Resolve a generated-data path and reject locations outside ``data/``."""
    path = Path(path).resolve()
    if not path.is_relative_to(DATA):
        raise ValueError(f"Generated data must be written under {DATA}")
    return path


def abides_paths(seed=0, end_time="10:00:00"):
    end = datetime.strptime(end_time, "%H:%M:%S").strftime("%H%M%S")
    name = f"rmsc04-seed-{int(seed)}-until-{end}"
    return DATA / "simulated/abides" / name, DATA / "processed/abides" / name / "smoke"
