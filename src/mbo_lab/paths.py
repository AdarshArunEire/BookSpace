"""Default data locations shared by BookSpace and the isolated ABIDES runtime."""

from datetime import datetime
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def abides_paths(seed=0, end_time="10:00:00"):
    end = datetime.strptime(end_time, "%H:%M:%S").strftime("%H%M%S")
    name = f"rmsc04-seed-{int(seed)}-until-{end}"
    return DATA / "simulated/abides" / name, DATA / "processed/abides" / name / "smoke"
