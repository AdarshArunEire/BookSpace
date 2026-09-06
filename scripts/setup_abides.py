"""Install pinned ABIDES sources and a separate Windows Python 3.9 runtime."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f9cbe51342b7dedd9587e4e069040d68a5c6477f"
REPOSITORY = "https://github.com/jpmorganchase/abides-jpmc-public.git"


def setup():
    vendor = ROOT / ".local/vendor/abides"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if not vendor.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(vendor)], check=True)
        subprocess.run(["git", "-C", str(vendor), "checkout", "--detach", COMMIT], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != COMMIT:
        raise RuntimeError("Existing ABIDES checkout has a different revision; inspect it first")
    # Windows defaults to 32-bit C long. These two upstream seed draws need uint64.
    patches = {
        "abides-core/abides_core/utils.py": (
            "pd.to_timedelta(string).to_timedelta64().astype(int)",
            "pd.to_timedelta(string).value",
        ),
        "abides-markets/abides_markets/configs/rmsc04.py": (
            "np.random.randint(low=0, high=2**32)",
            'np.random.randint(low=0, high=2**32, dtype="uint64")',
        ),
        "abides-markets/abides_markets/utils/__init__.py": (
            "np.random.randint(low=0, high=2 ** 32)",
            'np.random.randint(low=0, high=2 ** 32, dtype="uint64")',
        ),
    }
    for relative, (before, after) in patches.items():
        path = vendor / relative
        original = subprocess.check_output(
            ["git", "-C", str(vendor), "show", f"HEAD:{relative}"], text=True
        )
        expected = original.replace(before, after)
        current = path.read_text(encoding="utf-8")
        if current not in (original, expected):
            raise RuntimeError(f"Unexpected local ABIDES modifications: {relative}")
        path.write_text(expected, encoding="utf-8")
    uv = [shutil.which("uv")] if shutil.which("uv") else ["py", "-m", "uv"]
    environment = ROOT / ".local/abides-env39"
    interpreter = environment / "Scripts/python.exe"
    if not interpreter.exists():
        subprocess.run(uv + ["venv", "--python", "3.9", str(environment)], check=True)
    subprocess.run(
        uv
        + [
            "pip",
            "sync",
            "--python",
            str(interpreter),
            "--only-binary",
            ":all:",
            str(ROOT / "scripts/abides-requirements.txt"),
        ],
        check=True,
    )
    print(f"ABIDES {COMMIT}: ready at {interpreter}")


if __name__ == "__main__":
    setup()
