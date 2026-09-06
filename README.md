# BookSpace

BookSpace turns order-book observations into training samples. The current
pipeline is ABIDES simulation -> feature rows `x` -> history windows `X` and
future paths `Y` -> scalar pair targets `D`.

The current schema has 86 visible features per row, a 256-row history, and a
128-step future path. Encoder and training-loop implementations come later.

## Requirements

- PowerShell
- Python 3.13
- `uv` (install it with `py -m pip install uv` if `py -m uv` is unavailable)

## Setup

From the repository root, install the BookSpace environment:

```powershell
py -m uv sync --locked
```

BookSpace dependencies live in `.venv`. ABIDES has its own pinned Python 3.9
environment under `.local/abides-env39`. The `.local/` directory is ignored by
Git and is for ABIDES source and environments, not run data.

## Run the notebook pipeline

Set up ABIDES once:

```powershell
py -m uv run python scripts/setup_abides.py
```

Open the notebook after setup:

```powershell
py -m uv run python -m jupyterlab notebooks/abides_pipeline.ipynb
```

The first run cell generates the short smoke batch. The full-generation cells
let you choose a longer session and anchor count; set `RUN_FULL = True` before
running that cell. Generated files go to:

- `data/simulated/abides/<run>/`: observations, logs, and provenance
- `data/processed/abides/<run>/smoke/`: the derived batch and report

These payloads are ignored by Git. The tracked layout is documented in
[data/README.md](data/README.md).

For automation, the compatibility wrapper remains available:

```powershell
py -m uv run python scripts/training_smoke.py --seed 1 --end-time 11:00:00
```

## Check the code

```powershell
py -m uv run ruff check src scripts tests
py -m uv run python -m unittest discover -s tests -p test_samples.py -v
```

The test command starts ABIDES and may take a little while.

## Further reading

- [ABIDES setup, schema, and run details](docs/abides.md)
- [Data layout and retention](data/README.md)
