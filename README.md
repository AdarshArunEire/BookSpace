# BookSpace

BookSpace turns order-book observations into training samples. The current
working path is:

```text
ABIDES simulation -> observation rows x -> histories X and future paths Y -> pair distances D
```

The smoke pipeline uses 86 visible features, 256-row histories, and 128-step
future paths. It checks the data structures; learned encoders and the training
loop come later.

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

## Run the smoke pipeline

Set up ABIDES once:

```powershell
py -m uv run python scripts/setup_abides.py
```

Generate a run and build a training batch:

```powershell
py -m uv run python scripts/training_smoke.py
```

Choose another seed or a longer session with the same command:

```powershell
py -m uv run python scripts/training_smoke.py --seed 1 --end-time 11:00:00
```

Generated files go to:

- `data/simulated/abides/<run>/`: observations, logs, and provenance
- `data/processed/abides/<run>/smoke/`: the derived batch and report

These payloads are ignored by Git. The tracked layout is documented in
[data/README.md](data/README.md).

## Check the code

```powershell
py -m uv run ruff check src scripts tests
py -m uv run python -m unittest discover -s tests -p test_samples.py -v
```

The test command starts ABIDES and may take a little while.

## Further reading

- [ABIDES setup, schema, and run details](docs/abides.md)
- [Data layout and retention](data/README.md)
