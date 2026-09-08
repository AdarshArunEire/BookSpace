# BookSpace

**Learning Predictive Geometry in Limit Order Books**

In BookSpace, we ask: have we seen a market state **like this** before, and what was it's future like?

We take it one step further by asking what does ***“like this”*** *actually mean?*

Best work lives at [`final_trading_alg.py`](final_trading_alg.py).

#### ***STILL WORK IN PROGRESS***

MAKE SURE I CHECK SUNDAY SLICE FOR TESTING PHASE MODELS CUZ SUNDAYS SPARSER

## Requirements

- PowerShell
- Python 3.13
- `uv` on PATH

## Setup

From the repository root, install the BookSpace environment:

```powershell
uv sync --locked
```

BookSpace dependencies live in `.venv`. ABIDES has its own pinned Python 3.9
environment under `.local/abides-env39`. The `.local/` directory is ignored by
Git and is for ABIDES source and environments.

## Databento CLI

With `DATABENTO_API_KEY` set in your environment:

```powershell
uv run mbo estimate
uv run mbo download --dry-run
uv run mbo download
```

The default `es.toml` requests `GLBX.MDP3`, `ES.v.0`, from
2025-01-01 00:00 UTC through 2025-05-01 00:00 UTC (exclusive).
`ES.v.0` follows the contract selected by previous-day volume. The request includes
all recorded MBO events and instrument definitions, without a record limit.
April is reserved for the later held-out evaluation; downloading it does not train a model.

This download is complete locally (7 September 2026): 103 MBO and 103 definition
DBN archives, about 19.95 GB compressed. See [the saved-data inventory](data/README.md)
for the location and acquisition checks. A bounded five-minute real-data replay pilot
is validated; bulk reconstruction and preparation remain pending.
[Execution notes](docs/training_execution.md) record its primitives, timings, RAM
and the remaining training/validation/test plan.

`estimate` prints billable bytes, historical batch USD/GB rates and provider cost estimates.
`download --dry-run` is entirely offline. `download` submits two paid batch jobs
(MBO and definitions), waits for preparation, and saves compressed `.dbn.zst` files.
Manage spending controls in Databento; there is no local budget ledger or confirmation prompt.

Batch downloads are the default for this large request. Files are split by UTC day
and raw contract, with `split_size=5000000000`. This is Databento's file-splitting
setting, not a RAM limit or a guarantee about total storage. Their documentation
does not explicitly distinguish pre/post-compression bytes for this threshold.
Actual file sizes are reported after preparation. Files arrive already compressed;
the downloader never expands them or creates a combined ZIP archive.

Files are saved one at a time under `data/raw/<request-id>/<job-id>/` and checked
against Databento's SHA256 and byte counts. The small `batch.json` receipt records
job IDs and file inventories for resuming; it records no budget or spending reservations.
Rerun the same command with the same directory to reuse verified files and the
existing jobs. Keep the receipt. Batch files can be downloaded again for 30 days
without another data charge; expired jobs are not automatically repurchased.

If submission loses its response, the CLI stops instead of risking a duplicate purchase.
Find the existing job in Databento's Download center and attach it using
`--mbo-job JOB_ID` or `--definition-job JOB_ID` on the same download command.
The job's request settings are checked before it is reused.

Estimates send the entire range and symbol directly to Databento: two cost calls,
two billable-size calls and one unit-price call, regardless of range length.
Estimation does not split the range into daily API calls.

Override the range with both dates, keeping the start at UTC midnight:

```powershell
uv run mbo estimate --stream --start 2025-01-06T00:00:00Z --end 2025-01-06T00:10:00Z
uv run mbo download --stream --start 2025-01-06T00:00:00Z --end 2025-01-06T00:10:00Z
uv run mbo inspect --file PATH_TO_MBO.dbn.zst
uv run mbo book --mbo PATH_TO_MBO.dbn.zst --definitions PATH_TO_DEFINITIONS.dbn.zst
```

Use `--config PATH` for another profile, `download --directory PATH` to choose
the output folder, or `--json` on any command for machine-readable output.
`uv run mbo --help` lists commands, including dataset discovery.

`--stream` keeps the small-request path available; it has separate cache manifests
and repeated uncached streaming requests are billed again. Batch and stream caches
are not interchangeable.

Acquisition checks validate the downloaded bytes, not the reconstructed market book.
Reconstruction must begin with a valid snapshot and preserve order across size-split
files. Databento generates historical CME snapshots at weekday midnight UTC;
Sunday files and later parts of a split day must not be assumed to initialise a book.
The existing `book` command expects one complete, single-contract MBO file beginning
with a snapshot and its matching definitions. It is not a bulk corpus processor.

Databento estimates can overstate partial ten-minute MBO intervals; definitions
have 24-hour estimation granularity. The CLI notes these cases without changing
the requested range. Rates come from `metadata.list_unit_prices`; bytes and costs
come from `metadata.get_billable_size` and `metadata.get_cost` for matching requests.
See the [Databento metadata documentation](https://databento.com/docs/api-reference-historical/metadata).
The [historical API docs](https://databento.com/docs/api-reference-historical) recommend
batch downloads above 5 GB and describe repeat-download billing.
See also [batch submission options](https://databento.com/docs/api-reference-historical/batch/batch-submit-job)
and [snapshot boundaries](https://databento.com/docs/standards-and-conventions/mbo-snapshot).

Run the offline CLI/download checks:

```powershell
uv run python -m unittest discover -s tests -p test_databento.py -v
uv run python -m unittest discover -s tests -p test_batch.py -v
```

## Run the notebook pipeline

Set up ABIDES once:

```powershell
uv run python scripts/setup_abides.py
```

Open the notebook after setup:

```powershell
uv run python -m jupyterlab notebooks/abides_pipeline.ipynb
```

Run the smoke cell for a short pipeline check. In the generation cell, change
`sessions` and optionally `seed`. Each independent session runs 09:30–16:00;
20 sessions gives 130 simulated market hours. Completed sessions are reused when
rerunning the same configuration. Generated files go to:

- `data/simulated/abides/<run>/`: observations, logs, and provenance
- `data/simulated/abides/<dataset>/<date>/`: observations for each generated session
- `data/processed/abides/<run>/smoke/`: the derived batch and report

These payloads are ignored by Git. The tracked layout is documented in
[data/README.md](data/README.md).

For automation, the compatibility wrapper remains available:

```powershell
uv run python scripts/training_smoke.py --seed 1 --end-time 11:00:00
```

## Prepare saved sessions for a model

The saved-session loader supplies normalized histories, future paths and unique
eligible pair mappings without materializing the full dataset. See
[the data preparation guide](docs/training_data.md) for the frozen corpus, memory
settings and commands. Open [the data stream notebook](notebooks/data_stream.ipynb)
for a saved-data batch and reproducible checkpoint smoke check. No model is trained.

## Check the code

```powershell
uv run ruff check src scripts tests
uv run python -m unittest discover -s tests -p test_samples.py -v
```

The test command starts ABIDES and may take a little while.

## Further reading

- [ABIDES setup, schema, and run details](docs/abides.md)
- [Data layout and retention](data/README.md)
