# BookSpace
**Learning Predictive Geometry in Limit Order Books**

In BookSpace, we ask: have we seen a market state **like this** before, and what was it's future like?

We take it one step further by asking what does ***“like this”*** *actually mean?*

Best work lives at [`final_trading_alg.py`](final_trading_alg.py).

## Training sample structures

The full smoke run now uses **ABIDES RMSC04**, with all **84 features** per
observation, 256-row histories, 128-step cumulative future returns and scaled
pair distances. The old synthetic fixtures and generators have been removed.

```powershell
py -m uv sync --locked
py -m uv run python scripts/setup_abides.py
py -m uv run python scripts/training_smoke.py
```

Setup is needed once. ABIDES runs in a separate pinned Python 3.9 environment;
BookSpace remains on Python 3.13. Outputs are organized under `data/simulated/abides/` and `data/processed/abides/`.
See [ABIDES setup, schema and tests](docs/abides.md) for details.
Learned encoders and their training loop are not implemented yet.

## Databento reconstruction

Real DBN -> Nautilus L3 book reconstruction remains in `src/mbo_lab/data.py`.
`extract.py` samples raw F_LAST boundaries into the shared observation schema;
`samples.py` constructs windows within explicit chronological splits and valid
segments. Session boundaries must be supplied; no exchange calendar is inferred.

## Real ES data

Review `es.toml`. It requests one minute from weekday midnight UTC, resolving `ES.c.0`
to one dated contract. Resolution uses Databento's supported continuous -> instrument-ID
mapping, then downloads that dated contract by ID. The midnight snapshot is necessary to initialise a complete book.
For later intraday endpoints, the request must include all intervening updates.

Configure `DATABENTO_API_KEY` in your local environment. Do not put it in the config or notebook.

```powershell
python -m uv run mbo discover
python -m uv run mbo estimate --config es.toml
# Replace the cap with the maximum USD estimate you authorise.
python -m uv run mbo download --config es.toml --max-cost 0.10
```

The download command checks the combined MBO and definition estimate before calling
the historical API. The cap is a check against the provider's estimate, not a billing
guarantee. Original compressed files and a hashed request manifest are kept together
under `data/raw/`. A cache hit verifies hashes and file structure without creating a
Databento client. Interrupted or corrupt downloads fail explicitly; they are never
silently downloaded again. Inspect any `.part` files before manually removing them to retry.

The download command prints the two file paths. Pass those to `mbo book --mbo ... --definitions ...`.
The `book` and `inspect` commands never access the network.

## Python interface and boundaries

```python
from mbo_lab.data import build_book, load_deltas, summarize_book

book = build_book(mbo_path, definitions_path)
book.check_integrity()
summary = summarize_book(book)
```

`load_deltas` returns `(instrument, deltas)` if you want to inspect Nautilus objects
directly. `build_book` returns the actual Nautilus `OrderBook`, not another book model.
Summary prices and quantities are decimal strings to avoid float rounding.

This first version accepts one GLBX instrument per file and enforces a 128 MiB decoded
file limit because Nautilus's file loader materialises a list. It validates complete
Zstandard frames and DBN records, a starting snapshot CLEAR, matching instrument identity,
and F_LAST on the final **raw** record. Raw fill/status records may close an event even
when Nautilus emits no delta. Files remain in original order; timestamps are not used to
sort or invent event boundaries.

`build_book` exposes the final book; `extract_observations` exposes event rows. Integrity is checked after applying the whole file,
not during a partial snapshot or matching event. A structurally valid, crossed book is
also rejected; CME books can legitimately cross outside continuous trading, so this
strict initial interface is intended for continuous-trading endpoints. Integrity checks
do not prove that a provider delivered every exchange message. The historical API filters
MBO on receive time, and the pinned Nautilus decoder uses receive time for its event timestamp.

No Parquet catalog, server or custom matching engine is included. ABIDES runs separately.

## Paid historical downloads

The download command requires two separate limits: `--max-cost` is a per-request
ceiling, and the local append-only JSONL ledger is a cumulative ceiling. Before any
new transfer, the command asks Databento for a quote, checks the remaining ledger
budget, prints both, and requires the exact interactive word `DOWNLOAD`. There is no
`--yes` bypass. The quoted amount is reserved in the ledger before the transfer;
successful storage appends a `complete` record containing the same quoted USD amount.
Failed or interrupted transfers keep their reservation so they cannot be retried
without an explicit ledger review.

Create the ledger once using the remaining credit and expiry shown in the Databento
portal. Do not guess these values:

```powershell
python -m uv run mbo budget-init `
  --ledger .local/databento-budget.jsonl `
  --available-usd 125 `
  --expires-on 2027-01-01
$ledger = (Resolve-Path .local/databento-budget.jsonl).Path
[Environment]::SetEnvironmentVariable("MBO_BUDGET_LEDGER", $ledger, "User")
$env:MBO_BUDGET_LEDGER = $ledger
python -m uv run mbo budget
```

Then quote and, only after reviewing the printed estimate, request data:

```powershell
uv run mbo download --config es.toml --max-cost 0.10
```

Type `DOWNLOAD` only when the quote and remaining ledger budget are acceptable.
The current `es.toml` profile requests the continuous ES symbol `ES.c.0` from
`2024-06-17T00:00:00Z` through `2024-06-17T00:01:00Z`: one minute beginning at
weekday midnight UTC so the MBO stream includes the opening snapshot. The profile
does not request current data.

For a different date on the same ES profile, override both endpoints together:

```powershell
uv run mbo download --config es.toml --start 2024-06-24T00:00:00Z --end 2024-06-24T00:01:00Z --max-cost 0.10
```

The initial request validator requires a positive interval within one UTC day and a
weekday midnight UTC start. The symbol and dataset still come from `es.toml`; choosing
another instrument is a later CLI extension.

The local ledger cannot account for usage from other tools, API keys or team members,
and it cannot guarantee the provider's final bill. Set Databento's historical monthly
limit in the portal to the amount you are willing to pay, ideally below the local
budget, and monitor the provider's billing page as the authoritative record.

SDK references: [Nautilus loader at the pinned version](https://github.com/nautechsystems/nautilus_trader/blob/v1.231.0/nautilus_trader/adapters/databento/loaders.py),
[Databento snapshots](https://databento.com/docs/standards-and-conventions/mbo-snapshot).
