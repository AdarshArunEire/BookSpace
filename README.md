# BookSpace
**Learning Predictive Geometry in Limit Order Books**

In BookSpace, we ask: have we seen a market state **like this** before, and what was it's future like?

We take it one step further by asking what does ***“like this”*** *actually mean?*

Best work lives at [`final_trading_alg.py`](final_trading_alg.py).

---


### TEMP NOTES!!!! :

# MBO lab

Databento compressed DBN files -> Nautilus deltas -> Nautilus L3 order book.

The working code is in `src/mbo_lab/data.py` and `cli.py`. Research modules are empty,
as requested. This uses Python namespace packages: no `__init__.py` files are needed.
Git has not been initialised.

## Environment

From this directory in PowerShell:

```powershell
python -m pip install uv
python -m uv sync --locked
python -m uv run jupyter lab notebooks/smoke.ipynb
```

uv installs managed CPython 3.13 and the locked dependencies into `.venv`.
The SDK versions are Databento 0.86.0 and NautilusTrader 1.231.0. Use this project's
kernel when opening the notebook in another editor. The first cell reports versions.

## Synthetic smoke run

Run all cells in `notebooks/smoke.ipynb`. It writes two clearly labelled fixtures and
a provenance manifest under `data/synthetic/`, then exercises the real Nautilus loader.
Socket connections and Databento client construction are forbidden during offline checks.

**SYNTHETIC: pipeline checks only.** The prices, quantities, and order flow are invented.
The fixture covers snapshot/reset, add, modify, cancel, trade/fill attribution,
malformed input failures, deterministic reconstruction, and verified download caching.
The notebook saves its outputs so the assertions and resulting book can be reviewed.

The fixture generator is entirely in the notebook's **Disposable synthetic fixture
generator** section. Delete that section when it is no longer useful; keep the labelled
data. The later corruption tests also use its helpers and can be removed with it.
Production code contains no generator or synthetic fallback.

Inspect the generated data from the terminal:

```powershell
python -m uv run mbo inspect --file data/synthetic/SYNTH_ES.mbo.dbn.zst
python -m uv run mbo book --mbo data/synthetic/SYNTH_ES.mbo.dbn.zst --definitions data/synthetic/SYNTH_ES.definition.dbn.zst
```

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

To enable the notebook's separate real-data smoke section, set these **before launching
the kernel**, with your chosen cost cap and the API key already configured:

```powershell
$env:MBO_RUN_LIVE = '1'
$env:MBO_MAX_COST_USD = '0.10'
python -m uv run jupyter lab notebooks/smoke.ipynb
```

It checks discovery -> estimate -> download -> a populated ES book. The section is
skipped by default and never substitutes synthetic data. The offline checks remain offline.

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

Only the final book is exposed. Integrity is checked after applying the whole file,
not during a partial snapshot or matching event. A structurally valid, crossed book is
also rejected; CME books can legitimately cross outside continuous trading, so this
strict initial interface is intended for continuous-trading endpoints. Integrity checks
do not prove that a provider delivered every exchange message. The historical API filters
MBO on receive time, and the pinned Nautilus decoder uses receive time for its event timestamp.

No Parquet catalog, server, simulator, or custom matching engine is included.

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
