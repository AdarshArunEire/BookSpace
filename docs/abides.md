# ABIDES observations and training smoke run

The smoke run executes ABIDES RMSC04 and records visible book states after complete
exchange order requests. Both ABIDES and the Databento extractor use the same
84-feature builder: 10 levels per side with relative price, quantity, order count
and presence, plus spread, elapsed time, latest mid-price log return and an
initialization flag. Missing levels are masked; invalid states break continuity.

From BookSpace in PowerShell:

```powershell
py -m uv sync --locked
py -m uv run python scripts/setup_abides.py
py -m uv run python scripts/training_smoke.py
py -m uv run python -m unittest discover -s tests -p test_samples.py -v
```

Setup is needed once. It downloads ABIDES commit
`f9cbe51342b7dedd9587e4e069040d68a5c6477f` and installs a separate Python 3.9
environment in `.local/abides-env39`. BookSpace keeps Python 3.13. Three documented
Windows compatibility fixes preserve 64-bit random seeds and nanosecond durations;
agent strategies and the upstream order-size model are unchanged. Sources load
directly from the pinned checkout. Dependencies are pinned in
`scripts/abides-requirements.txt`.

The default simulation runs from 09:30 to 10:00 in a synthetic session. To extend
it or change the seed:

```powershell
py -m uv run python scripts/training_smoke.py --end-time 11:00:00 --seed 1
```

Every smoke run executes ABIDES -> x and one-step y -> chronological 256-row X and
128-step cumulative Y -> disjoint pair distances D. Feature transforms and the
positive-median distance scale are fitted only on training data. It then exercises
standardized Euclidean one-neighbour retrieval on a validation query. The batch
shape is `(64, 256, 84)`. There is no reduced-feature fallback or custom synthetic
history generator. Learned encoders and their training loop remain separate work.

Outputs use the run name `rmsc04-seed-<seed>-until-<HHMMSS>`:

- Source observations, provenance and simulation logs: `data/simulated/abides/<run>/`.
- Derived smoke batch and report: `data/processed/abides/<run>/smoke/`.

These payloads stay ignored. See [data layout](../data/README.md) for retention and
future dataset conventions. `--output` overrides the derived smoke directory;
`--observations-dir` overrides its simulation directory. When calling the exporter
directly, its `--output` refers to the simulation directory.

Files:

- `observations.npz`: raw x, mid-price, one-step y, timestamps, segments and feature names.
- `batch.npz`: transformed X, cumulative Y, pairs, raw/scaled D and fitted scales.
- `provenance.json`: source revision, patched-source hashes, runtime, configuration,
  observation hash, sampling/clock definitions and invalid-state counts.
- `report.json`: dimensions, train/validation counts and smoke results.

ABIDES uses simulated equities and integer-cent prices, not calibrated ES futures.
Its clock is simulation time; both timestamp arrays in the shared table carry that
clock and are labelled `simulation`. One row is a completed exchange order request,
including no-ops; it is not a Databento F_LAST event. Simulator-internal hidden
orders and latent fundamental values are excluded from x.

These runs test the pipeline, not real-market forecasting performance. Large-scale
streaming and multi-session generation are not implemented yet. The previous
synthetic DBN fixtures, array generators and their notebooks/plots were removed.
Tests execute ABIDES and check the full schema. The removed DBN regression notebook
no longer provides decoder coverage; ABIDES tests do not replace that coverage.

Upstream: https://github.com/jpmorganchase/abides-jpmc-public
