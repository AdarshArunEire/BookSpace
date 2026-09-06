# Data layout

All generated/downloaded datasets and smoke outputs belong here. Data payloads
are ignored by Git; this guide is tracked. Environments, vendor source and the
download-budget ledger remain in ignored `.local/`.

```text
data/
  raw/                         Original Databento DBN files and cache manifests
  simulated/abides/<run>/       ABIDES observations, provenance and logs/
  processed/abides/<run>/smoke/  Derived batch.npz and report.json
  .tests/                      Temporary test runs, removed after tests
```

Current run names look like `rmsc04-seed-0-until-100000`. The RMSC04 default
session starts at 09:30; the suffix records its end time. Each run's provenance
records the simulation parameters, runtime, pinned source and data hash.
Changing seed/end time selects a different directory. Repeating the same run
replaces its outputs. For experiments with other configuration changes, use
explicit output directories to retain both versions.

`observations.npz` holds raw x, mid-price, one-step y, timestamps, segments and
feature names. Its adjacent `provenance.json` describes their origin.
`batch.npz` holds derived X, Y, pair indices, D and fitted preprocessing scales.
`report.json` links to the source observations/provenance and records their hash.
Full overlapping histories need not be materialized beyond the requested batch.

Future real-data derived artifacts should use `processed/databento/<run>/`.
Prefer preserving provider originals and keeping derived outputs separate.
The legacy `manifests/` location also remains ignored; the current downloader
stores manifests alongside DBN files in `raw/`.

Small examples or result CSVs intended for publication can be added explicitly
under a future `examples/` directory. Do not force-add generated directories.
CSV is optional; NumPy is the current format for observations and tensors.

Defaults work from any working directory for the ABIDES scripts. Explicit
`--output` and `--observations-dir` paths are resolved relative to the caller.
