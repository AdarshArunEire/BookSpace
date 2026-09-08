# Data layout

All generated/downloaded datasets and smoke outputs belong here. Data payloads
are ignored by Git; this guide is tracked. Environments and vendor source
remain in ignored `.local/`. Databento spending controls are managed in Databento.

```text
data/
  raw/<request-id>/<job-id>/    Compressed Databento batch files, kept as delivered
  raw/<request-id>/batch.json   Job IDs and checksum inventory for resuming
  simulated/abides/<run>/       ABIDES observations, provenance and logs/
  processed/abides/<run>/<batch>/ Derived batch.npz and report.json (smoke or full)
  .tests/                      Temporary test runs, removed after tests
```

Small `--stream` requests retain their hashed DBN files and cache manifests directly
under `raw/`. Batch acquisition verifies provider checksums and stores compressed
files without decoding them. Deriving observations and enforcing the January–March
development / April test boundary happen in the later preparation stage.

The completed Databento download (7 September 2026) is in
`raw/5f5736543928343c/`: `GLBX.MDP3`, `ES.v.0`, MBO and definitions for
2025-01-01 00:00 UTC to 2025-05-01 00:00 UTC (exclusive).
MBO job `GLBX-20260907-YJH8CJSCCV` contains 103 compressed DBN archives;
definition job `GLBX-20260907-ER8GJU4WYC` contains 103, with three provider
metadata files per job. Total delivered size is 19,951,134,900 bytes (19.95 GB).
All 212 receipt-listed files are present and match the recorded byte counts;
this documentation check did not rerun SHA256 verification or decode/replay the books.
Keep `batch.json` and both job directories together. January–March is for development,
April remains held out, and these raw files are not yet part of the ABIDES training stream.

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

Session runs use `simulated/abides/<count>-sessions-seed-<seed>-until-160000/`.
Each date has its own observation file, provenance, and logs. `dataset.json`
records completed sessions, row counts, and source hashes. Dates are internal
timestamp labels for independent sessions. Verified completed sessions are reused
on rerun with the same configuration. All raw rows are retained; generation does not build
one giant window tensor or apply the smoke batch's anchor limit. Training windows
must remain within individual sessions, with shared train-only preprocessing.

Future real-data derived artifacts should use `processed/databento/<run>/`.
Prefer preserving provider originals and keeping derived outputs separate.
The legacy `manifests/` location also remains ignored; the current downloader
stores manifests alongside DBN files in `raw/`.

Small examples or result CSVs intended for publication can be added explicitly
under a future `examples/` directory. Do not force-add generated directories.
CSV is optional; NumPy is the current format for observations and tensors.

Defaults work from any working directory for the ABIDES scripts. Explicit
`--output` and `--observations-dir` paths are resolved relative to the caller.

## Frozen training preparation

`processed/abides/training-v1/` contains `corpus.json`, `preprocessing.json`,
`validation_pairs.json`, stream checkpoints and loader benchmark reports. These
are small indexes/statistics/reports; they do not duplicate the compressed source
observations or persist overlapping histories. The temporary scalar file used to
fit the exact time-gap median is removed after fitting.

See [the training-data guide](../docs/training_data.md) for split membership,
source recovery, pair eligibility, concurrency and the bounded RAM queue.

## Databento replay pilot

`processed/databento/pilot-20250102/` contains bounded observation chunks for
January 2, 00:00-00:05 UTC, plus source provenance, replay timing/RAM measurements,
window-cache benchmarks and a bounded reference prefix used for validation.
These are pilot artifacts; they are not the complete real training corpus.
See [training execution notes](../docs/training_execution.md) for measured results,
storage/timeline boundaries and the remaining split/preprocessing/loop plan.
