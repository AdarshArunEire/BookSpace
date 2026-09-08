# Saved-session training data

The data path is implemented; no ML runtime, encoder, forecast baseline or optimizer
is added. The sources remain compressed, separate ABIDES observation archives.

## Frozen corpus

`data/processed/abides/training-v1/corpus.json` records verified paths, hashes,
source settings, window ranges and split membership. Paths are relative to the
manifest. A changed source or fitted artifact is rejected when loaded.

The audit recovered 1,102 completed sessions with 56,381,255 observations. The
generator's `dataset.json` contained only 32 sessions after a resumed verification
pass reset it. Recovery enumerated only expected dates inside that dataset,
verified provenance/seeds and actual file hashes, and excluded the unfinished
`2025-04-23` directory. It did not rewrite the generator manifest.

| Split | Sessions | Rows | Eligible histories | Eligible unordered pairs |
|---|---:|---:|---:|---:|
| Train | 771 | 39,516,659 | 38,516,736 | 741,754,857,604,514 |
| Validation | 165 | 8,388,329 | 8,132,762 | 33,067,827,104,455 |
| Test | 166 | 8,476,267 | 8,249,281 | 34,022,188,598,621 |

Splits use whole sessions in date-label order: floor 70%, floor 15%, remainder.
Dates label independent simulations, not historical market conditions. Every file
passed schema, finite-value, book-level, mask, timestamp-order, source-offset,
segment-initialization, return and hash checks. Fragmented but valid sessions remain
included; per-session segment lengths, invalid-state counts, usable anchors,
zero-return-path fractions and future elapsed-time quantiles are in the manifest.
The most fragmented session retains 9,480 anchors from 22,513 rows across 141 segments.
This is simulation-data acceptance, not real ES replay validation.

## Pair universe and traversal

A sample is `(session_id, anchor_row)`. Its history ends at the anchor and its
future starts one row later. Default complete support is `[t-255,t+128]`.

- Each episode must remain inside one continuous segment and session.
- Within a session, pairs require anchor separation at least 384, including pairs
  drawn from different segments. Across independent sessions every combination
  of individually valid anchors is eligible.
- Only train/train pairs feed the training stream. Validation/validation pairs
  form a separate fixed distance-evaluation sample. No test sample is evaluated.
- `PairIndex` maps every eligible unordered pair to/from an integer ID. Range
  metadata is compact; only eight sessions' expanded anchor lookup arrays are cached.
- `PairTraversal` uses an eight-round keyed Feistel bijection plus cycle walking.
  It visits each pair at most once, with constant traversal state and no global
  pair list or seen-pair set. The seeded pseudorandom order is not a mathematically
  uniform draw from all possible permutations. The rank space gives every eligible
  pair equal representation; no distance-based or cache-based sampling filter exists.
- History reuse across different pairs is allowed. A new stream with the same
  seed starts the same sequence; resume its checkpoint to continue an existing run.

`validation_pairs.json` freezes 8,192 distinct pair IDs (seed 29) separately from
training. Later validation must reuse those IDs and the fitted training transforms.

## Shared preprocessing

`preprocessing.json` fits 39,081,051 unique rows covered by eligible training
histories, not overlapping windows as repeated observations. It records feature
counts, means, population standard deviations, constant channels, time-gap tau,
schema, source fingerprint and training pair IDs used for target units.

Quantities/order counts use log1p; time gaps use log1p(delta/tau). Level statistics
exclude absent levels, whose normalized fields stay zero. Presence, initialization
and time-of-day sine/cosine stay unchanged. Float64 calculations precede float32
model inputs. An exact positive-gap median is selected through bounded radix passes
over a temporary scalar file, deleted afterward. No persistent row/window copy is made.

Target distance is RMS between 128-step cumulative log-return paths. Its positive
median over 100,000 nonrepeating training pairs (seed 17) is approximately
`5.0913897106464105e-05`; the zero-distance fraction was zero. The two disjoint
50,000-pair halves gave medians `5.10402584807386e-05` and `5.079555335735098e-05`.
This is a training-only scale-sensitivity diagnostic, not predictive validation.

## Stream API

```python
from mbo_lab.paths import DATA
from mbo_lab.corpus import read_corpus
from mbo_lab.preprocessing import read_preprocessing
from mbo_lab.stream import PairStream

directory = DATA / "processed/abides/training-v1"
corpus, root = read_corpus(directory / "corpus.json")
scale = read_preprocessing(directory / "preprocessing.json", corpus)

with PairStream(corpus, root, scale, seed=0, queue_pairs=16384,
                batch_pairs=32, workers=4, memory_bytes=6 * 1024**3) as stream:
    batch = next(stream)
    # Consume this batch before saving the continuation point.
    stream.checkpoint(directory / "stream.checkpoint.json")
    print(stream.report())
```

| Field | Meaning |
|---|---|
| `X[U,256,86]` | Float32 normalized histories; U unique histories within this batch |
| `Y[U,128]` | Float64 raw cumulative future log-return paths |
| `pairs[P,2]` | Integer indexes into X/Y; default P=32, U at most 64 |
| `D_raw[P]`, `D[P]` | Raw RMS future distances and distances divided by the frozen scale |
| `sample_ids` | Stable `(session_id, anchor_row)` tuple for every history |
| `pair_ids[P]` | Unique integer IDs in this split's pair universe |

Restore with `stream.restore(checkpoint_path)` before requesting the next batch.
Checkpoints include corpus/scaler identity, traversal version/seed, delivered
counter, queue/batch sizes and coverage counts. A partial RAM queue is rebuilt
deterministically; worker count may change. Queue and batch sizes must match.
Restoration reproduces batch membership, order and numerical contents. As with any
checkpoint, work after the last saved checkpoint is replayed after a crash.

## Memory and concurrency

The stream draws a bounded queue globally, deduplicates its requested histories,
groups requests by session and gathers the X/Y rows into RAM. Each requested
session is loaded once per queue (or reused from cache), regardless of how many
different pair partners it has. Small batches then refer to those gathered histories.
All sampled pairs are delivered; grouping does not bias selection.

Defaults reserve a 6 GiB owned-data budget on the inspected 16 GB, 8-core/16-thread
machine. The budget includes the worst-case queue, one emitted batch, an LRU raw
session cache and conservative per-worker decompression/transformation reserves.
Requests too small to support those reserves are rejected. Allocations do not scale
with the number of possible pairs. Queue finite checks and transforms are chunked.

The budget excludes the Python interpreter, OS page cache, and batches retained by
the caller. Consume/release batches instead of collecting them in a list. Calls
to `next()` are synchronous; session loading/gathering inside a queue uses bounded
worker submissions, not one unbounded job per pair. Fitting statistics uses one
session at a time, with a bounded scalar-median pass and at most 200,000 fitting pairs.

## Commands and verification

Use the existing BookSpace environment. On PowerShell, from the repository root:

```powershell
.venv\Scripts\python.exe scripts/prepare_training.py audit --source data/simulated/abides/1348-sessions-seed-0-until-160000 --workers 4
.venv\Scripts\python.exe scripts/prepare_training.py fit
.venv\Scripts\python.exe scripts/prepare_training.py validation
.venv\Scripts\python.exe scripts/prepare_training.py benchmark --workers 4 --queue-pairs 16384 --memory-gib 6 --queues 2
.venv\Scripts\python.exe -m unittest discover -s tests -p test_training_data.py -v
```

Audit and fit refuse to overwrite frozen artifacts. They have already been run
for training-v1; use `--output data/processed/abides/<new-version>` for a new corpus.
Benchmarking never trains a model. It intentionally uses the same seed in separate
benchmark runs to compare worker counts on the same sampled workload.

Open `notebooks/data_stream.ipynb` for a saved-data smoke and exact resume check.
The integer fixtures in the tests check combinatorics only. Integration tests use
saved ABIDES sources and explicitly skip if the audited corpus is absent; there
is no synthetic fallback masquerading as a saved-data acceptance run.

## Measured loader performance

Two consecutive queues of 8,192 globally selected pairs, 32 pairs per minibatch,
6 GiB owned-data budget, on the current Windows machine. Separate processes used
identical pair IDs for the worker comparison. The first queue starts with an empty
application cache; OS filesystem caches were not controlled. No encoder work is included.

| Workers | First queue pairs/s | Second queue pairs/s | Peak process working set |
|---|---:|---:|---:|
| 1 | 127.9 | 147.4 | 5.67 GiB |
| 2 | 227.5 | 267.0 | 5.44 GiB |
| 4 | 343.7 | 334.8 | 4.90 GiB |

Four workers are the default based on this workload. The smaller observed RAM
peak with four workers reflects a larger reserved loading allowance and therefore
less space assigned to the persistent session cache. Timing remains machine/load
specific. Reports and resumable benchmark positions are saved alongside the corpus.

Validation completed: 12 focused data tests, 8 existing sample tests and 2 existing
session-generation tests passed. The repository-wide Ruff check still finds six
pre-existing issues in `src/mbo_lab/data.py`; new pipeline files pass Ruff.

Increasing the queue to 16,384 pairs at the same 6 GiB budget yielded **563.1 and
514.9 pairs/s**, with a **4.88 GiB** peak process working set. That larger queue
is now the default: more gathered histories per source load was more useful than
retaining a larger raw-session cache. The benchmark covered 32,768 distinct pairs
from the full training universe. These are preparation rates, not model-training rates.

The executed notebook produced X[64,256,86], Y[64,128] and 32 pair mappings,
then reproduced the next batch exactly after checkpoint reload. A second full
source audit, including session-date verification, reproduced the frozen manifest
exactly. Model modules remain empty and project dependencies are unchanged.
