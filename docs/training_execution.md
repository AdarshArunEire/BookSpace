# Training, validation and test execution notes

Updated 8 September 2026. This document records measured data costs, the real-data
preparation boundary, and the experiment execution path. No final test-set scoring
has been run.

## Data primitives and current boundary

The saved Databento pilot is `data/processed/databento/pilot-20250102/`, covering
2025-01-02 00:00–00:05 UTC, with an exclusive end. It uses ESH5.GLBX and the
definition-derived 0.25 tick. April has not been processed by this pilot.

- `dbn_replay.py`: ordered raw decoding, bounded SDK delta conversion, persistent
  L3 book, event-boundary feature extraction and an order-ID shadow-book check.
- `observation_store.py`: bounded compressed chunks, logical row/segment indexing,
  checksum-verified cached reads, cross-chunk episodes and pair-batch materialization.
- Existing `PairIndex`/`PairTraversal`: eligible unordered pairs and reproducible
  nonrepeating IDs. A new range-count primitive avoids expanding rows merely to count pairs.
- `materialize_pairs` uses logical timeline IDs and returns the existing `PairBatch`
  fields: X, Y, pair mappings, raw/scaled distances and source/pair IDs. It rejects
  an incompatible preprocessing corpus or reader source version.

The daily `observation-chunks-v1` stores now have a Databento-aware corpus audit,
resumable train-only preprocessing, and a chunk-backed implementation of the shared
pair-stream contract. The ABIDES and Databento loaders remain distinct behind that
contract; real and simulated sources are not silently mixed. Real ES normalization
has not finished fitting yet.

## Real daily corpus and preprocessing

`data/corpus.json` freezes 102 completed daily timelines: 51 January-February
training days, 26 March validation days, and 25 April test days. It contains
1,053,989,726 eligible anchors. The incomplete `20250418` store is explicitly
excluded. `data/validation_pairs.json` freezes 8,192 March pair IDs with seed 29.

Preprocessing scans only training history rows. It uses a deterministic one-million
value random-priority reservoir for the positive time-gap median, then makes a
second bounded pass for feature statistics and samples 100,000 training pairs for
target-distance units. Progress is saved after every day in
`data/preprocessing.progress.npz`; rerunning the same command resumes it. The current
checkpoint has completed 5 of 51 days in the first pass.

```powershell
uv run python .\scripts\prepare_training.py fit --output ".\data"
```

After this creates `data/preprocessing.json`, the experiment wrapper can consume
the real daily stores. `scripts/experiments.py` is a thin entry point over the
reusable `mbo_lab.experiment` API. Its `all` stage trains for the requested pair
budget, then deliberately visits every training anchor for prototype fitting and
future summaries and every validation anchor for forecast evaluation. On this
corpus those full-anchor stages are much larger than the 1,000-step encoder fit.

The installed Nautilus loader returns a whole-file list, so this prototype feeds
it small temporary compressed DBN blocks. Both record-block size and observation
chunk size are bounded. The bridge preserves SDK action conversion and can be
replaced if later profiling justifies it. It is not presented as the optimal
bulk decoder. Temporary conversion blocks are removed; provider archives are retained.

## Identity and continuity

Keep these distinct:

1. Source position: MBO-record ordinal across explicitly ordered source parts.
2. Logical timeline and segment: contract/session and valid contiguous history.
3. Storage chunk: a bounded group of observation rows, carrying no session semantics.

The initial snapshot initializes the book but does not become a training
observation. The first real completed event starts with initialization=1 and
zero previous-return/time-gap features. Subsequent observations follow raw F_LAST
boundaries, including completed events that produce no book mutation. Raw exchange
and receive timestamps remain separate. Bad receive-time events and invalid books
break sample continuity; histories cannot cross resets or logical segment edges.

A source part can end mid-event: the next ordered part continues the same book
and event. The first part must begin with a snapshot CLEAR. A contract change is
rejected by this single-contract primitive and must be scheduled separately.
File order is explicit, not guessed by sorting arbitrary filenames.

Storage changes do not discard windows or relax pair exclusion. The reader gathers
adjacent chunks when needed. Same-timeline anchors require separation of at least
384; putting them in different chunk files does not make them eligible.

## Replay measurements

First run: 4,096 records per conversion block, 512 observations per output chunk.
The small chunks deliberately exercise boundaries; they are not a production recommendation.

| Measurement | Observed |
|---|---:|
| Raw MBO records consumed, including initialization and stopping boundary | 33,070 |
| Snapshot records used only for initialization | 10,009 |
| Exported observation rows | 18,971 |
| Eligible 256/128 histories | 18,588 |
| Eligible unordered pairs | 165,701,910 |
| Total wall time, including source hashing/setup | 23.55 s |
| Filesystem reading + Zstandard decompression | 0.0035 s |
| DBN record decoding | 0.0018 s |
| Temporary-block creation + native SDK conversion | 0.1097 s |
| Nautilus book application | 0.0347 s |
| Feature extraction and book summarization | 21.6594 s |
| Observation buffering/chunk writes | 0.9128 s |
| Shadow-book comparisons | 20 checks; 0.0727 s |
| Maximum live orders in the shadow book | 10,008 |
| Peak process working set | 198.1 MiB |

Feature extraction accounted for about 92% of wall time. The existing summarizer
checks integrity, creates level/order objects, and formats decimal values at every
observation. Profile that path before attempting to accelerate decompression.
This timing does not establish a speedup for a replacement summarizer.

The decoder read ahead to 2 MiB of decompressed data; it did not decompress the
entire original archive. File reading and decompression are measured together,
not falsely reported as isolated CPU decompression time. Filesystem caches were
not controlled, and source hashing can warm them. Snapshot/shadow validation also
adds work that must be identified in any production benchmark.

Changing to 257-record conversion blocks and 1,024-row chunks produced exactly
the same observations in 23.80 s. All columns matched, and windows spanning chunks
matched. The bounded output matched the existing whole-prefix extractor, except
for the deliberate first-row initialization change after omitting the snapshot.
A separate test split the raw prefix inside an event and reproduced every column.

## Window-loading measurements and RAM movement

512 unique pairs, 32 pairs per batch. The same workload was repeated only to
measure warm-cache performance. Identity-like scaling constants exercise the
normalization interface; these are not fitted ES training statistics.

| Chunk cache | First pass | Warm pass | Warm p95 batch latency | Warm array bytes loaded |
|---|---:|---:|---:|---:|
| 2 MiB | 215 pairs/s | 222 pairs/s | 150 ms | 408.6 MB |
| 32 MiB | 842 pairs/s | 935 pairs/s | 36 ms | 0 |

The pilot's raw arrays occupy 13.81 MB. The small cache incurred 1,099 misses and
roughly 409 MB of repeated array loads per pass. The larger cache loaded 38 chunks
once; its warm pass had 1,099 hits and no misses. Peak process working set in this
profile was approximately 120 MiB. The profile delivered about 90.9 MB of X/Y
arrays over its batches; those are consumed batch by batch, not retained together.

The data path is compressed source -> bounded decoded records -> live book ->
bounded observation chunk on disk -> decoded chunk cache -> gathered X/Y batch ->
future model input. At training time, replay is not repeated: read the derived
observation chunks. Cache memory and output batch memory are separate allocations.
The reader bounds cache storage and limits row/batch requests; caller-retained
batches and each concurrently open reader's cache must be included in the total budget.

The 32 MiB result works because this tiny pilot fits completely. It does not imply
the multi-month corpus fits, nor that global training will sustain 935 pairs/s.
Benchmark several representative independent initialized streams, queue sizes and
worker counts after the scheduler exists. Never parallelize dependent updates to
one book out of order. More replay workers also multiply live-book memory.

## Target and split diagnostics

On this pilot, a 128-event future spans:

| Minimum | Median | 95th percentile | Maximum |
|---:|---:|---:|---:|
| 0.00247 s | 1.425 s | 5.804 s | 26.175 s |

About 2.30% of eligible future paths have no mid-price changes. The horizon is an
event count, not a fixed time interval. Repeat these diagnostics on representative
training periods before choosing model or evaluation budgets. A midnight pilot
does not represent daytime ES activity, roll periods or the complete dataset.

Proposed real-data split remains January–February training, March validation,
April final test. These are chronological timestamp boundaries, not random files
or pairs. Before freezing them, define the research session convention and enforce
the boundaries on the whole history-plus-future support. Contract rolls and invalid
intervals are additional boundaries. Do not use the ABIDES 70/15/15 assignment or
its fitted statistics for real ES data.

## Future loop plan

Training: fit feature and target scales only on training sources, once per corpus
version. Sample globally without replacement, group requested reads into bounded
queues, deduplicate histories per batch, then transfer model inputs. Record delivered
pair count, unique-history/session coverage, raw/scaled loss, loader wait, transfer
time, forward/backward time, optimizer time, peak RAM/VRAM, and checkpoint duration.
Report same-timeline versus cross-timeline pair counts and cache misses/bytes read.

Validation: freeze held-out pair/query IDs and reuse the training transforms.
Run without optimizer updates. Separately time loading, encoding and (once added)
reference-library search/forecast scoring. Keep distance-fit and forecast metrics
distinct. Select checkpoints using a declared validation criterion and patience,
with a maximum training budget. Estimate validation overhead from measured latency
and query counts, rather than choosing an arbitrary evaluation frequency.

Test: keep April out of fitting, architecture selection and stopping decisions.
After selecting a configuration using March, run the frozen final evaluation.
Report per-day results and dependence-aware uncertainty instead of treating
overlapping queries as independent. Do not repeatedly use April to choose changes.

Checkpoint model/optimizer state later alongside the sampler position, corpus
fingerprint and preprocessing fingerprint. Save after the batch has been consumed.
Pending RAM queues can be rebuilt; returning to the last saved checkpoint replays
work performed after that checkpoint. An epoch should be a declared pair/step
budget, not an attempted traversal of every possible pair.

## Remaining implementation before the real model

1. Build the metadata-driven file/definition schedule, with calendar, snapshot,
   instrument-roll, gap and January–March/April policies.
2. Profile/optimize feature extraction against the validated implementation, then
   run representative larger replay exports and verify later snapshots where feasible.
3. Extend shared corpus/preprocessing interfaces to stream logical timelines in
   chunks. The current pair index still expands a timeline for within-pair lookup;
   replace that lookup with range arithmetic if measured timeline size warrants it.
4. Fit real training-only scales, freeze validation membership, and integrate
   chunk-reader cache budgets into the multi-session prefetch/queue scheduler.
5. Recheck memory, throughput, split leakage and resume on the real corpus.

No first-model runtime or loop implementation is needed to complete these steps.

## Saved evidence and commands

- [Replay stages and RAM progress](../data/processed/databento/pilot-20250102/chunks-512/metrics.json)
- [RAM progress CSV](../data/processed/databento/pilot-20250102/chunks-512/memory-progress.csv)
- [First window/cache measurements](../data/processed/databento/pilot-20250102/loader-metrics.json)
- [Window/cache CSV](../data/processed/databento/pilot-20250102/loader-metrics.csv)

Reproduce the bounded window profile:

```powershell
uv run python scripts/profile_databento.py --directory data/processed/databento/pilot-20250102/chunks-512
uv run python -m unittest discover -s tests -p test_databento_replay.py -v
```

Run a new replay with `scripts/replay_databento.py --help`. It requires explicitly
ordered MBO paths, matching definitions, a new output directory, logical timeline
ID and exclusive UTC stop time. It makes no network calls. The bounded reference
prefix retained under the pilot is a validation artifact, not a replacement for
the original provider files. Six new tests and twelve existing training-data tests
passed; no result here is a predictive-performance claim.
