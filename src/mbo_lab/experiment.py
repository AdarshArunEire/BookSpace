#!/usr/bin/env python3
"""Reusable BookSpace experiment pipeline.

The script in ``scripts/experiments.py`` is intentionally only a CLI entry point.

Current end-to-end build
------------------------
1. Train ``f0`` on the existing global PairStream using distance MSE.
2. Evaluate held-out pair-distance fit (validation/test diagnostics).
3. Freeze ``f0`` and fit one global MiniBatchKMeans state codebook on every
   eligible training anchor exactly once.
4. Replay the training anchors once more.  For each frozen state code ``c``,
   accumulate only:
       - exact occurrence count,
       - streaming mean future path,
       - streaming variance future path,
       - a small uniform reservoir of real future paths.
   No per-anchor code archive and no full historical Y archive are persisted.
5. Evaluate validation/test anchors with the deliberately simple predictor
       X -> f0(X) -> nearest frozen c -> mean_future[c].

Future-path VQ and P(future_mode | state) are deliberately NOT implemented
here yet.  They come only after this basic compressed predictor works.

Expected existing project APIs
------------------------------
The wrapper is wired to the current BookSpace data-preparation interfaces:

    from mbo_lab.stream import PairStream
    from mbo_lab.pair_index import PairIndex
    from mbo_lab.corpus import verified_source
    from mbo_lab.preprocessing import transform_rows
    from mbo_lab.corpus import expand_ranges

PairStream is expected to iterate PairBatch objects with:

    X      [B,T,F] float32, unique histories within the pair minibatch
    pairs  [P,2]   indices into X
    D      [P]     already scaled train-unit target distances

The unique-anchor replay uses PairIndex(corpus, split).records, loads each
session via verified_source(), uses every corpus-declared anchor once, and applies
exactly the same transform_rows(..., preprocessing["features"]) call used by
PairStream.

Typical use
-----------
    python scripts/experiments.py all \
        --corpus data/processed/abides/training-v1/corpus.json \
        --preprocessing data/processed/abides/training-v1/preprocessing.json \
        --data-root data/simulated/abides/1348-sessions-seed-0-until-160000 \
        --run-dir runs/bookspace-linear-v1

Stages can also be run independently:

    train        train encoder + pair-distance diagnostics
    compress     fit/freeze state prototypes
    summarize    build tiny c -> future-statistics artifact
    validate     evaluate compressed mean predictor on validation split
    test         evaluate on untouched test split
    all          train -> compress -> summarize -> validate (does NOT touch test)

The wrapper refuses to silently use missing upstream artifacts.  Use --force
only when you deliberately want to replace an existing stage artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch

from mbo_lab.compressz import CompressionSpec, build_compressor, load_compressor
from mbo_lab.corpus import fingerprint, verified_source
from mbo_lab.metrics.learned import EncoderSpec, build_encoder, distance_mse
from mbo_lab.pair_index import PairIndex, PairTraversal
from mbo_lab.preprocessing import transform_rows
from mbo_lab.stream import open_pair_stream

Stage = Literal["train", "compress", "summarize", "validate", "test", "all"]


class ExperimentError(RuntimeError):
    """Experiment configuration, artifact, or upstream-contract error."""


@dataclass(frozen=True, slots=True)
class TrainSpec:
    steps: int = 10_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_pairs: int = 32
    queue_pairs: int = 16_384
    stream_memory_bytes: int = 6 * 1024**3
    workers: int = 4
    validation_pairs: int = 16_384
    test_pairs: int = 16_384
    validation_interval: int = 0
    validation_probe_pairs: int = 256

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be finite and positive")
        if not 1 <= self.batch_pairs <= self.queue_pairs:
            raise ValueError("need 1 <= batch_pairs <= queue_pairs")
        if self.workers not in (1, 2, 4):
            raise ValueError("workers must be one of 1, 2, 4")
        if self.stream_memory_bytes <= 0:
            raise ValueError("stream_memory_bytes must be positive")
        if self.validation_pairs < 0 or self.test_pairs < 0:
            raise ValueError("pair evaluation budgets must be nonnegative")
        if self.validation_interval < 0:
            raise ValueError("validation_interval must be nonnegative")
        if self.validation_probe_pairs < 1:
            raise ValueError("validation_probe_pairs must be positive")
        if self.validation_interval and self.validation_probe_pairs > self.validation_pairs:
            raise ValueError("validation_probe_pairs cannot exceed validation_pairs")


@dataclass(frozen=True, slots=True)
class FutureSummarySpec:
    reservoir_per_state: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if self.reservoir_per_state < 1:
            raise ValueError("reservoir_per_state must be positive")
        if self.seed < 0:
            raise ValueError("future-summary seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class AnchorBatch:
    X: np.ndarray
    Y: np.ndarray | None
    session_id: str
    anchors: np.ndarray


@dataclass(frozen=True, slots=True)
class FutureSummary:
    counts: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    reservoir: np.ndarray
    reservoir_counts: np.ndarray

    @property
    def n_prototypes(self) -> int:
        return int(self.counts.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.mean.shape[1])

    @property
    def global_mean(self) -> np.ndarray:
        total = int(self.counts.sum())
        if total <= 0:
            raise ExperimentError("future summary contains no observations")
        active = self.counts > 0
        return np.sum(
            self.mean[active] * self.counts[active, None], axis=0, dtype=np.float64
        ) / float(total)


class StateFutureAccumulator:
    """Bounded-memory future summaries for frozen state codes.

    Mean/variance use exact parallel-Welford merges in float64.  The path
    reservoir uses random-priority sampling: each observed path receives one
    iid U(0,1) key and each state retains only its R smallest keys.  This is a
    uniform sample without a Python loop over every historical observation.
    """

    def __init__(
        self,
        n_prototypes: int,
        horizon: int,
        spec: FutureSummarySpec,
    ) -> None:
        if n_prototypes < 1 or horizon < 1:
            raise ValueError("n_prototypes and horizon must be positive")
        self.n_prototypes = int(n_prototypes)
        self.horizon = int(horizon)
        self.spec = spec
        self._rng = np.random.default_rng(spec.seed)

        self.counts = np.zeros(self.n_prototypes, dtype=np.int64)
        self.mean = np.zeros((self.n_prototypes, self.horizon), dtype=np.float64)
        self.m2 = np.zeros((self.n_prototypes, self.horizon), dtype=np.float64)

        r = int(spec.reservoir_per_state)
        self._reservoir_keys = np.full((self.n_prototypes, r), np.inf, dtype=np.float64)
        self.reservoir = np.full((self.n_prototypes, r, self.horizon), np.nan, dtype=np.float32)
        self.reservoir_counts = np.zeros(self.n_prototypes, dtype=np.int32)

    def update(self, codes: np.ndarray, futures: np.ndarray) -> None:
        codes = np.asarray(codes)
        futures = np.asarray(futures)
        if codes.ndim != 1:
            raise ExperimentError(f"codes must have shape [B], got {codes.shape}")
        if futures.ndim != 2 or futures.shape[1] != self.horizon:
            raise ExperimentError(
                f"futures must have shape [B,{self.horizon}], got {futures.shape}"
            )
        if len(codes) != len(futures):
            raise ExperimentError("codes/futures batch lengths differ")
        if len(codes) == 0:
            return
        if codes.dtype.kind not in "iu":
            raise ExperimentError("state codes must be integer-valued")
        if int(codes.min()) < 0 or int(codes.max()) >= self.n_prototypes:
            raise ExperimentError("state code is outside the frozen codebook")
        if not np.isfinite(futures).all():
            raise ExperimentError("nonfinite future path in historical summary pass")

        y64 = np.asarray(futures, dtype=np.float64)
        codes64 = np.asarray(codes, dtype=np.int64)
        r = int(self.spec.reservoir_per_state)

        for code in np.unique(codes64):
            rows = np.flatnonzero(codes64 == code)
            group = y64[rows]
            nb = int(group.shape[0])

            # Exact merge of this batch-group into the state's running moments.
            mean_b = group.mean(axis=0)
            centered = group - mean_b
            m2_b = np.sum(centered * centered, axis=0, dtype=np.float64)
            na = int(self.counts[code])
            nt = na + nb
            if na == 0:
                self.mean[code] = mean_b
                self.m2[code] = m2_b
            else:
                delta = mean_b - self.mean[code]
                self.mean[code] += delta * (nb / nt)
                self.m2[code] += m2_b + delta * delta * (na * nb / nt)
            self.counts[code] = nt

            # Uniform bounded reservoir by random priorities.
            new_keys = self._rng.random(nb)
            current_n = int(self.reservoir_counts[code])
            current_keys = self._reservoir_keys[code, :current_n]
            current_paths = self.reservoir[code, :current_n]
            candidate_keys = np.concatenate((current_keys, new_keys))
            candidate_paths = np.concatenate(
                (current_paths, group.astype(np.float32, copy=False)), axis=0
            )
            keep_n = min(r, candidate_keys.size)
            if keep_n < candidate_keys.size:
                keep = np.argpartition(candidate_keys, keep_n - 1)[:keep_n]
                keep = keep[np.argsort(candidate_keys[keep])]
            else:
                keep = np.argsort(candidate_keys)
            self._reservoir_keys[code].fill(np.inf)
            self.reservoir[code].fill(np.nan)
            self._reservoir_keys[code, :keep_n] = candidate_keys[keep]
            self.reservoir[code, :keep_n] = candidate_paths[keep]
            self.reservoir_counts[code] = keep_n

    def summary(self) -> FutureSummary:
        variance = np.full_like(self.mean, np.nan, dtype=np.float64)
        active = self.counts > 0
        variance[active] = self.m2[active] / self.counts[active, None]
        mean = self.mean.copy()
        mean[~active] = np.nan
        return FutureSummary(
            counts=self.counts.copy(),
            mean=mean,
            variance=variance,
            reservoir=self.reservoir.copy(),
            reservoir_counts=self.reservoir_counts.copy(),
        )


# ---------------------------------------------------------------------------
# Small artifact helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"expected JSON object in {path}")
    return value


def _check_fingerprint(value: dict, path: Path) -> None:
    expected = value.get("fingerprint")
    actual = fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
    if not isinstance(expected, str) or actual != expected:
        raise ExperimentError(f"artifact fingerprint mismatch: {path}")


def _check_supported_sources(corpus: dict, root: Path) -> None:
    records = corpus.get("sessions")
    if not isinstance(records, list) or not records:
        raise ExperimentError("corpus contains no sessions")
    if not root.is_dir():
        raise ExperimentError(f"data root does not exist: {root}")
    first = root / str(records[0].get("path", ""))
    if corpus.get("source_format") == "observation-chunks-v1":
        if not (first / "observations.json").is_file():
            raise ExperimentError(f"missing observation-chunks-v1 source at {first}")
    elif not (first / "observations.npz").is_file() or not (first / "provenance.json").is_file():
        raise ExperimentError(
            f"corpus expects ABIDES observation archives, but none was found at {first}"
        )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _heartbeat(label: str, count: int, last: float) -> float:
    now = time.monotonic()
    if now - last >= 30:
        print(f"{label}: {count:,}", flush=True)
        return now
    return last


def _atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(body), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise ExperimentError(
            f"artifact already exists: {path}. Use --force only for an intentional replacement."
        )


def _update_manifest(run_dir: Path, stage: str, payload: dict) -> None:
    path = run_dir / "manifest.json"
    manifest = _read_json(path) if path.exists() else {"version": 1, "stages": {}}
    manifest.setdefault("stages", {})[stage] = {"updated_at_utc": _utc_now(), **payload}
    _atomic_json(path, manifest)


# ---------------------------------------------------------------------------
# Data contract adapters


def _session_id(record: dict, fallback: int) -> str:
    for key in ("id", "timeline_id", "session_id"):
        if key in record:
            return str(record[key])
    return f"session-{fallback}"


def _iter_anchor_ids(record: dict, batch_size: int) -> Iterator[np.ndarray]:
    for start, stop in record["anchor_ranges"]:
        for offset in range(start, stop, batch_size):
            yield np.arange(offset, min(offset + batch_size, stop), dtype=np.int64)


def iter_unique_anchor_batches(
    corpus: dict,
    root: Path,
    preprocessing: dict,
    *,
    split: str,
    anchor_batch_size: int,
    include_y: bool,
) -> Iterator[AnchorBatch]:
    """Yield every valid anchor in a split exactly once.

    PairIndex is used only to obtain the corpus-authoritative session set for the
    requested split.  Histories are reconstructed directly from each observation
    table and transformed with the exact transform_rows call used by PairStream.
    """

    if anchor_batch_size < 1:
        raise ValueError("anchor_batch_size must be positive")
    try:
        history = int(corpus["history"])
        horizon = int(corpus["horizon"])
        names = tuple(corpus["names"])
        feature_spec = preprocessing["features"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError(
            "corpus/preprocessing missing history, horizon, names, or features"
        ) from exc

    index = PairIndex(corpus, split)
    records = index.records
    for session_index, record in enumerate(records):
        if corpus.get("source_format") == "observation-chunks-v1":
            from mbo_lab.observation_store import ObservationReader

            source = (root / record["path"]).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ExperimentError("source escapes data root")
            reader = ObservationReader(source, cache_bytes=int(record["max_chunk_bytes"]))
            if reader.manifest["fingerprint"] != record["source_fingerprint"]:
                raise ExperimentError(f"changed observation source: {source}")
            sid = _session_id(record, session_index)
            batch_size = min(anchor_batch_size, 512)
            for a in _iter_anchor_ids(record, batch_size):
                if include_y:
                    raw, future = reader.episodes(a, history=history, horizon=horizon)
                else:
                    raw = reader.histories(a, history=history, horizon=horizon)
                    future = None
                normalized = transform_rows(
                    raw.reshape(-1, len(names)), reader.manifest["names"], feature_spec
                )
                X = np.asarray(
                    normalized.reshape(len(a), history, len(names)),
                    dtype=np.float32,
                    order="C",
                )
                Y = np.asarray(future, dtype=np.float64, order="C") if future is not None else None
                if not np.isfinite(X).all() or (Y is not None and not np.isfinite(Y).all()):
                    raise ExperimentError(f"nonfinite episode in {sid}")
                yield AnchorBatch(X=X, Y=Y, session_id=sid, anchors=a.copy())
            continue
        obs = verified_source(root, record)
        if tuple(obs.names) != names:
            raise ExperimentError(
                f"feature schema mismatch in {_session_id(record, session_index)}"
            )
        sid = _session_id(record, session_index)
        for a in _iter_anchor_ids(record, anchor_batch_size):
            past = a[:, None] + np.arange(1 - history, 1)
            raw = obs.x[past]
            normalized = transform_rows(raw.reshape(-1, len(names)), obs.names, feature_spec)
            X = np.asarray(
                normalized.reshape(len(a), history, len(names)), dtype=np.float32, order="C"
            )
            if not np.isfinite(X).all():
                raise ExperimentError(f"nonfinite transformed history in {sid}")

            Y = None
            if include_y:
                future = a[:, None] + np.arange(1, horizon + 1)
                Y = np.log(obs.mid[future] / obs.mid[a, None])
                Y = np.asarray(Y, dtype=np.float64, order="C")
                if not np.isfinite(Y).all():
                    raise ExperimentError(f"nonfinite future path in {sid}")
            yield AnchorBatch(X=X, Y=Y, session_id=sid, anchors=a.copy())


def _pair_stream(
    corpus: dict,
    root: Path,
    preprocessing: dict,
    train_spec: TrainSpec,
    *,
    split: str,
    seed: int,
    queue_pairs: int | None = None,
    batch_pairs: int | None = None,
    workers: int | None = None,
    progress=None,
):
    return open_pair_stream(
        corpus,
        root,
        preprocessing,
        split=split,
        seed=seed,
        queue_pairs=train_spec.queue_pairs if queue_pairs is None else queue_pairs,
        batch_pairs=train_spec.batch_pairs if batch_pairs is None else batch_pairs,
        memory_bytes=train_spec.stream_memory_bytes,
        workers=train_spec.workers if workers is None else workers,
        progress=progress,
    )


def _queue_progress(label: str, plot=None):
    last = [time.monotonic()]

    def update(completed: int, total: int) -> None:
        if plot:
            plot.pump()
        now = time.monotonic()
        if now - last[0] >= 30:
            print(f"{label}: {completed}/{total} session groups", flush=True)
            last[0] = now

    return update


# ---------------------------------------------------------------------------
# Encoder/checkpoint lifecycle


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_encoder_from_args(args, corpus: dict):
    spec = EncoderSpec(
        architecture=args.architecture,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        bias=not args.no_bias,
    )
    return build_encoder(
        spec,
        history=int(corpus["history"]),
        features=len(corpus["names"]),
    )


def _load_encoder(run_dir: Path, device: torch.device):
    path = run_dir / "encoder" / "checkpoint.pt"
    if not path.exists():
        raise ExperimentError(f"missing trained encoder checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        spec = EncoderSpec(**checkpoint["encoder_spec"])
        encoder = build_encoder(
            spec,
            history=int(checkpoint["history"]),
            features=int(checkpoint["features"]),
        )
        encoder.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ExperimentError(f"invalid encoder checkpoint {path}: {exc}") from exc
    encoder.to(device)
    encoder.eval()
    return encoder, checkpoint


def _check_checkpoint_data(checkpoint: dict, corpus: dict, preprocessing: dict) -> None:
    expected_corpus = corpus.get("fingerprint")
    expected_pre = preprocessing.get("fingerprint")
    if checkpoint.get("corpus_fingerprint") != expected_corpus:
        raise ExperimentError("encoder checkpoint/corpus fingerprint mismatch")
    if checkpoint.get("preprocessing_fingerprint") != expected_pre:
        raise ExperimentError("encoder checkpoint/preprocessing fingerprint mismatch")


def evaluate_pair_distance(
    encoder,
    corpus: dict,
    root: Path,
    preprocessing: dict,
    train_spec: TrainSpec,
    *,
    split: str,
    max_pairs: int,
    seed: int,
    device: torch.device,
    expected_pair_ids: np.ndarray | None = None,
) -> dict:
    if max_pairs <= 0:
        return {"split": split, "pairs": 0, "skipped": True}

    total_pairs = 0
    sum_sq_error = 0.0
    sum_abs_error = 0.0
    sum_pred = 0.0
    sum_target = 0.0
    encoder.eval()
    evaluation_queue_pairs = min(train_spec.queue_pairs, max_pairs)
    evaluation_batch_pairs = min(train_spec.batch_pairs, evaluation_queue_pairs)
    with _pair_stream(
        corpus,
        root,
        preprocessing,
        train_spec,
        split=split,
        seed=seed,
        queue_pairs=evaluation_queue_pairs,
        batch_pairs=evaluation_batch_pairs,
    ) as stream:
        with torch.inference_mode():
            for batch in stream:
                remaining = max_pairs - total_pairs
                if remaining <= 0:
                    break
                pair_count = min(len(batch.D), remaining)
                if pair_count <= 0:
                    break
                if expected_pair_ids is not None:
                    expected = expected_pair_ids[total_pairs : total_pairs + pair_count]
                    if not np.array_equal(batch.pair_ids[:pair_count], expected):
                        raise ExperimentError("validation stream disagrees with frozen pair IDs")

                X = torch.as_tensor(batch.X, dtype=torch.float32, device=device)
                pairs = torch.as_tensor(batch.pairs[:pair_count], dtype=torch.long, device=device)
                target = torch.as_tensor(batch.D[:pair_count], dtype=torch.float32, device=device)
                z = encoder(X)
                pred = torch.linalg.vector_norm(z[pairs[:, 0]] - z[pairs[:, 1]], dim=-1)
                err = pred - target
                total_pairs += pair_count
                sum_sq_error += float(torch.sum(err * err).cpu())
                sum_abs_error += float(torch.sum(torch.abs(err)).cpu())
                sum_pred += float(torch.sum(pred).cpu())
                sum_target += float(torch.sum(target).cpu())
                if total_pairs >= max_pairs:
                    break

    if total_pairs == 0:
        raise ExperimentError(f"no pair batches available for split={split!r}")
    return {
        "split": split,
        "pairs": total_pairs,
        "distance_mse": sum_sq_error / total_pairs,
        "distance_rmse": math.sqrt(sum_sq_error / total_pairs),
        "distance_mae": sum_abs_error / total_pairs,
        "mean_predicted_distance": sum_pred / total_pairs,
        "mean_target_distance": sum_target / total_pairs,
    }


def _frozen_validation_sample(args, corpus: dict, count: int) -> tuple[int, np.ndarray]:
    path = args.corpus.parent / "validation_pairs.json"
    sample = _read_json(path)
    _check_fingerprint(sample, path)
    if (
        sample.get("corpus") != corpus.get("fingerprint")
        or sample.get("split") != "validation"
        or sample.get("traversal_version") != PairTraversal.version
    ):
        raise ExperimentError(f"incompatible frozen validation sample: {path}")
    pair_ids = np.asarray(sample.get("pair_ids"), dtype=np.int64)
    if pair_ids.ndim != 1 or len(pair_ids) != sample.get("pair_count"):
        raise ExperimentError(f"invalid frozen validation pair IDs: {path}")
    if count > len(pair_ids):
        raise ExperimentError(
            f"validation-pairs={count} exceeds the frozen sample of {len(pair_ids)} pairs"
        )
    return int(sample["seed"]), pair_ids[:count]


def _load_validation_probe(
    args,
    corpus: dict,
    preprocessing: dict,
    spec: TrainSpec,
    plot=None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int, np.ndarray]:
    count = spec.validation_probe_pairs
    seed, expected_ids = _frozen_validation_sample(args, corpus, count)
    batches = []
    seen = 0
    probe_batch_pairs = min(spec.batch_pairs, count)
    with _pair_stream(
        corpus,
        args.data_root,
        preprocessing,
        spec,
        split="validation",
        seed=seed,
        queue_pairs=count,
        batch_pairs=probe_batch_pairs,
        progress=_queue_progress("Validation probe loading", plot),
    ) as stream:
        for batch in stream:
            pair_count = min(len(batch.D), count - seen)
            if pair_count <= 0:
                break
            expected = expected_ids[seen : seen + pair_count]
            if not np.array_equal(batch.pair_ids[:pair_count], expected):
                raise ExperimentError("validation probe disagrees with frozen pair IDs")
            batches.append(
                (
                    torch.as_tensor(batch.X, dtype=torch.float32),
                    torch.as_tensor(batch.pairs[:pair_count], dtype=torch.long),
                    torch.as_tensor(batch.D[:pair_count], dtype=torch.float32),
                )
            )
            seen += pair_count
            if seen >= count:
                break
    if seen != count:
        raise ExperimentError(f"validation probe produced {seen} of {count} requested pairs")
    return batches, seed, expected_ids


def _evaluate_validation_probe(encoder, batches, device: torch.device) -> float:
    was_training = encoder.training
    encoder.eval()
    total = 0
    sum_sq_error = 0.0
    with torch.inference_mode():
        for X, pairs, target in batches:
            X = X.to(device)
            pairs = pairs.to(device)
            target = target.to(device)
            z = encoder(X)
            pred = torch.linalg.vector_norm(z[pairs[:, 0]] - z[pairs[:, 1]], dim=-1)
            error = pred - target
            total += len(target)
            sum_sq_error += float(torch.sum(error * error).cpu())
    encoder.train(was_training)
    return sum_sq_error / total


class _LiveLearningCurve:
    def __init__(self) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ExperimentError("--live-plot requires matplotlib") from exc
        if matplotlib.get_backend().lower() == "agg":
            backend = matplotlib.get_backend()
            raise ExperimentError(
                f"--live-plot needs an interactive matplotlib backend, got {backend}"
            )
        self._plt = plt
        self._figure, self._axis = plt.subplots(num="BookSpace validation learning curve")
        (self._train_line,) = self._axis.plot([], [], label="training MSE", color="#4C78A8")
        (self._validation_line,) = self._axis.plot(
            [], [], label="validation probe MSE", color="#E45756", marker="o"
        )
        self._axis.set_xlabel("training step")
        self._axis.set_ylabel("distance MSE")
        self._axis.set_title("Encoder learning curve")
        self._axis.grid(alpha=0.25)
        self._axis.legend()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, history: list[dict]) -> None:
        if not self._plt.fignum_exists(self._figure.number):
            return
        training = [point for point in history if point["train_mse"] is not None]
        self._train_line.set_data(
            [point["step"] for point in training], [point["train_mse"] for point in training]
        )
        self._validation_line.set_data(
            [point["step"] for point in history],
            [point["validation_probe_mse"] for point in history],
        )
        self._axis.relim()
        self._axis.autoscale_view()
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    def pump(self) -> None:
        if self._plt.fignum_exists(self._figure.number):
            self._figure.canvas.flush_events()


def train_encoder_stage(args, corpus: dict, preprocessing: dict) -> None:
    run_dir = args.run_dir
    checkpoint_path = run_dir / "encoder" / "checkpoint.pt"
    metrics_path = run_dir / "encoder" / "metrics.json"
    _ensure_can_write(checkpoint_path, args.force)
    _ensure_can_write(metrics_path, args.force)

    device = _device(args.device)
    _seed_everything(args.seed)
    encoder = _build_encoder_from_args(args, corpus).to(device)
    encoder.train()
    spec = TrainSpec(
        steps=args.train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        batch_pairs=args.batch_pairs,
        queue_pairs=args.queue_pairs,
        stream_memory_bytes=args.stream_memory_gib * 1024**3,
        workers=args.workers,
        validation_pairs=args.validation_pairs,
        test_pairs=args.test_pairs,
        validation_interval=args.validation_interval,
        validation_probe_pairs=args.validation_probe_pairs,
    )
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )

    running_loss = 0.0
    running_pairs = 0
    last_loss = math.nan
    completed_steps = 0
    heartbeat = time.monotonic()
    curve = []
    plot = _LiveLearningCurve() if args.live_plot else None
    probe = None
    probe_seed = None
    probe_ids = None
    interval_loss = 0.0
    interval_pairs = 0
    if spec.validation_interval:
        print(
            f"Loading fixed validation probe: {spec.validation_probe_pairs:,} pairs",
            flush=True,
        )
        started = time.perf_counter()
        probe, probe_seed, probe_ids = _load_validation_probe(
            args, corpus, preprocessing, spec, plot
        )
        print(f"Validation probe loaded in {time.perf_counter() - started:.1f}s", flush=True)
        curve.append(
            {
                "step": 0,
                "train_mse": None,
                "validation_probe_mse": _evaluate_validation_probe(encoder, probe, device),
            }
        )
        if plot:
            plot.update(curve)

    training_queue_pairs = min(spec.queue_pairs, spec.steps * spec.batch_pairs)
    print(f"Loading training queue: {training_queue_pairs:,} pairs", flush=True)
    with _pair_stream(
        corpus,
        args.data_root,
        preprocessing,
        spec,
        split="train",
        seed=args.seed,
        queue_pairs=training_queue_pairs,
        progress=_queue_progress("Training queue loading", plot),
    ) as stream:
        for batch in stream:
            X = torch.as_tensor(batch.X, dtype=torch.float32, device=device)
            pairs = torch.as_tensor(batch.pairs, dtype=torch.long, device=device)
            target = torch.as_tensor(batch.D, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            z = encoder(X)
            loss = distance_mse(z[pairs[:, 0]], z[pairs[:, 1]], target)
            if not torch.isfinite(loss):
                raise ExperimentError("metric loss became nonfinite")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), spec.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise ExperimentError("encoder gradient norm became nonfinite")
            optimizer.step()

            pair_count = int(len(batch.D))
            last_loss = float(loss.detach().cpu())
            running_loss += last_loss * pair_count
            running_pairs += pair_count
            interval_loss += last_loss * pair_count
            interval_pairs += pair_count
            completed_steps += 1
            heartbeat = _heartbeat("Training steps", completed_steps, heartbeat)
            if plot:
                plot.pump()
            if spec.validation_interval and completed_steps % spec.validation_interval == 0:
                curve.append(
                    {
                        "step": completed_steps,
                        "train_mse": interval_loss / interval_pairs,
                        "validation_probe_mse": _evaluate_validation_probe(encoder, probe, device),
                    }
                )
                interval_loss = 0.0
                interval_pairs = 0
                if plot:
                    plot.update(curve)
            if completed_steps >= spec.steps:
                break

        stream_report = stream.report()

    if completed_steps == 0:
        raise ExperimentError("training PairStream produced no batches")
    if spec.validation_interval and curve[-1]["step"] != completed_steps:
        curve.append(
            {
                "step": completed_steps,
                "train_mse": interval_loss / interval_pairs,
                "validation_probe_mse": _evaluate_validation_probe(encoder, probe, device),
            }
        )
        if plot:
            plot.update(curve)

    checkpoint = {
        "version": 1,
        "created_at_utc": _utc_now(),
        "encoder_spec": encoder.spec.as_dict(),
        "history": encoder.history,
        "features": encoder.features,
        "latent_dim": encoder.latent_dim,
        "model_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": completed_steps,
        "train_spec": asdict(spec),
        "seed": args.seed,
        "corpus_fingerprint": corpus.get("fingerprint"),
        "preprocessing_fingerprint": preprocessing.get("fingerprint"),
    }
    _atomic_torch_save(checkpoint_path, checkpoint)

    encoder.eval()
    validation_seed, validation_ids = _frozen_validation_sample(args, corpus, spec.validation_pairs)
    validation = evaluate_pair_distance(
        encoder,
        corpus,
        args.data_root,
        preprocessing,
        spec,
        split="validation",
        max_pairs=spec.validation_pairs,
        seed=validation_seed,
        device=device,
        expected_pair_ids=validation_ids,
    )
    test_diag = evaluate_pair_distance(
        encoder,
        corpus,
        args.data_root,
        preprocessing,
        spec,
        split="test",
        max_pairs=spec.test_pairs,
        seed=args.seed + 202,
        device=device,
    )
    metrics = {
        "completed_steps": completed_steps,
        "train_pairs_seen": running_pairs,
        "train_pair_weighted_mean_loss": running_loss / running_pairs,
        "last_train_loss": last_loss,
        "stream_report": stream_report,
        "validation": validation,
        "learning_curve": {
            "interval": spec.validation_interval,
            "probe_pairs": spec.validation_probe_pairs if spec.validation_interval else 0,
            "probe_seed": probe_seed,
            "probe_pair_ids": probe_ids,
            "points": curve,
        },
        # This is diagnostic only. Do not use this value for tuning after a final
        # experiment protocol is frozen; the final test set remains untouched by
        # model selection.
        "test_distance_diagnostic": test_diag,
    }
    _atomic_json(metrics_path, metrics)
    _update_manifest(
        run_dir,
        "train",
        {"checkpoint": str(checkpoint_path), "metrics": str(metrics_path)},
    )
    print(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# State compression


def compress_stage(args, corpus: dict, preprocessing: dict) -> None:
    run_dir = args.run_dir
    output = run_dir / "compression" / "compressor.npz"
    _ensure_can_write(output, args.force)

    device = _device(args.device)
    encoder, checkpoint = _load_encoder(run_dir, device)
    _check_checkpoint_data(checkpoint, corpus, preprocessing)

    spec = CompressionSpec(
        n_prototypes=args.n_prototypes,
        batch_size=args.compressor_batch_size,
        init_size=args.compressor_init_size,
        n_init=args.compressor_n_init,
        reassignment_ratio=args.reassignment_ratio,
        seed=args.seed,
    )
    compressor = build_compressor(spec, latent_dim=encoder.latent_dim)

    anchor_count = 0
    heartbeat = time.monotonic()
    encoder.eval()
    with torch.inference_mode():
        for batch in iter_unique_anchor_batches(
            corpus,
            args.data_root,
            preprocessing,
            split="train",
            anchor_batch_size=args.anchor_batch_size,
            include_y=False,
        ):
            X = torch.as_tensor(batch.X, dtype=torch.float32, device=device)
            z = encoder(X)
            compressor.partial_fit(z)
            anchor_count += len(batch.X)
            heartbeat = _heartbeat("Compression anchors", anchor_count, heartbeat)

    compressor.finalize()
    compressor.save(output)
    metadata = {
        "training_anchors": anchor_count,
        "n_prototypes": compressor.n_prototypes,
        "latent_dim": compressor.latent_dim,
        "fit_observation_count": compressor.fit_observation_count,
        "spec": asdict(spec),
    }
    _atomic_json(run_dir / "compression" / "fit.json", metadata)
    _update_manifest(run_dir, "compress", {"compressor": str(output), **metadata})
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _save_future_summary(path: Path, summary: FutureSummary) -> None:
    _atomic_npz(
        path,
        counts=np.asarray(summary.counts, dtype=np.int64),
        mean=np.asarray(summary.mean, dtype=np.float64),
        variance=np.asarray(summary.variance, dtype=np.float64),
        reservoir=np.asarray(summary.reservoir, dtype=np.float32),
        reservoir_counts=np.asarray(summary.reservoir_counts, dtype=np.int32),
    )


def _load_future_summary(path: Path) -> FutureSummary:
    if not path.exists():
        raise ExperimentError(f"missing future summary artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {"counts", "mean", "variance", "reservoir", "reservoir_counts"}
            if set(artifact.files) != required:
                raise ExperimentError(f"unexpected future-summary fields: {sorted(artifact.files)}")
            summary = FutureSummary(
                counts=np.asarray(artifact["counts"], dtype=np.int64).copy(),
                mean=np.asarray(artifact["mean"], dtype=np.float64).copy(),
                variance=np.asarray(artifact["variance"], dtype=np.float64).copy(),
                reservoir=np.asarray(artifact["reservoir"], dtype=np.float32).copy(),
                reservoir_counts=np.asarray(artifact["reservoir_counts"], dtype=np.int32).copy(),
            )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"cannot load future summary {path}: {exc}") from exc
    if summary.mean.ndim != 2 or summary.counts.shape != (summary.mean.shape[0],):
        raise ExperimentError("invalid future-summary count/mean shapes")
    if summary.variance.shape != summary.mean.shape:
        raise ExperimentError("invalid future-summary variance shape")
    if summary.reservoir.ndim != 3 or summary.reservoir.shape[0] != summary.n_prototypes:
        raise ExperimentError("invalid future-summary reservoir shape")
    if summary.reservoir.shape[2] != summary.horizon:
        raise ExperimentError("future-summary reservoir horizon mismatch")
    if summary.reservoir_counts.shape != (summary.n_prototypes,):
        raise ExperimentError("invalid future-summary reservoir-count shape")
    return summary


def summarize_stage(args, corpus: dict, preprocessing: dict) -> None:
    run_dir = args.run_dir
    output = run_dir / "future" / "state_future_stats.npz"
    diagnostics_path = run_dir / "compression" / "assignment_metrics.json"
    _ensure_can_write(output, args.force)
    _ensure_can_write(diagnostics_path, args.force)

    device = _device(args.device)
    encoder, checkpoint = _load_encoder(run_dir, device)
    _check_checkpoint_data(checkpoint, corpus, preprocessing)
    compressor_path = run_dir / "compression" / "compressor.npz"
    if not compressor_path.exists():
        raise ExperimentError(f"missing compressor artifact: {compressor_path}")
    compressor = load_compressor(compressor_path)
    if compressor.latent_dim != encoder.latent_dim:
        raise ExperimentError("compressor/encoder latent dimension mismatch")

    horizon = int(corpus["horizon"])
    future_acc = StateFutureAccumulator(
        compressor.n_prototypes,
        horizon,
        FutureSummarySpec(
            reservoir_per_state=args.reservoir_per_state,
            seed=args.seed + 303,
        ),
    )
    compression_stats = compressor.make_stats_accumulator(
        reservoir_size=args.distance_reservoir_size,
        seed=args.seed + 404,
    )

    anchor_count = 0
    heartbeat = time.monotonic()
    encoder.eval()
    with torch.inference_mode():
        for batch in iter_unique_anchor_batches(
            corpus,
            args.data_root,
            preprocessing,
            split="train",
            anchor_batch_size=args.anchor_batch_size,
            include_y=True,
        ):
            assert batch.Y is not None
            X = torch.as_tensor(batch.X, dtype=torch.float32, device=device)
            z = encoder(X)
            z_cpu = z.detach().to(device="cpu", dtype=torch.float32).numpy()
            codes = compressor.encode(z_cpu)
            compression_stats.update(z_cpu, codes)
            future_acc.update(codes, batch.Y)
            anchor_count += len(codes)
            heartbeat = _heartbeat("Summary anchors", anchor_count, heartbeat)

    summary = future_acc.summary()
    _save_future_summary(output, summary)
    cstats = compression_stats.summary()
    diagnostics = {
        **asdict(cstats),
        "training_anchors": anchor_count,
        "future_summary": {
            "active_states": int(np.count_nonzero(summary.counts)),
            "total_state_occurrences": int(summary.counts.sum()),
            "min_active_support": int(summary.counts[summary.counts > 0].min()),
            "median_active_support": float(np.median(summary.counts[summary.counts > 0])),
            "max_support": int(summary.counts.max()),
            "reservoir_per_state": args.reservoir_per_state,
        },
    }
    _atomic_json(diagnostics_path, diagnostics)
    _update_manifest(
        run_dir,
        "summarize",
        {"future_summary": str(output), "assignment_metrics": str(diagnostics_path)},
    )
    print(json.dumps(_jsonable(diagnostics), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Compressed held-out evaluation


def _squared_path_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean((pred - truth) ** 2, axis=1, dtype=np.float64)


def evaluate_compressed_split(
    args,
    corpus: dict,
    preprocessing: dict,
    *,
    split: str,
) -> dict:
    run_dir = args.run_dir
    device = _device(args.device)
    encoder, checkpoint = _load_encoder(run_dir, device)
    _check_checkpoint_data(checkpoint, corpus, preprocessing)
    compressor = load_compressor(run_dir / "compression" / "compressor.npz")
    summary = _load_future_summary(run_dir / "future" / "state_future_stats.npz")
    if summary.n_prototypes != compressor.n_prototypes:
        raise ExperimentError("future summary/compressor prototype-count mismatch")
    if summary.horizon != int(corpus["horizon"]):
        raise ExperimentError("future summary/corpus horizon mismatch")

    global_mean = summary.global_mean
    total = 0
    supported = 0
    unsupported = 0
    state_sse = 0.0
    state_abs_terminal = 0.0
    state_sq_terminal = 0.0
    state_sum_rms = 0.0
    state_support_sum = 0.0
    state_distance_sum = 0.0
    state_distance_max = 0.0
    heartbeat = time.monotonic()

    global_sse = 0.0
    global_abs_terminal = 0.0
    global_sq_terminal = 0.0
    zero_sse = 0.0
    zero_abs_terminal = 0.0
    zero_sq_terminal = 0.0

    encoder.eval()
    with torch.inference_mode():
        for batch in iter_unique_anchor_batches(
            corpus,
            args.data_root,
            preprocessing,
            split=split,
            anchor_batch_size=args.anchor_batch_size,
            include_y=True,
        ):
            assert batch.Y is not None
            Y = batch.Y
            X = torch.as_tensor(batch.X, dtype=torch.float32, device=device)
            z = encoder(X)
            z_cpu = z.detach().to(device="cpu", dtype=torch.float32).numpy()
            codes = compressor.encode(z_cpu)

            # Baselines are defined for every held-out anchor.
            global_pred = np.broadcast_to(global_mean, Y.shape)
            global_e = _squared_path_error(global_pred, Y)
            global_sse += float(np.sum(global_e, dtype=np.float64))
            global_term = global_mean[-1] - Y[:, -1]
            global_abs_terminal += float(np.sum(np.abs(global_term), dtype=np.float64))
            global_sq_terminal += float(np.sum(global_term * global_term, dtype=np.float64))

            zero_e = np.mean(Y * Y, axis=1, dtype=np.float64)
            zero_sse += float(np.sum(zero_e, dtype=np.float64))
            zero_abs_terminal += float(np.sum(np.abs(Y[:, -1]), dtype=np.float64))
            zero_sq_terminal += float(np.sum(Y[:, -1] * Y[:, -1], dtype=np.float64))

            support_mask = summary.counts[codes] > 0
            batch_supported = int(np.count_nonzero(support_mask))
            supported += batch_supported
            unsupported += int(len(codes) - batch_supported)
            total += len(codes)
            heartbeat = _heartbeat(f"{split.title()} anchors", total, heartbeat)

            if batch_supported:
                scodes = codes[support_mask]
                truth = Y[support_mask]
                pred = summary.mean[scodes]
                e = _squared_path_error(pred, truth)
                state_sse += float(np.sum(e, dtype=np.float64))
                state_sum_rms += float(np.sum(np.sqrt(e), dtype=np.float64))
                terminal = pred[:, -1] - truth[:, -1]
                state_abs_terminal += float(np.sum(np.abs(terminal), dtype=np.float64))
                state_sq_terminal += float(np.sum(terminal * terminal, dtype=np.float64))
                state_support_sum += float(np.sum(summary.counts[scodes], dtype=np.float64))

                selected = compressor.prototypes[scodes]
                delta = z_cpu[support_mask] - selected
                distances = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64))
                state_distance_sum += float(np.sum(distances, dtype=np.float64))
                state_distance_max = max(state_distance_max, float(np.max(distances)))

    if total == 0:
        raise ExperimentError(f"split={split!r} contains no eligible anchors")

    metrics = {
        "split": split,
        "queries": total,
        "supported_queries": supported,
        "unsupported_queries": unsupported,
        "coverage": supported / total,
        "state_mean_predictor": None,
        "global_training_mean_baseline": {
            "path_mse": global_sse / total,
            "path_rmse": math.sqrt(global_sse / total),
            "terminal_mae": global_abs_terminal / total,
            "terminal_rmse": math.sqrt(global_sq_terminal / total),
        },
        "zero_return_path_baseline": {
            "path_mse": zero_sse / total,
            "path_rmse": math.sqrt(zero_sse / total),
            "terminal_mae": zero_abs_terminal / total,
            "terminal_rmse": math.sqrt(zero_sq_terminal / total),
        },
    }
    if supported:
        metrics["state_mean_predictor"] = {
            "path_mse_on_supported": state_sse / supported,
            "path_rmse_on_supported": math.sqrt(state_sse / supported),
            "mean_per_query_path_rms_error": state_sum_rms / supported,
            "terminal_mae_on_supported": state_abs_terminal / supported,
            "terminal_rmse_on_supported": math.sqrt(state_sq_terminal / supported),
            "mean_training_support_of_selected_state": state_support_sum / supported,
            "mean_query_to_state_distance": state_distance_sum / supported,
            "max_query_to_state_distance": state_distance_max,
        }
    return metrics


def evaluate_stage(args, corpus: dict, preprocessing: dict, split: str) -> None:
    output = args.run_dir / split / "metrics.json"
    _ensure_can_write(output, args.force)
    metrics = evaluate_compressed_split(args, corpus, preprocessing, split=split)
    _atomic_json(output, metrics)
    _update_manifest(args.run_dir, split, {"metrics": str(output), **metrics})
    print(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("train", "compress", "summarize", "validate", "test", "all"),
    )
    parser.add_argument("--corpus", type=Path, required=True, help="frozen corpus JSON")
    parser.add_argument(
        "--preprocessing", type=Path, required=True, help="train-only preprocessing JSON"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="root used by mbo_lab.corpus.verified_source(root, record)",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")

    # Encoder / metric training.
    parser.add_argument("--architecture", default="linear")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-pairs", type=int, default=32)
    parser.add_argument("--queue-pairs", type=int, default=16_384)
    parser.add_argument("--stream-memory-gib", type=int, default=6)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--validation-pairs", type=int, default=16_384)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=0,
        help="evaluate the fixed validation probe every N training steps; 0 disables it",
    )
    parser.add_argument("--validation-probe-pairs", type=int, default=256)
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="show an updating native learning-curve window during encoder training",
    )
    parser.add_argument(
        "--test-pairs",
        type=int,
        default=0,
        help="pair-distance diagnostic only; default 0 keeps final test untouched during training",
    )

    # Unique-anchor replay / VQ.
    parser.add_argument("--anchor-batch-size", type=int, default=128)
    parser.add_argument("--n-prototypes", type=int, default=1024)
    parser.add_argument("--compressor-batch-size", type=int, default=8192)
    parser.add_argument("--compressor-init-size", type=int, default=None)
    parser.add_argument("--compressor-n-init", type=int, default=3)
    parser.add_argument("--reassignment-ratio", type=float, default=0.01)

    # Tiny per-state future summary.
    parser.add_argument("--reservoir-per-state", type=int, default=128)
    parser.add_argument("--distance-reservoir-size", type=int, default=65_536)

    return parser.parse_args()


def _write_run_config(args: argparse.Namespace, corpus: dict, preprocessing: dict) -> None:
    path = args.run_dir / "config.json"
    if path.exists() and not args.force:
        return
    body = {
        "created_at_utc": _utc_now(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "corpus_fingerprint": corpus.get("fingerprint"),
        "preprocessing_fingerprint": preprocessing.get("fingerprint"),
        "history": corpus.get("history"),
        "horizon": corpus.get("horizon"),
        "features": len(corpus.get("names", [])),
    }
    _atomic_json(path, body)


def main() -> None:
    args = _parse_args()
    if args.seed < 0:
        raise ExperimentError("seed must be nonnegative")
    if args.stage in ("train", "all") and args.live_plot and args.validation_interval < 1:
        raise ExperimentError("--live-plot requires --validation-interval greater than zero")
    args.run_dir = args.run_dir.resolve()
    args.data_root = args.data_root.resolve()
    corpus_path = args.corpus.resolve()
    preprocessing_path = args.preprocessing.resolve()
    args.corpus = corpus_path
    args.preprocessing = preprocessing_path
    corpus = _read_json(corpus_path)
    preprocessing = _read_json(preprocessing_path)
    _check_fingerprint(corpus, corpus_path)
    _check_fingerprint(preprocessing, preprocessing_path)
    if preprocessing.get("corpus") != corpus.get("fingerprint"):
        raise ExperimentError("preprocessing/corpus fingerprint mismatch")
    _check_supported_sources(corpus, args.data_root)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args, corpus, preprocessing)

    stage: Stage = args.stage
    if stage in ("train", "all"):
        train_encoder_stage(args, corpus, preprocessing)
    if stage in ("compress", "all"):
        compress_stage(args, corpus, preprocessing)
    if stage in ("summarize", "all"):
        summarize_stage(args, corpus, preprocessing)
    if stage in ("validate", "all"):
        evaluate_stage(args, corpus, preprocessing, "validation")
    if stage == "test":
        evaluate_stage(args, corpus, preprocessing, "test")


if __name__ == "__main__":
    main()
