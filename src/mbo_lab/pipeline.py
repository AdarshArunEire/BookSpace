"""Reusable ABIDES observation and training-sample pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from mbo_lab.abides import run_abides_export
from mbo_lab.metrics.fixed import TargetScale, future_pair_distances, future_path_rms, latent_l2
from mbo_lab.observations import (
    FEATURE_SCHEMA_VERSION,
    TIME_OF_DAY_TIMEZONE,
    Observations,
    feature_names,
    load_observations,
)
from mbo_lab.paths import abides_paths, require_data_path
from mbo_lab.samples import FeatureScale, batch, eligible_pairs, valid_anchors


@dataclass
class TrainingBatch:
    """In-memory outputs shared by the notebook, tests, and CLI wrapper."""

    observations: Observations
    feature_scale: FeatureScale
    target_scale: TargetScale
    training_anchors: np.ndarray
    validation_anchors: np.ndarray
    anchors: np.ndarray
    training_rows: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    pairs: np.ndarray
    D: np.ndarray
    D_raw: np.ndarray
    query_X: np.ndarray
    query_Y: np.ndarray
    report: dict


def load_abides_source(source):
    """Load and verify one exported ABIDES observation directory."""
    source = Path(source).resolve()
    observation_path = source / "observations.npz"
    provenance_path = source / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    observation_hash = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    if observation_hash != provenance["observations_sha256"]:
        raise ValueError("ABIDES observation hash mismatch")
    if provenance.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("ABIDES provenance has an unsupported feature schema version")
    observations = load_observations(observation_path)
    expected_names = feature_names(10)
    if observations.names != expected_names or observations.x.shape[1] != len(expected_names):
        raise ValueError(f"ABIDES source must use the full {len(expected_names)}-feature schema")
    return observations, provenance, observation_path, provenance_path


def _training_rows(anchors, history, observation_count):
    coverage = np.zeros(observation_count + 1, dtype=np.int64)
    np.add.at(coverage, anchors - (history - 1), 1)
    np.add.at(coverage, anchors + 1, -1)
    return np.flatnonzero(np.cumsum(coverage[:-1]) > 0)


def _select_anchors(training_anchors, anchor_count, seed):
    training_anchors = np.asarray(training_anchors, dtype=np.int64)
    if anchor_count is None:
        return training_anchors.copy()
    if not isinstance(anchor_count, (int, np.integer)) or anchor_count < 2:
        raise ValueError("anchor_count must be at least 2 or None")
    if anchor_count > len(training_anchors):
        raise ValueError("anchor_count exceeds the available training anchors")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(training_anchors, size=anchor_count, replace=False))


def build_training_batch(
    observations,
    *,
    provenance=None,
    source_dir=None,
    seed=0,
    history=256,
    horizon=128,
    anchor_count=64,
    split_row=None,
):
    """Build X, Y, and scalar pair targets from an observation table."""
    split = len(observations.mid) * 2 // 3 if split_row is None else int(split_row)
    training_anchors = valid_anchors(observations, history, horizon, stop=split)
    validation_anchors = valid_anchors(observations, history, horizon, start=split)
    if not len(training_anchors) or not len(validation_anchors):
        raise ValueError("Need complete training and validation episodes")
    anchors = _select_anchors(training_anchors, anchor_count, seed)
    pairs = eligible_pairs(anchors, history, horizon)
    if not len(pairs):
        raise ValueError("Selected anchors do not produce an eligible pair")
    rows = _training_rows(training_anchors, history, len(observations.mid))
    feature_scale = FeatureScale.fit(observations, rows)
    transformed = replace(observations, x=feature_scale.transform(observations))
    X, Y = batch(transformed, anchors, history, horizon, stop=split)
    D_raw = future_pair_distances(Y, pairs)
    target_scale = TargetScale.fit(D_raw, seed=seed)
    D = future_pair_distances(Y, pairs, scale=target_scale)
    query_X, query_Y = batch(
        transformed,
        validation_anchors[:1],
        history,
        horizon,
        start=split,
    )
    distances = latent_l2(
        X.reshape(len(X), -1),
        np.broadcast_to(query_X.reshape(1, -1), (len(X), X[0].size)),
    )
    nearest = int(np.argmin(distances))
    source_dir = None if source_dir is None else Path(source_dir).resolve()
    observation_path = None if source_dir is None else source_dir / "observations.npz"
    provenance_path = None if source_dir is None else source_dir / "provenance.json"
    report = {
        "source": (provenance or {}).get("source", "ABIDES RMSC04"),
        "synthetic": bool((provenance or {}).get("synthetic", observations.synthetic)),
        "clock": observations.clock,
        "observations": None if observation_path is None else str(observation_path),
        "provenance": None if provenance_path is None else str(provenance_path),
        "observations_sha256": None
        if provenance is None
        else provenance["observations_sha256"],
        "book_rows": len(observations.mid),
        "segments": len(np.unique(observations.segment)),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(observations.names),
        "feature_names": list(observations.names),
        "x_shape": list(observations.x.shape),
        "y_shape": list(observations.y.shape),
        "history": history,
        "horizon": horizon,
        "split_row": split,
        "train_anchors": len(training_anchors),
        "validation_anchors": len(validation_anchors),
        "selected_anchors": len(anchors),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "D_shape": list(D.shape),
        "eligible_pairs": len(pairs),
        "target_scale": asdict(target_scale),
        "time_delta_transform": (
            "log1p(time_delta_seconds / time_delta_tau), then train-only mean/std"
        ),
        "time_delta_tau": feature_scale.time_delta_tau,
        "time_of_day": {
            "clock": observations.clock,
            "timezone": TIME_OF_DAY_TIMEZONE,
            "period_seconds": 86_400,
            "encoding": ["tod_sin", "tod_cos"],
        },
        "scaled_distance_range": [float(D.min()), float(D.max())],
        "pair_policy": "uniform anchor sample; unique pairs with disjoint complete episodes",
        "query_anchor": int(validation_anchors[0]),
        "nearest_anchor": int(anchors[nearest]),
        "forecast_path_rms_error": float(future_path_rms(Y[nearest], query_Y[0])),
        "purpose": "Full-schema pipeline smoke run; no encoder training or real-market claim",
    }
    return TrainingBatch(
        observations,
        feature_scale,
        target_scale,
        training_anchors,
        validation_anchors,
        anchors,
        rows,
        X,
        Y,
        pairs,
        D,
        D_raw,
        query_X,
        query_Y,
        report,
    )


def save_training_batch(result, output):
    """Write a reusable batch and report under a generated-data directory."""
    output = require_data_path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "batch.npz",
        X=result.X,
        Y=result.Y,
        pairs=result.pairs,
        D=result.D,
        D_raw=result.D_raw,
        anchors=result.anchors,
        feature_mean=result.feature_scale.mean,
        feature_scale=result.feature_scale.scale,
        time_delta_tau=result.feature_scale.time_delta_tau,
        feature_names=result.feature_scale.names,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        constant_features=result.feature_scale.constant,
        training_rows=result.training_rows,
        target_scale=result.target_scale.value,
        query_X=result.query_X,
        query_Y=result.query_Y,
    )
    report = dict(result.report)
    report["output"] = str(output)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def run_training_pipeline(
    output=None,
    *,
    seed=0,
    end_time="10:00:00",
    observations_dir=None,
    history=256,
    horizon=128,
    anchor_count=64,
    split_row=None,
    run_export=True,
):
    """Run ABIDES, build a batch, save artifacts, and return the batch object."""
    source_default, output_default = abides_paths(seed, end_time)
    source = Path(observations_dir) if observations_dir is not None else source_default
    output = Path(output) if output is not None else output_default
    if run_export:
        source = run_abides_export(output=source, seed=seed, end_time=end_time)
    observations, provenance, _, _ = load_abides_source(source)
    result = build_training_batch(
        observations,
        provenance=provenance,
        source_dir=source,
        seed=seed,
        history=history,
        horizon=horizon,
        anchor_count=anchor_count,
        split_row=split_row,
    )
    save_training_batch(result, output)
    result.report["output"] = str(Path(output).resolve())
    return result
