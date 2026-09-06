"""Full ABIDES -> 84-feature observations -> 256/128 windows -> scaled pair targets."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from mbo_lab.metrics.fixed import TargetScale, eligible_pairs, pair_targets, rms_distance
from mbo_lab.observations import feature_names, load_observations
from mbo_lab.paths import abides_paths
from mbo_lab.samples import FeatureScale, batch, valid_anchors

ROOT = Path(__file__).resolve().parents[1]


def run(output=None, seed=0, end_time="10:00:00", observations_dir=None):
    source_default, output_default = abides_paths(seed, end_time)
    source = (observations_dir or source_default).resolve()
    output = (output or output_default).resolve()
    interpreter = ROOT / ".local/abides-env39/Scripts/python.exe"
    if not interpreter.exists():
        raise RuntimeError("Run scripts/setup_abides.py first")
    subprocess.run(
        [
            str(interpreter),
            str(ROOT / "scripts/export_abides.py"),
            "--output",
            str(source),
            "--seed",
            str(seed),
            "--end-time",
            end_time,
        ],
        check=True,
        cwd=ROOT,
    )
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    path = source / "observations.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["observations_sha256"]:
        raise ValueError("ABIDES observation hash mismatch")
    obs = load_observations(path)
    if obs.names != feature_names(10) or obs.x.shape[1] != 84:
        raise ValueError("Smoke run must use the full 84-feature schema")
    split = len(obs.mid) * 2 // 3
    train = valid_anchors(obs, stop=split)
    validation = valid_anchors(obs, start=split)
    if len(train) < 64 or not len(validation):
        raise ValueError("ABIDES run too short for complete train/validation episodes; extend time")
    rng = np.random.default_rng(seed)
    anchors = np.sort(rng.choice(train, size=64, replace=False))
    pairs = eligible_pairs(anchors)
    coverage = np.zeros(len(obs.mid) + 1, dtype=np.int64)
    np.add.at(coverage, train - 255, 1)
    np.add.at(coverage, train + 1, -1)
    training_rows = np.flatnonzero(np.cumsum(coverage[:-1]) > 0)
    features = FeatureScale.fit(obs, training_rows)
    transformed = replace(obs, x=features.transform(obs))
    x, y = batch(transformed, anchors, stop=split)
    raw = pair_targets(y, pairs)
    scale = TargetScale.fit(raw, seed=seed)
    targets = pair_targets(y, pairs, scale)
    query_x, query_y = batch(transformed, validation[:1], start=split)
    distances = rms_distance(
        x.reshape(len(x), -1), np.broadcast_to(query_x.reshape(1, -1), (len(x), x[0].size))
    )
    nearest = int(np.argmin(distances))
    report = {
        "source": "ABIDES RMSC04",
        "synthetic": True,
        "clock": obs.clock,
        "observations": str(path),
        "provenance": str(source / "provenance.json"),
        "observations_sha256": provenance["observations_sha256"],
        "book_rows": len(obs.mid),
        "segments": len(np.unique(obs.segment)),
        "feature_count": len(obs.names),
        "feature_names": list(obs.names),
        "x_shape": list(obs.x.shape),
        "y_shape": list(obs.y.shape),
        "history": 256,
        "horizon": 128,
        "split_row": split,
        "train_anchors": len(train),
        "validation_anchors": len(validation),
        "X_shape": list(x.shape),
        "Y_shape": list(y.shape),
        "D_shape": list(targets.shape),
        "eligible_pairs": len(pairs),
        "target_scale": asdict(scale),
        "scaled_distance_range": [float(targets.min()), float(targets.max())],
        "pair_policy": "uniform anchor sample; unique pairs with disjoint complete episodes",
        "query_anchor": int(validation[0]),
        "nearest_anchor": int(anchors[nearest]),
        "forecast_path_rms_error": float(rms_distance(y[nearest], query_y[0])),
        "purpose": "Full-schema pipeline smoke run; no encoder training or real-market claim",
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "batch.npz",
        X=x,
        Y=y,
        pairs=pairs,
        D=targets,
        D_raw=raw,
        anchors=anchors,
        feature_mean=features.mean,
        feature_scale=features.scale,
        feature_names=features.names,
        constant_features=features.constant,
        training_rows=training_rows,
        target_scale=scale.value,
        query_X=query_x,
        query_Y=query_y,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_names"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Override processed smoke output directory")
    parser.add_argument("--observations-dir", type=Path, help="Override ABIDES source directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end-time", default="10:00:00")
    args = parser.parse_args()
    run(args.output, args.seed, args.end_time, args.observations_dir)
