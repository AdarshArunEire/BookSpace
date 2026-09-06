"""Contracts tested on actual ABIDES output; no alternate history generator."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mbo_lab.metrics.fixed import TargetScale, eligible_pairs, rms_distance
from mbo_lab.observations import feature_names, feature_row, load_observations
from mbo_lab.paths import DATA
from mbo_lab.samples import FeatureScale, batch, valid_anchors

ROOT = Path(__file__).resolve().parents[1]


class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = DATA / ".tests"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=scratch)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/training_smoke.py"),
                "--output",
                cls.temp.name,
                "--observations-dir",
                str(cls.output / "source"),
                "--end-time",
                "09:40:00",
            ],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        cls.obs = load_observations(cls.output / "source/observations.npz")

    def test_full_pipeline(self):
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["source"], "ABIDES RMSC04")
        self.assertEqual(report["X_shape"], [64, 256, 84])
        self.assertEqual(report["Y_shape"], [64, 128])
        self.assertGreater(report["eligible_pairs"], 0)
        self.assertEqual(self.obs.names, feature_names())
        self.assertEqual(self.obs.clock, "simulation")
        with np.load(self.output / "batch.npz") as data:
            anchors, futures = data["anchors"], data["Y"]
            for anchor, future in zip(anchors, futures):
                np.testing.assert_allclose(
                    future, np.log(self.obs.mid[anchor + 1 : anchor + 129] / self.obs.mid[anchor])
                )
            np.testing.assert_allclose(data["D"], data["D_raw"] / data["target_scale"])
            self.assertTrue(np.isfinite(data["X"]).all())
            self.assertTrue(np.all(np.abs(np.diff(anchors[data["pairs"]], axis=1)) >= 384))

    def test_causal_history_and_y(self):
        anchor = valid_anchors(self.obs)[0]
        x, y = batch(self.obs, [anchor])
        np.testing.assert_array_equal(x[0], self.obs.x[anchor - 255 : anchor + 1])
        np.testing.assert_allclose(
            y[0], np.cumsum(self.obs.y[anchor + 1 : anchor + 129]), atol=1e-14
        )
        changed = replace(self.obs, mid=self.obs.mid.copy(), x=self.obs.x.copy())
        changed.mid[anchor + 1 :] *= 2
        changed.x[anchor + 1 :] *= 2
        np.testing.assert_array_equal(batch(changed, [anchor])[0], x)

    def test_boundaries(self):
        obs = replace(self.obs, segment=self.obs.segment.copy())
        split = len(obs.mid) // 2
        obs.segment[split:] += 100
        anchors = valid_anchors(obs)
        self.assertTrue(np.all((anchors + 128 < split) | (anchors - 255 >= split)))
        self.assertTrue(np.all(valid_anchors(obs, stop=split) + 128 < split))
        with self.assertRaises(ValueError):
            batch(obs, [split])
        self.assertEqual(len(eligible_pairs([255, 638])), 0)
        self.assertEqual(len(eligible_pairs([255, 639])), 1)
        self.assertEqual(len(valid_anchors(obs, stop=383)), 0)

    def test_feature_arithmetic(self):
        # Literal book levels test the adapter, not a generated history.
        row = feature_row([(99, 15, 2)], [(101, 20, 3)], 100, 1, 2_000_000_000, 99, 1_000_000_000)
        self.assertEqual(len(row), 84)
        self.assertEqual(row[:4], [-1, 15, 2, 1])
        self.assertEqual(row[4:8], [0, 0, 0, 0])
        self.assertEqual(row[40:44], [1, 20, 3, 1])
        self.assertEqual(row[80:82], [2, 1])
        self.assertAlmostEqual(row[82], np.log(100 / 99))

    def test_repeatable_abides_run(self):
        second = self.output / "repeat"
        subprocess.run(
            [
                str(ROOT / ".local/abides-env39/Scripts/python.exe"),
                str(ROOT / "scripts/export_abides.py"),
                "--output",
                str(second),
                "--end-time",
                "09:40:00",
            ],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        repeated = load_observations(second / "observations.npz")
        np.testing.assert_array_equal(self.obs.x, repeated.x)
        np.testing.assert_array_equal(self.obs.mid, repeated.mid)
        np.testing.assert_array_equal(self.obs.ts_event, repeated.ts_event)

    def test_train_only_scaling(self):
        rows = np.arange(len(self.obs.mid) // 2)
        fitted = FeatureScale.fit(self.obs, rows)
        changed = replace(self.obs, x=self.obs.x.copy())
        changed.x[len(rows) :, 1] *= 10000
        again = FeatureScale.fit(changed, rows)
        np.testing.assert_array_equal(fitted.mean, again.mean)
        np.testing.assert_array_equal(fitted.scale, again.scale)
        values = fitted.transform(self.obs)
        for mask in range(3, 80, 4):
            absent = self.obs.x[:, mask] == 0
            self.assertTrue(np.all(values[absent, mask - 3 : mask] == 0))
            np.testing.assert_array_equal(values[:, mask], self.obs.x[:, mask])
        with self.assertRaises(ValueError):
            TargetScale.fit([0, 0])
        self.assertEqual(rms_distance(np.array([2.0, 5.0, 1.0]), np.array([2.0, 5.0, 1.0])), 0)


if __name__ == "__main__":
    unittest.main()
