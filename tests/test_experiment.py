"""Focused contracts for the reusable experiment engine."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from mbo_lab.experiment import (
    FutureSummarySpec,
    StateFutureAccumulator,
    TrainSpec,
    _evaluate_validation_probe,
    _iter_anchor_ids,
    iter_unique_anchor_batches,
)


class ExperimentTests(unittest.TestCase):
    def test_anchor_ids_stream_ranges_without_crossing_gaps(self):
        record = {"anchor_ranges": [[2, 7], [10, 12]]}

        batches = list(_iter_anchor_ids(record, 3))

        self.assertEqual([batch.tolist() for batch in batches], [[2, 3, 4], [5, 6], [10, 11]])

    def test_validation_probe_reuses_cached_tensors(self):
        encoder = torch.nn.Identity()
        batches = [
            (
                torch.tensor([[0.0], [2.0]]),
                torch.tensor([[0, 1]]),
                torch.tensor([1.0]),
            )
        ]

        mse = _evaluate_validation_probe(encoder, batches, torch.device("cpu"))

        self.assertEqual(mse, 1.0)
        self.assertTrue(encoder.training)

    def test_validation_probe_must_fit_final_validation_budget(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TrainSpec(validation_pairs=128, validation_interval=10, validation_probe_pairs=256)

    def test_anchor_replay_uses_frozen_corpus_ranges(self):
        observations = SimpleNamespace(
            names=("feature",),
            x=np.arange(20, dtype=np.float64)[:, None],
            mid=np.exp(np.arange(20, dtype=np.float64) / 100),
        )
        corpus = {
            "history": 2,
            "horizon": 2,
            "names": ["feature"],
            "fingerprint": "corpus",
            "sessions": [
                {
                    "id": "session",
                    "split": "train",
                    "anchor_ranges": [[4, 6]],
                    "anchors": 2,
                    "within_pairs": 0,
                },
                {
                    "id": "later",
                    "split": "train",
                    "anchor_ranges": [[8, 9]],
                    "anchors": 1,
                    "within_pairs": 0,
                },
            ],
        }
        preprocessing = {"features": {"unused": True}}

        def identity(rows, names, feature_spec):
            self.assertEqual(names, ("feature",))
            self.assertEqual(feature_spec, {"unused": True})
            return rows

        with (
            patch("mbo_lab.experiment.verified_source", return_value=observations),
            patch("mbo_lab.experiment.transform_rows", side_effect=identity),
        ):
            batches = list(
                iter_unique_anchor_batches(
                    corpus,
                    ".",
                    preprocessing,
                    split="train",
                    anchor_batch_size=8,
                    include_y=True,
                )
            )

        np.testing.assert_array_equal(batches[0].anchors, [4, 5])
        np.testing.assert_array_equal(batches[1].anchors, [8])
        self.assertEqual(sum(len(batch.X) for batch in batches), 3)

    def test_future_summary_is_exact_and_bounded(self):
        accumulator = StateFutureAccumulator(
            2,
            2,
            FutureSummarySpec(reservoir_per_state=1, seed=7),
        )
        accumulator.update(
            np.array([0, 1, 0]),
            np.array([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0]]),
        )
        summary = accumulator.summary()

        np.testing.assert_array_equal(summary.counts, [2, 1])
        np.testing.assert_allclose(summary.mean, [[2.0, 3.0], [10.0, 20.0]])
        np.testing.assert_allclose(summary.variance, [[1.0, 1.0], [0.0, 0.0]])
        self.assertEqual(summary.reservoir.shape, (2, 1, 2))


if __name__ == "__main__":
    unittest.main()
