"""Pair combinatorics plus integration against explicitly selected saved ABIDES sources."""

import copy
import itertools
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mbo_lab.corpus import expand_ranges, fingerprint, history_ranges, read_corpus, verified_source
from mbo_lab.pair_index import PairIndex, PairTraversal
from mbo_lab.preprocessing import exact_disk_median, fit_preprocessing, transform_rows
from mbo_lab.samples import FeatureScale, batch
from mbo_lab.stream import PairStream


def example_corpus(ranges, history=3, horizon=2):
    """Disposable integer-only combinatorial fixture; not observation/training data."""
    records = []
    for i, spans in enumerate(ranges):
        anchors = expand_ranges(spans)
        eligible = sum(b - a >= history + horizon for a, b in itertools.combinations(anchors, 2))
        records.append(
            {
                "id": str(i),
                "split": "train",
                "anchor_ranges": spans,
                "anchors": len(anchors),
                "within_pairs": int(eligible),
            }
        )
    return {"sessions": records, "history": history, "horizon": horizon, "fingerprint": "fixture"}


class PairCombinatoricsTests(unittest.TestCase):
    def test_exhaustive_rank_unrank_against_independent_enumeration(self):
        rng = np.random.default_rng(91)
        for _ in range(30):
            ranges = []
            for s in range(4):
                anchors = sorted(rng.choice(np.arange(2, 24), size=8, replace=False).tolist())
                ranges.append([[a, a + 1] for a in anchors])
            corpus = example_corpus(ranges)
            index = PairIndex(corpus)
            anchors = [(s, a) for s, spans in enumerate(ranges) for a in expand_ranges(spans)]
            expected = {
                (a, b)
                for a, b in itertools.combinations(anchors, 2)
                if a[0] != b[0] or b[1] - a[1] >= 5
            }
            actual = [index.unrank(i) for i in range(index.total)]
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(set(actual), expected)
            self.assertEqual([index.rank(*p) for p in actual], list(range(index.total)))

    def test_exact_383_384_boundary_and_cross_session_identity(self):
        index = PairIndex(example_corpus([[[255, 256], [638, 640]], [[255, 256]]], 256, 128))
        with self.assertRaises(ValueError):
            index.rank((0, 255), (0, 638))
        for pair in [((0, 255), (0, 639)), ((0, 255), (1, 255))]:
            self.assertEqual(index.unrank(index.rank(*pair)), pair)

    def test_permutation_and_resume_exhaustively(self):
        for size in [1, 2, 3, 4, 5, 15, 16, 17, 31, 64, 65, 257, 1000]:
            index = PairIndex(example_corpus([[[2, 3]], [[2, 3]]]))
            index.total = size
            for seed in (0, 1, 79):
                sampler = PairTraversal(index, seed)
                first = sampler.take(size // 3)
                restored = PairTraversal.restore(index, sampler.state())
                rest = restored.take(size)
                self.assertEqual(sorted(np.r_[first, rest].tolist()), list(range(size)))
                np.testing.assert_array_equal(rest, sampler.take(size))
                self.assertEqual(len(restored.take(10)), 0)

    def test_split_universes_disjoint(self):
        corpus = example_corpus([[[2, 12]]] * 6)
        for i, r in enumerate(corpus["sessions"]):
            r["split"] = ("train", "validation", "test")[i // 2]
        seen = []
        for split in ("train", "validation", "test"):
            index = PairIndex(corpus, split)
            pairs = [index.unrank(i) for i in range(index.total)]
            seen.append({index.records[s]["id"] for p in pairs for s, _ in p})
        self.assertFalse(seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])

    def test_exact_bounded_disk_median(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaps"
            rng = np.random.default_rng(17)
            for n in (1, 2, 13, 1024, 10001):
                values = rng.choice([1e-12, 0.002, 0.5, 1.0, 13.1], size=n).astype(np.float64)
                values.tofile(path)
                self.assertEqual(exact_disk_median(path, n, chunk=71), np.median(values))
            with self.assertRaises(ValueError):
                exact_disk_median(path, 0)


CORPUS = Path(__file__).resolve().parents[1] / "data/processed/abides/training-v1/corpus.json"


@unittest.skipUnless(CORPUS.exists(), "Requires audited saved ABIDES corpus; no synthetic fallback")
class SavedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full, cls.root = read_corpus(CORPUS)
        cls.corpus = copy.deepcopy(full)
        cls.corpus["sessions"] = cls.corpus["sessions"][:4]
        for i, record in enumerate(cls.corpus["sessions"]):
            first = record["anchor_ranges"][0][0]
            record["anchor_ranges"] = [[first, first + 8], [first + 1024, first + 1032]]
            anchors = expand_ranges(record["anchor_ranges"])
            record["anchors"] = len(anchors)
            record["within_pairs"] = int(
                np.sum(len(anchors) - np.searchsorted(anchors, anchors + 384))
            )
            record["split"] = "train" if i < 2 else "validation" if i == 2 else "test"
        cls.corpus["fingerprint"] = fingerprint(cls.corpus["sessions"])
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.scale = fit_preprocessing(
            cls.corpus,
            cls.root,
            Path(cls.temp.name) / "scale.json",
            target_pairs=200,
            progress=None,
        )

    def test_streamed_statistics_match_in_memory_unique_rows(self):
        rows = []
        for r in self.corpus["sessions"][:2]:
            obs = verified_source(self.root, r)
            rows.extend(obs.x[a:b] for a, b in history_ranges(r, 256))
        x = np.concatenate(rows)
        reference = replace(
            obs,
            x=x,
            mid=np.ones(len(x)),
            segment=np.zeros(len(x)),
            ts_recv=np.zeros(len(x)),
            ts_event=np.zeros(len(x)),
            source_row=np.arange(len(x)),
        )
        scale = FeatureScale.fit(reference, np.arange(len(x)))
        fitted = self.scale["features"]
        self.assertEqual(scale.time_delta_tau, fitted["time_delta_tau"])
        np.testing.assert_allclose(scale.mean, fitted["mean"], atol=1e-12)
        np.testing.assert_allclose(scale.scale, fitted["scale"], atol=1e-12)
        self.assertEqual(fitted["training_history_rows"], len(x))

    def test_fit_never_opens_heldout_sources(self):
        allowed = {r["id"] for r in self.corpus["sessions"][:2]}

        def guarded(root, record):
            self.assertIn(record["id"], allowed)
            return verified_source(root, record)

        changed = copy.deepcopy(self.corpus)
        for r in changed["sessions"][2:]:
            r["path"] = "does-not-exist-heldout-data"
        with patch("mbo_lab.preprocessing.verified_source", side_effect=guarded):
            result = fit_preprocessing(
                changed,
                self.root,
                Path(self.temp.name) / "guarded.json",
                target_pairs=200,
                progress=None,
            )
        self.assertEqual(result["features"], self.scale["features"])
        self.assertEqual(result["target"], self.scale["target"])

    def stream(self, **kwargs):
        return PairStream(
            self.corpus,
            self.root,
            self.scale,
            queue_pairs=13,
            batch_pairs=5,
            memory_bytes=1024**3,
            **kwargs,
        )

    def test_saved_data_contents_and_midqueue_resume_across_workers(self):
        checkpoint = Path(self.temp.name) / "checkpoint.json"
        with self.stream(workers=1) as original:
            first = next(original)
            original.checkpoint(checkpoint)
            expected = [next(original) for _ in range(5)]
            for i, (session, anchor) in enumerate(first.sample_ids):
                record = next(r for r in self.corpus["sessions"] if r["id"] == session)
                obs = verified_source(self.root, record)
                X, Y = batch(obs, [anchor])
                transformed = transform_rows(X[0], obs.names, self.scale["features"]).astype("f4")
                np.testing.assert_array_equal(first.X[i], transformed)
                np.testing.assert_array_equal(first.Y[i], Y[0])
            np.testing.assert_allclose(first.D, first.D_raw / self.scale["target"]["value"])
        with self.stream(workers=2) as restored:
            restored.restore(checkpoint)
            for reference in expected:
                actual = next(restored)
                for name in ("X", "Y", "D", "D_raw", "pairs", "pair_ids"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
                self.assertEqual(actual.sample_ids, reference.sample_ids)

    def test_full_small_stream_exhaustion_uniqueness_and_memory(self):
        with self.stream(workers=1) as stream:
            pair_ids = []
            for b in stream:
                pair_ids.extend(b.pair_ids.tolist())
                for a, c in b.pairs:
                    s, x = b.sample_ids[a]
                    r, y = b.sample_ids[c]
                    self.assertTrue(s != r or abs(x - y) >= 384)
            self.assertEqual(sorted(pair_ids), list(range(stream.index.total)))
            report = stream.report()
            self.assertLessEqual(report["cache_peak_bytes"], report["cache_capacity"])
            self.assertEqual(
                report["memory_budget_bytes"],
                report["cache_capacity"] + report["queue_budget_bytes"] + report["reserve_bytes"],
            )

    def test_changed_source_and_incompatible_checkpoint_rejected(self):
        bad = copy.deepcopy(self.corpus["sessions"][0])
        bad["provenance_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Changed provenance"):
            verified_source(self.root, bad)
        checkpoint = Path(self.temp.name) / "incompatible.json"
        with self.stream(workers=1) as stream:
            next(stream)
            stream.checkpoint(checkpoint)
        changed = copy.deepcopy(self.scale)
        changed["fingerprint"] = "different fitted statistics"
        with PairStream(
            self.corpus,
            self.root,
            changed,
            queue_pairs=13,
            batch_pairs=5,
            workers=1,
            memory_bytes=1024**3,
        ) as stream:
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                stream.restore(checkpoint)

    def test_failed_queue_does_not_skip_pairs(self):
        with self.stream(workers=2) as stream:
            expected = PairTraversal(stream.index).take(5)
            with patch.object(stream.cache, "get", side_effect=ValueError("damaged source")):
                with self.assertRaisesRegex(ValueError, "damaged source"):
                    next(stream)
            self.assertEqual(stream.traversal.counter, 0)
            np.testing.assert_array_equal(next(stream).pair_ids, expected)

    def test_budget_rejects_oversized_queue_before_loading(self):
        with patch("mbo_lab.stream.verified_source") as loader:
            with self.assertRaisesRegex(ValueError, "Memory budget"):
                PairStream(
                    self.corpus, self.root, self.scale, queue_pairs=8192, memory_bytes=128 * 1024**2
                )
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
