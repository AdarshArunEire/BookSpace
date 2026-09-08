"""Chunk-backed corpus and pair-stream contracts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mbo_lab.corpus import audit_databento_corpus, fingerprint, write_json
from mbo_lab.experiment import main as experiment_main
from mbo_lab.observation_store import ObservationWriter
from mbo_lab.observations import feature_names
from mbo_lab.pair_index import PairIndex, PairTraversal
from mbo_lab.preprocessing import fit_databento_preprocessing
from mbo_lab.stream import ChunkPairStream, open_pair_stream


class DatabentoTrainingTests(unittest.TestCase):
    def _write_day(self, root, day, rows=800):
        writer = ObservationWriter(root / day, f"ES.v.0-{day}", "ESH5.GLBX", chunk_rows=384)
        names = feature_names()
        x = np.zeros(len(names), dtype=np.float64)
        x[3:80:4] = 1
        x[81] = 1
        for row in range(rows):
            writer.append(x, 100.0 + row / 1000, 1, row, row, row)
        writer.finish({"source": "test"})

    def test_chunk_corpus_and_stream_reuse_existing_pair_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "daily"
            root.mkdir()
            for day in ("20250102", "20250303", "20250401"):
                self._write_day(root, day)
            (root / "20250418").mkdir()

            corpus_path = Path(temporary) / "corpus.json"
            corpus = audit_databento_corpus(root, corpus_path, history=2, horizon=2, progress=None)
            self.assertEqual(corpus["source_format"], "observation-chunks-v1")
            self.assertEqual(
                [record["split"] for record in corpus["sessions"]],
                [
                    "train",
                    "validation",
                    "test",
                ],
            )
            self.assertEqual(
                corpus["excluded"], [{"path": "20250418", "reason": "incomplete observation store"}]
            )

            preprocessing = fit_databento_preprocessing(
                corpus,
                root,
                Path(temporary) / "preprocessing.json",
                target_pairs=10,
                gap_sample=1000,
                progress=None,
            )

            with open_pair_stream(
                corpus,
                root,
                preprocessing,
                split="train",
                seed=3,
                queue_pairs=4,
                batch_pairs=2,
                memory_bytes=1024**3,
                workers=1,
            ) as stream:
                self.assertIsInstance(stream, ChunkPairStream)
                batch = next(stream)

            self.assertEqual(batch.X.shape, (4, 2, 86))
            self.assertEqual(batch.Y.shape, (4, 2))
            self.assertEqual(batch.pairs.shape, (2, 2))
            self.assertTrue(np.isfinite(batch.D).all())

            validation_index = PairIndex(corpus, "validation")
            validation_ids = PairTraversal(validation_index, 29).take(4).tolist()
            validation = {
                "version": 1,
                "corpus": corpus["fingerprint"],
                "split": "validation",
                "seed": 29,
                "traversal_version": PairTraversal.version,
                "pair_count": len(validation_ids),
                "pair_ids": validation_ids,
            }
            validation["fingerprint"] = fingerprint(validation)
            write_json(Path(temporary) / "validation_pairs.json", validation)

            run_dir = Path(temporary) / "run"
            arguments = [
                "experiments.py",
                "all",
                "--corpus",
                str(corpus_path),
                "--preprocessing",
                str(Path(temporary) / "preprocessing.json"),
                "--data-root",
                str(root),
                "--run-dir",
                str(run_dir),
                "--latent-dim",
                "2",
                "--train-steps",
                "1",
                "--batch-pairs",
                "2",
                "--queue-pairs",
                "4",
                "--stream-memory-gib",
                "1",
                "--validation-pairs",
                "4",
                "--workers",
                "1",
                "--anchor-batch-size",
                "64",
                "--n-prototypes",
                "1",
                "--compressor-batch-size",
                "64",
                "--reservoir-per-state",
                "2",
            ]
            with patch("sys.argv", arguments):
                experiment_main()
            self.assertTrue((run_dir / "encoder/checkpoint.pt").is_file())
            self.assertTrue((run_dir / "compression/compressor.npz").is_file())
            self.assertTrue((run_dir / "future/state_future_stats.npz").is_file())
            self.assertTrue((run_dir / "validation/metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
