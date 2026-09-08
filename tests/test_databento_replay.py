"""Chunk boundaries and real saved-data replay checks; no alternate market data."""

import itertools
import tempfile
import unittest
from pathlib import Path

import databento_dbn as dbn
import numpy as np
import zstandard as zstd

from mbo_lab.corpus import fingerprint
from mbo_lab.data import _records
from mbo_lab.dbn_replay import export_replay
from mbo_lab.extract import extract_observations
from mbo_lab.observation_store import (
    COLUMNS,
    ObservationReader,
    ObservationWriter,
    materialize_pairs,
)
from mbo_lab.pair_index import PairIndex, PairTraversal, count_range_pairs

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/processed/databento/pilot-20250102"


class StoreBoundaryTests(unittest.TestCase):
    def test_range_counts_against_enumeration(self):
        for gap in (1, 2, 5, 384):
            for ranges in ([[0, 15]], [[0, 4], [8, 13], [500, 504]], [[0, 1000]]):
                values = [a for start, stop in ranges for a in range(start, stop)]
                expected = sum(b - a >= gap for a, b in itertools.combinations(values, 2))
                self.assertEqual(count_range_pairs(ranges, gap), expected)
        self.assertEqual(
            count_range_pairs([[0, 10**10]], 384), (10**10 - 384) * (10**10 - 383) // 2
        )

    def test_storage_edge_is_not_a_segment_edge(self):
        # Disposable row fixture exercises storage only; real-data checks follow below.
        with tempfile.TemporaryDirectory(dir=ROOT / "data/.tests") as directory:
            path = Path(directory) / "rows"
            writer = ObservationWriter(path, "fixture", "fixture", chunk_rows=384)
            for i in range(1000):
                x = np.zeros(86)
                x[-4] = 0.001
                x[-3] = int(i in (0, 500))
                writer.append(x, 100 + i / 100, int(i >= 500), i, i, i)
            writer.finish({"purpose": "storage boundary fixture only"})
            reader = ObservationReader(path, cache_bytes=384 * 728)
            # This history crosses the chunk at 384 but remains wholly in segment 0.
            X, Y = reader.episodes([370])
            self.assertEqual(X.shape, (1, 256, 86))
            self.assertEqual(Y.shape, (1, 128))
            with self.assertRaisesRegex(ValueError, "logical segment"):
                reader.episodes([499])
            with self.assertRaisesRegex(ValueError, "logical segment"):
                reader.episodes([755 - 1])
            reader.episodes([755])
            self.assertLessEqual(reader.peak_bytes, reader.cache_bytes)
            self.assertGreater(reader.metrics["evicted_bytes"], 0)
            reader.cache.clear()
            reader.resident_bytes = 0
            (path / "rows-000000.npz").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "checksum"):
                reader.read_rows([0])


@unittest.skipUnless(
    (PILOT / "chunks-1024/observations.json").exists(), "Requires completed saved Databento pilot"
)
class SavedReplayTests(unittest.TestCase):
    def test_block_chunk_invariance_all_columns(self):
        a, b = ObservationReader(PILOT / "chunks-512"), ObservationReader(PILOT / "chunks-1024")
        self.assertEqual(a.manifest["rows"], b.manifest["rows"])
        for start in range(0, a.manifest["rows"], 4096):
            rows = np.arange(start, min(start + 4096, a.manifest["rows"]))
            left, right = a.read_rows(rows), b.read_rows(rows)
            for name in COLUMNS:
                np.testing.assert_array_equal(left[name], right[name])
        for t in (400, 511, 512, 1023, 1024, 16000):
            for left, right in zip(a.episodes([t]), b.episodes([t])):
                np.testing.assert_array_equal(left, right)

    def test_whole_prefix_reference(self):
        reader = ObservationReader(PILOT / "chunks-512")
        definitions = reader.manifest["provenance"]["definitions"]["path"]
        ref = extract_observations(PILOT / "reference-prefix.dbn.zst", definitions)
        data = reader.read_rows(np.arange(reader.manifest["rows"]))
        selected = np.flatnonzero(
            (ref.ts_recv >= data["ts_recv"][0]) & (ref.ts_recv <= data["ts_recv"][-1])
        )
        self.assertEqual(len(selected), len(data["mid"]))
        # The new exporter omits initialization snapshots and initializes at first real event.
        np.testing.assert_array_equal(data["x"][1:], ref.x[selected][1:])
        self.assertEqual(data["x"][0, -3], 1)
        self.assertEqual(data["x"][0, -4], 0)
        self.assertEqual(data["x"][0, -5], 0)
        for key in ("mid", "ts_recv", "ts_event"):
            np.testing.assert_array_equal(data[key], getattr(ref, key)[selected])
        np.testing.assert_array_equal(data["source_row"] + 1, ref.source_row[selected])

    def test_file_part_boundary_does_not_reset_book(self):
        original = ObservationReader(PILOT / "chunks-512")
        records = list(_records(PILOT / "reference-prefix.dbn.zst"))
        cut = next(i for i in range(11000, len(records)) if not records[i - 1].flags & dbn.F_LAST)
        with tempfile.TemporaryDirectory(dir=ROOT / "data/.tests") as directory:
            directory = Path(directory)
            files = []
            for i, body in enumerate((records[1:cut], records[cut:])):
                path = directory / f"part-{i}.dbn.zst"
                path.write_bytes(
                    zstd.ZstdCompressor().compress(
                        bytes(records[0]) + b"".join(bytes(r) for r in body)
                    )
                )
                files.append(path)
            provenance = original.manifest["provenance"]
            export_replay(
                files,
                provenance["definitions"]["path"],
                directory / "out",
                timeline_id=original.manifest["timeline_id"],
                stop_ns=provenance["stop_ns_exclusive"],
                block_records=513,
                chunk_rows=65536,
            )
            other = ObservationReader(directory / "out")
            for start in range(0, original.manifest["rows"], 4096):
                rows = np.arange(start, min(start + 4096, original.manifest["rows"]))
                a, b = original.read_rows(rows), other.read_rows(rows)
                for key in COLUMNS:
                    np.testing.assert_array_equal(a[key], b[key])

    def test_pair_primitive_uses_logical_timeline_and_reproducible_ids(self):
        reader = ObservationReader(PILOT / "chunks-512")
        record = reader.pair_record()
        corpus = {
            "history": 256,
            "horizon": 128,
            "sessions": [record],
            "fingerprint": fingerprint(record),
        }
        index = PairIndex(corpus)
        traversal = PairTraversal(index, seed=9)
        ids = traversal.take(32)
        np.testing.assert_array_equal(ids, PairTraversal(index, seed=9).take(32))
        # Identity-like constants test plumbing only; these are not fitted ES training statistics.
        scale = {
            "corpus": corpus["fingerprint"],
            "features": {
                "names": reader.manifest["names"],
                "mean": [0.0] * 86,
                "scale": [1.0] * 86,
                "time_delta_tau": 1.0,
            },
            "target": {"value": 1.0},
        }
        result = materialize_pairs(index, ids, {record["id"]: reader}, scale)
        self.assertEqual(result.X.dtype, np.float32)
        for i, j in result.pairs:
            self.assertGreaterEqual(abs(result.sample_ids[i][1] - result.sample_ids[j][1]), 384)
        np.testing.assert_allclose(
            result.D_raw,
            np.sqrt(
                np.mean((result.Y[result.pairs[:, 0]] - result.Y[result.pairs[:, 1]]) ** 2, axis=1)
            ),
        )


if __name__ == "__main__":
    unittest.main()
