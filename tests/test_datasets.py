"""Session counts, independent sessions, and restart behavior using ABIDES."""

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from mbo_lab.datasets import generate_abides_dataset, session_dates
from mbo_lab.observations import load_observations
from mbo_lab.paths import DATA


class DatasetTests(unittest.TestCase):
    def test_session_count(self):
        self.assertEqual(
            session_dates(2),
            [date(2021, 2, 1), date(2021, 2, 2)],
        )
        self.assertEqual(len(session_dates(20)), 20)
        for sessions in [0, -1, 1.5, True, "20"]:
            with self.assertRaises(ValueError):
                session_dates(sessions)

    def test_dated_exports_and_resume(self):
        scratch = DATA / ".tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            args = dict(output=directory, end_time="09:40:00")
            report = generate_abides_dataset(sessions=2, **args)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["sessions"]), 2)
            self.assertNotEqual(report["sessions"][0]["seed"], report["sessions"][1]["seed"])
            for session in report["sessions"]:
                obs = load_observations(session["observations"])
                self.assertEqual(obs.x.shape[1], 86)
                actual_day = datetime.fromtimestamp(
                    int(obs.ts_event[0]) / 1e9, timezone.utc
                ).date().isoformat()
                self.assertEqual(actual_day, session["date"])
                self.assertEqual(len(obs.mid), session["rows"])
            with patch("mbo_lab.datasets.run_abides_export") as exporter:
                resumed = generate_abides_dataset(sessions=2, **args)
                exporter.assert_not_called()
            self.assertEqual(report, resumed)
            manifest = json.loads((Path(directory) / "dataset.json").read_text())
            self.assertEqual(manifest, report)
            source = Path(report["sessions"][0]["observations"])
            source.write_bytes(b"corrupt")
            with patch("mbo_lab.datasets.run_abides_export") as exporter:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    generate_abides_dataset(sessions=2, **args)
                exporter.assert_not_called()
