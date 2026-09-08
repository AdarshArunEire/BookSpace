"""Disposable provider doubles; no paid API requests or real-market validation."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from mbo_lab.batch import _lock, download
from mbo_lab.cli import main
from mbo_lab.data import DataError, Request


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.request = Request(
            "ES.v.0", "2025-01-01T00:00:00Z", "2025-05-01T00:00:00Z", stype_in="continuous"
        )
        self.provider = Mock()
        self.jobs = {}
        self.payload = b"disposable batch fixture; not market data"

        def submit(**params):
            job_id = "TEST-" + params["schema"]
            self.jobs[job_id] = {**params, "id": job_id, "state": "done"}
            return self.jobs[job_id]

        def files(job_id):
            return [
                {
                    "filename": f"sample.{self.jobs[job_id]['schema']}.dbn.zst",
                    "hash": "sha256:" + hashlib.sha256(self.payload).hexdigest(),
                    "size": len(self.payload),
                }
            ]

        def fetch(job_id, output_dir, filename_to_download):
            path = Path(output_dir) / job_id / filename_to_download
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payload)
            return [path]

        self.provider.batch.submit_job.side_effect = submit
        self.provider.batch.get_job_details.side_effect = lambda job_id: self.jobs[job_id]
        self.provider.batch.list_files.side_effect = files
        self.provider.batch.download.side_effect = fetch

    def run_download(self, **kwargs):
        return download(self.request, self.temp.name, client=self.provider, **kwargs)

    def test_two_whole_range_jobs_and_offline_cached_repeat(self):
        result = self.run_download()
        self.assertEqual(result["downloaded_files"], 2)
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)
        for call in self.provider.batch.submit_job.call_args_list:
            self.assertEqual(call.kwargs["start"], self.request.start)
            self.assertEqual(call.kwargs["end"], self.request.end)
            self.assertEqual(call.kwargs["symbols"], ["ES.v.0"])
            self.assertEqual(call.kwargs["stype_in"], "continuous")
            self.assertEqual(call.kwargs["split_duration"], "day")
            self.assertTrue(call.kwargs["split_symbols"])
            self.assertEqual(call.kwargs["split_size"], 5_000_000_000)
            self.assertEqual(call.kwargs["compression"], "zstd")
            self.assertNotIn("limit", call.kwargs)
        self.provider.timeseries.get_range.assert_not_called()
        with patch("mbo_lab.batch.db.Historical", side_effect=AssertionError("No network")):
            repeat = download(self.request, self.temp.name)
        self.assertEqual(repeat["reused_files"], 2)
        self.assertEqual(repeat["downloaded_files"], 0)

    def test_failed_transfer_resumes_same_jobs(self):
        fetch = self.provider.batch.download.side_effect
        self.provider.batch.download.side_effect = RuntimeError("Disconnected")
        with self.assertRaisesRegex(RuntimeError, "Disconnected"):
            self.run_download()
        self.provider.batch.download.side_effect = fetch
        result = self.run_download()
        self.assertEqual(result["downloaded_files"], 2)
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)

    def test_checksum_failure_is_not_accepted(self):
        fetch = self.provider.batch.download.side_effect

        def corrupt(**params):
            paths = fetch(**params)
            paths[0].write_bytes(b"x" * len(self.payload))
            return paths

        self.provider.batch.download.side_effect = corrupt
        with self.assertRaisesRegex(DataError, "Checksum/size mismatch"):
            self.run_download()
        self.provider.batch.download.side_effect = fetch
        self.assertEqual(self.run_download()["downloaded_files"], 2)
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)

    def test_lost_submission_response_requires_explicit_job_recovery(self):
        submit = self.provider.batch.submit_job.side_effect

        def lose_response(**params):
            submit(**params)
            raise RuntimeError("Lost submission response")

        self.provider.batch.submit_job.side_effect = lose_response
        with self.assertRaisesRegex(RuntimeError, "Lost submission"):
            self.run_download()
        with self.assertRaisesRegex(DataError, "submission outcome is unknown"):
            self.run_download()
        self.assertEqual(self.provider.batch.submit_job.call_count, 1)
        self.provider.batch.submit_job.side_effect = submit
        result = self.run_download(jobs={"definition": "TEST-definition"})
        self.assertEqual(result["downloaded_files"], 2)
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)

    def test_expired_job_never_causes_another_purchase(self):
        result = self.run_download()
        job_id = result["jobs"]["mbo"]
        (result["directory"] / job_id / "sample.mbo.dbn.zst").unlink()
        self.jobs[job_id]["state"] = "expired"
        with self.assertRaisesRegex(DataError, "expired"):
            self.run_download()
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)

    def test_wrong_recovery_job_is_rejected_before_submission(self):
        self.jobs["WRONG"] = {"dataset": "WRONG"}
        with self.assertRaisesRegex(DataError, "does not match"):
            self.run_download(jobs={"mbo": "WRONG"})
        self.provider.batch.submit_job.assert_not_called()

    def test_wait_uses_job_details_and_reports_pending(self):
        original = self.provider.batch.get_job_details.side_effect
        calls = 0

        def pending_once(job_id):
            nonlocal calls
            calls += 1
            details = original(job_id)
            return {**details, "state": "queued"} if calls == 1 else details

        self.provider.batch.get_job_details.side_effect = pending_once
        report = Mock()
        with patch("mbo_lab.batch.time.sleep") as sleep:
            self.run_download(progress=report)
        sleep.assert_called_once_with(15)
        self.provider.batch.list_jobs.assert_not_called()
        self.assertTrue(any("queued" in call.args[0] for call in report.call_args_list))

    def test_bad_provider_filename_never_downloaded(self):
        self.provider.batch.list_files.side_effect = None
        self.provider.batch.list_files.return_value = [
            {"filename": "../escape", "hash": "sha256:" + "0" * 64, "size": 0}
        ]
        with self.assertRaisesRegex(DataError, "Unsafe filename"):
            self.run_download()
        self.provider.batch.download.assert_not_called()

    def test_lock_released_after_failure(self):
        path = Path(self.temp.name) / "lock"
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with _lock(path):
                with self.assertRaises(DataError):
                    with _lock(path):
                        self.fail("Second process lock acquired")
                raise RuntimeError("stop")
        with _lock(path):
            pass

    def test_cli_dry_run_is_offline_and_matches_training_request(self):
        output = io.StringIO()
        with patch("mbo_lab.cli.batch_download") as paid, redirect_stdout(output):
            self.assertEqual(main(["download", "--dry-run", "--json"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["request"]["symbol"], "ES.v.0")
        self.assertEqual(result["request"]["start"], self.request.start)
        self.assertEqual(result["request"]["end"], self.request.end)
        self.assertEqual(result["method"], "batch")
        self.assertEqual(result["paid_requests_submitted"], 0)
        paid.assert_not_called()

    def test_disk_space_failure_keeps_jobs_for_resume(self):
        from types import SimpleNamespace

        with patch("mbo_lab.batch.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(DataError, "Not enough free disk"):
                self.run_download()
        self.provider.batch.download.assert_not_called()
        self.assertEqual(self.run_download()["downloaded_files"], 2)
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)

    def test_corrupt_receipt_never_submits_more_jobs(self):
        result = self.run_download()
        result["receipt"].write_text('{"broken": true}', encoding="utf-8")
        with self.assertRaisesRegex(DataError, "Invalid batch receipt"):
            self.run_download()
        self.assertEqual(self.provider.batch.submit_job.call_count, 2)

    def test_definitive_submission_rejection_can_be_retried(self):
        from databento.common.error import BentoClientError

        submit = self.provider.batch.submit_job.side_effect
        self.provider.batch.submit_job.side_effect = BentoClientError(403, message="Not entitled")
        with self.assertRaises(BentoClientError):
            self.run_download()
        self.provider.batch.submit_job.side_effect = submit
        self.assertEqual(self.run_download()["downloaded_files"], 2)


if __name__ == "__main__":
    unittest.main()
