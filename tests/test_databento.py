import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mbo_lab.cli import main, print_estimate, request_from_args
from mbo_lab.data import DataError, Request, download, estimate


class DatabentoTests(unittest.TestCase):
    def setUp(self):
        self.request = Request(
            "5602", "2024-06-17T00:00:00Z", "2024-06-17T00:10:00Z", stype_in="instrument_id"
        )
        self.client = Mock()
        self.client.metadata.get_cost.side_effect = lambda **kw: {
            "definition": 0.000001,
            "mbo": 0.002,
        }[kw["schema"]]
        self.client.metadata.get_billable_size.side_effect = lambda **kw: {
            "definition": 1000,
            "mbo": 2000000,
        }[kw["schema"]]
        self.client.metadata.list_unit_prices.return_value = [
            {"mode": "historical", "unit_prices": {"definition": 2, "mbo": 2}},
            {"mode": "historical-streaming", "unit_prices": {"definition": 1, "mbo": 1}},
        ]

    def test_estimate_uses_identical_ranges_and_batch_rates(self):
        quote = estimate(self.request, client=self.client)
        self.assertAlmostEqual(quote["total_usd"], 0.002001)
        self.assertEqual(quote["billable_bytes"]["mbo"], 2000000)
        self.assertEqual(quote["usd_per_gb"]["mbo"], 2)
        self.assertEqual(
            self.client.metadata.get_cost.call_args_list,
            self.client.metadata.get_billable_size.call_args_list,
        )
        self.client.timeseries.get_range.assert_not_called()

    def test_year_estimate_passes_continuous_symbol_and_full_range_in_five_calls(self):
        request = Request(
            "ES.v.0", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z", stype_in="continuous"
        )
        with patch("mbo_lab.data.plan", side_effect=AssertionError("Must not plan downloads")):
            quote = estimate(request, client=self.client)
        self.assertEqual(quote["billable_bytes"]["mbo"], 2000000)
        self.assertAlmostEqual(quote["total_usd"], 0.002001)
        self.assertEqual(len(self.client.mock_calls), 5)
        for method in (self.client.metadata.get_cost, self.client.metadata.get_billable_size):
            self.assertEqual(method.call_count, 2)
            for call in method.call_args_list:
                self.assertEqual(call.kwargs["symbols"], ["ES.v.0"])
                self.assertEqual(call.kwargs["stype_in"], "continuous")
                self.assertEqual(call.kwargs["start"], request.start)
                self.assertEqual(call.kwargs["end"], request.end)
        self.client.metadata.list_unit_prices.assert_called_once_with(dataset="GLBX.MDP3")
        self.client.symbology.resolve.assert_not_called()
        self.client.timeseries.get_range.assert_not_called()

    def test_missing_streaming_rate_is_not_replaced_with_batch_rate(self):
        self.client.metadata.list_unit_prices.return_value = [
            {"mode": "historical", "unit_prices": {"mbo": 2}}
        ]
        self.assertIsNone(
            estimate(self.request, client=self.client, mode="historical-streaming")["usd_per_gb"][
                "mbo"
            ]
        )

    def test_invalid_provider_cost_rejected(self):
        self.client.metadata.get_cost.side_effect = None
        self.client.metadata.get_cost.return_value = float("nan")
        with self.assertRaisesRegex(DataError, "invalid cost"):
            estimate(self.request, client=self.client)

    def test_cli_readable_estimate_and_json(self):
        quote = estimate(self.request, client=self.client)
        with patch("mbo_lab.cli.estimate", return_value=quote):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["estimate"]), 0)
            self.assertIn("Billable bytes", output.getvalue())
            self.assertIn("$0.002001000", output.getvalue())
            self.assertIn("partial 24-hour", output.getvalue())
            self.assertNotIn("partial 10-minute", output.getvalue())
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["estimate", "--json"]), 0)
            self.assertEqual(json.loads(output.getvalue()), quote)

    def test_partial_interval_note_and_daily_alignment(self):
        for end, expected in (("2024-06-17T00:01:00Z", True), ("2024-06-18T00:00:00Z", False)):
            request = Request("5602", self.request.start, end, stype_in="instrument_id")
            output = io.StringIO()
            with redirect_stdout(output):
                print_estimate(estimate(request, client=self.client))
            self.assertEqual("partial 10-minute" in output.getvalue(), expected)
            self.assertEqual("partial 24-hour" in output.getvalue(), expected)

    def test_cli_download_needs_no_budget_options_or_prompt(self):
        with (
            patch("mbo_lab.cli.batch_download", return_value={"cached": True}) as transfer,
            patch("builtins.input", side_effect=AssertionError("Unexpected prompt")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["download", "--json"]), 0)
            self.assertEqual(transfer.call_args.kwargs, {"progress": None, "jobs": {}})

    def test_cli_date_validation_precedes_provider_call(self):
        args = SimpleNamespace(config="es.toml", start=self.request.start, end=self.request.end)
        self.assertEqual(request_from_args(args).end, self.request.end)
        with patch("mbo_lab.cli.estimate") as provider, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["estimate", "--start", self.request.start]), 1)
            self.assertEqual(
                main(["estimate", "--start", "2024-06-17T00:01:00Z", "--end", self.request.end]), 1
            )
            provider.assert_not_called()

    def test_missing_key_message_is_actionable(self):
        output = io.StringIO()
        with patch("mbo_lab.cli.estimate", side_effect=ValueError("invalid API key, was None")):
            with redirect_stderr(output):
                self.assertEqual(main(["estimate"]), 1)
        self.assertIn("Set DATABENTO_API_KEY in this terminal", output.getvalue())

    def test_download_without_ledger_and_cached_repeat_without_network(self):
        def write_fixture(**kwargs):
            Path(kwargs["path"]).write_bytes(b"disposable mocked DBN")

        self.client.timeseries.get_range.side_effect = write_fixture
        definitions = {"identities": [(1, 5602)], "symbols": ["ESM4"]}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("mbo_lab.data.inspect_file"),
            patch("mbo_lab.data._validate_pair", return_value=({}, definitions)),
        ):
            result = download(self.request, folder, client=self.client)
            self.assertFalse(result["cached"])
            self.assertEqual(self.client.timeseries.get_range.call_count, 2)
            self.client.metadata.get_cost.assert_not_called()
            manifest = json.loads(result["chunks"][0]["manifest_path"].read_text())
            self.assertNotIn("budget_reservation", manifest)
            with patch("mbo_lab.data.db.Historical") as constructor:
                self.assertTrue(download(self.request, folder)["cached"])
                constructor.assert_not_called()
            result["chunks"][0]["paths"]["mbo"].write_bytes(b"changed")
            with self.assertRaisesRegex(DataError, "missing or changed"):
                download(self.request, folder, client=self.client)

    def test_failed_transfer_preserves_partial_and_blocks_repeat(self):
        def fail(**kwargs):
            Path(kwargs["path"]).write_bytes(b"partial")
            raise RuntimeError("Connection lost")

        self.client.timeseries.get_range.side_effect = fail
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "Connection lost"):
                download(self.request, folder, client=self.client)
            self.assertEqual(len(list(Path(folder).glob("*.part"))), 1)
            with self.assertRaisesRegex(DataError, "Unfinished download"):
                download(self.request, folder, client=self.client)
            self.assertEqual(self.client.timeseries.get_range.call_count, 1)


if __name__ == "__main__":
    unittest.main()
