import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mbo_lab.budget import initialize, locked, reserve, status
from mbo_lab.cli import request_from_args
from mbo_lab.data import DataError, Request, download


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.jsonl"
        initialize(self.path, "1.00", "2099-01-01")

    def test_cumulative_cap_and_no_overwrite(self):
        reserve("0.60", {}, self.path)
        with self.assertRaisesRegex(ValueError, "cumulative"):
            reserve("0.41", {}, self.path)
        reserve("0.40", {}, self.path)
        self.assertEqual(status(self.path)["remaining_usd"], "0.00")
        with self.assertRaises(FileExistsError):
            initialize(self.path, "125", "2099-01-01")

    def test_lock_blocks_second_writer(self):
        with locked(self.path):
            with self.assertRaisesRegex(ValueError, "locked"):
                reserve("0.10", {}, self.path)
        self.assertEqual(status(self.path)["reserved_usd"], "0")

    def test_corruption_blocks_reservation(self):
        with self.path.open("a") as stream:
            stream.write('{"type":')
        with self.assertRaisesRegex(ValueError, "corrupt"):
            reserve("0.01", {}, self.path)

    def test_expiry_blocks_reservation(self):
        header = json.loads(self.path.read_text())
        header["expires_on"] = "2000-01-01"
        self.path.write_text(json.dumps(header) + "\n")
        with self.assertRaisesRegex(ValueError, "expired"):
            reserve("0.01", {}, self.path)

    def test_invalid_quotes_blocked(self):
        for value in ("NaN", "Infinity", "-1"):
            with self.assertRaises(ValueError):
                reserve(value, {}, self.path)

    def test_failed_transfer_keeps_reservation(self):
        request = Request("ESM4", "2024-06-17T00:00:00Z", "2024-06-17T00:01:00Z")
        from types import SimpleNamespace

        def fail(**kwargs):
            self.assertEqual(status(self.path)["reserved_usd"], "0.25")
            raise RuntimeError("Connection lost")

        client = SimpleNamespace(timeseries=SimpleNamespace(get_range=fail))
        quote = {
            "total_usd": 0.25,
            "request": {
                "symbol": "SYNTH_ES", "start": request.start, "end": request.end,
                "dataset": request.dataset, "stype_in": "raw_symbol",
            },
        }
        with patch("mbo_lab.data.estimate", return_value=quote):
            with self.assertRaisesRegex(RuntimeError, "Connection lost"):
                download(
                    request, Path(self.temp.name) / "raw", max_cost=1,
                    client=client, budget_path=self.path,
                    confirm=lambda details: True,
                )
        self.assertEqual(status(self.path)["reserved_usd"], "0.25")
        self.assertEqual(status(self.path)["pending_usd"], "0.25")

    def test_success_appends_completion_and_cost(self):
        request = Request("ESM4", "2024-06-17T00:00:00Z", "2024-06-17T00:01:00Z")
        quote = {
            "total_usd": 0.25,
            "request": {
                "symbol": "SYNTH_ES", "start": request.start, "end": request.end,
                "dataset": request.dataset, "stype_in": "raw_symbol",
            },
        }
        from types import SimpleNamespace

        def write_fake(**kwargs):
            # This path is reached only after the quote and confirmation.
            Path(kwargs["path"]).write_bytes(b"not-a-dbn")

        client = SimpleNamespace(timeseries=SimpleNamespace(get_range=write_fake))
        with patch("mbo_lab.data.estimate", return_value=quote), patch(
            "mbo_lab.data.inspect_file", side_effect=lambda path: {"records": 1}
        ), patch(
            "mbo_lab.data._validate_pair",
            return_value=(None, {"identities": [("SYNTH_ES.GLBX", 1)], "symbols": ["SYNTH_ES"]}),
        ):
            download(
                request, Path(self.temp.name) / "raw", max_cost=1,
                client=client, budget_path=self.path, confirm=lambda details: True,
            )
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(records[-1]["type"], "complete")
        self.assertEqual(records[-1]["usd"], "0.25")
        self.assertTrue(Path(records[-1]["manifest"]).is_file())
        self.assertEqual(status(self.path)["reserved_usd"], "0.25")
        self.assertEqual(status(self.path)["completed_usd"], "0.25")
        self.assertEqual(status(self.path)["pending_usd"], "0.00")

    def test_missing_ledger_blocks_before_client_creation(self):
        request = Request("ESM4", "2024-06-17T00:00:00Z", "2024-06-17T00:01:00Z")
        with patch("mbo_lab.data.db.Historical") as client:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                download(
                    request, Path(self.temp.name) / "raw", max_cost=1,
                    budget_path=Path(self.temp.name) / "missing.jsonl",
                )
            client.assert_not_called()

    def test_unconfirmed_download_never_transfers_or_reserves(self):
        request = Request("ESM4", "2024-06-17T00:00:00Z", "2024-06-17T00:01:00Z")
        from unittest.mock import Mock

        client = Mock()
        quote = {"total_usd": 0.25, "request": vars(request)}
        with patch("mbo_lab.data.estimate", return_value=quote):
            for confirm in (None, lambda details: False):
                with self.assertRaisesRegex(ValueError, "not confirmed"):
                    download(
                        request, Path(self.temp.name) / "raw", max_cost=1,
                        client=client, budget_path=self.path, confirm=confirm,
                    )
        client.timeseries.get_range.assert_not_called()
        self.assertEqual(status(self.path)["reserved_usd"], "0")

    def test_cli_rejects_piped_confirmation(self):
        from mbo_lab.cli import confirm_download

        with patch("sys.stdin.isatty", return_value=False), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "interactive terminal"):
                confirm_download({})

    def test_cli_date_overrides_are_paired_and_validated(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            config="es.toml",
            start="2024-06-17T00:00:00Z",
            end="2024-06-17T00:02:00Z",
        )
        request = request_from_args(args)
        self.assertEqual(request.symbol, "ES.c.0")
        self.assertEqual(request.start, args.start)
        self.assertEqual(request.end, args.end)

        with self.assertRaisesRegex(ValueError, "supplied together"):
            request_from_args(SimpleNamespace(config="es.toml", start=args.start, end=None))

        with self.assertRaises(DataError):
            request_from_args(
                SimpleNamespace(
                    config="es.toml",
                    start="2024-06-17T00:01:00Z",
                    end="2024-06-17T00:02:00Z",
                )
            )


if __name__ == "__main__":
    unittest.main()
