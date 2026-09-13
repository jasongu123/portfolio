from __future__ import annotations

import contextlib
from contextlib import closing
import datetime as dt
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest

from execution_cost_report import (
    DEFAULT_DB_PATH,
    count_quarantined_execution_fills,
    format_report,
    main,
    summarize_execution_costs,
)


_SCHEMA = """
CREATE TABLE order_intents (
    client_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    strategy TEXT,
    sleeve TEXT,
    symbol TEXT,
    PRIMARY KEY (client_id, order_id)
);

CREATE TABLE execution_costs (
    exec_id TEXT PRIMARY KEY,
    client_id INTEGER,
    order_id INTEGER,
    symbol TEXT,
    shares REAL,
    fill_price REAL,
    fill_timestamp TEXT,
    arrival_mid REAL,
    signed_slippage_bps REAL,
    commission REAL,
    realized_total_cost REAL,
    created_at TEXT
);
"""


class ExecutionCostReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "journal.sqlite3"
        self.now = dt.datetime(2026, 8, 2, 16, 0, tzinfo=dt.timezone.utc)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(_SCHEMA)
            connection.executemany(
                """
                INSERT INTO order_intents
                    (client_id, order_id, strategy, sleeve, symbol)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (1, 10, "trend", "core", "AAPL"),
                    (1, 11, "trend", "core", "AAPL"),
                    (1, 12, "trend", "core", "AAPL"),
                    (2, 20, "mean_reversion", "satellite", "MSFT"),
                    (3, 30, "old", "archive", "IBM"),
                ),
            )
            connection.commit()
            connection.executemany(
                """
                INSERT INTO execution_costs (
                    exec_id, client_id, order_id, symbol, shares, fill_price,
                    fill_timestamp, arrival_mid, signed_slippage_bps,
                    commission, realized_total_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        "aapl-1", 1, 10, "AAPL", 10, 100,
                        "2026-08-02T12:00:00+00:00", 99.9, 10, 0.50, 1.50,
                        "2026-08-02T12:00:00+00:00",
                    ),
                    (
                        "aapl-2", 1, 11, "AAPL", 5, 200,
                        "2026-08-02T13:00:00+00:00", 199.4, 30, 0.75, 3.75,
                        "2026-08-02T13:00:00+00:00",
                    ),
                    (
                        "aapl-missing", 1, 12, "AAPL", 1, 100,
                        # IBKR commonly emits a non-ISO broker timestamp.
                        # The report must fall back to the journal's ISO
                        # created_at rather than silently dropping this fill.
                        "20260802 14:00:00 US/Eastern", None, None, 0.25, 0.25,
                        "2026-08-02T14:00:00+00:00",
                    ),
                    (
                        "msft-1", 2, 20, "MSFT", 20, 50,
                        "2026-08-01T15:00:00+00:00", 50.1, -12.5, 1.00, -0.25,
                        "2026-08-01T15:00:00+00:00",
                    ),
                    (
                        "old-fill", 3, 30, "IBM", 10, 150,
                        "2026-06-01T12:00:00+00:00", 150, 0, 1.00, 1.00,
                        "2026-06-01T12:00:00+00:00",
                    ),
                ),
            )
            connection.commit()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_recent_fills_are_grouped_and_notional_weighted(self) -> None:
        rows = summarize_execution_costs(
            self.db_path,
            days=7,
            now=self.now,
        )
        by_symbol = {row["symbol"]: row for row in rows}

        self.assertEqual(set(by_symbol), {"AAPL", "MSFT"})
        aapl = by_symbol["AAPL"]
        self.assertEqual(aapl["sleeve"], "core")
        self.assertEqual(aapl["strategy"], "trend")
        self.assertEqual(aapl["fill_count"], 3)
        self.assertAlmostEqual(aapl["filled_notional"], 2_100.0)
        self.assertAlmostEqual(aapl["weighted_signed_slippage_bps"], 20.0)
        self.assertAlmostEqual(aapl["commissions"], 1.50)
        self.assertAlmostEqual(aapl["realized_total_cost"], 5.50)
        self.assertEqual(aapl["missing_arrival_mid_count"], 1)

        msft = by_symbol["MSFT"]
        self.assertEqual(msft["fill_count"], 1)
        self.assertAlmostEqual(msft["weighted_signed_slippage_bps"], -12.5)

    def test_report_contains_requested_columns_and_values(self) -> None:
        rows = summarize_execution_costs(self.db_path, days=7, now=self.now)
        output = format_report(rows, days=7, db_path=self.db_path)

        self.assertIn("FILLED NOTIONAL", output)
        self.assertIn("SIGNED SLIPPAGE BPS", output)
        self.assertIn("REALIZED TOTAL COST", output)
        self.assertIn("MISSING MID", output)
        self.assertIn("AAPL", output)
        self.assertIn("2,100.00", output)

    def test_join_uses_client_id_and_order_id_together(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO order_intents
                    (client_id, order_id, strategy, sleeve, symbol)
                VALUES (9, 10, 'stat_arb', 'hedge', 'QQQ')
                """
            )
            connection.execute(
                """
                INSERT INTO execution_costs (
                    exec_id, client_id, order_id, symbol, shares, fill_price,
                    fill_timestamp, arrival_mid, signed_slippage_bps,
                    commission, realized_total_cost, created_at
                ) VALUES (
                    'same-order-other-client', 9, 10, 'QQQ', 2, 500,
                    '2026-08-02T15:00:00+00:00', 500, 0, 0.35, 0.35,
                    '2026-08-02T15:00:00+00:00'
                )
                """
            )
            connection.commit()

        rows = summarize_execution_costs(self.db_path, days=7, now=self.now)
        qqq = next(row for row in rows if row["symbol"] == "QQQ")
        self.assertEqual(qqq["sleeve"], "hedge")
        self.assertEqual(qqq["strategy"], "stat_arb")

    def test_symbol_mismatch_is_quarantined_instead_of_attributed(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO execution_costs (
                    exec_id, client_id, order_id, symbol, shares, fill_price,
                    fill_timestamp, arrival_mid, signed_slippage_bps,
                    commission, realized_total_cost, created_at
                ) VALUES (
                    'wrong-symbol', 1, 10, 'JPM', 40, 365.02,
                    '2026-08-02T15:30:00+00:00', 115.805, 21520.23,
                    1.00, 9968.80, '2026-08-02T15:30:00+00:00'
                )
                """
            )
            connection.commit()

        rows = summarize_execution_costs(self.db_path, days=7, now=self.now)
        self.assertNotIn("JPM", {row["symbol"] for row in rows})
        aapl = next(row for row in rows if row["symbol"] == "AAPL")
        self.assertEqual(aapl["fill_count"], 3)
        self.assertEqual(
            count_quarantined_execution_fills(
                self.db_path, days=7, now=self.now
            ),
            1,
        )

    def test_missing_or_schema_empty_journal_is_cleanly_empty(self) -> None:
        missing = Path(self.temp.name) / "missing.sqlite3"
        self.assertEqual(summarize_execution_costs(missing, days=7), [])
        self.assertFalse(missing.exists())

        empty_schema = Path(self.temp.name) / "empty.sqlite3"
        sqlite3.connect(empty_schema).close()
        self.assertEqual(summarize_execution_costs(empty_schema, days=7), [])

    def test_cli_prints_clean_message_for_missing_journal(self) -> None:
        missing = Path(self.temp.name) / "missing.sqlite3"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--db", str(missing), "--days", "7"])

        self.assertEqual(exit_code, 0)
        self.assertIn("No execution fills found", stdout.getvalue())
        self.assertFalse(missing.exists())

    def test_default_database_is_beside_the_script(self) -> None:
        self.assertEqual(DEFAULT_DB_PATH.name, "trading_system.db")
        self.assertEqual(DEFAULT_DB_PATH.parent, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
