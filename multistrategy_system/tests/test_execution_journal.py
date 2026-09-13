from __future__ import annotations

from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

from execution_journal import ExecutionCostJournal


class ExecutionCostJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "execution_costs.sqlite3"
        self.journal = ExecutionCostJournal(self.db_path)
        self.assertTrue(self.journal.healthy)

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def record_intent(
        self,
        *,
        client_id: int = 7,
        order_id: int = 101,
        side: str = "BUY",
        mid: float = 100.0,
    ) -> None:
        self.assertTrue(
            self.journal.record_order_intent(
                client_id=client_id,
                order_id=order_id,
                strategy="trend",
                sleeve="core",
                symbol="AAPL",
                side=side,
                qty=100,
                decision_timestamp="2026-08-01T14:30:00+00:00",
                bid=mid - 0.05,
                ask=mid + 0.05,
                mid=mid,
                quoted_spread_bps=10.0,
                order_type="LMT",
                limit_price=mid + (0.10 if side == "BUY" else -0.10),
                predicted_commission=0.35,
                predicted_spread_cost=5.0,
                predicted_slippage_cost=2.0,
                predicted_impact_cost=1.0,
                predicted_total_cost=8.35,
            )
        )

    def test_schema_and_intent_fields(self):
        self.record_intent()
        intent = self.journal.fetch_order_intent(7, 101)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent["strategy"], "trend")
        self.assertEqual(intent["sleeve"], "core")
        self.assertEqual(intent["symbol"], "AAPL")
        self.assertEqual(intent["order_type"], "LMT")
        self.assertAlmostEqual(intent["predicted_total_cost"], 8.35)

        # sqlite3.Connection's context manager commits/rolls back but does not
        # close the handle.  Windows will not delete the temporary database
        # while that handle remains open, so close it explicitly.
        with closing(sqlite3.connect(self.db_path)) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            pk = {
                row[1]: row[5]
                for row in conn.execute("PRAGMA table_info(order_intents)")
            }
        self.assertIn("order_intents", tables)
        self.assertIn("execution_costs", tables)
        self.assertEqual(pk["client_id"], 1)
        self.assertEqual(pk["order_id"], 2)

    def test_buy_and_sell_slippage_signs(self):
        self.record_intent(client_id=1, order_id=10, side="BUY", mid=100.0)
        self.record_intent(client_id=1, order_id=11, side="SELL", mid=100.0)

        self.assertTrue(
            self.journal.record_execution(
                "buy-fill",
                1,
                10,
                "AAPL",
                "BOT",
                10,
                101.0,
                "2026-08-01T14:30:01+00:00",
            )
        )
        self.assertTrue(
            self.journal.record_execution(
                "sell-fill",
                1,
                11,
                "AAPL",
                "SLD",
                10,
                99.0,
                "2026-08-01T14:30:02+00:00",
            )
        )

        buy = self.journal.fetch_execution("buy-fill")
        sell = self.journal.fetch_execution("sell-fill")
        assert buy is not None and sell is not None
        self.assertEqual(buy["side"], "BUY")
        self.assertEqual(sell["side"], "SELL")
        self.assertAlmostEqual(buy["signed_slippage_bps"], 100.0)
        self.assertAlmostEqual(sell["signed_slippage_bps"], 100.0)
        self.assertAlmostEqual(buy["signed_slippage_dollars"], 10.0)
        self.assertAlmostEqual(sell["signed_slippage_dollars"], 10.0)

    def test_commission_before_fill_is_preserved_in_total_cost(self):
        self.record_intent()
        self.assertTrue(self.journal.record_commission("late-fill", 1.25))
        pending = self.journal.fetch_execution("late-fill")
        assert pending is not None
        self.assertEqual(pending["status"], "pending_execution")
        self.assertAlmostEqual(pending["realized_total_cost"], 1.25)

        self.assertTrue(
            self.journal.record_execution(
                "late-fill",
                7,
                101,
                "AAPL",
                "BUY",
                10,
                100.50,
                "2026-08-01T14:30:03+00:00",
            )
        )
        filled = self.journal.fetch_execution("late-fill")
        assert filled is not None
        self.assertAlmostEqual(filled["commission"], 1.25)
        self.assertAlmostEqual(filled["signed_slippage_dollars"], 5.0)
        self.assertAlmostEqual(filled["realized_total_cost"], 6.25)

    def test_partial_fills_are_aggregated_without_losing_fill_detail(self):
        self.record_intent()
        self.assertTrue(
            self.journal.record_execution(
                "part-1", 7, 101, "AAPL", "BUY", 40, 100.10, "fill-1"
            )
        )
        self.assertTrue(self.journal.record_commission("part-1", 0.40))
        self.assertTrue(
            self.journal.record_execution(
                "part-2", 7, 101, "AAPL", "BUY", 60, 99.90, "fill-2"
            )
        )
        self.assertTrue(self.journal.record_commission("part-2", 0.60))

        first = self.journal.fetch_execution("part-1")
        second = self.journal.fetch_execution("part-2")
        assert first is not None and second is not None
        self.assertAlmostEqual(first["arrival_mid"], 100.0)
        self.assertAlmostEqual(second["arrival_mid"], 100.0)
        self.assertAlmostEqual(first["signed_slippage_dollars"], 4.0)
        self.assertAlmostEqual(second["signed_slippage_dollars"], -6.0)

        summary = self.journal.get_order_summary(7, 101)
        self.assertEqual(summary["fill_count"], 2)
        self.assertAlmostEqual(summary["filled_shares"], 100.0)
        self.assertAlmostEqual(summary["signed_slippage_dollars"], -2.0)
        self.assertAlmostEqual(summary["commission"], 1.0)
        self.assertAlmostEqual(summary["realized_total_cost"], -1.0)

    def test_duplicate_callbacks_are_idempotent(self):
        self.record_intent()
        execution = (
            "duplicate",
            7,
            101,
            "AAPL",
            "BOT",
            10,
            100.25,
            "2026-08-01T14:30:04+00:00",
        )
        self.assertTrue(self.journal.record_execution(*execution))
        self.assertTrue(self.journal.record_execution(*execution))
        self.assertTrue(self.journal.record_commission("duplicate", 0.50))
        self.assertTrue(self.journal.record_commission("duplicate", 0.50))

        summary = self.journal.get_order_summary(7, 101)
        self.assertEqual(summary["fill_count"], 1)
        self.assertAlmostEqual(summary["filled_shares"], 10.0)
        self.assertAlmostEqual(summary["commission"], 0.50)
        self.assertAlmostEqual(summary["realized_total_cost"], 3.0)

    def test_same_order_id_from_two_clients_does_not_collide(self):
        self.record_intent(client_id=21, order_id=500, side="BUY", mid=100.0)
        self.record_intent(client_id=22, order_id=500, side="BUY", mid=200.0)

        self.assertTrue(
            self.journal.record_execution(
                "client-21", 21, 500, "AAPL", "BUY", 5, 101.0, "fill-a"
            )
        )
        self.assertTrue(
            self.journal.record_execution(
                "client-22", 22, 500, "AAPL", "BUY", 5, 201.0, "fill-b"
            )
        )

        first = self.journal.fetch_execution("client-21")
        second = self.journal.fetch_execution("client-22")
        assert first is not None and second is not None
        self.assertAlmostEqual(first["arrival_mid"], 100.0)
        self.assertAlmostEqual(second["arrival_mid"], 200.0)
        self.assertAlmostEqual(first["signed_slippage_dollars"], 5.0)
        self.assertAlmostEqual(second["signed_slippage_dollars"], 5.0)
        self.assertAlmostEqual(first["signed_slippage_bps"], 100.0)
        self.assertAlmostEqual(second["signed_slippage_bps"], 50.0)

    def test_symbol_mismatch_is_quarantined_and_perm_id_is_preserved(self):
        self.record_intent(client_id=1, order_id=24, mid=115.805)

        self.assertTrue(
            self.journal.record_execution(
                "foreign-fill",
                1,
                24,
                "JPM",
                "BOT",
                40,
                365.02,
                "2026-08-14T10:35:24-04:00",
                perm_id=987654321,
            )
        )

        fill = self.journal.fetch_execution("foreign-fill")
        assert fill is not None
        self.assertEqual(fill["perm_id"], 987654321)
        self.assertEqual(fill["status"], "order_identity_mismatch")
        self.assertIn("AAPL", fill["error"])
        self.assertIn("JPM", fill["error"])
        self.assertIsNone(fill["arrival_mid"])
        self.assertIsNone(fill["signed_slippage_bps"])
        self.assertIsNone(fill["signed_slippage_dollars"])

    def test_existing_database_is_migrated_with_perm_id_column(self):
        legacy_path = Path(self.temp.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE execution_costs (
                    exec_id TEXT PRIMARY KEY,
                    client_id INTEGER,
                    order_id INTEGER,
                    symbol TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """
            )

        legacy = ExecutionCostJournal(legacy_path)
        try:
            self.assertTrue(legacy.healthy, legacy.last_error)
            with closing(sqlite3.connect(legacy_path)) as conn:
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(execution_costs)")
                }
            self.assertIn("perm_id", columns)
        finally:
            legacy.close()

    def test_closed_database_returns_failure_instead_of_raising(self):
        self.journal.close()
        self.assertFalse(self.journal.record_commission("ignored", 1.0))
        status = self.journal.status()
        self.assertFalse(status["healthy"])
        self.assertIn("unavailable or closed", status["last_error"])


if __name__ == "__main__":
    unittest.main()
