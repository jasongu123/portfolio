from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from tech_sleeve_live import TECH_SYMBOLS, TechSleeve
from tech_sleeve_research import missing_data_files


class _RiskManager:
    def __init__(self, account_value=100_000):
        self.account_value = account_value

    def get_account_value(self):
        return self.account_value


class _Strategy:
    def __init__(self):
        self.symbol_data = defaultdict(
            lambda: {
                "current_position": 0,
                "intraday_data": None,
                "entry_price": None,
                "last_entry_price": None,
            }
        )
        self.signals = defaultdict(lambda: "HOLD")
        self.allocation_calls = []
        self.execution_calls = []
        self.use_rf_signals = {symbol: True for symbol in TECH_SYMBOLS}
        self.rf_models = {symbol: object() for symbol in TECH_SYMBOLS}

    def update_allocations(self, symbols, allocation_group="equity"):
        self.allocation_calls.append((tuple(symbols), allocation_group))

    def generate_signal(self, symbol):
        return self.signals[symbol]

    def execute_trade(self, symbol, signal, **kwargs):
        self.execution_calls.append((symbol, signal, kwargs))
        return len(self.execution_calls)


class TechSleeveTests(unittest.TestCase):
    def test_disabled_by_default_does_nothing(self):
        strategy = _Strategy()
        sleeve = TechSleeve(strategy, _RiskManager())

        self.assertFalse(sleeve.enabled)
        self.assertEqual(sleeve.position_symbols, set())
        self.assertEqual(sleeve.maybe_check(dt.datetime(2026, 8, 3)), 0)
        self.assertEqual(strategy.allocation_calls, [])

    def test_enabled_sleeve_routes_same_engine_with_separate_caps_and_label(self):
        strategy = _Strategy()
        strategy.signals["INTC"] = "BUY"
        sleeve = TechSleeve(
            strategy,
            _RiskManager(100_000),
            enabled=True,
            paper_mode=True,
            core_symbols=("SPY", "QQQ", "NVDA"),
        )

        count = sleeve.maybe_check(dt.datetime(2026, 8, 3, 10, 0))

        self.assertEqual(count, 1)
        self.assertEqual(
            strategy.allocation_calls,
            [(TECH_SYMBOLS, "tech")],
        )
        symbol, signal, kwargs = strategy.execution_calls[0]
        self.assertEqual((symbol, signal), ("INTC", "BUY"))
        self.assertEqual(kwargs["journal_sleeve"], "tech")
        self.assertAlmostEqual(kwargs["position_size_multiplier"], 0.10)
        self.assertAlmostEqual(kwargs["max_entry_notional"], 2_000.0)

    def test_existing_gross_can_block_new_entries(self):
        strategy = _Strategy()
        strategy.signals["INTC"] = "BUY"
        strategy.symbol_data["AMD"].update(
            current_position=100,
            entry_price=100.0,
        )
        sleeve = TechSleeve(
            strategy,
            _RiskManager(100_000),
            enabled=True,
            paper_mode=True,
        )

        self.assertEqual(sleeve.maybe_check(dt.datetime(2026, 8, 3, 10)), 0)
        self.assertEqual(strategy.execution_calls, [])

    def test_missing_rf_model_blocks_entire_cycle(self):
        strategy = _Strategy()
        strategy.signals["INTC"] = "BUY"
        strategy.rf_models.pop("INTC")
        sleeve = TechSleeve(
            strategy,
            _RiskManager(),
            enabled=True,
            paper_mode=True,
        )

        self.assertEqual(sleeve.maybe_check(dt.datetime(2026, 8, 3, 10)), 0)
        self.assertEqual(strategy.allocation_calls, [])
        self.assertEqual(strategy.execution_calls, [])

    def test_core_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            TechSleeve(
                _Strategy(),
                _RiskManager(),
                symbols=("INTC", "NVDA"),
                core_symbols=("NVDA", "AAPL"),
            )

    def test_real_trading_mode_forces_sleeve_off(self):
        sleeve = TechSleeve(
            _Strategy(),
            _RiskManager(),
            enabled=True,
            paper_mode=False,
        )
        self.assertFalse(sleeve.enabled)


class TechSleeveResearchTests(unittest.TestCase):
    def test_missing_data_check_requires_intraday_and_daily_for_every_symbol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for symbol in TECH_SYMBOLS:
                (root / f"{symbol}_1m.csv").touch()
                (root / f"{symbol}_daily.csv").touch()
            (root / "INTC_daily.csv").unlink()

            missing = missing_data_files(root)

        self.assertEqual(missing, [str(root / "INTC_daily.csv")])


if __name__ == "__main__":
    unittest.main()
