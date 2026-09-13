from dataclasses import FrozenInstanceError
import math
import unittest

from trading_costs import (
    CostEstimate,
    CostModelConfig,
    estimate_execution_cost,
    estimate_holding_cost,
)


class ExecutionCostTests(unittest.TestCase):
    def test_default_model_uses_minimum_commission_and_one_bp_fallback(self):
        estimate = estimate_execution_cost(shares=10, price=50)

        self.assertEqual(estimate.notional, 500)
        self.assertEqual(estimate.commission, 0.35)
        self.assertEqual(estimate.spread, 0.05)
        self.assertEqual(estimate.slippage, 0)
        self.assertAlmostEqual(estimate.total, 0.40)
        self.assertAlmostEqual(estimate.total_bps, 8.0)

    def test_valid_quote_uses_half_spread_instead_of_fallback(self):
        estimate = estimate_execution_cost(
            shares=200,
            price=25,
            bid=24.98,
            ask=25.02,
        )

        self.assertAlmostEqual(estimate.commission, 0.70)
        self.assertAlmostEqual(estimate.spread, 4.0)

    def test_missing_or_crossed_quote_uses_fallback(self):
        partial = estimate_execution_cost(100, 20, bid=19.99)
        crossed = estimate_execution_cost(100, 20, bid=20.01, ask=19.99)
        zero_quote = estimate_execution_cost(100, 20, bid=0, ask=0)

        self.assertAlmostEqual(partial.spread, 0.20)
        self.assertAlmostEqual(crossed.spread, 0.20)
        self.assertAlmostEqual(zero_quote.spread, 0.20)

    def test_extra_slippage_and_square_root_impact_are_separate(self):
        config = CostModelConfig(
            fallback_execution_bps=0,
            extra_slippage_bps=2,
            impact_coefficient=0.5,
        )
        estimate = estimate_execution_cost(
            shares=1_000,
            price=100,
            config=config,
            daily_volatility=0.02,
            dollar_adv=1_000_000,
        )

        self.assertAlmostEqual(estimate.slippage, 20.0)
        self.assertAlmostEqual(
            estimate.impact,
            100_000 * 0.5 * 0.02 * math.sqrt(100_000 / 1_000_000),
        )

    def test_signed_shares_have_symmetric_costs(self):
        buy = estimate_execution_cost(100, 40)
        sell = estimate_execution_cost(-100, 40)
        self.assertEqual(buy, sell)

    def test_zero_shares_have_no_minimum_commission(self):
        self.assertEqual(estimate_execution_cost(0, 100), CostEstimate())

    def test_invalid_inputs_are_rejected(self):
        for invalid_price in (-1, math.nan, math.inf):
            with self.subTest(price=invalid_price):
                with self.assertRaises(ValueError):
                    estimate_execution_cost(1, invalid_price)

        with self.assertRaises(ValueError):
            estimate_execution_cost(1, 10, bid=-1, ask=10)
        with self.assertRaises(ValueError):
            estimate_execution_cost(
                1,
                10,
                config=CostModelConfig(impact_coefficient=1),
                daily_volatility=0.02,
                dollar_adv=0,
            )


class HoldingCostTests(unittest.TestCase):
    def test_borrow_and_financing_use_configured_day_count(self):
        config = CostModelConfig(
            annual_borrow_rate=0.03,
            annual_financing_rate=0.06,
            day_count=360,
        )
        estimate = estimate_holding_cost(
            short_notional=100_000,
            borrowed_cash=50_000,
            holding_days=30,
            config=config,
        )

        self.assertEqual(estimate.notional, 150_000)
        self.assertAlmostEqual(estimate.borrow, 250.0)
        self.assertAlmostEqual(estimate.financing, 250.0)
        self.assertAlmostEqual(estimate.total, 500.0)
        self.assertAlmostEqual(estimate.total_bps, 500 / 150_000 * 10_000)

    def test_negative_holding_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_holding_cost(short_notional=-1)
        with self.assertRaises(ValueError):
            estimate_holding_cost(borrowed_cash=-1)
        with self.assertRaises(ValueError):
            estimate_holding_cost(holding_days=-1)


class ValueObjectTests(unittest.TestCase):
    def test_config_and_estimate_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            CostModelConfig().min_commission = 1
        with self.assertRaises(FrozenInstanceError):
            CostEstimate().commission = 1

    def test_as_dict_includes_computed_totals(self):
        estimate = CostEstimate(notional=1_000, commission=1, spread=2)
        self.assertEqual(
            estimate.as_dict(),
            {
                "notional": 1_000,
                "commission": 1,
                "spread": 2,
                "slippage": 0.0,
                "impact": 0.0,
                "borrow": 0.0,
                "financing": 0.0,
                "total": 3.0,
                "total_bps": 30.0,
            },
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            CostModelConfig(commission_per_share=-0.01)
        with self.assertRaises(ValueError):
            CostModelConfig(fallback_execution_bps=math.nan)
        with self.assertRaises(ValueError):
            CostModelConfig(day_count=0)


if __name__ == "__main__":
    unittest.main()
