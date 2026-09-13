import unittest

import numpy as np
import pandas as pd

from ibkr_backtest import StrategyBacktester
from trading_costs import CostModelConfig
from trend_follow_v2 import _apply_portfolio_costs


def _bare_backtester(config):
    """Build only the state needed by the centralized cost adapter."""
    backtester = object.__new__(StrategyBacktester)
    backtester.cost_model_config = config
    backtester.execution_cost_totals = backtester._empty_execution_cost_totals()
    return backtester


class BacktestCostTests(unittest.TestCase):
    def assert_cost_components(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=9, msg=name)

    def test_strategy_backtester_uses_minimum_commission_and_tracks_components(self):
        config = CostModelConfig(
            commission_per_share=0.0035,
            min_commission=0.35,
            fallback_execution_bps=1.0,
        )
        backtester = _bare_backtester(config)

        cost = backtester.apply_transaction_costs(price=100.0, shares=10)
        reversal = backtester.apply_transaction_costs(
            price=100.0,
            shares=20,
            return_estimate=True,
        )

        self.assertAlmostEqual(cost, 0.45)
        self.assertAlmostEqual(reversal.notional, 2_000.0)
        self.assertAlmostEqual(reversal.commission, 0.35)
        self.assertAlmostEqual(reversal.spread, 0.20)
        self.assertAlmostEqual(reversal.total, 0.55)
        self.assert_cost_components(backtester.execution_cost_totals, {
            'notional': 3_000.0,
            'commission': 0.70,
            'spread': 0.30,
            'slippage': 0.0,
            'impact': 0.0,
            'total': 1.0,
        })

    def test_intraday_orders_use_close_trade_points_and_full_reversal_turnover(self):
        orders = StrategyBacktester._exposure_change_orders(
            exposure=[0.0, 1.0, 1.0, -1.0, -1.0],
            close_prices=[100.0, 101.0, 102.0, 103.0, 104.0],
        )

        self.assertEqual(orders, [
            (0, 1.0, 100.0),
            (2, -2.0, 102.0),
            (4, 1.0, 104.0),
        ])
        self.assertEqual(sum(abs(delta) for _, delta, _ in orders), 4.0)

    def test_trend_costs_charge_double_turnover_on_reversal_and_short_borrow(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        gross_returns = pd.Series(0.0, index=dates)
        prices = pd.DataFrame({'A': 100.0}, index=dates)
        weights = pd.DataFrame({'A': [0.0, 1.0, -1.0]}, index=dates)
        config = CostModelConfig(
            commission_per_share=0.0,
            min_commission=0.0,
            fallback_execution_bps=10.0,
            annual_borrow_rate=0.252,
            day_count=252.0,
        )

        net = _apply_portfolio_costs(
            gross_returns,
            weights,
            prices,
            config,
            initial_capital=100_000.0,
        )

        # Entry trades one portfolio unit.  The direct long-to-short reversal
        # trades two units against the equity remaining after entry costs.
        self.assertAlmostEqual(net.iloc[1], -0.001)
        self.assertAlmostEqual(net.iloc[2], -0.003)
        self.assert_cost_components(net.attrs['cost_components'], {
            'notional': 299_800.0,
            'commission': 0.0,
            'spread': 299.8,
            'slippage': 0.0,
            'impact': 0.0,
            'borrow': 99.9,
            'financing': 0.0,
            'total': 399.7,
            'total_bps': 399.7 / 299_800.0 * 10_000.0,
        })

    def test_trend_execution_uses_prior_observable_price(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-02'])
        gross_returns = pd.Series(0.0, index=dates)
        prices = pd.DataFrame({'A': [100.0, 200.0]}, index=dates)
        weights = pd.DataFrame({'A': [0.0, 1.0]}, index=dates)
        config = CostModelConfig(
            commission_per_share=0.01,
            min_commission=0.0,
            fallback_execution_bps=0.0,
        )

        net = _apply_portfolio_costs(
            gross_returns,
            weights,
            prices,
            config,
            initial_capital=100_000.0,
        )

        # The rebalance occurs at the prior $100 close: 1,000 shares and $10
        # commission.  Using the current $200 close would leak the future price
        # into sizing and produce only $5 of commission.
        self.assertAlmostEqual(net.iloc[1], -0.0001)
        self.assertAlmostEqual(net.attrs['cost_components']['commission'], 10.0)

    def test_trend_costs_charge_financing_for_actual_calendar_days(self):
        dates = pd.to_datetime(['2024-01-05', '2024-01-08', '2024-01-09'])
        gross_returns = pd.Series(0.0, index=dates)
        prices = pd.DataFrame({'A': 100.0}, index=dates)
        weights = pd.DataFrame({'A': [0.0, 2.0, 2.0]}, index=dates)
        config = CostModelConfig(
            commission_per_share=0.0,
            min_commission=0.0,
            fallback_execution_bps=0.0,
            annual_financing_rate=0.36,
            day_count=360.0,
        )

        net = _apply_portfolio_costs(
            gross_returns,
            weights,
            prices,
            config,
            initial_capital=100_000.0,
        )

        np.testing.assert_allclose(net.to_numpy(), [0.0, -0.003, -0.001])
        self.assertAlmostEqual(net.attrs['cost_components']['financing'], 399.7)
        self.assertAlmostEqual(net.attrs['cost_components']['total'], 399.7)

    def test_market_neutral_trend_book_pays_borrow_but_not_cash_financing(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-02'])
        gross_returns = pd.Series(0.0, index=dates)
        prices = pd.DataFrame({'LONG': 100.0, 'SHORT': 100.0}, index=dates)
        weights = pd.DataFrame(
            {'LONG': [0.0, 1.0], 'SHORT': [0.0, -1.0]},
            index=dates,
        )
        config = CostModelConfig(
            commission_per_share=0.0,
            min_commission=0.0,
            fallback_execution_bps=0.0,
            annual_borrow_rate=0.36,
            annual_financing_rate=0.36,
            day_count=360.0,
        )

        net = _apply_portfolio_costs(
            gross_returns,
            weights,
            prices,
            config,
            initial_capital=100_000.0,
        )

        self.assertAlmostEqual(net.iloc[1], -0.001)
        self.assertAlmostEqual(net.attrs['cost_components']['borrow'], 100.0)
        self.assertAlmostEqual(net.attrs['cost_components']['financing'], 0.0)


if __name__ == '__main__':
    unittest.main()
