"""
Regenerate the headline equity sleeve (FULL-12 gated engine, combine_book's configuration, per-minute
VIX gate) on the audit-corrected backtester (BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md). A
correction, not a trial: no gate, no rule changes; the corrected series replaces the headline equity
whatever the direction, and the previous series stays in results/vixintraday/ for comparison.
Writes results/audit_rerun/headline_equity_{year}.csv (first day measured from initial capital).
Usage: python3 audit_rerun_headline.py 2020 [2021 ...]; then python3 headline_rebuild.py --equity audit
"""
import contextlib
import os
import sys
import time

import pandas as pd

import combine_book as cb
from feature_sweep import BASE_33, _ProgressFilter
from ibkr_backtest import StrategyBacktester

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'audit_rerun')
YEARS = {str(y): (f'{y}-01-01', f'{y}-12-31') for y in range(2020, 2025)}
YEARS['2025'] = ('2025-01-01', '2025-12-15')


def run(y):
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f'headline_equity_{y}.csv')
    if os.path.exists(f):
        print(f"  headline {y}: cached", flush=True)
        return
    t0 = time.time()
    start, end = YEARS[y]
    with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
        bt = StrategyBacktester(symbols=cb.SYMBOLS, start_date=start, end_date=end, initial_capital=100000,
                                use_stock_strategies=True, stock_strategy_mode='hybrid', gap_threshold=0.015,
                                gap_exit_minutes=60, use_stock_vol_regime=True, vol_lookback=20, use_rf=True,
                                use_lgbm=False, volatility_multiplier=2.0, commission_per_share=0.0035, slippage_bps=1)
        bt.load_from_csv()
        bt.ml_feature_names = list(BASE_33)
        eq = bt.run_backtest(long_only=False)
    pd.Series(bt.backtest_daily_returns, index=pd.to_datetime(eq['date']), name='equity').to_csv(f)
    print(f"  headline {y}: done ({len(eq)} days, {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == '__main__':
    for y in sys.argv[1:] or sorted(YEARS):
        run(y)
