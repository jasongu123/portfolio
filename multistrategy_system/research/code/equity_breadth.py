"""
Equity engine breadth: the validated engine on 12 names it has never seen — RESEARCH ONLY. Gate
pre-registered 2026-09-16 before any backtest ran. ibkr2.py is untouched.

Question: is the intraday engine's edge a property of the method or of the twelve names it was
built on? Every feature, threshold and the ML gate were developed on SPY QQQ IWM NVDA JPM GOOG
WMT AAPL MSFT META UNH XOM. Running the identical configuration on twelve other names is out of
sample for all of it.

Universe NEW12, chosen by rule before any data was pulled: the largest company by market cap in
each of the eleven GICS sectors that is not one of the twelve, was an S&P 500 member since
mid-2019 and is not a share-class duplicate, plus the next largest overall:
  AMZN AVGO LLY BAC COST CVX CAT NFLX LIN NEE PLD MU
Configuration: combine_book's exact StrategyBacktester settings (hybrid stock strategies, gap
1.5%/60 min, stock vol regime, RF gate, BASE_33 features, $0.0035/share + 1 bp), 2020-01-01..
2025-12-15, cached per year as equity_breadth_returns_{year}.csv. ORIG12 is the regenerated
equity_returns_{year}.csv. COMBO24 = equal-weight of the two 12-name books (each run on its own
$100k), the natural "24-name engine" proxy.

Pre-registered gate (no-rf daily Sharpe unless stated; rf = flat 5% where stated):
  Q1 the method carries: NEW12 net no-rf Sharpe >= 0.50 over 2020-2025 (ORIG12 reads 0.91) and
     positive in at least 4 of 6 calendar years.                                       binding
  Q2 recency: NEW12 2024-09..latest > 0.                                                binding
  Q3 concentration: no 24-month block > 75% of NEW12 cumulative profit.                 binding
  Q4 breadth pays: corr(NEW12, ORIG12) < 0.70 and COMBO24 no-rf Sharpe >= ORIG12 no-rf.  binding
  Q5 book: walk-forward COMBO24 + trend (as live) over 2020-2025 not worse than ORIG12 + trend
     on rf-Sharpe, worst year and MaxDD.                                                binding
  Informational: per-year table; per-symbol contribution if the backtester exposes it; the
     2 bp cost sensitivity is NOT run here (doubles the runtime); ORIG12's own 2 bp figure is
     the reference (it fails at 2 bp).
Decision: Q1-Q5 pass -> propose extending the live symbol list (versioned, paper first, data
feed permitting). Any fail -> not; NEW12 is not pruned to a subset.
Trials: T-2026-09-16-47 (NEW12), -48 (COMBO24).
Usage: /usr/local/bin/python3 equity_breadth.py [--run]   (--run executes missing yearly backtests)
"""

import os
import sys
import time
import contextlib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW12 = ['AMZN', 'AVGO', 'LLY', 'BAC', 'COST', 'CVX', 'CAT', 'NFLX', 'LIN', 'NEE', 'PLD', 'MU']
YEARS = {str(y): (f'{y}-01-01', f'{y}-12-31') for y in range(2020, 2025)}
YEARS['2025'] = ('2025-01-01', '2025-12-15')
ANN = np.sqrt(252)


def run_missing():
    import combine_book as cb
    from ibkr_backtest import StrategyBacktester
    from feature_sweep import BASE_33, _ProgressFilter
    for y, (start, end) in YEARS.items():
        f = os.path.join(SCRIPT_DIR, f'equity_breadth_returns_{y}.csv')
        if os.path.exists(f):
            print(f"  breadth {y}: cached", flush=True); continue
        t0 = time.time()
        print(f"  breadth {y}: running backtest...", flush=True)
        with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
            bt = StrategyBacktester(symbols=NEW12, start_date=start, end_date=end, initial_capital=100000,
                                    use_stock_strategies=True, stock_strategy_mode='hybrid', gap_threshold=0.015, gap_exit_minutes=60,
                                    use_stock_vol_regime=True, vol_lookback=20, use_rf=True, use_lgbm=False, volatility_multiplier=2.0,
                                    commission_per_share=0.0035, slippage_bps=1)
            bt.load_from_csv()
            bt.ml_feature_names = list(BASE_33)
            eq = bt.run_backtest(long_only=False)
        pv = eq['portfolio_value'] if hasattr(eq, 'columns') else pd.DataFrame(eq)['portfolio_value']
        r = pv.pct_change(); r.index = pd.to_datetime(eq['date']); r = r.dropna()
        r.to_csv(f)
        print(f"  breadth {y}: done ({len(r)} days, {(time.time() - t0) / 60:.0f} min)", flush=True)


def load(prefix):
    parts = [pd.read_csv(os.path.join(SCRIPT_DIR, f'{prefix}_{y}.csv'), index_col=0, parse_dates=True).iloc[:, 0] for y in YEARS]
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()]


def main():
    if '--run' in sys.argv:
        run_missing()
    import book_backtest as bb
    import trend_follow_v3 as v3
    new = load('equity_breadth_returns').rename('new12')
    orig = load('equity_returns').rename('orig12')
    j = pd.concat([new, orig], axis=1).dropna()
    combo = ((j.new12 + j.orig12) / 2).rename('combo24')
    sr = lambda x: float(x.mean() / x.std() * ANN)
    yr = lambda x: {int(y): sr(g) for y, g in x.groupby(x.index.year)}
    print(f"Overlap {len(j)} days {j.index.min().date()}..{j.index.max().date()}; corr(NEW12, ORIG12) {j.new12.corr(j.orig12):+.2f}")
    print(f"{'series':10s}{'no-rf':>7s}{'rf':>7s}{'vol':>7s}{'MDD':>8s}{'worst':>7s}  " + "".join(f"{y:>7d}" for y in range(2020, 2026)))
    for name, s in (('NEW12', j.new12), ('ORIG12', j.orig12), ('COMBO24', combo)):
        yy = yr(s)
        print(f"{name:10s}{sr(s):7.2f}{v3.sharpe(s):7.2f}{s.std() * ANN:7.1%}{v3.maxdd(s):8.1%}{min(yy.values()):7.2f}  " + "".join(f"{yy[y]:7.2f}" for y in range(2020, 2026)))
    yn = yr(j.new12)
    q1 = (sr(j.new12) >= 0.50) and (sum(v > 0 for v in yn.values()) >= 4)
    q2 = sr(j.new12.loc['2024-09-01':]) > 0
    m = j.new12.resample('ME').sum(); tot = (1 + m).prod() - 1
    blocks = {str(a.date()): (1 + m.loc[a:a + pd.DateOffset(months=23)]).prod() - 1 for a in m.index[:-23]}
    conc = max(blocks.values()) / tot if tot > 0 else np.inf
    q3 = conc <= 0.75
    q4 = (j.new12.corr(j.orig12) < 0.70) and (sr(combo) >= sr(j.orig12))
    tr = pd.read_csv(os.path.join(SCRIPT_DIR, 'results', 'headline_2026-09_daily_new.csv'), index_col=0, parse_dates=True)['trend_live'].rename('trend')
    books = {}
    for name, e in (('ORIG12', j.orig12), ('COMBO24', combo)):
        full = pd.concat([e.rename('equity'), tr], axis=1); bb.FULL_FRAME = full
        books[name] = bb.book_returns(full.loc['2020-01-01':'2025-12-31'].dropna(), 'WF')
    yo, yc = bb.yearly(books['ORIG12']), bb.yearly(books['COMBO24'])
    q5 = (bb.sharpe(books['COMBO24']) >= bb.sharpe(books['ORIG12'])) and (min(yc.values()) >= min(yo.values())) and (bb.maxdd(books['COMBO24']) >= bb.maxdd(books['ORIG12']))
    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if q1 else 'fail'}  Q1 NEW12 no-rf {sr(j.new12):.2f} (>= 0.50); positive years {sum(v > 0 for v in yn.values())}/6 (>= 4)")
    print(f"  {'PASS' if q2 else 'fail'}  Q2 2024-09+ {sr(j.new12.loc['2024-09-01':]):+.2f} (> 0)")
    print(f"  {'PASS' if q3 else 'fail'}  Q3 best 24-month block {conc:.0%} of profit (<= 75%)")
    print(f"  {'PASS' if q4 else 'fail'}  Q4 corr {j.new12.corr(j.orig12):+.2f} (< 0.70); COMBO24 {sr(combo):.2f} vs ORIG12 {sr(j.orig12):.2f}")
    print(f"  {'PASS' if q5 else 'fail'}  Q5 WF book rf {bb.sharpe(books['ORIG12']):.2f} -> {bb.sharpe(books['COMBO24']):.2f}; worst {min(yo.values()):+.2f} -> {min(yc.values()):+.2f}; MDD {bb.maxdd(books['ORIG12']):.1%} -> {bb.maxdd(books['COMBO24']):.1%}")
    print(f"\nVERDICT: {'the method carries; propose extending the live symbol list (versioned, paper first)' if all([q1, q2, q3, q4, q5]) else 'not adopted'}")
    pd.concat([j, combo], axis=1).to_csv(os.path.join(SCRIPT_DIR, 'results', 'equity_breadth_daily.csv'))


if __name__ == '__main__':
    main()
