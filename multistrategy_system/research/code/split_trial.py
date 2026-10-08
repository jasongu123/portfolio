"""
ETF / single-stock split trial — RESEARCH ONLY. Pre-registered 2026-10-01 before any run.
ibkr2.py untouched; ibkr_backtest.py untouched (variants live in the SplitBT subclass below).

Background (ENGINE_BREAKDOWN.md): the engine's ETF side (SPY/QQQ/IWM intraday momentum + a little
swing) earned ~nothing in 2020-25; the stock side earned everything, almost all of it from the
long-only swing book (~65% invested), while stock intraday mean reversion lost money before costs.
The ETF momentum departs from its source paper (Zarattini, Aziz & Barbon 2024, "Beat the Market")
in five ways: decisions every minute instead of every 30 minutes; a 2x noise band; a VIX >= 17
gate; an ML gate; an RSI size multiplier. The paper has no stock rule at all.

Each sleeve runs on its own $100k, one backtest per year 2020..2025-12-15, per-minute VIX,
$0.0035/share commission (min $0.35) + the stated slippage per side. Intraday arms log every
position (record_positions) so gross and net can be read separately.

ETF sleeve (SPY, QQQ, IWM; intraday only, no swing)
  E0p paper rules, paper costs: 30-minute decisions 10:00-15:30, band = 14-day average move
      (multiplier 1), no VIX gate, no ML gate, no RSI multiplier, slippage 0.03 bp (~$0.001/share)
  E0  paper rules, realistic costs: as E0p with 0.5 bp a side (about 2x the half-spread)
  E1  E0 + decisions every minute          E2  E0 + band multiplier 2.0
  E3  E0 + VIX >= 17 gate                  E4  E0 + pooled ML gate (forest, 0.55)
  E5  E0 + RSI size multiplier             E6  the engine as is: every minute, 2x band, VIX gate,
                                               ML gate, RSI multiplier, 1 bp
Stock sleeve (the 21 single stocks; 1 bp a side)
  S0  the engine as is: mean reversion + swing + ML gate
  S1  swing only (no intraday)
  S2  mean reversion only (no swing), with the ML gate
  S3  the paper's momentum rule on stocks, intraday only, paper rules as E0 (no swing)
  S4  passive: the 21 names equal-weight, rebalanced daily, at S1's average invested share of
      capital over 2020-25 (no backtest; computed from daily closes)

Pre-registered decisions (no-rf daily Sharpe, 2020-01..2025-12-15 unless stated; last 2y =
2024-01-01..2025-12-15; concentration = best 24-month block's share of total profit):
  E-A paper replication: E0 passes if Sharpe >= 0.50 AND last-2y Sharpe > 0 AND concentration
      <= 75%. Pass -> the ETF sleeve's intraday becomes a paper-rules candidate for a forward paper
      test (no live change from this trial). Fail -> recommend switching ETF intraday off.
      E1-E5 vs E0 and E6 are diagnostic: what each departure from the paper costs.
  S-A drop mean reversion: accepted if S1 Sharpe >= S0 AND S1 last-2y Sharpe >= S0's - 0.05
      (removing a part needs no improvement bar, only no damage).
  S-B stock momentum: S3 is a candidate stock intraday sleeve only if it passes E-A's three tests.
  S-C swing timing: the swing book's timing counts as an edge only if S1 Sharpe >= S4 + 0.10 AND
      S1 last-2y Sharpe >= S4's. Otherwise the stock sleeve is described as a stock holding with
      stops, and the next step is a redesign question, not a tuning one.
Trials: T-2026-10-01-59 (E0p) .. -71 (S4); see TRIAL_REGISTRY.md.

Usage:
  python3 split_trial.py --run E0 2023
  python3 split_trial.py --report
"""
import argparse
import contextlib
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate_trial import CORE, BREADTH, YEARS, ANN  # noqa: E402

# v2 (2026-10-01): rerun on the audit-corrected engine (BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md);
# the original results stay in results/split_trial/.
OUT = os.path.join(HERE, 'results', 'split_trial_v2')
ETFS = ['SPY', 'QQQ', 'IWM']
STOCKS = [s for s in CORE + BREADTH if s not in ETFS]
LAST2 = ('2024-01-01', '2025-12-15')
RECENT = '2024-09-01'

BASE = dict(initial_capital=100000, use_stock_strategies=True, stock_strategy_mode='hybrid',
            gap_threshold=0.015, gap_exit_minutes=60, use_stock_vol_regime=True, vol_lookback=20,
            use_lgbm=False, commission_per_share=0.0035)
# use_har_forecast=False: the paper has no HAR scaling and neither does the deployed ETF_PAPER_RULES
# (audit item 4; the v1 E0 inherited the engine's HAR default).
PAPER = dict(trade_freq=30, volatility_multiplier=1.0, vix_min=None, use_rf=False, use_swing=False, use_har_forecast=False,
             slippage_bps=0.5, no_rsi=True)
ASIS = dict(trade_freq=1, volatility_multiplier=2.0, vix_min=17, use_rf=True, slippage_bps=1, no_rsi=False)

ARMS = {
    'E0p': (ETFS, {**PAPER, 'slippage_bps': 0.03}),
    'E0': (ETFS, PAPER),
    'E1': (ETFS, {**PAPER, 'trade_freq': 1}),
    'E2': (ETFS, {**PAPER, 'volatility_multiplier': 2.0}),
    'E3': (ETFS, {**PAPER, 'vix_min': 17}),
    'E4': (ETFS, {**PAPER, 'use_rf': True}),
    'E5': (ETFS, {**PAPER, 'no_rsi': False}),
    'E6': (ETFS, {**ASIS, 'use_swing': False}),
    # Added 2026-10-01 after the implementation audit (BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md, item 4):
    # E0 inherited use_har_forecast=True; the deployed ETF_PAPER_RULES has no HAR. Diagnostic T-83, same E-A rule.
    'E0nh': (ETFS, {**PAPER, 'use_har_forecast': False}),
    'S0': (STOCKS, {**ASIS, 'use_swing': True}),
    'S1': (STOCKS, {**ASIS, 'use_swing': True, 'use_stock_strategies': False, 'use_rf': False}),
    'S2': (STOCKS, {**ASIS, 'use_swing': False}),
    'S3': (STOCKS, {**PAPER, 'slippage_bps': 1, 'all_momentum': True}),
}


def make_class():
    from ibkr_backtest import StrategyBacktester

    class SplitBT(StrategyBacktester):
        def __init__(self, *a, no_rsi=False, all_momentum=False, **kw):
            super().__init__(*a, **kw)
            self.no_rsi = no_rsi
            self.all_momentum = all_momentum

        def _rsi_gamma_multiplier(self, rsi):
            return 1.0 if self.no_rsi else StrategyBacktester._rsi_gamma_multiplier(rsi)

        def is_etf(self, symbol):
            # S3: route every symbol through the paper's momentum rule
            return True if self.all_momentum else super().is_etf(symbol)

    return SplitBT


def run(arm, y):
    from feature_sweep import BASE_33, _ProgressFilter
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f'{arm}_{y}.pkl')
    if os.path.exists(f):
        print(f"  {arm} {y}: cached", flush=True)
        return
    t0 = time.time()
    syms, opts = ARMS[arm]
    start, end = YEARS[str(y)]
    SplitBT = make_class()
    with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
        bt = SplitBT(symbols=list(syms), start_date=start, end_date=end, **{**BASE, **opts})
        bt.record_positions = True
        bt.load_from_csv()
        bt.ml_feature_names = list(BASE_33)
        eq = bt.run_backtest(long_only=False)
    # first day measured from initial capital (audit item 5)
    r = pd.Series(bt.backtest_daily_returns, index=pd.to_datetime(eq['date']))
    res = {'returns': r, 'positions': pd.DataFrame(getattr(bt, 'position_log', [])),
           'swing': pd.DataFrame(bt.swing_trades),
           'equity': eq[['date', 'portfolio_value', 'daily_pnl']].copy()}
    with open(f, 'wb') as fh:
        pickle.dump(res, fh)
    print(f"  {arm} {y}: {len(res['positions'])} positions, {len(res['swing'])} swing trades, "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)


def load(arm):
    parts = [pd.read_pickle(os.path.join(OUT, f'{arm}_{y}.pkl')) for y in YEARS]
    r = pd.concat([p['returns'] for p in parts]).sort_index()
    pos = pd.concat([p['positions'] for p in parts], ignore_index=True)
    sw = pd.concat([p['swing'] for p in parts], ignore_index=True)
    eqs = pd.concat([p['equity'] for p in parts], ignore_index=True)
    return r[~r.index.duplicated()].rename(arm), pos, sw, eqs


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def concentration(x):
    """Best 24-month block's share of total profit (sum of daily returns); inf if no profit."""
    x = x.dropna()
    total = x.sum()
    if total <= 0:
        return float('inf')
    return float(x.rolling(504, min_periods=400).sum().max() / total)


def exposure_from_swing(sw, eqs, syms):
    """Daily invested share of capital of a swing book (held notional at the prior close)."""
    px = {}
    for s in syms:
        d = pd.read_csv(os.path.join(HERE, 'backtest_data', f'{s}_daily.csv'))
        px[s] = d.assign(date=pd.to_datetime(d['date'])).set_index('date')['close']
    px = pd.DataFrame(px).sort_index()
    e = eqs.assign(date=pd.to_datetime(eqs['date'])).drop_duplicates('date').set_index('date').sort_index()
    pv = e['portfolio_value'].shift(1).fillna(BASE['initial_capital'])
    dates = e.index
    prior = px.reindex(dates).shift(1)
    held = pd.Series(0.0, index=dates)
    for r in sw.itertuples():
        m = (dates > pd.Timestamp(r.entry_date)) & (dates <= pd.Timestamp(r.exit_date))
        held[m] += r.shares * prior.loc[m, r.symbol]
    return (held / pv).dropna(), px.pct_change(fill_method=None)


def stats(x):
    x = x.dropna()
    eq = np.cumprod(1 + x.values)
    dd = float((eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1).min())
    return {'Sharpe': sr(x), 'rf Sharpe': float((x - 0.05 / 252).mean() / x.std() * ANN), 'return %/yr': x.mean() * 252 * 100,
            'MaxDD %': dd * 100, 'last 2y': sr(x.loc[LAST2[0]:LAST2[1]]), 'last 15m': sr(x.loc[RECENT:]),
            **{str(y): sr(x.loc[str(y)]) for y in YEARS}, 'conc': concentration(x)}


def report():
    pd.set_option('display.width', 240)
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    avail = [a for a in ARMS if all(os.path.exists(os.path.join(OUT, f'{a}_{y}.pkl')) for y in YEARS)]
    R, P, SW, EQ = {}, {}, {}, {}
    for a in avail:
        R[a], P[a], SW[a], EQ[a] = load(a)
    if 'S1' in avail:
        expo, rets = exposure_from_swing(SW['S1'], EQ['S1'], STOCKS)
        c = float(expo.mean())
        ew = rets[STOCKS].mean(axis=1).reindex(R['S1'].index).fillna(0.0)
        R['S4'] = (c * ew).rename('S4')
        R['S4 timed'] = (expo.reindex(R['S1'].index).fillna(0.0) * ew).rename('S4 timed')
        emit(f"S4: equal-weight 21 names at S1's average invested share {c:.1%} (S4 timed: S1's daily share, diagnostic)")
    df = pd.DataFrame({a: stats(R[a]) for a in R}).T
    emit(df.to_string(float_format=lambda v: f'{v:.2f}'))
    df.to_csv(os.path.join(OUT, 'summary.csv'))

    emit("\nIntraday positions, 2020-25 (bps of traded notional):")
    rows = {}
    for a in avail:
        p = P[a]
        if len(p):
            n = p['notional'].sum()
            rows[a] = {'positions': len(p), 'per year': len(p) / 6, 'win%': (p['net'] > 0).mean() * 100,
                       'gross bps': p['gross'].sum() / n * 1e4, 'cost bps': p['cost'].sum() / n * 1e4,
                       'net bps': p['net'].sum() / n * 1e4, 'net $k': p['net'].sum() / 1e3}
    emit(pd.DataFrame(rows).T.to_string(float_format=lambda v: f'{v:.2f}'))

    def has(*a):
        return all(x in df.index for x in a)

    emit("\nDECISIONS (pre-registered)")
    if has('E0'):
        r = df.loc['E0']
        t = [r.Sharpe >= 0.50, r['last 2y'] > 0, r.conc <= 0.75]
        emit(f"E-A paper replication (E0): Sharpe {r.Sharpe:.2f} >= 0.50 {'ok' if t[0] else 'fail'} | last 2y {r['last 2y']:+.2f} > 0 "
             f"{'ok' if t[1] else 'fail'} | concentration {r.conc:.0%} <= 75% {'ok' if t[2] else 'fail'} => "
             f"{'PASS: paper-rules ETF candidate for a forward test' if all(t) else 'FAIL: recommend ETF intraday off'}")
        for a in ('E0p', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6'):
            if a in df.index:
                emit(f"   {a}: Sharpe {df.loc[a].Sharpe:+.2f} ({df.loc[a].Sharpe - r.Sharpe:+.2f} vs E0), last 2y {df.loc[a]['last 2y']:+.2f}")
    if has('S0', 'S1'):
        a, b = df.loc['S1'], df.loc['S0']
        t = [a.Sharpe >= b.Sharpe, a['last 2y'] >= b['last 2y'] - 0.05]
        emit(f"S-A drop mean reversion: S1 {a.Sharpe:.2f} vs S0 {b.Sharpe:.2f} {'ok' if t[0] else 'fail'} | last 2y {a['last 2y']:+.2f} vs "
             f"{b['last 2y'] - 0.05:+.2f} {'ok' if t[1] else 'fail'} => {'ACCEPT: drop mean reversion' if all(t) else 'keep'}")
    if has('S3'):
        r = df.loc['S3']
        t = [r.Sharpe >= 0.50, r['last 2y'] > 0, r.conc <= 0.75]
        emit(f"S-B stock momentum (S3): Sharpe {r.Sharpe:.2f} {'ok' if t[0] else 'fail'} | last 2y {r['last 2y']:+.2f} {'ok' if t[1] else 'fail'} | "
             f"concentration {r.conc:.0%} {'ok' if t[2] else 'fail'} => {'PASS: candidate' if all(t) else 'FAIL'}")
    if has('S1', 'S4'):
        a, b = df.loc['S1'], df.loc['S4']
        t = [a.Sharpe >= b.Sharpe + 0.10, a['last 2y'] >= b['last 2y']]
        emit(f"S-C swing timing: S1 {a.Sharpe:.2f} vs passive {b.Sharpe:.2f} + 0.10 {'ok' if t[0] else 'fail'} | last 2y {a['last 2y']:+.2f} vs "
             f"{b['last 2y']:+.2f} {'ok' if t[1] else 'fail'} => {'timing is an edge' if all(t) else 'NO EDGE: a stock holding with stops'}")
    if has('S2'):
        emit(f"(S2 mean reversion alone: Sharpe {df.loc['S2'].Sharpe:+.2f}, last 2y {df.loc['S2']['last 2y']:+.2f})")
    with open(os.path.join(OUT, 'report.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs=2, metavar=('ARM', 'YEAR'))
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.run:
        run(a.run[0], int(a.run[1]))
    if a.report:
        report()
