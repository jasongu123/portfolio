"""
Equity engine breakdown — DIAGNOSTIC ONLY (2026-10-01). No decisions are taken from it; any fix it
suggests becomes a separate pre-registered trial that must have a mechanism and must not hurt
2021-23.

Question: which part of the engine stopped earning in 2024-25? The engine's daily P&L is split
into its sources: intraday positions tagged by the candidate rule that opened them (ETF noise-band
momentum, stock mean reversion, stock gap), plus swing holdings (the residual: change in portfolio
value minus intraday P&L). Each intraday position also carries direction, symbol group, entry time
and P&L before and after costs, so the sources can be cut by year, direction, group, VIX at entry,
market trend state and time of day.

Runs: the gate trial's G1 (pooled forest, the validated gate) and G0 (no gate) configs, 24 names,
2020..2025-12-15, one backtest per year, with ibkr_backtest's opt-in record_positions logger (it
records only; each year's returns must equal gate_trial's cached CSV exactly).

Usage:
  python3 engine_breakdown.py --run rf 2023     one arm-year (rf = gated, none = ungated)
  python3 engine_breakdown.py --report          tables to stdout and results/engine_breakdown/
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
from gate_trial import CORE, BREADTH, OUT as GATE_OUT, YEARS, ANN  # noqa: E402

# v2 (2026-10-01): audit-corrected engine; v1 results stay in results/engine_breakdown/.
OUT = os.path.join(HERE, 'results', 'engine_breakdown_v2')
INITIAL = 100000
ETFS = {'SPY', 'QQQ', 'IWM'}
RECENT = '2024-09-01'
# Verified 2026-10-01: a plain rerun (no recorder) on the new Mac (arm64) reproduces this run
# exactly but differs from the old Mac's cached rf 2021 (+32.1% vs +31.7% for the year); every
# other arm-year matches to 1e-16. A cross-machine difference, not the recorder.
KNOWN_MACHINE_DIFF = {('rf', '2021')}


def group_of(sym):
    return 'ETF' if sym in ETFS else ('core stock' if sym in CORE else 'breadth stock')


def run(arm, y):
    from ibkr_backtest import StrategyBacktester
    from feature_sweep import BASE_33, _ProgressFilter
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f'{arm}_{y}.pkl')
    if os.path.exists(f):
        print(f"  {arm} {y}: cached", flush=True)
        return
    t0 = time.time()
    start, end = YEARS[str(y)]
    kw = dict(symbols=CORE + BREADTH, start_date=start, end_date=end, initial_capital=100000,
              use_stock_strategies=True, stock_strategy_mode='hybrid', gap_threshold=0.015, gap_exit_minutes=60,
              use_stock_vol_regime=True, vol_lookback=20, use_rf=(arm != 'none'), use_lgbm=False,
              volatility_multiplier=2.0, commission_per_share=0.0035, slippage_bps=1)
    with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
        bt = StrategyBacktester(**kw)
        bt.record_positions = True
        bt.load_from_csv()
        bt.ml_feature_names = list(BASE_33)
        eq = bt.run_backtest(long_only=False)
    res = {'equity': eq[['date', 'portfolio_value', 'daily_pnl']].copy(),
           'returns': pd.Series(bt.backtest_daily_returns, index=pd.to_datetime(eq['date'])),
           'positions': pd.DataFrame(getattr(bt, 'position_log', [])),
           'swing': pd.DataFrame(bt.swing_trades)}
    with open(f, 'wb') as fh:
        pickle.dump(res, fh)
    print(f"  {arm} {y}: {len(res['positions'])} positions, {len(res['swing'])} swing trades, "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)


def load(arm):
    eqs, pos, sw = [], [], []
    for y in YEARS:
        r = pd.read_pickle(os.path.join(OUT, f'{arm}_{y}.pkl'))
        e = r['equity'].copy()
        e['date'] = pd.to_datetime(e['date'])
        e['prev_pv'] = e['portfolio_value'].shift(1).fillna(INITIAL)
        e['swing_pnl'] = e['portfolio_value'] - e['prev_pv'] - e['daily_pnl']
        mine = (e['portfolio_value'] / e['prev_pv'] - 1).values
        diff = float(np.abs(mine - r['returns'].values).max())
        if diff > 1e-12:
            sys.exit(f"SANITY FAILED: {arm} {y} curve and engine returns disagree ({diff:.2e})")
        p = r['positions'].copy()
        p['date'] = pd.to_datetime(p['date'])
        eqs.append(e)
        pos.append(p)
        s = r['swing'].copy()
        if len(s):
            s['year'] = y
        sw.append(s)
    return pd.concat(eqs, ignore_index=True), pd.concat(pos, ignore_index=True), pd.concat(sw, ignore_index=True)


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def periods():
    out = [(y, f'{y}-01-01', f'{y}-12-31') for y in YEARS]
    out.append(('last 15m', RECENT, '2025-12-31'))
    out.append(('2020-23', '2020-01-01', '2023-12-31'))
    out.append(('2024-25', '2024-01-01', '2025-12-31'))
    return out


def component_returns(eq, pos):
    """Daily return contribution of each source: P&L / prior-day portfolio value."""
    daily = pos.groupby(['date', 'source'])['net'].sum().unstack(fill_value=0.0)
    daily = daily.reindex(eq['date']).fillna(0.0)
    daily['swing'] = eq.set_index('date')['swing_pnl']
    # costs booked outside position segments (end-of-backtest liquidation) land in swing via the residual
    other = eq.set_index('date')['daily_pnl'] - daily.drop(columns='swing').sum(axis=1)
    daily['swing'] = daily['swing'] + other
    return daily.div(eq.set_index('date')['prev_pv'], axis=0)


def vix_minute_lookup():
    v = pd.read_csv(os.path.join(HERE, 'backtest_data', 'VIX_1m.csv'), parse_dates=['date'])
    return v.set_index('date')['close'].sort_index()


def spy_trend():
    d = pd.read_csv(os.path.join(HERE, 'backtest_data', 'SPY_daily.csv'))
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')['close'].sort_index()
    sma = d.rolling(200).mean()
    # state known at the open: yesterday's close vs yesterday's 200-day mean
    return (d > sma).shift(1).map({True: 'SPY above 200d', False: 'SPY below 200d'})


def swing_vs_passive(eq, comp, sw):
    """Swing P&L against holding the 24-name equal-weight basket at the swing book's own
    day-by-day exposure (held notional at the prior close / prior portfolio value)."""
    syms = CORE + BREADTH
    px = {}
    for s in syms:
        d = pd.read_csv(os.path.join(HERE, 'backtest_data', f'{s}_daily.csv'))
        px[s] = d.assign(date=pd.to_datetime(d['date'])).set_index('date')['close']
    px = pd.DataFrame(px).sort_index()
    rets = px.pct_change(fill_method=None)
    dates = comp.index
    pv = eq.set_index('date')['prev_pv']
    held = pd.DataFrame(0.0, index=dates, columns=syms)
    prior_close = px.reindex(dates).shift(1)
    for r in sw.itertuples():
        m = (dates > pd.Timestamp(r.entry_date)) & (dates <= pd.Timestamp(r.exit_date))
        held.loc[m, r.symbol] += r.shares * prior_close.loc[m, r.symbol]
    expo = held.sum(axis=1) / pv
    recon = (held * rets.reindex(dates)[syms]).sum(axis=1) / pv
    ew = rets[syms].mean(axis=1).reindex(dates).fillna(0.0)
    passive = expo * ew
    rows = {}
    for name, s, e in periods():
        y, pa = comp['swing'].loc[s:e], passive.loc[s:e]
        rows[name] = {'avg exposure %': expo.loc[s:e].mean() * 100, 'swing %': y.sum() * 100, 'passive %': pa.sum() * 100,
                      'swing SR': sr(y), 'passive SR': sr(pa), 'corr': y.corr(ew.loc[s:e])}
    return pd.DataFrame(rows).T, float(recon.corr(comp['swing']))


def bps_table(p, by):
    g = p.groupby(by)
    t = pd.DataFrame({'n': g.size(), 'win%': g.apply(lambda x: (x['net'] > 0).mean() * 100),
                      'gross bps': g.apply(lambda x: x['gross'].sum() / x['notional'].sum() * 1e4),
                      'cost bps': g.apply(lambda x: x['cost'].sum() / x['notional'].sum() * 1e4),
                      'net bps': g.apply(lambda x: x['net'].sum() / x['notional'].sum() * 1e4),
                      'net $k': g['net'].sum() / 1e3})
    return t


def report():
    pd.set_option('display.width', 230)
    pd.set_option('display.max_rows', 300)
    fmt = lambda v: f'{v:.2f}'
    lines = []

    def emit(s=''):
        print(s)
        lines.append(str(s))

    for arm in ('rf', 'none'):
        eq, pos, sw = load(arm)
        emit(f"\n{'=' * 100}\nARM {arm} ({'pooled forest gate, G1' if arm == 'rf' else 'no gate, G0'}): sanity ok, "
             f"{len(pos)} intraday positions, {len(sw)} swing trades")
        comp = component_returns(eq, pos)
        total = eq.set_index('date')['portfolio_value'] / eq.set_index('date')['prev_pv'] - 1
        comp.to_csv(os.path.join(OUT, f'components_{arm}.csv'))

        rows = {}
        for name, s, e in periods():
            c = comp.loc[s:e]
            row = {f'{k} %': c[k].sum() * 100 for k in comp.columns}
            row['total %'] = c.sum(axis=1).sum() * 100
            row['Sharpe total'] = sr(total.loc[s:e])
            row.update({f'SR {k}': sr(c[k]) for k in comp.columns})
            rows[name] = row
        emit("\nReturn contribution (sum of daily contributions, % of capital) and standalone no-rf Sharpe:")
        emit(pd.DataFrame(rows).T.to_string(float_format=fmt))

        if len(sw):
            t, rc = swing_vs_passive(eq, comp, sw)
            emit(f"\nSwing vs the 24-name equal-weight basket held at the swing book's own daily exposure "
                 f"(reconstruction corr with booked swing P&L {rc:.3f}):")
            emit(t.to_string(float_format=fmt))
        pos = pos.copy()
        pos['year'] = pos['date'].dt.year
        pos['period'] = np.where(pos['date'] >= '2024-01-01', '2024-25', '2020-23')
        pos['group'] = pos['symbol'].map(group_of)
        pos['side'] = pos['direction'].map({1: 'long', -1: 'short'})
        emit("\nIntraday positions by source and year (bps of traded notional):")
        emit(bps_table(pos, ['source', 'year']).to_string(float_format=fmt))
        emit("\nBy source, side and period:")
        emit(bps_table(pos, ['source', 'side', 'period']).to_string(float_format=fmt))
        emit("\nBy source, symbol group and period:")
        emit(bps_table(pos, ['source', 'group', 'period']).to_string(float_format=fmt))

        vix = vix_minute_lookup()
        et = pd.to_datetime(pos['entry_time'])
        pos['vix'] = vix.reindex(et, method='ffill').values
        pos['vix bucket'] = pd.cut(pos['vix'], [0, 15, 20, 25, 30, 200], labels=['<15', '15-20', '20-25', '25-30', '30+'])
        emit("\nBy source, VIX at entry and period:")
        emit(bps_table(pos, ['source', 'vix bucket', 'period']).to_string(float_format=fmt))
        tr = spy_trend()
        pos['trend'] = tr.reindex(pos['date'].dt.normalize()).values
        emit("\nBy source, market trend and period:")
        emit(bps_table(pos, ['source', 'trend', 'period']).to_string(float_format=fmt))
        pos['entry window'] = pd.cut(pos['entry_minute'], [0, 60, 120, 240, 330, 400],
                                     labels=['30-60m', '60-120m', '120-240m', '240-330m', '330m+'], right=False)
        emit("\nBy source, entry time and period:")
        emit(bps_table(pos, ['source', 'entry window', 'period']).to_string(float_format=fmt))
        emit("\nBy symbol, 2020-23 vs 2024-25 (net $k):")
        bysym = pos.pivot_table(index='symbol', columns='period', values='net', aggfunc='sum') / 1e3
        emit(bysym.sort_values('2024-25').to_string(float_format=fmt))

        if len(sw):
            sw = sw.copy()
            sw['group'] = sw['symbol'].map(group_of)
            emit("\nSwing trades by year (realized at exit, $k):")
            g = sw.groupby('year')
            emit(pd.DataFrame({'n': g.size(), 'win%': g.apply(lambda x: (x['pnl'] > 0).mean() * 100),
                               'net $k': g['pnl'].sum() / 1e3,
                               'avg return %': g['return'].mean() * 100}).to_string(float_format=fmt))
            emit("\nSwing by exit reason and year (net $k):")
            emit((sw.pivot_table(index='exit_reason', columns='year', values='pnl', aggfunc='sum') / 1e3).to_string(float_format=fmt))
            emit("\nSwing by group and year (net $k):")
            emit((sw.pivot_table(index='group', columns='year', values='pnl', aggfunc='sum') / 1e3).to_string(float_format=fmt))
        pos.to_pickle(os.path.join(OUT, f'positions_{arm}.pkl'))

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
