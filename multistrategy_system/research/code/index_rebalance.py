"""
S&P 500 additions and deletions: trading against index-fund rebalancing — RESEARCH ONLY.
Pre-registered 2026-10-02 before any return was computed. Nothing here touches the live engine.

Mechanism: index funds must buy every addition and sell every deletion at the close of the day
before the change takes effect (E-1), whatever the price. Classic findings: additions rise into
inclusion and partly reverse afterwards; deletions fall and partly rebound. The question is whether
anything is left to earn by taking the other side, now that the effect is famous (several studies
report it has shrunk sharply since the 2000s). This is a liquidity-provision trade with a named
forced counterparty, the kind of edge that should outlast pure price patterns, if it exists.

Data: IBKR adjusted daily closes (backtest_data/sp500_prices_long.csv); changes and EFFECTIVE dates
from the S&P 500 change log (backtest_data/sp500_changes.csv; e.g. TSLA 2020-12-21). Announcement
dates are not available, so the run-up into inclusion is reported but NOT traded (trading it would
assume knowing the change before it was announced). Stock-specific return = stock minus the
equal-weight return of the members priced that day.

Trades, entered at the close of E-1 (the closing auction where index funds trade; the change is
public by then), held 20 trading days (closes E .. E+19), market-hedged:
  ADD  short each added stock          (hypothesis: post-inclusion reversal)
  DEL  long each deleted stock that still trades afterwards (hypothesis: post-deletion rebound)
Each leg holds its open positions in equal weight (zero on days with none). Costs 6 bp per side
per position (5 bp stock + 1 bp hedge); shorts pay 0.5% a year borrow.

Main window: changes effective 2020-01-01 .. the last change with 20 days of prices after it.
Context (no decision): additions 2005-2019 (only names still members in 2020+ are priced, so this
is survivorship-biased) and deletions 2005-2019 where priced.

Pre-registered decision, per leg (two legs, so t thresholds are Bonferroni-adjusted):
  R1 mean net 20-day stock-specific return of the trade, averaged by effective date (each date
     counts once), > 0 with t >= 2.24
  R2 calendar-time leg net Sharpe >= 0.50
  R3 leg net return over the last 24 months > 0
  R4 best 24-month block <= 75% of the leg's total profit
  A leg passing all four is a candidate for a paper forward test (no live change from this trial).
Reported, no decision: event study of stock-specific returns over E-11..E-2 (run-up, not
tradable), E-1 (the index funds' trading day), E..E+4, E..E+19 and E..E+59; per-year counts;
correlation with the existing sleeves.
Trials: T-2026-10-02-86 (ADD), -87 (DEL).

Usage: python3 index_rebalance.py            (seconds; writes results/index_rebalance/)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taking_sides import membership  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'index_rebalance')
MAIN_START = '2020-01-01'
CONTEXT = ('2005-01-01', '2019-12-31')
HOLD = 20
COST_SIDE = 0.0006
BORROW = 0.005
T_MIN = 2.24
ANN = np.sqrt(252)
WINDOWS = {'run-up E-11..E-2': (-11, -2), 'E-1 (funds trade)': (-1, -1), 'E..E+4': (0, 4),
           'E..E+19 (traded)': (0, 19), 'E..E+59': (0, 59)}


def load():
    px = pd.read_csv(os.path.join(BD, 'sp500_prices_long.csv'), index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(os.path.join(BD, 'sp500_universe_2020_2026.csv')).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(os.path.join(BD, 'sp500_changes.csv'), parse_dates=['date'])
    mem = membership(chg, uni)
    days = px.index
    r = px.pct_change(fill_method=None)
    member = pd.DataFrame(False, index=days, columns=px.columns)
    for s in px.columns:
        if s in mem:
            a, rm = mem[s]
            m = np.ones(len(days), bool)
            if pd.notna(a):
                m &= days >= a
            if pd.notna(rm) and (pd.isna(a) or rm > a):
                m &= days < rm
            member[s] = m
    mkt = r.where(member & px.notna()).mean(axis=1)
    e = r.sub(mkt, axis=0)
    return px, e, days, chg


def events(px, e, days, chg):
    rows = []
    for d, add, rem, reason in zip(chg.date, chg.added, chg.removed, chg.reason):
        if d < pd.Timestamp(CONTEXT[0]):
            continue
        i = int(days.searchsorted(d))                   # first trading day on or after the effective date
        if i < 12 or i + HOLD >= len(days):
            continue
        for leg, sym in (('ADD', add), ('DEL', rem)):
            if not isinstance(sym, str):
                continue
            s = sym.replace('.', '-')
            if s not in e.columns:
                continue
            seg = e[s].iloc[i - 1:i + HOLD]
            if seg.notna().sum() < HOLD:                   # must trade through the holding period
                continue
            rows.append({'leg': leg, 'symbol': s, 'effective': days[i], 'i': i, 'reason': reason})
    return pd.DataFrame(rows)


def event_study(ev, e):
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    out = []
    for i, s in zip(ev.i, ev.symbol):
        j = col[s]
        out.append({k: (np.nansum(E[i + a:i + b + 1, j]) if i + b < len(E) else np.nan) for k, (a, b) in WINDOWS.items()})
    return pd.DataFrame(out, index=ev.index)


def leg_returns(ev, e, days, leg):
    sub = ev[ev.leg == leg]
    sgn = -1.0 if leg == 'ADD' else 1.0
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    n = len(days)
    net_sum, gross_sum, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    trade_net = []
    for i, s in zip(sub.i, sub.symbol):
        j = col[s]
        start, end = i, i + HOLD - 1                       # returns of closes E .. E+19 after entry at E-1
        tot = 0.0
        for t in range(start, end + 1):
            g = sgn * (0.0 if np.isnan(E[t, j]) else E[t, j])
            c = (COST_SIDE if t == start else 0.0) + (COST_SIDE if t == end else 0.0) + (BORROW / 252 if sgn < 0 else 0.0)
            gross_sum[t] += g
            net_sum[t] += g - c
            cnt[t] += 1
            tot += g - c
        trade_net.append(tot)
    net = np.where(cnt > 0, net_sum / np.maximum(cnt, 1), 0.0)
    gross = np.where(cnt > 0, gross_sum / np.maximum(cnt, 1), 0.0)
    return (pd.Series(gross, index=days), pd.Series(net, index=days), pd.Series(cnt, index=days),
            pd.Series(trade_net, index=sub.index))


def sr(x):
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def conc(x):
    tot = x.sum()
    return float(x.rolling(504, min_periods=400).sum().max() / tot) if tot > 0 else float('inf')


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    px, e, days, chg = load()
    ev = events(px, e, days, chg)
    ev['main'] = ev.effective >= pd.Timestamp(MAIN_START)
    es = event_study(ev, e)
    emit(f"Events with prices through the hold: " + ", ".join(
        f"{leg} {per}: {int(((ev.leg == leg) & (ev.main == m)).sum())}" for leg in ('ADD', 'DEL') for per, m in (('2020+', True), ('2005-19', False))))
    emit("\nEvent study: mean stock-specific return of the STOCK, % (t by effective date in brackets)")
    rows = {}
    for leg in ('ADD', 'DEL'):
        for per, m in (('2020+', True), ('2005-19 context', False)):
            mask = (ev.leg == leg) & (ev.main == m)
            if mask.sum() == 0:
                continue
            x = es[mask].assign(d=ev.effective[mask].values).groupby('d').mean()
            rows[(leg, per)] = {'events': int(mask.sum()), **{k: f"{x[k].mean() * 100:+.2f} ({x[k].mean() / x[k].std() * np.sqrt(len(x)):+.1f})"
                                                           for k in WINDOWS}}
    emit(pd.DataFrame(rows).T.to_string())

    emit(f"\nTrades {MAIN_START}+: entry at the close of E-1, {HOLD} days, hedged, net of costs")
    res, verdict = {}, {}
    for leg in ('ADD', 'DEL'):
        g, nret, cnt, tn = leg_returns(ev[ev.main], e, days, leg)
        last = ev[ev.main & (ev.leg == leg)].i.max() + HOLD
        g, nret, cnt = g.loc[MAIN_START:days[last]], nret.loc[MAIN_START:days[last]], cnt.loc[MAIN_START:days[last]]
        by_date = tn.groupby(ev.loc[tn.index, 'effective']).mean()
        t = by_date.mean() / by_date.std() * np.sqrt(len(by_date))
        last24 = nret.loc[nret.index[-1] - pd.DateOffset(months=24):].sum()
        res[leg] = {'trades': len(tn), 'dates': len(by_date), 'mean net trade %': tn.mean() * 100, 'win %': (tn > 0).mean() * 100,
                    't by date': t, 'net Sharpe': sr(nret), 'gross Sharpe': sr(g), 'last 24m %': last24 * 100,
                    'conc': conc(nret), 'days invested %': (cnt > 0).mean() * 100,
                    **{str(y): tn[ev.loc[tn.index, 'effective'].dt.year == y].mean() * 100 for y in range(2020, 2027)}}
        verdict[leg] = [t >= T_MIN and by_date.mean() > 0, res[leg]['net Sharpe'] >= 0.5, last24 > 0, res[leg]['conc'] <= 0.75]
        pd.DataFrame({'gross': g, 'net': nret, 'positions': cnt}).to_csv(os.path.join(OUT, f'{leg}_daily.csv'))
    emit(pd.DataFrame(res).T.to_string(float_format=lambda v: f'{v:.2f}'))
    emit("  (year columns: mean net trade return %, by effective year)")

    emit("\nDECISIONS (pre-registered):")
    for leg in ('ADD', 'DEL'):
        r, v = res[leg], verdict[leg]
        emit(f"  {leg} {'short additions' if leg == 'ADD' else 'long deletions'}: mean trade {r['mean net trade %']:+.2f}%, "
             f"t by date {r['t by date']:.2f} {'ok' if v[0] else 'FAIL'} | Sharpe {r['net Sharpe']:.2f} {'ok' if v[1] else 'FAIL'} | "
             f"last 24m {r['last 24m %']:+.2f}% {'ok' if v[2] else 'FAIL'} | concentration {r['conc']:.0%} {'ok' if v[3] else 'FAIL'} "
             f"=> {'PASS: candidate for a paper forward test' if all(v) else 'FAIL'}")
    ev.assign(**{k: es[k] for k in WINDOWS}).to_csv(os.path.join(OUT, 'events.csv'), index=False)
    with open(os.path.join(OUT, 'report.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
