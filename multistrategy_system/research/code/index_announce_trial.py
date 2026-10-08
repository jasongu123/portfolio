"""
S&P 500 additions: from announcement to inclusion — RESEARCH ONLY. Trial T-2026-10-03-95, pre-registered
2026-10-03 before any return using announcement dates was computed. Nothing here touches the live engine.

Background: index_rebalance.py (T-86/87) found added stocks run up +4.3% (t 4.2) over E-11..E-2 before
inclusion, but could not trade it without announcement dates. index_announcements.py now supplies them
(S&P press-release dates via Wikipedia; 2020+: 135 of 138 changes; median 7 calendar days before the
effective date; a few dates are late, e.g. TSLA, which delays entry and never gives look-ahead).

Mechanism: index funds must buy every addition at the close of the day before it takes effect (E-1),
whatever the price. Traders who buy after the announcement and sell to those funds at E-1 provide that
liquidity. S&P announces after the close, so the first-morning jump cannot be captured; the question
is whether anything is left between the first close after the announcement and E-1.

Data: IBKR adjusted daily closes (backtest_data/sp500_prices_long.csv); changes and effective dates
from backtest_data/sp500_changes.csv; announcement dates from backtest_data/sp500_announcements.csv
(matched on the added ticker and an effective date within 3 days). Stock-specific return = stock minus
the equal-weight return of the members priced that day (index_rebalance.load).
Day indices: A0 = last trading day on or before the announcement date (the release follows its close);
A1 = the next trading day; E = first trading day on or after the effective date.
Trade (ADD): buy the added stock at the close of A1, hedged with the market, sell at the close of E-1
  (returns of closes A1+1 .. E-1); needs A1 <= E-2. Costs 6 bp per side (5 bp stock + 1 bp hedge).
  The leg holds its open positions in equal weight (zero on days with none).
Main window: announcements 2020-01-01 .. the last with an exit.
Pre-registered decision (one leg):
  R1 mean net trade return, averaged by announcement date (each date counts once), > 0 with t >= 2.24
  R2 calendar-time leg net Sharpe >= 0.50
  R3 leg net return over the last 24 months > 0
  R4 best 24-month block <= 75% of the leg's total profit
  PASS -> candidate for a paper forward test (no live change from this trial).
Reported, no decision: event study of the added stock (A0-9..A0 before the release; A1 the jump, not
tradable; A1+1..E-1 the trade; E-1 alone; E..E+19 after); the same window for DELETED stocks (short,
context); 2005-2019 additions (survivorship: only stocks still in the 2020-26 universe are priced);
holding days; per-year means.

Usage: python3 index_announce_trial.py     (seconds; writes results/index_announce/)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from index_rebalance import load, sr, conc, COST_SIDE, BORROW, T_MIN  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'index_announce')
MAIN_START = '2020-01-01'
CONTEXT_START = '2005-01-01'
WINDOWS = {'A0-9..A0 (before release)': ('a0', -9, 'a0', 0), 'A1 (jump, not tradable)': ('a1', 0, 'a1', 0),
           'A1+1..E-1 (traded)': ('a1', 1, 'i', -1), 'E-1 alone': ('i', -1, 'i', -1), 'E..E+19 (after)': ('i', 0, 'i', 19)}


def events(e, days, chg):
    ann = pd.read_csv(os.path.join(BD, 'sp500_announcements.csv'), parse_dates=['effective', 'announce']).dropna(subset=['announce'])
    rows = []
    for d, add, rem in zip(chg.date, chg.added, chg.removed):
        if d < pd.Timestamp(CONTEXT_START) or not isinstance(add, str):
            continue
        m = ann[(ann.added.str.replace('.', '-', regex=False) == add.replace('.', '-')) & ((ann.effective - d).abs() <= pd.Timedelta(days=3))]
        if m.empty:
            continue
        a_date = m.announce.min()
        i = int(days.searchsorted(d))
        a0 = int(days.searchsorted(a_date, side='right')) - 1
        a1 = a0 + 1
        if a0 < 10 or a1 > i - 2 or i + 20 >= len(days):
            continue
        for leg, sym in (('ADD', add), ('DEL', rem)):
            if not isinstance(sym, str):
                continue
            s = sym.replace('.', '-')
            if s not in e.columns or e[s].iloc[a1 + 1:i].notna().sum() < (i - a1 - 1):
                continue
            rows.append({'leg': leg, 'symbol': s, 'announce': days[a0], 'effective': days[i], 'a0': a0, 'a1': a1, 'i': i})
    return pd.DataFrame(rows)


def event_study(ev, e):
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    out = []
    for _, r in ev.iterrows():
        j = col[r.symbol]
        row = {}
        for k, (sa, oa, sb, ob) in WINDOWS.items():
            lo, hi = r[sa] + oa, r[sb] + ob
            row[k] = np.nansum(E[lo:hi + 1, j]) if hi < len(E) and hi >= lo else np.nan
        out.append(row)
    return pd.DataFrame(out, index=ev.index)


def leg_returns(ev, e, days, sgn):
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    n = len(days)
    net_sum, cnt, trade_net = np.zeros(n), np.zeros(n), []
    for a1, i, s in zip(ev.a1, ev.i, ev.symbol):
        j = col[s]
        start, end = a1 + 1, i - 1
        tot = 0.0
        for t in range(start, end + 1):
            g = sgn * (0.0 if np.isnan(E[t, j]) else E[t, j])
            c = (COST_SIDE if t == start else 0.0) + (COST_SIDE if t == end else 0.0) + (BORROW / 252 if sgn < 0 else 0.0)
            net_sum[t] += g - c
            cnt[t] += 1
            tot += g - c
        trade_net.append(tot)
    net = np.where(cnt > 0, net_sum / np.maximum(cnt, 1), 0.0)
    return pd.Series(net, index=days), pd.Series(cnt, index=days), pd.Series(trade_net, index=ev.index)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    px, e, days, chg = load()
    ev = events(e, days, chg)
    ev['main'] = ev.announce >= pd.Timestamp(MAIN_START)
    ev['hold_days'] = ev.i - ev.a1 - 1
    ev.to_csv(os.path.join(OUT, 'events.csv'), index=False)
    es = event_study(ev, e)
    emit("Events (announcement date known, priced through the hold): " + ", ".join(
        f"{leg} {per}: {int(((ev.leg == leg) & (ev.main == m)).sum())}" for leg in ('ADD', 'DEL') for per, m in (('2020+', True), ('2005-19', False))))
    emit(f"Holding days A1+1..E-1, additions 2020+: median {ev[ev.main & (ev.leg == 'ADD')].hold_days.median():.0f}, "
         f"range {ev[ev.main & (ev.leg == 'ADD')].hold_days.min()}..{ev[ev.main & (ev.leg == 'ADD')].hold_days.max()}")
    emit("\nEvent study: mean stock-specific return of the STOCK, % (t by announcement date in brackets)")
    rows = {}
    for leg in ('ADD', 'DEL'):
        for per, m in (('2020+', True), ('2005-19 context', False)):
            mask = (ev.leg == leg) & (ev.main == m)
            if mask.sum() < 3:
                continue
            x = es[mask].assign(d=ev.announce[mask].values).groupby('d').mean()
            rows[(leg, per)] = {'events': int(mask.sum()), **{k: f"{x[k].mean() * 100:+.2f} ({x[k].mean() / x[k].std() * np.sqrt(len(x)):+.1f})"
                                                           for k in WINDOWS}}
    emit(pd.DataFrame(rows).T.to_string())

    res, checks = {}, None
    for leg, sgn in (('ADD', 1.0), ('DEL', -1.0)):
        sub = ev[ev.main & (ev.leg == leg)]
        nret, cnt, tn = leg_returns(sub, e, days, sgn)
        last = days[sub.i.max()]
        nret, cnt = nret.loc[MAIN_START:last], cnt.loc[MAIN_START:last]
        by_date = tn.groupby(sub.announce).mean()
        t = by_date.mean() / by_date.std() * np.sqrt(len(by_date))
        last24 = nret.loc[nret.index[-1] - pd.DateOffset(months=24):].sum()
        res[leg + (' (decision)' if leg == 'ADD' else ' short (context)')] = {
            'trades': len(tn), 'dates': len(by_date), 'mean net trade %': tn.mean() * 100, 'win %': (tn > 0).mean() * 100,
            't by date': t, 'net Sharpe': sr(nret), 'last 24m %': last24 * 100, 'conc': conc(nret),
            'days invested %': (cnt > 0).mean() * 100,
            **{str(y): tn[sub.announce.dt.year == y].mean() * 100 for y in range(2020, 2027)}}
        pd.DataFrame({'net': nret, 'positions': cnt}).to_csv(os.path.join(OUT, f'{leg}_daily.csv'))
        if leg == 'ADD':
            checks = [('R1 mean trade > 0, t by date >= 2.24', by_date.mean() > 0 and t >= T_MIN, f"{by_date.mean() * 100:+.2f}%, t {t:.2f}"),
                      ('R2 net Sharpe >= 0.50', sr(nret) >= 0.5, f"{sr(nret):.2f}"),
                      ('R3 last 24 months > 0', last24 > 0, f"{last24 * 100:+.2f}%"),
                      ('R4 best 24m <= 75% of profit', conc(nret) <= 0.75, f"{conc(nret):.0%}" if np.isfinite(conc(nret)) else 'no profit')]
    emit(f"\nTrades {MAIN_START}+: buy at the close of A1, sell at the close of E-1, hedged, net of costs")
    emit(pd.DataFrame(res).T.to_string(float_format=lambda v: f'{v:.2f}'))
    emit("  (year columns: mean net trade return %, by announcement year)")
    emit("\nGate (T-2026-10-03-95), additions:")
    for name, ok, val in checks:
        emit(f"  {name:38s} {val:20s} {'ok' if ok else 'FAIL'}")
    emit(f"\nT-95 VERDICT: {'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'}")
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
