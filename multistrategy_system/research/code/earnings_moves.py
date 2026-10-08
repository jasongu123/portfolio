"""
Big stock moves on earnings days versus other days — RESEARCH ONLY. Pre-registered 2026-10-02 before
any return was computed. Nothing here touches the live engine.

Hypothesis (Chan 2003 and the post-earnings-drift literature): a large stock-specific move that
comes with news tends to continue (the market underreacts to information); a large move without
news tends to partly reverse (temporary price pressure from someone trading in a hurry). Earnings
announcements are the most important, best-dated company news, so they stand in for "news".
Non-earnings days can still carry news, which biases the test against finding a difference.

Data
  prices    IBKR adjusted daily closes, backtest_data/sp500_prices_long.csv; point-in-time S&P 500
            membership from the change log (taking_sides.membership); price >= $5, >= 260 days of
            history. 2020-2026 is the main window; 2005-2019 is reported as context only (the
            universe there is the names that later were members: survivorship).
  earnings  SEC EDGAR filings index: every 8-K whose items include 2.02 (Results of Operations),
            with its acceptance time in UTC (backtest_data/sec_8k_202.csv, fetch_8k_dates(); item-2.02
            tagging starts Aug 2004, so context starts 2005). Reaction day:
            the trading day of acceptance if accepted before 16:00 ET, else the next trading day.
Events at the close of day t, per member stock:
  e  = stock return - equal-weight member return (the stock-specific move)
  z  = e / standard deviation of e over the previous 63 trading days (>= 50 observations)
  an EVENT is |z| >= 3; it is an EARNINGS event if a reaction day falls on t-1, t or t+1 (all
  known by the entry, the close of t+1), otherwise a NON-EARNINGS event.
Trades (both entered at the close of t+1: nothing trades on the signal close)
  NR  non-earnings reversal: position -sign(e), held 5 trading days
  ED  earnings drift:         position +sign(e), held 20 trading days
  Each position earns its stock-specific return (stock minus the equal-weight member return, i.e.
  hedged with the market). Each leg holds its open positions in equal weight (zero on days with
  none). Costs 6 bp at entry and 6 bp at exit per position (5 bp stock + 1 bp hedge); short
  positions pay 0.5% a year borrow.
Pre-registered decision, per leg, on 2020-01-01 .. the last complete holding period (two legs,
so the t threshold is Bonferroni-adjusted):
  P1 net Sharpe >= 0.50
  P2 net Sharpe over the last 24 months > 0
  P3 best 24-month block <= 75% of total profit
  P4 mechanism: the leg's mean daily return BEFORE costs has t >= 2.24 in the hypothesised direction
  A leg passing all four is a candidate sleeve for a paper forward test (no live change from this
  trial). Reported, no decision: the event study (mean signed stock-specific return over t+1,
  t+2..t+6 and t+2..t+21 for both groups; positive = continuation, negative = reversal), the
  2005-2019 context, and the correlation of each leg with the trend sleeve, swing-only stocks and
  the ETF paper rules.
Trials: T-2026-10-02-84 (NR), -85 (ED).

Usage:
  python3 earnings_moves.py --fetch     download SEC 8-K item 2.02 dates (a few minutes, cached)
  python3 earnings_moves.py             run the test (writes results/earnings_moves/)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from short_interest_splice import SEC_UA, SEC_TICKERS, SEC_SLEEP  # noqa: E402
from taking_sides import membership  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'earnings_moves')
EARN_FILE = os.path.join(BD, 'sec_8k_202.csv')
SUBMISSIONS = 'https://data.sec.gov/submissions/CIK{cik:010d}.json'
SUB_FILE = 'https://data.sec.gov/submissions/{name}'
MAIN = ('2020-01-01', '2026-12-31')
CONTEXT = ('2005-01-01', '2019-12-31')  # SEC item-2.02 tagging starts in Aug 2004
Z_EVENT = 3.0
HOLD = {'NR': 5, 'ED': 20}
COST_SIDE = 0.0006
BORROW = 0.005
T_MIN = 2.24
ANN = np.sqrt(252)


def _get(url, retries=4):
    for i in range(retries):
        try:
            r = requests.get(url, headers=SEC_UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(2 * (i + 1))
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


def _rows(block, sym, cik):
    out = []
    for form, fdate, acc, items in zip(block.get('form', []), block.get('filingDate', []),
                                       block.get('acceptanceDateTime', []), block.get('items', [])):
        if form == '8-K' and '2.02' in str(items).split(','):
            out.append({'symbol': sym, 'cik': cik, 'filing_date': fdate, 'acceptance': acc})
    return out


def fetch_8k_dates():
    uni = pd.read_csv(os.path.join(BD, 'sp500_universe_2020_2026.csv')).iloc[:, 0].astype(str).tolist()
    tick = _get(SEC_TICKERS)
    cik_of = {v['ticker'].upper().replace('.', '-'): int(v['cik_str']) for v in tick.values()}
    done = set(pd.read_csv(EARN_FILE).symbol) if os.path.exists(EARN_FILE) else set()
    rows, missing = [], []
    for k, sym in enumerate(uni):
        if sym in done:
            continue
        cik = cik_of.get(sym.upper().replace('.', '-'))
        if cik is None:
            missing.append(sym)
            continue
        sub = _get(SUBMISSIONS.format(cik=cik))
        time.sleep(SEC_SLEEP)
        if sub is None:
            missing.append(sym)
            continue
        rows += _rows(sub['filings']['recent'], sym, cik)
        for f in sub['filings'].get('files', []):
            more = _get(SUB_FILE.format(name=f['name']))
            time.sleep(SEC_SLEEP)
            if more:
                rows += _rows(more, sym, cik)
        if k % 50 == 0:
            print(f"  {k}/{len(uni)} {sym}: {len(rows)} filings so far", flush=True)
    df = pd.DataFrame(rows)
    if os.path.exists(EARN_FILE):
        df = pd.concat([pd.read_csv(EARN_FILE), df], ignore_index=True)
    df.drop_duplicates().to_csv(EARN_FILE, index=False)
    print(f"8-K item 2.02 filings: {len(df)} for {df.symbol.nunique()} symbols; no CIK or no data for {len(missing)}: {missing[:30]}")


def reaction_days(earn, days):
    """Trading day on which each filing's price reaction falls (acceptance before 16:00 ET: same
    trading day; otherwise the next). acceptanceDateTime is UTC (checked before any run: after-close
    filings cluster at 20:00-21:59 UTC), converted to New York time here."""
    acc = (pd.to_datetime(earn['acceptance'], utc=True, errors='coerce')
           .dt.tz_convert('America/New_York').dt.tz_localize(None))
    after = acc.dt.hour >= 16
    d = acc.dt.normalize() + pd.to_timedelta(after.astype(int), unit='D')
    idx = days.searchsorted(d)
    ok = idx < len(days)
    out = pd.Series(pd.NaT, index=earn.index)
    out[ok] = days[idx[ok]]
    return out


def build_events():
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
    ok = member & (px >= 5) & (px.notna().cumsum() >= 260)
    mkt = r.where(ok).mean(axis=1)
    e = r.sub(mkt, axis=0)
    sd = e.rolling(63, min_periods=50).std().shift(1)
    z = (e / sd).where(ok)
    earn = pd.read_csv(EARN_FILE)
    earn['react'] = reaction_days(earn, days)
    react = earn.dropna(subset=['react'])
    pos = {d: i for i, d in enumerate(days)}
    near = set()
    for sym, d in zip(react.symbol, react.react):
        i = pos[pd.Timestamp(d)]
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(days):
                near.add((sym, days[j]))
    covered = set(earn.symbol)
    ev = z.stack()
    ev = ev[ev.abs() >= Z_EVENT].rename('z').reset_index()
    ev.columns = ['date', 'symbol', 'z']
    ev = ev[ev.symbol.isin(covered)]
    ev['e'] = [e.at[d, s] for d, s in zip(ev.date, ev.symbol)]
    ev['earnings'] = [(s, d) in near for s, d in zip(ev.symbol, ev.date)]
    return ev.reset_index(drop=True), e, days, covered


def event_study(ev, e, days):
    pos = {d: i for i, d in enumerate(days)}
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    out = []
    for d, s, sign in zip(ev.date, ev.symbol, np.sign(ev.e)):
        i, j = pos[d], col[s]
        seg = lambda a, b: np.nansum(E[i + a:i + b + 1, j]) * sign if i + b < len(days) else np.nan
        out.append((seg(1, 1), seg(2, 6), seg(2, 21)))
    return pd.DataFrame(out, columns=['t+1', 't+2..6', 't+2..21'], index=ev.index)


def leg_returns(ev, e, days, leg):
    """Calendar-time leg: equal weight across open positions; position return = sign * e, net of costs."""
    k = HOLD[leg]
    sub = ev[~ev.earnings] if leg == 'NR' else ev[ev.earnings]
    direction = -1 if leg == 'NR' else 1
    pos = {d: i for i, d in enumerate(days)}
    E = e.to_numpy()
    col = {s: j for j, s in enumerate(e.columns)}
    n = len(days)
    gross_sum, net_sum, count = np.zeros(n), np.zeros(n), np.zeros(n)
    for d, s, ee in zip(sub.date, sub.symbol, sub.e):
        i, j = pos[d], col[s]
        sgn = direction * np.sign(ee)
        start, end = i + 2, i + 1 + k                     # returns earned after the close of t+1
        if end >= n:
            continue
        for t in range(start, end + 1):
            g = sgn * (0.0 if np.isnan(E[t, j]) else E[t, j])
            c = (COST_SIDE if t == start else 0.0) + (COST_SIDE if t == end else 0.0) + (BORROW / 252 if sgn < 0 else 0.0)
            gross_sum[t] += g
            net_sum[t] += g - c
            count[t] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        gross = np.where(count > 0, gross_sum / count, 0.0)
        net = np.where(count > 0, net_sum / count, 0.0)
    return pd.Series(gross, index=days), pd.Series(net, index=days), pd.Series(count, index=days)


def sr(x):
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def conc(x):
    tot = x.sum()
    return float(x.rolling(504, min_periods=400).sum().max() / tot) if tot > 0 else float('inf')


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    ev, e, days, covered = build_events()
    emit(f"Earnings dates for {len(covered)} symbols; events |z| >= {Z_EVENT}: {len(ev)} "
         f"({ev.earnings.mean():.0%} on earnings days); {time.time() - t0:.0f}s")
    es = event_study(ev, e, days)
    emit("\nEvent study: mean signed stock-specific return, % (positive = continuation, negative = reversal)")
    rows = {}
    for (lo, hi), per in ((MAIN, '2020-26'), (CONTEXT, '2005-19 (context)')):
        m = (ev.date >= lo) & (ev.date <= hi)
        for grp, mask in (('earnings', ev.earnings), ('non-earnings', ~ev.earnings)):
            x = es[m & mask]
            # t by date: average the events of each day first, so a crash day with 200 events
            # counts once (event-level t overstates significance when events cluster)
            dm = x.assign(date=ev.date[m & mask].values).groupby('date')[list(es.columns)].mean()
            rows[(per, grp)] = {'events': len(x), 'dates': len(dm), **{c: x[c].mean() * 100 for c in es.columns},
                                **{f't {c}': x[c].mean() / x[c].std() * np.sqrt(x[c].count()) for c in es.columns},
                                **{f't by date {c}': dm[c].mean() / dm[c].std() * np.sqrt(dm[c].count()) for c in es.columns}}
    emit(pd.DataFrame(rows).T.to_string(float_format=lambda v: f'{v:.2f}'))

    last_day = days[-1]
    legs = {}
    for leg in ('NR', 'ED'):
        g, nret, cnt = leg_returns(ev, e, days, leg)
        end = days[-1 - HOLD[leg] - 1]                    # last day every position could complete
        g, nret, cnt = g.loc[MAIN[0]:end], nret.loc[MAIN[0]:end], cnt.loc[MAIN[0]:end]
        legs[leg] = (g, nret, cnt)
    emit(f"\nLegs, {MAIN[0]} .. holding periods complete by {last_day.date()} (net of costs unless stated):")
    res = {}
    for leg, (g, nret, cnt) in legs.items():
        last2 = nret.loc[nret.index[-1] - pd.DateOffset(months=24):]
        t = g.mean() / g.std() * np.sqrt(len(g))
        sign_ok = t >= T_MIN
        res[leg] = {'net Sharpe': sr(nret), 'gross Sharpe': sr(g), 'ret %/yr': nret.mean() * 252 * 100,
                    'last 24m': sr(last2), 'conc': conc(nret), 't gross': t, 'days invested %': (cnt > 0).mean() * 100,
                    'avg positions': cnt[cnt > 0].mean(), **{str(y): sr(nret.loc[str(y)]) for y in range(2020, 2027)}}
    df = pd.DataFrame(res).T
    emit(df.to_string(float_format=lambda v: f'{v:.2f}'))
    for leg, (g, nret, cnt) in legs.items():
        pd.DataFrame({'gross': g, 'net': nret, 'positions': cnt}).to_csv(os.path.join(OUT, f'{leg}_daily.csv'))

    emit("\nCorrelation of net daily returns with existing sleeves (overlap):")
    others = {}
    try:
        tr = pd.read_csv(os.path.join(HERE, 'results', 'headline_2026-09_daily_audit.csv'), index_col=0, parse_dates=True)
        others['trend'] = tr['trend_live']
        import split_trial
        others['swing S1'] = split_trial.load('S1')[0]
        others['ETF E0'] = split_trial.load('E0')[0]
    except Exception as exc:
        emit(f"  (some sleeves unavailable: {exc})")
    for leg, (g, nret, cnt) in legs.items():
        emit(f"  {leg}: " + ", ".join(f"{k} {nret.corr(v.reindex(nret.index)):+.2f}" for k, v in others.items()))

    emit("\nDECISIONS (pre-registered):")
    for leg in ('NR', 'ED'):
        r = df.loc[leg]
        t = [r['net Sharpe'] >= 0.50, r['last 24m'] > 0, r['conc'] <= 0.75, r['t gross'] >= T_MIN]
        name = 'non-earnings reversal' if leg == 'NR' else 'earnings drift'
        emit(f"  {leg} {name}: Sharpe {r['net Sharpe']:.2f} {'ok' if t[0] else 'FAIL'} | last 24m {r['last 24m']:+.2f} "
             f"{'ok' if t[1] else 'FAIL'} | concentration {r['conc']:.0%} {'ok' if t[2] else 'FAIL'} | gross t {r['t gross']:.2f} "
             f"{'ok' if t[3] else 'FAIL'} => {'PASS: candidate for a paper forward test' if all(t) else 'FAIL'}")
    ev.to_csv(os.path.join(OUT, 'events.csv'), index=False)
    with open(os.path.join(OUT, 'report.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    emit(f"({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    a = ap.parse_args()
    if a.fetch:
        fetch_8k_dates()
    else:
        main()
