"""
Same-calendar-month seasonality in S&P 500 stocks — RESEARCH ONLY. Gate pre-registered
2026-09-15 before any return was computed.

Source: Heston & Sadka (JFE 2008) and Keloharju, Linnainmaa & Nyberg (JF 2016; JFE 2021): a
stock's average return in the same calendar month over the prior years predicts its return in
that month; the effect is cross-sectional, survives in large caps, and Coqueret & Perignon (2025)
report it persistent through 2022 across 576 method paths. Their samples end by 2022.

Signal at the last trading day of month m, for holding month m+1:
  SEAS = mean of the stock's monthly total return in calendar month (m+1) over the prior 20
         years (annual lags 1..20 from adjusted closes; at least 8 observations required).
Portfolio: long top decile / short bottom decile of SEAS within the dated S&P 500 universe
(TAKING_SIDES.md construction: change-log membership, price >= $5), equal-weight (EW) and
value-weight (VW), held one month, 10 bp per unit of one-way turnover (5 and 20 reported).
Prices: IBKR ADJUSTED_LAST, 25 years, 569 names (survivorship: 56 acquired/delisted names
absent; pre-2020 history is used only to form signals, never traded). Test window
2020-01..2026-08, entirely after the sources' samples.

Pre-registered gate (net Sharpe = no-rf, monthly*sqrt(12); EW unless stated):
  S1 alpha: EW net Sharpe >= 0.50 at 10 bp; VW >= 0.30.                               binding
  S2 recency: 2024-09..latest > 0.                                                     binding
  S3 concentration: no 24-month block > 75% of full-window profit.                     binding
  S4 orthogonality: monthly corr with equity and trend < 0.10; |beta to SPY| < 0.25;
     mean > 0 in the WF equity+trend book's worst-quintile months.                     binding
  S5 book: WF inverse-vol equity+trend+SEAS 2020-2025 not worse on rf-Sharpe, worst year, MaxDD. binding
  Informational: 5/20 bp; per year; quintile version; 10-year window version (not trials
  unless promoted, in which case they are registered before use).
Decision: S1-S5 pass -> paper-sleeve candidate. Any fail -> not.
Trials: T-2026-09-15-33 (EW decile, 20y), T-2026-09-15-34 (VW decile, 20y).
Usage: /usr/local/bin/python3 seasonality.py
"""

import os
import numpy as np
import pandas as pd

import book_backtest as bb
import trend_follow_v2 as t2
import taking_sides as ts
from short_interest_splice import shares_asof

RESULTS_DIR = ts.RESULTS_DIR
PX_LONG = os.path.join(ts.BD, 'sp500_prices_long.csv')
YEARS, MIN_OBS = 20, 8
START, END = '2020-01-01', '2026-08-31'


def build_signals(px, uni, chg, sec, sp, years=YEARS):
    pm = px.resample('ME').last()
    rm = pm.pct_change(fill_method=None)
    rm.index = rm.index.to_period('M')
    mem = ts.membership(chg, uni)
    shares = shares_asof(sec)
    sh_by = {k: v.set_index('filed')['val'].sort_index() for k, v in shares[shares.symbol.isin(uni)].groupby('symbol')}
    rebs = pd.DatetimeIndex(px.groupby(px.index.to_period('M')).apply(lambda g: g.index[-1]).values)
    rebs = rebs[(rebs >= pd.Timestamp(START) - pd.Timedelta(days=40)) & (rebs <= pd.Timestamp(END))]
    rows = []
    for d in rebs:
        hold = pd.Period(d, 'M') + 1
        lags = [hold - 12 * k for k in range(1, years + 1)]
        lags = [p for p in lags if p in rm.index]
        block = rm.loc[lags]
        for s in uni:
            if s not in px.columns or not ts.in_index(mem, s, d):
                continue
            p = px[s].loc[:d].dropna()
            if len(p) < ts.MIN_HIST or p.index[-1] != d or p.iloc[-1] < ts.MIN_PRICE:
                continue
            x = block[s].dropna()
            if len(x) < MIN_OBS:
                continue
            mcap = np.nan
            sh = sh_by.get(s)
            if sh is not None and len(sh.loc[:d]):
                mcap = p.iloc[-1] * sh.loc[:d].iloc[-1] * ts.split_factor_to_now(sp, s, sh.loc[:d].index[-1])
            rows.append({'date': d, 'symbol': s, 'SEAS': x.mean(), 'nobs': len(x), 'mcap': mcap})
    return pd.DataFrame(rows)


def portfolio_decile(sig, px, col, weighting, cost_bps, frac=0.10):
    rets = px.pct_change(fill_method=None)
    dates = sorted(sig.date.unique())
    w_prev = pd.Series(dtype=float)
    daily, turn = [], {}
    for i, d in enumerate(dates):
        g = sig[(sig.date == d)].dropna(subset=[col])
        if len(g) < 50:
            continue
        q = g[col].rank(pct=True)
        longs, shorts = g[q > 1 - frac], g[q <= frac]
        if weighting == 'VW':
            longs, shorts = longs.dropna(subset=['mcap']), shorts.dropna(subset=['mcap'])
            wl = longs.set_index('symbol').mcap / longs.mcap.sum()
            ws = -shorts.set_index('symbol').mcap / shorts.mcap.sum()
        else:
            wl = pd.Series(1.0 / len(longs), index=longs.symbol)
            ws = pd.Series(-1.0 / len(shorts), index=shorts.symbol)
        w = pd.concat([wl, ws])
        turn[d] = (w.subtract(w_prev, fill_value=0.0)).abs().sum()
        w_prev = w
        nxt = dates[i + 1] if i + 1 < len(dates) else px.index[-1]
        r = rets.loc[d + pd.Timedelta(days=1):nxt, w.index].fillna(0.0)
        cum = (1 + r).cumprod()
        val = (cum * w).sum(axis=1)
        pv = pd.concat([pd.Series([w.sum()], index=[d]), val])
        dr = pv.diff().iloc[1:]
        dr.iloc[0] -= turn[d] * cost_bps / 1e4
        daily.append(dr)
    out = pd.concat(daily).sort_index()
    return out[~out.index.duplicated()], pd.Series(turn)


sr, monthly = ts.sr, ts.monthly


def main():
    px = pd.read_csv(PX_LONG, index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(ts.UNI_FILE).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(ts.CHG_FILE, parse_dates=['date'])
    sec = pd.read_csv(ts.SEC_FILE)
    sp = pd.read_csv(os.path.join(ts.BD, 'sp500_splits_detected.csv'), parse_dates=['date', 'window_start'])
    print(f"Prices: {px.shape[1]} names, {px.index.min().date()}..{px.index.max().date()}; "
          f"names with >= 20y history: {(px.notna().idxmax() <= pd.Timestamp('2006-01-01')).sum()}")
    sig = build_signals(px, uni, chg, sec, sp)
    sig10 = build_signals(px, uni, chg, sec, sp, years=10)
    cov = sig.groupby('date').agg(n=('symbol', 'size'), nobs=('nobs', 'median'))
    print(f"Rebalances {len(cov)}: names/month median {cov.n.median():.0f}, median observations per signal {cov.nobs.median():.0f}")
    sig.to_csv(os.path.join(RESULTS_DIR, 'seasonality_signals.csv'), index=False)

    spy = t2.load_prices()['SPY'].pct_change(fill_method=None)
    series = {}
    print(f"\n{'portfolio':26s}{'net SR':>8s}{'5bp':>7s}{'20bp':>7s}{'ann':>8s}{'vol':>7s}{'MDD':>8s}{'beta':>7s}{'last2y':>8s}{'TO/mo':>7s}")
    for label, s_, frac in (('SEAS20 decile', sig, 0.10), ('SEAS20 quintile', sig, 0.20), ('SEAS10 decile', sig10, 0.10)):
        for wt in ('EW', 'VW'):
            r, to = portfolio_decile(s_, px, 'SEAS', wt, 10.0, frac)
            r = r.loc[START:END]
            r5 = portfolio_decile(s_, px, 'SEAS', wt, 5.0, frac)[0].loc[START:END]
            r20 = portfolio_decile(s_, px, 'SEAS', wt, 20.0, frac)[0].loc[START:END]
            m = monthly(r)
            j = pd.concat([m.rename('r'), monthly(spy.reindex(r.index)).rename('s')], axis=1).dropna()
            beta = j['r'].cov(j['s']) / j['s'].var()
            series[f'{label} {wt}'] = r
            print(f"{label + ' ' + wt:26s}{sr(m):8.2f}{sr(monthly(r5)):7.2f}{sr(monthly(r20)):7.2f}{r.mean() * 252:8.1%}{r.std() * np.sqrt(252):7.1%}"
                  f"{bb.maxdd(r):8.1%}{beta:7.2f}{sr(monthly(r.loc['2024-09-01':])):8.2f}{to.mean():7.2f}")
    prim = series['SEAS20 decile EW']
    m = monthly(prim)
    print("Per-year net Sharpe, SEAS20 decile EW: " + "  ".join(f"{y}:{sr(x):+.2f}" for y, x in m.groupby(m.index.year)))
    print("Per-calendar-month mean net return (bp), SEAS20 decile EW: " +
          "  ".join(f"{mo}:{v * 1e4:+.0f}" for mo, v in m.groupby(m.index.month).mean().items()))

    s1 = (sr(m) >= 0.50) and (sr(monthly(series['SEAS20 decile VW'])) >= 0.30)
    s2 = sr(m.loc['2024-09-01':]) > 0
    tot = (1 + m).prod() - 1
    blocks = {str(a.date()): (1 + m.loc[a:a + pd.DateOffset(months=23)]).prod() - 1 for a in m.index[:-23]}
    conc = max(blocks.values()) / tot if tot > 0 else np.inf
    s3 = conc <= 0.75
    eq, tr = bb.load_equity(), bb.load_trend()
    full = pd.concat([eq, tr, prim.rename('seas')], axis=1)
    bb.FULL_FRAME = full
    j = full.loc['2020-01-01':'2025-12-31'].dropna()
    jm = j.resample('ME').sum()
    c_eq, c_tr = jm['seas'].corr(jm['equity']), jm['seas'].corr(jm['trend'])
    js = pd.concat([m.rename('r'), monthly(spy.reindex(prim.index)).rename('s')], axis=1).dropna()
    beta = js['r'].cov(js['s']) / js['s'].var()
    book2 = bb.book_returns(j[['equity', 'trend']], 'WF')
    b2m = monthly(book2)
    bad = jm.loc[b2m[b2m <= b2m.quantile(0.2)].index, 'seas']
    s4 = (c_eq < 0.10) and (c_tr < 0.10) and (abs(beta) < 0.25) and (bad.mean() > 0)
    book3 = bb.book_returns(j[['equity', 'trend', 'seas']], 'WF')
    y2, y3 = bb.yearly(book2), bb.yearly(book3)
    s5 = (bb.sharpe(book3) >= bb.sharpe(book2)) and (min(y3.values()) >= min(y2.values())) and (bb.maxdd(book3) >= bb.maxdd(book2))

    print("\nGO/NO-GO (pre-registered, SEAS20 decile):")
    print(f"  {'PASS' if s1 else 'fail'}  S1 net Sharpe EW {sr(m):.2f} (>= 0.50); VW {sr(monthly(series['SEAS20 decile VW'])):.2f} (>= 0.30)")
    print(f"  {'PASS' if s2 else 'fail'}  S2 last two years {sr(m.loc['2024-09-01':]):+.2f} (> 0)")
    print(f"  {'PASS' if s3 else 'fail'}  S3 best 24-month block = {conc:.0%} of profit (<= 75%)")
    print(f"  {'PASS' if s4 else 'fail'}  S4 corr equity {c_eq:+.2f}, trend {c_tr:+.2f} (< 0.10); beta {beta:+.2f}; bad-month mean {bad.mean() * 100:+.2f}% (n={len(bad)}, hit {(bad > 0).mean():.0%})")
    print(f"  {'PASS' if s5 else 'fail'}  S5 WF book rf {bb.sharpe(book2):.2f} -> {bb.sharpe(book3):.2f}; worst {min(y2.values()):+.2f} -> {min(y3.values()):+.2f}; MDD {bb.maxdd(book2):.1%} -> {bb.maxdd(book3):.1%}")
    print(f"  info  book no-rf {bb.sharpe(book2, 0):.2f} -> {bb.sharpe(book3, 0):.2f}")
    verdict = all([s1, s2, s3, s4, s5])
    print(f"\nVERDICT: {'PROMOTE to paper-sleeve candidate' if verdict else 'not promoted'}")
    pd.DataFrame(series).to_csv(os.path.join(RESULTS_DIR, 'seasonality_daily.csv'))
    print("Saved results/seasonality_signals.csv, seasonality_daily.csv")


if __name__ == '__main__':
    main()
