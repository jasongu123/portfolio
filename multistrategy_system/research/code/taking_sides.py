"""
Taking Sides, free subset — RESEARCH ONLY. Gate pre-registered 2026-09-15 before any return
was computed.

Source: McLean, Pontiff & Reilly, "Taking Sides on Return Predictability", JFE 173 (2025),
sample 2006-10..2017-12. Finding used: firms (through net share issuance and repurchases) and
short sellers trade on the right side of return predictability; retail trades against it. The
two free pieces are firm trading (change in split-adjusted shares outstanding) and short-seller
trading (change in short interest). Everything after 2018 is untouched by the paper; our sample,
2020-05..2026-08, is fixed by the FINRA cache and is entirely out of sample.

Universe: S&P 500 members 2020-2026 reconstructed from the Wikipedia change log (current members
plus everyone added or removed since 2020-01; membership by add/remove date; 625 names). Prices:
Yahoo adjusted closes; splits from Yahoo. Survivorship: names Yahoo cannot serve are counted and
reported; the log itself is incomplete (it reconstructs 526 members at end-2019 against ~505).
Shares: SEC XBRL dei:EntityCommonStockSharesOutstanding, point in time by filing date, max 'end'
per filing (short_interest_splice.shares_asof), split-adjusted to the current basis. Splits:
Yahoo's split list if present (backtest_data/sp500_splits.csv); otherwise detected from the data:
candidates are jumps between consecutive SEC filings within 1.5% of 1.5 or within 4% of a
standard ratio >= 2 (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, or the inverse); from
2020-06 a candidate is pinned to the FINRA settlement date on which short interest jumps by the
ratio (within 30%; average daily volume within a factor 0.4-2.5 of the ratio, since a settlement
period straddling the split blends pre- and post-split volume) and, if no such settlement
exists near its window, is vetoed for ratios 1.5, 2 and 3 (a filing artefact or a stock merger)
but kept unpinned for ratios of 4 or more (never issuance; CTAS 4:1 and NFLX 10:1 were wrongly
vetoed by the stricter rule); before 2020-06 candidates are kept as
SEC-only, dated the day before the later filing. The short-interest signal is blanked for any rebalance whose two
settlement dates straddle a detection window. (Corrected 2026-09-15: SEC-only detection missed
TSLA's 2020 5:1, which coincided with a capital raise, and invented a WFC 2:1 from a filing
glitch; the first run's numbers are kept in the registry as the pre-correction row.) Prices: IBKR ADJUSTED_LAST daily closes (Yahoo rate-limited).
Short interest: FINRA consolidated, semi-monthly, treated as available 12 calendar days after
settlement.

Signals at each month-end rebalance date d (long = high signal):
  ISS  = -( log adj_shares(d) - log adj_shares(d - 12 months) )        firm trading, annual
  SHT  = -( log adj_SI(latest available) - log adj_SI(one month earlier) )   short-seller trading
  COMP = mean of the cross-sectional z-scores (winsorised 1/99) of ISS and SHT, both required.
Portfolio: long top quintile, short bottom quintile of COMP, equal-weight within leg (EW) and
value-weight (VW, price x adj shares); hold one month; cost 10 bp per unit of one-way turnover
(5 and 20 bp reported). Price >= $5 and >= 60 days of history required.

Pre-registered gate (composite, EW unless stated; net Sharpe = no-rf, monthly*sqrt(12)):
  T1 alpha: net Sharpe >= 0.50 over the full sample at 10 bp; VW net Sharpe >= 0.30.   binding
  T2 recency: net Sharpe over 2024-09..latest > 0.                                     binding
  T3 concentration: no 24-month block supplies > 75% of the full-sample profit.        binding
  T4 orthogonality: monthly corr with the equity sleeve and with trend each < 0.10;
     |beta to SPY| < 0.25; mean return > 0 in the WF equity+trend book's worst-quintile
     months (2020-2025).                                                               binding
  T5 book: walk-forward inverse-vol equity+trend+TS over 2020-2025 not worse than
     equity+trend on rf-Sharpe, worst year and MaxDD.                                  binding
  Informational: ISS alone and SHT alone on every line (trials T-31, T-32); 5/20 bp; legs
     long/short separately vs SPY; survivorship count.
Decision: T1-T5 pass -> paper-sleeve candidate (its own live design later). Any fail -> not.
Trials: T-2026-09-15-30 (COMP), -31 (ISS), -32 (SHT).
Usage: /usr/local/bin/python3 taking_sides.py
"""

import os
import numpy as np
import pandas as pd

import book_backtest as bb
import trend_follow_v2 as t2
from short_interest_splice import shares_asof

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
BD = os.path.join(SCRIPT_DIR, 'backtest_data')
PX_FILE, SPLIT_FILE = os.path.join(BD, 'sp500_prices_2020_2026.csv'), os.path.join(BD, 'sp500_splits.csv')
UNI_FILE, CHG_FILE = os.path.join(BD, 'sp500_universe_2020_2026.csv'), os.path.join(BD, 'sp500_changes.csv')
SEC_FILE, FINRA_FILE = os.path.join(RESULTS_DIR, 'sec_shares_outstanding.csv.gz'), os.path.join(RESULTS_DIR, 'finra_short_interest_raw.csv.gz')
COST_BPS = 10.0
FINRA_LAG_DAYS = 12
START, END = '2020-05-01', '2026-08-31'
MIN_PRICE, MIN_HIST = 5.0, 60
ANN12 = np.sqrt(12)


STD_RATIOS = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50]


def detect_splits_from_sec(shares, tol=0.015, tol_big=0.04):
    """Split candidates from consecutive point-in-time share counts: ratio within `tol` of 1.5
    (a 3:2 split is hard to tell from a stock-financed merger) or within `tol_big` of a standard
    ratio >= 2 (issuance or buybacks in the same quarter shift the ratio; TSLA 2020 was 5.10)."""
    rows = []
    for sym, g in shares.groupby('symbol'):
        g = g.sort_values('filed')
        v, f = g['val'].values, g['filed'].values
        for i in range(1, len(v)):
            if v[i - 1] <= 0 or v[i] <= 0:
                continue
            r = v[i] / v[i - 1]
            for k in STD_RATIOS:
                for cand in (k, 1.0 / k):
                    if abs(r / cand - 1) <= (tol if k == 1.5 else tol_big):
                        rows.append({'symbol': sym, 'date': pd.Timestamp(f[i]) - pd.Timedelta(days=1),
                                     'window_start': pd.Timestamp(f[i - 1]), 'ratio': float(cand), 'source': 'sec'})
    return pd.DataFrame(rows, columns=['symbol', 'date', 'window_start', 'ratio', 'source'])


def pin_or_veto_with_finra(cands, fin, uni, si_tol=0.30, adv_lo=0.4, adv_hi=2.5):
    """SEC-detected candidates from 2020-06 onward are pinned to the FINRA settlement date on
    which short interest jumps by the ratio (within si_tol; a settlement period straddling the
    split blends pre- and post-split volume, so ADV only has to land within [adv_lo, adv_hi] x
    ratio). A candidate with no such settlement near its window is a filing artefact and is
    vetoed. Earlier candidates are kept as SEC-only (FINRA starts 2020-04)."""
    f = fin[fin.symbolCode.isin(uni)].copy()
    f['si'] = pd.to_numeric(f.currentShortPositionQuantity, errors='coerce')
    f['adv'] = pd.to_numeric(f.averageDailyVolumeQuantity, errors='coerce')
    by = {k: v.sort_values('settlementDate').reset_index(drop=True) for k, v in f.groupby('symbolCode')}
    rows = []
    for c in cands.itertuples():
        if c.date < pd.Timestamp('2020-06-01'):
            rows.append({'symbol': c.symbol, 'date': c.date, 'window_start': c.window_start, 'ratio': c.ratio, 'source': 'sec'})
            continue
        g = by.get(c.symbol)
        pinned = False
        if g is not None:
            lo, hi = c.window_start - pd.Timedelta(days=5), c.date + pd.Timedelta(days=20)
            for i in range(1, len(g)):
                t = g.settlementDate[i]
                if t < lo or t > hi or min(g.si[i - 1], g.si[i], g.adv[i - 1], g.adv[i]) <= 0:
                    continue
                r_si, r_adv = g.si[i] / g.si[i - 1], g.adv[i] / g.adv[i - 1]
                if abs(r_si / c.ratio - 1) <= si_tol and adv_lo <= r_adv / c.ratio <= adv_hi:
                    rows.append({'symbol': c.symbol, 'date': t, 'window_start': g.settlementDate[i - 1], 'ratio': c.ratio, 'source': 'finra-pinned'})
                    pinned = True
                    break
        if not pinned:
            big = c.ratio >= 4 or c.ratio <= 0.25          # an integer ratio this large between quarterly filings is never issuance
            rows.append({'symbol': c.symbol, 'date': c.date, 'window_start': c.window_start, 'ratio': c.ratio,
                         'source': 'sec-unpinned' if big else 'vetoed'})
    out = pd.DataFrame(rows, columns=['symbol', 'date', 'window_start', 'ratio', 'source'])
    return out[out.source != 'vetoed'], out[out.source == 'vetoed']


def load():
    px = pd.read_csv(PX_FILE, index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(UNI_FILE).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(CHG_FILE, parse_dates=['date'])
    sec = pd.read_csv(SEC_FILE)
    fin = pd.read_csv(FINRA_FILE, parse_dates=['settlementDate'])
    if os.path.exists(SPLIT_FILE):
        sp = pd.read_csv(SPLIT_FILE, parse_dates=['date'])
        sp['window_start'] = sp['date']
        print(f"Splits: Yahoo list, {len(sp)} events")
    else:
        sh = shares_asof(sec)
        sh = sh[sh.symbol.isin(uni)]
        cands = detect_splits_from_sec(sh)
        sp, vetoed = pin_or_veto_with_finra(cands, fin, uni)
        sp = sp.sort_values(['symbol', 'date']).reset_index(drop=True)
        sp.to_csv(os.path.join(BD, 'sp500_splits_detected.csv'), index=False)
        vetoed.to_csv(os.path.join(BD, 'sp500_splits_vetoed.csv'), index=False)
        pinned = sp[sp.source == 'finra-pinned']
        print(f"Splits: {len(pinned)} FINRA-pinned from 2020-06, {(sp.source == 'sec').sum()} SEC-only before, {len(vetoed)} vetoed "
              f"({', '.join(f'{r.symbol} {r.ratio:g}' for r in vetoed.head(6).itertuples())}); pinned e.g. " +
              "; ".join(f"{r.symbol} {r.ratio:g}:1 on {r.date.date()}" for r in pinned.sort_values('date').head(8).itertuples()))
    return px, sp, uni, chg, sec, fin


def split_factor_to_now(sp, symbol, date):
    """Product of split ratios after `date`: multiply an old share count by this to reach today's basis."""
    s = sp[(sp.symbol == symbol) & (sp.date > date)]
    return float(np.prod(s.ratio.values)) if len(s) else 1.0


def membership(chg, uni):
    """(added_date, removed_date) per symbol from the change log; NaT = unknown (treated as always)."""
    norm = lambda x: str(x).replace('.', '-')
    added = chg.dropna(subset=['added']).assign(s=lambda d: d.added.map(norm)).groupby('s').date.max()
    removed = chg.dropna(subset=['removed']).assign(s=lambda d: d.removed.map(norm)).groupby('s').date.max()
    return {s: (added.get(s, pd.NaT), removed.get(s, pd.NaT)) for s in uni}


def in_index(mem, s, d):
    a, r = mem.get(s, (pd.NaT, pd.NaT))
    if pd.notna(a) and d < a:
        return False
    if pd.notna(r) and d >= r and (pd.isna(a) or r > a):
        return False
    return True


def build_signals(px, sp, uni, chg, sec, fin):
    shares = shares_asof(sec)
    shares = shares[shares.symbol.isin(uni)]
    sh_by = {k: v.set_index('filed')['val'].sort_index() for k, v in shares.groupby('symbol')}
    fin = fin[fin.symbolCode.isin(uni)].copy()
    fin['si'] = pd.to_numeric(fin.currentShortPositionQuantity, errors='coerce')
    fin['avail'] = fin.settlementDate + pd.Timedelta(days=FINRA_LAG_DAYS)
    si_by = {k: v.sort_values('settlementDate') for k, v in fin.groupby('symbolCode')}
    mem = membership(chg, uni)
    rebs = pd.DatetimeIndex(px.groupby(px.index.to_period('M')).apply(lambda g: g.index[-1]).values)
    rebs = rebs[(rebs >= pd.Timestamp(START) - pd.Timedelta(days=40)) & (rebs <= pd.Timestamp(END))]
    rows = []
    for d in rebs:
        for s in uni:
            if s not in px.columns or not in_index(mem, s, d):
                continue
            p = px[s].loc[:d].dropna()
            if len(p) < MIN_HIST or p.index[-1] != d or p.iloc[-1] < MIN_PRICE:
                continue
            iss = sht = np.nan
            sh = sh_by.get(s)
            if sh is not None:
                now = sh.loc[:d]
                old = sh.loc[:d - pd.DateOffset(months=12)]
                if len(now) and len(old):
                    f_now, f_old = now.index[-1], old.index[-1]
                    a_now = now.iloc[-1] * split_factor_to_now(sp, s, f_now)
                    a_old = old.iloc[-1] * split_factor_to_now(sp, s, f_old)
                    if a_now > 0 and a_old > 0 and f_now > f_old:
                        iss = -(np.log(a_now) - np.log(a_old))
            si = si_by.get(s)
            if si is not None:
                av = si[si.avail <= d]
                if len(av) >= 3:
                    t1 = av.iloc[-1]
                    prev = av[av.settlementDate <= t1.settlementDate - pd.Timedelta(days=25)]
                    if len(prev):
                        t0 = prev.iloc[-1]
                        v1 = t1.si * split_factor_to_now(sp, s, t1.settlementDate)
                        v0 = t0.si * split_factor_to_now(sp, s, t0.settlementDate)
                        win = sp[(sp.symbol == s) & (sp.window_start <= t1.settlementDate) & (sp.date >= t0.settlementDate)]
                        if v1 > 0 and v0 > 0 and len(win) == 0:
                            sht = -(np.log(v1) - np.log(v0))
            mcap = np.nan
            if sh is not None and len(sh.loc[:d]):
                mcap = p.iloc[-1] * sh.loc[:d].iloc[-1] * split_factor_to_now(sp, s, sh.loc[:d].index[-1])
            rows.append({'date': d, 'symbol': s, 'ISS': iss, 'SHT': sht, 'mcap': mcap})
    return pd.DataFrame(rows)


def zscore(x):
    lo, hi = x.quantile(0.01), x.quantile(0.99)
    x = x.clip(lo, hi)
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def portfolio(sig, px, col, weighting='EW', cost_bps=COST_BPS):
    """Long top quintile / short bottom quintile of `col`, rebalanced at each date, daily returns."""
    rets = px.pct_change(fill_method=None)
    dates = sorted(sig.date.unique())
    w_prev = pd.Series(dtype=float)
    daily, turn = [], {}
    for i, d in enumerate(dates):
        g = sig[(sig.date == d)].dropna(subset=[col])
        if col == 'COMP':
            g = g.dropna(subset=['ISS', 'SHT'])
        if len(g) < 50:
            continue
        q = g[col].rank(pct=True)
        longs, shorts = g[q > 0.8], g[q <= 0.2]
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
        # drift within the month: compound holdings
        cum = (1 + r).cumprod()
        val = (cum * w).sum(axis=1)
        pv = pd.concat([pd.Series([w.sum()], index=[d]), val])
        dr = (pv.diff().iloc[1:]) / 1.0                     # dollar P&L per $1 gross/2 of book, long-short is self-financing
        dr.iloc[0] -= turn[d] * cost_bps / 1e4
        daily.append(dr)
    out = pd.concat(daily).sort_index()
    out = out[~out.index.duplicated()]
    return out, pd.Series(turn)


def sr(x):
    x = x.dropna()
    return float(x.mean() / x.std() * ANN12) if len(x) > 6 and x.std() > 0 else np.nan


def monthly(s):
    return s.resample('ME').sum()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    px, sp, uni, chg, sec, fin = load()
    missing = [s for s in uni if s not in px.columns]
    print(f"Universe {len(uni)}; no Yahoo prices for {len(missing)} ({', '.join(missing[:12])}{'...' if len(missing) > 12 else ''})")
    sig = build_signals(px, sp, uni, chg, sec, fin)
    sig['ISS_z'] = sig.groupby('date')['ISS'].transform(zscore)
    sig['SHT_z'] = sig.groupby('date')['SHT'].transform(zscore)
    sig['COMP'] = sig[['ISS_z', 'SHT_z']].mean(axis=1)
    sig.loc[sig[['ISS', 'SHT']].isna().any(axis=1), 'COMP'] = np.nan
    cov = sig.groupby('date').agg(n=('symbol', 'size'), iss=('ISS', 'count'), sht=('SHT', 'count'), comp=('COMP', 'count'))
    print(f"Rebalances {len(cov)}: names/month median {cov.n.median():.0f}, with ISS {cov.iss.median():.0f}, SHT {cov.sht.median():.0f}, both {cov.comp.median():.0f}")
    print(f"corr(ISS_z, SHT_z) pooled {sig.ISS_z.corr(sig.SHT_z):+.2f}")
    sig.to_csv(os.path.join(RESULTS_DIR, 'taking_sides_signals.csv'), index=False)

    spy = t2.load_prices()['SPY'].pct_change(fill_method=None)
    series = {}
    print(f"\n{'portfolio':22s}{'net SR':>8s}{'5bp':>7s}{'20bp':>7s}{'ann':>8s}{'vol':>7s}{'MDD':>8s}{'beta':>7s}{'last2y':>8s}{'TO/mo':>7s}")
    for col in ('COMP', 'ISS', 'SHT'):
        for wt in ('EW', 'VW'):
            r, to = portfolio(sig, px, col, wt)
            r = r.loc[START:END]
            r5 = portfolio(sig, px, col, wt, cost_bps=5.0)[0].loc[START:END]
            r20 = portfolio(sig, px, col, wt, cost_bps=20.0)[0].loc[START:END]
            m = monthly(r)
            j = pd.concat([m.rename('r'), monthly(spy.reindex(r.index)).rename('s')], axis=1).dropna()
            beta = j['r'].cov(j['s']) / j['s'].var()
            series[f'{col}_{wt}'] = r
            print(f"{col + ' ' + wt:22s}{sr(m):8.2f}{sr(monthly(r5)):7.2f}{sr(monthly(r20)):7.2f}{r.mean() * 252:8.1%}{r.std() * np.sqrt(252):7.1%}"
                  f"{bb.maxdd(r):8.1%}{beta:7.2f}{sr(monthly(r.loc['2024-09-01':])):8.2f}{to.mean():7.2f}")
    comp = series['COMP_EW']
    m = monthly(comp)
    print("Per-year net Sharpe COMP EW: " + "  ".join(f"{y}:{sr(x):+.2f}" for y, x in m.groupby(m.index.year)))

    # ---- gate
    t1 = (sr(m) >= 0.50) and (sr(monthly(series['COMP_VW'])) >= 0.30)
    t2_ = sr(m.loc['2024-09-01':]) > 0
    tot = (1 + m).prod() - 1
    blocks = {str(a.date()): (1 + m.loc[a:a + pd.DateOffset(months=23)]).prod() - 1 for a in m.index[:-23]}
    conc = max(blocks.values()) / tot if tot > 0 else np.inf
    t3 = conc <= 0.75
    eq, tr = bb.load_equity(), bb.load_trend()
    ts = comp.rename('ts')
    full = pd.concat([eq, tr, ts], axis=1)
    bb.FULL_FRAME = full
    j = full.loc['2020-01-01':'2025-12-31'].dropna()
    jm = j.resample('ME').sum()
    c_eq, c_tr = jm['ts'].corr(jm['equity']), jm['ts'].corr(jm['trend'])
    js = pd.concat([m.rename('r'), monthly(spy.reindex(comp.index)).rename('s')], axis=1).dropna()
    beta = js['r'].cov(js['s']) / js['s'].var()
    book2 = bb.book_returns(j[['equity', 'trend']], 'WF')
    b2m = monthly(book2)
    bad = jm.loc[b2m[b2m <= b2m.quantile(0.2)].index, 'ts']
    t4 = (c_eq < 0.10) and (c_tr < 0.10) and (abs(beta) < 0.25) and (bad.mean() > 0)
    book3 = bb.book_returns(j[['equity', 'trend', 'ts']], 'WF')
    y2, y3 = bb.yearly(book2), bb.yearly(book3)
    t5 = (bb.sharpe(book3) >= bb.sharpe(book2)) and (min(y3.values()) >= min(y2.values())) and (bb.maxdd(book3) >= bb.maxdd(book2))

    print("\nGO/NO-GO (pre-registered, COMP):")
    print(f"  {'PASS' if t1 else 'fail'}  T1 net Sharpe EW {sr(m):.2f} (>= 0.50); VW {sr(monthly(series['COMP_VW'])):.2f} (>= 0.30)")
    print(f"  {'PASS' if t2_ else 'fail'}  T2 last two years {sr(m.loc['2024-09-01':]):+.2f} (> 0)")
    print(f"  {'PASS' if t3 else 'fail'}  T3 best 24-month block = {conc:.0%} of profit (<= 75%)")
    print(f"  {'PASS' if t4 else 'fail'}  T4 corr equity {c_eq:+.2f}, trend {c_tr:+.2f} (< 0.10); beta {beta:+.2f} (|.| < 0.25); bad-month mean {bad.mean() * 100:+.2f}% (n={len(bad)}, hit {(bad > 0).mean():.0%})")
    print(f"  {'PASS' if t5 else 'fail'}  T5 WF book rf {bb.sharpe(book2):.2f} -> {bb.sharpe(book3):.2f}; worst {min(y2.values()):+.2f} -> {min(y3.values()):+.2f}; MDD {bb.maxdd(book2):.1%} -> {bb.maxdd(book3):.1%}")
    print(f"  info  book no-rf {bb.sharpe(book2, 0):.2f} -> {bb.sharpe(book3, 0):.2f}")
    verdict = all([t1, t2_, t3, t4, t5])
    print(f"\nVERDICT: {'PROMOTE to paper-sleeve candidate' if verdict else 'not promoted'}")
    pd.DataFrame(series).to_csv(os.path.join(RESULTS_DIR, 'taking_sides_daily.csv'))
    print("Saved results/taking_sides_signals.csv, taking_sides_daily.csv")


if __name__ == '__main__':
    main()
