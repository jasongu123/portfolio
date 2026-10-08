"""
Intramonth legs 2+3: replication on other instruments — RESEARCH ONLY. Universes and gate fixed
2026-09-15 before any data was fetched.

Question: is the turn-of-month cycle dead, or crowded out of the nine sector SPDRs? The rule is
frozen (legs 2+3, 252-day ranking, long top third / short bottom third, equal weight, 1 bp) and
applied unchanged to three universes nobody tuned it on:
  U1 SPDR industry ETFs (sub-sector, thinner flow arbitrage):
     XHB XRT KRE XOP XME XBI XSD XPH KBE KIE XAR XTN            (12; most from 2006-2011)
  U2 Invesco equal-weight sector ETFs (same sectors, different holder base):
     RSPT RSPH RSPF RSPG RSPN RSPS RSPU RSPD RSPM               (9; 2006-11 onward)
  U3 largest three stocks per S&P sector, fixed today (survivorship: names are 2026 leaders,
     so pre-2020 numbers are biased upward; the since-2024 test is the one that counts):
     LIN SHW APD | XOM CVX COP | JPM BRK-B V | GE CAT RTX | AAPL MSFT NVDA | PG COST WMT |
     NEE SO DUK | LLY UNH JNJ | AMZN TSLA HD                     (27; 9 long / 9 short)
Data: Yahoo adjusted closes, cached in backtest_data/intramonth_replication_prices.csv.

Sample-size warning: since 2024-01 each universe has ~95 active days, Sharpe s.e. ~0.6. The
chance that at least one of three universes shows Sharpe >= 0.30 by luck when the true value
is zero is about two in three. The gate therefore pools.

Pre-registered gate ("the effect is alive somewhere"):
  R1 pooled recency: the equal-weight average of the three universes' daily net returns has
     net Sharpe >= 0.30 over 2024-01-01..latest, AND at least two of the three universes are
     individually positive over that window.                                       binding
  R2 pooled history: the pooled series has net Sharpe >= 0.30 over its full common sample and
     is positive in 2020-2023 (the effect must exist there in the years it worked on SPDRs;
     if the SPDR result was itself noise, it will not).                             binding
  R3 orthogonality: for any universe proposed for trading, monthly corr with equity and trend
     over 2020-2025 each < 0.10.                                                   binding
  R4 concentration: no four-year block supplies > 75% of the pooled series' post-2012 profit. binding
  Decision: R1-R4 all pass -> the effect is alive off the SPDRs; propose the best-supported
     universe for a paper sleeve (its own gate later). R1 fails -> the effect is dead
     everywhere since 2024; retire the intramonth candidate with that reason.
  Informational: per-universe full-sample, 2012-2019, 2020-2023, since-2024, last-2y Sharpe;
     0 and 2 bp; SPDR line for reference.
Trials: T-2026-09-15-25 (U1), -26 (U2), -27 (U3), -28 (pooled).
Usage: /usr/local/bin/python3 intramonth_replication.py
"""

import os
import time
import numpy as np
import pandas as pd

import intramonth_sector as ims
import book_backtest as bb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = ims.RESULTS_DIR
CACHE = os.path.join(SCRIPT_DIR, 'backtest_data', 'intramonth_replication_prices.csv')
UNIVERSES = {
    'U1_industry_spdr': ['XHB', 'XRT', 'KRE', 'XOP', 'XME', 'XBI', 'XSD', 'XPH', 'KBE', 'KIE', 'XAR', 'XTN'],
    'U2_equal_weight_sector': ['RSPT', 'RSPH', 'RSPF', 'RSPG', 'RSPN', 'RSPS', 'RSPU', 'RSPD', 'RSPM'],
    'U3_top3_stocks_per_sector': ['LIN', 'SHW', 'APD', 'XOM', 'CVX', 'COP', 'JPM', 'BRK-B', 'V', 'GE', 'CAT', 'RTX',
                                  'AAPL', 'MSFT', 'NVDA', 'PG', 'COST', 'WMT', 'NEE', 'SO', 'DUK', 'LLY', 'UNH', 'JNJ',
                                  'AMZN', 'TSLA', 'HD'],
}
RECENT = '2024-01-01'
LAST2Y = '2024-09-15'


def fetch(tickers, chunk=6, cooldown=120, tries=6):
    """Yahoo adjusted closes in small batched calls with a long cooldown on rate limits. The cache
    is written after every chunk so partial progress survives. Stooq (2026-09) sits behind a
    JavaScript challenge and is not usable; TWS is not reachable from the research machine."""
    import yfinance as yf
    have = pd.read_csv(CACHE, index_col=0, parse_dates=True) if os.path.exists(CACHE) else pd.DataFrame()
    out = {c: have[c] for c in have.columns}
    todo = [t for t in tickers if not (t in out and out[t].dropna().index.max() >= pd.Timestamp('2026-09-01'))]
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        for attempt in range(tries):
            try:
                df = yf.download(batch, period='max', auto_adjust=True, threads=False, progress=False, group_by='column')
                close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df[['Close']].rename(columns={'Close': batch[0]})
                got = [c for c in close.columns if close[c].notna().sum() > 100]
                if not got:
                    raise RuntimeError('empty response')
                for c in got:
                    s_ = close[c].dropna()
                    s_.index = pd.to_datetime(s_.index).tz_localize(None).normalize()
                    out[c] = s_[~s_.index.duplicated()]
                    print(f"  {c}: {len(s_)} rows from {s_.index.min().date()}", flush=True)
                miss = [c for c in batch if c not in got]
                if miss:
                    print(f"  no data for {miss}", flush=True)
                break
            except Exception as e:
                print(f"  batch {batch} attempt {attempt + 1} failed ({str(e)[:60]}); cooling {cooldown}s", flush=True)
                time.sleep(cooldown)
        pd.DataFrame(out).sort_index().to_csv(CACHE)
        time.sleep(5)
    df = pd.DataFrame(out).sort_index()
    df.to_csv(CACHE)
    return df


def run_universe(px, names, cost_bps=ims.COST_BPS):
    """Frozen legs-2+3 rule on an arbitrary universe: rank by 252-day return at the leg-2 entry
    day (D0 close), long top third / short bottom third equal-weight, earn D+1; flip at D+1 close,
    earn D+2 and D+3. Months need >= 11 days; next month >= 3."""
    px = px[names].dropna(how='all')
    n_top = max(1, len(names) // 3)
    mom = px / px.shift(ims.LOOKBACK) - 1.0
    rets = px.pct_change(fill_method=None)
    w = pd.DataFrame(0.0, index=px.index, columns=names)
    leg_of = pd.Series('', index=px.index)
    months = px.groupby(px.index.to_period('M')).apply(lambda g: list(g.index))
    keys = list(months.index)
    for i, k in enumerate(keys):
        days = months[k]
        if len(days) < 11 or i + 1 >= len(keys) or len(months[keys[i + 1]]) < 3:
            continue
        d0 = days[-1]
        m = mom.loc[d0].dropna()
        if len(m) < max(4, int(0.8 * len(names))):      # need most of the universe ranked
            continue
        ranked = m.sort_values()
        k_ = max(1, len(m) // 3)
        pos = pd.Series(0.0, index=names)
        pos[ranked.index[-k_:]] = 1.0 / k_
        pos[ranked.index[:k_]] = -1.0 / k_
        w.loc[d0] = pos.values
        leg_of.loc[d0] = 'L2'
        for d in months[keys[i + 1]][:2]:
            w.loc[d] = -pos.values
            leg_of.loc[d] = 'L3'
    gross = (w.shift(1) * rets.fillna(0.0)).sum(axis=1)
    cost = w.diff().abs().sum(axis=1) * cost_bps / 1e4
    net = (gross - cost).dropna()
    first = w.abs().sum(axis=1).gt(0).idxmax()
    return net.loc[first:], leg_of.shift(1).reindex(net.loc[first:].index)


def sr(x):
    return ims.sharpe(x)


def cum(x):
    return (1 + x).prod() - 1


def main():
    all_t = sorted({t for v in UNIVERSES.values() for t in v})
    print("Fetching / loading prices...")
    px = fetch(all_t)
    missing = [t for t in all_t if t not in px.columns or px[t].dropna().empty]
    print(f"Missing: {missing}")

    spdr = ims.run(ims.load_prices(), 'Strat2', legs=(2, 3))[0].rename('SPDR_L23')
    series = {'SPDR_L23 (reference)': spdr}
    windows = (('full', None, None), ('2012-2019', '2012-01-01', '2019-12-31'), ('2020-2023', '2020-01-01', '2023-12-31'),
               ('since 2024', RECENT, None), ('last 2y', LAST2Y, None))
    print(f"\n{'universe':30s}{'start':>11s}{'n':>4s}" + "".join(f"{w[0]:>12s}" for w in windows) + f"{'0bp full':>10s}{'2bp full':>10s}{'2024 cum':>10s}")
    for name, names in UNIVERSES.items():
        names = [t for t in names if t not in missing]
        if len(names) < 4:
            print(f"{name:30s} insufficient data ({len(names)} names) — skipped")
            continue
        net, leg = run_universe(px, names)
        series[name] = net.rename(name)
        n0 = run_universe(px, names, cost_bps=0.0)[0]
        n2 = run_universe(px, names, cost_bps=2.0)[0]
        line = f"{name:30s}{str(net.index.min().date()):>11s}{len(names):4d}"
        for _, a, b in windows:
            x = net.loc[a:b] if a else net
            line += f"{sr(x):12.2f}"
        line += f"{sr(n0):10.2f}{sr(n2):10.2f}{cum(net.loc[RECENT:]):10.1%}"
        print(line)
    line = f"{'SPDR_L23 (reference)':30s}{str(spdr.index.min().date()):>11s}{9:4d}"
    for _, a, b in windows:
        x = spdr.loc[a:b] if a else spdr
        line += f"{sr(x):12.2f}"
    print(line + f"{'':20s}{cum(spdr.loc[RECENT:]):10.1%}")

    # pooled
    U = pd.concat([series[k] for k in UNIVERSES if k in series], axis=1).dropna()
    if U.shape[1] < 2:
        print("Fewer than two universes available — gate cannot run."); return
    pooled = U.mean(axis=1).rename('pooled')
    print(f"\nPooled (equal-weight of the three universes), common sample {pooled.index.min().date()}..{pooled.index.max().date()}:")
    for lab, a, b in windows:
        x = pooled.loc[a:b] if a else pooled
        print(f"  {lab:12s} Sharpe {sr(x):+.2f}  cum {cum(x):+.1%}  ({len(x)} days)")
    print("  per year: " + "  ".join(f"{y}:{sr(x):+.2f}" for y, x in pooled.groupby(pooled.index.year)))
    print("  pairwise corr of universes (daily): " + ", ".join(f"{a[:2]}~{b[:2]} {U[a].corr(U[b]):+.2f}" for i, a in enumerate(U.columns) for b in U.columns[i + 1:]))

    # gate
    rec = pooled.loc[RECENT:]
    pos_each = sum(sr(series[k].loc[RECENT:]) > 0 for k in UNIVERSES)
    r1 = (sr(rec) >= 0.30) and (pos_each >= 2)
    r2 = (sr(pooled) >= 0.30) and (sr(pooled.loc['2020-01-01':'2023-12-31']) > 0)
    eq, tr = bb.load_equity(), bb.load_trend()
    corrs = {}
    for k in list(UNIVERSES) + ['pooled']:
        s_ = (series[k] if k != 'pooled' else pooled).rename('x')
        j = pd.concat([eq, tr, s_], axis=1).loc['2020':'2025'].dropna().resample('ME').sum()
        corrs[k] = (j['x'].corr(j['equity']), j['x'].corr(j['trend']))
    r3 = all(c[0] < 0.10 and c[1] < 0.10 for c in corrs.values())
    g = pooled.loc['2012':]
    blocks = {f'{y}-{y + 3}': cum(g.loc[str(y):str(y + 3)]) for y in range(2012, 2024)}
    conc = max(blocks.values()) / cum(g) if cum(g) > 0 else np.inf
    r4 = conc <= 0.75

    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if r1 else 'fail'}  R1 pooled since 2024 Sharpe {sr(rec):+.2f} (>= 0.30); universes positive since 2024: {pos_each}/3 (>= 2)")
    print(f"  {'PASS' if r2 else 'fail'}  R2 pooled full Sharpe {sr(pooled):+.2f} (>= 0.30); 2020-2023 {sr(pooled.loc['2020-01-01':'2023-12-31']):+.2f} (> 0)")
    print(f"  {'PASS' if r3 else 'fail'}  R3 monthly corr with equity / trend: " + "; ".join(f"{k[:2]} {c[0]:+.2f}/{c[1]:+.2f}" for k, c in corrs.items()))
    print(f"  {'PASS' if r4 else 'fail'}  R4 best 4-year block {max(blocks, key=blocks.get)} = {conc:.0%} of post-2012 pooled profit (<= 75%)")
    alive = all([r1, r2, r3, r4])
    print(f"\nVERDICT: {'ALIVE off the SPDRs — propose a universe for its own gate' if alive else ('DEAD everywhere since 2024 — retire the intramonth candidate' if not r1 else 'not alive on the registered criteria')}")
    pd.concat([series[k] for k in series] + [pooled], axis=1).to_csv(os.path.join(RESULTS_DIR, 'intramonth_replication_daily.csv'))
    print("Saved results/intramonth_replication_daily.csv")


if __name__ == '__main__':
    main()
