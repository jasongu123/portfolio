"""
Trend sleeve: rebalance timing off month-end — RESEARCH ONLY. Gate pre-registered 2026-09-16
before the run. Does not touch trend_sleeve_live.py.

Hypothesis (two mechanisms, stated first):
 1. Month-end is the most crowded execution window in the calendar: pension and fund rebalancing
    flows are anticipated and front-run (INTRAMONTH_REPLICATION.md showed the equity version of
    this since 2024). A trend sleeve that both evaluates its signals and trades at the month-end
    close pays that pressure and reads signals contaminated by it. Trading on an ordinary
    mid-month day should not.
 2. A single stamp day is one draw of "rebalance timing luck": the same rule on different days
    of the month gives materially different paths for no economic reason. Spreading the book
    over several stamp days removes that variance without changing the rule.

Configurations (V1_trail research baseline: v2 mean-sign ensemble 63/126/252, per-asset vol
targeting, 1/n, trailing 504-day scalar, 2 bp; live 16-ETF universe; identical everywhere):
  ME        stamp on the last trading day of the month (the live convention)
  D_k       stamp on the k-th trading day, k = 1..20 (diagnostic sweep, NOT a selection)
  STAG4     four equal tranches stamped on trading days 1, 6, 11, 16 (the candidate, fixed now)
  STAG2     two tranches on days 1 and 11 (informational)
Weights of a tranche are the rule's weights at its own stamp, active the next trading day;
the book is the average of tranche weights; turnover and costs are computed on the averaged
book (each tranche trades a quarter of the notional on its day).

Pre-registered gate (rf-Sharpe, flat 5%, daily; full window 2008-01-01..latest):
  R1 month-end is a bad day: ME rf-Sharpe is below the median of the 20 single-day stamps,
     and the mean of D_5..D_15 exceeds ME by >= 0.05.                                binding
  R2 the candidate: STAG4 rf-Sharpe >= ME + 0.05 and STAG4 MaxDD not deeper than ME.  binding
  R3 not one period: STAG4 - ME cumulative-return difference is positive in at least two of
     the three sub-periods 2008-2013, 2014-2019, 2020-2026, and the last-two-years
     rf-Sharpe difference is >= -0.05.                                                binding
  R4 book: walk-forward equity + STAG4 over 2020-2025 not worse than equity + ME on
     rf-Sharpe, worst year and MaxDD.                                                 binding
  Informational: turnover, cost, worst year, SPY corr of each; STAG2; the sweep table.
Decision: R1-R4 pass -> STAG4 replaces the month-end stamp in the live sleeve through a
versioned change (execution split over four days). Any fail -> not; the sweep is not mined.
Trials: T-2026-09-16-38 (STAG4), -39 (the single-day sweep as one diagnostic entry).
Usage: /usr/local/bin/python3 trend_rebalance_timing.py
"""

import os
import numpy as np
import pandas as pd

import trend_follow_v2 as t2
import trend_follow_v3 as v3
import book_backtest as bb

RESULTS_DIR = v3.RESULTS_DIR
FULL_START = v3.FULL_START
SUBS = (('2008-2013', '2008-01-01', '2013-12-31'), ('2014-2019', '2014-01-01', '2019-12-31'), ('2020-2026', '2020-01-01', None))
TRANCHES4, TRANCHES2 = (1, 6, 11, 16), (1, 11)


def stamps_kth(px, k):
    """Index of the k-th trading day of each month (last day if the month is shorter); k='ME' = last."""
    g = px.groupby(px.index.to_period('M'))
    if k == 'ME':
        return pd.DatetimeIndex(g.apply(lambda x: x.index[-1]).values)
    return pd.DatetimeIndex(g.apply(lambda x: x.index[min(k, len(x)) - 1]).values)


def weights_at(prices, universe, stamps):
    """V1_trail unscaled weights with the rule evaluated at `stamps` (monkeypatching v3's stamp index)."""
    orig = v3.month_end_index
    v3.month_end_index = lambda px, legacy=False: stamps
    try:
        normalized, px, rets, _, _ = v3.build_weights(prices, universe, 'V2', t2.HORIZONS, legacy=False)
    finally:
        v3.month_end_index = orig
    return normalized, px, rets


def run_config(prices, universe, ks):
    """ks: list of stamp keys ('ME' or ints). One tranche per key; book = average of tranche weights."""
    parts, all_stamps = [], []
    for k in ks:
        st = stamps_kth(prices[[c for c in universe if c in prices.columns]], k)
        normalized, px, rets = weights_at(prices, universe, st)
        parts.append(normalized)
        all_stamps.append(st)
    book = sum(parts) / len(parts)
    stamps = pd.DatetimeIndex(sorted(set().union(*[set(s) for s in all_stamps])))
    port = v3.scale_and_cost(book, px, rets, scalar_mode='trailing', month_ends=stamps)
    return port


def stats(r, label, spy):
    full = v3.window(r, FULL_START)
    yr = v3.yearly(full)
    return {'config': label, 'rf_sharpe': v3.sharpe(full), 'norf_sharpe': v3.sharpe(full, 0.0),
            'cagr': (1 + full).prod() ** (252 / len(full)) - 1, 'vol': full.std() * v3.ANN, 'maxdd': v3.maxdd(full),
            'worst_year': min(yr.values()), 'spy_corr': v3.corr_with(full, spy),
            'turnover': r.attrs['turnover_per_year'], 'last2y_rf': v3.sharpe(full.loc['2024-09-15':]),
            **{f'y{y}': s for y, s in yr.items()}}


def main():
    prices = t2.load_prices()
    live16 = [c for c in v3.LIVE16 if c in prices.columns]
    spy = prices['SPY'].pct_change(fill_method=None)
    eq = bb.load_equity()
    series, rows = {}, []
    for label, ks in [('ME', ['ME'])] + [(f'D_{k}', [k]) for k in range(1, 21)] + [('STAG4', list(TRANCHES4)), ('STAG2', list(TRANCHES2))]:
        port = run_config(prices, live16, ks)
        series[label] = port
        rows.append(stats(port, label, spy))
    df = pd.DataFrame(rows).set_index('config')
    chk = pd.concat([series['ME'].rename('a'), pd.read_csv(os.path.join(RESULTS_DIR, 'trend_v3_returns.csv'), index_col=0, parse_dates=True)['V1_trail'].rename('b')], axis=1).dropna()
    print(f"CHECK ME reproduces results/trend_v3_returns.csv V1_trail: max |diff| {(chk.a - chk.b).abs().max():.2e} over {len(chk)} days")

    print(f"\nFull window {FULL_START}..{series['ME'].index.max().date()}, live 16, V1_trail rule, 2 bp; rf-Sharpe headline")
    print(f"{'config':8s}{'rf SR':>7s}{'no-rf':>7s}{'CAGR':>7s}{'vol':>7s}{'MDD':>8s}{'worst':>7s}{'SPYc':>6s}{'TO/yr':>7s}{'last2y':>8s}")
    for c, r in df.iterrows():
        print(f"{c:8s}{r.rf_sharpe:7.2f}{r.norf_sharpe:7.2f}{r.cagr:7.1%}{r.vol:7.1%}{r.maxdd:8.1%}{r.worst_year:7.2f}{r.spy_corr:6.2f}{r.turnover:7.1f}{r.last2y_rf:8.2f}")
    singles = df.loc[[f'D_{k}' for k in range(1, 21)], 'rf_sharpe']
    mid = df.loc[[f'D_{k}' for k in range(5, 16)], 'rf_sharpe'].mean()
    me = df.loc['ME', 'rf_sharpe']
    print(f"\nSingle-day stamps: median rf-Sharpe {singles.median():.2f}, min {singles.min():.2f} ({singles.idxmin()}), max {singles.max():.2f} ({singles.idxmax()}); "
          f"ME {me:.2f} is at percentile {(singles < me).mean():.0%}; mean of D_5..D_15 {mid:.2f}")
    r1 = (me < singles.median()) and (mid - me >= 0.05)
    s4 = df.loc['STAG4']
    r2 = (s4.rf_sharpe >= me + 0.05) and (s4.maxdd >= df.loc['ME', 'maxdd'])
    diffs = []
    for name, a, b in SUBS:
        x = v3.window(series['STAG4'], a, b); y = v3.window(series['ME'], a, b)
        j = pd.concat([x.rename('s'), y.rename('m')], axis=1).dropna()
        diffs.append(((1 + j.s).prod() - 1) - ((1 + j.m).prod() - 1))
        print(f"  {name}: STAG4 rf {v3.sharpe(j.s):.2f} vs ME {v3.sharpe(j.m):.2f}; cumulative difference {diffs[-1]:+.1%}")
    last2 = s4.last2y_rf - df.loc['ME', 'last2y_rf']
    r3 = (sum(d > 0 for d in diffs) >= 2) and (last2 >= -0.05)
    bb.FULL_FRAME = pd.concat([eq, series['ME'].rename('trend_me'), series['STAG4'].rename('trend_s4')], axis=1)
    j = bb.FULL_FRAME.loc['2020-01-01':'2025-12-31'].dropna()
    b_me = bb.book_returns(j[['equity', 'trend_me']].rename(columns={'trend_me': 'trend'}), 'WF') if False else None
    # walk-forward books with each trend variant (rename so FULL_FRAME lookups work)
    def wf_book(col):
        f = bb.FULL_FRAME[['equity', col]].rename(columns={col: 'trend'})
        bb.FULL_FRAME = f
        out = bb.book_returns(f.loc['2020-01-01':'2025-12-31'].dropna(), 'WF')
        return out
    full_frame = bb.FULL_FRAME
    b_me = wf_book('trend_me'); bb.FULL_FRAME = full_frame
    b_s4 = wf_book('trend_s4'); bb.FULL_FRAME = full_frame
    y_me, y_s4 = bb.yearly(b_me), bb.yearly(b_s4)
    r4 = (bb.sharpe(b_s4) >= bb.sharpe(b_me)) and (min(y_s4.values()) >= min(y_me.values())) and (bb.maxdd(b_s4) >= bb.maxdd(b_me))

    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if r1 else 'fail'}  R1 ME {me:.2f} below single-stamp median {singles.median():.2f}; mid-month mean - ME = {mid - me:+.2f} (>= 0.05)")
    print(f"  {'PASS' if r2 else 'fail'}  R2 STAG4 rf {s4.rf_sharpe:.2f} vs ME {me:.2f} (need +0.05); MDD {s4.maxdd:.1%} vs {df.loc['ME', 'maxdd']:.1%}")
    print(f"  {'PASS' if r3 else 'fail'}  R3 sub-period differences positive in {sum(d > 0 for d in diffs)}/3; last-2y rf difference {last2:+.2f} (>= -0.05)")
    print(f"  {'PASS' if r4 else 'fail'}  R4 WF book equity+trend: rf {bb.sharpe(b_me):.2f} -> {bb.sharpe(b_s4):.2f}; worst {min(y_me.values()):+.2f} -> {min(y_s4.values()):+.2f}; MDD {bb.maxdd(b_me):.1%} -> {bb.maxdd(b_s4):.1%}")
    print(f"  info  STAG2 rf {df.loc['STAG2', 'rf_sharpe']:.2f}; book no-rf {bb.sharpe(b_me, 0):.2f} -> {bb.sharpe(b_s4, 0):.2f}")
    verdict = all([r1, r2, r3, r4])
    print(f"\nVERDICT: {'ADOPT STAG4 for the live trend sleeve (versioned change)' if verdict else 'not adopted'}")
    pd.DataFrame({k: v for k, v in series.items()}).to_csv(os.path.join(RESULTS_DIR, 'trend_rebalance_timing_returns.csv'))
    df.to_csv(os.path.join(RESULTS_DIR, 'trend_rebalance_timing_summary.csv'))
    print("Saved results/trend_rebalance_timing_returns.csv, _summary.csv")


if __name__ == '__main__':
    main()
