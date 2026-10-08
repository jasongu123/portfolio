"""
Cost-consistent, walk-forward rebuild of the headline book numbers (2026-09-16).

What changes versus the PORTFOLIO.md §1 table (the "legacy" version):
  - equity sleeve returns regenerated through the shared cost model (trading_costs.py, trade-point
    prices, commission minimum, 1 bp fallback spread) instead of the pre-2026-08 cost engine
  - trend sleeve reported AS LIVE: the 16-ETF disjoint universe, the live sleeve's conventions
    (v2 lookback and calendar month-end stamps) and its constant portfolio scalar 2.95, through
    the same shared cost model; the research baseline V1_trail (corrected conventions, trailing
    scalar) is shown alongside; the old 18-ETF full-sample-scalar series is shown for reference
  - book weights: walk-forward inverse-vol (trailing 252 days, monthly, equal weight until 63
    days of history) instead of full-overlap inverse-vol, which used the whole window's vol
  - drawdown with the high-water mark at initial capital (the legacy helper understated it)
  - one Sharpe convention per table, labelled; PSR/DSR on no-rf daily Sharpe with an auditable
    trial count from TRIAL_REGISTRY.md
Usage: /usr/local/bin/python3 headline_rebuild.py [--equity legacy|new|vixcausal|vixintraday]
(vixcausal = prior-day VIX in the backtester, 2026-09-24; vix_causal_rerun.py writes it)
Writes results/headline_2026-09_daily.csv and prints the tables for HEADLINE_2026-09.md.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

import trend_follow_v2 as t2
import trend_follow_v3 as v3
import book_backtest as bb
from trading_costs import CostModelConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
LIVE_SCALAR = 2.95
START, END = '2020-01-01', '2025-12-31'
ANN = np.sqrt(252)
EMC = 0.5772156649


def load_equity(which):
    parts = []
    for y in range(2020, 2026):
        f = (os.path.join(RESULTS_DIR, 'audit_rerun', f'headline_equity_{y}.csv') if which == 'audit'
             else os.path.join(SCRIPT_DIR, 'legacy_caches', f'equity_returns_legacy_{y}.csv') if which == 'legacy'
             else os.path.join(RESULTS_DIR, 'vixcausal', f'equity_returns_{y}.csv') if which == 'vixcausal'
             else os.path.join(RESULTS_DIR, 'vixintraday', f'equity_returns_{y}.csv') if which == 'vixintraday'
             else os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'))
        parts.append(pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0])
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()].rename('equity')


def trend_as_live(prices, universe):
    """Live conventions (legacy lookback/stamps) with the constant live scalar, shared cost model."""
    normalized, px, rets, _, _ = v3.build_weights(prices, universe, 'V2', t2.HORIZONS, legacy=True)
    target = normalized * LIVE_SCALAR
    gross = (target * rets).sum(axis=1)
    cfg = CostModelConfig(commission_per_share=t2.COMMISSION_PER_SHARE, min_commission=t2.MIN_COMMISSION,
                          fallback_execution_bps=t2.COST_BPS_PER_REBAL, extra_slippage_bps=0.0,
                          annual_borrow_rate=t2.ANNUAL_BORROW_RATE, annual_financing_rate=t2.ANNUAL_FINANCING_RATE, day_count=360.0)
    port = t2._apply_portfolio_costs(gross, target, px, cfg, t2.BACKTEST_CAPITAL)
    active = normalized.abs().sum(axis=1).gt(0)
    port = port.loc[active.idxmax():].dropna()
    port.attrs['turnover_per_year'] = float(target.diff().abs().sum(axis=1).loc[active.idxmax():].mean() * 252)
    return port


def stats(r):
    r = r.dropna()
    yr = v3.yearly(r)
    return {'rf': v3.sharpe(r), 'norf': v3.sharpe(r, 0.0), 'worst': min(yr.values()), 'worst_yr': min(yr, key=yr.get),
            'mdd': v3.maxdd(r), 'vol': r.std() * ANN, 'cagr': (1 + r).prod() ** (252 / len(r)) - 1, 'years': yr}


def psr(r, sr_star_ann=0.0):
    r = r.dropna(); sr = r.mean() / r.std(); g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False)); n = len(r)
    num = (sr - sr_star_ann / ANN) * np.sqrt(n - 1)
    den = np.sqrt(max(1e-9, 1 - g3 * sr + ((g4 - 1) / 4) * sr ** 2))
    return float(norm.cdf(num / den)), sr * ANN, g3, g4, n


def dsr(r, var_sr_daily, N):
    r = r.dropna(); sr = r.mean() / r.std(); g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False)); n = len(r)
    sr_star = np.sqrt(var_sr_daily) * ((1 - EMC) * norm.ppf(1 - 1.0 / N) + EMC * norm.ppf(1 - 1.0 / (N * np.e)))
    num = (sr - sr_star) * np.sqrt(n - 1)
    den = np.sqrt(max(1e-9, 1 - g3 * sr + ((g4 - 1) / 4) * sr ** 2))
    return float(norm.cdf(num / den)), sr_star * ANN


def registry_count():
    s = open(os.path.join(SCRIPT_DIR, 'TRIAL_REGISTRY.md')).read()
    return len(re.findall(r'^\| T-\d{4}-\d{2}-\d{2}-\d+ \|', s, flags=re.M))


def row(label, st, years):
    return (f"| {label} | {st['rf']:.2f} | {st['norf']:.2f} | {st['worst']:+.2f} ({st['worst_yr']}) | {st['mdd']:.1%} | {st['vol']:.1%} | {st['cagr']:+.1%} | "
            + " | ".join(f"{st['years'].get(y, np.nan):+.2f}" for y in years) + " |")


def main():
    which = 'new'
    if '--equity' in sys.argv:
        which = sys.argv[sys.argv.index('--equity') + 1]
    prices = t2.load_prices()
    live16 = [c for c in v3.LIVE16 if c in prices.columns]
    present = [c for c in t2.SECTOR if c in prices.columns]
    spy = prices['SPY'].pct_change(fill_method=None)
    eq = load_equity(which)
    eq_leg = load_equity('legacy')
    tr_live = trend_as_live(prices, live16).rename('trend_live')
    tr_v1 = v3.run_variant(prices, live16, 'V2', t2.HORIZONS, scalar_mode='trailing').rename('trend_v1trail')
    tr_old = t2.backtest(prices, present, t2.HORIZONS).rename('trend_legacy18')
    win = lambda s: s.loc[START:END]
    years = list(range(2020, 2026))

    print(f"Equity caches: {which}; days {len(win(eq))}; corr(new, legacy) {win(eq).corr(win(eq_leg)):.4f}; "
          f"mean daily diff {(win(eq) - win(eq_leg)).mean() * 1e4:+.2f} bp")
    print(f"Trend as-live vs V1_trail corr (2008+): {v3.window(tr_live, '2008-01-01').corr(v3.window(tr_v1, '2008-01-01')):.3f}; "
          f"as-live turnover/yr {tr_live.attrs['turnover_per_year']:.1f}")

    hdr = "| Series | rf-Sharpe | no-rf | worst year (rf) | MaxDD | vol | CAGR | " + " | ".join(str(y) for y in years) + " |"
    sep = "|" + "---|" * (7 + len(years))
    print("\nSLEEVES 2020-2025 (rf = flat 5%; MaxDD from initial capital)")
    print(hdr); print(sep)
    sleeves = {'Equity sleeve (shared cost model)': win(eq), 'Equity sleeve (legacy cache, prior version)': win(eq_leg),
               'Trend as live (16 ETFs, live conventions, scalar 2.95)': win(tr_live),
               'Trend research baseline V1_trail (16 ETFs, corrected, trailing scalar)': win(tr_v1),
               'Trend legacy headline (18 ETFs incl. SPY/IWM, full-sample scalar)': win(tr_old)}
    for k, s in sleeves.items():
        print(row(k, stats(s), years))
    print(f"\nSleeve correlations 2020-2025 (daily): equity~trend_live {win(eq).corr(win(tr_live)):+.2f}; equity~V1_trail {win(eq).corr(win(tr_v1)):+.2f}; "
          f"trend_live~SPY {win(tr_live).corr(spy.reindex(win(tr_live).index)):+.2f}; equity~SPY {win(eq).corr(spy.reindex(win(eq).index)):+.2f}")

    books = {}
    for tname, ts in (('trend_live', tr_live), ('trend_v1trail', tr_v1)):
        full = pd.concat([eq, ts.rename('trend')], axis=1)
        bb.FULL_FRAME = full
        j = full.loc[START:END].dropna()
        books[f'WF: equity + {tname}'] = bb.book_returns(j, 'WF')
        books[f'RP full-overlap (old convention): equity + {tname}'] = bb.book_returns(j, 'RP')
    full = pd.concat([eq_leg, tr_old.rename('trend')], axis=1)
    j = full.loc[START:END].dropna()
    books['Legacy headline reproduction: RP, legacy equity + legacy 18-ETF trend'] = bb.book_returns(j, 'RP')
    print("\nBOOKS 2020-2025")
    print(hdr); print(sep)
    for k, s in books.items():
        print(row(k, stats(s), years))
    W = bb.wf_weights(pd.concat([eq, tr_live.rename('trend')], axis=1).loc[START:END].dropna(), pd.concat([eq, tr_live.rename('trend')], axis=1)).dropna()
    print(f"\nWalk-forward weights (equity/trend_live): mean {W.mean().round(2).to_dict()}, min equity {W.equity.min():.2f}, max {W.equity.max():.2f}")

    head = books['WF: equity + trend_live']
    print("\nRIGOR (no-rf daily Sharpe convention throughout this block)")
    for label, r in (('equity (shared cost)', win(eq)), ('headline book, WF equity + trend_live', head)):
        p0, sr_ann, g3, g4, n = psr(r, 0.0); p05 = psr(r, 0.5)[0]
        print(f"  {label:40s} SR {sr_ann:.2f} | skew {g3:+.2f} | kurt {g4:.1f} | n={n} | PSR(SR>0) {p0:.1%} | PSR(SR>0.5) {p05:.1%}")
    # trial dispersion: standalone no-rf Sharpes of sleeve-level candidates on the 2020-2025 window, exact figures from the docs
    trials = {'equity': stats(win(eq))['norf'], 'trend_live': stats(win(tr_live))['norf'], 'intramonth (all legs)': 0.43, 'RFS': 0.49,
              'taking sides': -0.28, 'seasonality': -0.89, 'crowded spaces': -0.59, 'short-interest timing': 0.0, 'bearish positioning': 0.0}
    var_daily = np.var([v / ANN for v in trials.values()], ddof=1)
    nreg = registry_count()
    print(f"  trial dispersion from {len(trials)} sleeve-level candidates on 2020-2025 (no-rf): sd {np.sqrt(var_daily) * ANN:.2f}; registry rows {nreg}")
    print(f"  {'N':>6s}{'hurdle SR* (no-rf)':>20s}{'DSR':>8s}")
    for N in (len(trials), nreg, 100):
        d, h = dsr(head, var_daily, N)
        print(f"  {N:6d}{h:20.2f}{d:8.1%}")
    out = pd.concat([eq.rename('equity_new'), eq_leg.rename('equity_legacy'), tr_live, tr_v1, tr_old] + [v.rename(k) for k, v in books.items()], axis=1).loc[START:END]
    out.to_csv(os.path.join(RESULTS_DIR, f'headline_2026-09_daily_{which}.csv'))


if __name__ == '__main__':
    main()
