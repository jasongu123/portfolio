"""
Trend sleeve breadth: 16 ETFs -> 46 ETFs — RESEARCH ONLY. Gate pre-registered 2026-09-16 before
the run. Does not touch trend_sleeve_live.py.

Hypothesis (mechanism: diversification, not a signal): institutional trend programs earn their
Sharpe advantage from breadth across 60-100 markets. The live sleeve trades 16 ETFs and lacks
whole asset classes: credit, inflation-linked and international bonds, individual grains and
industrial metals, natural gas and refined products, several G10 currencies, single-country
equities, international real estate. Adding them with the same rule should raise the sleeve's
Sharpe through more independent bets. Futures are unavailable to the account; ETFs stand in.

Additions (fixed now; each enters when it has WARMUP_DAYS of history, as the sleeve already
handles late starters; all disjoint from the equity engine's symbols):
  commodities  CORN WEAT SOYB CPER PPLT PALL UNG UGA BNO
  rates/credit TIP EDV LQD HYG EMB BWX IGOV
  currencies   FXC FXA FXF CEW
  equities     EWG EWU EWC EWA EWY EWT EWZ INDA FXI
  real estate  VNQI
Costs: the shared cost model as in V1_trail (2 bp fallback) for the liquid names; the thin set
{CORN WEAT SOYB CPER PPLT PALL UGA BNO CEW IGOV BWX VNQI} is charged 5 bp per unit of turnover
on top (a pre-stated liquidity penalty; a 10 bp sensitivity is reported).

Configurations (V1_trail rule: corrected conventions, per-asset vol targeting, 1/n, trailing
scalar; the 16-ETF line reproduces results/trend_v3_returns.csv V1_trail):
  BASE16     the live universe
  BREADTH46  all additions
  Blocks (informational): BASE16 + commodities only; + rates/credit only; + currencies only;
             + equities only; and BREADTH46 minus the single largest contributor (drop-one).

Pre-registered gate (rf-Sharpe, flat 5%, daily, full window 2008-01-01..latest):
  B1 BREADTH46 rf-Sharpe >= BASE16 + 0.10 and MaxDD not deeper.                      binding
  B2 not one period: the improvement (rf-Sharpe difference) is positive in at least two of
     2008-2013, 2014-2019, 2020-2026, and the last-two-years difference >= -0.05.     binding
  B3 breadth, not one instrument: the effective number of independent bets (from the
     correlation matrix of instrument P&L contributions) rises by >= 25%, and the drop-one
     variant keeps at least half of the improvement.                                  binding
  B4 book: walk-forward equity + BREADTH46 over 2020-2025 not worse than equity + BASE16 on
     rf-Sharpe, worst year and MaxDD.                                                 binding
  B5 costs: B1 still holds with the thin set at 10 bp.                                binding
  Informational: block contributions; per-year table; turnover; SPY correlation.
Decision: B1-B5 pass -> the universe change goes to the live sleeve through a versioned change
(paper first). Any fail -> not; blocks are not mined for a subset.
Trials: T-2026-09-16-43 (BREADTH46), -44 (the four blocks and drop-one as one diagnostic entry).
Usage: /usr/local/bin/python3 trend_breadth.py
"""

import os
import numpy as np
import pandas as pd

import trend_follow_v2 as t2
import trend_follow_v3 as v3
import book_backtest as bb
from trading_costs import CostModelConfig

RESULTS_DIR = v3.RESULTS_DIR
PX_NEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'trend_breadth_prices.csv')
FULL_START = v3.FULL_START
BLOCKS = {
    'commodities': ['CORN', 'WEAT', 'SOYB', 'CPER', 'PPLT', 'PALL', 'UNG', 'UGA', 'BNO'],
    'rates_credit': ['TIP', 'EDV', 'LQD', 'HYG', 'EMB', 'BWX', 'IGOV'],
    'currencies': ['FXC', 'FXA', 'FXF', 'CEW'],
    'equities': ['EWG', 'EWU', 'EWC', 'EWA', 'EWY', 'EWT', 'EWZ', 'INDA', 'FXI'],
    'real_estate': ['VNQI'],
}
THIN = {'CORN', 'WEAT', 'SOYB', 'CPER', 'PPLT', 'PALL', 'UGA', 'BNO', 'CEW', 'IGOV', 'BWX', 'VNQI'}
THIN_BPS = 5.0
SUBS = (('2008-2013', '2008-01-01', '2013-12-31'), ('2014-2019', '2014-01-01', '2019-12-31'), ('2020-2026', '2020-01-01', None))


def matched_prices(prices, cols):
    """Audit finding 12 (2026-10-11): every instrument ends on the last date ALL of them have a real
    price (no zero returns for instruments whose data stopped earlier), and an interior missing price
    carries the previous mark, so the move across a gap is booked when trading resumes instead of
    being dropped. Returns the matched frame and the common end date."""
    end = min(prices[c].last_valid_index() for c in cols)
    p = prices.loc[:end].copy()
    p[cols] = p[cols].ffill()
    return p, end


def check_marks(normalized, rets):
    """Refuse to run if any held position lacks a return (a missing mark)."""
    bad = normalized.ne(0) & rets.reindex(normalized.index)[normalized.columns].isna()
    if bad.values.any():
        raise ValueError(f"{int(bad.values.sum())} held position-days have no price: {bad.sum()[bad.sum() > 0].to_dict()}")


def run(prices, universe, thin_bps=THIN_BPS):
    """V1_trail rule on `universe`; returns net port series (with attrs) and per-asset contributions."""
    normalized, px, rets, _, month_ends = v3.build_weights(prices, universe, 'V2', t2.HORIZONS, legacy=False)
    check_marks(normalized, rets)
    port = v3.scale_and_cost(normalized, px, rets, scalar_mode='trailing', month_ends=month_ends)
    # reconstruct the daily scalar to get target weights and contributions (same recipe as rates_attribution)
    gross = (normalized * rets).sum(axis=1)
    turnover = normalized.diff().abs().sum(axis=1)
    legacy = gross - turnover * (t2.COST_BPS_PER_REBAL / 1e4)
    active = normalized.abs().sum(axis=1).gt(0)
    first = active.idxmax()
    vals = {}
    for d in month_ends:
        h = legacy.loc[first:d].tail(v3.SCALAR_LOOKBACK)
        vals[d] = min(t2.PORT_TARGET_VOL / (h.std() * v3.ANN), t2.MAX_LEVERAGE) if len(h) >= v3.SCALAR_MIN_HISTORY and h.std() > 0 else 1.0
    scal = pd.Series(vals).reindex(px.index, method='ffill').shift(1).fillna(1.0)
    target = normalized.mul(scal, axis=0)
    contrib = (target * rets).fillna(0.0)
    # liquidity penalty on the thin set, per unit of turnover in that asset
    thin_cols = [c for c in universe if c in THIN and c in target.columns]
    penalty = target[thin_cols].diff().abs().sum(axis=1) * thin_bps / 1e4 if thin_cols else 0.0
    net = (port - penalty.reindex(port.index).fillna(0.0)) if thin_cols else port
    net.attrs = dict(port.attrs)
    return net, contrib.loc[net.index]


def effective_bets(contrib, start=FULL_START):
    c = contrib.loc[start:]
    c = c.loc[:, c.abs().sum() > 0]
    C = c.corr().values
    w = np.ones(len(C)) / len(C)
    ev = np.linalg.eigvalsh(np.nan_to_num(C))
    ev = ev[ev > 1e-8]
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum())), len(C)          # entropy-based effective number


def stats(r):
    full = v3.window(r, FULL_START)
    yr = v3.yearly(full)
    return {'rf': v3.sharpe(full), 'norf': v3.sharpe(full, 0.0), 'mdd': v3.maxdd(full), 'vol': full.std() * v3.ANN,
            'worst': min(yr.values()), 'worst_yr': min(yr, key=yr.get), 'to': r.attrs.get('turnover_per_year', np.nan),
            'last2y': v3.sharpe(full.loc['2024-09-15':]), 'years': yr}


def main():
    prices = t2.load_prices()
    newpx = pd.read_csv(PX_NEW, index_col=0, parse_dates=True).sort_index()
    for c in newpx.columns:
        if c not in prices.columns:
            prices = prices.join(newpx[c], how='outer')
    prices = prices.sort_index()
    live16 = [c for c in v3.LIVE16 if c in prices.columns]
    adds = [c for b in BLOCKS.values() for c in b if c in prices.columns]
    prices, common_end = matched_prices(prices, live16 + adds)
    print(f"Matched data: every instrument ends {common_end.date()}; interior gaps carry the last mark")
    missing = [c for b in BLOCKS.values() for c in b if c not in prices.columns]
    spy = prices['SPY'].pct_change(fill_method=None)
    print(f"Base {len(live16)}; additions present {len(adds)}, missing {missing}")
    print("First price of additions: " + ", ".join(f"{c} {prices[c].first_valid_index().year}" for c in adds))

    series, contribs, rows = {}, {}, {}
    configs = {'BASE16': live16, 'BREADTH46': live16 + adds}
    for b, names in BLOCKS.items():
        configs[f'BASE16+{b}'] = live16 + [c for c in names if c in prices.columns]
    for name, uni in configs.items():
        r, c = run(prices, uni)
        series[name], contribs[name], rows[name] = r, c, stats(r)
    chk = pd.concat([series['BASE16'].rename('a'), pd.read_csv(os.path.join(RESULTS_DIR, 'trend_v3_returns.csv'), index_col=0, parse_dates=True)['V1_trail'].rename('b')], axis=1).dropna()
    print(f"CHECK BASE16 reproduces V1_trail: max |diff| {(chk.a - chk.b).abs().max():.2e}")
    # drop-one: largest contributor among additions in BREADTH46
    cb = contribs['BREADTH46'].loc[FULL_START:]
    add_contrib = cb[[c for c in adds if c in cb.columns]].mean() * 252
    top = add_contrib.idxmax()
    r_drop, c_drop = run(prices, [c for c in configs['BREADTH46'] if c != top])
    series['BREADTH46 minus ' + top], rows['BREADTH46 minus ' + top] = r_drop, stats(r_drop)
    r10, _ = run(prices, configs['BREADTH46'], thin_bps=10.0)
    series['BREADTH46 thin@10bp'], rows['BREADTH46 thin@10bp'] = r10, stats(r10)

    print(f"\nFull window {FULL_START}..{series['BASE16'].index.max().date()}; rf-Sharpe headline")
    print(f"{'config':28s}{'rf SR':>7s}{'no-rf':>7s}{'vol':>7s}{'MDD':>8s}{'worst':>7s}{'(yr)':>6s}{'TO/yr':>7s}{'last2y':>8s}  " + "".join(f"{y:>6d}" for y in range(2008, 2027)))
    for k, s in rows.items():
        print(f"{k:28s}{s['rf']:7.2f}{s['norf']:7.2f}{s['vol']:7.1%}{s['mdd']:8.1%}{s['worst']:7.2f}{s['worst_yr']:6d}{s['to']:7.1f}{s['last2y']:8.2f}  " + "".join(f"{s['years'].get(y, np.nan):6.2f}" for y in range(2008, 2027)))
    print("\nAnnualised contribution of additions (BREADTH46, from 2008): " + ", ".join(f"{k} {v:+.2%}" for k, v in add_contrib.sort_values(ascending=False).items()))
    n16, k16 = effective_bets(contribs['BASE16']); n46, k46 = effective_bets(contribs['BREADTH46'])
    print(f"Effective independent bets: BASE16 {n16:.1f} of {k16}; BREADTH46 {n46:.1f} of {k46} ({n46 / n16 - 1:+.0%})")

    b, a = rows['BASE16'], rows['BREADTH46']
    b1 = (a['rf'] >= b['rf'] + 0.10) and (a['mdd'] >= b['mdd'])
    diffs = []
    for name, s0, s1 in SUBS:
        x, y = v3.window(series['BREADTH46'], s0, s1), v3.window(series['BASE16'], s0, s1)
        diffs.append(v3.sharpe(x) - v3.sharpe(y))
        print(f"  {name}: BREADTH46 rf {v3.sharpe(x):.2f} vs BASE16 {v3.sharpe(y):.2f} ({diffs[-1]:+.2f})")
    b2 = (sum(d > 0 for d in diffs) >= 2) and (a['last2y'] - b['last2y'] >= -0.05)
    imp = a['rf'] - b['rf']
    drop = rows['BREADTH46 minus ' + top]
    b3 = (n46 / n16 - 1 >= 0.25) and ((drop['rf'] - b['rf']) >= 0.5 * imp if imp > 0 else False)
    eq = bb.load_equity()
    books = {}
    for name in ('BASE16', 'BREADTH46'):
        full = pd.concat([eq, series[name].rename('trend')], axis=1)
        bb.FULL_FRAME = full
        j = full.loc['2020-01-01':'2025-12-31'].dropna()
        books[name] = bb.book_returns(j, 'WF')
    yb, ya = bb.yearly(books['BASE16']), bb.yearly(books['BREADTH46'])
    b4 = (bb.sharpe(books['BREADTH46']) >= bb.sharpe(books['BASE16'])) and (min(ya.values()) >= min(yb.values())) and (bb.maxdd(books['BREADTH46']) >= bb.maxdd(books['BASE16']))
    b5 = (rows['BREADTH46 thin@10bp']['rf'] >= b['rf'] + 0.10) and (rows['BREADTH46 thin@10bp']['mdd'] >= b['mdd'])

    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if b1 else 'fail'}  B1 rf {b['rf']:.2f} -> {a['rf']:.2f} ({imp:+.2f}, need +0.10); MDD {b['mdd']:.1%} -> {a['mdd']:.1%}")
    print(f"  {'PASS' if b2 else 'fail'}  B2 sub-period improvements positive in {sum(d > 0 for d in diffs)}/3; last-2y difference {a['last2y'] - b['last2y']:+.2f}")
    print(f"  {'PASS' if b3 else 'fail'}  B3 effective bets {n16:.1f} -> {n46:.1f} ({n46 / n16 - 1:+.0%}, need +25%); drop-one ({top}) keeps {(drop['rf'] - b['rf']):+.2f} of {imp:+.2f}")
    print(f"  {'PASS' if b4 else 'fail'}  B4 WF book rf {bb.sharpe(books['BASE16']):.2f} -> {bb.sharpe(books['BREADTH46']):.2f}; worst {min(yb.values()):+.2f} -> {min(ya.values()):+.2f}; MDD {bb.maxdd(books['BASE16']):.1%} -> {bb.maxdd(books['BREADTH46']):.1%}")
    print(f"  {'PASS' if b5 else 'fail'}  B5 thin set at 10 bp: rf {rows['BREADTH46 thin@10bp']['rf']:.2f} (need >= {b['rf'] + 0.10:.2f})")
    print(f"  info  book no-rf {bb.sharpe(books['BASE16'], 0):.2f} -> {bb.sharpe(books['BREADTH46'], 0):.2f}")
    print(f"\nVERDICT: {'ADOPT the 46-ETF universe for the live sleeve (versioned change, paper first)' if all([b1, b2, b3, b4, b5]) else 'not adopted'}")
    pd.DataFrame(series).to_csv(os.path.join(RESULTS_DIR, 'trend_breadth_returns.csv'))
    pd.DataFrame(rows).T.drop(columns='years').to_csv(os.path.join(RESULTS_DIR, 'trend_breadth_summary.csv'))
    print("Saved results/trend_breadth_returns.csv, _summary.csv")


if __name__ == '__main__':
    main()
