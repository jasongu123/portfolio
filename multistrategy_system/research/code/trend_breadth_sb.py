"""
Trend sleeve breadth, sector-balanced — RESEARCH ONLY. Gate pre-registered 2026-09-16 before the
run; hypothesis taken from the TREND_BREADTH.md diagnostic (T-43/44), so this is a second look at
the same data with a changed construction, and is registered as such.

Hypothesis: equal weight across 46 names let nine correlated country funds and four losing
currency funds crowd out the base. Trend programs handle breadth with sector balance: equal risk
per asset class, equal within. Under that weighting, adding credit, inflation-linked and
international bonds, and the commodity and equity funds inside their classes should add
independent bets without diluting the base.

Universe SB42 = the live 16 + the T-43 additions minus the currency block (CEW FXC FXA FXF),
which was pre-stated for exclusion by the block diagnostic. No other instrument is dropped.
Sectors (fixed now): Equity {EFA EEM EWJ EWG EWU EWC EWA EWY EWT EWZ INDA FXI}, Rates {SHY IEF TLT
TIP EDV BWX IGOV}, Credit {LQD HYG EMB}, Commodity {GLD SLV DBC USO DBA CORN WEAT SOYB CPER PPLT
PALL UNG UGA BNO}, FX {UUP FXE FXY FXB}, RealEstate {VNQ VNQI}.
Weighting: each active sector gets 1/S of the book, split equally among its active instruments;
everything else (signal, per-asset vol targeting, trailing scalar, costs, thin-set penalty) is
the T-43 setup. Attribution rows: BASE16 with 1/n (live), BASE16 sector-balanced (SB16), SB42.

Pre-registered gate (rf-Sharpe, flat 5%, daily, 2008-01-01..latest):
  S1 SB42 rf-Sharpe >= BASE16 + 0.10 and MaxDD not deeper; AND SB42 >= SB16 + 0.05 (breadth
     must add on the same weighting; a gain from the weighting alone is a different change). binding
  S2 SB42 - BASE16 positive in at least two of 2008-2013 / 2014-2019 / 2020-2026; last-2y
     difference >= -0.05.                                                                  binding
  S3 effective bets up >= 25% vs BASE16; drop-one (largest additive contributor) keeps >= half
     of the SB42 - BASE16 gain.                                                            binding
  S4 walk-forward equity + SB42 (2020-2025) not worse than equity + BASE16 on rf-Sharpe, worst
     year, MaxDD.                                                                          binding
  S5 S1 still holds with the thin set at 10 bp.                                            binding
Decision: S1-S5 pass -> SB42 goes to the live sleeve as a versioned change (paper first).
Trials: T-2026-09-16-45 (SB42), -46 (SB16, attribution).
Usage: /usr/local/bin/python3 trend_breadth_sb.py
"""

import os
import numpy as np
import pandas as pd

import trend_follow_v2 as t2
import trend_follow_v3 as v3
import book_backtest as bb
import trend_breadth as tb

RESULTS_DIR = v3.RESULTS_DIR
FULL_START = v3.FULL_START
SECTORS = {
    'Equity': ['EFA', 'EEM', 'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWY', 'EWT', 'EWZ', 'INDA', 'FXI'],
    'Rates': ['SHY', 'IEF', 'TLT', 'TIP', 'EDV', 'BWX', 'IGOV'],
    'Credit': ['LQD', 'HYG', 'EMB'],
    'Commodity': ['GLD', 'SLV', 'DBC', 'USO', 'DBA', 'CORN', 'WEAT', 'SOYB', 'CPER', 'PPLT', 'PALL', 'UNG', 'UGA', 'BNO'],
    'FX': ['UUP', 'FXE', 'FXY', 'FXB'],
    'RealEstate': ['VNQ', 'VNQI'],
}
SECTOR_OF = {c: s for s, cs in SECTORS.items() for c in cs}


def run_sb(prices, universe, thin_bps=tb.THIN_BPS, balanced=True):
    normalized, px, rets, _, month_ends = v3.build_weights(prices, universe, 'V2', t2.HORIZONS, legacy=False)
    tb.check_marks(normalized, rets)
    n_daily = normalized.ne(0).sum(axis=1).replace(0, 1)
    raw = normalized.mul(n_daily, axis=0)                      # undo 1/n: sig * size per asset
    if balanced:
        w = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
        active = raw.ne(0)
        sec_cols = {s: [c for c in cs if c in raw.columns] for s, cs in SECTORS.items()}
        k = pd.DataFrame({s: active[cols].sum(axis=1) for s, cols in sec_cols.items() if cols})
        S = k.gt(0).sum(axis=1).replace(0, 1)
        for s, cols in sec_cols.items():
            if not cols:
                continue
            ks = k[s].replace(0, np.nan)
            w[cols] = raw[cols].div(ks, axis=0).div(S, axis=0).fillna(0.0)
        normalized = w
    port = v3.scale_and_cost(normalized, px, rets, scalar_mode='trailing', month_ends=month_ends)
    gross = (normalized * rets).sum(axis=1)
    turnover = normalized.diff().abs().sum(axis=1)
    legacy = gross - turnover * (t2.COST_BPS_PER_REBAL / 1e4)
    first = normalized.abs().sum(axis=1).gt(0).idxmax()
    vals = {}
    for d in month_ends:
        h = legacy.loc[first:d].tail(v3.SCALAR_LOOKBACK)
        vals[d] = min(t2.PORT_TARGET_VOL / (h.std() * v3.ANN), t2.MAX_LEVERAGE) if len(h) >= v3.SCALAR_MIN_HISTORY and h.std() > 0 else 1.0
    scal = pd.Series(vals).reindex(px.index, method='ffill').shift(1).fillna(1.0)
    target = normalized.mul(scal, axis=0)
    contrib = (target * rets).fillna(0.0)
    thin_cols = [c for c in universe if c in tb.THIN and c in target.columns]
    net = port
    if thin_cols:
        penalty = target[thin_cols].diff().abs().sum(axis=1) * thin_bps / 1e4
        net = port - penalty.reindex(port.index).fillna(0.0)
        net.attrs = dict(port.attrs)
    return net, contrib.loc[net.index]


def main():
    prices = t2.load_prices()
    newpx = pd.read_csv(tb.PX_NEW, index_col=0, parse_dates=True).sort_index()
    for c in newpx.columns:
        if c not in prices.columns:
            prices = prices.join(newpx[c], how='outer')
    prices = prices.sort_index()
    live16 = [c for c in v3.LIVE16 if c in prices.columns]
    adds = [c for b, names in tb.BLOCKS.items() if b != 'currencies' for c in names if c in prices.columns]
    sb42 = live16 + adds
    prices, common_end = tb.matched_prices(prices, sb42)
    print(f"Matched data: every instrument ends {common_end.date()}; interior gaps carry the last mark")
    print(f"SB42 = {len(sb42)} instruments; sectors " + ", ".join(f"{s} {sum(1 for c in sb42 if SECTOR_OF.get(c) == s)}" for s in SECTORS))
    series, contribs, rows = {}, {}, {}
    for name, uni, bal in (('BASE16 (live, 1/n)', live16, False), ('SB16', live16, True), ('SB42', sb42, True)):
        r, c = run_sb(prices, uni, balanced=bal)
        series[name], contribs[name], rows[name] = r, c, tb.stats(r)
    chk = pd.concat([series['BASE16 (live, 1/n)'].rename('a'), pd.read_csv(os.path.join(RESULTS_DIR, 'trend_v3_returns.csv'), index_col=0, parse_dates=True)['V1_trail'].rename('b')], axis=1).dropna()
    print(f"CHECK BASE16 reproduces V1_trail: max |diff| {(chk.a - chk.b).abs().max():.2e}")
    c42 = contribs['SB42'].loc[FULL_START:]
    add_contrib = c42[[c for c in adds if c in c42.columns]].mean() * 252
    top = add_contrib.idxmax()
    r_drop, _ = run_sb(prices, [c for c in sb42 if c != top])
    series[f'SB42 minus {top}'], rows[f'SB42 minus {top}'] = r_drop, tb.stats(r_drop)
    r10, _ = run_sb(prices, sb42, thin_bps=10.0)
    series['SB42 thin@10bp'], rows['SB42 thin@10bp'] = r10, tb.stats(r10)

    print(f"\nFull window {FULL_START}..{series['SB42'].index.max().date()}; rf-Sharpe headline")
    print(f"{'config':22s}{'rf SR':>7s}{'no-rf':>7s}{'vol':>7s}{'MDD':>8s}{'worst':>7s}{'(yr)':>6s}{'TO/yr':>7s}{'last2y':>8s}  " + "".join(f"{y:>6d}" for y in range(2008, 2027)))
    for k, s in rows.items():
        print(f"{k:22s}{s['rf']:7.2f}{s['norf']:7.2f}{s['vol']:7.1%}{s['mdd']:8.1%}{s['worst']:7.2f}{s['worst_yr']:6d}{s['to']:7.1f}{s['last2y']:8.2f}  " + "".join(f"{s['years'].get(y, np.nan):6.2f}" for y in range(2008, 2027)))
    n16, _ = tb.effective_bets(contribs['BASE16 (live, 1/n)']); n42, _ = tb.effective_bets(contribs['SB42'])
    sec_contrib = {s: c42[[c for c in cs if c in c42.columns]].sum(axis=1).mean() * 252 for s, cs in SECTORS.items()}
    print("SB42 sector contributions /yr: " + ", ".join(f"{s} {v:+.2%}" for s, v in sec_contrib.items()))
    print(f"Effective bets BASE16 {n16:.1f} -> SB42 {n42:.1f} ({n42 / n16 - 1:+.0%}); largest additive contributor {top} {add_contrib.max():+.2%}/yr")

    b, s16, a = rows['BASE16 (live, 1/n)'], rows['SB16'], rows['SB42']
    imp = a['rf'] - b['rf']
    s1 = (imp >= 0.10) and (a['mdd'] >= b['mdd']) and (a['rf'] >= s16['rf'] + 0.05)
    diffs = []
    for name, d0, d1 in tb.SUBS:
        x, y = v3.window(series['SB42'], d0, d1), v3.window(series['BASE16 (live, 1/n)'], d0, d1)
        diffs.append(v3.sharpe(x) - v3.sharpe(y))
        print(f"  {name}: SB42 rf {v3.sharpe(x):.2f} vs BASE16 {v3.sharpe(y):.2f} ({diffs[-1]:+.2f}); SB16 {v3.sharpe(v3.window(series['SB16'], d0, d1)):.2f}")
    s2 = (sum(d > 0 for d in diffs) >= 2) and (a['last2y'] - b['last2y'] >= -0.05)
    drop = rows[f'SB42 minus {top}']
    s3 = (n42 / n16 - 1 >= 0.25) and (imp > 0 and (drop['rf'] - b['rf']) >= 0.5 * imp)
    eq = bb.load_equity()
    books = {}
    for name in ('BASE16 (live, 1/n)', 'SB42'):
        full = pd.concat([eq, series[name].rename('trend')], axis=1)
        bb.FULL_FRAME = full
        books[name] = bb.book_returns(full.loc['2020-01-01':'2025-12-31'].dropna(), 'WF')
    yb, ya = bb.yearly(books['BASE16 (live, 1/n)']), bb.yearly(books['SB42'])
    s4 = (bb.sharpe(books['SB42']) >= bb.sharpe(books['BASE16 (live, 1/n)'])) and (min(ya.values()) >= min(yb.values())) and (bb.maxdd(books['SB42']) >= bb.maxdd(books['BASE16 (live, 1/n)']))
    t10 = rows['SB42 thin@10bp']
    s5 = (t10['rf'] >= b['rf'] + 0.10) and (t10['mdd'] >= b['mdd']) and (t10['rf'] >= s16['rf'] + 0.05)
    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if s1 else 'fail'}  S1 rf BASE16 {b['rf']:.2f} -> SB42 {a['rf']:.2f} ({imp:+.2f}, need +0.10); SB16 {s16['rf']:.2f} (SB42 must be >= SB16 + 0.05); MDD {b['mdd']:.1%} -> {a['mdd']:.1%}")
    print(f"  {'PASS' if s2 else 'fail'}  S2 sub-period gains positive in {sum(d > 0 for d in diffs)}/3; last-2y difference {a['last2y'] - b['last2y']:+.2f}")
    print(f"  {'PASS' if s3 else 'fail'}  S3 effective bets {n16:.1f} -> {n42:.1f} ({n42 / n16 - 1:+.0%}); drop-one ({top}) keeps {(drop['rf'] - b['rf']):+.2f} of {imp:+.2f}")
    print(f"  {'PASS' if s4 else 'fail'}  S4 WF book rf {bb.sharpe(books['BASE16 (live, 1/n)']):.2f} -> {bb.sharpe(books['SB42']):.2f}; worst {min(yb.values()):+.2f} -> {min(ya.values()):+.2f}; MDD {bb.maxdd(books['BASE16 (live, 1/n)']):.1%} -> {bb.maxdd(books['SB42']):.1%}")
    print(f"  {'PASS' if s5 else 'fail'}  S5 thin set at 10 bp: rf {t10['rf']:.2f}")
    print(f"  info  book no-rf {bb.sharpe(books['BASE16 (live, 1/n)'], 0):.2f} -> {bb.sharpe(books['SB42'], 0):.2f}")
    print(f"\nVERDICT: {'ADOPT SB42 for the live sleeve (versioned change, paper first)' if all([s1, s2, s3, s4, s5]) else 'not adopted'}")
    pd.DataFrame(series).to_csv(os.path.join(RESULTS_DIR, 'trend_breadth_sb_returns.csv'))
    pd.DataFrame(rows).T.drop(columns='years').to_csv(os.path.join(RESULTS_DIR, 'trend_breadth_sb_summary.csv'))


if __name__ == '__main__':
    main()
