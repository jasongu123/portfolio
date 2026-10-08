"""
Intramonth sector cycle, legs 2+3 only — RESEARCH ONLY. Gate pre-registered 2026-09-14 before the run.

Hypothesis (mechanism first): leg 1 (ten-day sector momentum before month-end, D-10..D-5) is the
part of the strategy that (a) is a momentum exposure, so it overlaps the trend sleeve, and (b) rests
on pre-month-end institutional liquidation that has been arbitraged away since 2012. Legs 2 and 3
(turn-of-month flow on D0..D+1 and its reversal on D+1..D+3) are flow effects nothing else in the
book trades. Removing leg 1 should therefore raise orthogonality to the equity and trend sleeves,
cut turnover by about 40%, and leave a series that pays in the months the book loses.

Disclosure: T-2026-09-14-21 already looked at an approximation of this configuration (leg-1 days
zeroed in the Strat2 series) on the 2020-2025 book, so G4 below is not a blind test. G2's
2008-2019 part uses the trend sleeve alone as the book proxy and was NOT examined before this run.

Configuration: intramonth_sector.run(px, 'Strat2', legs=(2, 3), cost_bps=1). Nothing else changes.

Pre-registered gate (all on the legs-2+3 series at 1 bp):
  G1 orthogonality: monthly correlation with the equity sleeve and with trend V1_trail over
     2020-2025 each < 0.10 (signed; negative is fine).                                 binding
  G2 pays when the book loses: mean monthly return > 0 in the worst-quintile months of
     (a) the trend sleeve alone over 2008-2019 [blind], and (b) the walk-forward equity+trend
     book over 2020-2025.                                                              binding
  G3 the alpha survives: full-sample net Sharpe (no-rf, daily) >= 0.40; net Sharpe over
     2012-01..latest >= 0.30; each of leg 2 and leg 3 positive in each of 1999-2011,
     2012-2019, 2020-2026.                                                             binding
  G4 book: walk-forward inverse-vol equity+trend+L23 over 2020-2025 has rf-Sharpe >= the
     equity+trend book on the same convention and MaxDD not deeper.                    binding
  Informational: worst calendar year of the book (known from T-21 to be 2024, negative); 0 and
     2 bp results; turnover per month; active-day volatility and the inverse-vol weight it implies.
Verdict: promote to replace the all-legs configuration in the paper sleeve only if G1-G4 all pass.

Trial registry: T-2026-09-14-22.
Usage: /usr/local/bin/python3 intramonth_l23.py
"""

import os
import numpy as np
import pandas as pd

import intramonth_sector as ims
import book_backtest as bb
import trend_follow_v2 as t2

RESULTS_DIR = ims.RESULTS_DIR
PERIODS = (('1999-2011', '1999-01-01', '2011-12-31'), ('2012-2019', '2012-01-01', '2019-12-31'),
           ('2020-2026', '2020-01-01', '2026-12-31'))


def win(s, a, b=None):
    return s.loc[a:b] if b else s.loc[a:]


def main():
    px = ims.load_prices()
    full_net, full_gross, full_leg, _ = ims.run(px, 'Strat2', legs=(1, 2, 3))
    net, gross, leg, w = ims.run(px, 'Strat2', legs=(2, 3))
    net0 = ims.run(px, 'Strat2', legs=(2, 3), cost_bps=0.0)[0]
    net2 = ims.run(px, 'Strat2', legs=(2, 3), cost_bps=2.0)[0]
    leg2 = ims.run(px, 'Strat2', legs=(2,))[0]
    leg3 = ims.run(px, 'Strat2', legs=(3,))[0]

    print(f"Legs 2+3 at 1 bp, {net.index.min().date()}..{net.index.max().date()}: "
          f"net Sharpe {ims.sharpe(net):.2f} (0 bp {ims.sharpe(net0):.2f}, 2 bp {ims.sharpe(net2):.2f}) | "
          f"ann {ims.ann_ret(net):+.2%} | vol {net.std() * ims.ANN:.1%} | MDD {ims.maxdd(net):.1%}")
    print(f"All legs at 1 bp (reference):              net Sharpe {ims.sharpe(full_net):.2f} | ann {ims.ann_ret(full_net):+.2%} "
          f"| vol {full_net.std() * ims.ANN:.1%} | MDD {ims.maxdd(full_net):.1%}")
    to_full = (w.diff().abs().sum(axis=1)).resample('ME').sum().mean()
    print(f"Turnover per month (sum |dw|): L23 {to_full:.2f}x  vs all legs "
          f"{ims.run(px, 'Strat2')[3].diff().abs().sum(axis=1).resample('ME').sum().mean():.2f}x")
    active = leg.isin(['L2', 'L3'])
    print(f"Days active {active.mean():.1%}; vol on active days {net[active].std() * ims.ANN:.1%} vs series vol "
          f"{net.std() * ims.ANN:.1%}")

    print("\nPer period, net Sharpe at 1 bp:")
    print(f"{'':12s}{'L23':>8s}{'leg2':>8s}{'leg3':>8s}{'all':>8s}")
    leg_ok = True
    for name, a, b in PERIODS:
        s2, s3 = ims.sharpe(win(leg2, a, b)), ims.sharpe(win(leg3, a, b))
        leg_ok &= (s2 > 0) and (s3 > 0)
        print(f"{name:12s}{ims.sharpe(win(net, a, b)):8.2f}{s2:8.2f}{s3:8.2f}{ims.sharpe(win(full_net, a, b)):8.2f}")
    yr = net.groupby(net.index.year).apply(ims.sharpe)
    print("Per-year net Sharpe L23: " + "  ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))

    # ---- book pieces
    prices = t2.load_prices()
    spy = prices['SPY'].pct_change(fill_method=None)
    eq, tr = bb.load_equity(), bb.load_trend()
    l23 = net.rename('intramonth_L23')
    bb.FULL_FRAME = pd.concat([eq, tr, l23], axis=1)

    # G1
    j = pd.concat([eq, tr, l23], axis=1).loc['2020':'2025'].dropna()
    mo = j.resample('ME').sum()
    c_eq, c_tr = mo['intramonth_L23'].corr(mo['equity']), mo['intramonth_L23'].corr(mo['trend'])
    g1 = (c_eq < 0.10) and (c_tr < 0.10)

    # G2a blind: trend alone 2008-2019
    j2 = pd.concat([tr, l23], axis=1).loc['2008':'2019'].dropna().resample('ME').sum()
    q = j2['trend'].quantile(0.2)
    bad_a = j2[j2['trend'] <= q]['intramonth_L23']
    # G2b: WF equity+trend book 2020-2025
    book2 = bb.book_returns(j[['equity', 'trend']], 'WF')
    j3 = pd.concat([book2.rename('book2'), j['intramonth_L23']], axis=1).resample('ME').sum()
    q3 = j3['book2'].quantile(0.2)
    bad_b = j3[j3['book2'] <= q3]['intramonth_L23']
    g2 = (bad_a.mean() > 0) and (bad_b.mean() > 0)

    # G3
    sr_full, sr_2012 = ims.sharpe(net), ims.sharpe(win(net, '2012-01-01'))
    g3 = (sr_full >= 0.40) and (sr_2012 >= 0.30) and leg_ok

    # G4
    book3 = bb.book_returns(j[['equity', 'trend', 'intramonth_L23']], 'WF')
    W = bb.wf_weights(j[['equity', 'trend', 'intramonth_L23']], bb.FULL_FRAME).dropna()
    s2, s3_ = bb.sharpe(book2), bb.sharpe(book3)
    d2, d3 = bb.maxdd(book2), bb.maxdd(book3)
    g4 = (s3_ >= s2) and (d3 >= d2)
    y2, y3 = bb.yearly(book2), bb.yearly(book3)
    vm = book3 * (book2.std() / book3.std())

    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if g1 else 'fail'}  G1 monthly corr with equity {c_eq:+.2f}, trend {c_tr:+.2f} (< 0.10 each)")
    print(f"  {'PASS' if g2 else 'fail'}  G2 mean monthly return in bad months: trend-alone 2008-2019 [blind] {bad_a.mean() * 100:+.2f}% "
          f"(n={len(bad_a)}, hit {(bad_a > 0).mean():.0%}); WF book 2020-2025 {bad_b.mean() * 100:+.2f}% (n={len(bad_b)}, hit {(bad_b > 0).mean():.0%})")
    print(f"  {'PASS' if g3 else 'fail'}  G3 full Sharpe {sr_full:.2f} (>= 0.40); 2012+ {sr_2012:.2f} (>= 0.30); legs 2 and 3 positive in all periods: {leg_ok}")
    print(f"  {'PASS' if g4 else 'fail'}  G4 WF book rf-Sharpe {s2:.2f} -> {s3_:.2f}; MDD {d2:.1%} -> {d3:.1%}")
    print(f"  info  book no-rf {bb.sharpe(book2, 0):.2f} -> {bb.sharpe(book3, 0):.2f}; vol {book2.std() * bb.ANN:.1%} -> {book3.std() * bb.ANN:.1%}; "
          f"worst year {min(y2, key=y2.get)} {min(y2.values()):+.2f} -> {min(y3, key=y3.get)} {min(y3.values()):+.2f}; "
          f"vol-matched rf {bb.sharpe(vm):.2f} MDD {bb.maxdd(vm):.1%}; mean WF weights {W.mean().round(2).to_dict()}")
    print("  info  per-year book rf-Sharpe: " + "  ".join(f"{y}: {y2[y]:+.2f} -> {y3[y]:+.2f}" for y in sorted(y2)))
    verdict = all([g1, g2, g3, g4])
    print(f"\nVERDICT: {'PROMOTE legs 2+3 to replace all-legs in the paper sleeve' if verdict else 'not promoted'}")

    out = pd.concat([net.rename('l23_net'), leg.rename('leg')], axis=1)
    out.to_csv(os.path.join(RESULTS_DIR, 'intramonth_l23_daily.csv'))
    print("Saved results/intramonth_l23_daily.csv")


if __name__ == '__main__':
    main()
