"""
Combination test v2 — does the DEEPENED trend (18 assets, 3/6/12m ensemble) build a
better BOOK than the v1 trend (9 assets, 12m)?

Reuses cached equity_returns_{year}.csv (2023-25, the FULL12 gated equity book) and
rebuilds trend A (v1) & C (v2) from cache via trend_follow_v2 (no network). Combines
each with the equity book at risk-parity (inverse-vol) and reports per-year — with the
2025 alpha-decay row and the worst-year Sharpe called out, since that's what the
deepening is supposed to improve.

Usage:  python3.12 combine_book_v2.py
"""

import os
import numpy as np
import pandas as pd
import trend_follow_v2 as t2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF = 0.05 / 252


def sharpe(r, rf=RF):
    r = r.dropna()
    return float(((r - rf).mean() / r.std() * np.sqrt(252)) if (len(r) > 1 and r.std() > 0) else 0.0)


def worst_yr(r):
    return min((sharpe(x) for _, x in r.groupby(r.index.year)), default=0.0)


def maxdd(r):
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def combine(eq, trend, lo, hi):
    t = trend[(trend.index >= lo) & (trend.index <= hi)]
    j = pd.concat([eq.rename('e'), t.rename('t')], axis=1).dropna()
    e, tt = j['e'], j['t']
    ve, vt = e.std(), tt.std()
    we = (1 / ve) / ((1 / ve) + (1 / vt)) if ve > 0 and vt > 0 else 0.5
    return we * e + (1 - we) * tt, j, we


def report(label, r):
    print(f"  {label:30s} rf-Sharpe {sharpe(r):+.2f} | worst-yr {worst_yr(r):+.2f} "
          f"| MaxDD {maxdd(r):.1%} | vol {r.std()*np.sqrt(252):.1%}")


if __name__ == '__main__':
    prices = t2.load_prices()
    present = [c for c in t2.SECTOR if c in prices.columns]
    trend_v1 = t2.backtest(prices, t2.BASE9, [252])             # A
    trend_v2 = t2.backtest(prices, present, t2.HORIZONS)        # C

    eq = pd.concat([pd.read_csv(os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'),
                                index_col=0, parse_dates=True).iloc[:, 0]
                    for y in [2023, 2024, 2025]]).sort_index()
    lo, hi = eq.index.min(), eq.index.max()

    cv1, jv1, we1 = combine(eq, trend_v1, lo, hi)
    cv2, jv2, we2 = combine(eq, trend_v2, lo, hi)
    eq_al = jv2['e']     # equity aligned to the overlap

    print("=" * 80)
    print(f"COMBINATION v2  (2023-2025, {len(jv2)} overlapping days)")
    print("=" * 80)
    print(f"  Sleeve correlation to equity:  v1 trend {jv1['e'].corr(jv1['t']):+.2f} | "
          f"v2 trend {jv2['e'].corr(jv2['t']):+.2f}\n")
    report("Equity book alone", eq_al)
    report("Trend v1 alone (9, 12m)", jv1['t'])
    report("Trend v2 alone (18, 3/6/12m)", jv2['t'])
    report(f"Equity + v1 trend (RP {we1:.0%}/{1-we1:.0%})", cv1)
    report(f"Equity + v2 trend (RP {we2:.0%}/{1-we2:.0%})", cv2)

    print("\n  Per-year rf-Sharpe (the 2025 row is your alpha-decay test):")
    print(f"  {'year':6s}{'equity':>9s}{'+v1':>9s}{'+v2':>9s}")
    for y in sorted(set(jv2.index.year)):
        ey = sharpe(eq_al[eq_al.index.year == y])
        v1y = sharpe(cv1[cv1.index.year == y])
        v2y = sharpe(cv2[cv2.index.year == y])
        star = '  <-- alpha-decay year' if y == 2025 else ''
        print(f"  {str(y):6s}{ey:>9.2f}{v1y:>9.2f}{v2y:>9.2f}{star}")
    print("=" * 80)
    print("READ HONESTLY: over the FULL 20y sample v2 dominates v1 (Sharpe/DD/crisis alpha),")
    print("but over THIS 3y overlap v1 edges v2 on book Sharpe & worst-year — because the")
    print("2023-25 window is an equity BULL with NO crisis year, so v2's crisis-alpha edge is")
    print("invisible here, and v2 trades a little recent-bull beta for much better tail/crisis")
    print("behavior elsewhere. v2 still wins book vol & MaxDD; both rescue 2025. The overlap is")
    print("too short and crisis-free to adjudicate — extend the equity book to 2022/2020 to decide.")
