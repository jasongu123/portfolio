"""
Headline book for the RESTRUCTURED system (2026-10-09): the three sleeves that run live since the
October 2026 restructure, combined the way the existing headline is (headline_rebuild.py).

Sleeves (daily net returns, each on its own capital):
  trend   cross-asset trend as live: 16 ETFs, live conventions, constant scalar 2.95, shared cost
          model (headline_rebuild.trend_as_live)
  etf     ETF intraday momentum on the paper's rules as deployed (split_trial_v2 E0: 30-minute
          decisions, band x1, no HAR/VIX/ML/RSI, 0.5 bp a side), SPY/QQQ/IWM, corrected engine
  stocks  stock swing book only (split_trial_v2 S1: 21 names, 1 bp, no mean reversion, no gate)
Window: 2020-01-02 .. 2025-12-15 (the end of the corrected engine runs).

Books, fixed before running:
  MAIN   walk-forward inverse-volatility weights across the three sleeves (trailing 252 days,
         re-estimated at each month end, equal weight until 63 days of history; book_backtest.wf_weights)
  B      the old headline's two-sleeve shape with the new parts: one equity engine = 50/50 etf +
         stocks capital, then WF inverse-vol with trend
  OLD    the current headline recomputed on the same window: audit-corrected old engine + trend
  BASKET diagnostic for the open book-level question: MAIN with the stock sleeve replaced by an
         equal-weight basket of the same 21 stocks (held at the swing book's average exposure)
A correction of the headline, not a trial: no strategy rule is chosen here.
Limits: the live stock sleeve weights its 12 breadth names at half size; S1 weights all 21 equally.

Usage: python3 headline_restructured.py     (writes results/headline_restructured/)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import book_backtest as bb  # noqa: E402
import headline_rebuild as hr  # noqa: E402
import split_trial as st  # noqa: E402
import trend_follow_v2 as t2  # noqa: E402
import trend_follow_v3 as v3  # noqa: E402

OUT = os.path.join(HERE, 'results', 'headline_restructured')
START, END = '2020-01-01', '2025-12-15'
YEARS = list(range(2020, 2026))


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    prices = t2.load_prices()
    live16 = [c for c in v3.LIVE16 if c in prices.columns]
    trend = hr.trend_as_live(prices, live16).rename('trend')
    etf = st.load('E0')[0].rename('etf')
    R, _, SW, EQ = st.load('S1')
    stocks = R.rename('stocks')
    expo, rets = st.exposure_from_swing(SW, EQ, st.STOCKS)
    basket = (float(expo.mean()) * rets[st.STOCKS].mean(axis=1).reindex(stocks.index).fillna(0.0)).rename('stocks')
    old_eq = hr.load_equity('audit').rename('equity')
    spy = prices['SPY'].pct_change(fill_method=None)
    win = lambda s: s.loc[START:END]

    def wf(frame_cols):
        full = pd.concat(frame_cols, axis=1)
        bb.FULL_FRAME = full
        j = full.loc[START:END].dropna()
        return bb.book_returns(j, 'WF'), bb.wf_weights(j, full)

    main_book, W = wf([etf, stocks, trend])
    engine = (0.5 * etf + 0.5 * stocks).rename('equity')
    book_b, _ = wf([engine, trend])
    old_book, _ = wf([old_eq, trend])
    basket_book, _ = wf([etf, basket, trend])

    hdr = "| Series | Sharpe rf | Sharpe no-rf | worst year (rf) | MaxDD | vol | CAGR | " + " | ".join(map(str, YEARS)) + " |"
    sep = "|" + "---|" * (7 + len(YEARS))
    emit(f"Restructured book, {START} .. {END}; rf = flat 5%; MaxDD from initial capital; years = rf Sharpe\n")
    emit("SLEEVES")
    emit(hdr); emit(sep)
    for k, s in (('Trend as live (16 ETFs)', trend), ('ETF intraday momentum, paper rules', etf),
                 ('Stock swing book (21 names)', stocks), ('Equal-weight basket of the 21 at the same average exposure', basket),
                 ('Old equity engine (audit-corrected, pre-restructure)', old_eq)):
        emit(hr.row(k, hr.stats(win(s)), YEARS))
    j = pd.concat([win(etf), win(stocks), win(trend)], axis=1).dropna()
    emit("\nDaily correlations: " + ", ".join(f"{a}~{b} {j[a].corr(j[b]):+.2f}" for a, b in (('etf', 'stocks'), ('etf', 'trend'), ('stocks', 'trend')))
         + f"; book~SPY {main_book.corr(spy.reindex(main_book.index)):+.2f}")
    emit("\nBOOKS")
    emit(hdr); emit(sep)
    books = {'MAIN: WF inverse-vol, trend + ETF + stocks': main_book,
             'B: equity engine (50/50 ETF + stocks) + trend, WF': book_b,
             'OLD headline on the same window: old engine + trend, WF': old_book,
             'BASKET (diagnostic): trend + ETF + equal-weight basket, WF': basket_book}
    for k, s in books.items():
        emit(hr.row(k, hr.stats(s), YEARS))
    emit(f"\nMAIN walk-forward weights, mean: " + ", ".join(f"{c} {v:.2f}" for c, v in W.mean().items())
         + "; range: " + ", ".join(f"{c} {W[c].min():.2f}-{W[c].max():.2f}" for c in W.columns))
    last24 = main_book[main_book.index > main_book.index[-1] - pd.DateOffset(months=24)]
    emit(f"MAIN last 24 months: no-rf Sharpe {v3.sharpe(last24, 0.0):.2f}, rf {v3.sharpe(last24):.2f}")
    pd.DataFrame({'trend': win(trend), 'etf': win(etf), 'stocks': win(stocks), 'basket': win(basket),
                  **{k.split(':')[0]: v for k, v in books.items()}}).to_csv(os.path.join(OUT, 'daily.csv'))
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
