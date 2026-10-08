"""
Volatility-forecast sizing in the trend sleeve — RESEARCH ONLY. Trial T-2026-10-03-94, pre-registered
2026-10-03 before any sized portfolio was computed. Nothing here touches trend_sleeve_live.py.

Question: the trend sleeve sizes each ETF by 10% / (60-day trailing volatility). On fresh ETF data the
ridge volatility model (19 inputs, VOL_FORECAST.md, T-81) forecast next month's volatility ~10% better
than HAR (QLIKE) in 12 of 15 years, but missed its significance bar (t 2.16 < 2.24). Does sizing with
that forecast make the sleeve hit its risk target more accurately, without costing return?

Setup: trend_follow_v3 V1_trail exactly (live 16 ETFs, 3/6/12-month mean-sign signal, per-asset size
min(10% / vol, 3), 1/n, trailing 504-day portfolio scalar to 12%, 2 bp execution, shared cost model,
rebalanced on the last trading day of each month). The ONLY change is the vol in the per-asset size:
  BASE   60-day trailing realised volatility (the live rule)
  RIDGE  sqrt of the ridge forecast F_M1 of next month's variance, made at that month-end from data
         through that close (results/vol_forecast_etf/predictions.pkl: walk-forward, expanding window,
         holdout level correction; generated 2026-10-01 for T-81, before this trial)
  Months or ETFs without a forecast use the trailing vol in both (identical sizes), so only forecast
  months can differ. Context only, no decision: HAR (F_B1) and LightGBM (F_M2) sized the same way.
Window: 2012-02-01 (first forecast applies) .. 2026-08-31 (last forecast month).

Pre-registered decision, RIDGE vs BASE over the window:
  Z1 risk-targeting accuracy: per asset-month, e = (log(realised next-month vol / vol used for
     sizing))^2; the monthly mean of e across the 16 ETFs is at least 10% lower for RIDGE on average,
     with paired t (by month, Newey-West 3 lags) >= 2.0
  Z2 no-rf net Sharpe not lower than BASE's by more than 0.03
  Z3 no-rf net Sharpe over the last 24 months not lower than BASE's by more than 0.10
  Z4 maximum drawdown not deeper than BASE's by more than 1.0 percentage point
  PASS (all four) -> candidate to change the live sleeve's sizing (a separate, switch-guarded change).
Reported, no decision: rf Sharpe, 2020-2025 window, calendar years, turnover, the spread of monthly
realised portfolio volatility around its 12% target, HAR and LightGBM sizing.
Known limit: the forecast file holds only rows whose next month had >= 15 returns (an ETF's survival
for one month), immaterial for these 16 long-lived ETFs.

Usage: python3 vol_sizing_trial.py      (writes results/vol_sizing/)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trend_follow_v2 as t2  # noqa: E402
import trend_follow_v3 as t3  # noqa: E402

OUT = os.path.join(HERE, 'results', 'vol_sizing')
PRED = os.path.join(HERE, 'results', 'vol_forecast_etf', 'predictions.pkl')
START, END = '2012-02-01', '2026-08-31'
ANN = np.sqrt(252)
RF = 0.05 / 252
GATE = {'z1_gain': 0.10, 'z1_t': 2.0, 'z2': 0.03, 'z3': 0.10, 'z4': 0.01}


def forecast_table(col):
    p = pd.read_pickle(PRED)
    p = p[p.symbol.isin(t3.LIVE16)]
    return {(pd.Period(d, 'M'), s): float(np.sqrt(f)) for d, s, f in zip(p.date, p.symbol, p[col]) if np.isfinite(f) and f > 0}


def run(prices, table=None):
    """V1_trail with the per-asset vol replaced by table[(month, symbol)] when present. Returns the
    net daily returns and the vol used per (month-end, symbol)."""
    used = {}
    orig = t3._size

    def size(rets, c, d):
        trailing = rets[c].loc[:d].tail(t2.VOL_LOOKBACK).std() * ANN
        vol = table.get((pd.Period(d, 'M'), c), trailing) if table is not None else trailing
        used[(pd.Timestamp(d), c)] = vol
        if not np.isfinite(vol) or vol <= 0:
            return 0.0
        return min(t2.TARGET_VOL_PER_ASSET / vol, t2.MAX_LEVERAGE)

    t3._size = size
    try:
        port = t3.run_variant(prices, t3.LIVE16, 'V2', t2.HORIZONS, scalar_mode='trailing')
    finally:
        t3._size = orig
    return port, pd.Series(used)


def accuracy(used, rets):
    """Per month-end d: mean over ETFs of (log(realised vol over the next month / vol used))^2."""
    rows = {}
    stamps = sorted({d for d, _ in used.index})
    for i, d in enumerate(stamps[:-1]):
        nxt = stamps[i + 1]
        if d < pd.Timestamp(START) - pd.offsets.MonthEnd(1) or nxt > pd.Timestamp(END):
            continue
        win = rets.loc[(rets.index > d) & (rets.index <= nxt)]
        e = []
        for c in t3.LIVE16:
            v = used.get((d, c), np.nan)
            n = win[c].notna().sum()
            if np.isfinite(v) and v > 0 and n >= 15:
                e.append(np.log(win[c].std() * ANN / v) ** 2)
        if e:
            rows[d] = float(np.mean(e))
    return pd.Series(rows)


def nw_t(x, lags=3):
    x = np.asarray(x, float)
    n = len(x)
    u = x - x.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return float(x.mean() / np.sqrt(var / n))


def sharpe(r, rf=0.0):
    r = r.dropna()
    return float((r - rf).mean() / r.std() * ANN) if len(r) > 20 and r.std() > 0 else float('nan')


def maxdd(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax().clip(lower=1.0) - 1).min())


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    prices = t2.load_prices()
    rets = prices[t3.LIVE16].pct_change(fill_method=None)
    runs = {'BASE': run(prices)}
    for name, col in (('RIDGE', 'F_M1'), ('HAR', 'F_B1'), ('LGBM', 'F_M2')):
        runs[name] = run(prices, forecast_table(col))
    pd.DataFrame({k: v[0] for k, v in runs.items()}).to_csv(os.path.join(OUT, 'daily_returns.csv'))

    w = {k: v[0].loc[START:END] for k, v in runs.items()}
    acc = {k: accuracy(v[1], rets) for k, v in runs.items()}
    emit(f"Trend sleeve (live 16 ETFs), {START} .. {END}; only the per-asset vol differs.\n")
    emit(f"{'sizing':7s} {'no-rf':>6s} {'rf':>6s} {'last24 no-rf':>13s} {'MaxDD':>7s} {'2020-25 no-rf':>14s} "
         f"{'turnover/yr':>11s} {'vol err':>8s} {'port vol sd':>11s}")
    for k, r in w.items():
        last24 = r[r.index > r.index[-1] - pd.DateOffset(months=24)]
        mvol = r.groupby(r.index.to_period('M')).std() * ANN
        emit(f"{k:7s} {sharpe(r):+6.2f} {sharpe(r, RF):+6.2f} {sharpe(last24):+13.2f} {maxdd(r):7.1%} "
             f"{sharpe(r.loc['2020-01-01':'2025-12-31']):+14.2f} {runs[k][0].attrs['turnover_per_year']:11.1f} "
             f"{acc[k].mean():8.3f} {mvol.std():11.3f}")
    yrs = pd.DataFrame({k: r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1) for k, r in w.items()})
    emit("\nCalendar-year net returns:\n" + (yrs * 100).round(1).to_string())

    b, c = w['BASE'], w['RIDGE']
    d = (acc['BASE'] - acc['RIDGE']).dropna()
    gain = 1 - acc['RIDGE'].mean() / acc['BASE'].mean()
    t = nw_t(d.values)
    l24 = lambda r: r[r.index > r.index[-1] - pd.DateOffset(months=24)]
    g = [('Z1 vol error >= 10% lower, t >= 2.0', gain >= GATE['z1_gain'] and t >= GATE['z1_t'],
          f"{gain:+.1%} lower, t {t:.2f} ({len(d)} months)"),
         ('Z2 Sharpe >= BASE - 0.03', sharpe(c) >= sharpe(b) - GATE['z2'], f"{sharpe(c):.2f} vs {sharpe(b):.2f}"),
         ('Z3 last-24m Sharpe >= BASE - 0.10', sharpe(l24(c)) >= sharpe(l24(b)) - GATE['z3'],
          f"{sharpe(l24(c)):.2f} vs {sharpe(l24(b)):.2f}"),
         ('Z4 MaxDD within 1 pt of BASE', maxdd(c) >= maxdd(b) - GATE['z4'], f"{maxdd(c):.1%} vs {maxdd(b):.1%}")]
    emit("\nGate (T-2026-10-03-94), RIDGE vs BASE:")
    for name, ok, val in g:
        emit(f"  {name:38s} {val:34s} {'ok' if ok else 'FAIL'}")
    emit(f"\nT-94 VERDICT: {'PASS' if all(ok for _, ok, _ in g) else 'FAIL'}")
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
