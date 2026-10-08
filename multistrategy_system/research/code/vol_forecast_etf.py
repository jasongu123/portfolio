"""
Volatility forecasting on ETFs, fresh data — RESEARCH ONLY. Pre-registered 2026-10-01 before any
forecast was scored, after the S&P 500 stock trial (VOL_FORECAST.md, T-76..80).

Why a second test: on stocks, LightGBM (the registered candidate) failed against HAR while the
linear model on the same features beat HAR by 7-8% as a diagnostic. Picking that winner after
the fact would be data snooping, so both models are put to a fresh test on data never used for
volatility forecasting: the cross-asset ETF panel (the trend sleeve's 16 live ETFs plus SPY and
the 30 ETFs of TREND_BREADTH.md: equities by country, Treasuries, credit, inflation-linked and
international bonds, commodities, currencies, real estate), IBKR adjusted daily closes, 2006-2026.

Setup (identical to vol_forecast_trial.py except where stated):
  target    next month's close-to-close volatility (returns after the month-end close d through
            the next month-end, >= 15 required), annualised; models predict its log
  features  the same 19; the "market" return is the equal-weight mean of the ETFs priced that day
            (a cross-asset market), and beta and residual vol are measured against it; no
            membership or price filter (>= 260 days of history required)
  training  EXPANDING window: every sample whose target window ended by d (the panel is too
            small for 36-month windows); at least 1,000 samples
  calibration  holdout, fixed in advance for every model: fit on samples ending by d - 12
            months, take the residual variance on the samples dated in the last 12 months, refit
            on everything for the forecast (the method of T-80)
  out of sample  2012-01 .. the last complete month
Models: B0 random walk (reference), B1 HAR (benchmark), M1 ridge on 19 features, M2 LightGBM on
19 features (same hyper-parameters as the stock trial).

Pre-registered decision, applied to M1 and M2 separately against B1 (two candidates, so the
t threshold is Bonferroni-adjusted):
  V1 mean monthly QLIKE at least 3% lower
  V2 t-statistic of the monthly QLIKE difference >= 2.24
  V3 lower QLIKE in at least 60% of the calendar years 2012..2026
  V4 lower QLIKE over the last 24 scored months
  A candidate that passes all four is a validated volatility forecaster on fresh data; using it for
  the ETF or trend sleeve's position sizing is a separate trial.
Trials: T-2026-10-01-81 (M1 ridge), -82 (M2 LightGBM).

Usage: python3 vol_forecast_etf.py        (writes results/vol_forecast_etf/)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vol_forecast_trial import FEATURES, HAR, qlike  # noqa: E402

OUT = os.path.join(HERE, 'results', 'vol_forecast_etf')
OOS_START = '2012-01-01'
MIN_TRAIN = 1000
T_MIN = 2.24
ANN = np.sqrt(252)


def load_prices():
    import trend_follow_v2 as t2
    a = t2.load_prices()
    b = pd.read_csv(os.path.join(HERE, 'backtest_data', 'trend_breadth_prices.csv'), index_col=0, parse_dates=True)
    px = a.join(b[[c for c in b.columns if c not in a.columns]], how='outer').sort_index()
    return px


def build_panel(px):
    days = px.index
    r = px.pct_change(fill_method=None)
    mkt = r.mean(axis=1)
    lg = lambda x: np.log(x.clip(lower=1e-4))
    rv = {n: r.rolling(n, min_periods=int(n * 0.8)).std() * ANN for n in (5, 21, 63, 126, 252)}
    f = {'lrv5': lg(rv[5]), 'lrv21': lg(rv[21]), 'lrv63': lg(rv[63]), 'lrv126': lg(rv[126]), 'lrv252': lg(rv[252]),
         'lewma': lg(np.sqrt((r ** 2).ewm(alpha=0.06, min_periods=30).mean()) * ANN),
         'lsemi21': lg(np.sqrt((r.clip(upper=0) ** 2).rolling(21, min_periods=17).mean()) * ANN),
         'lmaxabs21': lg(r.abs().rolling(21, min_periods=17).max()),
         'lrv_q': lg(rv[21].shift(42)), 'lrv_y': lg(rv[21].shift(231)),
         'ret21': px / px.shift(21) - 1, 'ret63': px / px.shift(63) - 1, 'ret252': px / px.shift(252) - 1,
         'hi52': px / px.rolling(252).max() - 1, 'lprice': np.log(px)}
    f['beta'] = r.rolling(252, min_periods=200).cov(mkt).div(mkt.rolling(252, min_periods=200).var(), axis=0)
    f['livol63'] = lg(r.sub(mkt, axis=0).rolling(63, min_periods=50).std() * ANN)
    mk21, mk63 = lg(mkt.rolling(21).std() * ANN), lg(mkt.rolling(63).std() * ANN)
    month_ends = [pd.Timestamp(d) for d in pd.Series(days, index=days).groupby([days.year, days.month]).max().values]
    hist = px.notna().cumsum()
    rows = []
    for i, d in enumerate(month_ends[:-1]):
        nxt = month_ends[i + 1]
        ok = [s for s in px.columns if pd.notna(px.at[d, s]) and hist.at[d, s] >= 260]
        if not ok:
            continue
        win = r.loc[(r.index > d) & (r.index <= nxt), ok]
        n = win.notna().sum()
        g = pd.DataFrame({'date': d, 'end': nxt, 'symbol': ok})
        for k, v in f.items():
            g[k] = v.loc[d, ok].values
        g['lmkt21'], g['lmkt63'] = mk21.loc[d], mk63.loc[d]
        g['rv_next'] = np.where(n.values >= 15, (win.std() * ANN).values, np.nan)
        rows.append(g)
    p = pd.concat(rows, ignore_index=True)
    p['y'] = np.log(p['rv_next'].clip(lower=1e-4))
    return p


def run_models(p):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor
    scored = p.dropna(subset=['y'] + HAR).copy()
    out = []
    for d in sorted(scored.date.unique()):
        if d < pd.Timestamp(OOS_START):
            continue
        tr = scored[scored['end'] <= d]
        te = scored[scored.date == d].copy()
        cut = d - pd.DateOffset(months=12)
        fit_part, hold = tr[tr['end'] <= cut], tr[tr.date > cut]
        if len(tr) < MIN_TRAIN or te.empty or len(hold) < 50:
            continue
        s2 = float(np.var(hold['y'] - hold['lrv21']))
        te['F_B0'] = np.exp(2 * te['lrv21'] + 2 * s2)
        h0 = LinearRegression().fit(fit_part[HAR], fit_part['y'])
        s2 = float(np.var(hold['y'] - h0.predict(hold[HAR])))
        te['F_B1'] = np.exp(2 * LinearRegression().fit(tr[HAR], tr['y']).predict(te[HAR]) + 2 * s2)
        med = tr[FEATURES].median()
        X = lambda df: df[FEATURES].fillna(med)
        sc0 = StandardScaler().fit(X(fit_part))
        r0 = Ridge(alpha=1.0).fit(sc0.transform(X(fit_part)), fit_part['y'])
        s2 = float(np.var(hold['y'] - r0.predict(sc0.transform(X(hold)))))
        sc = StandardScaler().fit(X(tr))
        te['F_M1'] = np.exp(2 * Ridge(alpha=1.0).fit(sc.transform(X(tr)), tr['y']).predict(sc.transform(X(te))) + 2 * s2)
        mk = lambda: LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        g0 = mk().fit(fit_part[FEATURES], fit_part['y'])
        s2 = float(np.var(hold['y'] - g0.predict(hold[FEATURES])))
        te['F_M2'] = np.exp(2 * mk().fit(tr[FEATURES], tr['y']).predict(te[FEATURES]) + 2 * s2)
        out.append(te)
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    px = load_prices()
    emit(f"Panel: {px.shape[1]} ETFs, {px.index.min().date()}..{px.index.max().date()}")
    pred = run_models(build_panel(px))
    pred.to_pickle(os.path.join(OUT, 'predictions.pkl'))
    names = {'B0': 'random walk (last 21 days)', 'B1': 'HAR (benchmark)', 'M1': 'ridge, 19 features', 'M2': 'LightGBM, 19 features'}
    emit(f"Scored {pred.date.nunique()} months ({pred.date.min().date()}..{pred.date.max().date()}), {len(pred)} ETF-months, "
         f"median {pred.groupby('date').size().median():.0f} ETFs a month; {time.time() - t0:.0f}s")
    monthly, rows = {}, {}
    for m in names:
        pred[f'Q_{m}'] = qlike(pred['rv_next'], pred[f'F_{m}'])
        g = pred.groupby('date')
        monthly[m] = g[f'Q_{m}'].mean()
        rank = g.apply(lambda x: x[f'F_{m}'].corr(x['rv_next'], method='spearman'), include_groups=False)
        rows[m] = {'model': names[m], 'QLIKE': monthly[m].mean(), 'rank corr': rank.mean(),
                   'mean log(F/RV^2)': float((np.log(pred[f'F_{m}']) - 2 * np.log(pred['rv_next'])).mean())}
    df = pd.DataFrame(rows).T
    df['QLIKE vs HAR %'] = (df['QLIKE'].astype(float) / float(df.loc['B1', 'QLIKE']) - 1) * 100
    pd.set_option('display.width', 200)
    emit("\nOut-of-sample accuracy (lower QLIKE is better):")
    emit(df.to_string(float_format=lambda v: f'{v:.4f}' if abs(v) < 1 else f'{v:.1f}'))
    M = pd.DataFrame(monthly)
    by_year = M.groupby(M.index.year).mean()
    emit("\nMean QLIKE by year:")
    emit(by_year.to_string(float_format=lambda v: f'{v:.4f}'))
    emit("\nDECISIONS (pre-registered, each candidate vs HAR):")
    verdicts = {}
    for m in ('M1', 'M2'):
        diff = M[m] - M['B1']
        t = -diff.mean() / diff.std() * np.sqrt(len(diff))
        rel = M[m].mean() / M['B1'].mean() - 1
        won = int((by_year[m] < by_year['B1']).sum())
        last = M.iloc[-24:]
        v = [rel <= -0.03, t >= T_MIN, won >= 0.6 * len(by_year), last[m].mean() < last['B1'].mean()]
        verdicts[m] = all(v)
        emit(f"  {names[m]}: QLIKE {rel * 100:+.1f}% {'ok' if v[0] else 'FAIL'} | t {t:.2f} (need {T_MIN}) {'ok' if v[1] else 'FAIL'} | "
             f"years {won}/{len(by_year)} {'ok' if v[2] else 'FAIL'} | last 24m {last[m].mean():.4f} vs {last['B1'].mean():.4f} "
             f"{'ok' if v[3] else 'FAIL'} => {'PASS' if all(v) else 'FAIL'}")
    M.to_csv(os.path.join(OUT, 'monthly_qlike.csv'))
    df.to_csv(os.path.join(OUT, 'summary.csv'))
    with open(os.path.join(OUT, 'report.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    emit(f"({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
