"""
Volatility forecasting with machine learning, S&P 500 stocks — RESEARCH ONLY. Pre-registered
2026-10-01 before any forecast was scored. Nothing here touches the live engine.

Question: can a machine-learning model forecast each stock's volatility over the next month
better than the standard model (HAR, Corsi 2009)? Unlike returns, volatility is persistent and
predictable, so this is a target where ML has a fair chance; the bar is the strong linear
benchmark the field uses, not a naive guess.

Data: IBKR adjusted daily closes (backtest_data/sp500_prices_long.csv), point-in-time S&P 500
membership from the change log (taking_sides.membership), 569 of the 625 2020-2026 members
priced. Closes only, so volatility is close-to-close.

At each month-end close d (universe: members at d, price >= $5, >= 260 days of history):
  target  RV = standard deviation of the daily returns after d through the next month-end,
          annualised (>= 15 returns required); the model predicts log RV.
  features (known at the close of d): log realised vol over 5, 21, 63, 126, 252 days; log EWMA
          vol (lambda 0.94); log downside semivol 21d; log max |return| 21d; log vol of the same
          21-day window one quarter and one year earlier (earnings cycle); log vol of the
          equal-weight member return over 21 and 63 days; beta 252d; log residual vol 63d;
          returns over 21, 63, 252 days; distance to the 52-week high; log price.
Models, refitted every month-end on the previous 36 months of samples whose target window has
ended by d (no overlap with the future); out of sample 2021-01 .. the last complete month:
  B0 random walk  next month's vol = the last 21 days' vol
  B1 HAR          pooled OLS of log RV on log RV 5d, 21d, 63d  (the benchmark)
  M1 ridge        linear on all 19 features, standardised, alpha 1.0   (attribution: features)
  M2 LightGBM     same features; 400 trees, lr 0.03, 31 leaves, min 100 per leaf, subsample 0.8,
                  colsample 0.8, seed 42                              (the candidate)
Each forecast of variance is exp(2 x prediction + 2 x s^2), s^2 = the model's training residual
variance in logs (the lognormal mean correction; B0 uses the residual variance of log RV 21d
against the target on the same training window). All models are scored on the same rows.

Loss: QLIKE on variance, L = RV^2 / F - log(RV^2 / F) - 1 (robust to noise in the realised
proxy; Patton 2011), averaged over stocks each month, then over months. Also reported: MSE of
log RV, the monthly cross-sectional rank correlation of forecast with realised vol, and the
realised vol of the forecast's lowest and highest deciles.

Pre-registered decision, M2 against B1 out of sample:
  V1 mean monthly QLIKE at least 3% lower
  V2 t-statistic of the monthly QLIKE difference >= 2.0
  V3 lower QLIKE in at least 4 of the 6 calendar years 2021..2026
  V4 lower QLIKE over the last 24 scored months
  PASS -> ML forecasts stock volatility better than the standard model; using it for position
  sizing is a separate trial. Ridge versus HAR says how much comes from the extra features,
  LightGBM versus ridge how much from the nonlinearity (diagnostic, no decision).
Trials: T-2026-10-01-76 (B0) .. -79 (M2).

FOLLOW-UP T-2026-10-01-80, registered after T-79 FAILED and its diagnosis (so weaker evidence: it
reuses the same out-of-sample months). T-79's level correction used each model's IN-SAMPLE
residual variance, which flatters a flexible model: LightGBM's forecasts sat near the median of
realised variance (mean log(F/RV^2) +0.03 against HAR's +0.26) while its ranking and log error were
the best of all models. The fix, applied identically to every model: fit on the first 24 months of
the 36-month window, measure the residual variance on the last 12 (out of sample within the
training window), then refit on all 36 months and use that variance in the correction. Same four
rules V1..V4, LightGBM against HAR, both calibrated this way. Run with --calibration holdout
(writes results/vol_forecast_holdout/).

Usage: python3 vol_forecast_trial.py      (about a minute; writes results/vol_forecast/)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taking_sides import membership  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'vol_forecast')
CALIBRATION = 'insample'
TRAIN_START, OOS_START = '2017-01-01', '2021-01-01'
TRAIN_MONTHS = 36
HAR = ['lrv5', 'lrv21', 'lrv63']
FEATURES = HAR + ['lrv126', 'lrv252', 'lewma', 'lsemi21', 'lmaxabs21', 'lrv_q', 'lrv_y', 'lmkt21', 'lmkt63',
                  'beta', 'livol63', 'ret21', 'ret63', 'ret252', 'hi52', 'lprice']
ANN = np.sqrt(252)


def build_panel():
    px = pd.read_csv(os.path.join(BD, 'sp500_prices_long.csv'), index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(os.path.join(BD, 'sp500_universe_2020_2026.csv')).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(os.path.join(BD, 'sp500_changes.csv'), parse_dates=['date'])
    mem = membership(chg, uni)
    days = px.index
    r = px.pct_change(fill_method=None)
    member = pd.DataFrame(False, index=days, columns=px.columns)
    for s in px.columns:
        if s in mem:
            a, rm = mem[s]
            m = np.ones(len(days), bool)
            if pd.notna(a):
                m &= days >= a
            if pd.notna(rm) and (pd.isna(a) or rm > a):
                m &= days < rm
            member[s] = m
    mkt = r.where(member).mean(axis=1)
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
    mk21 = lg(mkt.rolling(21).std() * ANN)
    mk63 = lg(mkt.rolling(63).std() * ANN)
    month_ends = pd.Series(days, index=days).groupby([days.year, days.month]).max()
    month_ends = [pd.Timestamp(d) for d in month_ends.values if pd.Timestamp(d) >= pd.Timestamp(TRAIN_START)]
    hist = px.notna().cumsum()
    rows = []
    for i, d in enumerate(month_ends[:-1]):
        nxt = month_ends[i + 1]
        ok = [s for s in px.columns if member.at[d, s] and pd.notna(px.at[d, s]) and px.at[d, s] >= 5 and hist.at[d, s] >= 260]
        if len(ok) < 100:
            continue
        win = r.loc[(r.index > d) & (r.index <= nxt), ok]
        n = win.notna().sum()
        target = win.std() * ANN
        g = pd.DataFrame({'date': d, 'end': nxt, 'symbol': ok})
        for k, v in f.items():
            g[k] = v.loc[d, ok].values
        g['lmkt21'], g['lmkt63'] = mk21.loc[d], mk63.loc[d]
        g['rv_next'] = np.where(n.values >= 15, target.values, np.nan)
        rows.append(g)
    p = pd.concat(rows, ignore_index=True)
    p['y'] = np.log(p['rv_next'].clip(lower=1e-4))
    return p


def qlike(rv, var_f):
    x = rv ** 2 / var_f
    return x - np.log(x) - 1


def run_models(p):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor
    scored = p.dropna(subset=['y'] + HAR).copy()
    dates = sorted(scored.date.unique())
    out, imps = [], []
    for d in dates:
        if d < pd.Timestamp(OOS_START):
            continue
        tr = scored[(scored['end'] <= d) & (scored.date >= d - pd.DateOffset(months=TRAIN_MONTHS))]
        te = scored[scored.date == d].copy()
        if len(tr) < 5000 or te.empty:
            continue
        if CALIBRATION == 'holdout':
            cut = d - pd.DateOffset(months=12)
            fit_part, hold = tr[tr['end'] <= cut], tr[tr.date > cut]
        else:
            fit_part, hold = tr, tr
        # B0: random walk; its correction from the same training window
        s2 = float(np.var(hold['y'] - hold['lrv21']))
        te['F_B0'] = np.exp(2 * te['lrv21'] + 2 * s2)
        # B1: HAR
        har0 = LinearRegression().fit(fit_part[HAR], fit_part['y'])
        s2 = float(np.var(hold['y'] - har0.predict(hold[HAR])))
        har = LinearRegression().fit(tr[HAR], tr['y'])
        te['F_B1'] = np.exp(2 * har.predict(te[HAR]) + 2 * s2)
        # M1: ridge on all features (medians from the training window fill gaps)
        med = tr[FEATURES].median()
        sc0 = StandardScaler().fit(fit_part[FEATURES].fillna(med))
        r0 = Ridge(alpha=1.0).fit(sc0.transform(fit_part[FEATURES].fillna(med)), fit_part['y'])
        s2 = float(np.var(hold['y'] - r0.predict(sc0.transform(hold[FEATURES].fillna(med)))))
        sc = StandardScaler().fit(tr[FEATURES].fillna(med))
        ridge = Ridge(alpha=1.0).fit(sc.transform(tr[FEATURES].fillna(med)), tr['y'])
        te['F_M1'] = np.exp(2 * ridge.predict(sc.transform(te[FEATURES].fillna(med))) + 2 * s2)
        # M2: LightGBM
        mk = lambda: LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
        g0 = mk().fit(fit_part[FEATURES], fit_part['y'])
        s2 = float(np.var(hold['y'] - g0.predict(hold[FEATURES])))
        gbm = mk().fit(tr[FEATURES], tr['y'])
        te['F_M2'] = np.exp(2 * gbm.predict(te[FEATURES]) + 2 * s2)
        out.append(te)
        imps.append(pd.Series(gbm.booster_.feature_importance('gain'), index=FEATURES, name=d))
    return pd.concat(out, ignore_index=True), pd.DataFrame(imps)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    p = build_panel()
    pred, imps = run_models(p)
    pred.to_pickle(os.path.join(OUT, 'predictions.pkl'))
    models = {'B0': 'random walk (last 21 days)', 'B1': 'HAR (benchmark)', 'M1': 'ridge, all features', 'M2': 'LightGBM, all features'}
    emit(f"Scored {pred.date.nunique()} months ({pred.date.min().date()}..{pred.date.max().date()}), "
         f"{len(pred)} stock-months, median {pred.groupby('date').size().median():.0f} stocks a month; {time.time() - t0:.0f}s")
    monthly = {}
    rows = {}
    for m in models:
        F = pred[f'F_{m}']
        pred[f'Q_{m}'] = qlike(pred['rv_next'], F)
        pred[f'E_{m}'] = (np.log(np.sqrt(F)) - pred['y']) ** 2
        g = pred.groupby('date')
        monthly[m] = g[f'Q_{m}'].mean()
        rank = g.apply(lambda x: x[f'F_{m}'].corr(x['rv_next'], method='spearman'), include_groups=False)
        dec = pred.assign(dec=g[f'F_{m}'].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False)))
        lo = dec[dec.dec == 0].groupby('date').rv_next.mean().mean()
        hi = dec[dec.dec == 9].groupby('date').rv_next.mean().mean()
        rows[m] = {'model': models[m], 'QLIKE': monthly[m].mean(), 'MSE log vol': g[f'E_{m}'].mean().mean(),
                   'rank corr': rank.mean(), 'low-decile vol %': lo * 100, 'high-decile vol %': hi * 100}
    df = pd.DataFrame(rows).T
    df['QLIKE vs HAR %'] = (df['QLIKE'].astype(float) / float(df.loc['B1', 'QLIKE']) - 1) * 100
    pd.set_option('display.width', 220)
    emit("\nOut-of-sample accuracy (lower QLIKE and MSE are better):")
    emit(df.to_string(float_format=lambda v: f'{v:.4f}' if abs(v) < 1 else f'{v:.1f}'))
    M = pd.DataFrame(monthly)
    by_year = M.groupby(M.index.year).mean()
    emit("\nMean QLIKE by year:")
    emit(by_year.to_string(float_format=lambda v: f'{v:.4f}'))

    emit("\nLightGBM feature importance (mean gain share, %):")
    emit((imps.div(imps.sum(axis=1), axis=0).mean().sort_values(ascending=False) * 100).round(1).to_string())

    diff = M['M2'] - M['B1']
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    last = M.iloc[-24:]
    years_won = int((by_year['M2'] < by_year['B1']).sum())
    v = {'V1 QLIKE >= 3% lower': (M['M2'].mean() / M['B1'].mean() - 1) <= -0.03,
         'V2 t-stat >= 2.0': t <= -2.0,
         'V3 better in >= 4 of 6 years': years_won >= 4,
         'V4 better over the last 24 months': last['M2'].mean() < last['B1'].mean()}
    emit(f"\nDECISION (pre-registered), LightGBM vs HAR: QLIKE {(M['M2'].mean() / M['B1'].mean() - 1) * 100:+.1f}%, "
         f"t = {-t:.2f} (in HAR's favour if negative), years won {years_won}/{len(by_year)}, "
         f"last 24 months {last['M2'].mean():.4f} vs {last['B1'].mean():.4f}")
    for k, ok in v.items():
        emit(f"  {k}: {'ok' if ok else 'FAIL'}")
    emit(f"VERDICT: {'PASS - ML forecasts stock volatility better than HAR' if all(v.values()) else 'FAIL'}")
    emit(f"Attribution (diagnostic): ridge vs HAR {(M['M1'].mean() / M['B1'].mean() - 1) * 100:+.1f}% (extra features), "
         f"LightGBM vs ridge {(M['M2'].mean() / M['M1'].mean() - 1) * 100:+.1f}% (nonlinearity)")
    M.to_csv(os.path.join(OUT, 'monthly_qlike.csv'))
    df.to_csv(os.path.join(OUT, 'summary.csv'))
    with open(os.path.join(OUT, 'report.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    emit(f"({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', choices=['insample', 'holdout'], default='insample')
    a = ap.parse_args()
    if a.calibration == 'holdout':
        CALIBRATION = 'holdout'
        OUT = os.path.join(HERE, 'results', 'vol_forecast_holdout')
    main()
