"""
Machine-learning cross-sectional ranking of S&P 500 stocks — RESEARCH ONLY. Pre-registered
2026-10-01 before any return was computed. Nothing here touches the live engine.

Question: can a model that combines many weak price-based signals rank S&P 500 stocks by next
month's return well enough to trade, now that the single published signals on large caps have
died (TAKING_SIDES.md, SEASONALITY.md, CROWDED_SPACES.md)? And does machine learning add anything
over the textbook combination?

Data: IBKR adjusted daily closes (backtest_data/sp500_prices_long.csv, 569 of the 625 names that
were S&P 500 members at any time 2020-2026; the 56 missing are mostly delisted, so the universe
has survivorship bias, measured and reported). Point-in-time membership from the change log
(taking_sides.membership). Closes only: no volume, no fundamentals.

Rebalance: the last trading day of each month (signal at that close d). Positions are entered at
the NEXT trading day's close and held to the close of the trading day after the next signal (a
one-day lag; nothing trades on the signal close). Universe at d: index member at d, price >= $5,
>= 260 days of history.

Features at d (17, cross-sectionally ranked to [-0.5, 0.5], missing = 0):
  mom_12_1, mom_6_1, mom_3, rev_1m, rev_1w, vol_1m, vol_3m, ivol_3m (residual to the equal-weight
  member return), beta_1y, max_1m, min_1m, skew_3m, hi52 (distance to 52-week high), lo52,
  sma50, sma200, seas (mean same-calendar-month return over the prior 5 years).
Target: cross-sectional rank of the return from entry to exit (as traded), in [-0.5, 0.5].

Arms
  A0 composite   textbook, no fitting: rank(mom_12_1) - rank(rev_1m) - rank(vol_3m)
  A1 ridge       linear regression on the 17 ranks, L2 alpha = 1.0
  A2 lgbm        LightGBM regressor (300 trees, lr 0.03, 15 leaves, min 200 per leaf,
                 subsample 0.8, colsample 0.8, seed 42)
Models are refitted at every rebalance on the previous 36 months of samples whose exit is at or
before d (no overlap with the future). Out-of-sample predictions start 2021-01 (training from
2018-01, partly before the 2020-2026 membership window: survivorship there is worse).

Portfolio: long the top decile, short the bottom decile, equal weight within legs, $1 long and
$1 short per $1 of capital, holdings drift within the month; cost 5 bp per unit of one-way
turnover (2 and 10 bp reported); short borrow 0.5% a year on the short leg.

Pre-registered decision (daily returns, Sharpe of the self-financing long-short book, OOS
2021-01..2026-09-15; last 2y = 2024-09-15..2026-09-15):
  L1 best ML arm Sharpe >= 0.50
  L2 its last-2y Sharpe > 0
  L3 its best 24-month block <= 75% of total profit
  L4 it beats A0 by >= 0.10 Sharpe (otherwise ML adds nothing over the textbook composite)
  L5 |correlation| of its monthly returns with each existing sleeve (trend, swing-only stocks
     S1, ETF paper rules E0; overlap 2021-01..2025-12) < 0.30
  Model choice: ridge unless LightGBM >= ridge + 0.10. Pass -> candidate for a paper forward
  test (no live change from this trial). Fail -> written up with a mechanism.
Diagnostics (no decisions): IC (rank correlation of prediction with the next return), decile
spread monotonicity, long-only top decile minus the member average, beta to the member average,
LightGBM feature importance, cost sensitivity, survivorship coverage.
Trials: T-2026-10-01-73 (A0), -74 (A1), -75 (A2).

Usage: python3 ml_xsection_trial.py          (minutes; writes results/ml_xsection/)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taking_sides import membership, in_index  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'ml_xsection_v2')  # v1 kept in results/ml_xsection/
PX_FILE = os.path.join(BD, 'sp500_prices_long.csv')
UNI_FILE = os.path.join(BD, 'sp500_universe_2020_2026.csv')
CHG_FILE = os.path.join(BD, 'sp500_changes.csv')
FEATURES = ['mom_12_1', 'mom_6_1', 'mom_3', 'rev_1m', 'rev_1w', 'vol_1m', 'vol_3m', 'ivol_3m', 'beta_1y',
            'max_1m', 'min_1m', 'skew_3m', 'hi52', 'lo52', 'sma50', 'sma200', 'seas']
TRAIN_START, OOS_START, END = '2018-01-01', '2021-01-01', '2026-09-15'
LAST2 = ('2024-09-15', '2026-09-15')
TRAIN_MONTHS = 36
COST_BPS, BORROW = 5.0, 0.005
ANN = np.sqrt(252)


def load():
    px = pd.read_csv(PX_FILE, index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(UNI_FILE).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(CHG_FILE, parse_dates=['date'])
    return px, uni, chg


def rank_cs(s):
    r = s.rank(pct=True)
    return (r - 0.5 - 0.5 / r.count()).fillna(0.0) if r.count() else s * 0


def build_panel(px, uni, chg):
    rets = px.pct_change(fill_method=None)
    mem = membership(chg, uni)
    days = px.index
    month_ends = pd.Series(days, index=days).groupby([days.year, days.month]).max().values
    month_ends = [pd.Timestamp(d) for d in month_ends if pd.Timestamp(TRAIN_START) <= pd.Timestamp(d) <= pd.Timestamp(END)]
    # Audit 2026-10-01 item 9: the data's last day (mid-month) is not a month-end signal.
    month_ends = [d for d in month_ends if d != days[-1]]
    member_mask = pd.DataFrame(False, index=days, columns=px.columns)
    for s in px.columns:
        if s in mem:
            a, r = mem[s]
            m = np.ones(len(days), bool)
            if pd.notna(a):
                m &= days >= a
            if pd.notna(r) and (pd.isna(a) or r > a):
                m &= days < r
            member_mask[s] = m
    mkt = rets.where(member_mask).mean(axis=1)
    feats = {
        'mom_12_1': px.shift(21) / px.shift(252) - 1, 'mom_6_1': px.shift(21) / px.shift(126) - 1,
        'mom_3': px / px.shift(63) - 1, 'rev_1m': px / px.shift(21) - 1, 'rev_1w': px / px.shift(5) - 1,
        'vol_1m': rets.rolling(21).std(), 'vol_3m': rets.rolling(63).std(),
        'max_1m': rets.rolling(21).max(), 'min_1m': rets.rolling(21).min(), 'skew_3m': rets.rolling(63).skew(),
        'hi52': px / px.rolling(252).max() - 1, 'lo52': px / px.rolling(252).min() - 1,
        'sma50': px / px.rolling(50).mean() - 1, 'sma200': px / px.rolling(200).mean() - 1,
    }
    var_m = mkt.rolling(252).var()
    feats['beta_1y'] = rets.rolling(252).cov(mkt).div(var_m, axis=0)
    resid = rets.sub(mkt, axis=0)
    feats['ivol_3m'] = resid.rolling(63).std()
    monthly_ret = px.resample('ME').last().pct_change(fill_method=None)
    rows = []
    hist_len = px.notna().cumsum()
    for i, d in enumerate(month_ends):
        members = [s for s in px.columns if member_mask.at[d, s]]
        ok = [s for s in members if pd.notna(px.at[d, s]) and px.at[d, s] >= 5 and hist_len.at[d, s] >= 260]
        if len(ok) < 100:
            continue
        g = pd.DataFrame({'date': d, 'symbol': ok})
        for k, f in feats.items():
            g[k] = f.loc[d, ok].values
        prior = monthly_ret[(monthly_ret.index.month == d.month) & (monthly_ret.index < d - pd.Timedelta(days=200))].tail(5)
        g['seas'] = prior[ok].mean().values if len(prior) else np.nan
        # traded return: entry at the next close after d, exit at the close after the next signal
        di = days.get_loc(d)
        nxt = month_ends[i + 1] if i + 1 < len(month_ends) else None
        if di + 1 < len(days) and nxt is not None and days.get_loc(nxt) + 1 < len(days):
            e0, e1 = days[di + 1], days[days.get_loc(nxt) + 1]
            g['entry'], g['exit'] = e0, e1
            g['fwd'] = (px.loc[e1, ok] / px.loc[e0, ok] - 1).values
        elif di + 1 < len(days):
            # Audit item 9: the last executable signal is held to the end of the data and marked
            # to market; its incomplete forward return is never a training label.
            g['entry'], g['exit'], g['fwd'] = days[di + 1], days[-1], np.nan
        else:
            g['entry'], g['exit'], g['fwd'] = pd.NaT, pd.NaT, np.nan
        g['n_members'] = len(members)
        g['n_members_total'] = sum(1 for s in uni if in_index(mem, s, d))
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)
    for k in FEATURES:
        panel[k + '_r'] = panel.groupby('date')[k].transform(rank_cs)
    panel['y'] = panel.groupby('date')['fwd'].transform(rank_cs)
    return panel, rets, member_mask


def fit_predict(panel):
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    X = [k + '_r' for k in FEATURES]
    dates = sorted(panel.date.unique())
    out, imps = [], []
    for d in dates:
        if d < pd.Timestamp(OOS_START):
            continue
        tr = panel[(panel['exit'] <= d) & (panel.date >= d - pd.DateOffset(months=TRAIN_MONTHS)) & panel.fwd.notna()]
        te = panel[panel.date == d].copy()
        if len(tr) < 5000:
            continue
        ridge = Ridge(alpha=1.0).fit(tr[X], tr['y'])
        lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42,
                             verbose=-1).fit(tr[X], tr['y'])
        te['A1'] = ridge.predict(te[X])
        te['A2'] = lgbm.predict(te[X])
        te['A0'] = te['mom_12_1_r'] - te['rev_1m_r'] - te['vol_3m_r']
        out.append(te)
        imps.append(pd.Series(lgbm.booster_.feature_importance('gain'), index=FEATURES, name=d))
    return pd.concat(out, ignore_index=True), pd.DataFrame(imps)


def book(pred, col, px, cost_bps=COST_BPS, side='ls', member_mask=None):
    """Daily returns of the decile book on its own NAV. side: 'ls' long-short; 'lo' long top decile
    minus members. Audit fix 2026-10-01 (item 8): positions are fractions of the NAV at each
    rebalance; each day's return is that day's P&L over the previous day's NAV (it used to be the
    P&L increment in units of the rebalance NAV, compounded as if it were a return)."""
    rets = px.pct_change(fill_method=None)
    w_prev = pd.Series(dtype=float)
    parts, turns = [], []
    for d, g in pred.groupby('date'):
        g = g.dropna(subset=[col])
        if pd.isna(g['entry'].iloc[0]):
            continue
        e0, e1 = g['entry'].iloc[0], g['exit'].iloc[0]
        q = g[col].rank(pct=True)
        L, S = g[q > 0.9].symbol, g[q <= 0.1].symbol
        if side == 'ls':
            w = pd.concat([pd.Series(1 / len(L), index=L), pd.Series(-1 / len(S), index=S)])
        else:
            allm = g.symbol
            w = pd.Series(1 / len(L), index=L).subtract(pd.Series(1 / len(allm), index=allm), fill_value=0.0)
        turn = w.subtract(w_prev, fill_value=0.0).abs().sum()
        turns.append(turn)
        r = rets.loc[e0:e1, w.index].iloc[1:].fillna(0.0)
        cum = (1 + r).cumprod()
        pnl = (cum - 1).mul(w, axis=1).sum(axis=1)                  # cumulative P&L / NAV at entry
        nav = 1 - turn * cost_bps / 1e4 + pnl                        # cost paid at entry
        if side == 'ls':
            nav = nav - BORROW / 252 * np.arange(1, len(nav) + 1) * w[w < 0].abs().sum()
        prev = pd.concat([pd.Series([1.0]), nav.iloc[:-1].reset_index(drop=True)])
        dr = pd.Series(nav.values / prev.values - 1, index=nav.index)
        parts.append(dr)
        held = cum.iloc[-1] * w                                       # drifted holdings at exit
        w_prev = held / float(nav.iloc[-1])                           # as fractions of the exit NAV
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()], float(np.mean(turns))


def sr(x):
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def conc(x):
    tot = x.sum()
    return float(x.rolling(504, min_periods=400).sum().max() / tot) if tot > 0 else float('inf')


def mdd(x):
    eq = np.cumprod(1 + x.dropna().values)
    return float((eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1).min())


def existing_sleeves():
    out = {}
    try:
        tr = pd.read_csv(os.path.join(HERE, 'results', 'headline_2026-09_daily_new.csv'), index_col=0, parse_dates=True)
        out['trend'] = tr['trend_live']
    except Exception:
        pass
    try:
        import split_trial
        out['swing S1'] = split_trial.load('S1')[0]
        out['ETF E0'] = split_trial.load('E0')[0]
    except Exception:
        pass
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    px, uni, chg = load()
    missing = [s for s in uni if s not in px.columns]
    emit(f"Universe {len(uni)} names (2020-2026 members); no prices for {len(missing)}: {', '.join(missing)}")
    panel, rets, member_mask = build_panel(px, uni, chg)
    cov = panel.groupby('date').agg(n=('symbol', 'size'), members=('n_members_total', 'first'))
    oos_cov = cov.loc[OOS_START:]
    emit(f"Panel: {len(panel)} stock-months, {cov.shape[0]} rebalances; OOS names/month median {oos_cov.n.median():.0f} "
         f"of {oos_cov.members.median():.0f} point-in-time members ({(oos_cov.n / oos_cov.members).median():.0%} covered)")
    pred, imps = fit_predict(panel)
    pred.to_pickle(os.path.join(OUT, 'predictions.pkl'))
    imps.to_csv(os.path.join(OUT, 'lgbm_importance.csv'))
    emit(f"OOS rebalances {pred.date.nunique()} ({pred.date.min().date()}..{pred.date.max().date()}); {time.time() - t0:.0f}s")

    R, rows = {}, {}
    for col in ('A0', 'A1', 'A2'):
        for c in (2.0, 5.0, 10.0):
            r, turn = book(pred, col, px, cost_bps=c)
            R[(col, c)] = r.loc[OOS_START:END]
            if c == COST_BPS:
                tn = turn
        lo, _ = book(pred, col, px, side='lo')
        x = R[(col, COST_BPS)]
        mk = rets.where(member_mask).mean(axis=1).reindex(x.index)
        ic = pred.dropna(subset=['fwd']).groupby('date').apply(lambda g: g[col].corr(g['fwd'], method='spearman'))
        rows[col] = {'Sharpe': sr(x), 'ret %/yr': x.mean() * 252 * 100, 'vol %': x.std() * ANN * 100, 'MaxDD %': mdd(x) * 100,
                     'last 2y': sr(x.loc[LAST2[0]:LAST2[1]]), 'conc': conc(x),
                     **{str(y): sr(x.loc[str(y)]) for y in range(2021, 2027)},
                     'SR 2bp': sr(R[(col, 2.0)]), 'SR 10bp': sr(R[(col, 10.0)]), 'turnover/mo': tn,
                     'IC mean': ic.mean(), 'IC t': ic.mean() / ic.std() * np.sqrt(len(ic)),
                     'beta to mkt': x.cov(mk) / mk.var(), 'long-only active SR': sr(lo.loc[OOS_START:END])}
    df = pd.DataFrame(rows).T
    pd.set_option('display.width', 250)
    emit("\nResults (OOS 2021-01..2026-09-15, decile long-short, 5 bp, borrow 0.5%/yr):")
    emit(df.to_string(float_format=lambda v: f'{v:.2f}'))
    df.to_csv(os.path.join(OUT, 'summary.csv'))
    pd.DataFrame({f'{k[0]}_{k[1]:g}bp': v for k, v in R.items()}).to_csv(os.path.join(OUT, 'daily_returns.csv'))

    emit("\nMean next-month return by prediction decile (OOS, % per month; 1 = lowest):")
    dec = {}
    for col in ('A0', 'A1', 'A2'):
        p = pred.dropna(subset=['fwd']).copy()
        p['dec'] = p.groupby('date')[col].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False) + 1)
        dec[col] = p.groupby('dec')['fwd'].mean() * 100
    emit(pd.DataFrame(dec).T.to_string(float_format=lambda v: f'{v:.2f}'))
    emit("\nLightGBM feature importance (mean gain share):")
    emit((imps.div(imps.sum(axis=1), axis=0).mean().sort_values(ascending=False) * 100).round(1).to_string())

    ex = existing_sleeves()
    emit("\nMonthly correlation with existing sleeves (2021-01..2025-12):")
    corr = {}
    for col in ('A0', 'A1', 'A2'):
        m = R[(col, COST_BPS)].resample('ME').sum().loc['2021-01':'2025-12']
        corr[col] = {k: m.corr(v.resample('ME').sum().reindex(m.index)) for k, v in ex.items()}
    corr = pd.DataFrame(corr).T
    emit(corr.to_string(float_format=lambda v: f'{v:+.2f}'))

    emit("\nDECISION (pre-registered)")
    a1, a2 = df.loc['A1'], df.loc['A2']
    pick = 'A2' if a2.Sharpe >= a1.Sharpe + 0.10 else 'A1'
    r = df.loc[pick]
    c = corr.loc[pick] if pick in corr.index else pd.Series(dtype=float)
    t = {'L1 Sharpe >= 0.50': r.Sharpe >= 0.50, 'L2 last 2y > 0': r['last 2y'] > 0, 'L3 concentration <= 75%': r.conc <= 0.75,
         'L4 beats A0 by 0.10': r.Sharpe >= df.loc['A0'].Sharpe + 0.10,
         'L5 |corr| < 0.30': bool(len(c)) and bool((c.abs() < 0.30).all())}
    emit(f"Model chosen by the rule: {pick} ({'LightGBM' if pick == 'A2' else 'ridge'}); Sharpe {r.Sharpe:.2f}, last 2y {r['last 2y']:+.2f}, "
         f"concentration {r.conc:.0%}, A0 {df.loc['A0'].Sharpe:.2f}, max |corr| {c.abs().max() if len(c) else float('nan'):.2f}")
    for k, v in t.items():
        emit(f"  {k}: {'ok' if v else 'FAIL'}")
    emit(f"VERDICT: {'PASS - candidate for a paper forward test' if all(t.values()) else 'FAIL'}")
    with open(os.path.join(OUT, 'report.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    emit(f"({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
