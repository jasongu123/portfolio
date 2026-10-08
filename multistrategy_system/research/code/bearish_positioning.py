"""
Bearish-positioning composite: "follow the hedging" market timing. RESEARCH ONLY.
Pre-registered 2026-09-04 before any composite result was seen.

Motivation: the stock-level short-interest index (Rapach-Ringgenberg-Zhou) stopped moving after
2008 because bearish positioning migrated to index products. This composite measures bearish
positioning where it now lives. It is an original recombination of known series, not a paper
replication, so there is nothing to replicate; the evidence has to come from the walk-forward.

Components (all free, all to the present)
  SKEW      CBOE SKEW index, monthly mean of daily closes.  High = more tail-hedge demand.
  SPX_COT   CFTC legacy Commitments of Traders, non-commercial net long (long - short) as a
            fraction of open interest in S&P 500 futures (big contract + E-mini, E-mini scaled
            by 1/5; micro and "Consolidated" rows ignored). High = speculators long = bullish,
            so it enters with a MINUS sign.
  VIX_COT   Same for VIX futures (CFE). Non-commercials net long VIX = long volatility = bearish.
  Informational only (data end 2019-10): CBOE equity put/call ratio, monthly mean.

Timing of information: SKEW through month-end; COT report as-of Tuesday must be at least three
days before month-end (reports are released the Friday of the same week), so nothing used at
month-end t was published after t.

Composite: each component z-scored on an expanding window (minimum 36 months), signed so that
higher = more bearish positioning, then averaged; all legs required.
  Trial A (primary):   SKEW + SPX_COT + VIX_COT  (VIX futures exist from 2004; OOS from ~2012)
  Trial B (secondary): SKEW + SPX_COT            (E-mini from 1997; OOS from ~2005)

Rule: expanding OLS of next-month market excess return (French Mkt-RF) on the composite, data
through t, minimum 60 months; the SIGN IS ESTIMATED, not imposed. Weight = clip(forecast /
(3 * trailing 60-month variance), 0, 1.5), realised the following month; 1 bp per unit turnover.
Realised on the French market total return for the OOS window and on SPY for holdout and book.

Pre-registered gate (per trial; A is the one that counts, B is reported)
  G1 OOS window (first forecast .. 2021-12): Campbell-Thompson OOS R^2 > 0 AND timing net
     Sharpe > buy-and-hold AND MaxDD not deeper than buy-and-hold.                 binding
  G2 holdout 2022-01 .. latest on SPY: estimated slope has the same sign as at 2021-12 AND
     timing Sharpe >= buy-and-hold - 0.10.                                        informational
  G3 overlay value: active return (rule minus market) over OOS + holdout has net Sharpe > 0,
     AND its 2020-2025 monthly correlation with the equity sleeve and with trend V1_trail are
     each below 0.30.                                                             binding
  G4 book 2020-2025, inverse-vol, monthly: equity + trend + ACTIVE overlay has no-rf Sharpe >=
     equity + trend AND MaxDD not deeper. rf-Sharpe reported, not used.           binding
  Mechanism prediction, written first: if the composite's estimated sign is POSITIVE (bearish
  positioning -> higher returns) it is a fear gauge and hedgers are uninformed; if NEGATIVE,
  hedgers are informed like stock-level shorts were. Either is recorded.
Trial registry: 2 trials (A, B). Component univariate slopes are reported, not selected on.

Inputs: backtest_data/SKEW_History.csv, backtest_data/VIX_History.csv (CBOE),
        backtest_data/equitypc*.csv (CBOE, informational), COT annual zips (COT_DIR, default
        scratchpad; parsed once and cached to results/cot_sp_vix_weekly.csv),
        backtest_data/F-F_Research_Data_Factors.csv, trend_cache.csv, equity_returns_*.csv,
        results/trend_v3_returns.csv.
Outputs: results/bearish_positioning_monthly.csv, console report.
Usage:  COT_DIR=/path/to/deacot_zips python3 bearish_positioning.py
"""

import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
BD = os.path.join(SCRIPT_DIR, 'backtest_data')
COT_DIR = os.environ.get('COT_DIR', '')
COT_CACHE = os.path.join(RESULTS_DIR, 'cot_sp_vix_weekly.csv')
FF_CSV = os.path.join(BD, 'F-F_Research_Data_Factors.csv')
TREND_CACHE = os.path.join(SCRIPT_DIR, 'trend_cache.csv')
TREND_V3 = os.path.join(RESULTS_DIR, 'trend_v3_returns.csv')

Z_MIN = 36
MIN_TRAIN = 60
GAMMA, CAP, COST_BPS = 3.0, 1.5, 1.0
HOLDOUT_START = '2022-01'
EVAL = ('2020-01', '2025-12')
RF_FLAT_M = 0.05 / 12


# --------------------------------------------------------------------------- loaders
def load_cboe_index(path, col):
    df = pd.read_csv(path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    s = df.set_index('DATE')[col].astype(float)
    m = s.resample('ME').mean()
    m.index = m.index.to_period('M')
    return m


def load_equity_pc():
    frames = []
    for f in (os.path.join(BD, 'equitypcarchive.csv'), os.path.join(BD, 'equitypc.csv')):
        if not os.path.exists(f):
            continue
        raw = open(f, errors='ignore').read().splitlines()
        start = next((i for i, l in enumerate(raw) if l.upper().startswith('DATE')), None)
        if start is None:
            continue
        df = pd.read_csv(io.StringIO("\n".join(raw[start:])))
        df.columns = [c.strip().upper() for c in df.columns]
        ratio_col = next((c for c in df.columns if 'RATIO' in c or 'P/C' in c), None)
        if ratio_col is None:
            continue
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        frames.append(df.dropna(subset=['DATE']).set_index('DATE')[ratio_col].astype(float))
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep='last')]
    m = s.resample('ME').mean()
    m.index = m.index.to_period('M')
    return m.rename('equity_pc')


def parse_cot():
    if os.path.exists(COT_CACHE):
        w = pd.read_csv(COT_CACHE, parse_dates=['date'])
        return w
    rows = []
    for z in sorted(glob.glob(os.path.join(COT_DIR, 'cot*.zip'))):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith('.txt')][0]
            txt = zf.read(name).decode('latin-1')
        df = pd.read_csv(io.StringIO(txt), dtype=str, on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        need = {'Market and Exchange Names': 'name', 'As of Date in Form YYYY-MM-DD': 'date',
                'Open Interest (All)': 'oi', 'Noncommercial Positions-Long (All)': 'ncl',
                'Noncommercial Positions-Short (All)': 'ncs'}
        if not set(need).issubset(df.columns):
            continue
        d = df[list(need)].rename(columns=need)
        d['name'] = d['name'].str.strip().str.upper()
        mask = d['name'].str.contains(r'^(?:S&P 500 STOCK INDEX|E-MINI S&P 500|VIX FUTURES)', regex=True) & \
               ~d['name'].str.contains('MICRO|CONSOLIDATED|DIVIDEND|ADJUSTED|ANNUAL|QUARTERLY')
        d = d[mask].copy()
        for c in ('oi', 'ncl', 'ncs'):
            d[c] = pd.to_numeric(d[c].str.strip(), errors='coerce')
        rows.append(d)
    w = pd.concat(rows)
    w['date'] = pd.to_datetime(w['date'])
    w['group'] = np.where(w['name'].str.startswith('VIX'), 'VIX', 'SPX')
    w['scale'] = np.where(w['name'].str.startswith('E-MINI'), 0.2, 1.0)
    w = w.dropna(subset=['oi', 'ncl', 'ncs'])
    w.to_csv(COT_CACHE, index=False)
    return w


# Actual COT publication dates where they differ from the normal Friday release (as-of Tuesday + 3
# days), from CFTC press releases 6745-13, 7864-19 and 9138-25 (fixed 2026-09-05; previously the
# 3-day rule was assumed throughout, which used October 2025 data published in December).
COT_RELEASE_OVERRIDES = {
    # 2013 shutdown: first delayed report published 2013-10-25, two per week after, normal by 11-08
    '2013-10-01': '2013-10-25', '2013-10-08': '2013-10-29', '2013-10-15': '2013-11-01',
    '2013-10-22': '2013-11-05', '2013-10-29': '2013-11-08',
    # 2018-19 shutdown: 2018-12-24 report published 2019-02-01, then Tuesday/Friday until current
    '2018-12-24': '2019-02-01', '2018-12-31': '2019-02-05', '2019-01-07': '2019-02-08',
    '2019-01-14': '2019-02-12', '2019-01-22': '2019-02-15', '2019-01-29': '2019-02-19',
    '2019-02-05': '2019-02-22', '2019-02-12': '2019-02-26', '2019-02-19': '2019-03-01',
    '2019-02-26': '2019-03-05',
    # 2025 lapse in appropriations: published schedule
    '2025-09-30': '2025-11-19', '2025-10-07': '2025-11-21', '2025-10-14': '2025-11-25',
    '2025-10-21': '2025-12-02', '2025-10-28': '2025-12-05', '2025-11-04': '2025-12-09',
    '2025-11-10': '2025-12-12', '2025-11-18': '2025-12-16', '2025-11-25': '2025-12-19',
    '2025-12-02': '2025-12-23', '2025-12-09': '2025-12-30', '2025-12-16': '2026-01-06',
    '2025-12-23': '2026-01-09', '2025-12-30': '2026-01-13', '2026-01-06': '2026-01-16',
    '2026-01-13': '2026-01-20', '2026-01-20': '2026-01-23',
}


def cot_release_date(as_of):
    key = pd.Timestamp(as_of).strftime('%Y-%m-%d')
    if key in COT_RELEASE_OVERRIDES:
        return pd.Timestamp(COT_RELEASE_OVERRIDES[key])
    return pd.Timestamp(as_of) + pd.Timedelta(days=3)


def cot_monthly(w, group):
    g = w[w.group == group].copy()
    g['net'] = (g.ncl - g.ncs) * g.scale
    g['oi_s'] = g.oi * g.scale
    agg = g.groupby('date')[['net', 'oi_s']].sum()
    agg = agg[agg.oi_s > 0]
    pct = (agg.net / agg.oi_s).rename(f'{group.lower()}_cot')
    released = pd.Series([cot_release_date(d) for d in pct.index], index=pct.index)
    # month-end value: the latest report whose actual publication date is on or before month-end
    out = {}
    for m in pd.period_range(pct.index.min().to_period('M'), pct.index.max().to_period('M'), freq='M'):
        me = m.to_timestamp(how='end').normalize()
        ok = pct[(released <= me) & (pct.index <= me)]
        if len(ok):
            out[m] = float(ok.iloc[-1])
    return pd.Series(out).rename(pct.name)


def load_ff():
    rows = []
    with open(FF_CSV) as f:
        for line in f:
            p = [q.strip() for q in line.split(',')]
            if len(p) == 5 and len(p[0]) == 6 and p[0].isdigit():
                rows.append(p)
    df = pd.DataFrame(rows, columns=['ym', 'mktrf', 'smb', 'hml', 'rf']).set_index('ym').astype(float) / 100
    df.index = pd.PeriodIndex(pd.to_datetime(df.index, format='%Y%m'), freq='M')
    return df['mktrf'], df['rf']


def load_spy():
    px = pd.read_csv(TREND_CACHE, index_col=0, parse_dates=True)['SPY'].dropna()
    last = px.index[-1]
    if last != (last + pd.offsets.BMonthEnd(0)):
        px = px[px.index < last.to_period('M').to_timestamp()]
    m = px.resample('ME').last().pct_change().dropna()
    m.index = m.index.to_period('M')
    return m


def monthly_from_daily(d):
    m = (1 + d).resample('ME').prod() - 1
    m.index = m.index.to_period('M')
    return m


# --------------------------------------------------------------------------- composite & rule
def expanding_z(s, min_obs=Z_MIN):
    mu = s.expanding(min_obs).mean().shift(0)
    sd = s.expanding(min_obs).std()
    return (s - mu) / sd


def composite(parts):
    z = pd.concat([expanding_z(p[0]) * p[1] for p in parts], axis=1).dropna()
    return z.mean(axis=1).rename('composite')


def hac_t(y, x, lags=3):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    n, k = X.shape
    S = np.zeros((k, k))
    for lag in range(lags + 1):
        wgt = 1 - lag / (lags + 1)
        for t in range(lag, n):
            g = np.outer(X[t] * e[t], X[t - lag] * e[t - lag])
            S += wgt * (g + g.T if lag else g)
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    return beta, beta[1] / np.sqrt(V[1, 1])


def walk_forward(sig, excess, min_train=MIN_TRAIN):
    j = pd.concat([sig.rename('x'), excess.shift(-1).rename('y_next')], axis=1).dropna()
    fc, w, pm, slope = [], [], [], []
    for i in range(len(j)):
        if i < min_train:
            fc.append(np.nan); w.append(np.nan); pm.append(np.nan); slope.append(np.nan); continue
        hx, hy = j.x.values[:i], j.y_next.values[:i]
        b = np.polyfit(hx, hy, 1)
        f = np.polyval(b, j.x.values[i])
        var = np.var(hy[-60:])
        fc.append(f); pm.append(hy.mean()); slope.append(b[0])
        w.append(float(np.clip(f / (GAMMA * var), 0, CAP)))
    j['forecast'], j['prev_mean'], j['weight'], j['slope'] = fc, pm, w, slope
    return j.dropna()


def oos_r2(j):
    return float(1 - ((j.y_next - j.forecast) ** 2).sum() / ((j.y_next - j.prev_mean) ** 2).sum())


def realise(j, ret):
    pos = pd.Series(j.weight.values, index=j.index + 1)
    r = ret.reindex(pos.index)
    turnover = (pos - pos.shift(1).fillna(0)).abs()
    strat = (pos * r - turnover * COST_BPS / 1e4).dropna()
    return strat, pos.reindex(strat.index), r.reindex(strat.index)


def sr(r, rf=0.0):
    r = r.dropna()
    if len(r) < 6 or r.std() == 0:
        return np.nan
    ex = r - (rf.reindex(r.index) if isinstance(rf, pd.Series) else rf)
    return float(ex.mean() / r.std() * np.sqrt(12))


def mdd(r):
    """High-water mark starts at initial capital, so a first-period loss counts (fixed 2026-09-05)."""
    r = r.dropna()
    eq = (1 + r).cumprod().values
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((eq / peak - 1).min()) if len(eq) else 0.0


# --------------------------------------------------------------------------- main
def run_trial(label, sig, mktrf, rf, spy, eq_m, tr_m):
    print("\n" + "=" * 100 + f"\nTRIAL {label}: composite {sig.index.min()}..{sig.index.max()} ({len(sig)} months)")
    ins = pd.concat([sig.rename('x'), mktrf.shift(-1).rename('y')], axis=1).dropna()
    b_all, t_all = hac_t(ins.y.values, ins.x.values)
    print(f"  full-sample slope (informational): {b_all[1]*100:+.3f}%/month per s.d., HAC t {t_all:+.2f}, n={len(ins)}")
    wf = walk_forward(sig, mktrf)
    mkt_tot = mktrf + rf.reindex(mktrf.index)
    pre = wf.loc[:pd.Period('2021-11', 'M')]
    s_pre, w_pre, m_pre = realise(pre, mkt_tot)
    r2 = oos_r2(pre)
    sign_2021 = np.sign(pre.slope.iloc[-1])
    print(f"  G1 OOS {s_pre.index.min()}..2021-12 ({len(s_pre)} months): OOS R^2 {r2*100:+.2f}%, timing net Sharpe {sr(s_pre):.2f} "
          f"vs market {sr(m_pre):.2f}, MDD {mdd(s_pre):.1%} vs {mdd(m_pre):.1%}, avg weight {w_pre.mean():.2f}, "
          f"slope sign at 2021-12 {'+' if sign_2021 > 0 else '-'} ({pre.slope.iloc[-1]*100:+.3f}%/s.d.)")
    g1 = (r2 > 0) and (sr(s_pre) > sr(m_pre)) and (mdd(s_pre) >= mdd(m_pre))
    hold = wf.loc[pd.Period(HOLDOUT_START, 'M') - 1:]
    s_h, w_h, m_h = realise(hold, spy)
    # slope on SPY excess over the realised months only (fixed 2026-09-05)
    spy_ex = spy - rf.reindex(spy.index)
    hj = pd.concat([hold.x, spy_ex.shift(-1).reindex(hold.index)], axis=1).dropna()
    hj.columns = ['x', 'y']
    hb, ht = hac_t(hj.y.values, hj.x.values) if len(hj) >= 12 else ((np.nan, np.nan), np.nan)
    print(f"  G2 holdout {s_h.index.min()}..{s_h.index.max()} on SPY ({len(s_h)} months): slope on SPY excess ({len(hj)} obs) {hb[1]*100:+.3f}% t {ht:+.2f}; "
          f"timing Sharpe {sr(s_h):.2f} vs SPY {sr(m_h):.2f}; avg weight {w_h.mean():.2f}; MDD {mdd(s_h):.1%} vs {mdd(m_h):.1%}")
    g2 = (np.sign(hb[1]) == sign_2021) and (sr(s_h) >= sr(m_h) - 0.10)
    # active overlay over OOS + holdout, on the market pre-2007 / SPY after
    s_all, w_all, m_all = realise(wf, mkt_tot)
    active = (s_all - m_all).rename('active')
    g3_sr = sr(active)
    corr_eq = corr_tr = np.nan
    g4 = False
    if eq_m is not None and tr_m is not None:
        s_spy, w_spy, m_spy = realise(wf, spy)
        act_spy = (s_spy - m_spy).rename('active')
        jb = pd.concat([eq_m, tr_m, act_spy], axis=1).loc[pd.Period(EVAL[0], 'M'):pd.Period(EVAL[1], 'M')].dropna()
        c = jb.corr()
        corr_eq, corr_tr = float(c.loc['active', 'equity']), float(c.loc['active', 'trend'])

        def book(cols):
            inv = 1 / jb[cols].std(); wts = inv / inv.sum(); bk = (jb[cols] * wts).sum(axis=1)
            yr = {y: sr(x, RF_FLAT_M) for y, x in bk.groupby(bk.index.year)}
            return {'norf': sr(bk), 'rf': sr(bk, RF_FLAT_M), 'mdd': mdd(bk), 'vol': bk.std() * np.sqrt(12), 'worst': min(yr.values()), 'w': wts.round(2).to_dict()}
        b2, b3 = book(['equity', 'trend']), book(['equity', 'trend', 'active'])
        print(f"  G3 active overlay {active.index.min()}..{active.index.max()}: net Sharpe {g3_sr:.2f}; 2020-25 corr with equity {corr_eq:+.2f}, trend {corr_tr:+.2f} ({len(jb)} months)")
        print(f"  G4 book 2020-2025: equity+trend no-rf {b2['norf']:.2f} rf {b2['rf']:.2f} MDD {b2['mdd']:.1%} worst {b2['worst']:+.2f} | "
              f"+active no-rf {b3['norf']:.2f} rf {b3['rf']:.2f} MDD {b3['mdd']:.1%} worst {b3['worst']:+.2f} w {b3['w']}")
        g4 = (b3['norf'] >= b2['norf']) and (b3['mdd'] >= b2['mdd'])
    g3 = (g3_sr > 0) and (corr_eq < 0.30) and (corr_tr < 0.30)
    print(f"  GATE: G1 {'PASS' if g1 else 'fail'} | G2 {'PASS' if g2 else 'fail'} (informational) | G3 {'PASS' if g3 else 'fail'} | G4 {'PASS' if g4 else 'fail'}"
          f"  -> {'PROMOTE' if (g1 and g3 and g4) else 'not promoted'}")
    return wf, active


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    skew = load_cboe_index(os.path.join(BD, 'SKEW_History.csv'), 'SKEW').rename('skew')
    cot = parse_cot()
    spx_cot = cot_monthly(cot, 'SPX')
    vix_cot = cot_monthly(cot, 'VIX')
    pc = load_equity_pc()
    mktrf, rf = load_ff()
    spy = load_spy()
    print(f"SKEW {skew.index.min()}..{skew.index.max()} | SPX COT {spx_cot.index.min()}..{spx_cot.index.max()} ({len(spx_cot)}) | "
          f"VIX COT {vix_cot.index.min()}..{vix_cot.index.max()} ({len(vix_cot)}) | equity P/C {'none' if pc is None else f'{pc.index.min()}..{pc.index.max()}'}")

    print("\nUnivariate next-month Mkt-RF slopes (informational, full sample, signed as 'bearish'):")
    for name, s, sign in (('SKEW', skew, 1), ('SPX COT net spec (minus)', spx_cot, -1), ('VIX COT net spec', vix_cot, 1)) + ((('equity P/C', pc, 1),) if pc is not None else ()):
        z = expanding_z(s) * sign
        j = pd.concat([z.rename('x'), mktrf.shift(-1).rename('y')], axis=1).dropna()
        b, t = hac_t(j.y.values, j.x.values)
        print(f"  {name:26s} {j.index.min()}..{j.index.max()} n={len(j):4d}  slope {b[1]*100:+.3f}%/s.d.  HAC t {t:+.2f}")

    parts = []
    for y in range(2020, 2026):
        f = os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv')
        if os.path.exists(f):
            parts.append(pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0])
    eq_m = monthly_from_daily(pd.concat(parts).sort_index()).rename('equity') if parts else None
    tr_m = monthly_from_daily(pd.read_csv(TREND_V3, index_col=0, parse_dates=True)['V1_trail'].dropna()).rename('trend') if os.path.exists(TREND_V3) else None

    compA = composite([(skew, 1), (spx_cot, -1), (vix_cot, 1)])
    compB = composite([(skew, 1), (spx_cot, -1)])
    wfA, actA = run_trial('A: SKEW + SPX COT + VIX COT', compA, mktrf, rf, spy, eq_m, tr_m)
    wfB, actB = run_trial('B: SKEW + SPX COT', compB, mktrf, rf, spy, eq_m, tr_m)

    out = pd.concat([skew, spx_cot, vix_cot, compA.rename('composite_A'), compB.rename('composite_B'),
                     wfA.weight.rename('weight_A'), wfB.weight.rename('weight_B'), mktrf.rename('mktrf'), spy.rename('spy')], axis=1)
    if pc is not None:
        out = out.join(pc)
    out.to_csv(os.path.join(RESULTS_DIR, 'bearish_positioning_monthly.csv'))
    print("\nSaved results/bearish_positioning_monthly.csv")
