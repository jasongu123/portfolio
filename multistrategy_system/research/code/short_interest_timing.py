"""
Aggregate short-interest market timing — RESEARCH ONLY, DATA-LIMITED (pre-registered 2026-09-04).

Reference: Rapach, Ringgenberg & Zhou, "Short interest and aggregate stock returns" (JFE 2016);
Gorbenko, "Short Interest and Aggregate Stock Returns: International Evidence" (RAPS 2023).
Those papers use decades of Compustat short interest. The only free source is FINRA's
consolidated short-interest API, which starts 2020-04. This is therefore a ~6-year
preliminary with roughly 50 out-of-sample months, not a test with statistical power. Its
job is to build the data path and record a first read; it cannot promote anything on its own.

Data (cached, resumable): results/finra_short_interest_agg.csv, one row per settlement date.
Exchange-listed classes only (NYSE, NNM, SC, AMEX); OTC, ARCA and BZX excluded so ETF and
pink-sheet short positions do not dominate. Per date:
  ew_dtc  = equal-weighted mean days-to-cover across names with ADV >= 100k shares,
            winsorised at the 1st/99th percentiles
  agg_dtc = total short quantity / total average daily volume
FINRA has no shares-outstanding field, so the papers' short-interest-to-shares ratio cannot
be built; days-to-cover is the closest free analogue and is stated as such.

Timing rule (pre-registered): use the MID-MONTH settlement (published about eight business
days later, so known before the next month starts) to predict the NEXT calendar month's SPY
excess return. Signal = expanding-window linear-detrended log(ew_dtc), standardised expanding
(min 24 observations). Walk-forward: expanding OLS forecast (min 24 months),
w = clip(forecast / (3 * trailing 36-month variance), 0, 1.5), realised on SPY, 1 bp cost.

Reported: predictive-regression slope and HAC t-stat, walk-forward net Sharpe versus SPY,
correlation with the equity and trend sleeves. No go/no-go is attached: the sample is too
short to pass or fail anything. Trial registry: 1 trial (data-limited).

Usage:  python3 short_interest_timing.py          # fetch (cached) + analyse
"""

import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
CACHE = os.path.join(RESULTS_DIR, 'finra_short_interest_agg.csv')
API = 'https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest'
CLASSES = ['NYSE', 'NNM', 'SC', 'AMEX']
FIELDS = ['symbolCode', 'marketClassCode', 'currentShortPositionQuantity',
          'averageDailyVolumeQuantity', 'daysToCoverQuantity']
MIN_ADV = 100_000
START = dt.date(2020, 4, 1)
PAGE = 5000
TREND_CACHE = os.path.join(SCRIPT_DIR, 'trend_cache.csv')
FF_CSV = os.path.join(SCRIPT_DIR, 'backtest_data', 'F-F_Research_Data_Factors.csv')
TREND_V3 = os.path.join(RESULTS_DIR, 'trend_v3_returns.csv')


# --------------------------------------------------------------------------- fetch
def _post(body, retries=4):
    for i in range(retries):
        try:
            r = requests.post(API, json=body, headers={'Accept': 'application/json'}, timeout=90)
            if r.status_code == 200:
                return r.json(), int(r.headers.get('record-total', -1))
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (i + 1))
                continue
            return None, -1
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return None, -1


def _exists(date):
    body = {'limit': 1, 'fields': ['settlementDate'],
            'compareFilters': [{'compareType': 'EQUAL', 'fieldName': 'settlementDate', 'fieldValue': date.isoformat()}]}
    data, total = _post(body)
    return bool(data)


def candidate_dates(start, end):
    """FINRA settlement dates: the 15th and the last calendar day of each month, moved back to a
    business day when needed. We probe +/- 2 business days around each to be safe."""
    out = []
    d = pd.Timestamp(start).to_period('M')
    while d <= pd.Timestamp(end).to_period('M'):
        for anchor in (pd.Timestamp(d.year, d.month, 15), d.to_timestamp(how='end').normalize()):
            for shift in (0, -1, -2, 1, 2):
                c = anchor + pd.offsets.BDay(shift) if shift else anchor
                out.append((anchor, c.date()))
        d += 1
    return out


def fetch_date(date):
    """Returns None (and caches nothing) if any page fails or a class comes back incomplete, so
    the date is retried next run instead of being frozen as complete (fixed 2026-09-05)."""
    rows = []
    for cls in CLASSES:
        offset, expected, got = 0, None, 0
        while True:
            body = {'limit': PAGE, 'offset': offset, 'fields': FIELDS,
                    'compareFilters': [{'compareType': 'EQUAL', 'fieldName': 'settlementDate', 'fieldValue': date.isoformat()},
                                       {'compareType': 'EQUAL', 'fieldName': 'marketClassCode', 'fieldValue': cls}]}
            data, total = _post(body)
            if data is None:
                return None
            if expected is None:
                expected = total
            rows.extend(data)
            got += len(data)
            offset += PAGE
            if len(data) < PAGE:
                break
        if expected is not None and expected >= 0 and got != expected:
            return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for c in ('currentShortPositionQuantity', 'averageDailyVolumeQuantity', 'daysToCoverQuantity'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[(df.averageDailyVolumeQuantity >= MIN_ADV) & (df.currentShortPositionQuantity > 0)]
    if df.empty:
        return None
    dtc = df.currentShortPositionQuantity / df.averageDailyVolumeQuantity
    lo, hi = dtc.quantile(0.01), dtc.quantile(0.99)
    return {'settlementDate': date.isoformat(), 'n': int(len(df)),
            'ew_dtc': float(dtc.clip(lo, hi).mean()),
            'agg_dtc': float(df.currentShortPositionQuantity.sum() / df.averageDailyVolumeQuantity.sum()),
            'total_short': float(df.currentShortPositionQuantity.sum())}


def build_cache():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=['settlementDate'])
    have = set(done.settlementDate.astype(str))
    found_anchor = set()
    if len(done):
        found_anchor = {pd.Timestamp(s).to_period('M').strftime('%Y-%m') + ('-mid' if pd.Timestamp(s).day <= 20 else '-end')
                        for s in done.settlementDate}
    end = dt.date.today()
    records = done.to_dict('records')
    for anchor, cand in candidate_dates(START, end):
        key = anchor.to_period('M').strftime('%Y-%m') + ('-mid' if anchor.day == 15 else '-end')
        if key in found_anchor or cand > end:
            continue
        if not _exists(cand):
            continue
        rec = fetch_date(cand)
        if rec is None:
            continue
        records.append(rec)
        found_anchor.add(key)
        pd.DataFrame(records).sort_values('settlementDate').to_csv(CACHE, index=False)
        print(f"  {cand} n={rec['n']:5d} ew_dtc={rec['ew_dtc']:.2f} agg_dtc={rec['agg_dtc']:.2f}", flush=True)
    return pd.DataFrame(records).sort_values('settlementDate') if records else pd.DataFrame()


# --------------------------------------------------------------------------- analysis
def load_spy_monthly():
    px = pd.read_csv(TREND_CACHE, index_col=0, parse_dates=True)['SPY'].dropna()
    last = px.index[-1]
    if last != (last + pd.offsets.BMonthEnd(0)):
        px = px[px.index < last.to_period('M').to_timestamp()]
    m = px.resample('ME').last().pct_change().dropna()
    m.index = m.index.to_period('M')
    return m


def load_rf_monthly():
    rows = []
    with open(FF_CSV) as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 5 and len(parts[0]) == 6 and parts[0].isdigit():
                rows.append((parts[0], float(parts[4]) / 100.0))
    s = pd.Series(dict(rows))
    s.index = pd.PeriodIndex(pd.to_datetime(s.index, format='%Y%m'), freq='M')
    return s


def monthly_series(sleeve_daily):
    m = (1 + sleeve_daily).resample('ME').prod() - 1
    m.index = m.index.to_period('M')
    return m


def hac_tstat(y, x, lags=3):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    n, k = X.shape
    S = np.zeros((k, k))
    for lag in range(lags + 1):
        w = 1.0 - lag / (lags + 1)
        for t in range(lag, n):
            g = np.outer(X[t] * e[t], X[t - lag] * e[t - lag])
            S += w * (g + g.T if lag else g)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    return beta[1], beta[1] / np.sqrt(V[1, 1])


def analyse(agg):
    agg = agg.copy()
    agg['settlementDate'] = pd.to_datetime(agg.settlementDate)
    mid = agg[agg.settlementDate.dt.day <= 20].set_index('settlementDate').sort_index()
    sig_month = mid.index.to_period('M')
    level = pd.Series(np.log(mid.ew_dtc.values), index=sig_month)

    # expanding linear detrend, then expanding standardisation (min 24)
    detr, z = [], []
    t = np.arange(len(level))
    for i in range(len(level)):
        if i + 1 < 24:
            detr.append(np.nan)
            continue
        b = np.polyfit(t[:i + 1], level.values[:i + 1], 1)
        detr.append(level.values[i] - np.polyval(b, t[i]))
    detr = pd.Series(detr, index=level.index)
    for i in range(len(detr)):
        h = detr.iloc[:i + 1].dropna()
        z.append((detr.iloc[i] - h.mean()) / h.std() if len(h) >= 12 and h.std() > 0 else np.nan)
    z = pd.Series(z, index=level.index, name='sii')

    spy = load_spy_monthly()
    rf = load_rf_monthly()
    ex = (spy - rf.reindex(spy.index)).rename('spy_excess')
    # signal in month m predicts month m+1
    pairs = pd.concat([z.shift(0), ex.shift(-1).reindex(z.index)], axis=1).dropna()
    pairs.columns = ['sii', 'next_excess']
    print(f"\nMid-month observations: {len(level)} ({level.index.min()}..{level.index.max()}); usable pairs: {len(pairs)}")
    if len(pairs) < 12:
        print("Too few observations to analyse.")
        return
    b, tst = hac_tstat(pairs.next_excess.values, pairs.sii.values)
    r2 = np.corrcoef(pairs.sii, pairs.next_excess)[0, 1] ** 2
    print(f"Predictive regression next-month SPY excess on SII: slope {b*100:+.2f}% per 1 s.d., HAC t {tst:+.2f}, R2 {r2:.3f}")
    print("  (RRZ report a NEGATIVE slope: high short interest -> low future returns)")

    # walk-forward timing
    w, fc = [], []
    for i in range(len(pairs)):
        if i < 24:
            w.append(np.nan); fc.append(np.nan); continue
        hx, hy = pairs.sii.values[:i], pairs.next_excess.values[:i]
        bb = np.polyfit(hx, hy, 1)
        f = np.polyval(bb, pairs.sii.values[i])
        var = np.var(hy[-36:]) if i >= 36 else np.var(hy)
        w.append(float(np.clip(f / (3 * var), 0, 1.5))); fc.append(f)
    pairs['weight'] = w
    pairs['forecast'] = fc
    ret_month = pairs.index + 1
    pos = pd.Series(pairs.weight.values, index=ret_month).dropna()
    realised = spy.reindex(pos.index)
    turnover = (pos - pos.shift(1).fillna(0)).abs()
    strat = (pos * realised - turnover * 1e-4).dropna()
    bh = spy.reindex(strat.index)
    sr = lambda r: float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan

    def mdd(r):   # high-water mark starts at initial capital (fixed 2026-09-05)
        eq = (1 + r.dropna()).cumprod().values
        peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
        return float((eq / peak - 1).min()) if len(eq) else 0.0
    print(f"Walk-forward timing {strat.index.min()}..{strat.index.max()} ({len(strat)} months): "
          f"net Sharpe {sr(strat):.2f} vs SPY buy-and-hold {sr(bh):.2f}; avg weight {pos.mean():.2f}; "
          f"MDD {mdd(strat):.1%} vs SPY {mdd(bh):.1%}")

    # correlations with sleeves
    parts = []
    for y in range(2020, 2026):
        f = os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv')
        if os.path.exists(f):
            parts.append(pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0])
    if parts and os.path.exists(TREND_V3):
        eq = monthly_series(pd.concat(parts).sort_index()).rename('equity')
        tr = monthly_series(pd.read_csv(TREND_V3, index_col=0, parse_dates=True)['V1_trail'].dropna()).rename('trend')
        j = pd.concat([eq, tr, strat.rename('si_timing'), bh.rename('spy')], axis=1).dropna()
        print(f"Monthly correlations over {len(j)} overlapping months:\n{j.corr().round(2).to_string()}")
    pairs.to_csv(os.path.join(RESULTS_DIR, 'short_interest_timing_pairs.csv'))
    print("Saved results/short_interest_timing_pairs.csv")


if __name__ == '__main__':
    if '--no-fetch' in sys.argv and os.path.exists(CACHE):
        agg = pd.read_csv(CACHE)
    else:
        print("Fetching FINRA consolidated short interest (cached; safe to re-run)...", flush=True)
        agg = build_cache()
    if agg is None or agg.empty:
        print("No data fetched.")
        sys.exit(1)
    print(f"Cache: {len(agg)} settlement dates {agg.settlementDate.min()}..{agg.settlementDate.max()}")
    analyse(agg)
