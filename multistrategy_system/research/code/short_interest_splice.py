"""
Aggregate short-interest timing with the authors' own index, spliced to the present.
RESEARCH ONLY (pre-registered 2026-09-04, before the holdout was computed).

Sources
  - Rapach, Ringgenberg & Zhou (JFE 2016) short interest index, updated by Ringgenberg to
    1973-01..2021-12: columns siiInSample (full-sample linear detrend of log equal-weighted
    short interest, standardised) and siiOutOfSample (recursive detrend, from 1978-01).
    Downloaded from https://www.matthewringgenberg.com/data for research; kept OUT of the repo.
    Local copy: SII_XLSX (scratchpad path passed on the command line or env RRZ_SII_XLSX).
  - Raw EWSI 1973-01..2014-12 from the JFE replication package (same site), used only to recover
    the affine map from siiInSample back to log EWSI so the raw level can be extended.
  - FINRA consolidated short interest (free API, 2020-04 onward), per-symbol, exchange-listed
    classes NYSE/NNM/SC/AMEX.
  - SEC XBRL company-facts: dei:EntityCommonStockSharesOutstanding per filer, point-in-time by
    filing date, to turn FINRA short quantities into the papers' short-interest-to-shares ratio.
  - Ken French Mkt-RF and RF (backtest_data/F-F_Research_Data_Factors.csv) for market excess
    returns; SPY monthly (trend_cache.csv) for the implementable series.

Construction
  1. log EWSI 1973-01..2021-12 = a + b*t + s*siiInSample, with (a, b, s) fitted on the 1973-2014
     overlap with the raw EWSI (R^2 checked at run time; must be ~1).
  2. Proxy 2020-04..2026-08: equal-weighted mean across mapped names of short quantity / shares
     outstanding as of the latest SEC filing filed on or before the settlement date (filing no
     older than 400 days; shares >= 1m; ratio <= 100%; trimmed 0.5/99.5). Mid-month settlement
     only, one observation per month.
  3. Splice: OLS of log EWSI on log proxy over the 2020-04..2021-12 overlap (21 months); the
     fitted map extends log EWSI to 2026-08. The overlap correlation is reported; if it is below
     0.6 the splice is declared unreliable and the holdout results are labelled as such.
  4. SII_oos_t: recursive linear detrend of log EWSI using data through t, residual standardised
     by the standard deviation of the recursive-fit residuals through t, from 1978-01. This does
     NOT reproduce the authors' siiOutOfSample column (correlation about 0.88 overall and near
     zero after 2010; their exact recursive construction is not documented in the files). So:
     G2 uses the authors' own OOS column unchanged; the holdout (G3) and the book test (G4) use
     this internally consistent recursive construction, which is the only way to carry one
     definition across the 2021-12 seam. G2b reports the same 1990-2021 test on this
     construction so the two can be compared.
  5. Timing rule (RRZ Section 5 style): forecast_t = alpha + beta * SII_oos_t from an expanding
     OLS of next-month market excess return on SII_oos using data through t; weight
     w_t = clip(forecast_t / (3 * trailing 60-month variance), 0, 1.5); realised on the market
     (long history) and on SPY (2007 onward); 1 bp per unit turnover.

Pre-registered gate (binding unless marked)
  G1 replication: in-sample 1973-01..2021-12 predictive regression of next-month Mkt-RF on
     siiInSample has a NEGATIVE slope with HAC |t| >= 2.5.
  G2 OOS rule check, 1990-01..2021-12: Campbell-Thompson OOS R^2 > 0 versus the prevailing-mean
     forecast, AND timing-rule net Sharpe (no-rf) >= buy-and-hold Sharpe over the same window.
  G3 holdout 2022-01..latest (informational; ~55 months): slope sign negative, and timing-rule
     net Sharpe not below buy-and-hold by more than 0.20.
  G4 book 2020-01..2025-12 monthly, inverse-vol weights: three-sleeve (equity, trend V1_trail,
     SI timing on SPY) no-rf Sharpe >= two-sleeve AND MaxDD not deeper. rf-Sharpe reported
     alongside but, after the RFS lesson, not used as the criterion.
  G5 informational: monthly correlations with the equity sleeve, trend, and SPY.
Trial registry: one trial (this rule); no variants are run.

Outputs: results/finra_short_interest_raw.csv.gz, results/sec_shares_outstanding.csv.gz,
results/short_interest_splice_monthly.csv, console report.
Usage:  python3 short_interest_splice.py --fetch-only     # FINRA per-symbol + SEC shares (cached)
        python3 short_interest_splice.py                  # analysis (fetches if caches missing)
"""

import datetime as dt
import gzip
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
RAW_CACHE = os.path.join(RESULTS_DIR, 'finra_short_interest_raw.csv.gz')
SEC_CACHE = os.path.join(RESULTS_DIR, 'sec_shares_outstanding.csv.gz')
AGG_CACHE = os.path.join(RESULTS_DIR, 'finra_short_interest_agg.csv')     # dates already discovered
FF_CSV = os.path.join(SCRIPT_DIR, 'backtest_data', 'F-F_Research_Data_Factors.csv')
TREND_CACHE = os.path.join(SCRIPT_DIR, 'trend_cache.csv')
TREND_V3 = os.path.join(RESULTS_DIR, 'trend_v3_returns.csv')
SII_XLSX = os.environ.get('RRZ_SII_XLSX', '')
RAW_XLSX = os.environ.get('RRZ_RAW_XLSX', '')

FINRA_API = 'https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest'
CLASSES = ['NYSE', 'NNM', 'SC', 'AMEX']
FIELDS = ['symbolCode', 'marketClassCode', 'currentShortPositionQuantity', 'averageDailyVolumeQuantity']
PAGE = 5000
# The SEC asks automated clients to identify themselves: set SEC_USER_AGENT to "your-name your-email".
SEC_UA = {'User-Agent': os.environ.get('SEC_USER_AGENT', 'research-script contact@example.com'), 'Accept-Encoding': 'gzip, deflate'}
SEC_TICKERS = 'https://www.sec.gov/files/company_tickers.json'
SEC_CONCEPT = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json'
SEC_SLEEP = 0.11
SEC_REFRESH_DAYS = 30

OVERLAP = ('2020-04', '2021-12')
OOS_START = '1990-01'
HOLDOUT_START = '2022-01'
EVAL = ('2020-01', '2025-12')
RF_FLAT_M = 0.05 / 12


# --------------------------------------------------------------------------- FINRA per-symbol
def _finra_post(body, retries=4):
    """Return (rows, record_total). rows is None on failure, [] on a genuine empty page."""
    for i in range(retries):
        try:
            r = requests.post(FINRA_API, json=body, headers={'Accept': 'application/json'}, timeout=90)
            if r.status_code == 200:
                return r.json(), int(r.headers.get('record-total', -1))
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (i + 1))
                continue
            return None, -1
        except requests.RequestException:
            time.sleep(3 * (i + 1))
    return None, -1


def fetch_finra_raw():
    """Per-symbol rows for every settlement date. A date is cached only when every class was
    fetched completely (row count equals the API's record-total); failed or partial dates are
    not cached and are retried on the next run (fixed 2026-09-05)."""
    dates = sorted(pd.read_csv(AGG_CACHE).settlementDate.astype(str))
    have = set()
    frames = []
    if os.path.exists(RAW_CACHE):
        old = pd.read_csv(RAW_CACHE)
        frames.append(old)
        have = set(old.settlementDate.astype(str))
    failed = []
    for d in dates:
        if d in have:
            continue
        rows, ok = [], True
        for cls in CLASSES:
            offset, expected, got = 0, None, 0
            while True:
                data, total = _finra_post({'limit': PAGE, 'offset': offset, 'fields': FIELDS,
                                           'compareFilters': [{'compareType': 'EQUAL', 'fieldName': 'settlementDate', 'fieldValue': d},
                                                              {'compareType': 'EQUAL', 'fieldName': 'marketClassCode', 'fieldValue': cls}]})
                if data is None:
                    ok = False
                    break
                if expected is None:
                    expected = total
                rows.extend(data)
                got += len(data)
                offset += PAGE
                if len(data) < PAGE:
                    break
            if not ok or (expected is not None and expected >= 0 and got != expected):
                ok = False
                break
        if not ok:
            failed.append(d)
            print(f"  FINRA {d}: FAILED or incomplete, not cached", flush=True)
            continue
        if rows:
            df = pd.DataFrame(rows)
            df['settlementDate'] = d
            frames.append(df)
            pd.concat(frames).to_csv(RAW_CACHE, index=False, compression='gzip')
            print(f"  FINRA {d}: {len(df)} rows", flush=True)
    if failed:
        print(f"FINRA: {len(failed)} dates not cached, retry next run: {failed}", flush=True)
    return pd.concat(frames) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- SEC shares outstanding
def _norm(sym):
    return str(sym).strip().upper().replace('.', '-')


def fetch_sec_shares(symbols):
    sess = requests.Session()
    sess.headers.update(SEC_UA)
    tick = sess.get(SEC_TICKERS, timeout=60).json()
    t2c = {}
    for v in tick.values():
        t2c[_norm(v['ticker'])] = int(v['cik_str'])
    today = pd.Timestamp.today().normalize()
    if os.path.exists(SEC_CACHE):
        done = pd.read_csv(SEC_CACHE)
        if 'fetched' not in done.columns:      # cache written before the refresh policy existed
            done['fetched'] = pd.Timestamp(os.path.getmtime(SEC_CACHE), unit='s').strftime('%Y-%m-%d')
    else:
        done = pd.DataFrame(columns=['symbol', 'fetched', 'form'])
    # retry transient failures (anything but a definitive 404) and refresh entries older than
    # SEC_REFRESH_DAYS so newer filings arrive (fixed 2026-09-05)
    form = done['form'].astype(str)
    transient = done[form.str.startswith(('HTTP', 'ERR')) & (form != 'HTTP404')].symbol.astype(str)
    stale = done[pd.to_datetime(done['fetched']) < today - pd.Timedelta(days=SEC_REFRESH_DAYS)].symbol.astype(str)
    redo = set(transient) | set(stale)
    keep = done[~done.symbol.astype(str).isin(redo)]
    have = set(keep.symbol.astype(str))
    out = [keep] if len(keep) else []
    todo = [s for s in sorted(set(map(_norm, symbols))) if s in t2c and s not in have]
    print(f"SEC: {len(todo)} symbols to fetch ({len(have)} cached, {len(redo)} retried/refreshed; "
          f"{sum(1 for s in set(map(_norm, symbols)) if s not in t2c)} unmapped)", flush=True)
    batch = []
    stamp = today.strftime('%Y-%m-%d')
    for k, s in enumerate(todo):
        cik = t2c[s]
        try:
            r = sess.get(SEC_CONCEPT.format(cik=cik), timeout=60)
            if r.status_code == 200:
                for u in r.json().get('units', {}).get('shares', []):
                    batch.append({'symbol': s, 'cik': cik, 'end': u.get('end'), 'filed': u.get('filed'),
                                  'val': u.get('val'), 'form': u.get('form'), 'fetched': stamp})
            else:
                batch.append({'symbol': s, 'cik': cik, 'end': None, 'filed': None, 'val': np.nan, 'form': f'HTTP{r.status_code}', 'fetched': stamp})
        except requests.RequestException:
            batch.append({'symbol': s, 'cik': cik, 'end': None, 'filed': None, 'val': np.nan, 'form': 'ERR', 'fetched': stamp})
        time.sleep(SEC_SLEEP)
        if (k + 1) % 200 == 0 or k + 1 == len(todo):
            out.append(pd.DataFrame(batch))
            batch = []
            pd.concat(out).to_csv(SEC_CACHE, index=False, compression='gzip')
            print(f"  SEC {k + 1}/{len(todo)}", flush=True)
    return pd.concat(out) if out else pd.DataFrame()


def shares_asof(sec):
    """Per symbol: filed-date -> total shares as of that filing's latest cover date.

    A filing can carry facts for more than one 'end' date (an amended prior quarter alongside the
    current cover page). Only facts with the filing's maximum 'end' are kept, then distinct values
    (share classes) are summed. Summing across 'end' dates double-counted 378 filings since 2020
    (fixed 2026-09-05)."""
    s = sec.dropna(subset=['val', 'filed', 'end']).copy()
    s = s[s.form.astype(str).str.startswith(('10-', '20-', '40-', '8-', '6-'))]
    s['end'] = pd.to_datetime(s['end'])
    s['filed'] = pd.to_datetime(s['filed'])
    s['max_end'] = s.groupby(['symbol', 'filed'])['end'].transform('max')
    s = s[s['end'] == s['max_end']].drop_duplicates(subset=['symbol', 'filed', 'end', 'val'])
    g = s.groupby(['symbol', 'filed'], as_index=False)['val'].sum()
    return g.sort_values(['symbol', 'filed'])


def build_proxy(raw, shares):
    raw = raw.copy()
    raw['symbol'] = raw.symbolCode.map(_norm)
    raw['short'] = pd.to_numeric(raw.currentShortPositionQuantity, errors='coerce')
    raw['date'] = pd.to_datetime(raw.settlementDate)
    raw = raw[raw.date.dt.day <= 20]                    # mid-month settlements only
    recs = []
    by_sym = {k: v for k, v in shares.groupby('symbol')}
    for d, grp in raw.groupby('date'):
        ratios = []
        for sym, sh in zip(grp.symbol, grp.short):
            f = by_sym.get(sym)
            if f is None or not np.isfinite(sh) or sh <= 0:
                continue
            f = f[(f.filed <= d) & (f.filed >= d - pd.Timedelta(days=400))]
            if f.empty:
                continue
            so = float(f.iloc[-1].val)
            if so < 1e6:
                continue
            ratio = sh / so
            if 0 < ratio <= 1.0:
                ratios.append(ratio)
        if len(ratios) >= 500:
            r = pd.Series(ratios)
            lo, hi = r.quantile(0.005), r.quantile(0.995)
            recs.append({'m': d.to_period('M'), 'n': len(r), 'ewsi_proxy': float(r.clip(lo, hi).mean())})
    return pd.DataFrame(recs).set_index('m').sort_index()


# --------------------------------------------------------------------------- authors' index
def load_authors():
    sii = pd.read_excel(SII_XLSX)
    sii['m'] = pd.to_datetime(sii['date']).dt.to_period('M')
    sii = sii.set_index('m')[['siiInSample', 'siiOutOfSample']]
    raw = pd.read_excel(RAW_XLSX, sheet_name='Short interest')
    raw['m'] = pd.PeriodIndex(pd.to_datetime(raw['Date'].astype(int).astype(str), format='%Y%m'), freq='M')
    raw = raw.set_index('m')[['EWSI']]
    return sii, raw


def reconstruct_log_ewsi(sii, raw):
    j = raw.join(sii, how='inner')
    j['t'] = np.arange(len(j))
    y = np.log(j.EWSI.values)
    X = np.column_stack([np.ones(len(j)), j.t, j.siiInSample])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - np.var(y - X @ beta) / np.var(y)
    t_full = np.arange(len(sii))
    log_ewsi = pd.Series(beta[0] + beta[1] * t_full + beta[2] * sii.siiInSample.values, index=sii.index, name='log_ewsi')
    return log_ewsi, r2


def recursive_sii(x, min_obs=60):
    x = x.dropna()
    t = np.arange(len(x))
    out = np.full(len(x), np.nan)
    for i in range(min_obs - 1, len(x)):
        b = np.polyfit(t[:i + 1], x.values[:i + 1], 1)
        r = x.values[:i + 1] - np.polyval(b, t[:i + 1])
        out[i] = r[-1] / r.std(ddof=1)
    return pd.Series(out, index=x.index, name='sii_oos')


# --------------------------------------------------------------------------- returns & stats
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


def hac_t(y, x, lags=3):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    n, k = X.shape
    S = np.zeros((k, k))
    for lag in range(lags + 1):
        w = 1 - lag / (lags + 1)
        for t in range(lag, n):
            g = np.outer(X[t] * e[t], X[t - lag] * e[t - lag])
            S += w * (g + g.T if lag else g)
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    return beta, beta[1] / np.sqrt(V[1, 1])


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


def walk_forward(sii_oos, excess, start, gamma=3.0, cap=1.5, min_train=60):
    """Expanding OLS of next-month excess return on SII through t; MV weight applied to month t+1."""
    j = pd.concat([sii_oos.rename('x'), excess.shift(-1).rename('y_next')], axis=1).dropna()
    fc, w, pm = [], [], []
    for i in range(len(j)):
        if i < min_train or j.index[i] < pd.Period(start, 'M'):
            fc.append(np.nan); w.append(np.nan); pm.append(np.nan); continue
        hx, hy = j.x.values[:i], j.y_next.values[:i]
        b = np.polyfit(hx, hy, 1)
        f = np.polyval(b, j.x.values[i])
        var = np.var(hy[-60:])
        fc.append(f); pm.append(hy.mean())
        w.append(float(np.clip(f / (gamma * var), 0, cap)))
    j['forecast'], j['prev_mean'], j['weight'] = fc, pm, w
    return j.dropna()


def oos_r2(j):
    e_model = (j.y_next - j.forecast) ** 2
    e_mean = (j.y_next - j.prev_mean) ** 2
    return float(1 - e_model.sum() / e_mean.sum())


def realise(j, ret, cost_bps=1.0):
    pos = pd.Series(j.weight.values, index=j.index + 1)      # weight decided at t, earns month t+1
    r = ret.reindex(pos.index)
    turnover = (pos - pos.shift(1).fillna(0)).abs()
    return (pos * r - turnover * cost_bps / 1e4).dropna(), pos


# --------------------------------------------------------------------------- main
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    need_fetch = not (os.path.exists(RAW_CACHE) and os.path.exists(SEC_CACHE))
    if '--fetch-only' in sys.argv or need_fetch:
        print("Stage 1: FINRA per-symbol short interest", flush=True)
        raw = fetch_finra_raw()
        syms = raw.symbolCode.map(_norm).unique()
        print(f"Stage 2: SEC shares outstanding for {len(syms)} symbols", flush=True)
        fetch_sec_shares(syms)
        if '--fetch-only' in sys.argv:
            print("fetch complete")
            sys.exit(0)

    if not SII_XLSX or not RAW_XLSX:
        sys.exit("Set RRZ_SII_XLSX and RRZ_RAW_XLSX to the downloaded authors' files (kept outside the repo).")
    raw = pd.read_csv(RAW_CACHE)
    sec = pd.read_csv(SEC_CACHE)
    sii, raw_ewsi = load_authors()
    log_ewsi, r2 = reconstruct_log_ewsi(sii, raw_ewsi)
    print(f"Affine reconstruction of log EWSI from siiInSample: R^2 = {r2:.6f} (must be ~1)")

    proxy = build_proxy(raw, shares_asof(sec))
    print(f"Proxy: {len(proxy)} mid-month observations {proxy.index.min()}..{proxy.index.max()}, "
          f"median names {int(proxy.n.median())}")
    ov = pd.concat([log_ewsi, np.log(proxy.ewsi_proxy).rename('log_proxy')], axis=1).loc[pd.Period(OVERLAP[0], 'M'):pd.Period(OVERLAP[1], 'M')].dropna()
    corr = float(ov.corr().iloc[0, 1])
    b = np.polyfit(ov.log_proxy, ov.log_ewsi, 1)
    print(f"Splice overlap {OVERLAP[0]}..{OVERLAP[1]}: n={len(ov)}, corr(log EWSI, log proxy) = {corr:+.2f}, "
          f"map: log EWSI = {b[1]:.3f} + {b[0]:.3f} * log proxy")
    splice_ok = corr >= 0.6
    ext = pd.Series(np.polyval(b, np.log(proxy.ewsi_proxy)), index=proxy.index)
    log_ewsi_full = pd.concat([log_ewsi, ext.loc[pd.Period(OVERLAP[1], 'M') + 1:]]).sort_index()
    sii_oos = recursive_sii(log_ewsi_full)
    chk = pd.concat([sii_oos, sii.siiOutOfSample], axis=1).dropna()
    print(f"Recursive SII replication vs authors' OOS column: corr {chk.corr().iloc[0, 1]:.4f}, "
          f"mean |diff| {(chk.iloc[:, 0] - chk.iloc[:, 1]).abs().mean():.3f} over {len(chk)} months")

    mktrf, rf = load_ff()
    spy = load_spy()
    spy_ex = spy - rf.reindex(spy.index)

    # G1 replication, in-sample 1973-2021
    g = pd.concat([sii.siiInSample.rename('x'), mktrf.shift(-1).rename('y')], axis=1).loc[:pd.Period('2021-12', 'M')].dropna()
    beta1, t1 = hac_t(g.y.values, g.x.values)
    print(f"\nG1 in-sample 1973-2021 (n={len(g)}): slope {beta1[1]*100:+.3f}%/month per s.d., HAC t {t1:+.2f}")

    # G2 OOS 1990-2021 with the authors' own OOS column (no splice involved)
    wf = walk_forward(sii.siiOutOfSample, mktrf, OOS_START)
    wf_pre = wf.loc[:pd.Period('2021-11', 'M')]
    strat_pre, _ = realise(wf_pre, mktrf + rf.reindex(mktrf.index))
    bh_pre = (mktrf + rf).reindex(strat_pre.index)
    r2_oos = oos_r2(wf_pre)
    print(f"G2 OOS {OOS_START}..2021-12 ({len(strat_pre)} months): OOS R^2 {r2_oos*100:+.2f}%, "
          f"timing net Sharpe {sr(strat_pre):.2f} vs buy-and-hold {sr(bh_pre):.2f}; avg weight {wf_pre.weight.mean():.2f}; "
          f"MDD {mdd(strat_pre):.1%} vs {mdd(bh_pre):.1%}")

    # G2b same OOS test on the internally consistent recursive construction (informational)
    wf_mine = walk_forward(sii_oos.loc[:pd.Period('2021-12', 'M')], mktrf, OOS_START)
    wf_mine_pre = wf_mine.loc[:pd.Period('2021-11', 'M')]
    strat_mine, _ = realise(wf_mine_pre, mktrf + rf.reindex(mktrf.index))
    print(f"G2b same test on own recursive index ({len(strat_mine)} months): OOS R^2 {oos_r2(wf_mine_pre)*100:+.2f}%, "
          f"timing net Sharpe {sr(strat_mine):.2f} vs buy-and-hold {sr((mktrf + rf).reindex(strat_mine.index)):.2f}; "
          f"avg weight {wf_mine_pre.weight.mean():.2f}")

    # G3 holdout 2022-01.. with the spliced series (own recursive construction throughout)
    wf_all = walk_forward(sii_oos, mktrf, OOS_START)
    hold = wf_all.loc[pd.Period(HOLDOUT_START, 'M') - 1:]
    strat_h, pos_h = realise(hold, spy)
    bh_h = spy.reindex(strat_h.index)
    # slope on the same instrument and months as the reported performance: SPY excess return,
    # realised months only (fixed 2026-09-05; previously French market through a later month)
    hj = pd.concat([hold.x, spy_ex.shift(-1).reindex(hold.index)], axis=1).dropna()
    hj.columns = ['x', 'y']
    hb, ht = hac_t(hj.y.values, hj.x.values) if len(hj) >= 12 else ((np.nan, np.nan), np.nan)
    print(f"G3 holdout {HOLDOUT_START}..{strat_h.index.max()} on SPY ({len(strat_h)} months, splice {'OK' if splice_ok else 'UNRELIABLE'}): "
          f"slope on SPY excess ({len(hj)} obs) {hb[1]*100:+.3f}% t {ht:+.2f}; timing net Sharpe {sr(strat_h):.2f} vs SPY {sr(bh_h):.2f}; "
          f"avg weight {pos_h.mean():.2f}; MDD {mdd(strat_h):.1%} vs {mdd(bh_h):.1%}")

    # G4/G5 book 2020-2025 on SPY-realised timing
    wf_eval = wf_all.loc[pd.Period(EVAL[0], 'M') - 1:pd.Period(EVAL[1], 'M')]
    strat_e, pos_e = realise(wf_eval, spy)
    parts = [pd.read_csv(os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'), index_col=0, parse_dates=True).iloc[:, 0]
             for y in range(2020, 2026) if os.path.exists(os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'))]
    if parts and os.path.exists(TREND_V3):
        eq = monthly_from_daily(pd.concat(parts).sort_index()).rename('equity')
        tr = monthly_from_daily(pd.read_csv(TREND_V3, index_col=0, parse_dates=True)['V1_trail'].dropna()).rename('trend')
        jb = pd.concat([eq, tr, strat_e.rename('si_timing'), spy.rename('spy')], axis=1).loc[pd.Period(EVAL[0], 'M'):pd.Period(EVAL[1], 'M')].dropna()
        print(f"\nG5 correlations 2020-2025 ({len(jb)} months):\n{jb.corr().round(2).to_string()}")

        def book(cols):
            inv = 1 / jb[cols].std(); w = inv / inv.sum(); b = (jb[cols] * w).sum(axis=1)
            yr = {y: sr(x, RF_FLAT_M) for y, x in b.groupby(b.index.year)}
            return {'w': w.round(2).to_dict(), 'norf': sr(b), 'rf': sr(b, RF_FLAT_M), 'mdd': mdd(b), 'vol': b.std() * np.sqrt(12), 'worst': min(yr.values())}
        b2, b3 = book(['equity', 'trend']), book(['equity', 'trend', 'si_timing'])
        for n_, bb in (('equity+trend', b2), ('equity+trend+SI timing', b3)):
            print(f"  {n_:24s} no-rf SR {bb['norf']:.2f} | rf-SR {bb['rf']:.2f} | MDD {bb['mdd']:.1%} | vol {bb['vol']:.1%} | worst yr {bb['worst']:+.2f} | w {bb['w']}")
        g4 = (b3['norf'] >= b2['norf']) and (b3['mdd'] >= b2['mdd'])
    else:
        g4 = False
        print("Book test skipped: equity or trend series missing")

    g1 = (beta1[1] < 0) and (abs(t1) >= 2.5)
    g2 = (r2_oos > 0) and (sr(strat_pre) >= sr(bh_pre))
    g3 = (hb[1] < 0) and (sr(strat_h) >= sr(bh_h) - 0.20) if np.isfinite(ht) else False
    print("\nGO/NO-GO (pre-registered):")
    print(f"  {'PASS' if g1 else 'fail'}  G1 replication: negative slope, |t| >= 2.5")
    print(f"  {'PASS' if g2 else 'fail'}  G2 OOS 1990-2021: R^2 > 0 and timing Sharpe >= buy-and-hold")
    print(f"  {'PASS' if g3 else 'fail'}  G3 holdout 2022+: negative slope, Sharpe within 0.20 of SPY (informational{'' if splice_ok else '; splice unreliable'})")
    print(f"  {'PASS' if g4 else 'fail'}  G4 book 2020-2025: no-rf Sharpe not lower, MDD not deeper")
    verdict = g1 and g2 and g4
    print(f"VERDICT: {'PROMOTE to paper-trade candidate' if verdict else 'not promoted'} (G1, G2, G4 binding; G3 informational)")

    out = pd.concat([log_ewsi_full.rename('log_ewsi_spliced'), sii_oos, sii, proxy.ewsi_proxy, mktrf.rename('mktrf'), spy.rename('spy')], axis=1)
    out.to_csv(os.path.join(RESULTS_DIR, 'short_interest_splice_monthly.csv'))
    print("Saved results/short_interest_splice_monthly.csv")
