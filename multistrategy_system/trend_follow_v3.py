"""
Trend sleeve v3 — RESEARCH ONLY. Does not import or modify trend_sleeve_live.py.

Purpose (pre-registered 2026-09-04): test two published refinements of the validated
cross-asset trend sleeve on the LIVE 16-ETF universe (SECTOR minus SPY/QQQ/IWM), while
removing the known full-sample volatility-scalar leak in trend_follow_v2.

Variants (all use the shared cost model at the trend_follow_v2 defaults: 2 bp fallback
execution, no commission/borrow/financing):

  REF_v2_18      trend_follow_v2 as cached for the headline: 18 assets, FULL-SAMPLE scalar (leaks)
  REF_v2_16      trend_follow_v2 signals on the live 16, FULL-SAMPLE scalar
  V1_trail       v2 signals (mean sign over 63/126/252 days) + TRAILING scalar      <- BASELINE
  V2a_MED        Goulding, Harvey & Mazzoleni (JFE 2023) MED: w = ½(SLOW + FAST) per asset,
                 SLOW = +1 if mean of trailing 12 monthly returns >= 0 else -1,
                 FAST = +1 if last monthly return >= 0 else -1 (paper, Table 2 notes).
                 Turning-point assets (Correction / Rebound) get zero exposure.
  V2b_DYN        GHM DYN: w = (1 - a_s) SLOW + a_s FAST with a_Bu = a_Be = ½ and a_Co, a_Re from
                 the paper's eqs (32)-(33), estimated on POOLED, vol-scaled prior data across the
                 universe (expanding window, no look-ahead, clipped to [0, 1], MED until enough data).
  V3a_63_126     mean-sign ensemble over {63, 126}         (Quantica 2025: 60-120 business-day band)
  V3b_126_252    mean-sign ensemble over {126, 252}        (drop the fastest leg)
  V3c_84_126_168 mean-sign ensemble over {84, 126, 168}    (band-centred)
  V4_DYN_ENS     DYN state weighting with the 3/6/12 ensemble as the slow leg:
                 w = (1 - a_s) ENS + a_s FAST, state from (sign(ENS), FAST). Pre-registered heuristic.

Trailing scalar: at each month-end, realized vol of the UNSCALED portfolio (per-asset vol
targeting, 1/n) over the last SCALAR_LOOKBACK trading days (min SCALAR_MIN_HISTORY);
scalar = min(PORT_TARGET_VOL / realized, MAX_LEVERAGE); 1.0 until enough history. Applied to
the next month's positions, so it never sees the returns it scales.

Normalisation: 1/n with n = number of assets that have a valid size (enough price history and
a finite positive vol). For v2 signals this equals v2's count of nonzero weights, so V1 differs
from REF_v2_16 only in the scalar path (checked at run time). For MED/DYN it keeps
turning-point assets at zero instead of re-concentrating the book into the rest.

Go/no-go (pre-registered): promote a variant over V1 only if it beats V1 on full-sample
rf-Sharpe AND worst-calendar-year rf-Sharpe, keeps 2022 rf-Sharpe > 0, and its 2020-2025
SPY correlation is not above V1's by more than 0.02. Otherwise V1 (the leak fix) ships alone.

Sharpe conventions (labelled everywhere): rf subtracts 5%/yr daily, no-rf is mean/std; both
use pandas' default ddof=1, matching combine_book_v2.py / extend_history.py.

Outputs: results/trend_v3_returns.csv (daily net returns per variant),
results/trend_v3_summary.csv, results/trend_v3_dyn_speeds.csv, console report.
Trial registry: 7 new trials (V1, V2a, V2b, V3a, V3b, V3c, V4); the two REF rows are not trials.

Usage:  python3 trend_follow_v3.py
"""

import os
import numpy as np
import pandas as pd

import trend_follow_v2 as t2
from trading_costs import CostModelConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
RF_DAILY = 0.05 / 252
ANN = np.sqrt(252)

LIVE16 = [c for c in t2.SECTOR if c not in ('SPY', 'QQQ', 'IWM')]
SCALAR_LOOKBACK = 504          # ~2 years of daily portfolio returns
SCALAR_MIN_HISTORY = 126       # scalar stays 1.0 before this
DYN_MIN_OBS = 60               # pooled Co and Re observations needed before DYN speeds are used
WARMUP_DAYS = max(t2.HORIZONS) + 5   # 257, identical to trend_follow_v2
FULL_START = '2008-01-01'      # first year in which all 16 ETFs have signals; includes the 2008 crisis
EVAL_START, EVAL_END = '2020-01-01', '2025-12-31'
CRISIS = {2008, 2020, 2022}

VARIANTS = [
    # name,            mode,      horizons
    ('V1_trail',       'V2',      t2.HORIZONS),
    ('V2a_MED',        'MED',     t2.HORIZONS),
    ('V2b_DYN',        'DYN',     t2.HORIZONS),
    ('V3a_63_126',     'V2',      [63, 126]),
    ('V3b_126_252',    'V2',      [126, 252]),
    ('V3c_84_126_168', 'V2',      [84, 126, 168]),
    ('V4_DYN_ENS',     'DYN_ENS', t2.HORIZONS),
]


# --------------------------------------------------------------------------- signals
def _size(rets, c, d):
    vol = rets[c].loc[:d].tail(t2.VOL_LOOKBACK).std() * ANN
    if not np.isfinite(vol) or vol <= 0:
        return 0.0
    return min(t2.TARGET_VOL_PER_ASSET / vol, t2.MAX_LEVERAGE)


def _ens(p, horizons, legacy=False):
    """Mean sign of trailing h-day returns. legacy=True reproduces trend_follow_v2 (and the live
    sleeve), whose p.iloc[-h] spans h-1 returns; the corrected form uses p.iloc[-h-1], h returns."""
    off = 0 if legacy else 1
    return float(np.mean([1.0 if p.iloc[-1] / p.iloc[-h - off] - 1.0 > 0 else -1.0 for h in horizons]))


def month_end_index(px, legacy=False):
    """Rebalance stamps. legacy=True uses calendar month-ends as trend_follow_v2 does, which
    activates weights one trading day late after weekend month-ends; the corrected form stamps the
    last trading day of each month so weights apply from the next trading day."""
    if legacy:
        return px.resample('ME').last().index
    return pd.DatetimeIndex(px.groupby(px.index.to_period('M')).apply(lambda g: g.index[-1]).values)


def _slow_fast(rm_c, d):
    """GHM definitions: SLOW from the mean of the trailing 12 monthly returns, FAST from the last."""
    h = rm_c.loc[:pd.Period(d, 'M')].dropna()
    if len(h) < 12:
        return None
    slow = 1.0 if h.tail(12).mean() >= 0 else -1.0
    fast = 1.0 if h.iloc[-1] >= 0 else -1.0
    return slow, fast


def _state(slow, fast):
    if slow > 0 and fast > 0:
        return 'Bu'
    if slow < 0 and fast < 0:
        return 'Be'
    return 'Co' if slow > 0 else 'Re'


def _dyn_speeds(realized):
    """Eqs (32)-(33) of GHM on realized (state, next-month vol-scaled return) pairs.

    K = E[r^2|Bu or Be] P[Bu or Be] / (E[r|Bu] P[Bu] - E[r|Be] P[Be]); the sample sizes cancel, so
    K = sum_{Bu,Be} r^2 / (sum_Bu r - sum_Be r).
    a_Co = ½ (1 - K E[r|Co] / E[r^2|Co]);  a_Re = ½ (1 + K E[r|Re] / E[r^2|Re]); clipped to [0, 1].
    """
    if not realized:
        return 0.5, 0.5, 'MED(no history)'
    df = pd.DataFrame(realized)
    co, re_ = df.y[df.state == 'Co'], df.y[df.state == 'Re']
    if len(co) < DYN_MIN_OBS or len(re_) < DYN_MIN_OBS:
        return 0.5, 0.5, 'MED(insufficient)'
    bu, be = df.y[df.state == 'Bu'], df.y[df.state == 'Be']
    denom = bu.sum() - be.sum()
    if denom <= 0:
        return 0.5, 0.5, 'MED(denom<=0)'
    k = (bu.pow(2).sum() + be.pow(2).sum()) / denom
    a_co = 0.5 * (1.0 - k * co.mean() / co.pow(2).mean())
    a_re = 0.5 * (1.0 + k * re_.mean() / re_.pow(2).mean())
    return float(np.clip(a_co, 0, 1)), float(np.clip(a_re, 0, 1)), 'DYN'


def build_weights(prices, universe, mode, horizons, legacy=False):
    """Unscaled daily target weights (per-asset vol targeting, 1/n) and diagnostics."""
    cols = [c for c in universe if c in prices.columns]
    px = prices[cols]
    rets = px.pct_change(fill_method=None)
    pm = px.resample('ME').last()
    rm = pm.pct_change(fill_method=None)
    rm.index = rm.index.to_period('M')
    month_ends = month_end_index(px, legacy)
    reb = pd.DataFrame(index=month_ends, columns=cols, dtype=float)
    nvalid = pd.Series(np.nan, index=month_ends)
    dyn_log, realized, pending = [], [], []

    for d in month_ends:
        # realize last month's (state, next-month return) pairs before anything else is decided
        for rec in pending:
            r_next = rm.at[pd.Period(d, 'M'), rec['asset']]
            if np.isfinite(r_next):
                realized.append({'state': rec['state'], 'y': r_next * rec['size']})
        pending = []

        hist = px.loc[:d]
        if len(hist) < WARMUP_DAYS:
            continue
        a_co = a_re = 0.5
        status = 'MED'
        if mode in ('DYN', 'DYN_ENS'):
            a_co, a_re, status = _dyn_speeds(realized)
            dyn_log.append({'date': d, 'a_Co': a_co, 'a_Re': a_re, 'status': status,
                            'pooled_obs': len(realized)})
        w, n = {}, 0
        for c in cols:
            p = hist[c].dropna()
            if len(p) < WARMUP_DAYS:
                w[c] = 0.0
                continue
            size = _size(rets, c, d)
            if size <= 0:
                w[c] = 0.0
                continue
            if mode == 'V2':
                sig = _ens(p, horizons, legacy)
            else:
                sf = _slow_fast(rm[c], d)
                if sf is None:
                    w[c] = 0.0
                    continue
                slow, fast = sf
                if mode == 'DYN_ENS':
                    slow_leg = _ens(p, horizons, legacy)
                    slow_dir = 1.0 if slow_leg > 0 else -1.0
                else:
                    slow_leg, slow_dir = slow, slow
                st = _state(slow_dir, fast)
                a = {'Bu': 0.5, 'Be': 0.5, 'Co': a_co, 'Re': a_re}[st]
                sig = (1.0 - a) * slow_leg + a * fast
                pending.append({'asset': c, 'state': st, 'size': size})
            n += 1
            w[c] = sig * size
        if n == 0:
            continue
        reb.loc[d] = pd.Series(w)
        nvalid.loc[d] = n

    weights = reb.reindex(px.index, method='ffill').shift(1).fillna(0.0)
    n_daily = nvalid.reindex(px.index, method='ffill').shift(1).fillna(1.0).replace(0, 1.0)
    normalized = weights.div(n_daily, axis=0)
    return normalized, px, rets, pd.DataFrame(dyn_log), month_ends


# --------------------------------------------------------------------------- scaling + costs
def scale_and_cost(normalized, px, rets, scalar_mode='trailing', execution_bps=t2.COST_BPS_PER_REBAL,
                   month_ends=None):
    gross = (normalized * rets).sum(axis=1)
    turnover = normalized.diff().abs().sum(axis=1)
    legacy = gross - turnover * (execution_bps / 10_000.0)
    active = normalized.abs().sum(axis=1).gt(0)
    first_active = active.idxmax() if active.any() else px.index[0]

    if scalar_mode == 'full':
        realized = legacy.std() * ANN
        s = min((t2.PORT_TARGET_VOL / realized) if realized > 0 else 1.0, t2.MAX_LEVERAGE)
        scal = pd.Series(s, index=px.index)
    else:
        vals = {}
        stamps = month_ends if month_ends is not None else px.resample('ME').last().index
        for d in stamps:
            h = legacy.loc[first_active:d].tail(SCALAR_LOOKBACK)
            if len(h) >= SCALAR_MIN_HISTORY and h.std() > 0:
                vals[d] = min(t2.PORT_TARGET_VOL / (h.std() * ANN), t2.MAX_LEVERAGE)
            else:
                vals[d] = 1.0
        scal = pd.Series(vals).reindex(px.index, method='ffill').shift(1).fillna(1.0)

    target = normalized.mul(scal, axis=0)
    gross_scaled = (target * rets).sum(axis=1)
    config = CostModelConfig(
        commission_per_share=t2.COMMISSION_PER_SHARE,
        min_commission=t2.MIN_COMMISSION,
        fallback_execution_bps=execution_bps,
        extra_slippage_bps=0.0,
        annual_borrow_rate=t2.ANNUAL_BORROW_RATE,
        annual_financing_rate=t2.ANNUAL_FINANCING_RATE,
        day_count=360.0,
    )
    port = t2._apply_portfolio_costs(gross_scaled, target, px, config, t2.BACKTEST_CAPITAL)
    cost_components = port.attrs['cost_components']
    port = port.loc[legacy.notna()].dropna()
    port.attrs['cost_components'] = cost_components
    port.attrs['scalar_mean'] = float(scal.loc[first_active:].mean())
    port.attrs['scalar_last'] = float(scal.iloc[-1])
    port.attrs['turnover_per_year'] = float(target.diff().abs().sum(axis=1).loc[first_active:].mean() * 252)
    port.attrs['first_active'] = first_active
    return port


def run_variant(prices, universe, mode, horizons, scalar_mode='trailing', legacy=False):
    normalized, px, rets, dyn_log, month_ends = build_weights(prices, universe, mode, horizons, legacy=legacy)
    port = scale_and_cost(normalized, px, rets, scalar_mode=scalar_mode, month_ends=month_ends)
    port.attrs['dyn_log'] = dyn_log
    return port


# --------------------------------------------------------------------------- statistics
def sharpe(r, rf=RF_DAILY):
    r = r.dropna()
    return float(((r - rf).mean() / r.std()) * ANN) if len(r) > 1 and r.std() > 0 else 0.0


def maxdd(r):
    """Max drawdown with the high-water mark starting at initial capital (1.0), so a loss in the
    first period counts. The legacy helper in combine_book*/trend_follow_v2 starts the mark after
    the first return and understates drawdowns."""
    r = r.dropna()
    eq = (1 + r).cumprod().values
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((eq / peak - 1).min()) if len(eq) else 0.0


def yearly(r):
    return {int(y): sharpe(x) for y, x in r.groupby(r.index.year)}


def window(r, start, end=None):
    return r.loc[start:end] if end else r.loc[start:]


def corr_with(r, spy):
    j = pd.concat([r.rename('r'), spy.rename('s')], axis=1).dropna()
    return float(j['r'].corr(j['s'])) if len(j) > 30 else np.nan


def rp_book(eq, trend):
    """Risk-parity combination on the overlap. Weights are full-overlap inverse vol, the
    headline convention (extend_history.py); walk-forward weights are a separate task."""
    j = pd.concat([eq.rename('e'), trend.rename('t')], axis=1).dropna()
    if len(j) < 60:
        return None
    ve, vt = j['e'].std(), j['t'].std()
    we = (1 / ve) / ((1 / ve) + (1 / vt))
    book = we * j['e'] + (1 - we) * j['t']
    return {'we': float(we), 'rf_sharpe': sharpe(book), 'worst_year': min(yearly(book).values()),
            'maxdd': maxdd(book), 'vol': float(book.std() * ANN), 'days': len(j)}


def load_equity():
    parts = []
    for y in range(2020, 2026):
        f = os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv')
        if os.path.exists(f):
            parts.append(pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0])
    return pd.concat(parts).sort_index() if parts else None


# --------------------------------------------------------------------------- report
def summarize(name, port, spy, eq):
    full = window(port, FULL_START)
    ev = window(port, EVAL_START, EVAL_END)
    yr = yearly(full)
    cc = port.attrs['cost_components']
    exec_cost = sum(cc[k] for k in ('commission', 'spread', 'slippage', 'impact'))
    exec_bps = exec_cost / cc['notional'] * 1e4 if cc['notional'] > 0 else 0.0
    book = rp_book(eq, ev) if eq is not None else None
    row = {
        'variant': name,
        'full_rf_sharpe': sharpe(full), 'full_norf_sharpe': sharpe(full, 0.0),
        'full_vol': float(full.std() * ANN), 'full_maxdd': maxdd(full),
        'full_worst_year': min(yr.values()), 'full_spy_corr': corr_with(full, spy),
        'eval_rf_sharpe': sharpe(ev), 'eval_norf_sharpe': sharpe(ev, 0.0),
        'eval_vol': float(ev.std() * ANN), 'eval_maxdd': maxdd(ev),
        'eval_spy_corr': corr_with(ev, spy),
        'y2022_rf_sharpe': yr.get(2022, np.nan),
        'scalar_mean': port.attrs['scalar_mean'], 'scalar_last': port.attrs['scalar_last'],
        'turnover_per_year': port.attrs['turnover_per_year'], 'exec_cost_bps': exec_bps,
        'book_rf_sharpe': book['rf_sharpe'] if book else np.nan,
        'book_worst_year': book['worst_year'] if book else np.nan,
        'book_maxdd': book['maxdd'] if book else np.nan,
        'book_vol': book['vol'] if book else np.nan,
        'book_equity_weight': book['we'] if book else np.nan,
    }
    for y, s in yr.items():
        row[f'y{y}'] = s
    return row


def print_report(rows, years):
    print("\n" + "=" * 118)
    print(f"TREND v3 — live 16-ETF universe, shared cost model 2 bp. Full window from {FULL_START}; "
          f"eval {EVAL_START[:4]}-{EVAL_END[:4]}. rf-Sharpe unless marked.")
    print("=" * 118)
    hdr = f"{'variant':16s}{'full rf':>8s}{'no-rf':>7s}{'vol':>7s}{'MDD':>8s}{'worst':>7s}{'SPYc':>7s}" \
          f"{'| eval rf':>10s}{'MDD':>8s}{'SPYc':>7s}{'| scal':>8s}{'TO/yr':>7s}{'bps':>6s}" \
          f"{'| book rf':>10s}{'worst':>7s}{'MDD':>8s}"
    print(hdr)
    for r in rows:
        print(f"{r['variant']:16s}{r['full_rf_sharpe']:8.2f}{r['full_norf_sharpe']:7.2f}"
              f"{r['full_vol']:7.1%}{r['full_maxdd']:8.1%}{r['full_worst_year']:7.2f}{r['full_spy_corr']:7.2f}"
              f"{r['eval_rf_sharpe']:10.2f}{r['eval_maxdd']:8.1%}{r['eval_spy_corr']:7.2f}"
              f"{r['scalar_mean']:8.2f}{r['turnover_per_year']:7.1f}{r['exec_cost_bps']:6.2f}"
              f"{r['book_rf_sharpe']:10.2f}{r['book_worst_year']:7.2f}{r['book_maxdd']:8.1%}")
    print("\nPer-year rf-Sharpe (* = crisis year):")
    print(f"{'variant':16s}" + "".join(f"{str(y) + ('*' if y in CRISIS else ''):>8s}" for y in years))
    for r in rows:
        print(f"{r['variant']:16s}" + "".join(f"{r.get(f'y{y}', np.nan):8.2f}" for y in years))


def gate(rows):
    base = next(r for r in rows if r['variant'] == 'V1_trail')
    print("\nGO/NO-GO versus V1_trail (pre-registered):")
    print(f"{'variant':16s}{'full rf >':>10s}{'worst-yr >':>11s}{'2022 > 0':>10s}{'SPYc ok':>9s}{'verdict':>10s}")
    verdicts = {}
    for r in rows:
        if r['variant'] in ('V1_trail', 'REF_v2_18', 'REF_v2_16'):
            continue
        c1 = r['full_rf_sharpe'] > base['full_rf_sharpe']
        c2 = r['full_worst_year'] > base['full_worst_year']
        c3 = r['y2022_rf_sharpe'] > 0
        c4 = r['eval_spy_corr'] <= base['eval_spy_corr'] + 0.02
        ok = c1 and c2 and c3 and c4
        verdicts[r['variant']] = ok
        print(f"{r['variant']:16s}{'PASS' if c1 else 'fail':>10s}{'PASS' if c2 else 'fail':>11s}"
              f"{'PASS' if c3 else 'fail':>10s}{'PASS' if c4 else 'fail':>9s}{'PROMOTE' if ok else 'no':>10s}")
    return verdicts


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prices = t2.load_prices()
    present = [c for c in t2.SECTOR if c in prices.columns]
    live16 = [c for c in LIVE16 if c in prices.columns]
    spy = prices['SPY'].pct_change(fill_method=None)
    eq = load_equity()
    print(f"Loaded {len(prices)} days {prices.index.min().date()}..{prices.index.max().date()}; "
          f"live universe {len(live16)}: {', '.join(live16)}")
    if eq is None:
        print("No cached equity_returns_2020..2025 found; book columns will be NaN.")

    series, rows = {}, []

    ref18 = t2.backtest(prices, present, t2.HORIZONS)
    ref16 = t2.backtest(prices, live16, t2.HORIZONS)
    for name, s in (('REF_v2_18', ref18), ('REF_v2_16', ref16)):
        s.attrs.setdefault('scalar_mean', s.attrs['leverage_scalar'])
        s.attrs.setdefault('scalar_last', s.attrs['leverage_scalar'])
        s.attrs.setdefault('turnover_per_year', np.nan)
        series[name] = s
        rows.append(summarize(name, s, spy, eq))

    # reimplementation check: v2 signals + full-sample scalar must reproduce trend_follow_v2 exactly
    chk = run_variant(prices, live16, 'V2', t2.HORIZONS, scalar_mode='full', legacy=True)
    j = pd.concat([chk.rename('a'), ref16.rename('b')], axis=1).dropna()
    print(f"Reimplementation check (legacy timing + lookback) vs trend_follow_v2 on live 16: max |diff| = "
          f"{(j['a'] - j['b']).abs().max():.2e} over {len(j)} days (expect ~0)")
    print("Corrections 2026-09-05 applied to all V* variants: rebalance stamped on the last trading day "
          "(weights active the next trading day, not one day later after weekend month-ends); h-day "
          "lookbacks use h returns (v2/live use h-1); max drawdown counts a first-period loss. "
          "REF rows keep the v2 conventions.")

    for name, mode, horizons in VARIANTS:
        port = run_variant(prices, live16, mode, horizons, scalar_mode='trailing')
        series[name] = port
        rows.append(summarize(name, port, spy, eq))
        if mode in ('DYN', 'DYN_ENS') and len(port.attrs['dyn_log']):
            log = port.attrs['dyn_log']
            log.to_csv(os.path.join(RESULTS_DIR, f'trend_v3_dyn_speeds_{name}.csv'), index=False)
            used = log[log.status == 'DYN']
            if len(used):
                print(f"{name}: DYN speeds active from {used.date.min().date()} "
                      f"(a_Co mean {used.a_Co.mean():.2f} last {used.a_Co.iloc[-1]:.2f}; "
                      f"a_Re mean {used.a_Re.mean():.2f} last {used.a_Re.iloc[-1]:.2f})")
            else:
                print(f"{name}: DYN speeds never activated (fell back to MED throughout)")

    years = sorted({y for r in rows for k in r if k.startswith('y') and k[1:].isdigit() for y in [int(k[1:])]})
    print_report(rows, years)
    verdicts = gate(rows)

    wide = pd.concat({k: v for k, v in series.items()}, axis=1)
    wide.to_csv(os.path.join(RESULTS_DIR, 'trend_v3_returns.csv'))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'trend_v3_summary.csv'), index=False)
    print(f"\nSaved results/trend_v3_returns.csv, results/trend_v3_summary.csv. "
          f"Trials logged for the DSR registry: {len(VARIANTS)}.")
    promoted = [k for k, v in verdicts.items() if v]
    print("Promoted over V1_trail: " + (", ".join(promoted) if promoted else "none — ship V1_trail (leak fix) alone."))
