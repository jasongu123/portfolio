"""
Forward paper test of the ETF sleeve on the source paper's rules (T-2026-10-01-72). Read-only on
trading_system.db.

    python etf_forward_report.py --start YYYY-MM-DD --account-value 1000000 [--etf-share 0.15]
    python etf_forward_report.py --make-thresholds          (research machine; needs split_trial E0)

--start is the first session ibkr2.py ran with ETF_PAPER_RULES on (the install date of
vm_sync_2026-10-01.zip, or the next session if installed after the open). --etf-share is the
three ETFs' combined allocation (the sum of the SPY, QQQ and IWM "Allocation:" lines that
ibkr2.py prints when it sizes a trade); drawdown is measured on account x share.

Method, fixed when the test was registered (2026-10-01, before any forward data existed):
  - fills: the execution journal's matched fills for SPY, QQQ, IWM (client 1; client, order and
    symbol must match, as in breadth_forward_report.py)
  - P&L: realized, average-cost, at the actual paper fill prices, less $0.0035 a share; booked on
    the session of the closing fill; every NYSE session from --start counts
  - G1 net no-rf Sharpe (daily x sqrt 252) > 0 after 126 sessions
  - G2 median signed slippage of the ETF fills against the decision-time midpoint <= 1.0 bp
    (the backtest edge is ~3.4 bp gross per position and is gone at about 2 bp a side)
  - G3 early stop, checked at every run: drawdown as a share of the sleeve's capital must not
    exceed the 5th-percentile drawdown of the corrected backtest E0 (results/split_trial_v2; no HAR, as deployed) over the same number of sessions
    (results/etf_forward_thresholds.csv). If it does: set ETF_INTRADAY_ENABLED=false and restart
    ibkr2.py (not ETF_PAPER_RULES=false, which restores the losing version), and record the stop
    in TRIAL_REGISTRY.md.
"""
import argparse
import datetime as dt
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

from breadth_forward_report import QUERY, sessions, threshold_for

HERE = os.path.dirname(os.path.abspath(__file__))
ETFS = ('SPY', 'QQQ', 'IWM')
HORIZON = 126
COMMISSION_PER_SHARE = 0.0035
MAX_MEDIAN_SLIPPAGE_BPS = 1.0
THRESHOLDS = os.path.join(HERE, 'results', 'etf_forward_thresholds.csv')
SLIPPAGE_QUERY = """
SELECT UPPER(TRIM(symbol)) AS symbol, signed_slippage_bps
FROM execution_costs
WHERE client_id = ? AND signed_slippage_bps IS NOT NULL
  AND COALESCE(fill_timestamp, created_at) >= ?
"""


def load(db, start, client_id=1):
    conn = sqlite3.connect(db)
    try:
        fills = pd.read_sql_query(QUERY, conn, params=(client_id,))
        slip = pd.read_sql_query(SLIPPAGE_QUERY, conn, params=(client_id, start.isoformat()))
    finally:
        conn.close()
    fills = fills[fills.symbol.isin(ETFS)].copy()
    fills['ts'] = pd.to_datetime(fills['ts'], utc=True, errors='coerce', format='mixed')
    fills = fills.dropna(subset=['ts'])
    fills['session'] = fills['ts'].dt.tz_convert('America/New_York').dt.date
    return fills[fills.session >= start], slip[slip.symbol.isin(ETFS)]


def realized(fills):
    """Average-cost realized P&L at fill prices less commission, per session; open positions."""
    pos, avg, rows = {}, {}, []
    for f in fills.itertuples():
        q = float(f.shares) * (1 if f.side == 'BUY' else -1)
        p, s = float(f.price), f.symbol
        cur, a = pos.get(s, 0.0), avg.get(s, 0.0)
        pnl = 0.0
        if cur == 0 or np.sign(cur) == np.sign(q):
            new = cur + q
            avg[s] = (a * abs(cur) + p * abs(q)) / abs(new)
            pos[s] = new
        else:
            closed = min(abs(q), abs(cur))
            pnl = closed * (p - a) * np.sign(cur)
            new = cur + q
            pos[s] = new
            if new == 0:
                avg[s] = 0.0
            elif np.sign(new) != np.sign(cur):
                avg[s] = p
        rows.append({'session': f.session, 'pnl': pnl - COMMISSION_PER_SHARE * abs(q)})
    return pd.DataFrame(rows, columns=['session', 'pnl']), {s: q for s, q in pos.items() if q != 0}


def report(db, start, account_value, etf_share=0.15, thresholds=None, today=None):
    fills, slip = load(db, start)
    pnl, open_pos = realized(fills)
    end = today or max([start] + list(fills.session))
    days = sessions(start, end)
    daily = pnl.groupby('session').pnl.sum().reindex(days).fillna(0.0)
    n = len(days)
    cap = account_value * etf_share
    eq = cap + daily.cumsum()
    dd = float((eq / np.maximum.accumulate(np.concatenate([[cap], eq.values]))[1:] - 1).min()) if n else 0.0
    sr = float(daily.mean() / daily.std() * np.sqrt(252)) if n > 1 and daily.std() > 0 else float('nan')
    med = float(slip.signed_slippage_bps.median()) if len(slip) else float('nan')
    thr = threshold_for(n, thresholds) if thresholds is not None and n else None
    res = {'sessions': n, 'sharpe': sr, 'pnl': float(daily.sum()), 'dd': dd, 'dd_threshold': thr,
           'median_slippage_bps': med, 'slippage_fills': int(len(slip)), 'fills': int(len(fills)),
           'open_positions': open_pos}
    if thr is not None and dd < thr:
        res['status'] = 'STOP: drawdown beyond the backtest 5th percentile - set ETF_INTRADAY_ENABLED=false'
    elif n < HORIZON:
        res['status'] = f'IN PROGRESS ({n}/{HORIZON} sessions)'
    else:
        g1 = sr > 0
        g2 = not np.isnan(med) and med <= MAX_MEDIAN_SLIPPAGE_BPS
        res['status'] = 'PASS' if (g1 and g2) else f"FAIL (G1 {'ok' if g1 else 'fail'}, G2 {'ok' if g2 else 'fail'})"
    return res


def window_mdd(r):
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1).min())


def make_thresholds():
    import split_trial
    # E0 from results/split_trial_v2: the audit-corrected engine, paper rules without HAR (as deployed)
    r = split_trial.load('E0')[0].values
    rows = []
    for h in (5, 10, 21, 42, 63, 84, 105, 126):
        dds = np.array([window_mdd(r[i:i + h]) for i in range(0, len(r) - h + 1)])
        rows.append({'horizon': h, 'dd_p05': float(np.percentile(dds, 5)), 'dd_min': float(dds.min()), 'windows': len(dds)})
    out = pd.DataFrame(rows)
    out.to_csv(THRESHOLDS, index=False)
    print(out.to_string(index=False, formatters={'dd_p05': '{:.1%}'.format, 'dd_min': '{:.1%}'.format}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(HERE, 'trading_system.db'))
    ap.add_argument('--start', type=dt.date.fromisoformat)
    ap.add_argument('--account-value', type=float)
    ap.add_argument('--etf-share', type=float, default=0.15)
    ap.add_argument('--thresholds', default=THRESHOLDS)
    ap.add_argument('--make-thresholds', action='store_true')
    a = ap.parse_args(argv)
    if a.make_thresholds:
        make_thresholds()
        return 0
    if a.start is None or a.account_value is None:
        ap.error('--start and --account-value are required')
    thr = pd.read_csv(a.thresholds) if os.path.exists(a.thresholds) else None
    if thr is None:
        print(f"warning: {a.thresholds} not found; early-stop check skipped")
    r = report(a.db, a.start, a.account_value, a.etf_share, thr)
    print(f"ETF paper-rules forward test T-2026-10-01-72, from {a.start}: {r['status']}")
    print(f"  sessions {r['sessions']}; ETF fills {r['fills']}")
    print(f"  net P&L at paper fills less commission: ${r['pnl']:,.0f}; no-rf Sharpe {r['sharpe']:.2f} (G1 needs > 0)")
    print(f"  median slippage vs decision midpoint {r['median_slippage_bps']:.2f} bp over {r['slippage_fills']} fills (G2 needs <= {MAX_MEDIAN_SLIPPAGE_BPS} bp)")
    thr_txt = f"{r['dd_threshold']:.1%}" if r['dd_threshold'] is not None else 'n/a'
    print(f"  drawdown {r['dd']:.1%} of sleeve capital; early-stop line {thr_txt}")
    if r['open_positions']:
        print(f"  open positions (not in realized P&L): {r['open_positions']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
