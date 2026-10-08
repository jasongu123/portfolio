"""
Gate-model trial — RESEARCH ONLY. Pre-registered 2026-09-28 before any run. ibkr2.py untouched.

Question: does the equity engine's ML gate depend on its model class, and does the gate add
anything at all? The backtester's gate (the one behind the headline) is a single random forest
pooled across the run's symbols, trained on the most recent 200 candidate outcomes (every
candidate is labelled, taken or not: win = simulated trade > +0.1%), retrained every 30 new
outcomes, entry allowed when P(win) >= 0.55. Arms change ONLY the model; data, features
(BASE_33), scaling, window, retrain cadence, class weighting and the 0.55 threshold are the same.

Book: the 24 names the paper engine now trades (core 12 + breadth 12) in one backtest per year,
2020-01-01..2025-12-15, combine_book's configuration, per-minute VIX gate, $0.0035/share + 1 bp.
Each year starts with no gate model (candidates pass until the first model trains), as in the
headline runs.

Arms
  G0 none      no ML gate                                   reference, not a candidate
  G1 rf        RandomForest(50 trees, depth 5, leaf 3)      control (the validated gate)
  G2 logit     L2 logistic regression, C=0.1                is the forest better than linear?
  G3 lgbm      LightGBM, 100 trees, 7 leaves, depth 3, lr 0.05, min 10 per leaf, bagging 0.8
  G4 lgbm_cal  G3 + sigmoid calibration on the walk-forward validation slice (the newest 20%)

Pre-registered decision (no-rf daily Sharpe; rf flat 5% where stated; MaxDD from initial capital):
  An arm G2-G4 replaces the forest only if ALL hold versus G1 on 2020-2025:
    M1 no-rf Sharpe >= G1 + 0.10
    M2 better in at least 2 of the 3 blocks 2020-21, 2022-23, 2024-25
    M3 2024-09..2025-12 no-rf Sharpe >= G1's - 0.05
    M4 MaxDD not deeper than G1's by more than 1.0 point
    M5 walk-forward book with the live trend sleeve: rf-Sharpe >= the G1 book's
  If several pass, the highest M1 wins; within 0.05, the simpler model (logit < lgbm < lgbm_cal).
  Reading fixed in advance for G0: if G1 does not beat G0 by >= 0.10 no-rf Sharpe, the gate adds
  nothing measurable in this backtest, which bears on the live gate too.
Trials: T-2026-09-28-50 (G0) .. -54 (G4).

Usage:
  /usr/local/bin/python3 gate_trial.py --arm rf            run missing years for one arm
  /usr/local/bin/python3 gate_trial.py --report            gate + tables from cached results
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'gate_trial')
CORE = ['SPY', 'QQQ', 'IWM', 'AAPL', 'NVDA', 'GOOG', 'MSFT', 'META', 'JPM', 'WMT', 'UNH', 'XOM']
BREADTH = ['AMZN', 'AVGO', 'LLY', 'BAC', 'COST', 'CVX', 'CAT', 'NFLX', 'LIN', 'NEE', 'PLD', 'MU']
YEARS = {str(y): (f'{y}-01-01', f'{y}-12-31') for y in range(2020, 2025)}
YEARS['2025'] = ('2025-01-01', '2025-12-15')
ARMS = ['none', 'rf', 'logit', 'lgbm', 'lgbm_cal']
ANN = np.sqrt(252)


def factory_for(arm):
    if arm == 'logit':
        from sklearn.linear_model import LogisticRegression
        return lambda cw: LogisticRegression(C=0.1, class_weight=cw, max_iter=2000)
    if arm in ('lgbm', 'lgbm_cal'):
        from lightgbm import LGBMClassifier
        return lambda cw: LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7, max_depth=3,
                                         min_child_samples=10, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, reg_lambda=1.0, class_weight=cw,
                                         random_state=42, verbose=-1)
    return None


def run_arm(arm):
    from ibkr_backtest import StrategyBacktester
    from feature_sweep import BASE_33, _ProgressFilter
    os.makedirs(OUT, exist_ok=True)
    for y, (start, end) in YEARS.items():
        f = os.path.join(OUT, f'{arm}_{y}.csv')
        if os.path.exists(f):
            print(f"  {arm} {y}: cached", flush=True)
            continue
        t0 = time.time()
        print(f"  {arm} {y}: running...", flush=True)
        kw = dict(symbols=CORE + BREADTH, start_date=start, end_date=end, initial_capital=100000,
                  use_stock_strategies=True, stock_strategy_mode='hybrid', gap_threshold=0.015, gap_exit_minutes=60,
                  use_stock_vol_regime=True, vol_lookback=20, use_rf=(arm != 'none'), use_lgbm=False,
                  volatility_multiplier=2.0, commission_per_share=0.0035, slippage_bps=1,
                  gate_model_factory=factory_for(arm), gate_calibration=('sigmoid' if arm == 'lgbm_cal' else None))
        with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
            bt = StrategyBacktester(**kw)
            bt.load_from_csv()
            bt.ml_feature_names = list(BASE_33)
            eq = bt.run_backtest(long_only=False)
        pv = eq['portfolio_value'] if hasattr(eq, 'columns') else pd.DataFrame(eq)['portfolio_value']
        r = pv.pct_change()
        r.index = pd.to_datetime(eq['date'])
        r = r.dropna()
        stats = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                 for k, v in getattr(bt, 'ml_gate_stats', {}).items()}
        with open(os.path.join(OUT, f'{arm}_{y}_gate_stats.json'), 'w') as fh:
            json.dump(stats, fh, indent=1)
        r.to_csv(f)
        print(f"  {arm} {y}: done ({len(r)} days, {(time.time() - t0) / 60:.0f} min)", flush=True)


def load(arm):
    parts = [pd.read_csv(os.path.join(OUT, f'{arm}_{y}.csv'), index_col=0, parse_dates=True).iloc[:, 0] for y in YEARS]
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()].rename(arm)


def sr(x):
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def mdd(x):
    eq = np.cumprod(1 + x.dropna().values)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((eq / peak - 1).min())


def report():
    import book_backtest as bb
    avail = [a for a in ARMS if all(os.path.exists(os.path.join(OUT, f'{a}_{y}.csv')) for y in YEARS)]
    if 'rf' not in avail:
        sys.exit("G1 (rf) not complete yet")
    S = pd.concat([load(a) for a in avail], axis=1).dropna()
    blocks = (('2020-21', '2020', '2021'), ('2022-23', '2022', '2023'), ('2024-25', '2024', '2025'))
    tr = pd.read_csv(os.path.join(HERE, 'results', 'headline_2026-09_daily_new.csv'), index_col=0, parse_dates=True)['trend_live']
    rows = {}
    for a in avail:
        x = S[a]
        full = pd.concat([x.rename('equity'), tr.rename('trend')], axis=1)
        bb.FULL_FRAME = full
        book = bb.book_returns(full.loc['2020-01-01':'2025-12-31'].dropna(), 'WF')
        stats = {}
        for y in YEARS:
            f = os.path.join(OUT, f'{a}_{y}_gate_stats.json')
            if os.path.exists(f):
                for k, v in json.load(open(f)).items():
                    if isinstance(v, (int, float)):
                        stats[k] = stats.get(k, 0) + v
        rows[a] = {'norf': sr(x), 'rf': float(((x - 0.05 / 252).mean() / x.std()) * ANN), 'vol': float(x.std() * ANN),
                   'mdd': mdd(x), 'recent': sr(x.loc['2024-09-01':]),
                   **{b: sr(x.loc[s:e]) for b, s, e in blocks},
                   **{str(y): sr(x.loc[y]) for y in YEARS},
                   'book_rf': bb.sharpe(book), 'book_norf': bb.sharpe(book, 0), 'book_mdd': bb.maxdd(book),
                   'approved': stats.get('active_approved', np.nan), 'blocked': stats.get('active_blocked', np.nan),
                   'candidates': stats.get('base_candidates', np.nan)}
    df = pd.DataFrame(rows).T
    pd.set_option('display.width', 200)
    cols = ['norf', 'rf', 'vol', 'mdd', 'recent', '2020-21', '2022-23', '2024-25'] + list(YEARS) + ['book_rf', 'book_norf', 'book_mdd', 'candidates', 'approved', 'blocked']
    print(df[cols].to_string(float_format=lambda v: f'{v:.2f}' if abs(v) < 50 else f'{v:,.0f}'))
    print(f"\ncorr of daily returns with G1: " + ", ".join(f"{a} {S[a].corr(S['rf']):.2f}" for a in avail if a != 'rf'))
    g1 = df.loc['rf']
    print("\nGO/NO-GO (pre-registered, vs G1 rf):")
    passing = []
    for a in [x for x in ('logit', 'lgbm', 'lgbm_cal') if x in df.index]:
        r = df.loc[a]
        m1 = r.norf >= g1.norf + 0.10
        m2 = sum(r[b] > g1[b] for b, _, _ in blocks) >= 2
        m3 = r.recent >= g1.recent - 0.05
        m4 = r.mdd >= g1.mdd - 0.01
        m5 = r.book_rf >= g1.book_rf
        ok = all([m1, m2, m3, m4, m5])
        passing.append((a, r.norf)) if ok else None
        print(f"  {a:9s} M1 {r.norf:.2f} vs {g1.norf + 0.10:.2f} {'ok' if m1 else 'fail'} | M2 {sum(r[b] > g1[b] for b, _, _ in blocks)}/3 {'ok' if m2 else 'fail'} | "
              f"M3 {r.recent:+.2f} vs {g1.recent - 0.05:+.2f} {'ok' if m3 else 'fail'} | M4 {r.mdd:.1%} vs {g1.mdd - 0.01:.1%} {'ok' if m4 else 'fail'} | "
              f"M5 {r.book_rf:.2f} vs {g1.book_rf:.2f} {'ok' if m5 else 'fail'} => {'PASS' if ok else 'fail'}")
    if 'none' in df.index:
        gv = g1.norf - df.loc['none'].norf
        print(f"\nGate value (G1 - G0) no-rf Sharpe {gv:+.2f}: {'the gate adds value' if gv >= 0.10 else 'the gate adds nothing measurable (pre-registered reading)'}")
    order = {'logit': 0, 'lgbm': 1, 'lgbm_cal': 2}
    if passing:
        best = max(passing, key=lambda t: t[1])
        near = [p for p in passing if best[1] - p[1] <= 0.05]
        pick = min(near, key=lambda t: order[t[0]])
        print(f"VERDICT: replace the forest with {pick[0]}")
    else:
        print("VERDICT: keep the forest (no arm passed)")
    df.to_csv(os.path.join(OUT, 'summary.csv'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.arm:
        run_arm(a.arm)
    if a.report:
        report()
