"""
Per-name vs pooled gate trial — RESEARCH ONLY. Pre-registered 2026-10-01 before any run.
ibkr2.py and ibkr_backtest.py untouched (everything lives in the GateBT subclass below).

Question 1: is one gate model per stock better than one model pooled across stocks?
Question 2: is the LIVE gate any good? Live ibkr2.py differs from the validated backtest gate in
four ways at once: (a) one forest per symbol; (b) fitted once at startup on the last ~1000 NYSE
sessions instead of rolling on the latest 200 outcomes; (c) long-only labels (BUY candidate that
made > +0.2% = 1, every SELL candidate = 0) instead of direction-aware > +0.1%; (d) asymmetric
0.65 threshold: a BUY needs P(win) >= 0.65, a SELL is allowed unless P >= 0.65.

Book and costs exactly as gate_trial.py (24 names, hybrid engine, per-minute VIX, $0.0035/share +
1 bp, one backtest per year). Minute data starts late 2019, so every model here can only learn
from 2020 onward: the evaluation window is 2021-01-01..2025-12-15, and the static arms' "last
1000 sessions" window holds 1, 2, 3, 4, 4 years of history in 2021..2025 (full only in 2024-25).

Step 0 (harvest): re-run the ungated arm for 2020-2024 and save every labelled candidate
(symbol, direction, exit time, simulated profit, the 33 features). This is the ungated candidate
stream, which is what live trains on too. Sanity check fixed in advance: each harvest year's
daily returns must equal gate_trial's cached none_{year}.csv (max abs diff < 1e-9); otherwise
stop and investigate before reading any arm.

Arms (all random forests with the validated hyper-parameters; same features BASE_33)
  Q0 pooled      pooled, rolling 200 / retrain every 30 / min 50, P>=0.55, warm-started from the
                 harvest before the year (CONTROL: the validated design, warm)
  Q1 pername     one model per symbol, otherwise identical to Q0 (each symbol's own last 200,
                 retrain every 30 of its own outcomes, min 50; a symbol with no model passes
                 ungated, as the pooled gate does before its first model)
  Q2 perstatic   per symbol, fitted once at the year start on the harvest from the last 1000
                 NYSE sessions (all of it, live-style), backtest labels, P>=0.55 symmetric
                 (diagnostic: separates (a)+(b) from (c)+(d))
  Q3 live        per symbol, fitted once on the last 1000 sessions, live labels, min 30 samples,
                 live asymmetric 0.65 rule (the live gate, as close as the backtester allows)
  References, cached from gate_trial: G0 none (no gate), G1 rf (pooled, cold start each year).

Pre-registered decisions (no-rf daily Sharpe on 2021-01-01..2025-12-15; rf flat 5% where stated;
MaxDD from initial capital; blocks 2021-22, 2023-24, 2025; recent = 2024-09..2025-12):
  D1 per-name vs pooled: Q1 replaces pooled only if ALL hold versus Q0:
     M1 Sharpe >= Q0 + 0.10; M2 better in >= 2 of 3 blocks; M3 recent >= Q0 - 0.05;
     M4 MaxDD not deeper by > 1.0 point; M5 walk-forward book with the live trend sleeve rf-Sharpe
     >= Q0's book. Otherwise pooled stays the validated design.
  D2 live gate: acceptable only if ALL hold:
     L1 Q3 Sharpe >= Q0 - 0.10; L2 Q3 recent >= Q0 recent - 0.10; L3 Q3 Sharpe >= G0 (no gate).
     Otherwise the recommendation is to move the live gate to the validated design (Q0, or Q1
     if D1 passed). Q2 is diagnostic only: it says whether the gap comes from per-name static
     fitting or from the live labels and threshold.
Trials: T-2026-10-01-55 (Q0) .. -58 (Q3).

Usage:
  python3 pergate_trial.py --harvest 2021      ungated run of one year, saves its candidates
  python3 pergate_trial.py --arm pername --year 2023
  python3 pergate_trial.py --report
"""
import argparse
import contextlib
import datetime
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate_trial import CORE, BREADTH, OUT as GATE_OUT, load as gate_load, sr, mdd, ANN  # noqa: E402

OUT = os.path.join(HERE, 'results', 'pergate_trial')
HARVEST_YEARS = [2020, 2021, 2022, 2023, 2024]
EVAL_YEARS = [2021, 2022, 2023, 2024, 2025]
ARMS = ['pooled', 'pername', 'perstatic', 'live']
WINDOW_SESSIONS = 1000


def year_span(y):
    return f'{y}-01-01', ('2025-12-15' if y == 2025 else f'{y}-12-31')


def base_kwargs(y, gated):
    start, end = year_span(y)
    return dict(symbols=CORE + BREADTH, start_date=start, end_date=end, initial_capital=100000,
                use_stock_strategies=True, stock_strategy_mode='hybrid', gap_threshold=0.015, gap_exit_minutes=60,
                use_stock_vol_regime=True, vol_lookback=20, use_rf=gated, use_lgbm=False,
                volatility_multiplier=2.0, commission_per_share=0.0035, slippage_bps=1)


def make_class():
    from ibkr_backtest import StrategyBacktester
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    def forest(class_weight):
        return RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=5, min_samples_leaf=3,
                                      random_state=42, bootstrap=True, oob_score=True, class_weight=class_weight)

    class GateBT(StrategyBacktester):
        """Tags each candidate's features with its symbol and direction, and swaps the gate."""

        def __init__(self, *a, style='none', **kw):
            super().__init__(*a, **kw)
            self.style = style
            self.sym_models = {}
            self.sym_data = {}

        def extract_rf_features(self, symbol, trade_date, bar_idx, day_bars, prev_close, candidate_direction=0):
            f = super().extract_rf_features(symbol, trade_date, bar_idx, day_bars, prev_close,
                                            candidate_direction=candidate_direction)
            if f is None:
                return None
            f = dict(f)
            f['__sym'] = symbol
            f['__dir'] = int(np.sign(candidate_direction))
            return f

        # -- warm start / static fit from the harvest ------------------------------------
        def prepare(self, prior):
            """prior: harvest DataFrame of samples whose exit is before this run's start."""
            if self.style == 'pooled':
                self.rf_training_data = [self._sample(r) for r in prior.itertuples()]
                if len(self.rf_training_data) >= self.rf_min_samples:
                    self.train_rf_model(walk_forward_window=self.rf_walk_forward_window)
            elif self.style == 'pername':
                for sym, g in prior.groupby('sym'):
                    self.sym_data[sym] = [self._sample(r) for r in g.itertuples()]
                    self._train_sym(sym)
            elif self.style in ('perstatic', 'live'):
                live = self.style == 'live'
                min_n = 30 if live else 50
                for sym, g in prior.groupby('sym'):
                    g = g.sort_values('ts')
                    X = g[self.ml_feature_names].to_numpy(float)
                    y = (((g['dir'] > 0) & (g['profit'] > 0.002)) if live else (g['profit'] > self.ml_profit_target_pct)).astype(int).to_numpy()
                    u, c = np.unique(y, return_counts=True)
                    if len(y) < min_n or len(u) < 2:
                        continue
                    cw = 'balanced' if c.max() / c.min() > 3 else None
                    sc = StandardScaler().fit(X)
                    self.sym_models[sym] = (forest(cw).fit(sc.transform(X), y), sc)

        def _sample(self, r):
            feats = {n: getattr(r, n) for n in self.ml_feature_names}
            return {'features': feats, 'label': int(r.profit > self.ml_profit_target_pct),
                    'profit_pct': r.profit, 'timestamp': r.ts}

        def _train_sym(self, sym):
            data = self.sym_data.get(sym, [])
            if len(data) < self.rf_min_samples:
                return
            saved = (self.rf_training_data, self.rf_model, self.rf_scaler)
            self.rf_training_data, self.rf_model, self.rf_scaler = data, None, StandardScaler()
            try:
                self.train_rf_model(walk_forward_window=self.rf_walk_forward_window)
                if self.rf_model is not None:
                    self.sym_models[sym] = (self.rf_model, self.rf_scaler)
            finally:
                self.rf_training_data, self.rf_model, self.rf_scaler = saved

        # -- online updates -----------------------------------------------------------
        def record_trade_outcome(self, features, entry_price, exit_price, signal, timestamp):
            if self.style in ('none', 'pooled'):
                return super().record_trade_outcome(features, entry_price, exit_price, signal, timestamp)
            if self.style != 'pername' or features is None:
                return  # static arms are frozen for the year, as live is between restarts
            profit = exit_price / entry_price - 1 if signal == 'BUY' else entry_price / exit_price - 1
            sym = features['__sym']
            self.sym_data.setdefault(sym, []).append({'features': features, 'label': int(profit > self.ml_profit_target_pct),
                                                      'profit_pct': profit, 'timestamp': timestamp})
            if len(self.sym_data[sym]) % 30 == 0:
                self._train_sym(sym)

        # -- the gate -----------------------------------------------------------------
        def _ml_gate_ready(self):
            if self.style in ('pername', 'perstatic', 'live'):
                return True  # per-symbol readiness is decided in get_rf_signal
            return super()._ml_gate_ready()

        def get_ml_signal(self, features):
            # The base routes to get_rf_signal only when the pooled rf_model exists.
            if self.style in ('pername', 'perstatic', 'live'):
                return self.get_rf_signal(features)
            return super().get_ml_signal(features)

        def get_rf_signal(self, features):
            if self.style not in ('pername', 'perstatic', 'live'):
                return super().get_rf_signal(features)
            if features is None:
                return None, 0.0
            m = self.sym_models.get(features.get('__sym'))
            if m is None:
                return 'TRADE', 1.0  # no model for this symbol yet: pass ungated
            model, sc = m
            x = sc.transform(np.array([self._ordered_feature_values(features)]))
            proba = model.predict_proba(x)[0]
            classes = list(model.classes_)
            p = float(proba[classes.index(1)]) if 1 in classes else 0.0
            if self.style == 'live':
                ok = p >= 0.65 if features.get('__dir', 0) > 0 else p < 0.65
                return ('TRADE', 1.0) if ok else ('HOLD', 0.0)
            return ('TRADE' if model.predict(x)[0] == 1 else 'HOLD'), p

    return GateBT


def _run(bt):
    eq = bt.run_backtest(long_only=False)
    r = eq['portfolio_value'].pct_change()
    r.index = pd.to_datetime(eq['date'])
    return r.dropna()


def harvest(y):
    from feature_sweep import BASE_33, _ProgressFilter
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f'harvest_{y}.pkl')
    if os.path.exists(f):
        print(f"  harvest {y}: cached", flush=True)
        return
    t0 = time.time()
    GateBT = make_class()
    with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
        bt = GateBT(**base_kwargs(y, gated=False), style='none')
        bt.load_from_csv()
        bt.ml_feature_names = list(BASE_33)
        r = _run(bt)
    rows = [{'sym': s['features']['__sym'], 'dir': s['features']['__dir'], 'ts': pd.Timestamp(s['timestamp']),
             'profit': float(s['profit_pct']), **{n: float(s['features'].get(n, 0.0)) for n in BASE_33}}
            for s in bt.rf_training_data]
    df = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, f'harvest_{y}_returns.csv'))
    with open(f, 'wb') as fh:
        pickle.dump(df, fh)
    print(f"  harvest {y}: {len(df)} samples, {(time.time() - t0) / 60:.0f} min", flush=True)


def prior_samples(y, style):
    parts = [pd.read_pickle(os.path.join(OUT, f'harvest_{h}.pkl')) for h in HARVEST_YEARS if h < y]
    df = pd.concat(parts, ignore_index=True)
    start = pd.Timestamp(f'{y}-01-01')
    df = df[df['ts'] < start]
    if style in ('perstatic', 'live'):
        import index_coverage
        first = index_coverage.recent_sessions(WINDOW_SESSIONS, datetime.date(y, 1, 1))[0]
        df = df[df['ts'] >= pd.Timestamp(first)]
    return df.sort_values('ts').reset_index(drop=True)


def run_arm(arm, y):
    from feature_sweep import BASE_33, _ProgressFilter
    f = os.path.join(OUT, f'{arm}_{y}.csv')
    if os.path.exists(f):
        print(f"  {arm} {y}: cached", flush=True)
        return
    t0 = time.time()
    prior = prior_samples(y, arm)
    GateBT = make_class()
    with contextlib.redirect_stdout(_ProgressFilter(sys.stdout)):
        bt = GateBT(**base_kwargs(y, gated=True), style=arm)
        bt.load_from_csv()
        bt.ml_feature_names = list(BASE_33)
        bt.prepare(prior)
        n_models = len(bt.sym_models) if arm != 'pooled' else int(bt.rf_model is not None)
        r = _run(bt)
    r.to_csv(f)
    print(f"  {arm} {y}: done ({len(r)} days, {len(prior)} prior samples, {n_models} models at start, "
          f"{(time.time() - t0) / 60:.0f} min)", flush=True)


def check_harvest():
    worst = 0.0
    for y in HARVEST_YEARS:
        a = pd.read_csv(os.path.join(OUT, f'harvest_{y}_returns.csv'), index_col=0, parse_dates=True).iloc[:, 0]
        b = pd.read_csv(os.path.join(GATE_OUT, f'none_{y}.csv'), index_col=0, parse_dates=True).iloc[:, 0]
        d = float((a - b).abs().max())
        worst = max(worst, d)
        print(f"  harvest {y} vs cached G0: max abs diff {d:.2e}")
    return worst < 1e-9


def load(arm):
    parts = [pd.read_csv(os.path.join(OUT, f'{arm}_{y}.csv'), index_col=0, parse_dates=True).iloc[:, 0] for y in EVAL_YEARS]
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()].rename(arm)


def report():
    import book_backtest as bb
    ok = check_harvest()
    print("SANITY: harvest reproduces G0" if ok else "SANITY FAILED: harvest differs from G0; do not read the arms")
    avail = [a for a in ARMS if all(os.path.exists(os.path.join(OUT, f'{a}_{y}.csv')) for y in EVAL_YEARS)]
    series = [load(a) for a in avail] + [gate_load('none').rename('G0_none'), gate_load('rf').rename('G1_cold')]
    S = pd.concat(series, axis=1).loc['2021-01-01':'2025-12-15'].dropna()
    blocks = (('2021-22', '2021', '2022'), ('2023-24', '2023', '2024'), ('2025', '2025', '2025'))
    tr = pd.read_csv(os.path.join(HERE, 'results', 'headline_2026-09_daily_new.csv'), index_col=0, parse_dates=True)['trend_live']
    rows = {}
    for a in S.columns:
        x = S[a]
        full = pd.concat([x.rename('equity'), tr.rename('trend')], axis=1)
        bb.FULL_FRAME = full
        book = bb.book_returns(full.loc['2021-01-01':'2025-12-15'].dropna(), 'WF')
        rows[a] = {'norf': sr(x), 'rf': float(((x - 0.05 / 252).mean() / x.std()) * ANN), 'vol': float(x.std() * ANN),
                   'mdd': mdd(x), 'recent': sr(x.loc['2024-09-01':]),
                   **{b: sr(x.loc[s:e]) for b, s, e in blocks},
                   **{str(y): sr(x.loc[str(y)]) for y in EVAL_YEARS},
                   'corr_Q0': float(x.corr(S['pooled'])) if 'pooled' in S else np.nan,
                   'book_rf': bb.sharpe(book), 'book_norf': bb.sharpe(book, 0), 'book_mdd': bb.maxdd(book)}
    df = pd.DataFrame(rows).T
    pd.set_option('display.width', 220)
    print(df.to_string(float_format=lambda v: f'{v:.2f}'))
    df.to_csv(os.path.join(OUT, 'summary.csv'))
    if 'pooled' not in df.index:
        return
    q0 = df.loc['pooled']
    if 'pername' in df.index:
        r = df.loc['pername']
        nb = sum(r[b] > q0[b] for b, _, _ in blocks)
        m = [r.norf >= q0.norf + 0.10, nb >= 2, r.recent >= q0.recent - 0.05, r.mdd >= q0.mdd - 0.01, r.book_rf >= q0.book_rf]
        print(f"\nD1 per-name vs pooled: M1 {r.norf:.2f} vs {q0.norf + 0.10:.2f} {'ok' if m[0] else 'fail'} | M2 {nb}/3 {'ok' if m[1] else 'fail'} | "
              f"M3 {r.recent:+.2f} vs {q0.recent - 0.05:+.2f} {'ok' if m[2] else 'fail'} | M4 {r.mdd:.1%} vs {q0.mdd - 0.01:.1%} {'ok' if m[3] else 'fail'} | "
              f"M5 {r.book_rf:.2f} vs {q0.book_rf:.2f} {'ok' if m[4] else 'fail'} => {'PER-NAME WINS' if all(m) else 'pooled stays'}")
    if 'live' in df.index:
        r = df.loc['live']
        g0 = df.loc['G0_none']
        l = [r.norf >= q0.norf - 0.10, r.recent >= q0.recent - 0.10, r.norf >= g0.norf]
        print(f"D2 live gate: L1 {r.norf:.2f} vs {q0.norf - 0.10:.2f} {'ok' if l[0] else 'fail'} | L2 {r.recent:+.2f} vs {q0.recent - 0.10:+.2f} "
              f"{'ok' if l[1] else 'fail'} | L3 {r.norf:.2f} vs no-gate {g0.norf:.2f} {'ok' if l[2] else 'fail'} => "
              f"{'live gate acceptable' if all(l) else 'move the live gate to the validated design'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--harvest', type=int)
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--year', type=int)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.harvest:
        harvest(a.harvest)
    if a.arm:
        run_arm(a.arm, a.year)
    if a.report:
        report()
