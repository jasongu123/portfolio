"""
Pairs trading across the whole S&P 500 — RESEARCH ONLY. Trial T-2026-10-07-96, pre-registered 2026-10-07
before any return was computed. Nothing here touches the live engine.

Why: the earlier stat-arb sleeve (stat_arb.py, PORTFOLIO.md scoreboard) traded at most 12 hand-related
large-cap pairs (cointegration, formed 2018-22, traded 2023-26) and died when one relationship broke
(AMD vs MSFT/QQQ ran to 5.2 sigma, -33%). The user's hypothesis: with many more pairs, one break cannot
sink the book. More pairs cut that risk but cannot create an edge, so the test asks whether the AVERAGE
pair still earns money net of costs since 2020.

Method: the distance method of Gatev, Goetzmann & Rouwenhorst (RFS 2006), the standard reference.
  universe   S&P 500 point-in-time members (taking_sides.membership) with IBKR adjusted closes, a member
             and >= $5 on the last formation day, no missing price in the formation window
  formation  252 trading days ending the day before trading starts; prices normalised to 1 at the start;
             distance = sum of squared differences of the normalised prices; the N closest pairs
             (a stock may be in several); sigma = std of the normalised price difference
  trading    the next 126 trading days; prices re-normalised to 1 at the last formation close.
             OPEN when |spread| > 2 sigma at a close: long the lower, short the higher, $1 each side,
             executed at the NEXT close (one-day wait removes bid-ask bounce); CLOSE at the next close
             after the spread crosses zero, or at the end of the period, or when a leg stops trading.
             A pair may reopen after it closes. Costs 5 bp per side per leg at entry and exit
             (20 bp per round trip per pair); 0.5% a year borrow on the short leg while open.
  returns    committed capital: each portfolio's daily return is the sum over its N pairs divided by N
             (idle pairs count as cash at 0%). A new portfolio starts on the first trading day of every
             month, so six overlap; the strategy's daily return is their average.
Main configuration: N = 100 pairs. Diagnostics (no decision): N = 20 (the paper's), 50, 200.
Main window: 2020-01-01 .. the last day with data (the last portfolios are cut short).
Context (no decision): 2005-2019, survivorship-biased (only names priced in the 2020-26 file).

Pre-registered decision (N = 100, main window), net of costs:
  P1 net Sharpe >= 0.50
  P2 Newey-West (20 lags) t of the mean daily net return >= 2.24 (second look at the pairs family)
  P3 net return over the last 24 months > 0
  P4 best 24-month block <= 75% of total net profit
  P5 |correlation with SPY daily returns| <= 0.30 (it should be market-neutral)
  PASS (all five) -> candidate for a paper forward test; no live change from this trial.
Reported: gross vs net; per-year returns; the N curve; how many positions converged versus were forced
shut at the end of the period, and the mean return of each (the de-cointegration diagnostic).

Usage: python3 pairs_broad.py      (minutes; writes results/pairs_broad/)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taking_sides import membership  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'pairs_broad')
FORM, TRADE, K_SIG = 252, 126, 2.0
COST_SIDE, BORROW = 0.0005, 0.005
N_MAIN, N_DIAG = 100, (20, 50, 200)
MAIN_START, CONTEXT = '2020-01-01', ('2005-01-01', '2019-12-31')
GATE = {'sharpe': 0.50, 't': 2.24, 'conc': 0.75, 'corr': 0.30}
ANN = np.sqrt(252)


def load():
    px = pd.read_csv(os.path.join(BD, 'sp500_prices_long.csv'), index_col=0, parse_dates=True).sort_index()
    uni = pd.read_csv(os.path.join(BD, 'sp500_universe_2020_2026.csv')).iloc[:, 0].astype(str).tolist()
    chg = pd.read_csv(os.path.join(BD, 'sp500_changes.csv'), parse_dates=['date'])
    mem = membership(chg, uni)
    days = px.index
    member = np.zeros(px.shape, bool)
    for k, s in enumerate(px.columns):
        if s in mem:
            a, rm = mem[s]
            m = np.ones(len(days), bool)
            if pd.notna(a):
                m &= days >= a
            if pd.notna(rm) and (pd.isna(a) or rm > a):
                m &= days < rm
            member[:, k] = m
    return px, member


def form_pairs(P, n):
    """P: formation prices (days x stocks), no NaN. Returns the n closest pairs and their sigma."""
    X = P / P[0]
    sq = (X ** 2).sum(0)
    ssd = sq[:, None] + sq[None, :] - 2 * X.T @ X
    iu = np.triu_indices(X.shape[1], 1)
    d = ssd[iu]
    pick = np.argsort(d)[:n]
    i, j = iu[0][pick], iu[1][pick]
    sig = (X[:, i] - X[:, j]).std(0)
    return i, j, sig


def trade_portfolio(Pt, R, i, j, sig, n):
    """Pt: prices from the last formation day through the trading period (T+1 x stocks), R: daily
    returns over the trading days (T x stocks). Returns gross and net daily returns and position logs."""
    T = R.shape[0]
    gross, net = np.zeros(T), np.zeros(T)
    logs = []
    for a, b, s in zip(i, j, sig):
        pa, pb = Pt[:, a] / Pt[0, a], Pt[:, b] / Pt[0, b]
        ra, rb = R[:, a], R[:, b]
        pos, want, opened, acc = 0, 0, None, 0.0
        for t in range(T):
            # returns of day t accrue to the position held at the close of t-1
            if pos != 0:
                if np.isnan(ra[t]) or np.isnan(rb[t]):
                    logs.append((acc, 'stopped trading'))
                    pos, opened, acc = 0, None, 0.0
                    continue
                g = pos * (ra[t] - rb[t])            # pos=+1: long a, short b
                c = BORROW / 252
                gross[t] += g
                net[t] += g - c
                acc += g - c
            # decisions at the close of t, executed at the close of t+1 (one-day wait)
            if want != 0 and pos == 0:               # execute a pending open at this close
                pos, opened, acc = want, t, -2 * COST_SIDE
                net[t] -= 2 * COST_SIDE
                want = 0
            elif want == 2 and pos != 0:             # execute a pending close
                net[t] -= 2 * COST_SIDE
                logs.append((acc - 2 * COST_SIDE, 'converged'))
                pos, opened, acc, want = 0, None, 0.0, 0
                continue
            if np.isnan(pa[t + 1]) or np.isnan(pb[t + 1]):
                continue
            spread = pa[t + 1] - pb[t + 1]
            if pos == 0 and want == 0 and abs(spread) > K_SIG * s:
                want = -1 if spread > 0 else 1         # a rich: short a, long b
            elif pos != 0 and want == 0 and np.sign(spread) != np.sign(-pos) and spread != 0:
                want = 2                               # spread crossed zero: close next close
        if pos != 0:
            net[T - 1] -= 2 * COST_SIDE
            logs.append((acc - 2 * COST_SIDE, 'forced at period end'))
    return gross / n, net / n, logs


def run(px, member, n):
    days = px.index
    P_all = px.to_numpy()
    R_all = px.pct_change(fill_method=None).to_numpy()
    starts = [k for k in range(FORM, len(days)) if days[k].month != days[k - 1].month]
    g_tot, n_tot, cnt = np.zeros(len(days)), np.zeros(len(days)), np.zeros(len(days))
    logs = []
    for k in starts:
        if days[k] < pd.Timestamp('2005-01-01'):
            continue
        f0, f1 = k - FORM, k - 1                      # formation days f0..f1 (prices), trading k..k+125
        win = P_all[f0:f1 + 1]
        ok = member[f1] & ~np.isnan(win).any(0) & (P_all[f1] >= 5)
        cols = np.where(ok)[0]
        if len(cols) < 50:
            continue
        i, j, sig = form_pairs(win[:, cols], n)
        end = min(k + TRADE, len(days))
        Pt = P_all[f1:end][:, cols]
        R = R_all[k:end][:, cols]
        g, ne, lg = trade_portfolio(Pt, R, i, j, sig, n)
        g_tot[k:end] += g
        n_tot[k:end] += ne
        cnt[k:end] += 1
        logs += [(days[k], *x) for x in lg]
    with np.errstate(invalid='ignore', divide='ignore'):
        gross = pd.Series(np.where(cnt > 0, g_tot / cnt, np.nan), index=days)
        net = pd.Series(np.where(cnt > 0, n_tot / cnt, np.nan), index=days)
    return gross.dropna(), net.dropna(), pd.DataFrame(logs, columns=['start', 'ret', 'exit'])


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 20 and x.std() > 0 else float('nan')


def nw_t(x, lags=20):
    x = np.asarray(x.dropna(), float)
    u = x - x.mean()
    n = len(x)
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return float(x.mean() / np.sqrt(var / n))


def conc(x):
    tot = x.sum()
    return float(x.rolling(504, min_periods=400).sum().max() / tot) if tot > 0 else float('inf')


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    px, member = load()
    spy = pd.read_csv(os.path.join(BD, 'sector_spdr_adjclose.csv'), index_col=0, parse_dates=True)['SPY'].pct_change()
    res = {}
    for n in (N_MAIN,) + N_DIAG:
        g, ne, lg = run(px, member, n)
        res[n] = (g, ne, lg)
        pd.DataFrame({'gross': g, 'net': ne}).to_csv(os.path.join(OUT, f'daily_N{n}.csv'))
    emit("S&P 500 pairs, distance method (GGR 2006), committed capital, net of 5 bp/side/leg + borrow\n")
    emit(f"{'N':>4s} {'period':10s} {'gross SR':>9s} {'net SR':>7s} {'net %/yr':>9s} {'NW t':>6s} {'corr SPY':>9s}")
    for n, (g, ne, _) in res.items():
        for (lo, hi), per in (((MAIN_START, '2100-01-01'), '2020+'), (CONTEXT, '2005-19')):
            x, y = ne[lo:hi], g[lo:hi]
            c = x.corr(spy.reindex(x.index))
            emit(f"{n:4d} {per:10s} {sharpe(y):+9.2f} {sharpe(x):+7.2f} {x.mean() * 252 * 100:+9.2f} {nw_t(x):+6.2f} {c:+9.2f}")
    g, ne, lg = res[N_MAIN]
    yr = ne.groupby(ne.index.year).sum() * 100
    emit(f"\nN={N_MAIN} net return by year %: " + ", ".join(f"{y} {v:+.1f}" for y, v in yr.items()))
    lg = lg[lg.start >= MAIN_START]
    emit(f"\nPositions opened 2020+ (N={N_MAIN}): {len(lg)}; by exit: " + "; ".join(
        f"{k} {len(v)} ({len(v) / len(lg):.0%}), mean {v.ret.mean() * 100:+.2f}%" for k, v in lg.groupby('exit')))

    x = ne[MAIN_START:]
    last24 = x[x.index > x.index[-1] - pd.DateOffset(months=24)].sum()
    c = x.corr(spy.reindex(x.index))
    checks = [('P1 net Sharpe >= 0.50', sharpe(x) >= GATE['sharpe'], f"{sharpe(x):.2f}"),
              ('P2 NW t >= 2.24', nw_t(x) >= GATE['t'], f"{nw_t(x):.2f}"),
              ('P3 last 24 months > 0', last24 > 0, f"{last24 * 100:+.2f}%"),
              ('P4 best 24m <= 75% of profit', conc(x) <= GATE['conc'], f"{conc(x):.0%}" if np.isfinite(conc(x)) else 'no profit'),
              ('P5 |corr with SPY| <= 0.30', abs(c) <= GATE['corr'], f"{c:+.2f}")]
    emit(f"\nGate (T-2026-10-07-96), N={N_MAIN}, {x.index[0]:%Y-%m-%d} .. {x.index[-1]:%Y-%m-%d}:")
    for name, ok, val in checks:
        emit(f"  {name:32s} {val:14s} {'ok' if ok else 'FAIL'}")
    emit(f"\nT-96 VERDICT: {'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'}")
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
