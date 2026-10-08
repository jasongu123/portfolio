"""
Residual statistical arbitrage after Avellaneda & Lee (2010) — RESEARCH ONLY. Trial T-2026-10-07-97,
pre-registered 2026-10-07 before any return was computed. Nothing here touches the live engine.

Source: M. Avellaneda and J.-H. Lee, "Statistical arbitrage in the US equities market", Quantitative
Finance 10(7), 2010. Their ETF-based version earned a Sharpe of 1.1 after costs over 1997-2007, weaker
after 2002. This is a replication of that version on data after their sample, with the changes below.
The textbook pairs method (T-96) earned 0.22; this tests the factor-residual design that hedge funds
use in industrial form.

Method (the ETF version of the paper; deviations marked *):
  hedge ETF  * each stock is matched to the one of the 9 Select Sector SPDRs (XLB XLE XLF XLI XLK XLP XLU
             XLV XLY) whose daily returns correlate most with it over the previous 252 days, re-chosen
             at each month end (the paper used each stock's industry ETF; no sector map is available)
  residual   OLS of the stock's daily returns on its ETF's over the last 60 trading days:
             R_s = alpha + beta R_etf + eps; X = cumulative eps over the window
  OU fit     AR(1) of X: X[k+1] = a + b X[k] + zeta; kappa = -ln(b) * 252; keep only 0 < b < 1 and
             kappa > 252/30 (mean reversion faster than 30 trading days, as in the paper);
             m = a / (1 - b), sigma_eq = sqrt(var(zeta) / (1 - b^2))
  s-score    s = (-m + mean of m across that day's eligible stocks) / sigma_eq (the paper's centred
             score; X at the window end is 0 by construction of OLS)
  rules      the paper's: open SHORT (sell the stock, buy beta of the ETF) if s > +1.25; open LONG if
             s < -1.25; close a short when s < +0.75; close a long when s > -0.50
  timing     * signals at a close are executed at the NEXT close (the paper traded at the signal close;
             that version is reported as a diagnostic, not a decision)
  returns    each open position earns pos * (stock return - beta * ETF return), beta from the latest
             estimate; the book holds its open positions in equal weight (employed capital)
  costs      * 5 bp per side on the stock and 1 bp on the ETF leg (6 bp per side per position; the paper
             used 5 bp); 0.5% a year borrow while short a stock
  universe   S&P 500 point-in-time members, >= $5, 60 complete days in the window (taking_sides.membership,
             IBKR adjusted closes)
Main window: 2020-01-01 .. the last day with data. Context (no decision): 2005-2019, survivorship-biased.

Pre-registered decision (main window, next-close execution, net of costs):
  P1 net Sharpe >= 0.50
  P2 Newey-West (20 lags) t of the mean daily net return >= 2.24
  P3 net return over the last 24 months > 0
  P4 best 24-month block <= 75% of total net profit
  P5 |correlation with SPY daily returns| <= 0.30
  PASS (all five) -> candidate for a paper forward test; no live change from this trial.
Reported: gross vs net; same-close execution (the paper's timing); turnover and holding days; per-year.

Usage: python3 statarb_al.py      (a few minutes; writes results/statarb_al/)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pairs_broad import load, sharpe, nw_t, conc  # noqa: E402

BD = os.path.join(HERE, 'backtest_data')
OUT = os.path.join(HERE, 'results', 'statarb_al')
ETFS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
WIN, CORR_WIN = 60, 252
S_OPEN, S_CLOSE_SHORT, S_CLOSE_LONG = 1.25, 0.75, -0.50
KAPPA_MIN = 252 / 30
COST_STOCK, COST_ETF, BORROW = 0.0005, 0.0001, 0.005
MAIN_START, CONTEXT = '2020-01-01', ('2005-01-01', '2019-12-31')
GATE = {'sharpe': 0.50, 't': 2.24, 'conc': 0.75, 'corr': 0.30}


def scores(px, member, etf_px):
    """Daily s-scores and betas for every stock (NaN where not eligible)."""
    days = px.index
    R = px.pct_change(fill_method=None).to_numpy()
    E = etf_px.reindex(days).pct_change(fill_method=None).to_numpy()
    P = px.to_numpy()
    nd, ns = R.shape
    S = np.full((nd, ns), np.nan)
    B = np.full((nd, ns), np.nan)
    match = np.zeros(ns, int)
    etf_of = np.zeros((nd, ns), int)
    month_end = np.r_[days.month[1:] != days.month[:-1], True]
    start = max(WIN, CORR_WIN) + 1
    for d in range(start, nd):
        if month_end[d - 1] or d == start:                     # re-choose ETFs at each month end
            r, e = R[d - CORR_WIN:d], E[d - CORR_WIN:d]
            ok_e = ~np.isnan(e).any(0)
            c = np.full((ns, len(ETFS)), -np.inf)
            for k in np.where(ok_e)[0]:
                rr = np.where(np.isnan(r), 0.0, r)
                ek = e[:, k] - e[:, k].mean()
                rc = rr - rr.mean(0)
                c[:, k] = (rc * ek[:, None]).sum(0) / np.sqrt((rc ** 2).sum(0) * (ek ** 2).sum() + 1e-18)
            match = c.argmax(1)
        etf_of[d] = match
        r = R[d - WIN + 1:d + 1]                               # window ends at today's close
        e = E[d - WIN + 1:d + 1][:, match]
        elig = member[d] & (P[d] >= 5) & ~np.isnan(r).any(0) & ~np.isnan(e).any(0)
        if elig.sum() < 30:
            continue
        r, e = r[:, elig], e[:, elig]
        em, rm = e.mean(0), r.mean(0)
        beta = ((e - em) * (r - rm)).sum(0) / ((e - em) ** 2).sum(0)
        eps = r - rm - beta * (e - em)
        X = np.cumsum(eps, 0)
        x0, x1 = X[:-1], X[1:]
        x0m, x1m = x0.mean(0), x1.mean(0)
        b = ((x0 - x0m) * (x1 - x1m)).sum(0) / ((x0 - x0m) ** 2).sum(0)
        a = x1m - b * x0m
        zeta = x1 - a - b * x0
        good = (b > 0) & (b < 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            kappa = -np.log(np.where(good, b, np.nan)) * 252
            m = a / (1 - b)
            seq = np.sqrt(zeta.var(0) / (1 - b ** 2))
        good &= kappa > KAPPA_MIN
        if good.sum() < 10:
            continue
        mbar = np.nanmean(np.where(good, m, np.nan))
        s = np.where(good, (-m + mbar) / seq, np.nan)
        idx = np.where(elig)[0]
        S[d, idx] = s
        B[d, idx] = beta
    return S, B, R, E, etf_of


def simulate(S, B, R, E, etf_of, delay):
    """delay=1: signal at close d, trade at close d+1; delay=0: trade at the signal close (paper).
    At each close: earn the day's P&L on what was held, execute any order decided at the previous
    close, then decide from today's scores (executed now if delay=0, at the next close if delay=1)."""
    nd, ns = S.shape
    pos = np.zeros(ns)
    beta_held = np.zeros(ns)
    gross, net, n_open = np.zeros(nd), np.zeros(nd), np.zeros(nd)
    trades = 0
    pending = None

    def execute(target, d, k_held):
        nonlocal trades
        change = target != pos
        if not change.any():
            return 0.0
        cost = np.abs(target[change] - pos[change]) * (COST_STOCK + COST_ETF)
        k = max(k_held, int((target != 0).sum()), 1)
        trades += int((target[change] != 0).sum())
        pos[change] = target[change]
        nb = B[d, change]
        beta_held[change] = np.where(np.isnan(nb), beta_held[change], nb)
        return cost.sum() / k

    def decide(s):
        tgt = pos.copy()
        flat = pos == 0
        tgt[flat & (s > S_OPEN)] = -1
        tgt[flat & (s < -S_OPEN)] = 1
        tgt[(pos < 0) & ((s < S_CLOSE_SHORT) | np.isnan(s))] = 0
        tgt[(pos > 0) & ((s > S_CLOSE_LONG) | np.isnan(s))] = 0
        return tgt

    for d in range(1, nd):
        held = pos != 0
        if held.any():
            rs = R[d, held]
            re = E[d, etf_of[d - 1, held]]
            pr = pos[held] * (np.nan_to_num(rs) - beta_held[held] * np.nan_to_num(re))
            c = np.where(pos[held] < 0, BORROW / 252, 0.0)
            gross[d] = pr.mean()
            net[d] = (pr - c).mean()
            n_open[d] = held.sum()
        if delay == 1:
            if pending is not None:
                net[d] -= execute(pending, d, int(held.sum()))
            pending = decide(S[d])
        else:
            net[d] -= execute(decide(S[d]), d, int(held.sum()))
        upd = (pos != 0) & ~np.isnan(B[d])
        beta_held[upd] = B[d, upd]
    return gross, net, n_open, trades


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def emit(s=''):
        print(s, flush=True)
        lines.append(str(s))

    px, member = load()
    etf = pd.read_csv(os.path.join(BD, 'sector_spdr_adjclose.csv'), index_col=0, parse_dates=True)
    spy = etf['SPY'].pct_change(fill_method=None)
    S, B, R, E, etf_of = scores(px, member, etf[ETFS])
    days = px.index
    res = {}
    for name, delay in (('next close (decision)', 1), ('same close (paper timing)', 0)):
        g, ne, no, tr = simulate(S, B, R, E, etf_of, delay)
        res[name] = (pd.Series(g, index=days), pd.Series(ne, index=days), pd.Series(no, index=days), tr)
        pd.DataFrame({'gross': g, 'net': ne, 'open': no}, index=days).to_csv(
            os.path.join(OUT, f"daily_{'next' if delay else 'same'}.csv"))
    emit("Residual stat arb (Avellaneda & Lee 2010, ETF version), S&P 500, net of 6 bp/side + borrow\n")
    emit(f"{'execution':28s} {'period':8s} {'gross SR':>9s} {'net SR':>7s} {'net %/yr':>9s} {'NW t':>6s} {'corr SPY':>9s} {'avg open':>9s}")
    for name, (g, ne, no, tr) in res.items():
        for (lo, hi), per in (((MAIN_START, '2100-01-01'), '2020+'), (CONTEXT, '2005-19')):
            x, y = ne[lo:hi], g[lo:hi]
            act = no[lo:hi] > 0
            x, y = x[x.index >= x[act].index.min()], y[y.index >= x[act].index.min()]
            c = x.corr(spy.reindex(x.index))
            emit(f"{name:28s} {per:8s} {sharpe(y):+9.2f} {sharpe(x):+7.2f} {x.mean() * 252 * 100:+9.2f} "
                 f"{nw_t(x):+6.2f} {c:+9.2f} {no[lo:hi][act].mean():9.0f}")
    g, ne, no, tr = res['next close (decision)']
    yr = ne.groupby(ne.index.year).sum() * 100
    emit("\nNext-close net return by year %: " + ", ".join(f"{y} {v:+.1f}" for y, v in yr.items() if y >= 2005))
    yrs = (days[-1] - days[days.searchsorted(pd.Timestamp('2005-01-01'))]).days / 365.25
    emit(f"Trades opened 2005+: {tr} (~{tr / yrs:.0f} a year)")

    x = ne[MAIN_START:]
    last24 = x[x.index > x.index[-1] - pd.DateOffset(months=24)].sum()
    c = x.corr(spy.reindex(x.index))
    checks = [('P1 net Sharpe >= 0.50', sharpe(x) >= GATE['sharpe'], f"{sharpe(x):.2f}"),
              ('P2 NW t >= 2.24', nw_t(x) >= GATE['t'], f"{nw_t(x):.2f}"),
              ('P3 last 24 months > 0', last24 > 0, f"{last24 * 100:+.2f}%"),
              ('P4 best 24m <= 75% of profit', conc(x) <= GATE['conc'], f"{conc(x):.0%}" if np.isfinite(conc(x)) else 'no profit'),
              ('P5 |corr with SPY| <= 0.30', abs(c) <= GATE['corr'], f"{c:+.2f}")]
    emit(f"\nGate (T-2026-10-07-97), next-close execution, {x.index[0]:%Y-%m-%d} .. {x.index[-1]:%Y-%m-%d}:")
    for name, ok, val in checks:
        emit(f"  {name:32s} {val:14s} {'ok' if ok else 'FAIL'}")
    emit(f"\nT-97 VERDICT: {'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'}")
    open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
