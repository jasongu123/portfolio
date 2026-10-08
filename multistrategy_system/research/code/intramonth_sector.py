"""
Sectoral intramonth momentum cycle — RESEARCH ONLY. Pre-registered 2026-09-07 before any run.

Source of rules: Quantpedia, "Sectoral Intramonth Momentum Cycle" (Aug 2026), implementing
Nathan, Suominen & Tasa, "The Intramonth Momentum Cycle" (SSRN 6426026, 2026). Mechanism claimed:
institutions raise month-end cash by selling their most dispensable (recent-loser) holdings,
ending about four trading days before month-end; month-start inflows then reverse it.

Universe: nine Select Sector SPDRs (XLB XLE XLF XLI XLK XLP XLU XLV XLY) plus SPY. Daily adjusted
closes in backtest_data/sector_spdr_adjclose.csv (Dec-1998 onward).

Rules (D0 = last trading day of the month; D-k = k trading days before D0; D+k = k-th trading day
of the next month; ranking = trailing 252-trading-day total return, top 3 = winners, bottom 3 =
losers, equal-weighted 1/3 each):
  Leg 1  enter at the close of D-10 (ranking at D-10), exit at the close of D-5:
         Strat2: long winners, short losers.  Strat1: long winners, short SPY equal notional.
  Leg 2  enter at the close of D0 (ranking at D0), hold through the close of D+1.
  Leg 3  flip the leg-2 portfolio at the close of D+1, hold through the close of D+3.
  Flat otherwise. Self-financing, no leverage beyond the implicit short.
Cost: COST_BPS per unit of absolute weight change per ETF (reported at 0, 1 and 2 bp; the gate
uses 1 bp). Months with fewer than 11 trading days are skipped.

Reported by the source (Dec-1998..Jun-2026, cost not stated): Strat2 5.99%/yr, vol 10.87%,
Sharpe 0.55, MaxDD -21.7%; Strat1-SPY 3.77%/yr, vol 6.99%, Sharpe 0.54, MaxDD -14.7%.
There is NO untouched holdout: the source's sample runs to mid-2026.

Pre-registered gate (Strat2 is the gated trial; Strat1 is reported as a second trial)
  G1 replication at 1 bp: net Sharpe (no-rf, daily*sqrt252) >= 0.40 over the full sample,
     and every one of the three legs has a positive net contribution over the full sample.  binding
  G2 stability: rolling 5-year net Sharpe positive in >= 70% of windows.                    binding
  G3 recent sub-period 2017-01..latest (after the month-end-flow literature's samples): net
     Sharpe >= 0.30.                                                                        binding
  G4 book 2020-2025, monthly, inverse-vol: correlation with equity sleeve and trend V1_trail each
     < 0.30, AND three-sleeve no-rf Sharpe >= two-sleeve AND MaxDD not deeper.              binding
  Informational: results at 0 and 2 bp; per-leg Sharpe; per-year Sharpe; Strat1 numbers.
Trial registry: 2 trials (Strat2, Strat1). Leg decompositions are diagnostics, not trials.

Outputs: results/intramonth_sector_daily.csv, console report.
Usage:  python3 intramonth_sector.py
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PRICES = os.path.join(SCRIPT_DIR, 'backtest_data', 'sector_spdr_adjclose.csv')
TREND_V3 = os.path.join(RESULTS_DIR, 'trend_v3_returns.csv')
FF_CSV = os.path.join(SCRIPT_DIR, 'backtest_data', 'F-F_Research_Data_Factors.csv')

SECTORS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
LOOKBACK = 252
N_TOP = 3
COST_BPS = 1.0
RECENT_START = '2017-01-01'
EVAL = ('2020-01', '2025-12')
RF_FLAT_M = 0.05 / 12
ANN = np.sqrt(252)


# --------------------------------------------------------------------------- data
def load_prices():
    px = pd.read_csv(PRICES, index_col=0, parse_dates=True).sort_index()
    px = px[SECTORS + ['SPY']].dropna(how='all')
    return px


def load_ff_rf():
    rows = []
    with open(FF_CSV) as f:
        for line in f:
            p = [q.strip() for q in line.split(',')]
            if len(p) == 5 and len(p[0]) == 6 and p[0].isdigit():
                rows.append((p[0], float(p[4]) / 100))
    s = pd.Series(dict(rows))
    s.index = pd.PeriodIndex(pd.to_datetime(s.index, format='%Y%m'), freq='M')
    return s


# --------------------------------------------------------------------------- strategy
def build_weights(px, variant='Strat2', legs=(1, 2, 3)):
    """Daily target weights, indexed by the day whose close they are set at; they earn the next
    day's return. Returns (weights DataFrame, leg label Series)."""
    rets = px.pct_change(fill_method=None)
    mom = px / px.shift(LOOKBACK) - 1.0
    cols = SECTORS + ['SPY']
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    leg_of = pd.Series('', index=px.index)
    months = px.groupby(px.index.to_period('M')).apply(lambda g: list(g.index))
    keys = list(months.index)

    def mom_position(day):
        m = mom.loc[day, SECTORS].dropna()
        if len(m) < len(SECTORS):
            return None
        ranked = m.sort_values()
        losers, winners = list(ranked.index[:N_TOP]), list(ranked.index[-N_TOP:])
        pos = pd.Series(0.0, index=cols)
        pos[winners] = 1.0 / N_TOP
        if variant == 'Strat2':
            pos[losers] = -1.0 / N_TOP
        else:
            pos['SPY'] = -1.0
        return pos

    for i, k in enumerate(keys):
        days = months[k]
        if len(days) < 11:
            continue
        d0 = days[-1]
        # Leg 1: set at D-10 close, held through D-5 close -> weights on D-10 .. D-6
        if 1 in legs:
            pos = mom_position(days[-11])
            if pos is not None:
                for d in days[-11:-6]:
                    w.loc[d] = pos.values
                    leg_of.loc[d] = 'L1'
        # Legs 2 and 3 need the next month's first three days
        if i + 1 < len(keys):
            nxt = months[keys[i + 1]]
            if len(nxt) >= 3:
                pos = mom_position(d0)
                if pos is not None:
                    if 2 in legs:                     # set at D0 close, earns D+1
                        w.loc[d0] = pos.values
                        leg_of.loc[d0] = 'L2'
                    if 3 in legs:                     # set at D+1 close, earns D+2 and D+3
                        for d in nxt[:2]:
                            w.loc[d] = -pos.values
                            leg_of.loc[d] = 'L3'
    return w, leg_of, rets


def run(px, variant='Strat2', legs=(1, 2, 3), cost_bps=COST_BPS):
    w, leg_of, rets = build_weights(px, variant, legs)
    gross = (w.shift(1) * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).shift(0)          # trades at the close of the day the weight changes
    cost = turnover * cost_bps / 1e4
    net = (gross - cost).dropna()
    net = net.loc[px.index[LOOKBACK + 1]:]                    # after the first full ranking
    return net.rename('net'), gross.reindex(net.index), leg_of.shift(1).reindex(net.index), w


# --------------------------------------------------------------------------- statistics
def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * ANN) if len(r) > 20 and r.std() > 0 else np.nan


def maxdd(r):
    r = r.dropna()
    eq = (1 + r).cumprod().values
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((eq / peak - 1).min()) if len(eq) else 0.0


def ann_ret(r):
    r = r.dropna()
    return float((1 + r).prod() ** (252 / len(r)) - 1) if len(r) else np.nan


def monthly(r):
    m = (1 + r).resample('ME').prod() - 1
    m.index = m.index.to_period('M')
    return m


def rolling_positive(r, years=5):
    n = years * 252
    rs = r.rolling(n).apply(lambda x: x.mean() / x.std() if x.std() > 0 else np.nan, raw=True).dropna()
    return float((rs > 0).mean()), len(rs)


def sr_m(r, rf=0.0):
    r = r.dropna()
    return float((r - rf).mean() / r.std() * np.sqrt(12)) if len(r) > 6 and r.std() > 0 else np.nan


# --------------------------------------------------------------------------- main
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    px = load_prices()
    print(f"Prices {px.index.min().date()}..{px.index.max().date()} ({len(px)} days); "
          f"first ranking {px.index[LOOKBACK].date()}")

    results = {}
    for variant in ('Strat2', 'Strat1'):
        print("\n" + "=" * 96 + f"\n{variant}: {'long top-3 / short bottom-3' if variant == 'Strat2' else 'long top-3 / short SPY'}")
        for bps in (0.0, 1.0, 2.0):
            net, gross, leg, w = run(px, variant, cost_bps=bps)
            print(f"  cost {bps:.0f} bp: ann ret {ann_ret(net):+.2%} | vol {net.std()*ANN:.2%} | Sharpe {sharpe(net):.2f} | "
                  f"MaxDD {maxdd(net):.1%} | days invested {(w.shift(1).abs().sum(axis=1).reindex(net.index) > 0).mean():.0%}")
            if bps == COST_BPS:
                results[variant] = (net, gross, leg, w)
        net, gross, leg, w = results[variant]
        print("  per-leg net Sharpe and annual contribution (1 bp), each leg run alone:")
        leg_ok = True
        for L in (1, 2, 3):
            n_l, _, _, _ = run(px, variant, legs=(L,), cost_bps=COST_BPS)
            contrib = ann_ret(n_l)
            leg_ok &= contrib > 0
            print(f"    leg {L}: Sharpe {sharpe(n_l):.2f}, ann contribution {contrib:+.2%}")
        yr = {int(y): sharpe(x) for y, x in net.groupby(net.index.year)}
        print("  per-year Sharpe: " + "  ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
        frac, nwin = rolling_positive(net)
        recent = net.loc[RECENT_START:]
        print(f"  rolling 5y Sharpe positive in {frac:.0%} of {nwin} windows | {RECENT_START[:4]}+ Sharpe {sharpe(recent):.2f} "
              f"({len(recent)} days) | MaxDD {maxdd(recent):.1%}")
        results[variant] += (leg_ok, frac, sharpe(recent))

    # book test on Strat2
    net = results['Strat2'][0]
    parts = [pd.read_csv(os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'), index_col=0, parse_dates=True).iloc[:, 0]
             for y in range(2020, 2026) if os.path.exists(os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'))]
    g4 = False
    corr_eq = corr_tr = np.nan
    if parts and os.path.exists(TREND_V3):
        eq = monthly(pd.concat(parts).sort_index()).rename('equity')
        tr = monthly(pd.read_csv(TREND_V3, index_col=0, parse_dates=True)['V1_trail'].dropna()).rename('trend')
        im = monthly(net).rename('intramonth')
        jb = pd.concat([eq, tr, im], axis=1).loc[pd.Period(EVAL[0], 'M'):pd.Period(EVAL[1], 'M')].dropna()
        c = jb.corr()
        corr_eq, corr_tr = float(c.loc['intramonth', 'equity']), float(c.loc['intramonth', 'trend'])

        def book(cols):
            inv = 1 / jb[cols].std(); wts = inv / inv.sum(); b = (jb[cols] * wts).sum(axis=1)
            yrs = {y: sr_m(x, RF_FLAT_M) for y, x in b.groupby(b.index.year)}
            return {'norf': sr_m(b), 'rf': sr_m(b, RF_FLAT_M), 'mdd': maxdd(b), 'vol': b.std() * np.sqrt(12), 'worst': min(yrs.values()), 'w': wts.round(2).to_dict()}
        b2, b3 = book(['equity', 'trend']), book(['equity', 'trend', 'intramonth'])
        print("\n" + "=" * 96 + f"\nBook 2020-2025 ({len(jb)} months): corr intramonth~equity {corr_eq:+.2f}, ~trend {corr_tr:+.2f}")
        for n_, bb in (('equity+trend', b2), ('equity+trend+intramonth', b3)):
            print(f"  {n_:26s} no-rf {bb['norf']:.2f} | rf {bb['rf']:.2f} | MDD {bb['mdd']:.1%} | vol {bb['vol']:.1%} | worst {bb['worst']:+.2f} | w {bb['w']}")
        g4 = (corr_eq < 0.30) and (corr_tr < 0.30) and (b3['norf'] >= b2['norf']) and (b3['mdd'] >= b2['mdd'])

    net, gross, leg, w, leg_ok, frac, sr_recent = results['Strat2']
    g1 = (sharpe(net) >= 0.40) and leg_ok
    g2 = frac >= 0.70
    g3 = sr_recent >= 0.30
    print("\nGO/NO-GO (pre-registered, Strat2 at 1 bp):")
    print(f"  {'PASS' if g1 else 'fail'}  G1 full-sample net Sharpe >= 0.40 ({sharpe(net):.2f}) and all legs positive ({leg_ok})")
    print(f"  {'PASS' if g2 else 'fail'}  G2 rolling 5y Sharpe > 0 in >= 70% ({frac:.0%})")
    print(f"  {'PASS' if g3 else 'fail'}  G3 2017+ net Sharpe >= 0.30 ({sr_recent:.2f})")
    print(f"  {'PASS' if g4 else 'fail'}  G4 book: corr < 0.30 ({corr_eq:+.2f}, {corr_tr:+.2f}), Sharpe and MDD not worse")
    print(f"VERDICT: {'PROMOTE to paper-trade candidate' if all([g1, g2, g3, g4]) else 'not promoted'}")

    out = pd.concat([results['Strat2'][0].rename('strat2_net'), results['Strat1'][0].rename('strat1_net'),
                     results['Strat2'][2].rename('leg')], axis=1)
    out.to_csv(os.path.join(RESULTS_DIR, 'intramonth_sector_daily.csv'))
    print("Saved results/intramonth_sector_daily.csv")
