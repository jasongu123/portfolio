"""
Rates-bucket attribution for the trend sleeve — RESEARCH ONLY (2026-09-11).

Question: what do the Treasury ETFs (SHY / IEF / TLT) add to the cross-asset trend sleeve?

Two views, both on the live 16-ETF universe:

  1. IN-RUN DECOMPOSITION (the actual attribution). One backtest of the full sleeve; each day's
     return is split into per-asset contributions (target weight x asset return) and summed by
     sector. The sector pieces add up exactly to the sleeve's gross return, so "what Rates added"
     is read off the run the sleeve actually made, not from a different portfolio.

  2. RE-RUN COUNTERFACTUALS (the A/B/C design). The sleeve rebuilt three ways:
        A  full universe
        B  without the Rates sector (13 assets: 1/n and the vol scalar re-set themselves)
        C  the Rates sector alone
     B is "the sleeve as it would have been built without Treasuries", not A minus the Rates
     contribution; the difference between the two views is the re-weighting.

Conventions: research baseline V1_trail from trend_follow_v3 (corrected lookback and rebalance
stamps, trailing 504-day vol scalar capped at 3.0, 2 bp cost model, drawdown mark at initial
capital). Reference rows with trend_follow_v2 legacy conventions are printed for A/B/C so the
numbers tie to the v2 stats. Headline Sharpe is rf (flat 5%/yr); no-rf is shown alongside and
labelled. Windows: full from 2008-01-01 (all signals live), eval 2020-2025 (the validated book
window). SHY prices start 2017-08 in the cache, so the Rates bucket is IEF + TLT until 2018-08.

Run:  /usr/local/bin/python3 rates_attribution.py
Writes results/rates_attribution_contrib.csv, results/rates_attribution_runs.csv.
Does not touch trend_sleeve_live.py, trend_follow_v2.py or any headline series.
"""

import os
import numpy as np
import pandas as pd

import trend_follow_v2 as t2
import trend_follow_v3 as v3

RESULTS_DIR = v3.RESULTS_DIR
FULL_START, EVAL_START, EVAL_END = v3.FULL_START, v3.EVAL_START, v3.EVAL_END
ANN = v3.ANN
CRISIS_YEARS = (2008, 2020, 2022)


# --------------------------------------------------------------------------- view 1: decomposition
def daily_scalar(normalized, px, rets, month_ends):
    """Trailing scalar exactly as trend_follow_v3.scale_and_cost builds it (scalar_mode='trailing')."""
    gross = (normalized * rets).sum(axis=1)
    turnover = normalized.diff().abs().sum(axis=1)
    legacy = gross - turnover * (t2.COST_BPS_PER_REBAL / 10_000.0)
    active = normalized.abs().sum(axis=1).gt(0)
    first_active = active.idxmax() if active.any() else px.index[0]
    vals = {}
    for d in month_ends:
        h = legacy.loc[first_active:d].tail(v3.SCALAR_LOOKBACK)
        if len(h) >= v3.SCALAR_MIN_HISTORY and h.std() > 0:
            vals[d] = min(t2.PORT_TARGET_VOL / (h.std() * ANN), t2.MAX_LEVERAGE)
        else:
            vals[d] = 1.0
    return pd.Series(vals).reindex(px.index, method='ffill').shift(1).fillna(1.0)


def decomposition(prices, universe):
    normalized, px, rets, _, month_ends = v3.build_weights(prices, universe, 'V2', t2.HORIZONS)
    scal = daily_scalar(normalized, px, rets, month_ends)
    target = normalized.mul(scal, axis=0)
    contrib = (target * rets).fillna(0.0)                       # per-asset daily contribution
    port = v3.scale_and_cost(normalized, px, rets, scalar_mode='trailing', month_ends=month_ends)
    gross_scaled = contrib.sum(axis=1)
    j = pd.concat([gross_scaled.rename('g'), port.rename('p')], axis=1).dropna()
    cost_drag = j['g'] - j['p']
    return contrib.loc[j.index], target.loc[j.index], port.loc[j.index], cost_drag, scal.loc[j.index]


def sector_table(contrib, target, port, sectors, start, end=None):
    c = contrib.loc[start:end]
    t = target.loc[start:end]
    p = port.loc[start:end]
    gross = c.sum(axis=1)
    rows = []
    var_p = gross.var()
    for sec in sorted(set(sectors.values())):
        cols = [a for a in c.columns if sectors[a] == sec]
        cs = c[cols].sum(axis=1)
        rows.append({
            'sector': sec, 'assets': len(cols),
            'ann_contrib': cs.mean() * 252,
            'share_of_gross': cs.mean() / gross.mean() if gross.mean() != 0 else np.nan,
            'risk_contrib': cs.cov(gross) / var_p if var_p > 0 else np.nan,
            'avg_gross_exposure': t[cols].abs().sum(axis=1).mean(),
            'exposure_share': t[cols].abs().sum(axis=1).mean() / t.abs().sum(axis=1).mean(),
            'avg_net_exposure': t[cols].sum(axis=1).mean(),
            'corr_with_rest': cs.corr(gross - cs),
            'hit_rate_months': (cs.resample('ME').sum() > 0).mean(),
        })
    df = pd.DataFrame(rows).set_index('sector')
    df.loc['TOTAL (gross of costs)'] = {
        'assets': len(c.columns), 'ann_contrib': gross.mean() * 252, 'share_of_gross': 1.0,
        'risk_contrib': 1.0, 'avg_gross_exposure': t.abs().sum(axis=1).mean(), 'exposure_share': 1.0,
        'avg_net_exposure': t.sum(axis=1).mean(), 'corr_with_rest': np.nan, 'hit_rate_months': np.nan,
    }
    return df


def yearly_sector(contrib, sectors):
    by = {}
    for sec in sorted(set(sectors.values())):
        cols = [a for a in contrib.columns if sectors[a] == sec]
        by[sec] = contrib[cols].sum(axis=1).groupby(contrib.index.year).sum()
    by['TOTAL'] = contrib.sum(axis=1).groupby(contrib.index.year).sum()
    return pd.DataFrame(by)


# --------------------------------------------------------------------------- view 2: re-runs
def run_stats(r, spy, eq, label):
    full = v3.window(r, FULL_START)
    ev = v3.window(r, EVAL_START, EVAL_END)
    yr = v3.yearly(full)
    book = v3.rp_book(eq, ev) if eq is not None else None
    cy = full.groupby(full.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {
        'run': label,
        'full_rf_sharpe': v3.sharpe(full), 'full_norf_sharpe': v3.sharpe(full, 0.0),
        'full_cagr': (1 + full).prod() ** (252 / len(full)) - 1,
        'full_vol': full.std() * ANN, 'full_maxdd': v3.maxdd(full),
        'worst_year_rf_sharpe': min(yr.values()), 'spy_corr': v3.corr_with(full, spy),
        'eval_rf_sharpe': v3.sharpe(ev), 'eval_maxdd': v3.maxdd(ev),
        'y2008': cy.get(2008, np.nan), 'y2020': cy.get(2020, np.nan), 'y2022': cy.get(2022, np.nan),
        'scalar_mean': r.attrs.get('scalar_mean', r.attrs.get('leverage_scalar', np.nan)),
        'book_rf_sharpe': book['rf_sharpe'] if book else np.nan,
        'book_worst_year': book['worst_year'] if book else np.nan,
        'book_maxdd': book['maxdd'] if book else np.nan,
        'book_trend_weight': 1 - book['we'] if book else np.nan,
    }


def print_runs(rows):
    hdr = (f"{'run':32s}{'rf SR':>7s}{'no-rf':>7s}{'CAGR':>8s}{'vol':>7s}{'MDD':>8s}{'worst':>7s}{'SPYc':>7s}"
           f"{'| 2008':>8s}{'2020':>7s}{'2022':>7s}{'| eval rf':>10s}{'MDD':>8s}{'| book rf':>10s}{'worst':>7s}{'MDD':>8s}")
    print(hdr)
    for r in rows:
        print(f"{r['run']:32s}{r['full_rf_sharpe']:7.2f}{r['full_norf_sharpe']:7.2f}{r['full_cagr']:8.1%}"
              f"{r['full_vol']:7.1%}{r['full_maxdd']:8.1%}{r['worst_year_rf_sharpe']:7.2f}{r['spy_corr']:7.2f}"
              f"{r['y2008']:8.0%}{r['y2020']:7.0%}{r['y2022']:7.0%}"
              f"{r['eval_rf_sharpe']:10.2f}{r['eval_maxdd']:8.1%}"
              f"{r['book_rf_sharpe']:10.2f}{r['book_worst_year']:7.2f}{r['book_maxdd']:8.1%}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prices = t2.load_prices()
    live16 = [c for c in v3.LIVE16 if c in prices.columns]
    sectors = {c: t2.SECTOR[c] for c in live16}
    rates = [c for c in live16 if sectors[c] == 'Rates']
    no_rates = [c for c in live16 if sectors[c] != 'Rates']
    spy = prices['SPY'].pct_change(fill_method=None)
    eq = v3.load_equity()
    print(f"Live universe {len(live16)}: {', '.join(live16)}")
    print(f"Rates bucket: {rates}   first price: " +
          ", ".join(f"{c} {prices[c].first_valid_index().date()}" for c in rates))

    # ---------------- view 1
    contrib, target, port, cost_drag, scal = decomposition(prices, live16)
    print(f"\nCheck: sum of asset contributions - sleeve return = cost drag, "
          f"{cost_drag.mean() * 252:+.2%}/yr (max daily {cost_drag.abs().max():.2e}); "
          f"sleeve gross {contrib.sum(axis=1).loc[FULL_START:].mean() * 252:+.2%}/yr from {FULL_START}.")

    out = []
    for wname, s, e in (('full 2008-', FULL_START, None), ('eval 2020-2025', EVAL_START, EVAL_END)):
        tab = sector_table(contrib, target, port, sectors, s, e)
        tab.insert(0, 'window', wname)
        out.append(tab)
        print(f"\n=== VIEW 1: in-run sector decomposition, {wname} (gross of costs; pieces sum to TOTAL) ===")
        with pd.option_context('display.float_format', lambda x: f'{x:8.3f}', 'display.width', 160):
            show = tab.drop(columns='window').copy()
            for col in ('ann_contrib', 'share_of_gross', 'risk_contrib', 'exposure_share', 'hit_rate_months'):
                show[col] = show[col].map(lambda x: f'{x:+.1%}' if pd.notna(x) else '')
            print(show.to_string())
    # per-asset inside Rates
    print("\n--- Rates bucket by asset (full window; SHY only from 2018-08) ---")
    gross_full = contrib.loc[FULL_START:].sum(axis=1)
    for c in rates:
        cs = contrib[c].loc[FULL_START:]
        live = target[c].loc[FULL_START:].abs().gt(0)
        print(f"  {c}: ann contrib {cs.mean() * 252:+.2%} | risk contrib {cs.cov(gross_full) / gross_full.var():+.1%} "
              f"| avg |w| {target[c].loc[FULL_START:].abs().mean():.3f} | days active {int(live.sum())} "
              f"| avg net w {target[c].loc[FULL_START:].mean():+.3f}")
    ys = yearly_sector(contrib.loc[FULL_START:], sectors)
    print("\n--- Yearly return contribution by sector (simple sum of daily contributions; * crisis) ---")
    with pd.option_context('display.float_format', lambda x: f'{x:+.1%}', 'display.width', 160):
        ys.index = [f"{y}{'*' if y in CRISIS_YEARS else ''}" for y in ys.index]
        print(ys.to_string())
    rc = contrib[rates].sum(axis=1).loc[FULL_START:]
    rest = gross_full - rc
    print(f"\ncorr(Rates contribution, rest of sleeve) {rc.corr(rest):+.2f} | corr(Rates contribution, SPY) "
          f"{rc.corr(spy.reindex(rc.index)):+.2f} | corr(rest, SPY) {rest.corr(spy.reindex(rest.index)):+.2f}")
    pd.concat(out).to_csv(os.path.join(RESULTS_DIR, 'rates_attribution_contrib.csv'))
    ys.to_csv(os.path.join(RESULTS_DIR, 'rates_attribution_yearly.csv'))

    # ---------------- view 2
    runs = {'A  full sleeve': live16, 'B  without Rates': no_rates, 'C  Rates alone': rates}
    rows, series = [], {}
    for label, uni in runs.items():
        r = v3.run_variant(prices, uni, 'V2', t2.HORIZONS, scalar_mode='trailing')
        series[label] = r
        rows.append(run_stats(r, spy, eq, label + ' [V1_trail]'))
    for label, uni in runs.items():
        r = t2.backtest(prices, uni, t2.HORIZONS)
        rows.append(run_stats(r, spy, eq, label + ' [v2 legacy ref]'))
    print(f"\n=== VIEW 2: re-run counterfactuals. Full {FULL_START[:4]}-, eval {EVAL_START[:4]}-{EVAL_END[:4]}; "
          f"rf-Sharpe headline, no-rf labelled; MDD mark at initial capital ===")
    print_runs(rows)
    a, b, c = (v3.window(series[k], FULL_START) for k in runs)
    j = pd.concat([a.rename('a'), b.rename('b'), c.rename('c')], axis=1).dropna()
    print(f"\nA vs B on identical dates ({len(j)} days): rf-Sharpe {v3.sharpe(j['a']):.2f} vs {v3.sharpe(j['b']):.2f} "
          f"({v3.sharpe(j['a']) - v3.sharpe(j['b']):+.2f}); MDD {v3.maxdd(j['a']):.1%} vs {v3.maxdd(j['b']):.1%}; "
          f"corr(B, C) {j['b'].corr(j['c']):+.2f}; corr(A, B) {j['a'].corr(j['b']):+.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'rates_attribution_runs.csv'), index=False)
    print("\nWrote results/rates_attribution_contrib.csv, _yearly.csv, _runs.csv")


if __name__ == '__main__':
    main()
