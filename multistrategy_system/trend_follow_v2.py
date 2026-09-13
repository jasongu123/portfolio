"""
Trend sleeve v2 — BREADTH + MULTI-HORIZON. Deepen the one validated diversifier.

Rationale (from the carry result): the next 'classic factor' (carry/value/defensive)
is short-vol and won't diversify a long-risk book. The high-conviction move is to
strengthen TREND — the long-vol / crisis-alpha engine that DOES diversify — along its
two real levers:
  1. BREADTH  — up to 18 assets across 5 sectors (equity / rates / commodity / FX /
     real estate). Trend's Sharpe comes from many UNCORRELATED trends, not one bet.
  2. HORIZON  — a 3/6/12-month signal ensemble instead of a single 12-month lookback;
     more robust, fewer whipsaws from one arbitrary window.

ABLATION (isolate what helps — institutional rigor):
  A   9 assets,  12m only          (= v1 baseline, trend_follow.py)
  B   N assets,  12m only          (breadth alone)
  C   N assets,  3/6/12m ensemble  (breadth + horizon)
Reports each: Sharpe (no-rf & rf), per-year with CRISIS years (2008/2020/2022) starred,
SPY correlation. The keeper must hold its crisis alpha while lifting risk-adjusted return.

Data: trend_cache.csv (existing 9, yfinance) + trend_extra.csv (9 breadth ETFs from
pull_trend_ibkr.py, IBKR ADJUSTED_LAST). Runs on whatever subset is present.

Usage:  python3.12 trend_follow_v2.py
"""

import os
import numpy as np
import pandas as pd

from trading_costs import (
    CostModelConfig,
    estimate_execution_cost,
    estimate_holding_cost,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, 'trend_cache.csv')
EXTRA = os.path.join(SCRIPT_DIR, 'trend_extra.csv')
RF_DAILY = 0.05 / 252

SECTOR = {
    'SPY': 'Equity', 'IWM': 'Equity', 'EFA': 'Equity', 'EEM': 'Equity', 'EWJ': 'Equity',
    'SHY': 'Rates', 'IEF': 'Rates', 'TLT': 'Rates',
    'GLD': 'Commodity', 'SLV': 'Commodity', 'DBC': 'Commodity', 'USO': 'Commodity', 'DBA': 'Commodity',
    'UUP': 'FX', 'FXE': 'FX', 'FXY': 'FX', 'FXB': 'FX',
    'VNQ': 'RealEstate',
}
BASE9 = ['SPY', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'UUP', 'VNQ']

HORIZONS = [63, 126, 252]      # ~3 / 6 / 12 months
VOL_LOOKBACK = 60
TARGET_VOL_PER_ASSET = 0.10
PORT_TARGET_VOL = 0.12
MAX_LEVERAGE = 3.0
COST_BPS_PER_REBAL = 2
COMMISSION_PER_SHARE = 0.0
MIN_COMMISSION = 0.0
ANNUAL_BORROW_RATE = 0.0
ANNUAL_FINANCING_RATE = 0.0
BACKTEST_CAPITAL = 100_000.0
CRISIS = {2008, 2020, 2022}


def load_prices():
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    if os.path.exists(EXTRA):
        ex = pd.read_csv(EXTRA, index_col=0, parse_dates=True)
        for c in ex.columns:
            if c not in df.columns:
                df = df.join(ex[c], how='outer')
    return df.sort_index()


def _apply_portfolio_costs(
    gross_returns,
    target_weights,
    prices,
    config,
    initial_capital,
):
    """Apply shared execution and holding costs to target portfolio weights.

    Weight changes are charged at the last observable close.  Holding costs use
    the current day's short and net-funded exposures and the actual calendar-day
    gap from the prior observation, so weekends are not silently free.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    target_weights = target_weights.reindex(gross_returns.index).fillna(0.0)
    prior_prices = prices.ffill().shift(1).reindex(gross_returns.index)
    weight_changes = target_weights.diff().fillna(0.0)
    equity = float(initial_capital)
    prior_date = None
    net_returns = []
    totals = {
        'notional': 0.0,
        'commission': 0.0,
        'spread': 0.0,
        'slippage': 0.0,
        'impact': 0.0,
        'borrow': 0.0,
        'financing': 0.0,
        'total': 0.0,
    }

    for date in gross_returns.index:
        execution_cost = 0.0
        for symbol, delta_weight in weight_changes.loc[date].items():
            if not np.isfinite(delta_weight) or abs(delta_weight) <= 1e-15:
                continue
            price = prior_prices.at[date, symbol]
            if not np.isfinite(price) or price <= 0:
                continue
            shares = abs(delta_weight) * equity / price
            estimate = estimate_execution_cost(shares, price, config=config)
            execution_cost += estimate.total
            for component in ('notional', 'commission', 'spread', 'slippage', 'impact'):
                totals[component] += getattr(estimate, component)

        weights_today = target_weights.loc[date]
        short_exposure = float((-weights_today.clip(upper=0.0)).sum())
        net_exposure = float(weights_today.sum())
        holding_days = 0 if prior_date is None else max((date - prior_date).days, 0)
        holding = estimate_holding_cost(
            short_notional=short_exposure * equity,
            # Short-sale proceeds fund offsetting longs in this cash-account
            # approximation.  Financing therefore applies only when net long
            # exposure exceeds equity; stock borrow is charged independently.
            borrowed_cash=max(net_exposure - 1.0, 0.0) * equity,
            holding_days=holding_days,
            config=config,
        )
        totals['borrow'] += holding.borrow
        totals['financing'] += holding.financing

        total_cost = execution_cost + holding.total
        gross_return = float(gross_returns.loc[date])
        net_return = gross_return - total_cost / equity
        net_returns.append(net_return)
        totals['total'] += total_cost
        equity *= 1.0 + net_return
        prior_date = date

    result = pd.Series(net_returns, index=gross_returns.index, name='trend_return')
    totals['total_bps'] = (
        totals['total'] / totals['notional'] * 10_000.0
        if totals['notional'] > 0 else 0.0
    )
    result.attrs['cost_components'] = totals
    return result


def backtest(
    prices,
    universe,
    horizons,
    long_only=False,
    *,
    execution_bps=COST_BPS_PER_REBAL,
    annual_borrow_rate=ANNUAL_BORROW_RATE,
    annual_financing_rate=ANNUAL_FINANCING_RATE,
    commission_per_share=COMMISSION_PER_SHARE,
    min_commission=MIN_COMMISSION,
    initial_capital=BACKTEST_CAPITAL,
):
    cols = [c for c in universe if c in prices.columns]
    px = prices[cols]
    rets = px.pct_change(fill_method=None)
    month_ends = px.resample('ME').last().index
    reb = pd.DataFrame(index=month_ends, columns=cols, dtype=float)
    maxlb = max(horizons)
    for d in month_ends:
        hist = px.loc[:d]
        if len(hist) < maxlb + 5:
            continue
        w = {}
        for c in cols:
            p = hist[c].dropna()
            if len(p) < maxlb + 5:
                w[c] = 0.0
                continue
            # multi-horizon ensemble: average the sign across lookbacks
            sigs = [(1.0 if p.iloc[-1] / p.iloc[-h] - 1.0 > 0 else (0.0 if long_only else -1.0))
                    for h in horizons]
            sig = float(np.mean(sigs))
            vol = rets[c].loc[:d].tail(VOL_LOOKBACK).std() * np.sqrt(252)
            size = min(TARGET_VOL_PER_ASSET / vol, MAX_LEVERAGE) if vol and vol > 0 else 0.0
            w[c] = sig * size
        reb.loc[d] = pd.Series(w)

    weights = reb.reindex(px.index, method='ffill').shift(1).fillna(0.0)
    n = weights.abs().gt(0).sum(axis=1).replace(0, 1)
    normalized_weights = weights.div(n, axis=0)
    gross = (normalized_weights * rets).sum(axis=1)
    turnover = normalized_weights.diff().abs().sum(axis=1)

    # Preserve the historical scalar so this cost integration does not silently
    # change reported signals or exposure. FIXME: replace this full-sample
    # volatility estimate with a lagged walk-forward scalar in a separate,
    # explicitly versioned strategy change.
    legacy_port = gross - turnover * (execution_bps / 10000.0)
    realized = legacy_port.std() * np.sqrt(252)
    scalar = min((PORT_TARGET_VOL / realized) if realized > 0 else 1.0, MAX_LEVERAGE)
    target_weights = normalized_weights * scalar
    gross_scaled = (target_weights * rets).sum(axis=1)

    config = CostModelConfig(
        commission_per_share=commission_per_share,
        min_commission=min_commission,
        fallback_execution_bps=execution_bps,
        extra_slippage_bps=0.0,
        annual_borrow_rate=annual_borrow_rate,
        annual_financing_rate=annual_financing_rate,
        day_count=360.0,
    )
    port = _apply_portfolio_costs(
        gross_scaled,
        target_weights,
        px,
        config,
        initial_capital,
    )
    cost_components = port.attrs['cost_components']
    port = port.loc[legacy_port.notna()].dropna()
    port.attrs['cost_components'] = cost_components
    port.attrs['leverage_scalar'] = scalar
    port.attrs['cost_config'] = config
    return port


def stats(r, label, spy=None):
    cost_components = r.attrs.get('cost_components', {})
    r = r.dropna()
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sh_norf = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    sh_rf = ((r - RF_DAILY).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    eq = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    corr = ""
    if spy is not None:
        j = pd.concat([r.rename('a'), spy.rename('b')], axis=1).dropna()
        if len(j) > 30:
            corr = f" | SPYcorr {j['a'].corr(j['b']):+.2f}"
    print(f"\n{label}")
    print(f"  CAGR {ann:+.1%} | Vol {vol:.1%} | Sharpe no-rf {sh_norf:.2f} | rf {sh_rf:.2f} "
          f"| MaxDD {mdd:.1%}{corr}")
    if cost_components:
        execution_total = sum(
            cost_components[name]
            for name in ('commission', 'spread', 'slippage', 'impact')
        )
        execution_bps = (
            execution_total / cost_components['notional'] * 10_000.0
            if cost_components['notional'] > 0 else 0.0
        )
        print(
            f"  costs: execution ${execution_total:,.0f} "
            f"({execution_bps:.2f}bp) | "
            f"borrow ${cost_components['borrow']:,.0f} | "
            f"financing ${cost_components['financing']:,.0f} | "
            f"total ${cost_components['total']:,.0f}"
        )
    by = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    ysh = r.groupby(r.index.year).apply(lambda x: (x.mean()/x.std()*np.sqrt(252)) if x.std()>0 else 0)
    cells = []
    for y, v in by.items():
        star = '*' if y in CRISIS else ' '
        cells.append(f"{y}{star}{v:+.0%}({ysh[y]:.1f})")
    print("  year: " + "  ".join(cells))


if __name__ == '__main__':
    prices = load_prices()
    present = [c for c in SECTOR if c in prices.columns]
    by_sector = {}
    for c in present:
        by_sector.setdefault(SECTOR[c], []).append(c)
    print(f"Loaded {len(prices)} days. Universe present ({len(present)} assets):")
    for s, members in by_sector.items():
        print(f"  {s:11s}: {', '.join(members)}")
    if not os.path.exists(EXTRA):
        print("\n(NOTE: trend_extra.csv not found — run pull_trend_ibkr.py for full breadth.\n"
              " Showing horizon effect on the base 9 for now; B will equal A until extras load.)")

    spy = prices['SPY'].pct_change(fill_method=None) if 'SPY' in prices.columns else None
    A = backtest(prices, BASE9, [252])
    B = backtest(prices, present, [252])
    C = backtest(prices, present, HORIZONS)

    print("\n" + "=" * 84)
    print("ABLATION — what actually moves the needle? (* = crisis year, the diversification test)")
    print("=" * 84)
    stats(A, f"A  9 assets, 12m only            (v1 baseline)", spy)
    stats(B, f"B  {len(present)} assets, 12m only           (breadth alone)", spy)
    stats(C, f"C  {len(present)} assets, 3/6/12m ensemble   (breadth + horizon)", spy)

    # save the deepest variant for the combination test
    C.to_csv(os.path.join(SCRIPT_DIR, 'trend_v2_returns.csv'))
    print("\nSaved C (deepest) to trend_v2_returns.csv for the combine-book test.")
    print("Gate: C should LIFT Sharpe vs A while KEEPING crisis-year alpha (2008/2020/2022 *) "
          "and low SPY corr.")
