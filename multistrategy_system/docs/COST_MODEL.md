# Trading-cost model

## Status

The shared cost model is implemented in research and live paper trading. Live
integration is **observe-only**: it records predicted and realized execution
costs but does not block, resize, or reroute orders.

Behavioral cost gates remain deliberately disabled until they pass historical
tests and a live shadow comparison.

## Cost components

`trading_costs.py` models all values as one-way dollar costs:

1. Commission: `max(abs(shares) * per_share, minimum_commission)`.
2. Spread: quoted half-spread when a valid bid/ask exists; otherwise a
   configurable bps-of-notional fallback.
3. Extra slippage: an optional bps buffer for adverse selection beyond the
   displayed spread.
4. Market impact: optional square-root impact using daily volatility and
   dollar ADV. The coefficient defaults to zero until calibrated.
5. Borrow: annualized cost on short notional over the actual holding days.
6. Financing: annualized cost on borrowed cash over the actual holding days.

The backtest and live observer use the same `CostModelConfig` and
`CostEstimate` types.

## Backtest integration

### Equity sleeve

`ibkr_backtest.py` now:

- applies one shared commission minimum and execution-cost formula;
- charges each exposure change at its trade-point close rather than charging
  every trade at the day's open;
- counts the full notional of a long/short reversal;
- records commission, spread, slippage, impact, notional, and total cost;
- includes final liquidation costs in the equity curve;
- uses cash accounting for swing positions without adding the same entry fee
  or realized P&L through `daily_pnl` a second time.

Historical minute bars do not contain reliable bid/ask quotes. Consequently,
the legacy `slippage_bps` argument maps to `fallback_execution_bps` and appears
in the reported `spread_cost` component. This is an assumption, not a measured
historical spread.

### Trend sleeve

`trend_follow_v2.py` applies execution cost to actual scaled weight changes and
can accrue short borrow and leveraged-cash financing by calendar day. Defaults
for borrow, financing, and per-share commission remain zero so the default trend
series stays backward compatible. Supply explicit assumptions for a realistic
stress run.

The historical full-sample portfolio-volatility scalar is intentionally
preserved in this cost-only version. Replacing it with lagged walk-forward
scaling is a separate strategy-version change.

### Technology sleeve

The opt-in paper technology sleeve reuses the equity signal and execution path
but records order intent with `sleeve='tech'`. Its disjoint universe and hard
entry caps are documented in `TECH_SLEEVE.md`. It remains disabled by default
pending a cost-aware backtest and combined-book review.

### Stress test

Run one year or the complete resumable grid:

```bash
python3.12 cost_stress.py 2025
python3.12 cost_stress.py
```

The result file now includes execution notional, every cost component, total
execution cost, and effective bps. Rows produced by the former formula remain
auditable but are not reused as shared-model cache hits.

## Live integration

Immediately before submission, `ibkr2.py` and `trend_sleeve_live.py` record:

- composite client/order ID;
- strategy and sleeve;
- symbol, side, quantity, and order type;
- bid, ask, midpoint, and quoted spread;
- predicted commission, spread, slippage, impact, and total cost.

`execDetails` then records every partial fill and calculates side-adjusted
slippage from the decision-time midpoint. `commissionReport` attaches the
broker's actual commission whether it arrives before or after the fill.
Duplicate IBKR callbacks are idempotent.

Data is stored in `trading_system.db`:

- `order_intents`
- `execution_costs`

If quote collection or SQLite logging fails, the order path continues unchanged
and prints an observer warning.

Summarize recent observations with:

```bash
python3.12 execution_cost_report.py --days 30
```

## Live environment controls

```text
COST_OBSERVE_ENABLED=true
COST_COMMISSION_PER_SHARE=0.0035
COST_MIN_COMMISSION=0.35
COST_FALLBACK_EXECUTION_BPS=1.0
COST_EXTRA_SLIPPAGE_BPS=0.0
COST_IMPACT_COEFFICIENT=0.0

# Unvalidated paper technology candidate; leave false until research passes.
TECH_SLEEVE_ENABLED=false
TECH_SLEEVE_SIZING_MULTIPLIER=0.10
TECH_SLEEVE_MAX_POSITION_PCT=0.02
TECH_SLEEVE_GROSS_CAP_PCT=0.10
```

`COST_IMPACT_COEFFICIENT` should remain zero until order size, volatility, ADV,
and observed implementation shortfall support a calibration. Paper-account
fills are useful for plumbing tests but should not be treated as proof of live
execution quality.

## Activation gate

The next behavioral version should be activated only after:

1. sufficient live fills have valid arrival midpoints;
2. predicted and realized cost are compared by symbol, sleeve, side, order
   type, time of day, spread, and participation;
3. conservative cost estimates are rerun through the historical backtest;
4. a no-trade rule improves net walk-forward performance;
5. the same rule wins in live shadow mode without blocking exits.

Until then, cost output is analytics rather than an order veto.
