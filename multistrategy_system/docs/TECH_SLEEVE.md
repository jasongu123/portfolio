# Technology sleeve

The technology sleeve is a paper-only allocation tilt that reuses the existing
individual-stock signal engine. It is not yet a validated source of independent
alpha and is disabled by default.

## Where it lives

- `tech_sleeve_live.py`: isolated universe, paper guard, risk budget, and order
  routing into the shared equity strategy.
- `tech_sleeve_research.py`: cost-aware standalone backtest entry point.
- `ibkr2.py`: creates the sleeve and calls it from the serialized one-minute
  trading loop.

The shared IBKR equity connection is intentional. The legacy market-data path
temporarily installs callbacks and uses global queues, so a second copy of the
same signal engine could race the core engine. Logical isolation is provided by
disjoint symbols, separate allocation timing, risk caps, and journal labels.

## Universe and risk boundary

The initial disjoint universe is:

```text
INTC AMD AVGO QCOM MU
```

Existing core names such as NVDA, AAPL, MSFT, GOOG, and META remain in the core
equity sleeve. A constructor guard rejects exact ticker overlap.

Paper defaults:

- 10% sizing multiplier relative to the core position formula;
- 2% of NetLiquidation maximum entry notional per tech name;
- 10% of NetLiquidation estimated gross cap across the five tech names;
- cost and fill attribution under `sleeve='tech'`;
- no signal checks until RF models exist for all five symbols;
- forcibly disabled whenever `LIVE_TRADING=True`.

These limits apply to new entries. Exits are never blocked by a sleeve cap.

## Research workflow

Do not run the historical extraction while the live process is running. With
TWS logged into paper and API port 7497 enabled, use a separate process:

```powershell
py -3 -c "from ibkr_backtest import extract_ibkr_data; extract_ibkr_data(['INTC','AMD','AVGO','QCOM','MU'], years_back=6, timeframe='1 min')"
```

This is a large, paced IBKR download. Check completeness:

```powershell
py -3 .\tech_sleeve_research.py --check-data
```

Then run the same cost-aware stock logic used by the core backtest:

```powershell
py -3 .\tech_sleeve_research.py --start 2023-01-01 --end 2026-07-31
```

Before activation, compare the sleeve standalone and combined with the core
equity and trend sleeves. At minimum review net Sharpe, worst calendar year,
maximum drawdown, turnover, execution cost, and daily-return correlation. The
sleeve should not be promoted merely because its standalone return is positive.

## Paper activation

Activation is intentionally explicit:

```powershell
$env:TECH_SLEEVE_ENABLED = "true"
$env:TECH_SLEEVE_SIZING_MULTIPLIER = "0.10"
$env:TECH_SLEEVE_MAX_POSITION_PCT = "0.02"
$env:TECH_SLEEVE_GROSS_CAP_PCT = "0.10"
py -3 .\ibkr2.py
```

Leave `TECH_SLEEVE_ENABLED` unset or set it to `false` until the research gate
passes. Enabling it also adds the five symbols to RF initialization so the live
paper sleeve uses the same model pipeline rather than silently bypassing it.

## Current limitations

- The gross cap uses the signal engine's most recent prices and reserves the
  full per-name cap during a cycle; it is conservative but not broker-level
  real-time portfolio accounting.
- QQQ in the core equity sleeve creates economic technology overlap even though
  exact ticker ownership is disjoint. Combined exposure still needs reporting.
- The sleeve uses market orders because it inherits the core execution path.
  Cost monitoring is observe-only; it does not block or reroute orders.
- This addition does not establish that Intel or the technology basket improves
  the portfolio. That is the purpose of the research stage.
