# Portfolio overview

*Updated 9 October 2026. This replaces the September 2026 version, which described an older
configuration (intraday mean reversion with an ML gate) and pre-audit numbers. The full research
record is in [`../research/README.md`](../research/README.md).*

## What runs (paper account, since October 2026)

| Sleeve | What it does | Status |
|---|---|---|
| Cross-asset trend | Time-series momentum on 16 ETFs (equities, Treasuries, commodities, currencies, real estate), 3/6/12-month signal ensemble, rebalanced at month-end | Validated |
| ETF intraday momentum | SPY, QQQ and IWM on the exact rules of Zarattini, Aziz & Barbon (2024): decisions every 30 minutes, trailing stop, flat by the close | Six-month forward paper test |
| Stock swing book | 21 large-cap stocks, multi-day long positions with ATR trailing stops | Running; treated as risk-managed market exposure, not alpha |

Removed after research:
- stock intraday mean reversion, which lost money before costs;
- gap trades, which never fired;
- the ML trade gate, whose whole value was shrinking the losses of the removed strategies.

## Headline backtest (2 Jan 2020 to 15 Dec 2025)

| | Sharpe, rf | Sharpe, no-rf | Max drawdown |
|---|---|---|---|
| Restructured book: trend + ETF momentum (paper rules) + stock swing book, walk-forward inverse-volatility weights | 0.98 | 1.70 | −5.8% |
| Old book (original engine + trend), same method and window | 0.44 | 1.04 | −8.2% |
| Trend sleeve alone | 0.41 | 0.81 | −22.2% |

The three sleeves barely move together (daily correlations −0.12 to +0.01), and every year was
positive. The figures are in-sample: the restructure was chosen on these same years, so the live
forward tests are the real check. The ETF sleeve assumes 0.5 bp a side and is cost-sensitive. "rf"
subtracts a 5% cash rate; "no-rf" does not; the two are never mixed. Details are in
[`../research/writeups/HEADLINE_RESTRUCTURED.md`](../research/writeups/HEADLINE_RESTRUCTURED.md).

## What the research found

- **The equity engine's profit was its swing holdings.** The intraday strategies subtracted from it.
  - The swing book matched an equal-weight basket of the same stocks on risk-adjusted return:
    Sharpe 1.19 vs 1.13, against 0.78 for SPY.
  - It cut the maximum drawdown from 31% to 13%.
  - The benefit came from exposure timing, not stock selection.
- **Replicating the source paper exposed my own modifications.** The paper's rules earned a Sharpe
  of 0.81; the engine's modified version earned −0.89. Deciding every minute instead of every 30
  minutes did the most damage.
- **Trend following earns at month-end.** It takes the other side of pension and fund rebalancing;
  month-end was the best of all 21 rebalance days.
- **Large US stocks since 2020 are efficient for public news and published patterns.** All of the
  following failed under pre-registered rules:
  - earnings and guidance drift;
  - index additions, priced within minutes of S&P's announcement;
  - six published anomalies;
  - ML stock ranking;
  - pairs trading and residual statistical arbitrage, where the gross edge is real but retail costs
    and execution delay remove it.

97 pre-registered trials are logged in [`../research/TRIAL_REGISTRY.md`](../research/TRIAL_REGISTRY.md), and
the failures are grouped by cause in [`../research/FAILURE_AUTOPSY.md`](../research/FAILURE_AUTOPSY.md).

## Known limits

- **Paper trading only.** IBKR declined futures permission, so the system uses ETFs.
- **Short backtests.** Minute data starts in late 2019, so most backtests cover 2020–2025.
- **Survivorship in older periods.** The stock list was partly chosen with hindsight.
