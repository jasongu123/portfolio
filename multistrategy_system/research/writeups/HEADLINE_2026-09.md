# Headline book, rebuilt on consistent conventions (2026-09-16)

> **Corrected again 2026-10-02 (AUDIT_RESPONSE_2026-10-01.md):** backtester defects fixed (future-vol swing sizing, close-to-morning capital leak, ETF label bands, cold annual warm-up, first-day return). Equity sleeve rf 0.24 / no-rf 0.72, MaxDD −20.2%; **headline book rf 0.44 / no-rf 1.04, worst year −0.40 (2022), MaxDD −8.2%**, positive in 5 of 6 years; DSR 63.4% at N = 9 (16% at N = 83). Series: `results/headline_2026-09_daily_audit.csv`.

> **Corrected 2026-09-24 — see VIX_LOOKAHEAD_2026-09.md.** The equity sleeve below read the trade
> date's VIX close (a look-ahead). Corrected (per-minute VIX gate, as live): equity sleeve rf 0.40 /
> no-rf 0.88, MaxDD −17.1%; **headline book rf 0.51 / no-rf 1.13, worst year −0.16 (2022), MaxDD
> −7.3%**, positive in 5 of 6 years; DSR 67.9% at N = 9. The tables below are the 2026-09-16
> figures, kept as recorded; `results/headline_2026-09_daily_new.csv` now holds the corrected
> series and the 2026-09-16 one is `results/headline_2026-09_daily_sameday_vix.csv`.


`headline_rebuild.py`; series in `results/headline_2026-09_daily_new.csv`. This replaces the
PORTFOLIO.md §1 table. The prior table is reproduced below as the labelled legacy version.

## What changed and why

| | Legacy headline (PORTFOLIO.md, to 2026-09-15) | Rebuilt |
|---|---|---|
| Equity series | caches built June 2026; 2024 had 246 of 251 days | regenerated 2026-09-16 with the current backtester and complete minute data; identical to legacy for 2020–2022, different for 2023–2025 |
| Trend series | 18 ETFs including SPY and IWM, which the equity engine also trades; full-sample vol scalar | the sleeve as it runs: 16 disjoint ETFs, live conventions, the constant 2.95 scalar, shared cost model |
| Book weights | inverse-vol from the whole window's volatility (look-ahead) | walk-forward inverse-vol, trailing 252 days, monthly, implementable |
| Drawdown | high-water mark after the first return (understated) | mark at initial capital |
| Sharpe | rf headline; DSR mixed conventions | rf headline, no-rf alongside; PSR and DSR on no-rf with a registry trial count |

The cost engine is not the source of any change: 2020–2022 regenerate to the basis point. The
2023–2025 equity years moved because the legacy caches were built on minute data with missing
days and an earlier code state (the swing cash-accounting correction and the execution-cost
identity fix post-date them); without version control the change cannot be attributed line by
line. 2024 fell from 1.28 to 0.71 rf-Sharpe and 2025 rose from 0.14 to 0.47.

## Sleeves, 2020–2025 (rf = flat 5%/yr; drawdown from initial capital)

| Series | rf-Sharpe | no-rf | worst year | MaxDD | vol | CAGR | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Equity sleeve, regenerated | 0.44 | 0.91 | −1.28 (2022) | −14.9% | 10.4% | +9.4% | +0.40 | +1.43 | −1.28 | +1.39 | +0.71 | +0.47 |
| Equity sleeve, legacy cache | 0.54 | 0.96 | −1.28 (2022) | −14.9% | 11.7% | +11.2% | +0.40 | +1.44 | −1.28 | +1.36 | +1.28 | +0.14 |
| Trend as live (16 ETFs, scalar 2.95) | 0.42 | 0.83 | −1.87 (2023) | −22.2% | 12.4% | +9.9% | +0.76 | +0.08 | +1.11 | −1.87 | +0.43 | +0.80 |
| Trend research baseline V1_trail | 0.45 | 0.88 | −1.87 (2023) | −20.4% | 11.6% | +10.0% | +0.62 | +0.23 | +1.35 | −1.87 | +0.28 | +0.78 |
| Trend legacy headline (18 ETFs) | 0.37 | 0.77 | −1.95 (2023) | −22.7% | 12.6% | +9.3% | +0.62 | +0.22 | +0.87 | −1.95 | +0.56 | +0.85 |

Daily correlations: equity to trend-as-live +0.01; trend-as-live to SPY −0.22; equity to SPY +0.55.

## Books, 2020–2025

| Book | rf-Sharpe | no-rf | worst year | MaxDD | vol | CAGR | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Walk-forward: equity + trend as live (the headline)** | **0.54** | **1.16** | **−0.02 (2022)** | **−7.4%** | **8.1%** | **+9.5%** | +0.79 | +0.78 | −0.02 | +0.21 | +0.68 | +0.76 |
| Same, old full-overlap weighting | 0.58 | 1.20 | +0.09 | −7.4% | 8.0% | +9.8% | +0.77 | +0.88 | +0.09 | +0.26 | +0.72 | +0.77 |
| Walk-forward: equity + V1_trail | 0.55 | 1.20 | +0.10 (2022) | −7.3% | 7.7% | +9.4% | +0.68 | +0.87 | +0.10 | +0.25 | +0.64 | +0.75 |
| Legacy headline, reproduced | 0.60 | 1.17 | −0.07 (2022) | −7.4% | 8.8% | +10.4% | +0.68 | +0.91 | −0.07 | +0.39 | +1.13 | +0.47 |

Walk-forward weights: equity 0.53 on average, ranging 0.41 to 0.61. The look-ahead in the old
weighting is worth about 0.04 to 0.07 of rf-Sharpe and a tenth to a third of a point on the
worst year, consistently across trend variants.

## Rigor, one convention (no-rf daily Sharpe)

| | Sharpe | skew | kurtosis | n | P(true SR > 0) | P(true SR > 0.5) |
|---|---|---|---|---|---|---|
| Equity sleeve | 0.91 | −0.35 | 7.9 | 1,491 | 98.6% | 84.0% |
| Headline book | 1.16 | −0.22 | 6.2 | 1,491 | 99.7% | 94.4% |

Deflated Sharpe of the headline book. The trial dispersion is the standard deviation (0.62) of
the standalone no-rf Sharpes of the nine sleeve-level candidates examined for inclusion on this
window (equity, trend, intramonth, RFS, short-interest timing, bearish positioning, Taking
Sides, seasonality, Crowded Spaces); N counts trials.

| N | What it counts | Hurdle SR* | DSR = P(SR > SR*) |
|---|---|---|---|
| 9 | sleeve-level candidates | 0.94 | 70.1% |
| 42 | every row in TRIAL_REGISTRY.md, including design variants never adopted | 1.37 | 30.9% |
| 100 | an upper bound including the unregistered pre-2026-09 feature and parameter sweeps | 1.57 | 16.4% |

Reading: against the search that actually selected the book (nine candidates), the book clears
the luckiest-of-N hurdle with 70% probability; against every configuration ever examined it is
plausible, not established. The recorded 88/77/63% figures mixed conventions and assumed a
narrower trial spread; they are withdrawn.

## What the rebuild changes in the claims

- The headline book is rf 0.54 / no-rf 1.16, worst year −0.02, drawdown −7.4%, on implementable
  weights and the sleeves as they run. The old 0.60 / −0.07 / −7.4% used look-ahead weights, an
  18-ETF trend that overlapped the equity engine's symbols, and an equity series with missing
  days in its best year.
- "Positive in 5 of 6 years" stands; 2022 is −0.02.
- The live trend sleeve (0.42) is better than the legacy table's trend row (0.33 in the
  original, 0.37 reproduced) because it drops the two symbols the equity engine trades.
- The equity sleeve's 2024 is materially weaker than recorded and its 2025 stronger; the
  2024–2025 average is about the same.
