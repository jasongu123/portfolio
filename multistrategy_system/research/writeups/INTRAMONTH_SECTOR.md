# Sectoral intramonth momentum cycle (2026-09-07)

Research only. `intramonth_sector.py`. Nothing live was touched. Rules follow Quantpedia's
"Sectoral Intramonth Momentum Cycle" (August 2026), which implements Nathan, Suominen & Tasa,
"The Intramonth Momentum Cycle" (SSRN 6426026, 2026), on the nine Select Sector SPDRs.

## Verdict and the caveat that goes with it

**Passed all four pre-registered criteria. Promoted to paper-trade candidate.** It is the first
candidate in this program to do so.

Read the next sentence before the tables. The record is front-loaded: the full-sample Sharpe of
0.58 is 0.91 over 1999–2011 and 0.22 over 2012–2026, and the leg that carries the paper's
headline mechanism, pre-month-end liquidation, has been negative since 2012. The gate's
recent-period criterion (2017 onward, threshold 0.30) was met at 0.37; a 2012-onward criterion
would have failed. The gate was written before the run and is not being rewritten after it, but
the honest description is "a decayed anomaly with a still-live turn-of-month component," and
paper trading is what "promoted" means here, not sizing.

## Rules

- Universe: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY; SPY only as the short in the secondary
  variant. Adjusted closes from December 1998 (yfinance, dividend-adjusted).
- Ranking: trailing 252-trading-day total return; top three = winners, bottom three = losers,
  equal-weighted at one third each.
- D0 is the last trading day of the month, D−k is k trading days before it, D+k the k-th trading
  day of the next month. All stamps use actual trading days.
- Leg 1: enter at the close of D−10 with the ranking at D−10, exit at the close of D−5.
- Leg 2: enter at the close of D0 with the ranking at D0, hold through the close of D+1.
- Leg 3: flip the leg-2 portfolio at the close of D+1, hold through the close of D+3.
- Flat otherwise, about 38% of days invested. Strat2 is long winners / short losers; Strat1 is
  long winners / short SPY.
- Cost: 1 bp per unit of absolute weight change per ETF; 0 and 2 bp reported. Turnover is about
  11.5 units of notional per month, so 1 bp costs roughly 1.4% a year.

## Pre-registered gate (Strat2 at 1 bp)

| | Criterion | Result | |
|---|---|---|---|
| G1 | Full-sample net Sharpe ≥ 0.40 and every leg's contribution positive | 0.58; legs +1.58%, +1.18%, +2.77% a year | pass |
| G2 | Rolling 5-year Sharpe positive in ≥ 70% of windows | 98% of 5,471 windows | pass |
| G3 | 2017 onward net Sharpe ≥ 0.30 | 0.37 on 2,432 days | pass |
| G4 | 2020–2025 book: correlations with equity and trend < 0.30, three-sleeve no-rf Sharpe not lower, MaxDD not deeper | −0.11 and −0.16; 1.52 → 1.66; −5.2% → −3.1% | pass |

## Replication

| | Source (cost not stated) | Here, 0 bp | Here, 1 bp | Here, 2 bp |
|---|---:|---:|---:|---:|
| Annual return | 5.99% | 7.16% | 5.63% | 4.12% |
| Volatility | 10.87% | 10.40% | 10.40% | 10.40% |
| Sharpe | 0.55 | 0.72 | 0.58 | 0.44 |
| MaxDD | −21.7% | −22.5% | −22.9% | −23.3% |

The source's figures sit between the 0 and 1 bp rows, consistent with a small cost assumption
or minor rule differences. Strat1 (short SPY) is weaker: Sharpe 0.41 at 1 bp, and its leg 2 is
negative, so it is recorded as a second trial and not pursued.

## Where the return comes from

| Period | Days | Annual return | Vol | Sharpe | MaxDD |
|---|---:|---:|---:|---:|---:|
| 1999-12 to 2011-12 | 3,040 | +10.79% | 12.0% | **0.91** | −22.9% |
| 2012-01 to 2019-12 | 2,012 | −0.19% | 6.8% | **0.01** | −15.8% |
| 2020-01 to 2026-09 | 1,678 | +3.70% | 10.9% | 0.39 | −19.5% |
| 2012-01 to 2026-09 | 3,690 | +1.56% | 8.9% | 0.22 | −19.5% |

Per-leg Sharpe, each leg run alone at 1 bp:

| Leg | 1999–2011 | 2012–2019 | 2020–2026 |
|---|---:|---:|---:|
| 1, pre-month-end momentum (D−10 to D−5) | 0.61 | −0.11 | −0.23 |
| 2, turn-of-month momentum (D0 to D+1) | 0.21 | 0.11 | 0.78 |
| 3, month-start reversal (D+1 to D+3) | 0.70 | 0.09 | 0.62 |

Per-year Sharpe, 2020 onward: 0.62, 1.18, 0.93, 0.56, −1.02, −0.22, and +0.15 for 2026 to
date. The two most recent full years are negative.

Reading: the pre-month-end liquidation effect that the paper is built around looks arbitraged
away after 2011; what survives is the turn-of-month pair, momentum into the first day of the
month and reversal over the next two, which is the pattern the older month-end-flow literature
documents. The strategy as published is therefore two-thirds a historical artefact and one-third
a live effect. A leg-2-plus-leg-3 version would be a new trial and was not run.

## Book test, 2020–2025 monthly, inverse-vol weights

| Book | No-rf Sharpe | rf-Sharpe | Mean/yr | Vol | MaxDD | Worst year |
|---|---:|---:|---:|---:|---:|---:|
| equity + trend | 1.52 | 0.80 | 10.6% | 7.0% | −5.2% | +0.39 |
| equity + trend + intramonth | 1.66 | 0.66 | 8.3% | 5.0% | −3.1% | −0.01 |
| same, vol-matched to 7.0% (diagnostic) | 1.66 | 0.95 | 11.6% | 7.0% | −4.3% | +0.32 |

The rf-Sharpe drop is the flat-5% convention again: a lower-vol book is penalised. At equal vol
the three-sleeve book is better on every column except worst year, and the worst year is 2024–25,
where the strategy lost. Per-year rf-Sharpe with the sleeve added: 2020 0.68 → 1.00, 2021 1.48 →
2.02, 2022 0.39 → 0.66, 2023 0.56 → 0.45, 2024 1.20 → 0.07, 2025 0.81 → −0.01.

## Implementation notes

- Two trades per leg per side at the close; market-on-close orders in the most liquid sector
  ETFs. Shorting three sector SPDRs at one third each is general-collateral borrow; not modelled.
- No survivorship or look-ahead issues: ETFs, rankings at the entry close, positions earn the
  next day, month boundaries on actual trading days, drawdown from initial capital.
- No untouched holdout exists; the source's sample ends June 2026. Forward paper months are the
  only true test, and given the decay pattern they are the test that matters.

## Next step

Shadow paper trading: a daily script that computes the target weights from the calendar and
rankings and logs them without sending orders, so a forward record starts now. If a forward year
looks like 2020–2023 rather than 2024–2025, size it at a fraction of inverse-vol weight; if it
looks like 2012–2019, close it.

Trial registry: two trials (Strat2, Strat1). See `TRIAL_REGISTRY.md`.
