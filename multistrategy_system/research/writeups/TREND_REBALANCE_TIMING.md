# Trend sleeve: rebalance timing off month-end (2026-09-16)

Research only: `trend_rebalance_timing.py`, gate pre-registered in its docstring; the live sleeve
is unchanged. Registry T-2026-09-16-38 (four-tranche candidate), -39 (the single-day sweep as one
diagnostic entry). Outputs `results/trend_rebalance_timing_returns.csv`, `_summary.csv`.

**Hypothesis.** Month-end is the most crowded execution window in the calendar, so a trend sleeve
that evaluates and trades at the month-end close pays anticipated rebalancing pressure; trading
mid-month, or spreading the book over four stamp days, should do better.

**Answer: the opposite.** The month-end stamp is the best of all twenty-one possible stamp days,
and the early-month stamps are the worst. Every gate criterion fails. The live convention stays.

## Setup

V1_trail rule (v2 mean-sign ensemble 63/126/252, per-asset vol targeting, 1/n, trailing scalar,
2 bp), live 16-ETF universe, 2008–2026, identical everywhere; only the day of the month on which
the rule is evaluated and traded changes. The month-end run reproduces the recorded V1_trail
series to machine precision.

## Results, rf-Sharpe by stamp day (full window)

| Stamp | rf-Sharpe | no-rf | MaxDD | | Stamp | rf-Sharpe | no-rf | MaxDD |
|---|---|---|---|---|---|---|---|---|
| Month-end (live) | **0.19** | 0.63 | −20.4% | | Day 11 | 0.03 | 0.47 | −27.8% |
| Day 1 | −0.02 | 0.43 | −22.2% | | Day 12 | 0.04 | 0.47 | −20.7% |
| Day 2 | 0.02 | 0.46 | −22.7% | | Day 13 | 0.03 | 0.46 | −22.9% |
| Day 3 | 0.01 | 0.46 | −20.2% | | Day 14 | −0.02 | 0.42 | −23.1% |
| Day 4 | −0.03 | 0.41 | −23.4% | | Day 15 | 0.13 | 0.57 | −19.3% |
| Day 5 | 0.08 | 0.52 | −21.3% | | Day 16 | 0.11 | 0.55 | −19.4% |
| Day 6 | 0.02 | 0.46 | −24.3% | | Day 17 | 0.16 | 0.60 | −20.2% |
| Day 7 | 0.17 | 0.62 | −19.4% | | Day 18 | 0.18 | 0.63 | −22.5% |
| Day 8 | 0.11 | 0.55 | −18.1% | | Day 19 | 0.15 | 0.59 | −22.2% |
| Day 9 | 0.00 | 0.44 | −23.1% | | Day 20 | 0.16 | 0.60 | −19.2% |
| Day 10 | 0.03 | 0.47 | −20.7% | | Four tranches (1, 6, 11, 16) | 0.05 | 0.50 | −19.1% |

Single-day median 0.03; month-end sits at the 100th percentile. Mean of days 5–15: 0.06.

| | Criterion | Result | |
|---|---|---|---|
| R1 | Month-end below the single-stamp median; mid-month mean ≥ month-end + 0.05 | 0.19 vs median 0.03; −0.13 | fail |
| R2 | Four tranches ≥ month-end + 0.05, drawdown not deeper | 0.05 vs 0.19 | fail |
| R3 | Difference positive in two of three sub-periods | 0 of 3 (−5.7%, −17.7%, −19.9% cumulative) | fail |
| R4 | Walk-forward book not worse | 0.59 → 0.47; worst year +0.11 → −0.08 | fail |

## Reading

The day-of-month profile is not noise. Stamps in the last week of the month (days 17–20 and
month-end) cluster at 0.15–0.19; stamps in the first week (days 1–4) cluster at −0.03–0.02; the
middle is mixed. A random draw would not order itself by proximity to month-end.

The mechanism that fits is the mirror of the one I registered. Month-end rebalancing flow is
anti-trend by construction: rebalancers sell what rose and buy what fell to restore weights.
A trend follower who trades at the month-end close is on the other side of that flow, buying
its winners while rebalancers are selling them, at prices the pressure has pushed in its
favour, and the pressure reverses in the first days of the month. Trading on day 1 to 4 means
buying after that reversal has run, which is the worst cluster in the table. The month-end
stamp is not paying the crowded window; it is being paid by it. This is also consistent with
the equity intramonth result: the turn-of-month move is real, and the side that profits from it
is the one providing liquidity to the rebalancers, which a month-end trend rebalance does
incidentally.

Two consequences:

1. **Do not move the trend sleeve's rebalance.** Any drift away from month-end costs about
   0.1–0.2 of rf-Sharpe on this record; spreading over the month costs 0.14.
2. **The sleeve's record has a timing component.** The rule's expected rf-Sharpe over a random
   stamp day is about 0.05, and month-end is worth roughly +0.14 above that. That is a
   structural advantage, not luck, as long as month-end rebalancing flow exists; it is also a
   dependency to know about. If turn-of-month flow ever gets arbitraged the way the equity
   version was, the trend sleeve loses part of its edge with it. Worth watching as a forward
   diagnostic: the month-end minus mid-month gap, re-estimated yearly.

## Caveats

- Sharpe differences between stamp days over 18 years have modest standard errors because the
  configurations are highly correlated, but a single sub-period could flip the ordering; the
  ordering is consistent in all three sub-periods for the candidate, which is the strongest
  evidence here.
- Stamping late in the month (day 17–20) is statistically indistinguishable from month-end and
  offers no improvement; not pursued.
