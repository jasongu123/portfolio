# Taking Sides, free subset (2026-09-15)

Research only: `taking_sides.py`, gate pre-registered in its docstring. Registry T-2026-09-15-30
(composite), -31 (issuance), -32 (short interest). Data: `backtest_data/sp500_prices_2020_2026.csv`
(IBKR ADJUSTED_LAST, 569 of 625 names), `sp500_universe_2020_2026.csv` and `sp500_changes.csv`
(Wikipedia change log), `results/sec_shares_outstanding.csv.gz`, `results/finra_short_interest_raw.csv.gz`,
`backtest_data/sp500_splits_detected.csv`. Outputs `results/taking_sides_signals.csv`, `taking_sides_daily.csv`.

**Question.** McLean, Pontiff and Reilly (JFE 2025) find that firms, through net issuance and
repurchases, and short sellers trade on the right side of return predictability. Those two are
the free pieces. Does a monthly long/short tilt on them earn anything in large caps, 2020–2026,
entirely after the paper's sample?

**Answer: no.** The composite loses money net of a 10 bp cost, both components are weak or
negative, and the book is worse with it. Not promoted; four of five criteria fail.

## Construction

S&P 500 members 2020–2026 from the change log (current plus everyone added or removed since
2020, by date). Issuance signal: minus the 12-month log change in point-in-time, split-adjusted
shares outstanding from SEC filings (available on the filing date). Short-seller signal: minus
the one-month log change in split-adjusted FINRA short interest, available 12 calendar days after
settlement. Composite: mean of the two cross-sectional z-scores. Long top quintile, short bottom
quintile, monthly, equal-weight and value-weight, 10 bp per unit of one-way turnover.

## Results, net Sharpe (no-rf, monthly), 2020-05 to 2026-08

| Portfolio | 10 bp | 5 bp | 20 bp | Ann. return | Vol | MaxDD | Beta | Last 2y | Turnover /mo |
|---|---|---|---|---|---|---|---|---|---|
| Composite EW | −0.28 | +0.04 | −0.91 | −1.4% | 6.6% | −18% | −0.06 | −0.18 | 2.6 |
| Composite VW | −0.81 | −0.71 | −1.02 | −14.0% | 17.4% | −64% | −0.08 | −1.22 | 2.9 |
| Issuance EW | +0.17 | +0.19 | +0.13 | +1.4% | 10.2% | −22% | −0.13 | −0.56 | 0.3 |
| Issuance VW | −0.20 | −0.19 | −0.23 | −4.5% | 20.4% | −55% | −0.28 | −0.12 | 0.6 |
| Short interest EW | −0.70 | −0.27 | −1.55 | −3.3% | 6.1% | −24% | −0.01 | −0.05 | 3.3 |
| Short interest VW | −0.38 | −0.21 | −0.72 | −4.7% | 14.1% | −46% | +0.08 | −0.57 | 3.6 |

Per-year composite EW: 2020 −1.11, 2021 +0.91, 2022 −0.27, 2023 −0.27, 2024 −1.76, 2025 −0.05,
2026 +0.57. The two signals are uncorrelated (pooled +0.01).

## Gate

| | Criterion | Result | |
|---|---|---|---|
| T1 | EW net Sharpe ≥ 0.50, VW ≥ 0.30 | −0.28, −0.81 | fail |
| T2 | Last two years > 0 | −0.18 | fail |
| T3 | No 24-month block above 75% of profit | no profit | fail |
| T4 | Corr with equity and trend < 0.10, beta under 0.25, positive in the book's bad months | −0.05, +0.10, −0.06, +0.12% | fail (trend corr on the line) |
| T5 | Walk-forward book not worse on rf-Sharpe, worst year, drawdown | 0.50 → −0.14 | fail |

## Reading

- **Issuance carries a faint positive signal in equal weight** (0.17 net; the annual signal
  turns over little so costs barely matter) and nothing in value weight. That is the net
  issuance anomaly of Pontiff and Woodgate (2008) at its post-publication, large-cap level:
  present, and too small to trade.
- **Monthly short-interest changes have no alpha in this universe** and churn the book: 3.3
  turnover a month, so the 10 bp cost alone is 4% a year. Gross it is roughly flat. Whatever
  short sellers know about S&P 500 names is in the level of short interest, which is a known
  and crowded signal, not in its month-to-month change.
- **Value weight is dominated by a few names.** With quintiles of about 95 names, the top
  long weight was 36% (Microsoft) and the top short 17% (Tesla) at the last rebalance. The
  paper's value-weighted results come from the full CRSP universe, where that concentration
  does not arise. In the S&P 500 a value-weighted quintile portfolio is a bet on three stocks.
- **The paper's finding is about who is on the right side of *other* anomalies.** Firms and
  short sellers trade consistently with a composite of 130 predictors; that is different from
  their trading being a strong predictor on its own, which is what the free subset can test.
  The reading is that the paper is right and its free subset does not contain a sleeve.

## Data notes and corrections

- Split handling was corrected twice during the pass, before any verdict was written. SEC-only
  detection at 1.5% tolerance missed Tesla's 2020 five-for-one, which coincided with a capital
  raise (ratio 5.10), and invented a Wells Fargo two-for-one from a filing glitch; the corrupted
  issuance signal shorted Tesla through 2020–21. Splits are now pinned to the FINRA settlement
  where short interest jumps by the ratio and confirmed against the SEC counts; ratios of four
  or more between quarterly filings are kept without FINRA confirmation. Registry keeps the
  first run's row (−0.33 EW) as the pre-correction figure. No verdict changed.
- 56 of 625 names have no IBKR history, almost all acquired or delisted during the sample; the
  survivorship gap is 9% of the universe and biases in the strategy's favour if anything.
- The Wikipedia change log is incomplete (it reconstructs 526 members at end-2019 against
  about 505); membership dates are approximate.
- A 1.5 ratio cannot separate a 3:2 split from a stock-financed merger; International Paper's
  DS Smith deal is one such case and is treated as a split.

## Consequence

Not a candidate. The cached FINRA and SEC data remain useful for a level-of-short-interest
signal or for the crowding measure in "Crowded Spaces", which are different trials. The registry
now holds 32 trials.
