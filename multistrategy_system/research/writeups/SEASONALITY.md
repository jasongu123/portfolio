# Same-calendar-month seasonality in S&P 500 stocks (2026-09-15)

Research only: `seasonality.py`, gate pre-registered in its docstring. Registry T-2026-09-15-33
(equal weight), -34 (value weight). Data: `backtest_data/sp500_prices_long.csv` (IBKR
ADJUSTED_LAST, 569 names, 2001–2026), the dated S&P 500 universe from TAKING_SIDES.md. Outputs
`results/seasonality_signals.csv`, `seasonality_daily.csv`.

**Question.** Heston and Sadka (2008) and Keloharju, Linnainmaa and Nyberg (2016, 2021): a
stock's average return in the same calendar month over prior years predicts its return in that
month. Does it still, in large caps, after 2020?

**Answer: no, it has reversed.** Long the top decile, short the bottom, the strategy loses
10% a year net in 2020–2026, and the raw decile spread has the wrong sign. The same code on
2010–2019 reproduces the published effect, so this is the effect's fate and not a construction
error.

## Construction

Signal at each month-end: mean monthly return of the stock in the coming calendar month over
the prior twenty years (at least eight observations; median 20). Decile long/short within the
dated S&P 500 universe, monthly, equal- and value-weight, 10 bp per unit of one-way turnover.
Median 444 names per month. Test window 2020-01..2026-08, after every source sample.

## Results, net Sharpe (no-rf, monthly)

| Portfolio | 10 bp | 5 bp | 20 bp | Ann. return | Vol | MaxDD | Beta | Last 2y | Turnover /mo |
|---|---|---|---|---|---|---|---|---|---|
| 20y decile EW | −0.89 | −0.71 | −1.26 | −10.1% | 13.0% | −55% | +0.08 | −1.65 | 3.4 |
| 20y decile VW | −0.25 | −0.17 | −0.42 | −6.5% | 24.2% | −66% | +0.22 | −0.63 | 3.5 |
| 20y quintile EW | −0.75 | −0.51 | −1.21 | −6.0% | 10.2% | −40% | +0.08 | −1.88 | 3.1 |
| 10y decile EW | −0.55 | −0.41 | −0.83 | −8.4% | 15.6% | −57% | +0.03 | −1.49 | 3.5 |

Per year, 20y decile EW: 2020 −1.61, 2021 −0.38, 2022 −1.75, 2023 +0.58, 2024 +0.05, 2025 −2.51,
2026 −1.83. Negative in ten of twelve calendar months.

Raw decile spread (next-month mean return, top minus bottom):

| | Top decile | Bottom decile | Universe | Spread t |
|---|---|---|---|---|
| 2010–2019 | +1.68% | +1.17% | +1.26% | +2.34 |
| 2020–2026 | +1.08% | +1.56% | +1.05% | −1.35 |

## Gate

| | Criterion | Result | |
|---|---|---|---|
| S1 | EW net Sharpe ≥ 0.50, VW ≥ 0.30 | −0.89, −0.25 | fail |
| S2 | Last two years > 0 | −1.65 | fail |
| S3 | Concentration | no profit | fail |
| S4 | Corr with equity and trend < 0.10, beta, bad months | +0.21, −0.19; bad-month mean −0.37% | fail |
| S5 | Walk-forward book not worse | 0.59 → −0.01 | fail |

## Implementation check (diagnostic, not a trial)

The same construction on 2010–2019, with the 2020–2026 membership applied backward (which
biases that decade upward), gives net Sharpe +0.24 EW and +0.59 VW at 10 bp, gross +0.74 and
+0.93, positive in seven of ten years. That is the published effect at roughly the published
size. Two things follow: the code is right, and the strategy was always cost-fragile, since 3.4
turnover a month at 10 bp is 4% a year and took the equal-weight version from 0.74 to 0.24 even
in its good decade.

## Reading

Cross-sectional seasonality in large caps did not fade after 2020; it inverted. The plausible
account is the usual one for a fully public, cheap-to-compute signal: once enough capital
positions for the seasonal month ahead of time, the expected move is pulled forward and the
month itself pays the other way. The pattern is the same as the intramonth sleeve's death, and
it is consistent with four of four cross-sectional large-cap effects tested this week (short
interest, issuance, intramonth flow, seasonality) being flat or reversed since 2020 while their
pre-2020 records replicate.

## Consequence

Not a candidate. Registry at 34 trials. The 25-year price file is reusable for anything
cross-sectional on this universe.
