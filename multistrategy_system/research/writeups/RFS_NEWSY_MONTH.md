# RFS newsy-month equity signal

`RFS` is shorthand in this repository for Hongye Guo's 2025 *Review of Financial
Studies* paper, [Earnings Extrapolation and Predictable Stock Market
Returns](https://academic.oup.com/rfs/article/38/6/1730/8101501). It is not a random
forest and it does not read individual earnings announcements.

## The idea

January, April, July, and October are the first months of calendar quarters and contain
an unusually large amount of earnings news. The paper calls these **newsy months**.
Returns from the four most recent newsy months have opposite predictive implications:

- In February/March, May/June, August/September, and November/December, recent newsy-month
  strength predicts continuation.
- In January, April, July, and October, the same strength predicts reversal.

For each target month the aggregate-market implementation:

1. sums returns from the four most recent completed newsy months;
2. subtracts the expanding historical mean of that four-month sum;
3. changes the sign for a newsy target month;
4. estimates an expanding OLS return-on-signal slope from prior months only; and
5. holds `slope × signal` units of the market for one month.

The author-provided replication package is available at [Harvard Dataverse
(doi:10.7910/DVN/AFOM9J)](https://doi.org/10.7910/DVN/AFOM9J). The implementation in
`rfs_newsy_month.py` follows `TSMomentumImplementation_final.do` from that package.

The paper-history plus post-June-2021 SPY extension can be run with:

```powershell
python .\rfs_newsy_month.py `
  --monthly-returns .\TSImplementationCombined.tab `
  --fallback-return-col ret_gfd `
  --extend-prices .\trend_cache.csv --symbol SPY --extension-start 2021-07 `
  --start 2021-07 --cost-bps 1 --max-abs-position 1
```

`TSImplementationCombined.tab` is file ID `10802956` in the official Dataverse package;
it is third-party research data and is therefore not silently copied into this repository.

## Run the local preliminary test

```powershell
python .\rfs_newsy_month.py --prices .\trend_cache.csv --symbol SPY `
  --cost-bps 1 --output .\rfs_spy_monthly.csv
```

On macOS/Linux:

```bash
python3 rfs_newsy_month.py --prices trend_cache.csv --symbol SPY \
  --cost-bps 1 --output rfs_spy_monthly.csv
```

The final daily-price month is excluded by default so an incomplete cached month cannot
enter the backtest. Use `--include-final-month` only when the last price is known to be a
completed month-end.

## Important limits

- This is standalone research. It is not imported by `ibkr2.py` and cannot place orders.
- The local SPY history begins in 2007, versus the paper's 1871 estimation history and
  1926–2021 reported evaluation period. The local result is therefore a short preliminary
  replication, not independent validation.
- The paper's final volatility multiplier is estimated from the full completed sample.
  The script reproduces and labels that convention. A live candidate needs trailing-only
  volatility scaling, a leverage cap, and a fixed capital budget.
- The stronger cross-sectional industry version requires point-in-time industry returns
  and market-cap weights. The current repository does not contain that panel.
- Trading costs are modeled as one-way basis points times the absolute monthly change in
  market exposure. The paper's headline tests do not subtract this implementation cost.

Nothing in this file or the research script changes the live equity or trend sleeves.

## Current preliminary results

These figures use no risk-free-rate subtraction and are not yet a keep/reject decision.

| Test | Months | Cost / cap | Net Sharpe | MaxDD | Net mean/month |
|---|---:|---|---:|---:|---:|
| Author-data reproduction, 1926-11–2021-06 | 1,136 | 0 bp / paper uncapped | 0.441 | -43.3% | 0.846% |
| Local SPY history, 2010-03–2026-05 | 195 | 1 bp / 1.0x | 0.204 | -33.6% | 0.153% |
| After paper data cutoff, 2021-07–2026-05 | 59 | 1 bp / 1.0x | 0.699 | -17.1% | 0.609% |

The author reports 0.829% per month in Table A3; the Python reproduction is 0.017
percentage point higher while matching the approximately 0.44 Sharpe. That small level
difference remains a replication note, likely related to the Dataverse tabular export
versus the author's original Stata input, and should not be hidden.

The post-June-2021 slice is encouraging and survives a simple 1/2/5-bp cost stress, but it
has only 59 observations. The full local ETF-era sample is weak, the 1.0x cap is an added
design trial rather than a paper parameter, and 2024 was sharply negative. The next gate
is a frozen-specification walk-forward and correlation test against the existing equity
and trend sleeves—not live integration.

## Gate results (2026-09-04)

The frozen-specification gate with trailing-only sizing and a century of free estimation
history (Ken French market series) is in `rfs_gate.py` and `RFS_GATE.md`. Outcome: the signal
passed the standalone tests (1950–2026 net Sharpe 0.44, positive in every decade, post-paper
0.80, correlation with both sleeves near zero) and failed the pre-registered book test because
of a flat-rf sizing artefact plus a bad 2024. Not promoted; held as a shadow paper candidate.
