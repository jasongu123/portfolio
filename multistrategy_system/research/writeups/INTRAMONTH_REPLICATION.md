# Intramonth legs 2+3: replication on other instruments (2026-09-15)

Research only: `intramonth_replication.py`, universes and gate fixed before any data was fetched.
Registry T-2026-09-15-25 (U1), -26 (U2), -27 (U3), -28 (pooled). Data: Yahoo adjusted closes,
`backtest_data/intramonth_replication_prices.csv` (48 symbols; the IBKR fetcher
`fetch_ibkr_daily.py` was written as a fallback and was not needed). Output
`results/intramonth_replication_daily.csv`.

**Question.** Is the turn-of-month sector cycle dead, or crowded out of the nine sector SPDRs?

**Answer: dead everywhere since 2024.** The frozen rule (legs 2+3, 252-day ranking, long top
third / short bottom third, 1 bp) applied unchanged to three universes nobody tuned it on shows
the same shape as the SPDRs: a 2020–2023 burst and losses since, in all three.

| Universe | Start | n | Full | 2012–2019 | 2020–2023 | Since 2024 | Last 2y | Since-2024 cumulative |
|---|---|---|---|---|---|---|---|---|
| U1 SPDR industry ETFs | 2007 | 12 | 0.07 | 0.07 | 0.63 | −0.08 | −0.28 | −2.3% |
| U2 equal-weight sector ETFs | 2007 | 9 | 0.49 | 0.19 | 1.53 | −0.71 | −0.51 | −10.3% |
| U3 top-3 stocks per sector | 1987 | 27 | 0.06 | 0.17 | 1.67 | −1.25 | −1.25 | −21.6% |
| Sector SPDRs (reference) | 1999 | 9 | 0.64 | 0.14 | 1.92 | −0.69 | −0.65 | −10.6% |
| Pooled, equal weight of U1–U3 | 2007 | | 0.37 | 0.16 | 1.46 | −0.85 | −0.83 | −11.6% |

Net Sharpe, no-rf, daily. U3's pre-2020 numbers carry survivorship bias (names fixed in 2026),
which makes its since-2024 collapse the more telling. Pairwise daily correlations between the
universes are 0.46 to 0.63: it is one effect, and it died in all of them together.

## Gate

| | Criterion | Result | |
|---|---|---|---|
| R1 | Pooled since-2024 Sharpe ≥ 0.30 and at least two universes positive | −0.85; 0 of 3 positive | **fail** |
| R2 | Pooled full-sample Sharpe ≥ 0.30 and positive in 2020–2023 | 0.37; 1.46 | pass |
| R3 | Monthly corr with equity and trend < 0.10 | all between −0.14 and +0.07 | pass |
| R4 | No four-year block above 75% of post-2012 profit | 2020–2023 is 126% | **fail** |

Pre-registered decision on R1 failing: the effect is dead everywhere since 2024; retire the
intramonth candidate.

## What this settles

- R2 passing says the effect was real: three independent universes and the SPDRs all show it in
  2020–2023, and the pooled full-sample Sharpe clears the bar. It was not an SPDR artefact.
- R1 and R4 failing say it has been gone since the start of 2024 in every instrument set,
  including single stocks and thin sub-sector ETFs, so it was not crowded out of the SPDRs in
  particular. Whatever produced turn-of-month sector pressure in 2020–2023 (the volume of
  rebalancing flow in high-dispersion years is the obvious candidate) is not present now.
- Orthogonality was never the problem. The series were uncorrelated with the book throughout.

## Consequences

1. **The intramonth candidate is retired.** All configurations (all legs, legs 2+3, the ±1 day
   shifts, and the three replication universes) are closed in the registry with this reason.
2. **The paper sleeve should be switched off.** `INTRAMONTH_SLEEVE_ENABLED=false` in the VM's
   environment, or the default in `intramonth_sleeve_live.py` changed to disabled. Not done
   here; the live file is changed only on request. The legs-2+3 package sent 2026-09-14 is moot.
3. **What would reopen it.** A forward record, on the SPDRs or the pooled universes, with a
   positive two-year Sharpe. Nothing in the backtest can; the recency criterion is binding.
4. **Trial count.** This pass added ten registered trials for the intramonth family (T-15, 16,
   21–28). The deflated-Sharpe accounting for any future intramonth-style candidate starts from
   that count.
