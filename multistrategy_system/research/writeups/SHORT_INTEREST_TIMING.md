# Aggregate short-interest timing with the authors' index, spliced to the present (2026-09-04, corrected 2026-09-05)

Research only. `short_interest_splice.py` supersedes the data-limited preliminary that
`short_interest_timing.py` produced earlier the same day (kept for the record). Nothing live was
touched. The authors' data files were downloaded to the scratchpad for research and are not in
the repository.

## Corrections applied 2026-09-05

An independent review found the following, all fixed:

1. **Shares outstanding.** Facts filed on the same day were summed across different cover dates,
   so a filing that carried an amended prior quarter alongside the current one double-counted
   (378 filings since 2020; Advance Auto Parts on 2024-05-30 came out at 119m shares instead of
   60m). Only facts at the filing's latest cover date are now used. The overlap correlation with
   the authors' series is unchanged at 0.98.
2. **Holdout regression.** The slope is now estimated on SPY excess returns over the same 53
   realised months as the reported performance, not on the French market through a later month.
3. **Maximum drawdown** now counts a first-period loss.
4. **Fetch robustness.** A failed or incomplete FINRA page no longer marks a settlement date as
   complete; SEC cache entries are retried after transient failures and refreshed after 30 days.
   An audit of the existing cache against the API's record totals found all 612 date-class groups
   complete, so no re-fetch was needed.

The verdict is unchanged.

## Data

- **Authors' index.** Rapach, Ringgenberg & Zhou (JFE 2016), updated by Ringgenberg to
  1973-01..2021-12 and posted on his data page: `siiInSample` (full-sample linear detrend of
  log equal-weighted short interest, standardised) and `siiOutOfSample` (recursive detrend, from
  1978-01). The JFE replication package supplies raw EWSI 1973-01..2014-12, S&P 500 and CRSP
  returns, and the Goyal–Welch predictors.
- **Reconstruction.** log EWSI 1973–2021 recovered from `siiInSample` by the affine map fitted
  on the 1973–2014 overlap with raw EWSI: R² 0.99975.
- **Extension 2020-04..2026-08.** FINRA consolidated short interest per symbol (free API,
  exchange-listed classes only, 153 settlement dates), divided by shares outstanding from SEC
  XBRL company facts (point-in-time by filing date, 7,082 filers mapped), equal-weighted across
  mapped names (median 3,517 per date), mid-month settlements only. Over the 21-month overlap
  with the authors' series the correlation of log levels is **0.98**.
- **Returns.** Ken French Mkt-RF and RF for the long history; SPY monthly for the implementable
  series from 2007.
- Caches: `results/finra_short_interest_raw.csv.gz` and `results/sec_shares_outstanding.csv.gz`,
  reusable as the short-seller leg of the "Taking Sides" candidate.

## Pre-registered gate (written before the holdout was computed)

| | Criterion | Binding | Outcome |
|---|---|---|---|
| G1 | 1973–2021 in-sample regression of next-month market excess return on `siiInSample`: negative slope, HAC \|t\| ≥ 2.5 | yes | **fail** (t −1.86) |
| G2 | 1990–2021 walk-forward on the authors' OOS column: OOS R² > 0 and timing-rule net Sharpe ≥ buy-and-hold | yes | **pass** (R² +1.35%, 0.88 vs 0.79) |
| G3 | 2022-01 onward holdout on the spliced series: negative slope and timing Sharpe within 0.20 of SPY | informational | **fail** (slope +2.31%, t +1.72; 0.76 vs 0.81) |
| G4 | 2020–2025 inverse-vol book: three-sleeve no-rf Sharpe ≥ two-sleeve and MaxDD not deeper | yes | **fail** (Sharpe 1.54 vs 1.52; MaxDD −5.6% vs −5.2%) |

Verdict: **not promoted.** One trial; no variants were run.

## G1: replication, and what the full sample shows

| Regression | Slope per s.d. | HAC t | n |
|---|---:|---:|---:|
| Paper-style index (trend fitted 1973–2014), S&P 500 excess, 1973–2014 | −0.49% | −2.51 | 503 |
| Same, log excess returns | −0.50% | −2.50 | 503 |
| Same, CRSP VW excess | −0.51% | −2.39 | 503 |
| Authors' `siiInSample` (trend fitted 1973–2021), French Mkt-RF, 1973–2014 | −0.31% | −1.14 | 503 |
| Authors' `siiInSample`, French Mkt-RF, **1973–2021** | **−0.37%** | **−1.86** | 588 |

The paper's slope of about −0.5% per standard deviation is reproduced on the paper's sample. G1
still fails, and the table shows why: the index is a linear detrend of log short interest, and
log EWSI peaked in 2008 (2.17) and fell to 1.14 by 2021. Fitting the trend through 2021 flattens
it (+0.072/yr versus +0.086/yr) and reshapes the in-sample index; the two versions correlate only
0.80. The predictability lives in deviations from a trend line that the data stopped following
after 2008.

## G2: walk-forward 1990–2021 on the authors' OOS column

| Window | Months | OOS R² | Timing net Sharpe | Buy-and-hold Sharpe | Avg weight | MaxDD timing / market |
|---|---:|---:|---:|---:|---:|---|
| 1990-01 to 2021-12 | 383 | +1.35% | **0.88** | 0.79 | 1.23 | −48.6% / −50.3% |
| Paper OOS window 1990–2014 | 299 | +1.17% | 0.85 | 0.73 | | |
| Post-paper 2015–2021 | 84 | +2.30% | 1.05 | 1.05 | 1.50 (at cap) | |
| Own recursive construction, 1990–2021 (G2b) | 383 | +0.41% | 0.82 | 0.79 | 1.31 | |

G2 passes. The post-2015 row already shows the failure mode: the index sat far below trend, the
rule sat at its 1.5× cap for all 84 months, and the Sharpe equals buy-and-hold because the
position never moved.

Replication caveat: the authors' recursive construction is not documented in the package and my
recursive detrend does not reproduce it (correlation 0.88 overall, near zero after 2010). G2 uses
their column as published; G3 and G4 use my construction because it is the only definition that
can be carried across the 2021-12 seam.

## G3: holdout 2022-01 to 2026-05 on SPY

| | Timing rule | SPY |
|---|---:|---:|
| Months | 53 | 53 |
| Predictive slope on SPY excess (per s.d.) | **+2.31%**, t +1.72 | |
| Net Sharpe | 0.76 | 0.81 |
| Average weight | 1.41 | 1.00 |
| MaxDD | −34.7% | −23.9% |

The sign is wrong in the holdout, though not significantly on 53 months. What is unambiguous is
the behaviour: the index stayed near its lows, the rule stayed close to maximum leverage, and the
strategy rode the 2022 bear at 1.4× the market.

## G4: book test 2020–2025 (monthly, inverse-vol weights)

| Book | No-rf Sharpe | rf-Sharpe | MaxDD | Vol | Worst year |
|---|---:|---:|---:|---:|---:|
| equity + trend | 1.52 | 0.80 | −5.2% | 7.0% | +0.39 |
| equity + trend + SI timing | 1.54 | 0.93 | −5.6% | 8.2% | −0.38 |

Correlations over the 72 months: SI timing with SPY **1.00**, with the equity sleeve +0.68, with
trend −0.35. The strategy is SPY at 1.4×, so adding it adds equity beta. G4 fails as registered.

## Mechanism to record

The index measures short interest relative to a linear trend. Short interest rose for 35 years
and then structurally halved after 2008 as hedging moved to ETFs and derivatives and retail
participation changed the shorting base. Since about 2015 the detrended index has been pinned
several standard deviations below trend, which the rule reads as a permanent buy signal. It has
not made a timing decision in ten years, and it adds beta to a book that already has plenty.

## What is retained

- The per-symbol short-interest and point-in-time shares-outstanding caches.
- The reconstruction and splice code, in case a de-trending that respects the 2008 break is ever
  pre-registered as a separate trial. It is not being tried now.

Trial registry: one trial. See `TRIAL_REGISTRY.md`. Regression tests: `tests/test_research_fixes.py`.
