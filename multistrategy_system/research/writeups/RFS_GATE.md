# RFS newsy-month gate (2026-09-04, corrected 2026-09-05)

Research only. `rfs_gate.py` imports the frozen signal and regression from `rfs_newsy_month.py`
and adds trailing-only sizing, a century of free estimation history, walk-forward checks and
the book test. Nothing live was touched. Outputs are under `results/`.

## Corrections applied 2026-09-05

- The maximum-drawdown helper now counts a first-period loss (high-water mark starts at initial
  capital). The 1970-onward drawdown moves from −22.7% to −24.8%; other windows are unchanged.
- The book rows use the corrected trend baseline from `trend_follow_v3.py` (rebalance timing and
  lookback fixes), so the two-sleeve reference changed slightly. The gate outcome is unchanged.

## Frozen specification

- Signal: sum of market returns over the four most recent completed newsy months
  (January, April, July, October) minus its expanding mean, sign flipped in a newsy target month.
- Slope: expanding OLS of realised return on signal using strictly prior months, minimum 24.
- Raw position: slope × signal, in units of the market.
- Sizing (new, replaces the paper's ex-post volatility match): raw position × (12%/yr target ÷
  trailing 120-month volatility of the raw strategy through the prior month, minimum 36 months),
  clipped to ±1.0× the market.
- Cost: 1 bp × |change in position|.
- Data: Ken French monthly Mkt-RF + RF, 1926-07 to 2026-07 (the CRSP value-weighted total return,
  which is the paper's series). Positions are also realised on SPY monthly total returns from
  2007 as the implementable series.

## Pre-registered gate and outcome

| | Criterion | Result | |
|---|---|---|---|
| G1 | 1950–2026 walk-forward net Sharpe (no-rf, monthly) ≥ 0.30 | **0.44** | pass |
| G2 | Rolling 10-year net Sharpe positive in ≥ 70% of windows | **96%** of 800 windows | pass |
| G3 | Post-paper window 2021-07 onward net Sharpe > 0 (informational) | **0.80** on 61 months | pass |
| G4 | 2020–2025 monthly correlation with equity sleeve and with trend each < 0.30 | **+0.02, −0.03** | pass |
| G5 | 3-sleeve inverse-vol book 2020–2025: rf-Sharpe ≥ 2-sleeve, worst year not lower, MaxDD not deeper | rf-Sharpe 0.80 → 0.71, worst year +0.39 → −0.10, MaxDD −5.2% → −4.5% | **fail** |

Verdict under the pre-registered gate: **not promoted.** The verdict stands as registered; the
decomposition below is recorded so the next gate can be written properly, not to overturn this one.

## Results

Net Sharpe is no-rf (mean/std × √12) unless marked; "ex-RF" subtracts the actual monthly T-bill.

| Window | Months | Net Sharpe | Net ex-RF | Market Sharpe | Market ex-RF | MaxDD | Avg position | Avg abs position |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1950-01 onward | 919 | 0.44 | 0.05 | 0.82 | 0.55 | −25.5% | 0.01 | 0.51 |
| 1970-01 onward | 679 | 0.48 | 0.07 | 0.75 | 0.47 | −24.8% | 0.00 | 0.48 |
| 2000-01 onward | 319 | 0.50 | 0.32 | 0.60 | 0.48 | −17.7% | 0.00 | 0.48 |
| 2007-01 onward | 235 | 0.46 | 0.30 | 0.74 | 0.65 | −17.7% | 0.03 | 0.43 |
| Post-paper 2021-07 onward | 61 | 0.80 | 0.44 | 0.78 | 0.56 | −17.0% | 0.01 | 0.53 |
| 2020-01 to 2025-12 | 72 | 0.59 | 0.37 | 0.87 | 0.72 | −17.7% | 0.06 | 0.62 |

Implementable on SPY (positions from the French estimation, realised on SPY): 2007 onward
net Sharpe 0.46, post-paper 0.77, 2020–2025 0.55; correlation with SPY +0.08, −0.03 and +0.13.

Per-decade net Sharpe on the market: 1930s 0.80, 1940s 0.21, 1950s 0.41, 1960s 0.24, 1970s 0.17,
1980s 0.77, 1990s 0.51, 2000s 0.50, 2010s 0.57, 2020s 0.60. Every decade positive.

Per-year net Sharpe on SPY, 2007 onward: 2007 0.92, 2008 0.96, 2009 0.06, 2010 1.89, 2011 −0.83,
2012 0.96, 2013 0.57, 2014 0.77, 2015 0.35, 2016 −0.92, 2017 1.62, 2018 0.42, 2019 0.97, 2020 0.15,
2021 0.82, 2022 2.91, 2023 0.52, **2024 −2.71**, 2025 0.99, 2026 (5 months) 1.13.

Mean net monthly return since 1950: ordinary months +0.43%, newsy months +0.26%. Average
absolute position about half the market; long 51% of months; the cap binds in 39 of 235 months
since 2007; average monthly turnover 0.56× the market. Sensitivities logged as trials: cap 1.5
gives 0.45 long-run; 5 bp cost gives 0.41.

## Why G5 failed

| Book, 2020–2025 monthly | rf-Sharpe | no-rf | Mean/yr | Vol | MaxDD | Worst year |
|---|---:|---:|---:|---:|---:|---:|
| equity + trend | 0.80 | 1.52 | 10.6% | 7.0% | −5.2% | +0.39 |
| equity + trend + RFS | 0.71 | 1.53 | 9.3% | 6.1% | −4.5% | −0.10 |
| same, vol-matched to 7.0% (diagnostic only) | 0.82 | 1.53 | 10.8% | 7.0% | −5.2% | +0.04 |

Two separate things are inside the failure.

1. **A sizing artefact.** Inverse-vol weights give RFS a third of the book and the book's vol falls
   from 7.0% to 6.1%. The project's rf-Sharpe subtracts a flat 5%, so a lower-vol book is
   penalised even when its no-rf Sharpe rises and its drawdown shrinks. Scaled back to the same
   vol, the three-sleeve book has the higher rf-Sharpe (0.82) and the same drawdown. This is the
   flat-rf convention, not the signal.
2. **2024.** RFS lost money in ten of twelve months (−3.62 Sharpe on the year), which turned the
   book's 2024 from +1.20 to −0.10 and became its worst year. In a persistent bull market with
   strong earnings news, the reversal leg in newsy months keeps fighting the trend. That is a real
   property of the signal, visible in the long history as the 2011 and 2016 losses, and the
   vol-matched book still has a lower worst year (+0.04 vs +0.39) because of it.

Per-year rf-Sharpe, 2020–2025: equity+trend 0.68, 1.48, 0.39, 0.56, 1.20, 0.81; with RFS 0.47,
1.36, **1.63**, 0.39, **−0.10**, 0.69. RFS is the reason 2022 goes from modest to strongly positive
and the reason 2024 goes from strong to flat.

## Reading

The signal cleared every standalone test that could be written in advance: a century of
walk-forward history at the paper's Sharpe with honest sizing, positive in every decade, stronger
after the paper's data end, and uncorrelated with both existing sleeves. What it did not clear is
the book test as registered, and the registered test mixes a convention artefact with a genuine
weakness. The correct response is to record this outcome, fix the book convention before anything
else is gated, and let forward paper months add to the 61-month post-paper record.

## Next step for this candidate

- Run RFS in shadow on paper: compute and log the monthly SPY position without trading it.
- Re-gate under the corrected book convention when that convention is fixed, with the gate text
  written before the rerun. On these numbers a vol-targeted book test would pass on Sharpe and
  drawdown and fail narrowly on worst year.
- Do not size it in a live book while 2024-type years are unexplained.

Trial registry: three trials (frozen spec; cap 1.5; 5 bp). See `TRIAL_REGISTRY.md`.
