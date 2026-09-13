# Trend sleeve v3 research pass (2026-09-04, corrected 2026-09-05)

Research only. `trend_follow_v3.py` does not touch `trend_sleeve_live.py`, `trend_follow_v2.py`,
or any cached headline series. Outputs are under `results/`.

## Corrections applied 2026-09-05

An independent review found three implementation errors in the 2026-09-04 version. All are fixed
and every number below is from the corrected run; the pre-registered gate is unchanged.

1. **Rebalance timing.** Weights were stamped on calendar month-ends and forward-filled onto
   trading days, then shifted, so after a weekend month-end they activated one trading day late
   (July 2021 weights on August 3 instead of August 2). Now stamped on the last trading day of the
   month; the same fix applies to the trailing scalar. Inherited from `trend_follow_v2.py`.
2. **Lookback length.** `p.iloc[-h]` spans h−1 returns, so the advertised 63/126/252-day signals
   used 62/125/251 returns. Now `p.iloc[-h-1]`. Inherited from v2 and shared by the live sleeve
   (`trend_sleeve_live.py` line 361), so research and live are internally consistent; the live
   sleeve is not changed here.
3. **Maximum drawdown.** The helper started the high-water mark after the first return, so a
   first-period loss counted as zero drawdown. Now the mark starts at initial capital.

The REF rows keep the v2 conventions on purpose; the reimplementation check (legacy conventions)
still reproduces `trend_follow_v2` to machine precision. **The same drawdown helper is used by
`combine_book.py`, `combine_book_v2.py` and `trend_follow_v2.stats`, which produced the
PORTFOLIO.md headline drawdowns (−7.4% book, −14.9% equity, −22.1% trend). Those are understated
by construction and should be regenerated in the cost-consistent rebuild.**

## What was tested and why

Two published refinements of the validated cross-asset trend sleeve, plus removal of the
full-sample volatility-scalar leak, on the **live 16-ETF universe** (SECTOR minus SPY/QQQ/IWM),
shared cost model at the v2 defaults (2 bp fallback execution, no commission, borrow or financing).

| Variant | Definition | Source |
|---|---|---|
| REF_v2_18 | trend_follow_v2 as cached for the headline: 18 assets, full-sample scalar, v2 conventions | reference, not a trial |
| REF_v2_16 | same signals on the live 16, full-sample scalar, v2 conventions | reference, not a trial |
| **V1_trail** | v2 signals (mean sign over 63/126/252 days) with a **trailing** portfolio scalar | leak fix, **baseline** |
| V2a_MED | per asset w = ½(SLOW + FAST); SLOW = sign of mean of trailing 12 monthly returns, FAST = sign of last monthly return; turning-point assets get zero | Goulding, Harvey & Mazzoleni, JFE 149 (2023), Table 2 definitions |
| V2b_DYN | state-dependent speed: a_Bu = a_Be = ½, a_Co and a_Re from the paper's eqs (32)–(33), estimated on pooled vol-scaled prior data, expanding, clipped to [0,1] | same paper, Section 5 |
| V3a_63_126 | mean-sign ensemble over {63,126} | Quantica, "The Speed Factor" (Nov 2025) |
| V3b_126_252 | ensemble over {126,252} | speed test |
| V3c_84_126_168 | ensemble over {84,126,168} | speed test |
| V4_DYN_ENS | DYN state weighting with the 3/6/12 ensemble as the slow leg | pre-registered heuristic |

Trailing scalar: at each month-end, realized vol of the unscaled (per-asset vol-targeted, 1/n)
portfolio over the last 504 trading days (minimum 126), scalar = min(12% / realized, 3.0), 1.0
until enough history, applied to the following month.

Pre-registered go/no-go versus V1: beat V1 on full-sample rf-Sharpe **and** worst-calendar-year
rf-Sharpe, keep 2022 rf-Sharpe > 0, and keep 2020–2025 SPY correlation within +0.02 of V1.

## Results (corrected)

Full window 2008-01-01 to 2026-06-29; eval window 2020–2025; rf-Sharpe subtracts a flat 5%/yr.
Book = risk-parity combination with the cached 2020–2025 equity series, full-overlap inverse-vol
weights (the headline convention).

| Variant | Full rf | Full no-rf | Vol | MaxDD | Worst yr | SPY corr | 2020–25 rf | 2020–25 MDD | 2020–25 SPY corr | 2022 rf | Scalar mean | TO/yr | Book rf | Book worst | Book MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF_v2_18 | 0.18 | 0.59 | 12.2% | −22.7% | −1.95 | −0.15 | 0.37 | −22.7% | −0.19 | 0.87 | 2.92 | — | **0.60** | **−0.07** | **−7.4%** |
| REF_v2_16 | 0.16 | 0.58 | 12.1% | −22.2% | −1.87 | −0.21 | 0.42 | −22.2% | −0.22 | 1.11 | 2.95 | — | 0.63 | 0.20 | −6.8% |
| **V1_trail** | 0.19 | 0.63 | 11.3% | −20.4% | −1.87 | −0.13 | 0.45 | −20.4% | −0.13 | 1.35 | 2.77 | 11.7 | 0.66 | 0.43 | −6.6% |
| V2a_MED | −0.03 | 0.43 | 10.9% | −21.8% | −1.81 | −0.13 | 0.43 | −18.6% | −0.15 | 1.07 | 2.83 | 19.9 | 0.65 | 0.03 | −7.6% |
| V2b_DYN | 0.06 | 0.49 | 11.8% | −27.9% | −2.06 | +0.05 | 0.58 | −19.0% | **+0.20** | 0.24 | 2.71 | 18.1 | 0.72 | −0.69 | −7.9% |
| V3a_63_126 | 0.23 | 0.66 | 11.6% | −22.2% | −2.18 | −0.15 | 0.41 | −20.9% | −0.17 | 1.33 | 2.64 | 13.3 | 0.65 | −0.03 | −8.1% |
| V3b_126_252 | 0.17 | 0.61 | 11.5% | −20.6% | −2.08 | −0.08 | 0.43 | −20.6% | −0.02 | 1.30 | 2.72 | 10.3 | 0.64 | 0.36 | −6.7% |
| V3c_84_126_168 | 0.18 | 0.62 | 11.5% | −26.4% | −2.91 | −0.13 | 0.49 | −19.7% | −0.14 | 1.34 | 2.69 | 11.2 | 0.69 | 0.33 | −7.8% |
| V4_DYN_ENS | 0.22 | 0.65 | 11.6% | −22.1% | −1.59 | −0.07 | 0.50 | −17.6% | −0.08 | 0.37 | 2.77 | 18.9 | 0.69 | −0.54 | −7.4% |

REF_v2_18's book row reproduces the PORTFOLIO.md headline (0.60 / −0.07 / −7.4%) exactly, under
the legacy conventions.

Gate outcome: **no variant promoted. V1_trail ships alone.** Unchanged from the first run.

| Variant | Full rf > V1 | Worst yr > V1 | 2022 > 0 | SPY corr ok | Verdict |
|---|---|---|---|---|---|
| V2a_MED | fail | pass | pass | pass | no |
| V2b_DYN | fail | fail | pass | fail | no |
| V3a_63_126 | pass | fail | pass | pass | no |
| V3b_126_252 | fail | fail | pass | fail | no |
| V3c_84_126_168 | fail | fail | pass | pass | no |
| V4_DYN_ENS | pass | pass | pass | fail | no |

## Findings

1. **The scalar leak was not flattering the numbers.** With corrected conventions, V1_trail beats
   the legacy REF_v2_16 on every full-sample column (rf 0.19 vs 0.16, drawdown −20.4% vs −22.2%)
   and on the book row (0.66 / +0.43 / −6.6%). The trailing scalar sits at the 3.0 cap most of the
   time and is 3.00 today, so the live constant of 2.95 is within its range.
2. **Turning-point rules still break the crisis hedge on a multi-asset universe.** The pooled DYN
   estimate gives a_Re ≈ 0.9, so after any Rebound the rule re-longs almost fully. Applied per asset
   across bonds, commodities and FX, that flips the 2020–2025 SPY correlation to +0.20, cuts 2022
   from 1.35 to 0.24, and makes the book's worst year −0.69. The ensemble-based V4 is closer than
   before (it now passes three of four criteria) but still raises equity correlation and gives the
   book a −0.54 worst year, which is the diversification objection in a different form.
3. **Speed changes remain within noise, with the sign now slightly favouring the faster band on the
   full sample.** V3a's full-sample rf-Sharpe is 0.23 versus 0.19, but its worst year (2012, −2.18)
   and 2023–2025 are weaker, and V3c's worst year is −2.91. Differences of 0.04 on a 0.2 Sharpe over
   eighteen years are not evidence. Nothing is promoted; V3a and V4 are the two to re-examine as
   forward paper months accumulate.

## Caveats

- A flat 5% risk-free rate over 2008–2021 makes full-sample rf-Sharpe close to meaningless (0.19 rf
  versus 0.63 no-rf). Variant rankings are the same under no-rf.
- DYN speeds are estimated pooled across asset classes on vol-scaled returns; per-asset-class
  estimation was deliberately not run, to keep the trial count at seven.
- Book rows use full-overlap risk-parity weights; the walk-forward weighting task remains open.
- The cap binding most of the time means realized vol is below the 12% target (11.3%).

## What ships

- V1_trail (corrected conventions) is the research baseline for the trend sleeve;
  `results/trend_v3_returns.csv` column `V1_trail`.
- Recommended live change, not made here: replace the constant `PORTFOLIO_SCALAR = 2.95` in
  `trend_sleeve_live.py` with the trailing computation, and consider adopting the corrected
  lookback and rebalance-stamp conventions at the same time so live matches research. Until
  deployed, 2.95 is consistent with the trailing estimate and the live conventions match v2.
- Regenerate the headline caches with the corrected drawdown helper in the cost-consistent rebuild.

Trial registry: seven trials (V1, V2a, V2b, V3a, V3b, V3c, V4); see `TRIAL_REGISTRY.md`.
Regression tests for the fixes: `tests/test_research_fixes.py`.
