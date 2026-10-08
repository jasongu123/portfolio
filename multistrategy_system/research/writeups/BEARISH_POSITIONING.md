# Bearish-positioning composite, "follow the hedging" (2026-09-04, corrected 2026-09-05)

Research only. `bearish_positioning.py`. Nothing live was touched. This was an original
recombination, not a paper replication: the stock-level short-interest index died because
bearish positioning migrated to index products after 2008, so this composite measured it where
it now lives and asked whether it times the market. Gate and mechanism prediction were written
before any composite result was seen.

## Corrections applied 2026-09-05

1. **CFTC publication dates.** The first run assumed every Commitments of Traders report was
   public three days after its as-of date. During the 2013, 2018–19 and 2025 government shutdowns
   reports were published weeks late (the 28 October 2025 report on 5 December 2025), so several
   month-end signals used data that did not yet exist. Actual publication dates from CFTC press
   releases 6745-13, 7864-19 and 9138-25 are now used; a month whose latest report had not been
   published by month-end carries the last published one.
2. **Holdout slope** is now estimated on SPY excess returns over the same 53 realised months as
   the reported performance, instead of on the French market through a later month.
3. **Maximum drawdown** now counts a first-period loss.

All numbers below are from the corrected run. The verdicts are unchanged.

## Components (all free, all to the present)

| Leg | Series | Sign in composite | History |
|---|---|---|---|
| SKEW | CBOE SKEW index, monthly mean | + (more tail-hedge demand = bearish) | 1990– |
| SPX COT | CFTC non-commercial net long ÷ open interest, S&P 500 futures (big + E-mini, E-mini scaled 1/5) | − (speculators long = bullish) | 1997– |
| VIX COT | Same for VIX futures | + (speculators long vol = bearish) | 2004– |
| Equity put/call | CBOE equity P/C, monthly mean | informational only; CBOE's free file ends 2019-10 | 2003–2019 |

Each leg is z-scored on an expanding window (minimum 36 months), signed, and averaged. Trial A is
the three-leg composite (OOS from 2012-07), Trial B drops the VIX leg for a longer window (OOS
from 2005-01). The rule is the expanding mean-variance timing rule used for the short-interest
test, with the sign estimated from data.

## Gate and outcome

| | Criterion | Trial A | Trial B |
|---|---|---|---|
| G1, binding | OOS window to 2021-12: OOS R² > 0, timing Sharpe > market, MaxDD not deeper | **fail**: R² −1.83%, Sharpe 0.88 vs 1.20, MaxDD −30.0% vs −20.2% (114 months) | **fail**: R² −0.39%, Sharpe 0.49 vs 0.77 (204 months) |
| G2, informational | Holdout 2022–2026 on SPY: same sign as 2021, Sharpe within 0.10 of SPY | pass: slope +0.42% (t 0.5), 0.73 vs 0.81 | fail: sign flipped (−0.15%, t −0.2), 0.69 vs 0.81 |
| G3, binding | Active overlay (rule minus market) net Sharpe > 0; 2020–25 correlation with equity and trend sleeves < 0.30 | **fail**: Sharpe 0.00, equity correlation +0.37 | **fail**: Sharpe −0.38 |
| G4, binding | Book 2020–2025 with the active overlay: no-rf Sharpe not lower, MaxDD not deeper | **fail**: 1.52 → 1.23, MaxDD −5.2% → −5.4%, worst year +0.39 → −1.56 | **fail**: 1.52 → 1.08, MaxDD −6.3% |

Verdict: **not promoted**, both trials. Two trials logged.

## What the data said

- Univariate slopes on next-month market excess return, full sample: SKEW +0.07% per s.d. (t 0.4),
  SPX positioning +0.18% (t 0.9), VIX positioning +0.71% (t 1.4), equity put/call −0.19% (t −0.4).
  Nothing is individually significant.
- The mechanism prediction resolved on the fear-gauge side: the composite's estimated sign is
  positive, meaning index hedgers behave as uninformed insurance buyers rather than as the
  informed shorts the stock-level index captured. The effect is too weak to forecast with: out of
  sample the regression forecasts are worse than the historical mean in both trials, and the rule
  ends up as a noisy leverage dial around 1× that adds drawdown.
- Using true publication dates moved the numbers slightly in the composite's favour (Trial A's
  OOS R² from −3.1% to −1.8%) and changed no conclusion.

## Why this closes the line of thought

The stock-level index worked, when it worked, because single-stock short sellers are informed.
Following the hedging to index products replaces informed traders with insurance buyers, and
insurance buying is a weak contrarian signal that the options-and-volatility literature has
already mined. There is no version of "where did the shorts go" that recovers the original
mechanism, because the mechanism was who was doing the shorting, not how much shorting there was.

Retained: the parsed CFTC positioning history (`results/cot_sp_vix_weekly.csv`, 1997–2026, S&P
and VIX futures) with the publication-date table in the script, and the CBOE SKEW, VIX and
put/call files under `backtest_data/`.

Trial registry: two trials. See `TRIAL_REGISTRY.md`. Regression tests: `tests/test_research_fixes.py`.
