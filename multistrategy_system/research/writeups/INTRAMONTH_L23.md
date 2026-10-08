# Intramonth sector cycle, legs 2 and 3 only (2026-09-14)

Research: `intramonth_l23.py` (gate pre-registered in its docstring before the run). Live:
`intramonth_sleeve_live.py` now defaults to `INTRAMONTH_LEGS=23`. Registry T-2026-09-14-22.

**Question.** Does dropping leg 1 (ten-day sector momentum before month-end, D−10 to D−5) make
the sleeve more orthogonal to the equity and trend sleeves without losing the alpha?

**Answer.** Yes on every pre-registered criterion, including the blind one. Legs 2 and 3
(turn-of-month flow and its reversal) have a monthly correlation of 0.01 with the equity sleeve
and 0.03 with the trend sleeve, pay in the months the book loses, and lift the book on the
implementable convention. The caveat is that the series has lost money since January 2024, and
the paper record has to settle whether that is noise or decay.

## Mechanism, stated before the run

Leg 1 is a momentum exposure, so it overlaps the trend sleeve (its daily correlation with trend
is 0.12; legs 2 and 3 have none), and its premise, pre-month-end institutional liquidation, has
been arbitraged since 2012 (leg 1 Sharpe −0.11 in 2012–2019, −0.23 in 2020–2026). Legs 2 and 3
are flow effects nothing else in the book trades. Dropping leg 1 removes two of five order rounds
a month.

## Disclosure

Trial 21 on the same day looked at an approximation (leg-1 days zeroed in the all-legs series) on
the 2020–2025 book, so G4 was not blind. G2(a), the 2008–2019 test against the trend sleeve alone,
was not examined before this run and is the honest part of the evidence.

## Results at 1 bp

| | Legs 2+3 | All legs |
|---|---|---|
| Net Sharpe, no-rf, 1999–2026 | 0.64 (0 bp 0.79, 2 bp 0.49) | 0.58 |
| Annual return | +4.0% | +5.6% |
| Vol | 6.4% (16.9% on active days) | 10.4% |
| MaxDD | −14.0% | −22.9% |
| Turnover per month | 7.7× | 11.5× |
| Days in the market | 14% | 38% |

Per period, net Sharpe: 1999–2011 0.70 (leg 2 0.21, leg 3 0.70); 2012–2019 0.14 (0.11, 0.09);
2020–2026 0.96 (0.78, 0.62). Per year since 2020: +2.52, +2.34, +1.20, +1.58, −1.17, −0.28, and
−0.54 for 2026 to September.

## Gate

| | Criterion | Result | |
|---|---|---|---|
| G1 | Monthly corr with equity and with trend each < 0.10, 2020–2025 | +0.01, +0.03 | pass |
| G2 | Mean monthly return > 0 in the worst-quintile months of (a) trend alone 2008–2019, blind; (b) walk-forward equity+trend book 2020–2025 | (a) +0.84%, hit rate 66% of 29 months; (b) +0.67%, 67% of 15 | pass |
| G3 | Full-sample Sharpe ≥ 0.40; 2012 onward ≥ 0.30; legs 2 and 3 each positive in all three periods | 0.64; 0.58; yes | pass |
| G4 | Walk-forward book rf-Sharpe not lower, MaxDD not deeper | 0.59 → 0.74; −7.0% → −4.9% | pass |

Verdict as registered: promote legs 2+3 to replace the all-legs configuration in the paper sleeve.

## Sizing, which matters more than usual here

The sleeve is in the market one day in seven, so its full-series volatility (6.4%) understates
the risk it carries when it is on (16.9%). Inverse-vol weighting on the full series gives it 46%
of the book, which is the raw G4 number above. Sized on its active-day volatility instead, the
book still improves and does so without a losing year:

| Book 2020–2025, walk-forward | rf-Sharpe | no-rf | Vol | MaxDD | Worst year |
|---|---|---|---|---|---|
| equity + trend | 0.59 | 1.19 | 8.2% | −7.0% | 2022 +0.11 |
| + legs 2+3, sized on full-series vol (46%) | 0.74 | 1.66 | 5.4% | −4.9% | 2025 −0.29 |
| + legs 2+3, sized on active-day vol (28%) | 0.72 | 1.53 | 6.2% | −6.1% | 2022 +0.16 |
| + legs 2+3 at a fixed 20% | 0.65 | 1.39 | 6.7% | −6.2% | 2022 +0.15 |

Recommendation for any book convention: size this sleeve on active-day volatility, which puts it
near a quarter of the book. The live sleeve sizes by fixed notional and is unaffected.

## The caveat, quantified (2026-09-15)

From January 2024 to September 2026 the legs-2+3 series has a Sharpe of −0.69 and a cumulative
loss of −10.6% over 672 trading days; from its November 2023 peak it is down 12.3%, with a
maximum of −12.9%. Against the backtest's own distribution of 672-day windows before 2024:

| | Since 2024 | Backtest windows before 2024 (1,078) |
|---|---|---|
| Cumulative return | −10.6% | 5th percentile −3.1%; no window lost more; worst since 2012 was −7.1% |
| Sharpe | −0.69 | 0.7% of windows lower; gap to the pre-2024 Sharpe of 0.77 is 2.4 standard errors |
| Max drawdown | −12.9% | 5th percentile −14.0%, worst −14.0% |

The drawdown has not crossed the stopping rule's line (−14.0%), but the cumulative loss and the
Sharpe are outside anything the backtest produced. This is what decay looks like, not a bad draw.
The gate was pre-registered and passed because the 1999–2023 record dominates it; the earlier
version of this section called the stretch "within the historical range", which the table shows
is wrong on two of three measures. Corrected.

**How G3 passed anyway.** G3 asked for a Sharpe of at least 0.30 from 2012 onward and got 0.58.
That number is four years:

| Window | Sharpe | Cumulative |
|---|---|---|
| 2012–2019 | +0.14 | +4.2% |
| 2020–2023 | +1.92 | +67.4% |
| 2024–2026 | −0.69 | −10.6% |
| 2012–2026 excluding 2020–2023 | −0.12 | −6.9% |

2020–2023 supplied 120% of the post-2012 profit. A criterion on the average over a window whose
boundaries were fixed at 2012 and 2020 cannot see that, and the gate had no "most recent N
years" test. One would have failed: the last two years are −0.66 and the last three −0.50. That
is a defect in the gate's design, not in the data, and it is not repaired after the fact by
adding the criterion now. It is added to every future gate (see TRIAL_REGISTRY.md, lessons).
The reading of the evidence: outside 2020–2023 there is no post-2012 edge in this series. The
2020–2023 years were the high-volatility, high-flow years, which is a hypothesis about when
turn-of-month flow is large, and a post-hoc one.

**Stance.** Paper costs nothing, and the legs-2+3 configuration is better than all-legs on every
measure regardless, so the file ships. The sleeve is on probation and counts for nothing in any
book figure until the forward record says otherwise. Two pre-registered kill conditions, written
2026-09-15: (1) forward drawdown from the November 2023 peak reaches −14.0%, the backtest's worst;
(2) the Sharpe from 2024-01 onward is still below zero on 2027-09-15. Either closes the sleeve
and retires the candidate.

## Live change

`intramonth_sleeve_live.py`: new `INTRAMONTH_LEGS` setting, default `23`; a calendar role for
a disabled leg is mapped to flat before any order logic, so the reconciler closes anything left
over from that leg; the journal strategy label becomes `sector_intramonth_cycle_L23` and the state
file records the legs. Set `INTRAMONTH_LEGS=123` to restore the original. The sleeve now trades
on three days a month: D0 (enter), D+1 (flip), D+3 (exit). Test added in
`tests/test_live_sleeves.py`. This starts a fresh paper record; fills before the change are
labelled with the old strategy name.
