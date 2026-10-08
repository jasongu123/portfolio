# Intramonth legs 2+3: T+1 settlement timing trial and leg diagnostic (2026-09-15)

Research only: `intramonth_t1.py`, gate pre-registered in its docstring. Registry
T-2026-09-15-23 (+1 shift) and T-2026-09-15-24 (−1 shift, control). Output
`results/intramonth_t1_daily.csv`.

**Hypothesis.** T+1 settlement (2024-05-28) lets month-end sellers act a day later, so the
turn-of-month cycle should have moved one trading day later. Prediction: the +1 shift beats the
original after the break; the −1 shift is the control; the original must beat the +1 shift before
the break.

**Result: rejected.** The +1 shift is worse after the break, not better.

| Shift | Pre-break Sharpe (1999–May 2024) | Post-break Sharpe (May 2024–Sep 2026, 84 active days) | Last two years | Full sample |
|---|---|---|---|---|
| −1 (control) | −0.13 | +0.46 | +0.36 | −0.09 |
| 0 (registered rule) | 0.74 | −0.65 | −0.65 | 0.64 |
| +1 (predicted) | −0.38 | −1.03 | −1.16 | −0.43 |

Gate: H1 fail (gap −0.38, needed +0.50), H2 pass, H3 fail, H4 pass, H5 fail. Not adopted.

**The control is the only positive line, and it is not adopted either.** The −1 shift's
post-break Sharpe of +0.46 rests on 84 active days (standard error about 0.65), it is negative
over the full 24 years and lost 22% before the break, and it is the opposite of the mechanism
that was written down. Adopting it would be selecting the best of three configurations on the
only window where it works, which is the window it was found on. It is registered as examined
and rejected. If a mechanism for an *earlier* cycle is proposed later, it needs a blind test
that is not 2024–2026, and there is none available in this series.

**Leg diagnostic (not a trial).** Both legs made money only in 2020–2023 and both lose since
2024; leg 2 (the turn-of-month entry) died harder.

| Net Sharpe, shift 0 | Leg 2 | Leg 3 |
|---|---|---|
| 2012–2019 | +0.11 | +0.09 |
| 2020–2023 | +1.77 | +1.07 |
| 2024-01 to latest | −0.79 | −0.24 |

**Reading.** Nothing in the timing of the rule explains the break, and no leg survived it. On
the recency standard (the edge has to be working now), the sector-SPDR intramonth cycle is not
working, in any of its legs, at any one-day timing. The candidates left are conditioning on flow
intensity (blind-testable on 1999–2019) and replication on less crowded instruments, which is
its own out-of-sample test and the one that would say whether the effect is dead or merely
crowded out of the nine SPDRs. The paper sleeve stays on probation; it counts for nothing.
