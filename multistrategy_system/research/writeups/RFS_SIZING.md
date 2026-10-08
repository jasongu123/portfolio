# RFS in the book: hedge-based sizing (2026-09-15)

Research only: `rfs_sizing.py`, gate pre-registered in its docstring. Registry T-2026-09-15-29.
Output `results/rfs_sizing_daily.csv`.

**Question.** Sized as a hedge on its bad-month payoff rather than by inverse volatility, does
RFS earn a place in the equity + trend book?

**Answer as registered: no.** The sizing rule, derived only from 2008–2019, produced a weight of
0.70, and at that weight the 2020–2025 book is much worse. The rule was badly designed and the
gate is recorded as failed. A fixed small weight would have looked better, but that is a sweep
read after the fact, and it is not promoted.

## The rule and why it produced 0.70

In the trend sleeve's worst-quintile months of 2008–2019 (29 months) the sleeve lost 3.91% a
month and RFS made 0.41% with a 66% hit rate. The rule asked RFS to offset a quarter of that loss,
which needs w = 0.25 × 3.91 / (0.41 + 0.25 × 3.91) = 0.70. The hedge payoff is real but an order
of magnitude too small to offset a trend drawdown at any sane weight; the target of a quarter
was the design error, and it was fixed before the run, so it stands.

## Out of sample, 2020–2025, walk-forward equity + trend book

| Book | rf-Sharpe | no-rf | Vol | MaxDD | Worst year |
|---|---|---|---|---|---|
| equity + trend | 0.59 | 1.19 | 8.2% | −7.0% | 2022 +0.11 |
| + RFS at the derived w = 0.70 | 0.27 | 0.76 | 10.1% | −12.9% | 2024 −2.21 |
| walk-forward inverse-vol (BOOK_BACKTEST.md) | 0.47 | 1.15 | 7.3% | −7.3% | 2024 −0.35 |

RFS did what the mechanism says in the bad months (+1.03% a month in the book's 15 worst months,
hit rate 60%, against a book loss of −1.77%), and 2022 was +1.85 with it. It also delivered a
−2.21 year in 2024 at that weight.

## Gate

| | Criterion | Result | |
|---|---|---|---|
| S1 | In-sample hedge value: RFS > 0 in the trend sleeve's bad months, hit rate ≥ 50% | +0.41%, 66% | pass |
| S2 | Out-of-sample book at the derived weight: rf-Sharpe, worst year and MaxDD not worse | 0.59 → 0.27; +0.11 → −2.21; −7.0% → −12.9% | **fail** |
| S3 | RFS recency: last two years > 0; best 4-year block ≤ 75% | +0.24; 73% | pass |

## The sweep, disclosed and not acted on

Fixed fractions on the same out-of-sample window (a sweep, read after the gate):

| w | rf-Sharpe | no-rf | MaxDD | Worst year |
|---|---|---|---|---|
| 0 | 0.59 | 1.19 | −7.0% | +0.11 |
| 0.05 | 0.59 | 1.23 | −6.8% | +0.30 |
| 0.10 | 0.60 | 1.26 | −6.6% | +0.37 |
| 0.15 | 0.59 | 1.27 | −6.5% | +0.36 |
| 0.20 | 0.58 | 1.28 | −6.6% | +0.34 |
| 0.30 | 0.54 | 1.24 | −6.9% | −0.11 |

At 10–15% of the book, RFS raises no-rf Sharpe by about 0.07, shallows the drawdown slightly,
and lifts the worst year from +0.11 to +0.37, with rf-Sharpe unchanged. That is the size of the
prize: small, and consistent with the quadrature arithmetic for a 0.5-Sharpe uncorrelated
sleeve. It is also the best point of a seven-point sweep on six years, so it is not evidence of
the kind the gate accepts. If a 10% weight is ever adopted, the justification has to be a
forward paper record at that size, not this table.

## Decision

RFS stays a paper sleeve at zero book weight, as registered. Its current paper size (0.5 × a
150k budget) is fine for building the record. Lesson recorded: a hedge-sizing rule needs an a
priori cap on the weight; a rule that can output 0.70 for a 0.5-Sharpe sleeve is not a sizing
rule.
