# Book backtest: all sleeves together (2026-09-14)

Research only. `book_backtest.py` combines the four sleeves' existing net daily return series and
reports the book under three conventions. It re-optimises nothing. Outputs under
`results/book_backtest_*.csv`; checks in `tests/test_book_backtest.py`.

**Question.** With the two paper candidates (sector intramonth cycle, RFS newsy-month) added to
the equity + trend book, is the book better?

**Answer.** Not on any implementable convention. Both additions cut the book's volatility and
neither raises its Sharpe once the weights are computed without look-ahead. Their gate results
were flattered by the headline full-overlap inverse-vol weighting, which uses the whole window's
volatility. The 2024 year is the problem: both candidates lost money that year while the equity
and trend sleeves made it, so the "no losing years" property of the current book goes away.
Nothing in this run changes a registered verdict; the paper sleeves keep running to build the
forward record that will settle it.

## Inputs

| Sleeve | Series | Costs already in the series |
|---|---|---|
| equity | engine-reported daily net returns, `equity_returns_2020..2025.csv` | engine cost model |
| trend | `results/trend_v3_returns.csv` column V1_trail (research baseline, corrected conventions, trailing scalar) | 2 bp |
| intramonth | `results/intramonth_sector_daily.csv` Strat2 net | 1 bp |
| rfs | monthly position from `results/rfs_gate_monthly.csv` applied to SPY daily returns, 1 bp on the month-start change (live trades VOO, same exposure) | 1 bp |

Two checks run first and both pass: the equity + trend headline book reproduces the TREND_V3.md
V1_trail line (rf 0.66, worst year 0.43, MDD −6.6%), and the RFS daily reconstruction matches the
gate's monthly net series (correlation 0.998, mean absolute difference 6 bp per month, which is
daily compounding versus monthly product).

## Conventions

- **RP, headline.** Inverse-vol weights from the whole window's volatility. This is the
  PORTFOLIO.md convention and the one the intramonth and RFS gates used for their book criterion.
  It describes a book; nobody could have run it, because the weights use future volatility.
- **WF, walk-forward.** Inverse-vol weights re-estimated at each month-end from each sleeve's
  trailing 252 days (pre-window history counts where it exists), applied from the next trading
  day; equal weight until every sleeve has 63 days. Implementable. Covers the same dates as RP.
- **WF vol-matched.** WF levered or delevered to the equity + trend RP book's volatility (8.3%).
  Diagnostic only: the flat-5% rf convention penalises a lower-vol book, so this shows what the
  additions do at equal risk.
- **EW.** Equal weight, reference.

Sharpe: rf (flat 5%/yr) headline, no-rf labelled. Drawdown mark at initial capital.

## Window 1: 2020–2025, all four sleeves (1,477 days)

Daily correlations are all below 0.10 in absolute value; the additions do diversify.

Standalone:

| Sleeve | rf-Sharpe | no-rf | Vol | MDD | Worst year (rf-Sharpe) |
|---|---|---|---|---|---|
| equity | 0.54 | 0.96 | 11.7% | −14.9% | 2022 −1.28 |
| trend | 0.41 | 0.84 | 11.7% | −19.3% | 2023 −1.80 |
| intramonth | −0.03 | 0.43 | 10.9% | −19.5% | 2024 −1.37 |
| rfs | 0.13 | 0.49 | 13.9% | −21.0% | 2024 −2.58 |

Books, walk-forward (implementable):

| Book | rf-Sharpe | no-rf | Vol | MDD | Worst year | Vol-matched rf-Sharpe | Vol-matched MDD |
|---|---|---|---|---|---|---|---|
| equity + trend (current) | **0.59** | 1.19 | 8.2% | −7.0% | 2022 +0.11 | 0.59 | −7.0% |
| + intramonth | 0.43 | 1.17 | 6.8% | −6.2% | 2025 −0.12 | 0.56 | −7.6% |
| + rfs | 0.47 | 1.15 | 7.3% | −7.3% | 2024 −0.35 | 0.55 | −8.3% |
| + both | 0.39 | 1.21 | 6.1% | −7.0% | 2024 −0.88 | 0.61 | −9.5% |

Same books, headline RP (the gates' convention): equity + trend 0.66 / 1.27; + intramonth
0.50 / 1.24; + rfs 0.60 / 1.29; + both 0.51 / 1.32. The headline convention is 0.07 rf-Sharpe
above the implementable one on the current book, and it is the only convention under which
adding both candidates raises no-rf Sharpe by more than rounding.

Reading:

1. Under WF the no-rf Sharpe is flat (1.19 → 1.17 / 1.15 / 1.21): the additions bring
   diversification and about the same return per unit of risk as what is already there, so
   they neither help nor hurt on that measure. Under rf they hurt because the book's vol falls
   from 8.2% to 6.1% with nothing to show for it.
2. Vol-matched, adding both gives 0.61 against 0.59, for a deeper drawdown (−9.5% vs −7.0%) and
   a losing 2024. That is not a trade worth making on this evidence.
3. The worst-year column is the one to watch. The current book's worst year is a positive 0.11.
   Every addition makes it negative, and with both added 2024 is −0.88: intramonth and RFS both
   lost that year, and their daily correlation of −0.09 did not stop them losing together.

## Window 2: 2008–2026, the three diversifiers without equity (4,648 days)

| Book (WF) | rf-Sharpe | no-rf | Vol | MDD | Worst year |
|---|---|---|---|---|---|
| trend alone | 0.19 | 0.63 | 11.3% | −20.4% | 2023 −1.87 |
| trend + intramonth | 0.06 | 0.73 | 7.5% | −13.5% | 2012 −2.54 |
| trend + rfs | −0.04 | 0.61 | 7.7% | −15.8% | 2016 −2.29 |
| trend + intramonth + rfs | −0.04 | 0.75 | 6.3% | −9.4% | 2016 −2.90 |

Over the long history intramonth helps on no-rf and drawdown (the front-loaded years before 2012
are in this window); RFS does not help on any measure, and its 2016–2017 stretch is the worst in
the file. Standalone over 2008–2026: intramonth rf 0.02 / no-rf 0.50, RFS rf −0.07 / no-rf 0.33.

## What this changes

- **A convention finding for the headline rebuild.** The PORTFOLIO.md book Sharpe uses
  full-overlap inverse-vol weights. The implementable equivalent is about 0.07 lower on the
  current book (0.59 vs 0.66 with V1_trail). The cost-consistent rebuild should report the
  walk-forward number.
- **No verdict changes.** The intramonth gate (G4) and RFS gate (G5) were scored on the headline
  convention as pre-registered; re-scoring them on WF would be changing the rule after seeing
  the result. Recorded here: on WF neither would have passed the "book not worse" criterion.
  Future book criteria should be written on the WF convention.
- **The paper sleeves keep running.** The forward record is the test that does not depend on
  which convention flattered what. The stopping rule in LIVE_SLEEVES.md stands.
- **If a decision had to be made today** on this evidence alone, the answer would be to keep the
  intramonth sleeve at reduced size (it improves drawdown in both windows and its no-rf effect is
  neutral) and to not fund RFS beyond paper. Not a decision; a reading.

## Caveats

- 2020–2025 is six years and 2024 is one of them; the worst-year conclusion rests on one year.
- Equity is the engine's own reported net series; the other three are research series with
  modelled costs. Paper fills will say whether 1 bp holds for the intramonth sleeve's MOC orders.
- RFS is reconstructed on SPY; the live instrument is VOO. Same exposure, immaterial difference.
- No book-level leverage cost is modelled in the vol-matched diagnostic.

## Registry

Logged as T-2026-09-14-19 (book combinations under the walk-forward convention). It is an
examination of registered trials in combination, not a new signal.
