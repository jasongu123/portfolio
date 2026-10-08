# Equity engine on twelve names it has never seen (2026-09-19)

Research only: `equity_breadth.py`, gate pre-registered in its docstring before any backtest ran;
`ibkr2.py` untouched. Registry T-2026-09-16-47 (NEW12), -48 (COMBO24). Data: seven years of
IBKR minute and daily bars for the twelve names (`backtest_data/{SYM}_1m.csv`, `_daily.csv`).
Yearly caches `equity_breadth_returns_{2020..2025}.csv`; series in `results/equity_breadth_daily.csv`.

**Question.** Is the intraday engine's edge a property of the method or of the twelve names it
was built on?

**Answer: the method.** On twelve names chosen by rule before any data was pulled, with every
parameter, feature and the ML gate unchanged, the engine earned a higher Sharpe than on its own
names. It still fails the gate as registered, on recency and on book drawdown, so nothing
changes in the live system today. This is the strongest single piece of evidence for the
engine in the project, because there was no selection on these names at all.

## Universe and setup

NEW12 = the largest company by market cap in each GICS sector not already traded, in the index
since mid-2019, plus the next largest overall: AMZN AVGO LLY BAC COST CVX CAT NFLX LIN NEE PLD
MU. Configuration identical to the validated book's (hybrid stock strategies, gap 1.5% / 60 min,
stock vol regime, RF gate on the 33 features, $0.0035 a share plus 1 bp), 2020-01 to 2025-12-15.
ORIG12 is the regenerated headline equity series. COMBO24 is the equal-weight of the two books.

## Results, 2020–2025 (no-rf daily Sharpe unless stated; rf = flat 5%)

| Series | no-rf | rf | Vol | MaxDD | Worst year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEW12 | **1.11** | 0.73 | 13.0% | −19.0% | −0.42 (2025) | 1.03 | 2.85 | 0.51 | 1.75 | 1.64 | −0.42 |
| ORIG12 | 0.91 | 0.44 | 10.4% | −14.9% | −0.83 (2022) | 0.74 | 2.05 | −0.83 | 1.89 | 1.35 | 1.06 |
| COMBO24 | 1.13 | 0.66 | 10.6% | −12.7% | −0.08 (2022) | 0.96 | 2.74 | −0.08 | 2.00 | 1.69 | 0.22 |

Daily correlation between the two books 0.63 over the window; by year 0.73, 0.65, 0.72, 0.65,
0.60, 0.25. The new names made money in 2022 (+0.51) when the original names lost (−0.83), and
the 24-name combination's worst year is −0.08 instead of −0.83.

## Gate

| | Criterion | Result | |
|---|---|---|---|
| Q1 | NEW12 no-rf ≥ 0.50, positive in ≥ 4 of 6 years | 1.11; 5 of 6 | pass |
| Q2 | 2024-09 onward > 0 | −0.61 | **fail** |
| Q3 | No 24-month block > 75% of profit | 51% | pass |
| Q4 | Corr < 0.70 and COMBO24 ≥ ORIG12 | 0.63; 1.13 vs 0.91 | pass |
| Q5 | Walk-forward book with trend not worse on rf-Sharpe, worst year, MaxDD | 0.54 → 0.70; −0.02 → +0.17; **−7.4% → −9.6%** | **fail** (drawdown) |

Verdict as registered: not adopted.

## What the two failures are

**Q2.** From September 2024 the new names ran at −0.61, cumulative −10%, while the original
names ran at +0.59. The damage is concentrated: March and April 2025 were −4.8% and −8.2%, the
tariff shock, and the second half of 2025 was positive again (+0.18). The recency criterion
exists precisely so that a strong average cannot hide a recent break, and it did its job; but
fifteen months, one of them the worst equity shock since 2020, is also the kind of stretch the
original names had in 2022, from which they recovered. The correlation between the two books
fell to 0.25 in 2025, which says the two sets stopped behaving alike in the year that decides
the criterion.

**Q5.** The book's rf-Sharpe rises from 0.54 to 0.70 and its worst year from −0.02 to +0.17;
the criterion failed on drawdown alone, −9.6% against −7.4%, with the trough on 2025-04-21.
That is the same April 2025 episode.

## Reading

1. The edge generalises. A method that earns 1.11 on names it has never seen, after a 0.91 on
   the names it was tuned on, is not fitted to those names. This should change how the engine
   is described: not "twelve stocks that work" but "a method that works on large liquid names".
2. Breadth would have paid over the six years: the 24-name book has the same Sharpe with a
   third less drawdown and no losing year worse than −0.08, because the two sets hedged each
   other in 2022.
3. The recent fifteen months are the open question, and they are exactly what a forward paper
   record answers. The registered decision rule does not allow adoption from this backtest;
   it does allow a pre-registered forward test.

## Proposed next step (not done; a live-file change, so it is the user's call)

Add NEW12 to the paper engine's symbol list as a separately labelled group, so the execution
journal attributes fills by symbol, and pre-register the forward criterion now: after six
months of paper, NEW12's net Sharpe must be positive and its correlation with ORIG12 below 0.70,
at which point Q2 is re-evaluated on data that did not exist when this doc was written. Costs
in paper carry no information; the modelled 1 bp applies. Data feed: the twelve names need
minute bars on the VM, which the engine already fetches for its symbols.

## Caveats

- Minute bars begin October 2019, so the 2020 backtest year runs on about two months of warmup
  for the ML gate, as the original 2020 did.
- 2 bp cost sensitivity was not run; the original names fail at 2 bp and the new names are of
  similar liquidity, so the same fragility should be assumed until measured.
- Six years, one universe draw. A second draw (the next-largest name per sector) would be the
  cheap replication if the forward record is ambiguous.

## Re-verified on the corrected engine (2026-09-25)

The numbers above used the backtester's same-day VIX close (VIX_LOOKAHEAD_2026-09.md). Re-run on
the per-minute VIX gate that the live engine uses, same universe, same configuration; the
same-day caches are kept as `legacy_caches/equity_breadth_returns_sameday_vix_{y}.csv`.

| Series | no-rf | rf | Vol | MaxDD | Worst year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEW12 | **1.02** | 0.64 | 13.2% | −18.5% | −0.24 (2025) | 0.88 | 2.68 | 0.49 | 1.70 | 1.40 | −0.24 |
| ORIG12 (headline equity sleeve) | 0.88 | 0.40 | 10.4% | −17.1% | −0.99 (2022) | 0.80 | 1.98 | −0.99 | 1.81 | 1.31 | 1.13 |
| COMBO24 | 1.05 | 0.59 | 10.7% | −12.6% | −0.18 (2022) | 0.91 | 2.61 | −0.18 | 1.93 | 1.52 | 0.32 |

Correlation 0.64. Gate unchanged: Q1, Q3, Q4 pass; Q2 fails (2024-09 onward −0.55) and Q5 fails
on drawdown (book −7.3% → −9.6%, with rf 0.51 → 0.65 and worst year −0.16 → +0.17). The
conclusion holds: the edge carries to names the engine never saw.

## Live change and forward test (2026-09-25, T-2026-09-25-49)

The user chose to add the twelve names to the paper engine as a forward test. The backtest
verdict is not overridden: the headline stays 0.51 / 1.13 until the forward test passes.

What changed in `ibkr2.py` (vm_sync_2026-09-25.zip):
- `EQUITY_BREADTH_SYMBOLS` added as allocation group `equity_breadth`; the core twelve are
  `CORE_EQUITY_SYMBOLS`. `EQUITY_BREADTH_ENABLED=false` restores the original exactly.
- Same total capital: each group's inverse-vol shares (rounded as before) are scaled by its
  budget, `EQUITY_BREADTH_WEIGHT` (0.5) for the new group and the rest for the core. That is the
  tested COMBO24. The two sizing paths that ignored the allocation are scaled by the same budget:
  the mean-reversion floor (10% of account → 5% at 0.5) and swing sizing (40% of capital →
  40% of the group's half). Without that, twelve more names would have roughly doubled MR and
  swing exposure. The backtester sizes every trade by allocation and has no MR floor, so the
  live floor was already a live/backtest difference; scaling it keeps it from growing.
- Entries in a breadth name need a trained RF model; until then the name stays flat (exits
  are never blocked). Untrained symbols otherwise pass entries ungated, and the VM needs about
  five hours to fetch history and train the new names on first start.
- Fills in breadth names are journalled with sleeve `equity_breadth`.
- The disabled tech sleeve refuses to construct while AVGO and MU are equity names; it logs
  "TechSleeve unavailable" at startup.

Forward test, registered before any forward data existed:
- window: 126 NYSE sessions from the first session with all twelve breadth models trained
- F1 breadth net no-rf Sharpe > 0; F2 correlation with the core's daily P&L < 0.70
- F3 early stop at any time: breadth drawdown on its capital beyond the corrected NEW12
  backtest's 5th-percentile drawdown for the elapsed sessions
  (`results/breadth_forward_thresholds.csv`: 21 sessions −6.8%, 63 −10.9%, 126 −15.2%)
- scoring: `breadth_forward_report.py --start <date> --account-value <NetLiq>`, realized P&L
  from matched journal fills, charged $0.0035 a share + 1 bp (paper fills carry no cost)
- on PASS: re-score Q2 on the forward data and add the group to the headline with the forward
  record disclosed; on FAIL or STOP: `EQUITY_BREADTH_ENABLED=false`
