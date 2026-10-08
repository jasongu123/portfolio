# Trend sleeve breadth: 16 ETFs to 46 (2026-09-16)

Research only: `trend_breadth.py`, gate pre-registered in its docstring; the live sleeve is
unchanged. Registry T-2026-09-16-43 (46-ETF universe), -44 (asset-class blocks and drop-one as
one diagnostic entry). Data `backtest_data/trend_breadth_prices.csv` (IBKR, 30 ETFs, 20 years).
Outputs `results/trend_breadth_returns.csv`, `_summary.csv`.

**Hypothesis.** Institutional trend programs earn their edge from breadth across 60–100 markets.
Adding 30 ETFs (grains, industrial and precious metals, gas and refined products, credit and
inflation-linked and international bonds, four currencies, nine single-country equity funds,
international real estate) to the 16-ETF sleeve with the same rule should raise its Sharpe
through more independent bets.

**Answer: no, under equal weighting it lowers it.** Effective independent bets more than
doubled and the Sharpe fell. Every asset-class block reduced the sleeve on its own.

## Results, full window 2008–2026 (rf = flat 5%)

| Configuration | rf-Sharpe | no-rf | Vol | MaxDD | Worst year | Last 2y | Effective bets |
|---|---|---|---|---|---|---|---|
| BASE16, the live universe | 0.18 | 0.63 | 11.2% | −20.4% | −1.87 (2023) | 0.23 | 12.0 of 16 |
| BREADTH46 | −0.05 | 0.43 | 10.6% | −20.7% | −1.57 (2023) | 0.05 | 27.1 of 46 |
| BASE16 + commodities (9) | 0.08 | 0.56 | | −22.9% | | | |
| BASE16 + rates and credit (7) | 0.15 | 0.59 | | −20.4% | | | |
| BASE16 + currencies (4) | −0.04 | 0.41 | | −23.3% | | | |
| BASE16 + equities (9) | 0.08 | 0.50 | | −24.3% | | | |
| BASE16 + real estate (1) | 0.15 | 0.60 | | −21.0% | | | |
| BREADTH46 minus LQD | −0.07 | 0.40 | | | | | |
| BREADTH46, thin set at 10 bp | −0.06 | 0.41 | | | | | |

Sub-periods, BREADTH46 minus BASE16 rf-Sharpe: 2008–2013 −0.08, 2014–2019 −0.29, 2020–2026 −0.31.
Walk-forward book with equity, 2020–2025: rf 0.55 → 0.37, worst year +0.10 → −0.39.

| | Criterion | Result | |
|---|---|---|---|
| B1 | +0.10 rf-Sharpe, drawdown not deeper | −0.23 | fail |
| B2 | Improvement in two of three sub-periods, last two years | 0 of 3; −0.17 | fail |
| B3 | Effective bets +25%, drop-one keeps half the gain | +126%; no gain to keep | fail |
| B4 | Walk-forward book not worse | 0.55 → 0.37 | fail |
| B5 | Holds with thin set at 10 bp | −0.06 | fail |

## Why breadth hurt: dilution, not correlation

Per-instrument no-rf Sharpe of the rule's own P&L in each instrument, 2008 onward:

| Block | n | Mean Sharpe | Share positive | Contribution /yr in the 46-book |
|---|---|---|---|---|
| Base 16 | 16 | +0.36 | 100% | +2.54% |
| Commodities added | 9 | +0.16 | 78% | +0.71% |
| Rates and credit added | 7 | +0.21 | 100% | +1.03% |
| Currencies added | 4 | −0.15 | 25% | −0.37% |
| Equities added | 9 | +0.18 | 100% | +0.95% |
| Real estate added | 1 | +0.02 | 100% | +0.01% |

The best additions are as good as the best base names: LQD 0.46, HYG 0.41, TIP 0.36, EWT and
EWY 0.35, CORN and BNO 0.34. The worst are the currency funds (FXF −0.25, FXA −0.21, FXC −0.19)
and platinum. Under equal weighting each base name falls from a sixteenth of the book to a
forty-sixth, and the thirty slots that replace that weight earn about half as much per slot
(0.078% a year per slot against 0.159% for the base), so the book's expected return per unit of
risk falls even though the P&L streams are more numerous and less correlated. The
effective-bets diagnostic is real and the diversification is real; the average quality of what
was added is not high enough for equal weighting to survive it.

Why ETF breadth is not futures breadth: the thin commodity funds carry 0.6–1.1% expense ratios
and roll drag; currency ETFs are low-volatility products levered to the cap by the vol target;
and the nine country funds are largely the same bet as EFA and EEM, already held. Institutional
breadth works on futures with no wrapper cost and genuinely distinct markets.

## What would be a legitimate follow-up (not run, not registered)

The diagnostic points at the weighting, not the instruments: equal weight across 46 names lets
nine correlated country funds and four losing currency funds crowd out the base. A
sector-balanced construction, equal risk per asset class and equal within, is how trend
programs handle breadth and it would keep the base's balance while adding credit, TIPS and the
better commodity funds within their classes. That is a design change with a mechanism, it comes
from this diagnostic rather than from a prior, and it would be registered as its own trial with
the same gate before any number is looked at. The currency additions would not be part of it.

## Consequence

Not adopted. The live sleeve keeps its 16 ETFs. Data for the 30 additions is cached.

## Follow-up: sector-balanced weighting, SB42 (2026-09-16, T-2026-09-16-45 / -46)

`trend_breadth_sb.py`, gate pre-registered; the hypothesis came from the diagnostic above and is
registered as a second look at the same data. Universe: the live 16 plus the additions minus the
currency block (42). Weighting: equal risk per asset class (Equity, Rates, Credit, Commodity, FX,
RealEstate), equal within. Attribution row SB16 applies the same weighting to the base alone.

| Configuration | rf-Sharpe | no-rf | MaxDD | Worst year | Last 2y | 2008–2013 | 2014–2019 | 2020–2026 |
|---|---|---|---|---|---|---|---|---|
| BASE16, live | 0.18 | 0.63 | −20.4% | −1.87 | 0.23 | −0.05 | 0.10 | 0.45 |
| SB16, base under sector weights | 0.16 | 0.61 | −19.4% | −1.39 | 0.11 | 0.10 | 0.02 | 0.34 |
| SB42 | 0.05 | 0.50 | −26.4% | −1.51 | −0.19 | 0.14 | 0.02 | 0.01 |
| SB42 minus LQD | 0.01 | 0.45 | −30.3% | | | | | |

All five criteria fail. The weighting change itself is neutral (SB16 within 0.02 of the base,
with a better worst year). The breadth is what hurts, and it hurts most recently: 2020–2026 at
0.01 against 0.45, with 2021 at −0.86 and the last two years at −0.19. The walk-forward book
falls from 0.55 to 0.29 and its drawdown nearly doubles. Effective bets rose 113%; it did not
matter.

**Reading.** Under both constructions the ETF additions lose the sleeve money in the period
that matters most, the last six years, while the base names made theirs. The credit block is
the one positive standout (+1.72% a year in the 42-book, LQD the largest single contributor),
which is the kind of observation the registry records and does not act on. ETF-wrapped breadth
is closed as a route: two constructions, ten criteria, ten failures, and the direction of the
failure is recent. If futures ever become available the hypothesis is worth re-opening on
futures data, where the wrapper costs that drive this result do not exist.
