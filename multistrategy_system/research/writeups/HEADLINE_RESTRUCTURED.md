# Headline book after the October 2026 restructure (2026-10-09)

`headline_restructured.py` combines the three sleeves that run live since the restructure, the same
way the existing headline is built: walk-forward inverse-volatility weights (trailing 252 days,
re-estimated monthly). Window 2 Jan 2020 to 15 Dec 2025; rf = flat 5%; drawdown from initial
capital. This is a correction of the headline, not a trial: no strategy rule is chosen here.

| | Sharpe rf | Sharpe no-rf | Max drawdown | Volatility | CAGR | Worst year (rf Sharpe) |
|---|---|---|---|---|---|---|
| **Restructured book: trend + ETF momentum (paper rules) + stock swing book** | **0.98** | **1.70** | **−5.8%** | 6.9% | +12.2% | +0.04 (2025) |
| Old headline (old engine + trend), same window | 0.44 | 1.04 | −8.2% | 8.3% | +8.7% | −0.40 (2022) |
| Diagnostic: the swing book replaced by an equal-weight basket of the same 21 stocks | 1.02 | 1.71 | −7.2% | 7.2% | +12.8% | +0.55 (2025) |

- **Sleeves:**
  - trend 0.81 no-rf;
  - ETF momentum 0.81;
  - stock swing book 1.19.
- **The sleeves barely move together:** daily correlations −0.12, −0.02 and +0.01. That is why the
  book's Sharpe is far above any single sleeve's.
- **Weights:** about one third each on average (ranges 0.24–0.44).
- **Every year was positive** on rf Sharpe; 2025 was the weakest (+0.04).
- **Last 24 months:** 1.50 no-rf.

**Read with these limits:**
1. **Not out of sample.** The restructure was chosen on this same 2020–25 data, by the split test
   (dropping mean reversion, adopting the paper's ETF rules). The ETF rules come from a 2024 paper
   and were not tuned here, but the choice to use them was made after seeing these years. Expect the
   live record to be weaker. The forward tests (T-72 for ETF, T-49 for breadth) are the real check.
2. **The ETF sleeve is cost-sensitive.** These figures use 0.5 bp a side; at 1 bp the ETF sleeve's
   Sharpe roughly halves in the earlier run.
3. **The stock list carries hindsight.** The 21 names were chosen in 2025–26.
4. **The live stock sleeve weights its 12 breadth names at half size**; this run weights all 21 equally.
5. **The swing book adds nothing at book level either.** Replacing it with a plain basket gives the
   same book (1.71 vs 1.70 no-rf). This answers the open book-level question as a diagnostic, not a
   pre-registered trial.
