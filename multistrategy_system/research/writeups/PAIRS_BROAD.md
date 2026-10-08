# Pairs trading across the S&P 500 (T-2026-10-07-96): FAIL

**Question.** The first stat-arb sleeve traded 12 related large-cap pairs and died when one
relationship broke (AMD vs MSFT/QQQ, −33%). Does trading many more pairs fix it?

**Method.**
- The textbook distance method (Gatev, Goetzmann & Rouwenhorst, RFS 2006) across the whole S&P 500,
  using point-in-time membership.
- Each month: take the 100 closest pairs over the past year; trade them for six months.
- Open at a 2-sigma gap with a one-day wait; close when the gap crosses zero.
- Costs 5 bp per side per leg plus borrow, on committed capital.
- Pre-registered in `pairs_broad.py`.

**Result, 2020 to Sep 2026 (100 pairs):**
- Net Sharpe 0.22 (gross 0.46), about +0.8% a year, t 0.71.
- The last 24 months were −1.75%.
- The best two years supplied 247% of the profit.
- Four of five rules failed. Correlation with SPY (+0.21) passed.

| Pairs | 20 | 50 | 100 | 200 |
|---|---|---|---|---|
| Net Sharpe 2020+ | 0.02 | 0.16 | 0.22 | 0.27 |
| Net Sharpe 2005–19 (context) | 0.11 | 0.13 | 0.14 | 0.21 |

**What it shows.**
1. **More pairs do help, but only by smoothing.** Sharpe rises with the number of pairs because one
   broken pair matters less. The average pair still earns little more than its costs: gross 0.46,
   net 0.22.
2. **Most positions never converge.** 57% were still open when the six months ran out, with a mean
   of −5.1%. The 43% that converged averaged +7.3%. Divergence risk eats the profit.
3. **The profit comes from crises.** Net return was +5.6% in 2008, +10.2% in 2009 and +5.9% in 2020.
   In the calm years it is about zero or negative, including 2024–26. That is the known pattern:
   wide, fast-reverting spreads appear when volatility spikes.
4. **This is not a post-2020 decay.** The method was already weak in 2005–19 (0.14). That matches
   published work showing large-cap pairs profits shrinking since the 1990s.

**Status.** Pairs trading on large caps is closed. A less-watched version (stock vs sector in
small and mid caps) stays inside hypothesis H1 of FAILURE_AUTOPSY.md, pending the data check.
