# Equity engine sized by the trend rule's market state (2026-09-16)

Research only: `equity_trend_gate.py`, gate pre-registered in its docstring; `ibkr2.py` untouched.
Registry T-2026-09-16-40 (GATE), -41 (FLAT), -42 (MIRROR control). Output
`results/equity_trend_gate_daily.csv`.

**Hypothesis.** The intraday engine's edge is regime-dependent; its worst year (2022) was the
trend sleeve's best. Halving its size when the trend rule on SPY (mean sign of 63/126/252-day
returns, month-end stamp, active next day) is bearish should cut the bad years cheaply.

**Answer: no. The engine is better in bearish regimes, not worse.**

| Engine daily returns 2020–2025 by state | Days | No-rf Sharpe |
|---|---|---|
| State −1 (all three lookbacks down) | 171 | +1.77 |
| State −⅓ | 145 | +0.94 |
| State +⅓ | 248 | −0.47 |
| State +1 (all three up) | 913 | +1.22 |
| Bearish (≤ 0) vs bullish (> 0) | 316 vs 1,161 | 1.41 vs 0.81 |

The 2022 loss did not come from the bearish-state days; it came from the bear-market rallies,
when the state read weakly bullish. Halving in bearish states therefore made 2022 worse (−1.28 →
−2.13 vol-matched) and lowered the full-window rf-Sharpe (0.54 → 0.47). Flat in bear states is
worse still (−3.28). The mirror control, halving in bullish states, had the best worst year
(−0.67) and a lower full-window rf-Sharpe (0.51), so the control criterion passed by 0.03; that
is an observation, not a result.

## Gate

| | Criterion | Result | |
|---|---|---|---|
| E1 | Engine Sharpe on bullish days exceeds bearish by ≥ 0.30 | −0.60 | fail |
| E2 | GATE vol-matched: rf-Sharpe +0.05, worst year +0.30, drawdown not deeper | 0.47 vs 0.54; −2.13 vs −1.28 | fail |
| E3 | Not one year | ex-2022 0.83 vs 0.88 | fail |
| E4 | MIRROR worse than engine | 0.51 vs 0.54 | pass |
| E5 | Walk-forward book not worse | 0.59 → 0.55; worst +0.11 → −0.01 | fail |

## Reading

- The engine's edge is not long-market beta in disguise; it earns most on days when the market
  trend is confirmed down. That is good news for orthogonality and bad news for this overlay.
- The weak spot is the transitional state (+⅓, 248 days, −0.47), which is where bear-market
  rallies land. That is a post-hoc reading of one state bucket on six years; it would need an
  out-of-sample engine record to become a hypothesis, and none exists before 2020.
- Both "blend" trials this week (trend timing off month-end; engine gated by trend state) were
  rejected, and both rejections came with a mechanism that runs the other way: the trend
  sleeve is paid by month-end flow, and the engine is paid in confirmed bear regimes. The two
  sleeves do not need each other's state; their independence is why the book works.
