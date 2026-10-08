# Machine-learning cross-sectional ranking (2026-10-01, T-2026-10-01-73..75)

> **Under correction (2026-10-01):** an implementation audit (BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md) found engine defects affecting the numbers below: daily P&L increments compounded as returns, and the last open month dropped. Corrected reruns: done: results/ml_xsection_v2/ (ridge -0.32, LightGBM -0.39, composite -0.30; verdict unchanged). Treat the figures below as superseded until then.

Research only: `ml_xsection_trial.py` (pre-registered in its docstring before any return was
computed). Results in `results/ml_xsection/` (report.txt, summary.csv, daily_returns.csv,
predictions.pkl, lgbm_importance.csv). 17 seconds on the new Mac.

**Setup.** S&P 500 point-in-time members (569 of 625 with IBKR prices; 95% of members covered in
the out-of-sample months), monthly rebalance with a one-day lag, 17 price-based features ranked
cross-sectionally, models refitted monthly on the previous 36 months, decile long-short, 5 bp per
unit of turnover + 0.5% a year borrow. Out of sample 2021-01..2026-09-15 (69 rebalances).

| Arm | Sharpe | Return/yr | MaxDD | Last 2y | 2023 | 2024 | 2025 | IC (mean, t) | Turnover/mo |
|---|---|---|---|---|---|---|---|---|---|
| A0 textbook composite | −0.31 | −5.8% | −38% | −0.93 | −0.87 | 0.76 | −0.83 | −0.01, −0.3 | 2.8 |
| A1 ridge | −0.34 | −5.5% | −50% | 0.37 | −4.04 | 1.20 | 0.13 | −0.01, −0.7 | 3.0 |
| A2 LightGBM | −0.39 | −5.2% | −42% | 0.13 | −3.07 | 1.09 | −0.50 | −0.01, −0.7 | 3.1 |

Mean next-month return by prediction decile is flat (≈0.8–1.4% a month, no monotone pattern).
LightGBM importance is spread evenly (4–8% per feature): no feature carries information.
Monthly correlation with the existing sleeves is low (|corr| ≤ 0.28), but there is nothing to
diversify with.

**Verdict: FAIL** (ridge chosen by the rule): L1 −0.34 < 0.50, L3 no profit, L4 does not beat
the composite; L2 (last 2y +0.37) and L5 pass. Information coefficient ≈ 0 for every arm.

**Mechanism.** Price-only signals on large caps are among the most crowded data in finance: every
feature here is a published anomaly that earlier trials (TAKING_SIDES, SEASONALITY,
CROWDED_SPACES) found dead since 2020. Combining dead signals, linearly or with trees, gives a
dead combination; ML can only find structure that is in the data. The 2023 collapse (ridge −4.0)
is a model trained on 2020–22's rotations meeting the narrow mega-cap rally.

**What would change the answer:** information the market prices less efficiently than past
prices: fundamentals (SEC XBRL), analyst revisions, text (the news pipeline once it collects data),
or smaller, less-arbitraged stocks (which IBKR paper can trade but costs more).

## Corrected rerun (2026-10-01, audit items 8 and 9)

`book()` now computes each day's return as that day's P&L over the previous day's NAV (it had
compounded P&L increments measured in units of the rebalance NAV), and the last executable signal
(2026-08-31) is held and marked to the end of the data instead of being dropped; the data's final
mid-month day is no longer treated as a month-end signal. Results in `results/ml_xsection_v2/`:
composite −0.30, ridge −0.32 (last 2y +0.37), LightGBM −0.39; information coefficient unchanged at
about −0.01. **Verdict unchanged: FAIL.** A correction, not a new trial.
