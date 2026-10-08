# Equity engine breakdown (2026-10-01)

> **Corrected 2026-10-02** (AUDIT_RESPONSE_2026-10-01.md, results/engine_breakdown_v2/): swing +80.6% of capital 2020–25 (was +94.8), ETF momentum −2.7, stock mean reversion −8.2; swing 2022 −2.8% vs passive −11.7%; 2024–25 swing +19.2% vs passive +37.2%. Conclusions unchanged. The tables below are the original run.

Diagnostic only: `engine_breakdown.py`, using a new opt-in recorder in `ibkr_backtest.py`
(`record_positions = True`, `_log_positions`; records only; returns unchanged to the cent;
backup `legacy_caches/code_backups/ibkr_backtest_2026-10-01.py`). Results in
`results/engine_breakdown/` (report.txt, components_*.csv, positions_*.pkl). `ibkr2.py` untouched.

**Setup.** The 24-name engine as in GATE_TRIAL.md, 2020-01..2025-12-15, gated (G1, pooled forest)
and ungated (G0). Each intraday position is tagged by the candidate rule that opened it; swing
holdings are the rest (change in portfolio value minus intraday P&L). Every arm-year reproduced the
gate trial's cached returns exactly, except gated 2021, where the new Mac (arm64) gives +32.1%
for the year against the old Mac's +31.7%; a plain rerun without the recorder gives the same
+32.1%, so the recorder is not the cause.

## Where the money comes from (gated engine, % of capital, sum of daily contributions)

| Period | Swing holdings | ETF intraday momentum | Stock intraday mean reversion | Stock gap | Total | Engine Sharpe |
|---|---|---|---|---|---|---|
| 2020 | +15.8 | −1.7 | +1.6 | 0 trades | +15.7 | 0.82 |
| 2021 | +29.6 | −0.3 | −0.9 | 0 | +28.4 | 2.69 |
| 2022 | +3.8 | −0.6 | −3.1 | 0 | +0.1 | 0.00 |
| 2023 | +22.4 | −0.4 | −1.7 | 0 | +20.3 | 1.72 |
| 2024 | +18.9 | +0.6 | −1.7 | 0 | +17.8 | 1.76 |
| 2025 | +4.3 | −0.5 | −1.9 | 0 | +2.0 | 0.20 |
| Last 15 months | +3.7 | +0.1 | −3.6 | 0 | +0.3 | 0.02 |
| Standalone Sharpe 2020–25 | 1.22 | about −1 | about −1 | — | | 1.07 |

Ungated, the intraday losses are larger (ETF momentum −8.2, mean reversion −10.0 over 2020–25):
the gate's whole value is shrinking intraday losses. Swing holdings are not gated.

## Findings

1. **All of the engine's profit is the swing book.** It holds about 11 of the 24 names at about
   70% of capital (multi-day, long only). Every intraday part loses money after costs in almost
   every year.
2. **Stock mean reversion loses before costs** (gross −3 to −11 bps a trade in 2021–25; it made
   money only in 2020's VIX 30+ crash). Weakest: core stocks, long side, the first half hour and
   the last 90 minutes. **ETF momentum** is about zero before costs and −3 to −6 bps after
   (3–4 bps of costs a trade); 2024 is its only positive year.
3. **Gap trades never fire in the backtest.** The rule only enters between 9:31 and 9:35, but the
   backtester acts only from 10:00. Live `ibkr2.py` allows gap entries in the first 15 minutes
   (long only), so any live gap trades are untested.
4. **The swing book is mostly market exposure with a defensive tilt.** Compared with the 24-name
   equal-weight basket held at the swing book's own daily exposure: 2020–25 +94.8% vs +97.1%
   (same Sharpe, 1.22 vs 1.21). It avoided 2022 (+3.8% vs −10.1%) and beat in 2021; it lagged in
   2020, 2024 and 2025 (last 15 months +3.7% vs +15.3%). Daily correlation with the basket
   0.65–0.70.
5. **The recent weakness is two things:** the swing book lagging a rising market (2025: stop and
   below-SMA exits −22k, max-hold exits +20k) and mean reversion's steady bleed.

## What this changes

- EQUITY_BREADTH.md's "the edge is the method, not the names" (NEW12 1.02) is mostly the swing
  book holding stocks in a rising market; that generalises trivially.
- The engine is long-biased equity exposure, not an intraday edge. The book's orthogonality story
  (engine vs trend) needs checking against that.
- Candidate trials (not run; each to be pre-registered): (a) intraday off, swing only; (b) swing
  timing versus constant exposure, inside the book with the trend sleeve.
