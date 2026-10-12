# Response to the backtest implementation audit (2026-10-01/02)

The audit (`BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md`) was checked by running its read-only
reproduction script: every reproduction held. Fixes are in `ibkr_backtest.py` (pre-fix backup
`legacy_caches/code_backups/ibkr_backtest_2026-10-01_preaudit.py`) with one test per defect in
`tests/test_backtest_audit_fixes.py`. All seven fail on the pre-fix engine and pass after the fix.
These are corrections, not trials: no strategy rule was changed in response to a corrected number.

| # | Finding | Status |
|---|---|---|
| 1 | Swing sizing used volatility from the end of the daily file | Fixed: window ends before the trade date |
| 2 | Close-of-day swing cash changed the same morning's intraday sizes | Fixed: intraday sized from start-of-day cash |
| 3 | ETF candidate labels simulated against unfilled future bands | Fixed: all bands computed before the signal pass |
| 4 | "Paper rules" E0 inherited HAR | Fixed in split_trial v2 (`use_har_forecast=False`, as deployed) |
| 5 | First day of each annual run dropped | Fixed (engine returns and research scripts) |
| 6 | Terminal swing liquidations missing from the ledger | Fixed (`exit_reason='BACKTEST_END'`); swing reconstruction now matches booked P&L (corr 1.000) |
| 7 | Annual runs started cold | Fixed: `warmup_days=120` history loaded before the start, not traded |
| 8 | ML x-section compounded P&L increments as returns | Fixed: NAV-based returns (`tests/test_ml_xsection_accounting.py`) |
| 9 | ML x-section dropped the last open month | Fixed: held and marked to the data's end |
| 10 | Daily vol window ended the day before yesterday | Fixed |
| 11 | 13F publication look-ahead (crowded spaces) | Fixed 2026-10-11: each quarter's aggregate uses only filings published before its first use (`parse_13f.point_in_time`; 8,246 late manager filings, 0.01% of shares, removed). Rerun: CS −0.59, ALL −0.35, mirror +0.44, unchanged; daily returns identical. Test: `tests/test_audit_open_items.py` |
| 12 | Trend breadth: missing marks treated as zero | Fixed 2026-10-11: all instruments end on the common last real price (2026-06-24), interior gaps carry the last mark, and a held position without a price stops the run (`trend_breadth.matched_prices`, `check_marks`). Rerun: BASE16 0.19 → BREADTH46 −0.04 rf (was 0.18 → −0.05); SB42 0.06 (was 0.05); verdicts unchanged. The same interior-gap fix moves the live 16-ETF trend sleeve by 0.001 Sharpe. Tests: `tests/test_audit_open_items.py` |
| — | Gate trials (GATE_TRIAL, PERGATE_TRIAL) | Not rerun yet (the gate is off live); figures superseded |
| — | Vol forecast years grouped by forecast origin | By design; noted in VOL_FORECAST.md |

## What changed (corrected reruns)

**Headline book** (`results/audit_rerun/`, `headline_rebuild.py --equity audit`): equity sleeve rf
0.40 → **0.24**, no-rf 0.88 → **0.72**, MaxDD −17.1% → −20.2%; **book rf 0.51 → 0.44, no-rf 1.13
→ 1.04**, worst year −0.16 → −0.40 (2022), MaxDD −7.3% → −8.2%; DSR at N = 9: 67.9% → 63.4%
(16% at the registry's 83 rows). The book is still positive in 5 of 6 years and still beats both
sleeves on Sharpe. The equity sleeve here is the pre-restructure gated hybrid engine, not the
configuration now live.

**Split trial v2** (`results/split_trial_v2/`): every pre-registered decision is unchanged.
- **ETF paper rules (E0, no HAR):** 0.81, last 2y 0.39 → PASS.
- **Departures from the paper, each still lowering Sharpe:** every minute −1.20, ML gate −0.38, band ×2 −0.43, VIX gate −0.19, RSI −0.32.
- **As-is ETF intraday (E6):** −0.89.
- **Stocks:** swing only 1.19 vs as-is 1.08 → drop mean reversion; stock momentum −1.11 → FAIL.
- **Swing vs passive:** 1.19 vs 1.13 (needed 1.23) and last 2y 1.27 vs 1.74 → no timing edge.

**Engine breakdown v2** (`results/engine_breakdown_v2/`, gated 24 names, % of capital 2020–25):
- **Swing:** +80.6 (was +94.8).
- **ETF momentum:** −2.7.
- **Stock mean reversion:** −8.2.
- Same picture: all profit is the swing book. Swing in 2022 is now −2.8% (was +3.8%), still well ahead of the passive basket (−11.7%).
- 2024–25: swing +19.2% vs passive +37.2%.

**ML x-section v2:** ridge −0.32, LightGBM −0.39, composite −0.30; verdict unchanged.

**ETF forward-test thresholds** rebuilt from the corrected E0 (changes of at most 0.5 points);
they ship with the next VM package.
