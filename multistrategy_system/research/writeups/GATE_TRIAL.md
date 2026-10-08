# Gate-model trial (2026-09-28, T-2026-09-28-50..54)

> **Under correction (2026-10-01):** an implementation audit (BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md) found engine defects affecting the numbers below: ETF candidate labels simulated against unfilled future bands, future-volatility swing sizing, close-of-day swing cash changing morning sizes, cold annual warm-up. Corrected reruns: not yet scheduled (the gate is off live). Treat the figures below as superseded until then.

Research only: `gate_trial.py` (pre-registered in its docstring before any run), hook
`gate_model_factory` / `gate_calibration` in `ibkr_backtest.train_rf_model` (default path
unchanged; tests/test_gate_trial_hook.py). Results in `results/gate_trial/` (per arm-year returns,
gate stats, summary.csv, report.txt). `ibkr2.py` untouched.

**Question.** Does the engine's ML gate depend on its model class, and does the gate add anything?

**Setup.** The backtest gate behind the headline is one model pooled across the run's symbols,
trained on the latest 200 candidate outcomes (every candidate labelled, taken or not: win =
simulated trade > +0.1%), retrained every 30 outcomes, entry when P(win) >= 0.55. Arms change only
the model. Book: the 24 names the paper engine now trades, one backtest per year 2020–2025,
per-minute VIX, $0.0035 a share + 1 bp.

## Results (no-rf daily Sharpe unless stated)

| Arm | Model | 2020–25 | rf | MaxDD | Last 15 months | 2020–21 | 2022–23 | 2024–25 | 2025 | Book with trend, rf / no-rf |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 | no gate | 0.96 | 0.59 | −14.3% | −0.25 | 1.36 | 0.71 | 0.74 | −0.10 | 0.66 / 1.21 |
| **G1** | **random forest (control)** | **1.07** | 0.69 | −14.1% | +0.02 | 1.42 | 0.77 | 0.98 | 0.20 | **0.72 / 1.27** |
| G2 | logistic, L2 | 1.12 | 0.75 | −14.0% | +0.13 | 1.41 | 0.90 | 1.02 | 0.42 | 0.76 / 1.31 |
| G3 | LightGBM | 1.08 | 0.70 | −13.9% | −0.12 | 1.46 | 0.85 | 0.85 | 0.01 | 0.74 / 1.29 |
| G4 | LightGBM, calibrated | 1.04 | 0.66 | −14.5% | −0.14 | 1.43 | 0.79 | 0.82 | −0.09 | 0.71 / 1.26 |

Daily-return correlation with G1: 0.98–0.99 for every arm. The candidate and approval counts in
the gate stats count bar-minutes, not distinct trades (a blocked signal is re-counted every minute
it persists), so they are not approval rates and are not reported.

## Gate (pre-registered)

| Arm | M1 Sharpe ≥ G1 + 0.10 | M2 2 of 3 blocks | M3 recent ≥ G1 − 0.05 | M4 MaxDD | M5 book | Verdict |
|---|---|---|---|---|---|---|
| G2 logistic | 1.12 vs 1.17, fail | pass | pass | pass | pass | fail |
| G3 LightGBM | fail | pass | fail | pass | pass | fail |
| G4 calibrated | fail | pass | fail | pass | fail | fail |

Verdict as registered: **keep the forest.** Gate value G1 − G0 = +0.10, exactly at the
pre-registered line: the gate adds value, but only just.

## Reading

1. **The model class barely matters.** All four gated arms are 98–99% the same portfolio, and
   their Sharpes span 1.04–1.12. With about 200 noisy labels per fit, no model has much room to
   differ.
2. **A linear model does at least as well as the forest.** Logistic regression was higher on
   every summary measure (full period, recent 15 months, two of three blocks, the book) but
   short of the +0.10 bar, so it does not replace the forest. The forest's nonlinearity is not
   earning anything measurable here; the gate behaves like a linear filter.
3. **Boosting and calibration did not help,** and were weaker than the forest recently.
4. **The gate itself is worth about 0.1 of Sharpe** on this book, most of it in 2024–25 (0.98 vs
   0.74) and 2025 (0.20 vs −0.10). Without it the last 15 months are −0.25.
5. **The larger open question is not the model class but the live gate's design.** The live
   engine uses a separate forest per name, trained on ~1,000 days at startup, with a 0.65
   threshold; none of that is what these backtests validate. The next trial should run that
   design inside the backtester against G1.

By-product: the 24-name book with the validated gate and the live trend sleeve reads rf 0.72 /
no-rf 1.27 with a −10% book drawdown over 2020–2025 (headline twelve-name book: 0.51 / 1.13 /
−7.3%). This is informational; the headline changes only through the forward test T-49.
