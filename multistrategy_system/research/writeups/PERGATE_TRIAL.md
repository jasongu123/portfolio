# Per-name vs pooled gate trial (2026-10-01, T-2026-10-01-55..58)

> **Under correction (2026-10-01):** an implementation audit (BACKTEST_IMPLEMENTATION_AUDIT_2026-10-01.md) found engine defects affecting the numbers below: the same label and engine defects as GATE_TRIAL.md. Corrected reruns: not yet scheduled (the gate is off live). Treat the figures below as superseded until then.

Research only: `pergate_trial.py` (pre-registered in its docstring before any run). All gate
changes live in a subclass inside that script; `ibkr_backtest.py` and `ibkr2.py` untouched.
Results in `results/pergate_trial/` (harvest_*.pkl, per arm-year returns, summary.csv,
report.txt). Ran on the new Mac (M5 Pro, 24 GB): 10 runs at once, 57 minutes end to end.

**Questions.** (1) Is one gate model per stock better than one pooled model? (2) Is the live
gate, which differs from the validated backtest gate in four ways at once, any good?

**Setup.** 24-name book, hybrid engine, per-minute VIX, $0.0035 a share + 1 bp, as GATE_TRIAL.md.
Minute data starts late 2019, so 2020 is history only and the evaluation window is
2021-01-01..2025-12-15. Every model is warm-started from an ungated harvest of all labelled
candidates since 2020 (the harvest re-run reproduced the cached no-gate returns exactly). A single
stock produces only ~40–90 candidates a year (all 24: ~1,200–3,400), so a per-stock model's last
200 outcomes span 3–4 years, while the pooled model's span about a month.

| Arm | Gate | 2021–25 no-rf | rf | MaxDD | Last 15 months | 2021–22 | 2023–24 | 2025 | Corr with Q0 | Book with trend, rf / no-rf |
|---|---|---|---|---|---|---|---|---|---|---|
| **Q0** | **pooled, rolling 200, 0.55 (control)** | **1.22** | 0.79 | −12% | +0.05 | 1.17 | 1.74 | 0.20 | 1.00 | **0.72 / 1.31** |
| Q1 | one model per stock, otherwise Q0 | 1.19 | 0.77 | −12% | +0.04 | 1.12 | 1.76 | 0.17 | 0.99 | 0.71 / 1.30 |
| Q2 | per stock, fitted yearly on last 1000 sessions, 0.55 (diagnostic) | 1.26 | 0.83 | −12% | +0.10 | 1.22 | 1.75 | 0.33 | 0.99 | 0.76 / 1.35 |
| Q3 | **live replica**: per stock, last 1000 sessions, long-only labels, 0.65 asymmetric | 1.20 | 0.77 | −12% | +0.01 | 1.14 | 1.80 | 0.11 | 0.97 | 0.72 / 1.33 |
| G0 | no gate (reference) | 1.03 | 0.61 | −14% | −0.25 | 1.05 | 1.57 | −0.10 | 0.98 | 0.61 / 1.20 |
| G1 | pooled, cold start each year (gate trial) | 1.18 | 0.76 | −13% | +0.02 | 1.10 | 1.73 | 0.20 | 1.00 | 0.69 / 1.29 |

## Decisions (pre-registered)

- **D1 per stock vs pooled: pooled stays.** Q1 1.19 vs the 1.32 bar (M1 fail), better in 1 of 3
  blocks (M2 fail), book 0.71 vs 0.72 (M5 fail); M3 and M4 pass. It is a tie, not a loss.
- **D2 live gate: acceptable.** L1 1.20 ≥ 1.12, L2 +0.01 ≥ −0.05, L3 1.20 ≥ no gate 1.03. No
  change to `ibkr2.py` is indicated.

## Reading

1. **Per stock vs pooled makes no measurable difference.** The two are 99% the same portfolio.
   Pooling gives fresh data; per-stock gives stock-specific but years-old data; on this engine
   the two effects cancel.
2. **The live gate holds up.** Its four departures from the validated design (per stock,
   trained once on ~4 years, long-only labels, 0.65 asymmetric) cost nothing measurable:
   1.20 vs 1.22. The live gate barely blocks shorts (a short is stopped only when the model is
   ≥ 65% sure a long would win), and that did not hurt.
3. **Every gate design lands within 1.19–1.26, and every one beats no gate by 0.16–0.23.** Having
   a gate matters; how it is built does not. Together with GATE_TRIAL.md (model class doesn't
   matter either), further gate engineering is unlikely to pay.
4. **Warm-starting helps a little** (Q0 1.22 vs cold G1 1.18), as live does by training at startup.
5. **No gate design fixes the recent weakness.** The last 15 months are +0.01 to +0.10 for every
   gated arm (−0.25 ungated). The gate stops losses there; it does not create gains.

**Limits.** The live replica learns from the backtester's candidates, not from `ibkr2.py`'s
simpler `simulate_historical_trades`; it refits yearly, while live keeps its cached models until
the cache is rebuilt; the "last 1000 sessions" window is full only in 2024–25 (1–3 years before);
live also adds that day's executed trades, which the replica does not. Q2's lead (1.26) is
diagnostic only and inside the noise of the others.
