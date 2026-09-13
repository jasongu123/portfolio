"""
Rigor layer — Probabilistic & Deflated Sharpe (Bailey & Lopez de Prado).

Two questions a quant audience always asks, answered honestly:
  1. PSR  — given only ~3 years and non-normal returns, how confident are we the book's
            TRUE Sharpe is > 0 (and > 0.5)?  [Probabilistic Sharpe Ratio]
  2. DSR  — we searched ~N sleeves and kept the best; how much of the book's Sharpe
            survives that multiple-testing selection?  [Deflated Sharpe Ratio]

Inputs are the cached returns already on disk; nothing is re-run.

Usage:  python3.12 deflated_sharpe.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANN = np.sqrt(252)
EMC = 0.5772156649  # Euler-Mascheroni

# --- assemble the book (equity + trend v2 at risk-parity, the 2023-25 overlap) ---
eq = pd.concat([pd.read_csv(os.path.join(SCRIPT_DIR, f'equity_returns_{y}.csv'),
                            index_col=0, parse_dates=True).iloc[:, 0]
                for y in [2023, 2024, 2025]]).sort_index()
tv = pd.read_csv(os.path.join(SCRIPT_DIR, 'trend_v2_returns.csv'),
                 index_col=0, parse_dates=True).iloc[:, 0]
lo, hi = eq.index.min(), eq.index.max()
t = tv[(tv.index >= lo) & (tv.index <= hi)]
j = pd.concat([eq.rename('e'), t.rename('t')], axis=1).dropna()
ve, vt = j['e'].std(), j['t'].std()
we = (1 / ve) / ((1 / ve) + (1 / vt))
book = we * j['e'] + (1 - we) * j['t']


def psr(r, sr_star_ann=0.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_star), adjusting for skew/kurtosis & n."""
    r = r.dropna()
    sr = r.mean() / r.std()                       # per-period (daily)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))         # Pearson (normal = 3)
    n = len(r)
    sr_star = sr_star_ann / ANN
    num = (sr - sr_star) * np.sqrt(n - 1)
    den = np.sqrt(max(1e-9, 1 - g3 * sr + ((g4 - 1) / 4) * sr ** 2))
    return float(norm.cdf(num / den)), sr * ANN, g3, g4, n


print("=" * 72)
print("PROBABILISTIC SHARPE — is the book's positive Sharpe statistically reliable?")
print("=" * 72)
for label, r in [('Equity book alone', j['e']),
                 ('Combined book (equity + trend v2, risk-parity)', book)]:
    p0, sr_ann, g3, g4, n = psr(r, 0.0)
    p05, *_ = psr(r, 0.5)
    print(f"\n{label}")
    print(f"  SR(ann, no-rf) {sr_ann:.2f} | skew {g3:+.2f} | kurtosis {g4:.1f} | n={n} (~{n/252:.1f}y)")
    print(f"  PSR(true SR > 0)   = {p0:5.1%}")
    print(f"  PSR(true SR > 0.5) = {p05:5.1%}")

print("\n" + "=" * 72)
print("DEFLATED SHARPE — does the book survive multiple-testing (we searched N sleeves)?")
print("=" * 72)
# annualized standalone Sharpes of the sleeves we searched over (the trial set)
TRIALS_ANN = {'trend': 0.16, 'reversal': -0.82, 'low-vol': -0.33,
              'stat-arb': -1.23, 'carry': -0.54, 'event-filter': 0.00}
var_daily = np.var([s / ANN for s in TRIALS_ANN.values()], ddof=1)
print(f"  trial sleeves (ann SR): " + ", ".join(f"{k} {v:+.2f}" for k, v in TRIALS_ANN.items()))


def expected_max_sr(var_sr, N):
    return np.sqrt(var_sr) * ((1 - EMC) * norm.ppf(1 - 1.0 / N) + EMC * norm.ppf(1 - 1.0 / (N * np.e)))


r = book.dropna()
sr = r.mean() / r.std()
g3 = float(skew(r))
g4 = float(kurtosis(r, fisher=False))
n = len(r)
print(f"\n  {'N trials':>9s}{'hurdle SR* (ann)':>18s}{'DSR = P(SR>SR*)':>18s}")
for N in [6, 12, 24]:
    sr_star0 = expected_max_sr(var_daily, N)
    num = (sr - sr_star0) * np.sqrt(n - 1)
    den = np.sqrt(max(1e-9, 1 - g3 * sr + ((g4 - 1) / 4) * sr ** 2))
    dsr = float(norm.cdf(num / den))
    print(f"  {N:>9d}{sr_star0*ANN:>18.2f}{dsr:>17.1%}")

print("\n" + "=" * 72)
print("HONEST READ: PSR>0 should be high (the book IS positive); PSR>0.5 and the DSR")
print("show the real uncertainty — 3 years + a sleeve search means moderate, not")
print("slam-dunk, confidence. Stating that number IS the rigor. More years shrink it.")
print("=" * 72)
