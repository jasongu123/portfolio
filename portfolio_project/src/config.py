"""
Portfolio Optimization Project Configuration
=============================================
Comparing ERI, GARCH-EVT-CVaR, and classical approaches on NASDAQ stocks.

Based on:
- Mainik, Mitov & Rüschendorf (2015): ERI optimization
- Bedoui et al. (2023): GARCH-EVT-Vine Copula-CVaR
- Damjanovic et al. (2025): Diversification metrics
"""

from datetime import datetime

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Stock universe: NASDAQ-100 subset
# We'll filter to stocks with complete history
NUM_STOCKS = 100  # Target number of stocks (will filter by data availability)

# Time period
START_DATE = "2012-01-01"  # Extra history for expanding window estimation
END_DATE = "2024-12-31"
BACKTEST_START = "2015-01-01"  # Actual backtest period starts here

# Data source
DATA_SOURCE = "yahoo"

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================

# Rebalancing frequency
REBALANCE_FREQ = "M"  # Monthly

# Estimation window
WINDOW_TYPE = "expanding"  # expanding or rolling
MIN_HISTORY_DAYS = 500  # Minimum days required before first optimization

# Portfolio constraints
MIN_WEIGHT = 0.0  # Minimum weight per asset
MAX_WEIGHT = 0.20  # Maximum weight per asset (20% cap for diversification)
ALLOW_SHORT = False  # No short selling

# =============================================================================
# ERI OPTIMIZATION PARAMETERS (Mainik et al.)
# =============================================================================

# Tail fraction for Hill estimator
ERI_TAIL_FRACTION = 0.10  # Top 10% of observations (as in Mainik)

# Minimum tail index for inclusion
ERI_MIN_TAIL_INDEX = 1.5  # Exclude assets with extremely heavy tails

# =============================================================================
# GARCH-EVT-CVaR PARAMETERS (Bedoui et al.)
# =============================================================================

# GARCH specification
GARCH_P = 1
GARCH_Q = 1

# GPD threshold (quantile for POT)
GPD_THRESHOLD_QUANTILE = 0.90  # Use top 10% for tail fitting

# CVaR confidence level
CVAR_ALPHA = 0.95  # 95% CVaR (equivalent to ES at 5% level)

# Monte Carlo simulations for CVaR estimation
MC_SIMULATIONS = 10000

# =============================================================================
# VINE COPULA PARAMETERS
# =============================================================================

# Copula type for vine structure
VINE_TYPE = "c-vine"  # c-vine or d-vine

# Copula families to consider
COPULA_FAMILIES = ["gaussian", "student-t", "clayton", "gumbel", "frank"]

# =============================================================================
# DIVERSIFICATION METRICS (Damjanovic et al.)
# =============================================================================

# Metrics to compute
DIVERSIFICATION_METRICS = [
    "ENC",      # Effective Number of Constituents
    "shannon",  # Shannon Entropy
    "HHI",      # Herfindahl-Hirschman Index
    "PDI",      # Portfolio Diversification Index
    "ENB",      # Effective Number of Bets
]

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

RISK_FREE_RATE = 0.02  # Annual risk-free rate for Sharpe ratio
TRADING_DAYS_PER_YEAR = 252

# =============================================================================
# OUTPUT PATHS
# =============================================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

# =============================================================================
# NASDAQ-100 TICKERS (as of late 2024)
# =============================================================================

NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "GOOG", "AVGO", "COST",
    "PEP", "CSCO", "ADBE", "NFLX", "AMD", "CMCSA", "TMUS", "INTC", "INTU", "QCOM",
    "TXN", "AMGN", "AMAT", "ISRG", "HON", "BKNG", "SBUX", "LRCX", "VRTX", "MDLZ",
    "GILD", "ADI", "ADP", "REGN", "PANW", "KLAC", "SNPS", "CDNS", "ASML", "MELI",
    "PYPL", "MAR", "CRWD", "ORLY", "MNST", "NXPI", "CTAS", "MRVL", "FTNT", "CSX",
    "ADSK", "PCAR", "DXCM", "CHTR", "WDAY", "CPRT", "ROST", "AEP", "ODFL", "PAYX",
    "KDP", "MCHP", "KHC", "MRNA", "AZN", "LULU", "EXC", "IDXX", "FAST", "CTSH",
    "EA", "VRSK", "CSGP", "GEHC", "BKR", "XEL", "FANG", "ON", "DDOG", "TTWO",
    "ANSS", "CDW", "BIIB", "DLTR", "GFS", "ILMN", "WBD", "ZS", "MDB", "TEAM",
    "SPLK", "PDD", "JD", "LCID", "RIVN", "ARM", "DASH", "COIN", "ZM", "ROKU"
]
