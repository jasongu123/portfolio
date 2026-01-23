# Portfolio Optimization Project

**Comparing Extreme Value Theory Approaches to Portfolio Optimization**

This project implements and compares multiple portfolio optimization methodologies:

1. **Equal Weight (EW)** - Benchmark naive diversification
2. **Minimum Variance (MV)** - Classical Markowitz approach
3. **Extreme Risk Index (ERI)** - Based on Mainik, Mitov & Rüschendorf (2015)
4. **GARCH-EVT-CVaR** - Based on Bedoui et al. (2023)

With diversification analysis using metrics from Damjanovic et al. (2025).

## Project Structure

```
portfolio_project/
├── src/
│   ├── main.py                  # Main entry point
│   ├── config.py                # Configuration parameters
│   ├── data_module.py           # Yahoo Finance data download
│   ├── synthetic_data.py        # Synthetic data for testing
│   ├── benchmark_portfolios.py  # EW and MV optimizers
│   ├── eri_optimizer.py         # ERI optimization (Mainik)
│   ├── garch_evt_cvar.py        # GARCH-EVT-CVaR (Bedoui)
│   ├── diversification_metrics.py  # Metrics (Damjanovic)
│   ├── backtester.py            # Backtesting framework
│   └── visualization.py         # Plotting functions
├── data/                        # Data files (generated)
├── results/                     # Output visualizations
└── README.md
```

## Installation

### Requirements

```bash
pip install numpy pandas scipy yfinance matplotlib seaborn arch statsmodels cvxpy
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
cd src

# Option 1: Run complete pipeline with synthetic data
python main.py --all

# Option 2: Download real NASDAQ data then run
python main.py --download
python main.py --backtest
python main.py --visualize
```

## Usage

### Command Line Options

| Command | Description |
|---------|-------------|
| `python main.py --download` | Download NASDAQ-100 data from Yahoo Finance |
| `python main.py --synthetic` | Generate synthetic data for testing |
| `python main.py --backtest` | Run full backtest |
| `python main.py --visualize` | Create result visualizations |
| `python main.py --all` | Run complete pipeline |

### Configuration

Edit `src/config.py` to modify:

| Parameter | Default | Description |
|-----------|---------|-------------|
| NUM_STOCKS | 50 | Number of stocks to include |
| START_DATE | 2012-01-01 | Data start date |
| BACKTEST_START | 2015-01-01 | Backtest start date |
| REBALANCE_FREQ | M | Rebalancing frequency (M=monthly) |
| MAX_WEIGHT | 0.20 | Maximum weight per asset |
| CVAR_ALPHA | 0.95 | CVaR confidence level |

## Methodologies

### 1. Minimum Variance (MV)

Classical Markowitz portfolio minimizing portfolio variance:

```
minimize   w' Σ w
subject to Σ w_i = 1
           0 ≤ w_i ≤ max_weight
```

### 2. Extreme Risk Index (ERI)

From Mainik et al. (2015). Uses multivariate regular variation to minimize tail risk:

```
minimize   ERI(w) = E[|w'Θ|^α]
```

Where:
- α is the tail index (estimated using Hill estimator)
- Θ follows the spectral measure of extreme observations

Key insight: ERI outperforms MV on assets with heavy tails (α ≤ 2.2).

### 3. GARCH-EVT-CVaR

From Bedoui et al. (2023). Three-stage approach:

1. **GARCH(1,1)**: Model conditional volatility
2. **GPD**: Model tail behavior of standardized residuals
3. **CVaR**: Optimize expected shortfall

```
minimize   CVaR_α(w)
subject to Σ w_i = 1
           0 ≤ w_i ≤ max_weight
```

### 4. Diversification Metrics

From Damjanovic et al. (2025):

| Metric | Type | Description |
|--------|------|-------------|
| ENC | Weight-based | Effective Number of Constituents |
| HHI | Weight-based | Herfindahl-Hirschman Index |
| Shannon | Weight-based | Shannon Entropy of weights |
| PDI | Covariance-based | Portfolio Diversification Index |
| ENB | Combined | Effective Number of Bets |
| EDP | Combined | Exploited Diversification Potential |

## Output

### Performance Metrics

The backtest produces:

| Metric | Description |
|--------|-------------|
| Annual Return | Geometric annualized return |
| Annual Volatility | Annualized standard deviation |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| VaR (95%) | Value at Risk |
| CVaR (95%) | Conditional Value at Risk |

### Visualizations

Generated in `results/`:

1. `cumulative_returns.png` - Portfolio value over time
2. `drawdowns.png` - Drawdown periods
3. `rolling_sharpe.png` - Rolling 1-year Sharpe ratio
4. `performance_comparison.png` - Bar chart comparison
5. `diversification_comparison.png` - Diversification metrics

## Extending the Project

### Adding Vine Copula

The current GARCH-EVT-CVaR uses Gaussian copula. To add Vine Copula:

1. Install `pyvinecopulib`:
   ```bash
   pip install pyvinecopulib
   ```

2. Modify `garch_evt_cvar.py` to use C-vine or D-vine structure

### Using Different Assets

1. Edit `NASDAQ100_TICKERS` in `config.py`
2. Or modify `data_module.py` for different markets (e.g., FTSE 100)

### Adding More Optimization Methods

Create a new optimizer class following the pattern:

```python
class NewOptimizer:
    def __init__(self, **params):
        self.weights = None
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        # Your optimization logic
        self.weights = ...
        return self.weights
    
    def get_weights(self) -> np.ndarray:
        return self.weights
```

Then add to the strategies dict in `backtester.py`.

## References

1. Mainik, G., Mitov, G., & Rüschendorf, L. (2015). Portfolio optimization for heavy-tailed assets: Extreme Risk Index vs. Markowitz. *Journal of Empirical Finance*, 32, 115-134.

2. Bedoui, R., Benkraiem, R., Guesmi, K., & Kedidi, I. (2023). Portfolio optimization through hybrid deep learning and genetic algorithms vine Copula-GARCH-EVT-CVaR model. *Technological Forecasting and Social Change*, 197, 122887.

3. Damjanovic, A., et al. (2025). Price vs. market-cap-weighted portfolio diversification: does it matter? *Working Paper*.

## License

MIT License

## Author

Portfolio Optimization Course Project
December 2024
