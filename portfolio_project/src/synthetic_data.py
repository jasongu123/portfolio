"""
Synthetic Data Generator
========================
Creates realistic synthetic stock return data for algorithm development.
You will replace this with real data from Yahoo Finance on your local machine.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def generate_synthetic_returns(n_stocks: int = 50, 
                                n_days: int = 3000,
                                start_date: str = "2012-01-01") -> pd.DataFrame:
    """
    Generate synthetic stock returns with realistic properties:
    - Heavy tails (Student-t distributed)
    - Volatility clustering (GARCH-like)
    - Cross-sectional correlation
    - Varying tail heaviness across stocks
    
    Parameters
    ----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of trading days
    start_date : str
        Start date for the index
        
    Returns
    -------
    pd.DataFrame
        Synthetic daily returns
    """
    # Generate trading day index
    dates = pd.bdate_range(start=start_date, periods=n_days)
    
    # Create correlation matrix (factor model structure)
    # Market factor + sector factors
    n_sectors = 5
    stocks_per_sector = n_stocks // n_sectors
    
    # Base correlation from market factor
    market_loading = np.random.uniform(0.3, 0.8, n_stocks)
    
    # Sector membership
    sector_loadings = np.zeros((n_stocks, n_sectors))
    for i in range(n_stocks):
        sector = i // stocks_per_sector
        if sector >= n_sectors:
            sector = n_sectors - 1
        sector_loadings[i, sector] = np.random.uniform(0.2, 0.5)
    
    # Build covariance matrix
    idio_var = np.random.uniform(0.0001, 0.0004, n_stocks)  # Daily variance
    
    cov_matrix = np.outer(market_loading, market_loading) * 0.0002
    cov_matrix += sector_loadings @ sector_loadings.T * 0.0001
    cov_matrix += np.diag(idio_var)
    
    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if eigvals.min() < 0:
        cov_matrix += np.eye(n_stocks) * (abs(eigvals.min()) + 1e-6)
    
    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(cov_matrix)
    
    # Generate returns with GARCH-like volatility clustering
    returns = np.zeros((n_days, n_stocks))
    
    # Tail indices for each stock (lower = heavier tails)
    # Range from 2.5 (heavy) to 6 (moderate)
    tail_indices = np.random.uniform(2.5, 6.0, n_stocks)
    
    # Initial volatilities
    vols = np.sqrt(np.diag(cov_matrix))
    
    for t in range(n_days):
        # Generate correlated innovations
        z = np.random.standard_normal(n_stocks)
        correlated_z = L @ z
        
        # Apply Student-t transformation for heavy tails
        for i in range(n_stocks):
            # Transform to Student-t with stock-specific df
            df = tail_indices[i]
            u = stats.norm.cdf(correlated_z[i])
            correlated_z[i] = stats.t.ppf(u, df) / np.sqrt(df / (df - 2))
        
        # Scale by volatility
        returns[t, :] = correlated_z * vols
        
        # Update volatility (simple GARCH(1,1) dynamics)
        omega = 0.00001
        alpha = 0.08
        beta = 0.90
        vols = np.sqrt(omega + alpha * returns[t, :]**2 + beta * vols**2)
    
    # Create ticker names
    tickers = [f"STOCK_{i:02d}" for i in range(n_stocks)]
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
    
    # Store tail indices as metadata
    returns_df.attrs['tail_indices'] = dict(zip(tickers, tail_indices))
    
    return returns_df


def generate_synthetic_prices(returns: pd.DataFrame, 
                               initial_prices: np.ndarray = None) -> pd.DataFrame:
    """
    Generate prices from returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Log returns
    initial_prices : np.ndarray
        Starting prices (default: random 50-500)
        
    Returns
    -------
    pd.DataFrame
        Price series
    """
    n_stocks = returns.shape[1]
    
    if initial_prices is None:
        np.random.seed(42)
        initial_prices = np.random.uniform(50, 500, n_stocks)
    
    # Cumulative returns
    cum_returns = returns.cumsum()
    
    # Convert to prices
    prices = pd.DataFrame(
        initial_prices * np.exp(cum_returns.values),
        index=returns.index,
        columns=returns.columns
    )
    
    # Add initial row
    initial_row = pd.DataFrame(
        [initial_prices],
        index=[returns.index[0] - pd.Timedelta(days=1)],
        columns=returns.columns
    )
    prices = pd.concat([initial_row, prices])
    
    return prices


def compute_tail_statistics(returns: pd.DataFrame, 
                            tail_fraction: float = 0.10) -> pd.DataFrame:
    """
    Compute tail statistics using Hill estimator.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return series
    tail_fraction : float
        Fraction of observations to use for tail estimation
        
    Returns
    -------
    pd.DataFrame
        Tail statistics for each stock
    """
    stats_list = []
    
    for col in returns.columns:
        r = returns[col].dropna().values
        n = len(r)
        k = int(n * tail_fraction)
        
        if k < 10:
            continue
        
        # Absolute returns for tail analysis
        abs_r = np.abs(r)
        sorted_abs = np.sort(abs_r)[::-1]  # Descending
        
        # Hill estimator for tail index
        log_ratios = np.log(sorted_abs[:k] / sorted_abs[k])
        hill_estimate = k / np.sum(log_ratios)
        
        # Standard error
        hill_se = hill_estimate / np.sqrt(k)
        
        stats_list.append({
            'ticker': col,
            'tail_index': hill_estimate,
            'tail_se': hill_se,
            'kurtosis': pd.Series(r).kurtosis(),
            'skewness': pd.Series(r).skew(),
            'n_obs': n
        })
    
    return pd.DataFrame(stats_list).set_index('ticker')


def prepare_synthetic_dataset():
    """
    Create complete synthetic dataset for algorithm development.
    """
    print("Generating synthetic dataset...")
    print("(Replace with real Yahoo Finance data on your local machine)")
    print()
    
    # Generate returns
    returns = generate_synthetic_returns(n_stocks=50, n_days=3000)
    
    # Generate prices
    prices = generate_synthetic_prices(returns)
    
    # Compute tail statistics
    tail_stats = compute_tail_statistics(returns)
    
    # Save to files
    prices.to_csv("data/prices.csv")
    returns.to_csv("data/returns.csv")
    tail_stats.to_csv("data/tail_statistics.csv")
    
    # Summary
    print(f"Dataset created:")
    print(f"  Stocks: {returns.shape[1]}")
    print(f"  Trading days: {returns.shape[0]}")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print()
    print("Tail index distribution:")
    print(f"  Min:  {tail_stats['tail_index'].min():.2f}")
    print(f"  Mean: {tail_stats['tail_index'].mean():.2f}")
    print(f"  Max:  {tail_stats['tail_index'].max():.2f}")
    print()
    print("Files saved: data/prices.csv, data/returns.csv, data/tail_statistics.csv")
    
    return prices, returns, tail_stats


if __name__ == "__main__":
    prices, returns, tail_stats = prepare_synthetic_dataset()
    
    print("\n" + "="*60)
    print("TAIL STATISTICS (Hill Estimator)")
    print("="*60)
    print(tail_stats.sort_values('tail_index').head(10))
