"""
Data Module
===========
Downloads and preprocesses NASDAQ stock data from Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    NASDAQ100_TICKERS, START_DATE, END_DATE, NUM_STOCKS,
    DATA_DIR, MIN_HISTORY_DAYS, BACKTEST_START
)


def download_stock_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    
    Parameters
    ----------
    tickers : list
        List of stock ticker symbols
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True
    )
    
    # Extract Close prices (yfinance returns MultiIndex if multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers
    
    print(f"Downloaded {len(prices)} trading days")
    return prices


def filter_stocks_by_availability(prices: pd.DataFrame, 
                                   min_history: int,
                                   backtest_start: str) -> pd.DataFrame:
    """
    Filter stocks that have sufficient history and no missing data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Raw price data
    min_history : int
        Minimum number of trading days required before backtest start
    backtest_start : str
        Date when backtest begins
        
    Returns
    -------
    pd.DataFrame
        Filtered price data
    """
    backtest_date = pd.to_datetime(backtest_start)
    
    valid_tickers = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        
        if len(series) == 0:
            continue
            
        # Check if ticker has data before backtest start
        pre_backtest = series[series.index < backtest_date]
        
        if len(pre_backtest) >= min_history:
            # Check for gaps during backtest period
            backtest_data = series[series.index >= backtest_date]
            
            # Allow up to 5% missing data
            if len(backtest_data) > 0:
                missing_pct = backtest_data.isna().sum() / len(backtest_data)
                if missing_pct < 0.05:
                    valid_tickers.append(ticker)
    
    print(f"Found {len(valid_tickers)} stocks with sufficient history")
    return prices[valid_tickers]


def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """
    Compute returns from price series.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    method : str
        'log' for log returns, 'simple' for arithmetic returns
        
    Returns
    -------
    pd.DataFrame
        Returns data
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns.dropna()


def compute_descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for return series.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
        
    Returns
    -------
    pd.DataFrame
        Statistics for each stock
    """
    stats = pd.DataFrame(index=returns.columns)
    
    stats['mean'] = returns.mean() * 252  # Annualized
    stats['std'] = returns.std() * np.sqrt(252)  # Annualized
    stats['skewness'] = returns.skew()
    stats['kurtosis'] = returns.kurtosis()  # Excess kurtosis
    stats['min'] = returns.min()
    stats['max'] = returns.max()
    stats['sharpe'] = stats['mean'] / stats['std']
    
    # Jarque-Bera test for normality
    from scipy import stats as scipy_stats
    jb_pvals = []
    for col in returns.columns:
        _, pval = scipy_stats.jarque_bera(returns[col].dropna())
        jb_pvals.append(pval)
    stats['jb_pvalue'] = jb_pvals
    
    return stats


def select_top_stocks(prices: pd.DataFrame, 
                      returns: pd.DataFrame,
                      n_stocks: int,
                      method: str = 'liquidity') -> list:
    """
    Select top N stocks based on specified criteria.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Returns data  
    n_stocks : int
        Number of stocks to select
    method : str
        Selection method: 'liquidity', 'variance', or 'random'
        
    Returns
    -------
    list
        Selected ticker symbols
    """
    if method == 'liquidity':
        # Proxy for liquidity: fewer missing values and higher average price
        scores = prices.count() * prices.mean()
        selected = scores.nlargest(n_stocks).index.tolist()
        
    elif method == 'variance':
        # Select stocks with highest variance (more interesting for tail analysis)
        variances = returns.var()
        selected = variances.nlargest(n_stocks).index.tolist()
        
    elif method == 'random':
        np.random.seed(42)
        selected = np.random.choice(prices.columns, size=n_stocks, replace=False).tolist()
        
    else:
        # Default: just take first n
        selected = prices.columns[:n_stocks].tolist()
    
    return selected


def prepare_dataset():
    """
    Main function to prepare the complete dataset.
    
    Returns
    -------
    tuple
        (prices, returns, selected_tickers, stats)
    """
    # Create data directory
    Path(DATA_DIR).mkdir(exist_ok=True)
    
    # Download data
    prices_raw = download_stock_data(NASDAQ100_TICKERS, START_DATE, END_DATE)
    
    # Filter by availability
    prices_filtered = filter_stocks_by_availability(
        prices_raw, MIN_HISTORY_DAYS, BACKTEST_START
    )
    
    # Compute returns
    returns = compute_returns(prices_filtered, method='log')
    
    # Select top stocks
    selected_tickers = select_top_stocks(
        prices_filtered, returns, NUM_STOCKS, method='liquidity'
    )
    
    # Final datasets
    prices_final = prices_filtered[selected_tickers]
    returns_final = returns[selected_tickers]
    
    # Forward fill any remaining small gaps
    prices_final = prices_final.ffill().bfill()
    returns_final = returns_final.fillna(0)
    
    # Compute statistics
    stats = compute_descriptive_stats(returns_final)
    
    # Save to files
    prices_final.to_csv(f"{DATA_DIR}/prices.csv")
    returns_final.to_csv(f"{DATA_DIR}/returns.csv")
    stats.to_csv(f"{DATA_DIR}/statistics.csv")
    
    print(f"\nDataset prepared:")
    print(f"  Stocks: {len(selected_tickers)}")
    print(f"  Date range: {prices_final.index[0].date()} to {prices_final.index[-1].date()}")
    print(f"  Trading days: {len(prices_final)}")
    print(f"\nFiles saved to {DATA_DIR}/")
    
    return prices_final, returns_final, selected_tickers, stats


if __name__ == "__main__":
    prices, returns, tickers, stats = prepare_dataset()
    
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS (Annualized)")
    print("="*60)
    print(stats[['mean', 'std', 'skewness', 'kurtosis', 'sharpe']].describe())
