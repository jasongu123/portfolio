"""
IBKR Strategy Backtester
========================
Backtests the EGARCH-based momentum strategy with proper metrics.

Key Features:
- Walk-forward validation (no look-ahead bias)
- Transaction costs (commission + slippage)
- Performance metrics (Sharpe, Sortino, Calmar, Max DD)
- Statistical significance testing
- Regime analysis

Data Sources:
- IBKR historical data via CSV (run extract_ibkr_data() first)
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from datetime import datetime, timedelta, time
import os
import pickle
import hashlib
import time as time_module
import warnings
warnings.filterwarnings('ignore')

# ML imports for Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Market hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
FIRST_TRADE_TIME = time(10, 0)  # Skip first 30 minutes
ETF_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA']
class ZarattiniSigmaForecaster:
    """
    HAR model to forecast tomorrow's Zarattini noise area.
    
    Model: σ_tomorrow = β0 + β_d·σ_today + β_w·σ_week + β_m·σ_month
    
    Where σ is the Zarattini noise area (average move from open),
    not returns volatility.
    """
    
    def __init__(self, min_history=30):
        self.model = None
        self.is_fitted = False
        self.min_history = min_history
        self.scaler_X = None
        self.scaler_y = None
    
    def fit(self, sigma_history):
        """
        Fit HAR model on historical noise areas.
        
        Args:
            sigma_history: Array of daily noise areas (chronological)
        
        Returns:
            dict with R², coefficients, and diagnostics
        """
        if len(sigma_history) < self.min_history:
            return None
        
        # Create DataFrame
        df = pd.DataFrame({'sigma': sigma_history})
        
        # HAR components
        df['sigma_d'] = df['sigma'].rolling(1).mean()   # Daily (yesterday)
        df['sigma_w'] = df['sigma'].rolling(5).mean()   # Weekly (last 5 days)
        df['sigma_m'] = df['sigma'].rolling(22).mean()  # Monthly (last 22 days)
        
        # Drop NaN
        df = df.dropna()
        
        if len(df) < 10:
            return None
        
        # Prepare regression (predict next day)
        X = df[['sigma_d', 'sigma_w', 'sigma_m']].values[:-1]
        y = df['sigma'].values[1:]
        
        # Standardize features for better numerical stability
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Fit OLS
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_scaled)
        
        # Calculate R²
        r2 = self.model.score(X_scaled, y_scaled)
        
        # Transform coefficients back to original scale
        # Unscaled coef = scaled_coef * (y_std / X_std)
        coef_unscaled = self.model.coef_ * (
            self.scaler_y.scale_[0] / self.scaler_X.scale_
        )
        intercept_unscaled = (
            self.model.intercept_ * self.scaler_y.scale_[0] + 
            self.scaler_y.mean_[0] -
            np.sum(coef_unscaled * self.scaler_X.mean_)
        )
        
        self.is_fitted = True
        
        return {
            'r2': r2,
            'coef_d': coef_unscaled[0],
            'coef_w': coef_unscaled[1],
            'coef_m': coef_unscaled[2],
            'intercept': intercept_unscaled,
            'n_samples': len(X)
        }
    
    def forecast(self, recent_sigmas):
        """
        Forecast tomorrow's sigma.
        
        Args:
            recent_sigmas: Array of recent noise areas (need 22+)
        
        Returns:
            Forecasted sigma for tomorrow
        """
        if not self.is_fitted:
            # Fallback to simple average
            return np.mean(recent_sigmas[-14:])
        
        if len(recent_sigmas) < 22:
            return np.mean(recent_sigmas)
        
        # Calculate components
        sigma_d = recent_sigmas[-1]
        sigma_w = np.mean(recent_sigmas[-5:])
        sigma_m = np.mean(recent_sigmas[-22:])
        
        # Create feature vector
        X = np.array([[sigma_d, sigma_w, sigma_m]])
        
        # Scale
        X_scaled = self.scaler_X.transform(X)
        
        # Predict (scaled)
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Unscale
        forecast = (
            y_pred_scaled[0] * self.scaler_y.scale_[0] + 
            self.scaler_y.mean_[0]
        )
        
        # Sanity check: forecast should be positive and reasonable
        if forecast < 0.001:
            forecast = 0.001
        if forecast > 0.10:
            forecast = 0.10
        
        return forecast

# Data directory for cached IBKR data
DATA_DIR = os.path.join(os.path.dirname(__file__), 'backtest_data')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'backtest_cache')


def extract_ibkr_data(
    symbols: list,
    years_back: int = 6,
    timeframe: str = '1 min',
    save_dir: str = None
):
    """
    Extract historical data from IBKR in chunks going backwards.

    For 1-minute data: uses 1-month chunks (IBKR limit ~30 days for 1-min bars)
    For 15-minute or larger: uses 1-year chunks

    Args:
        symbols: List of symbols to fetch
        years_back: How many years back to fetch (default 7)
        timeframe: Bar size ('1 min', '15 mins', '5 mins', '1 day', etc.)
        save_dir: Directory to save CSV files

    Usage:
        from ibkr_backtest import extract_ibkr_data
        extract_ibkr_data(['SPY'], years_back=5, timeframe='1 min')
    """
    import time as time_module

    # Import from your main trading file
    try:
        from ibkr2 import get_recent_bars, PTLClient, TRADING_PORT
    except Exception as e:
        print(f"Error importing from ibkr2.py: {e}")
        print("Make sure TWS/Gateway is running and ibkr2.py is in the same directory")
        return

    # Create our own client with a different ID to avoid conflicts
    print("Connecting to TWS...")
    client = PTLClient('127.0.0.1', TRADING_PORT, 200)
    time_module.sleep(2)

    if save_dir is None:
        save_dir = DATA_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Determine chunk size based on timeframe
    # IBKR limits: 1-min data ~30 days max per request, 15-min ~1 year
    is_1min = '1 min' in timeframe or timeframe == '1min'
    if is_1min:
        chunk_duration = '1 M'  # 1 month chunks for 1-minute data
        chunk_days = 30
        total_chunks = years_back * 12  # 12 months per year
        chunk_label_unit = 'month'
    else:
        chunk_duration = '1 Y'  # 1 year chunks for larger timeframes
        chunk_days = 365
        total_chunks = years_back
        chunk_label_unit = 'year'

    print(f"Extracting IBKR data to {save_dir}")
    print(f"Fetching {years_back} years back in {chunk_duration} chunks ({total_chunks} chunks total)")
    print(f"Timeframe: {timeframe}")
    print(f"Symbols: {symbols}")
    print("-" * 50)

    for symbol in symbols:
        try:
            print(f"\n{'='*50}")
            print(f"Fetching {symbol}...")
            print(f"{'='*50}")

            all_intraday = []

            failed_chunks = []  # Track failed chunks for retry

            # Fetch in chunks going backwards
            for i in range(total_chunks):
                # Calculate end_datetime for this chunk
                if i == 0:
                    end_dt = ""  # Now
                    chunk_label = f"now to 1 {chunk_label_unit} ago"
                else:
                    # End datetime is i chunks ago from today
                    end_date = datetime.now() - timedelta(days=i * chunk_days)
                    end_dt = end_date.strftime("%Y%m%d %H:%M:%S")
                    chunk_label = f"{i} {chunk_label_unit}s ago"

                print(f"\n  Chunk {i+1}/{total_chunks}: {chunk_label}")

                # Retry logic for transient failures
                df_chunk = pd.DataFrame()
                for attempt in range(3):  # Up to 3 attempts
                    df_chunk = get_recent_bars(
                        symbol,
                        timeframe,
                        chunk_duration,
                        for_calculation=True,
                        client_instance=client,
                        end_datetime=end_dt
                    )

                    if not df_chunk.empty:
                        break  # Success, exit retry loop

                    # Failed, wait longer and retry
                    if attempt < 2:
                        wait_time = 10 * (attempt + 1)  # 10s, 20s
                        print(f"    Retry {attempt + 1}/3 in {wait_time}s...")
                        time_module.sleep(wait_time)

                if not df_chunk.empty:
                    print(f"    Got {len(df_chunk)} bars")
                    all_intraday.append(df_chunk)
                else:
                    print(f"    FAILED after 3 attempts - will retry at end")
                    failed_chunks.append((i, end_dt, chunk_label))

                # Pacing - IBKR rate limits historical data requests
                # More aggressive pacing for 1-min data to avoid hitting limits
                if is_1min:
                    time_module.sleep(5)  # 5 seconds between 1-min requests
                else:
                    time_module.sleep(3)

            # Retry failed chunks with longer delays
            if failed_chunks:
                print(f"\n  Retrying {len(failed_chunks)} failed chunks...")
                time_module.sleep(30)  # Wait 30s before retrying

                for i, end_dt, chunk_label in failed_chunks:
                    print(f"\n  Retry chunk {i+1}: {chunk_label}")
                    df_chunk = get_recent_bars(
                        symbol,
                        timeframe,
                        chunk_duration,
                        for_calculation=True,
                        client_instance=client,
                        end_datetime=end_dt
                    )

                    if not df_chunk.empty:
                        print(f"    SUCCESS: Got {len(df_chunk)} bars")
                        all_intraday.append(df_chunk)
                    else:
                        print(f"    Still failed - skipping this chunk")

                    time_module.sleep(10)

            # Combine all chunks
            if all_intraday:
                df_intraday = pd.concat(all_intraday, ignore_index=True)
                df_intraday = df_intraday.drop_duplicates(subset=['date']).sort_values('date')

                # Use timeframe in filename (e.g., SPY_1m.csv, SPY_15m.csv)
                tf_suffix = timeframe.replace(' ', '').replace('mins', 'm').replace('min', 'm')
                intraday_file = os.path.join(save_dir, f'{symbol}_{tf_suffix}.csv')
                df_intraday.to_csv(intraday_file, index=False)
                print(f"\n  TOTAL: {len(df_intraday)} intraday bars saved")
                print(f"  Range: {df_intraday['date'].min()} to {df_intraday['date'].max()}")
            else:
                print(f"  No intraday data for {symbol}")
                continue

            # Fetch daily data (longer history available)
            print(f"\n  Fetching daily data...")
            df_daily = get_recent_bars(
                symbol,
                '1 day',
                '10 Y',
                for_calculation=True,
                client_instance=client
            )

            if not df_daily.empty:
                daily_file = os.path.join(save_dir, f'{symbol}_daily.csv')
                df_daily.to_csv(daily_file, index=False)
                print(f"  Saved {len(df_daily)} daily bars")

        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDisconnecting from TWS...")
    client.disconnect()

    print("\n" + "=" * 50)
    print("Data extraction complete!")
    print(f"Files saved to: {save_dir}")


def check_and_fill_gaps(
    symbols: list = None,
    timeframe: str = '1 min',
    save_dir: str = None,
    auto_fill: bool = True
):
    """
    Check for gaps in existing data and optionally fill them.

    Args:
        symbols: List of symbols to check. If None, checks SPY.
        timeframe: Bar size ('1 min', '15 mins', etc.)
        save_dir: Directory containing CSV files
        auto_fill: If True, automatically fetch missing data

    Usage:
        from ibkr_backtest import check_and_fill_gaps
        check_and_fill_gaps(['SPY'])  # Check and fill gaps
        check_and_fill_gaps(['SPY'], auto_fill=False)  # Just check, don't fill
    """
    import time as time_module

    if symbols is None:
        symbols = ['SPY']
    if save_dir is None:
        save_dir = DATA_DIR

    # Determine filename suffix
    tf_suffix = timeframe.replace(' ', '').replace('mins', 'm').replace('min', 'm')

    print("=" * 60)
    print("CHECKING DATA GAPS")
    print("=" * 60)

    for symbol in symbols:
        intraday_file = os.path.join(save_dir, f'{symbol}_{tf_suffix}.csv')

        if not os.path.exists(intraday_file):
            print(f"\n{symbol}: No data file found at {intraday_file}")
            continue

        # Load existing data
        df = pd.read_csv(intraday_file)
        df['datetime'] = pd.to_datetime(df['date'])
        df['year_month'] = df['datetime'].dt.to_period('M')
        df['date_only'] = df['datetime'].dt.date

        # Get date range
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()

        print(f"\n{symbol}:")
        print(f"  File: {intraday_file}")
        print(f"  Date range: {min_date.date()} to {max_date.date()}")
        print(f"  Total bars: {len(df):,}")

        # Check for monthly gaps
        all_months = pd.period_range(start=min_date, end=max_date, freq='M')
        existing_months = set(df['year_month'].unique())

        missing_months = []
        sparse_months = []

        for month in all_months:
            if month not in existing_months:
                missing_months.append(month)
            else:
                # Check if month has suspiciously few bars
                month_bars = len(df[df['year_month'] == month])
                # For 1-min data, expect ~20 trading days * 390 mins = ~7800 bars/month
                # For 15-min data, expect ~20 * 26 = ~520 bars/month
                is_1min = '1m' in tf_suffix
                expected_min = 5000 if is_1min else 300
                if month_bars < expected_min:
                    sparse_months.append((month, month_bars))

        # Report gaps
        if missing_months:
            print(f"\n  MISSING MONTHS ({len(missing_months)}):")
            for m in missing_months:
                print(f"    - {m}")
        else:
            print(f"\n  No missing months")

        if sparse_months:
            print(f"\n  SPARSE MONTHS (may have gaps):")
            for m, bars in sparse_months:
                print(f"    - {m}: only {bars} bars")

        # Daily gap analysis
        trading_dates = sorted(df['date_only'].unique())
        date_gaps = []
        for i in range(1, len(trading_dates)):
            prev_date = trading_dates[i-1]
            curr_date = trading_dates[i]
            gap_days = (curr_date - prev_date).days
            # More than 4 days suggests missing data (accounting for weekends)
            if gap_days > 4:
                date_gaps.append((prev_date, curr_date, gap_days))

        if date_gaps:
            print(f"\n  DATE GAPS (>{4} days):")
            for start, end, days in date_gaps[:10]:  # Show first 10
                print(f"    - {start} to {end} ({days} days)")
            if len(date_gaps) > 10:
                print(f"    ... and {len(date_gaps) - 10} more gaps")

        # Auto-fill missing months
        if auto_fill and (missing_months or sparse_months):
            print(f"\n  Filling gaps...")

            try:
                from ibkr2 import get_recent_bars, PTLClient, TRADING_PORT
            except Exception as e:
                print(f"  Error importing from ibkr2.py: {e}")
                continue

            client = PTLClient('127.0.0.1', TRADING_PORT, 201)
            time_module.sleep(2)

            all_new_data = []
            months_to_fill = list(missing_months) + [m for m, _ in sparse_months]

            for month in months_to_fill:
                # End datetime is last day of the month
                end_date = month.to_timestamp() + pd.offsets.MonthEnd(1)
                end_dt = end_date.strftime("%Y%m%d 23:59:59")

                print(f"\n    Fetching {month}...")

                df_chunk = pd.DataFrame()
                for attempt in range(3):
                    df_chunk = get_recent_bars(
                        symbol,
                        timeframe,
                        '1 M',
                        for_calculation=True,
                        client_instance=client,
                        end_datetime=end_dt
                    )

                    if not df_chunk.empty:
                        break

                    if attempt < 2:
                        print(f"      Retry {attempt + 1}/3...")
                        time_module.sleep(10)

                if not df_chunk.empty:
                    print(f"      Got {len(df_chunk)} bars")
                    all_new_data.append(df_chunk)
                else:
                    print(f"      Failed to fetch")

                time_module.sleep(5)

            # Merge with existing data
            if all_new_data:
                print(f"\n    Merging {len(all_new_data)} chunks with existing data...")
                new_df = pd.concat(all_new_data, ignore_index=True)

                # Combine with existing - ensure date column is string type for consistency
                df['date'] = df['date'].astype(str)
                new_df['date'] = new_df['date'].astype(str)
                combined = pd.concat([df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date'])
                combined['date'] = pd.to_datetime(combined['date'])
                combined = combined.sort_values('date')
                combined['date'] = combined['date'].astype(str)  # Back to string for CSV

                # Save
                combined.to_csv(intraday_file, index=False)
                print(f"    Saved: {len(df):,} -> {len(combined):,} bars (+{len(combined) - len(df):,})")

            client.disconnect()

    print("\n" + "=" * 60)
    print("Gap check complete!")


def extract_specific_year(
    symbols: list,
    year: int,
    timeframe: str = '15 mins',
    save_dir: str = None
):
    """
    Extract data for a specific year and merge with existing CSV.

    Usage:
        from ibkr_backtest import extract_specific_year
        extract_specific_year(['NVDA', 'SPY'], 2021)
    """
    import time as time_module

    try:
        from ibkr2 import get_recent_bars, PTLClient, TRADING_PORT
    except Exception as e:
        print(f"Error importing from ibkr2.py: {e}")
        return

    print(f"Connecting to TWS...")
    client = PTLClient('127.0.0.1', TRADING_PORT, 200)
    time_module.sleep(2)

    if save_dir is None:
        save_dir = DATA_DIR

    print(f"Extracting {year} data for {symbols}")
    print("-" * 50)

    # End datetime is Dec 31 of that year
    end_dt = f"{year}1231 23:59:59"

    for symbol in symbols:
        try:
            print(f"\nFetching {symbol} for {year}...")

            df_chunk = get_recent_bars(
                symbol,
                timeframe,
                '1 Y',
                for_calculation=True,
                client_instance=client,
                end_datetime=end_dt
            )

            if df_chunk.empty:
                print(f"  No data returned for {symbol} {year}")
                continue

            print(f"  Got {len(df_chunk)} bars")

            # Load existing data and merge
            intraday_file = os.path.join(save_dir, f'{symbol}_15m.csv')
            if os.path.exists(intraday_file):
                existing = pd.read_csv(intraday_file)
                # Ensure date columns are same type (string) for comparison
                existing['date'] = existing['date'].astype(str)
                df_chunk['date'] = df_chunk['date'].astype(str)
                combined = pd.concat([existing, df_chunk], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date']).sort_values('date')
                combined.to_csv(intraday_file, index=False)
                print(f"  Merged: {len(existing)} + {len(df_chunk)} -> {len(combined)} bars")
            else:
                df_chunk.to_csv(intraday_file, index=False)
                print(f"  Saved {len(df_chunk)} bars")

            time_module.sleep(3)

        except Exception as e:
            print(f"  Error: {e}")

    print("\nDisconnecting...")
    client.disconnect()
    print("Done!")


def check_data_coverage(symbols: list = None, save_dir: str = None, auto_extract: bool = True):
    """
    Check data coverage by year for each symbol. Auto-extracts missing years.

    Usage:
        from ibkr_backtest import check_data_coverage
        check_data_coverage()  # Check and auto-extract missing
        check_data_coverage(auto_extract=False)  # Just check, don't extract
    """
    if save_dir is None:
        save_dir = DATA_DIR
    if symbols is None:
        symbols = ['NVDA', 'JPM', 'GOOG', 'WMT', 'QQQ', 'SPY']

    print("Data coverage by year:")
    print("=" * 60)

    missing = []  # List of (symbol, year) tuples

    for symbol in symbols:
        intraday_file = os.path.join(save_dir, f'{symbol}_15m.csv')
        if not os.path.exists(intraday_file):
            print(f"{symbol}: No data file")
            continue

        df = pd.read_csv(intraday_file)
        df['datetime'] = pd.to_datetime(df['date'])
        df['year'] = df['datetime'].dt.year
        df['date_only'] = df['datetime'].dt.date

        days_per_year = df.groupby('year')['date_only'].nunique()
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())

        print(f"\n{symbol}:")
        for year, days in days_per_year.items():
            year = int(year)
            if year == min_year or year == max_year:
                status = "✓ (partial)"
            elif days < 100:
                status = "❌ MISSING"
                missing.append((symbol, year))
            else:
                status = "✓"
            print(f"  {year}: {days:3d} days {status}")

    if missing and auto_extract:
        print(f"\n{'='*60}")
        print(f"Auto-extracting {len(missing)} missing symbol+year pairs:")
        for sym, yr in missing:
            print(f"  {sym} {yr}")
        print("=" * 60)

        # Group by year to minimize connections
        from collections import defaultdict
        by_year = defaultdict(list)
        for sym, yr in missing:
            by_year[yr].append(sym)

        for year in sorted(by_year.keys()):
            syms = by_year[year]
            print(f"\n>>> Extracting {year} for {syms}...")
            extract_specific_year(syms, year, save_dir=save_dir)

        print("\n" + "=" * 60)
        print("Re-checking coverage after extraction:")
        check_data_coverage(symbols, save_dir, auto_extract=False)
    elif missing:
        print(f"\nMissing: {missing}")
        print("Run with auto_extract=True to fetch them.")


def extract_vix_data(years_back: int = 6, save_dir: str = None):
    """
    Extract VIX daily data from IBKR.

    VIX is an index, so it uses a different contract type.

    Usage:
        from ibkr_backtest import extract_vix_data
        extract_vix_data(years_back=6)
    """
    import time as time_mod

    try:
        from ibkr2 import get_recent_bars, PTLClient, TRADING_PORT
    except Exception as e:
        print(f"Error importing from ibkr2.py: {e}")
        return

    print("Connecting to TWS for VIX data...")
    client = PTLClient('127.0.0.1', TRADING_PORT, 202)
    time_mod.sleep(2)

    if save_dir is None:
        save_dir = DATA_DIR

    os.makedirs(save_dir, exist_ok=True)

    print(f"Extracting VIX data ({years_back} years)...")

    # VIX daily data - fetch in 1-year chunks
    all_vix_data = []

    for i in range(years_back):
        if i == 0:
            end_dt = ""
        else:
            end_date = datetime.now() - timedelta(days=i * 365)
            end_dt = end_date.strftime("%Y%m%d %H:%M:%S")

        print(f"  Fetching year {i+1}/{years_back}...")

        df_chunk = get_recent_bars(
            'VIX',
            '1 day',
            '1 Y',
            for_calculation=True,
            client_instance=client,
            end_datetime=end_dt
        )

        if not df_chunk.empty:
            print(f"    Got {len(df_chunk)} bars")
            all_vix_data.append(df_chunk)
        else:
            print(f"    No data returned")

        time_mod.sleep(3)

    if all_vix_data:
        df_vix = pd.concat(all_vix_data, ignore_index=True)
        df_vix = df_vix.drop_duplicates(subset=['date']).sort_values('date')

        vix_file = os.path.join(save_dir, 'VIX_daily.csv')
        df_vix.to_csv(vix_file, index=False)
        print(f"\nSaved {len(df_vix)} VIX daily bars to {vix_file}")
        print(f"Range: {df_vix['date'].min()} to {df_vix['date'].max()}")

    client.disconnect()
    print("VIX extraction complete!")


def load_csv_data(symbols: list, data_dir: str = None) -> dict:
    """
    Load historical data from CSV files.

    Args:
        symbols: List of symbols to load
        data_dir: Directory containing CSV files

    Returns:
        Dict of {symbol: {'intraday': df, 'daily': df}}
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data = {}

    for symbol in symbols:
        # Try 1-minute data first (paper spec), then fall back to 15-minute
        intraday_file_1m = os.path.join(data_dir, f'{symbol}_1m.csv')
        daily_file = os.path.join(data_dir, f'{symbol}_daily.csv')

        if os.path.exists(intraday_file_1m):
            intraday_file = intraday_file_1m
            print(f"  {symbol}: Using 1-minute data (paper spec)")
        else:
            print(f"  Warning: No intraday data for {symbol}")
            continue

        try:
            # Load intraday
            df_intraday = pd.read_csv(intraday_file)
            df_intraday['datetime'] = pd.to_datetime(df_intraday['date'])
            df_intraday['date'] = df_intraday['datetime'].dt.date
            df_intraday['time'] = df_intraday['datetime'].dt.time
            df_intraday['symbol'] = symbol

            # Calculate minutes from market open
            df_intraday['minutes_from_open'] = (
                df_intraday['datetime'].dt.hour * 60 +
                df_intraday['datetime'].dt.minute -
                (9 * 60 + 30)
            )

            # Calculate VWAP
            df_intraday['typical_price'] = (
                df_intraday['high'] + df_intraday['low'] + df_intraday['close']
            ) / 3
            df_intraday['tp_volume'] = df_intraday['typical_price'] * df_intraday['volume']
            df_intraday['cum_tp_vol'] = df_intraday.groupby('date')['tp_volume'].cumsum()
            df_intraday['cum_vol'] = df_intraday.groupby('date')['volume'].cumsum()
            df_intraday['vwap'] = df_intraday['cum_tp_vol'] / df_intraday['cum_vol']
            df_intraday['vwap'] = df_intraday['vwap'].fillna(df_intraday['close'])

            # Load daily
            if os.path.exists(daily_file):
                df_daily = pd.read_csv(daily_file)
                
                # Robust date parsing - handle multiple formats
                date_col = df_daily['date']
                if date_col.dtype == 'int64' or date_col.dtype == 'float64':
                    # Check if it looks like Unix timestamp (large numbers)
                    sample = date_col.iloc[0]
                    if sample > 1e9:  # Unix timestamp in seconds (after 2001)
                        df_daily['date'] = pd.to_datetime(date_col, unit='s')
                    elif sample > 1e6:  # Unix timestamp in milliseconds
                        df_daily['date'] = pd.to_datetime(date_col, unit='ms')
                    else:
                        # Could be days since some epoch (like Excel)
                        # Try parsing as string first
                        df_daily['date'] = pd.to_datetime(date_col.astype(str))
                else:
                    # String format - try multiple approaches
                    try:
                        df_daily['date'] = pd.to_datetime(date_col)
                    except:
                        # Try with explicit format
                        df_daily['date'] = pd.to_datetime(date_col, format='%Y-%m-%d', errors='coerce')
                
                # Validate dates are reasonable (2010-2030)
                first_date = df_daily['date'].iloc[0]
                if first_date.year < 2010 or first_date.year > 2030:
                    print(f"  WARNING: Daily dates look wrong (first: {first_date})")
                    print(f"    Raw date column sample: {date_col.iloc[:3].tolist()}")
                    print(f"    Attempting to create daily from intraday instead...")
                    # Fall back to creating from intraday
                    df_daily = df_intraday.groupby('date').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).reset_index()
                    df_daily['date'] = pd.to_datetime(df_daily['date'])
                
                df_daily['daily_return'] = df_daily['close'].pct_change()
                df_daily['daily_log_return'] = np.log(df_daily['close'] / df_daily['close'].shift(1))
            else:
                # Create daily from intraday if no daily file
                df_daily = df_intraday.groupby('date').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()
                df_daily['date'] = pd.to_datetime(df_daily['date'])
                df_daily['daily_return'] = df_daily['close'].pct_change()
                df_daily['daily_log_return'] = np.log(df_daily['close'] / df_daily['close'].shift(1))

            data[symbol] = {
                'intraday': df_intraday,
                'daily': df_daily
            }

            print(f"  {symbol}: {len(df_intraday)} intraday bars, {len(df_daily)} daily bars")

        except Exception as e:
            print(f"  Error loading {symbol}: {e}")

    return data


class StrategyBacktester:
    """
    Backtests the EGARCH momentum strategy with noise boundaries.
    Matches original school project methodology.
    """

    def __init__(
        self,
        symbols: list,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        # Strategy parameters (matching Zarattini paper)
        lag: int = 14,
        target_vol: float = 0.02,  # Paper: 2% daily target volatility
        volatility_multiplier: float = 1.0,
        p_terms: int = 1,
        o_terms: int = 0,
        q_terms: int = 1,
        max_leverage: float = 4.0,  # Paper: 4x leverage cap
        trade_freq: int = 1,  # Check every bar (paper: continuous monitoring)
        # Transaction costs
        commission_per_share: float = 0.0035,
        slippage_bps: float = 1,
        # RF parameters
        use_rf: bool = True,
        rf_min_samples: int = 50,
        rf_confidence_threshold: float = 0.65,
        rf_walk_forward_window: int = 200,  # Rolling window for walk-forward training
        # VIX filter: only trade ETFs when VIX is above minimum (paper says higher VIX = better)
        # NOTE: This filter only applies to ETFs. Stocks trade regardless of VIX level.
        vix_min: float = 17,  # Minimum VIX to trade ETFs
        vix_max: float = None,  # Maximum VIX to trade (for bucket analysis)
        # Danger zone parameters (VIX 15-17 is historically toxic)
        skip_danger_zone: bool = False,  # Skip VIX danger zone entirely
        danger_zone_min: float = 15,  # Lower bound of danger zone
        danger_zone_max: float = 17,  # Upper bound of danger zone
        # Microstructure filters
        early_stop_pct: float = 0,  # Exit if loss exceeds X% within first 30 min (0=disabled, e.g. 0.003 = 0.3%)
        last_entry_time: str = "15:50",  # No new entries after this time (gamma pinning)
        skip_opex: bool = False,  # Skip monthly OPEX (3rd Friday) and triple witching
        # Exit: Paper uses max(upper_bound, vwap) as trailing stop - no other stops
        # Swing strategy parameters (for low VIX regimes)
        use_swing: bool = True,  # Enable swing strategy when VIX < vix_min
        swing_momentum_lookback: int = 12,  # Days for momentum calculation
        swing_momentum_threshold: float = 0.01,  # 1% min momentum to enter
        swing_sma_period: int = 20,  # Trend filter
        swing_atr_period: int = 14,  # For trailing stop
        swing_atr_multiplier: float = 2.0,  # Trailing stop = 2x ATR
        swing_min_hold_days: int = 3,  # Minimum holding period
        swing_max_hold_days: int = 30,  # Maximum holding period
        # Enhanced swing filters (Option B: stricter entry)
        swing_strict_mode: bool = False,  # Enable strict swing entry conditions
        swing_max_vix: float = 15.0,  # Only swing when VIX < this (very calm)
        swing_min_momentum: float = 0.03,  # 3% momentum required (was 1%)
        swing_min_autocorr: float = 0.05,  # Positive autocorrelation required (trending)
        swing_wider_stops: bool = False,  # Use 3x ATR instead of 2x in low vol
        # HMM Regime Detection
        use_hmm_regime: bool = True,  # Enable HMM-based regime detection
        hmm_n_states: int = 3,  # Number of hidden states
        hmm_lookback: int = 252,  # Days of history for HMM training
        hmm_retrain_frequency: int = 20,  # Retrain every N days
        use_har_forecast: bool = True,  # Enable HAR forecasting
        har_refit_frequency: int = 20,
        # VIX-adaptive parameters
        use_vix_adaptive: bool = False,  # Enable VIX-based parameter adaptation
        vix_adaptive_mode: str = 'moderate',  # 'conservative', 'moderate', 'aggressive'
        # Adaptive VIX threshold parameters
        use_adaptive_threshold: bool = False,  # Enable rolling Sharpe-based threshold
        adaptive_lookback: int = 30,  # Days for rolling Sharpe calculation
        adaptive_thresholds: dict = None,  # {sharpe_cutoff: vix_threshold}
        # Autocorrelation-aware VIX filter (smart regime detection)
        use_autocorr_filter: bool = False,  # Override VIX filter if market is trending
        autocorr_lookback: int = 20,  # Days for autocorrelation calculation
        autocorr_threshold: float = 0.05,  # Min autocorr to consider "trending"
        # Stock-specific strategy parameters (for individual stocks, not ETFs)
        use_stock_strategies: bool = True,  # Enable stock-specific strategies (mean reversion, gap capture)
        stock_strategy_mode: str = 'hybrid',  # 'mean_reversion', 'gap_capture', 'hybrid', 'vol_adaptive'
        gap_threshold: float = 0.015,  # Minimum gap % to trade (1.5%)
        gap_exit_minutes: int = 60,  # Exit gap trades after N minutes (e.g., 60 = exit by 10:30)
        use_stock_vol_regime: bool = True,  # Use stock's own vol instead of VIX
        vol_momentum_threshold: float = 0.2,  # Vol rank threshold to switch momentum (20% above avg)
        vol_lookback: int = 20,  # Days for stock vol calculation
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # Strategy params
        self.lag = lag
        self.target_vol = target_vol
        self.volatility_multiplier = volatility_multiplier
        self.p_terms = p_terms
        self.o_terms = o_terms
        self.q_terms = q_terms
        self.max_leverage = max_leverage
        self.trade_freq = trade_freq

        # Costs
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

        # RF parameters
        self.use_rf = use_rf
        self.rf_min_samples = rf_min_samples
        self.rf_confidence_threshold = rf_confidence_threshold
        self.rf_walk_forward_window = rf_walk_forward_window
        self.rf_model = None
        self.rf_scaler = StandardScaler()
        self.rf_training_data = []
        self.rf_val_scores = []  # Track walk-forward validation scores

        # VIX filter
        self.vix_min = vix_min
        self.vix_max = vix_max  # Max VIX for bucket analysis
        self.vix_data = {}  # {date: vix_close}
        
        # Danger zone parameters (VIX 15-17 is historically toxic)
        self.skip_danger_zone = skip_danger_zone
        self.danger_zone_min = danger_zone_min
        self.danger_zone_max = danger_zone_max

        # Swing strategy parameters
        self.use_swing = use_swing
        self.swing_momentum_lookback = swing_momentum_lookback
        self.swing_momentum_threshold = swing_momentum_threshold
        self.swing_sma_period = swing_sma_period
        self.swing_atr_period = swing_atr_period
        self.swing_atr_multiplier = swing_atr_multiplier
        self.swing_min_hold_days = swing_min_hold_days
        self.swing_max_hold_days = swing_max_hold_days
        
        # Enhanced swing filters (Option B)
        self.swing_strict_mode = swing_strict_mode
        self.swing_max_vix = swing_max_vix
        self.swing_min_momentum = swing_min_momentum
        self.swing_min_autocorr = swing_min_autocorr
        self.swing_wider_stops = swing_wider_stops
        
        # HMM Regime Detection
        self.use_hmm_regime = use_hmm_regime
        self.hmm_n_states = hmm_n_states
        self.hmm_lookback = hmm_lookback
        self.hmm_retrain_frequency = hmm_retrain_frequency
        self.hmm_model = None
        self.hmm_trained = False
        self.hmm_last_train_idx = 0
        self.hmm_regime_history = []  # Track regime predictions
        self.current_hmm_regime = None

        # Swing position tracking (persists across days)
        self.swing_positions = {}  # {symbol: {quantity, entry_price, entry_date, highest_price, trailing_stop, atr}}
        self.swing_trades = []  # Track swing trades separately

        # Microstructure filters
        self.early_stop_pct = early_stop_pct
        self.last_entry_time = datetime.strptime(last_entry_time, "%H:%M").time()
        self.skip_opex = skip_opex

        # Monthly allocation tracking (rebalanced monthly based on prev month vol)
        self.allocations = {s: 1.0 / len(symbols) for s in symbols}
        print(self.allocations)
        self.current_month = None

        # Pre-computed statistics (like original project)
        self.daily_returns = {}  # {symbol: {date: return}}
        self.daily_log_returns = {}
        self.open_returns = {}  # First 30 min return
        self.dvol = {}  # Rolling daily volatility
        self.egarch_vol = {}  # EGARCH forecast
        self.prev_month_vol = {}  # Previous month's volatility
        self.regression_params = {}  # {symbol: {month: (intercept, coef)}}
        self.noise_area = {}  # {symbol: {date: avg_move}} - Zarattini 14-day avg move from open
        self.daily_move_from_open = {}  # {symbol: {date: |close - open|}}
        self.intraday_sigma_cache = {}  # {symbol: {date_str: {minute: sigma}}} - PRE-COMPUTED
        # HAR forecasting
        self.use_har_forecast = use_har_forecast
        self.har_refit_frequency = har_refit_frequency
        self.har_forecaster = {}  # {symbol: ZarattiniSigmaForecaster()}
        self.har_sigma_history = {}  # {symbol: [daily_sigmas]}
        self.har_forecasts = {}  # {symbol: {date: forecast}}
        self.har_diagnostics = {}  # Track model quality over time
        # VIX-adaptive parameters
        self.use_vix_adaptive = use_vix_adaptive
        self.vix_adaptive_mode = vix_adaptive_mode
        
        # Adaptive VIX threshold based on rolling Sharpe
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_lookback = adaptive_lookback
        # Default thresholds: {min_sharpe: vix_threshold}
        # If rolling Sharpe < -0.5, use VIX >= 25 (very selective)
        # If rolling Sharpe < 0, use VIX >= 20 (selective)
        # If rolling Sharpe < 0.5, use VIX >= 17 (normal)
        # Otherwise use VIX >= 12 (aggressive)
        self.adaptive_thresholds = adaptive_thresholds or {
            -0.5: 25,   # Very hostile market - only trade high VIX
            0.0: 20,    # Hostile market - be selective
            0.5: 17,    # Normal market - standard threshold
            float('inf'): 12  # Good market - trade aggressively
        }
        self.rolling_returns = []  # Track daily returns for rolling Sharpe
        self.current_adaptive_threshold = self.vix_min  # Start with default
        
        # Autocorrelation-aware VIX filter
        # If autocorr > threshold, market is trending - trade even on low VIX
        # If autocorr < threshold, market is choppy - apply VIX filter
        self.use_autocorr_filter = use_autocorr_filter
        self.autocorr_lookback = autocorr_lookback
        self.autocorr_threshold = autocorr_threshold
        self.current_market_autocorr = None  # Track for debugging
        
        # Regime thresholds and parameter mappings
        self.vix_regimes = {
            'conservative': {
                'crisis': {'vix': 30, 'lookback': 14, 'vm': 2.0},
                'elevated': {'vix': 20, 'lookback': 30, 'vm': 1.5},
                'normal': {'vix': 15, 'lookback': 60, 'vm': 1.2},
                'low': {'vix': 0, 'lookback': 90, 'vm': 1.0}
            },
            'moderate': {
                'crisis': {'vix': 30, 'lookback': 10, 'vm': 2.5},
                'elevated': {'vix': 20, 'lookback': 20, 'vm': 1.8},
                'normal': {'vix': 15, 'lookback': 45, 'vm': 1.3},
                'low': {'vix': 0, 'lookback': 90, 'vm': 1.0}
            },
            'aggressive': {
                'crisis': {'vix': 30, 'lookback': 7, 'vm': 3.0},
                'elevated': {'vix': 20, 'lookback': 14, 'vm': 2.0},
                'normal': {'vix': 15, 'lookback': 30, 'vm': 1.5},
                'low': {'vix': 0, 'lookback': 60, 'vm': 1.0}
            }
        }

        # Stock-specific strategy parameters
        self.use_stock_strategies = use_stock_strategies
        self.stock_strategy_mode = stock_strategy_mode
        self.gap_threshold = gap_threshold
        self.gap_exit_minutes = gap_exit_minutes
        self.use_stock_vol_regime = use_stock_vol_regime
        self.vol_momentum_threshold = vol_momentum_threshold
        self.vol_lookback = vol_lookback

        # Stock volatility regime cache
        self.stock_vol_regimes = {}  # {symbol: {date: (current_vol, vol_rank)}}

        # Gap trade tracking
        self.gap_positions = {}  # {symbol: {entry_time, entry_price, shares}}

        # Data storage
        self.data = {}
        self.results = None
        self.backtest_daily_returns = None

    def load_from_csv(self, data_dir: str = None):
        """
        Load historical data from CSV files (extracted from IBKR).

        Args:
            data_dir: Directory containing CSV files. If None, uses default.

        Usage:
            bt = StrategyBacktester(symbols, start, end)
            bt.load_from_csv()  # Load from default directory
            bt.run_backtest()
        """
        print(f"Loading data from CSV files...")

        self.data = load_csv_data(self.symbols, data_dir)

        # Filter to date range
        for symbol in list(self.data.keys()):
            intraday = self.data[symbol]['intraday']

            # Filter intraday to date range
            start_date = pd.to_datetime(self.start_date).date()
            end_date = pd.to_datetime(self.end_date).date()

            mask = (intraday['date'] >= start_date) & (intraday['date'] <= end_date)
            self.data[symbol]['intraday'] = intraday[mask].copy()

            print(f"  {symbol}: Filtered to {len(self.data[symbol]['intraday'])} bars "
                  f"({start_date} to {end_date})")

        # Load VIX data (always load if file exists - needed for diagnostics)
        vix_file = os.path.join(data_dir or DATA_DIR, 'VIX_daily.csv')
        if os.path.exists(vix_file):
            df_vix = pd.read_csv(vix_file)
            df_vix['date'] = pd.to_datetime(df_vix['date']).dt.date
            for _, row in df_vix.iterrows():
                self.vix_data[row['date']] = row['close']
            print(f"  VIX: Loaded {len(self.vix_data)} days")
        else:
            print(f"  Warning: No VIX data found at {vix_file}")
            print(f"  To extract VIX: python -c \"from ibkr_backtest import extract_vix_data; extract_vix_data()\"")

        # Pre-compute statistics like original project
        self._precompute_statistics()

        return self

    def _precompute_statistics(self):
        """
        Pre-compute all statistics needed for the strategy.
        Matches original school project methodology exactly.
        """
        print("  Pre-computing statistics (EGARCH, regression, volatility)...")

        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            intraday = self.data[symbol]['intraday']
            daily_groups = intraday.groupby('date')
            all_days = sorted(intraday['date'].unique())

            # Initialize storage for this symbol
            self.daily_returns[symbol] = {}
            self.daily_log_returns[symbol] = {}
            self.open_returns[symbol] = {}
            self.dvol[symbol] = {}
            self.egarch_vol[symbol] = {}
            self.prev_month_vol[symbol] = {}
            self.regression_params[symbol] = {}
            self.noise_area[symbol] = {}
            self.daily_move_from_open[symbol] = {}

            # Calculate daily statistics
            for d in range(1, len(all_days)):
                current_day = all_days[d]
                prev_day = all_days[d - 1]

                try:
                    current_day_data = daily_groups.get_group(current_day)
                    prev_day_data = daily_groups.get_group(prev_day)

                    # Daily return
                    daily_ret = current_day_data['close'].iloc[-1] / prev_day_data['close'].iloc[-1] - 1
                    daily_log_ret = np.log(current_day_data['close'].iloc[-1] / prev_day_data['close'].iloc[-1])
                    self.daily_returns[symbol][current_day] = daily_ret
                    self.daily_log_returns[symbol][current_day] = daily_log_ret

                    # Open return (first 30 min return from prev close)
                    at_30min = current_day_data[current_day_data['minutes_from_open'] >= 30]
                    if not at_30min.empty:
                        open_ret = at_30min.iloc[0]['close'] / prev_day_data['close'].iloc[-1] - 1
                    else:
                        # Fallback to 90 min if 30 min not available
                        at_90min = current_day_data[current_day_data['minutes_from_open'] >= 90]
                        if not at_90min.empty:
                            open_ret = at_90min.iloc[0]['close'] / prev_day_data['close'].iloc[-1] - 1
                        else:
                            open_ret = 0.0
                    self.open_returns[symbol][current_day] = open_ret

                    # Zarattini noise area: |close - open| / open (daily move from open as %)
                    day_open = current_day_data['open'].iloc[0]
                    day_close = current_day_data['close'].iloc[-1]
                    move_from_open = abs(day_close - day_open) / day_open
                    self.daily_move_from_open[symbol][current_day] = move_from_open

                    # Calculate 14-day rolling average of moves from open (Zarattini noise area)
                    if d >= self.lag:
                        recent_days = all_days[d - self.lag:d]
                        recent_moves = [self.daily_move_from_open[symbol].get(day, 0) for day in recent_days
                                        if day in self.daily_move_from_open[symbol]]
                        if recent_moves:
                            self.noise_area[symbol][current_day] = np.mean(recent_moves)

                    # Rolling daily volatility (dvol) over lag days
                    # Paper uses 15 days (d-15:d-1), excluding current day for walk-forward validation
                    if d > 14:  # Need at least 15 days of history
                        recent_days = all_days[d - 15:d - 1]  # 14 days of returns, excluding today
                        recent_returns = [self.daily_returns[symbol].get(day, 0) for day in recent_days
                                          if day in self.daily_returns[symbol]]
                        if len(recent_returns) >= 14:
                            self.dvol[symbol][current_day] = np.std(recent_returns)

                            # EGARCH volatility forecast
                            training_returns = np.array(recent_returns) * 100  # Scale for arch
                            try:
                                model = arch_model(
                                    training_returns,
                                    vol='GARCH',
                                    p=self.p_terms,
                                    o=self.o_terms,
                                    q=self.q_terms
                                )
                                res = model.fit(disp='off', show_warning=False, options={'maxiter': 100})
                                forecast = res.forecast(horizon=1)
                                vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
                                self.egarch_vol[symbol][current_day] = vol_forecast
                            except Exception:
                                self.egarch_vol[symbol][current_day] = self.dvol[symbol][current_day]

                except Exception as e:
                    continue

            # Calculate monthly statistics (prev_month_vol and regression params)
            intraday['year_month'] = pd.to_datetime(intraday['datetime']).dt.strftime('%Y-%m')
            monthly_groups = intraday.groupby('year_month')
            all_months = sorted(intraday['year_month'].unique())

            for m in range(1, len(all_months)):
                current_month = all_months[m]
                prev_month = all_months[m - 1]

                try:
                    prev_month_data = monthly_groups.get_group(prev_month)
                    prev_month_days = sorted(prev_month_data['date'].unique())

                    # Previous month volatility (std of daily returns)
                    prev_month_returns = [self.daily_returns[symbol].get(day, 0) for day in prev_month_days
                                          if day in self.daily_returns[symbol]]
                    if prev_month_returns:
                        self.prev_month_vol[symbol][current_month] = np.std(prev_month_returns)

                    # Regression: open_return predicts daily_return (previous month data)
                    X = [self.open_returns[symbol].get(day, 0) for day in prev_month_days
                         if day in self.open_returns[symbol] and day in self.daily_returns[symbol]]
                    y = [self.daily_returns[symbol].get(day, 0) for day in prev_month_days
                         if day in self.open_returns[symbol] and day in self.daily_returns[symbol]]

                    if len(X) >= 5:
                        X = np.array(X).reshape(-1, 1)
                        y = np.array(y)
                        # Simple linear regression
                        X_mean = X.mean()
                        y_mean = y.mean()
                        denom = np.sum((X.flatten() - X_mean) ** 2)
                        if denom > 1e-10:
                            coef = np.sum((X.flatten() - X_mean) * (y - y_mean)) / denom
                            intercept = y_mean - coef * X_mean
                            self.regression_params[symbol][current_month] = (intercept, coef)
                        else:
                            self.regression_params[symbol][current_month] = (0.0, 0.0)
                    else:
                        self.regression_params[symbol][current_month] = (0.0, 0.0)

                except Exception as e:
                    continue

        # ============================================================
        # PRE-COMPUTE INTRADAY SIGMA FOR ALL (date, minute) PAIRS
        # This is the expensive calculation - do it once upfront!
        # Uses DISK CACHING to avoid recomputation on subsequent runs
        # ============================================================
        self._precompute_intraday_sigma_with_cache()

        print("  Statistics pre-computed.")

    def _get_cache_hash(self, symbol: str) -> str:
        """Generate a hash based on data characteristics to validate cache."""
        if symbol not in self.data:
            return ""
        intraday = self.data[symbol]['intraday']
        # Hash based on: row count, date range, lag parameter
        hash_input = f"{len(intraday)}_{intraday['date'].min()}_{intraday['date'].max()}_{self.lag}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _load_sigma_cache(self, symbol: str) -> bool:
        """Try to load sigma cache from disk. Returns True if successful."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f'{symbol}_sigma_cache.pkl')

        if not os.path.exists(cache_file):
            return False

        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)

            # Validate cache hash matches current data
            expected_hash = self._get_cache_hash(symbol)
            if cached.get('hash') != expected_hash:
                print(f"    {symbol}: Cache outdated (data changed), will recompute")
                return False

            self.intraday_sigma_cache[symbol] = cached['data']
            print(f"    {symbol}: Loaded {len(cached['data'])} days from disk cache")
            
            # FIXED: Populate HAR sigma history when loading from cache
            if self.use_har_forecast:
                if symbol not in self.har_sigma_history:
                    self.har_sigma_history[symbol] = []
                
                # Calculate daily average sigmas from cached data
                for date_str, minute_sigmas in cached['data'].items():
                    if minute_sigmas:
                        avg_sigma = np.mean(list(minute_sigmas.values()))
                        trade_date = pd.to_datetime(date_str).date()
                        self.har_sigma_history[symbol].append({
                            'date': trade_date,
                            'sigma': avg_sigma
                        })
                
                # Sort by date (HAR needs chronological order)
                self.har_sigma_history[symbol].sort(key=lambda x: x['date'])
                print(f"    {symbol}: Populated {len(self.har_sigma_history[symbol])} days for HAR from cache")
            
            return True

        except Exception as e:
            print(f"    {symbol}: Cache load failed ({e}), will recompute")
            return False

    def _save_sigma_cache(self, symbol: str):
        """Save sigma cache to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f'{symbol}_sigma_cache.pkl')

        try:
            cached = {
                'hash': self._get_cache_hash(symbol),
                'data': self.intraday_sigma_cache[symbol],
                'created': datetime.now().isoformat()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached, f)
            print(f"    {symbol}: Saved cache to disk")
        except Exception as e:
            print(f"    {symbol}: Failed to save cache ({e})")

    def _precompute_intraday_sigma_with_cache(self):
        """Pre-compute intraday sigma with disk caching and progress reporting."""
        print("  Pre-computing intraday sigma values...")

        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            # Try to load from disk cache first
            if self._load_sigma_cache(symbol):
                continue  # Cache loaded successfully, skip computation

            print(f"    {symbol}: Computing sigma values (this is a one-time operation)...")
            start_time = time_module.time()

            intraday = self.data[symbol]['intraday']
            all_dates = sorted(intraday['date'].unique())
            total_days = len(all_dates)

            self.intraday_sigma_cache[symbol] = {}

            # Pre-build: for each day, store {minute: price_at_minute} for quick lookup
            print(f"    {symbol}: Building price lookup structure...")
            day_minute_prices = {}

            for trade_date in all_dates:
                day_data = intraday[intraday['date'] == trade_date].sort_values('datetime')
                if day_data.empty:
                    continue

                open_price = day_data.iloc[0]['open']
                minute_prices = {}

                for _, row in day_data.iterrows():
                    mins = row['minutes_from_open']
                    minute_prices[mins] = row['close']

                day_minute_prices[trade_date] = {
                    'open': open_price,
                    'prices': minute_prices
                }

            # Now compute sigma for each (date, minute) using pre-built data
            last_report_time = time_module.time()
            processed_days = 0

            for day_idx, trade_date in enumerate(all_dates):
                # Progress reporting every 2 seconds or every 50 days
                current_time = time_module.time()
                if current_time - last_report_time >= 2.0 or day_idx % 50 == 0:
                    elapsed = current_time - start_time
                    if day_idx > self.lag:
                        days_done = day_idx - self.lag
                        days_total = total_days - self.lag
                        pct = (days_done / days_total) * 100 if days_total > 0 else 0
                        rate = days_done / elapsed if elapsed > 0 else 0
                        eta = (days_total - days_done) / rate if rate > 0 else 0
                        print(f"    {symbol}: {days_done}/{days_total} days ({pct:.1f}%) | "
                              f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | "
                              f"Rate: {rate:.1f} days/sec")
                    last_report_time = current_time

                # Need at least 'lag' days of history
                if day_idx < self.lag:
                    continue

                # Get lookback dates
                lookback_dates = all_dates[day_idx - self.lag:day_idx]

                # Get all unique minutes for this day
                if trade_date not in day_minute_prices:
                    continue

                today_minutes = sorted(day_minute_prices[trade_date]['prices'].keys())
                date_str = str(trade_date)
                self.intraday_sigma_cache[symbol][date_str] = {}

                # For each minute, calculate sigma using lookback
                for minute in today_minutes:
                    moves = []

                    for lookback_date in lookback_dates:
                        if lookback_date not in day_minute_prices:
                            continue

                        day_info = day_minute_prices[lookback_date]
                        open_price = day_info['open']
                        prices = day_info['prices']

                        # Find price at this minute (or closest minute >= this one)
                        price_at_time = None
                        for m in sorted(prices.keys()):
                            if m >= minute:
                                price_at_time = prices[m]
                                break

                        if price_at_time is not None and open_price > 0:
                            move = abs((price_at_time / open_price) - 1)
                            moves.append(move)

                    # Store sigma
                    if moves:
                        self.intraday_sigma_cache[symbol][date_str][minute] = np.mean(moves)
                    else:
                        self.intraday_sigma_cache[symbol][date_str][minute] = 0.02

            # Final stats
            elapsed = time_module.time() - start_time
            print(f"    {symbol}: Complete! {len(self.intraday_sigma_cache[symbol])} days in {elapsed:.1f}s")
            # NEW: Collect daily average sigmas for HAR model
            if self.use_har_forecast:
                if symbol not in self.har_sigma_history:
                    self.har_sigma_history[symbol] = []
                
                # For each day, calculate the average sigma across all minutes
                for trade_date in all_dates:
                    date_str = str(trade_date)
                    if date_str in self.intraday_sigma_cache[symbol]:
                        minute_sigmas = list(self.intraday_sigma_cache[symbol][date_str].values())
                        if minute_sigmas:
                            avg_sigma = np.mean(minute_sigmas)
                            self.har_sigma_history[symbol].append({
                                'date': trade_date,
                                'sigma': avg_sigma
                            })
                
                print(f"    {symbol}: Collected {len(self.har_sigma_history[symbol])} days of sigma history for HAR")
            # Save to disk for future runs
            self._save_sigma_cache(symbol)

    def _is_opex_day(self, trade_date) -> bool:
        """
        Check if a date is monthly OPEX (3rd Friday) or triple witching.
        Triple witching: 3rd Friday of March, June, September, December.
        """
        d = pd.to_datetime(trade_date)

        # Check if it's a Friday
        if d.weekday() != 4:  # 4 = Friday
            return False

        # Check if it's the 3rd Friday (days 15-21)
        if not (15 <= d.day <= 21):
            return False

        # It's the 3rd Friday - this is monthly OPEX
        # Triple witching is 3rd Friday of Mar, Jun, Sep, Dec
        return True
    
    def _get_adaptive_params(self, trade_date):
        """
        Get lookback and VM based on VIX regime.
        
        Returns:
            (lookback, volatility_multiplier)
        """
        if not self.use_vix_adaptive:
            return self.lag, self.volatility_multiplier
        
        # Get VIX for this date
        vix = self.vix_data.get(trade_date, 15.0)
        
        # Determine regime
        regime_map = self.vix_regimes[self.vix_adaptive_mode]
        
        if vix >= regime_map['crisis']['vix']:
            regime = 'crisis'
        elif vix >= regime_map['elevated']['vix']:
            regime = 'elevated'
        elif vix >= regime_map['normal']['vix']:
            regime = 'normal'
        else:
            regime = 'low'
        
        params = regime_map[regime]
        
        return params['lookback'], params['vm']

    def _update_monthly_allocations(self, trade_date):
        """
        Update allocations at the start of each month based on previous month volatility.
        Matches original school project: inverse volatility weighting.
        """
        current_month = pd.to_datetime(trade_date).strftime('%Y-%m')

        if current_month != self.current_month:
            self.current_month = current_month

            # Calculate inverse-vol weighted allocations
            inv_vols = {}
            for symbol in self.symbols:
                prev_vol = self.prev_month_vol.get(symbol, {}).get(current_month, None)
                if prev_vol and prev_vol > 0:
                    inv_vols[symbol] = self.target_vol / prev_vol * (1.0 / len(self.symbols))
                else:
                    inv_vols[symbol] = 1.0 / len(self.symbols)

            # Normalize allocations
            total_inv_vol = sum(inv_vols.values())
            if total_inv_vol > 0:
                for symbol in self.symbols:
                    self.allocations[symbol] = round(inv_vols[symbol] / total_inv_vol, 2)

    # ==========================================================================
    # RANDOM FOREST METHODS (matching ibkr2.py)
    # ==========================================================================

    def extract_rf_features(
        self,
        symbol: str,
        trade_date,
        bar_idx: int,
        day_bars: pd.DataFrame,
        prev_close: float
    ) -> dict:
        """
        Extract features for RF prediction (matching ibkr2.py).
        Returns dict of features or None if extraction fails.
        """
        try:
            bar = day_bars.iloc[bar_idx]
            current_price = bar['close']
            open_price = day_bars.iloc[0]['open']
            current_vwap = bar['vwap']
            minutes_from_open = bar['minutes_from_open']

            # Get datetime for day/hour features
            bar_datetime = pd.to_datetime(bar['datetime'])

            features = {}

            # Price features
            features['price_vs_open'] = (current_price / open_price) - 1
            features['price_vs_vwap'] = (current_price / current_vwap) - 1 if current_vwap > 0 else 0

            # Volatility features
            features['realized_vol'] = self.egarch_vol.get(symbol, {}).get(trade_date, 0.02)

            # Gap size (open vs prev close)
            features['gap_size'] = (open_price / prev_close) - 1 if prev_close > 0 else 0

            # Time features
            features['minutes_from_open'] = minutes_from_open
            features['hour_of_day'] = bar_datetime.hour
            features['day_of_week'] = bar_datetime.dayofweek

            # Volume ratio (today vs average) - simplified
            today_volume = day_bars.iloc[:bar_idx+1]['volume'].sum() if bar_idx > 0 else bar['volume']
            avg_volume = day_bars['volume'].mean() * (bar_idx + 1) if bar_idx > 0 else day_bars['volume'].mean()
            features['volume_ratio'] = today_volume / avg_volume if avg_volume > 0 else 1.0

            # First half-hour return (open to 10:00 AM)
            features['first_half_return'] = self.open_returns.get(symbol, {}).get(trade_date, 0)

            return features

        except Exception as e:
            return None

    def train_rf_model(self, walk_forward_window: int = None):
        """
        Train RF model using walk-forward methodology.

        Walk-forward approach:
        1. Only use data from a rolling window (prevents using future data)
        2. Train on oldest 80% of window, validate on newest 20%
        3. Ensures no look-ahead bias

        Args:
            walk_forward_window: Number of samples to use for training.
                                If None, uses all available data (less rigorous).
        """
        if len(self.rf_training_data) < self.rf_min_samples:
            return

        try:
            # Sort chronologically (critical for walk-forward)
            sorted_data = sorted(self.rf_training_data, key=lambda x: x.get('timestamp', datetime.min))

            # Walk-forward: only use most recent window of data
            if walk_forward_window and len(sorted_data) > walk_forward_window:
                training_data = sorted_data[-walk_forward_window:]
            else:
                training_data = sorted_data

            X = []
            y = []

            for sample in training_data:
                X.append(list(sample['features'].values()))
                y.append(sample['label'])

            X = np.array(X)
            y = np.array(y)

            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                return

            # Handle imbalance
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            class_weight = 'balanced' if imbalance_ratio > 3 else None

            # Walk-forward split: train on first 80%, validate on last 20%
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            if len(X_train) < 20 or len(X_val) < 5:
                # Not enough data for proper split, use all data
                X_train, y_train = X, y
                X_val, y_val = None, None

            # Scale features (fit on training only to prevent leakage)
            self.rf_scaler.fit(X_train)
            X_train_scaled = self.rf_scaler.transform(X_train)

            # Train RF
            self.rf_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                bootstrap=True,
                oob_score=True,
                class_weight=class_weight
            )

            self.rf_model.fit(X_train_scaled, y_train)

            # Walk-forward validation score
            if X_val is not None and len(X_val) > 0:
                X_val_scaled = self.rf_scaler.transform(X_val)
                val_score = self.rf_model.score(X_val_scaled, y_val)
                print(f"    RF Walk-Forward: train={len(X_train)}, val={len(X_val)}, val_acc={val_score:.2f}")

                # Track validation scores for model selection
                if not hasattr(self, 'rf_val_scores'):
                    self.rf_val_scores = []
                self.rf_val_scores.append(val_score)

        except Exception as e:
            print(f"    Error training RF: {e}")

    def get_rf_signal(self, features: dict) -> tuple:
        """
        Get RF prediction and confidence.
        Returns (signal, confidence) where signal is 'BUY' or 'HOLD'.
        """
        if self.rf_model is None or features is None:
            return None, 0.0

        try:
            feature_array = np.array([list(features.values())])
            feature_scaled = self.rf_scaler.transform(feature_array)

            rf_signal = self.rf_model.predict(feature_scaled)[0]
            rf_proba = self.rf_model.predict_proba(feature_scaled)[0]

            # Binary: 1=BUY, 0=HOLD
            signal_str = 'BUY' if rf_signal == 1 else 'HOLD'

            # Confidence
            if len(rf_proba) > 1:
                confidence = rf_proba[1]  # Probability of BUY
            else:
                confidence = rf_proba[0] if rf_signal == 1 else 1 - rf_proba[0]

            return signal_str, confidence

        except Exception:
            return None, 0.0

    def record_trade_outcome(self, features: dict, entry_price: float, exit_price: float,
                             signal: str, timestamp):
        """Record trade outcome for RF training."""
        if features is None:
            return

        # Calculate outcome
        if signal == 'BUY':
            profit_pct = (exit_price / entry_price) - 1
        else:
            profit_pct = (entry_price / exit_price) - 1

        # Binary label: 1=profitable trade, 0=not profitable
        label = 1 if profit_pct > 0.002 else 0

        self.rf_training_data.append({
            'features': features,
            'label': label,
            'profit_pct': profit_pct,
            'timestamp': timestamp
        })

        # Retrain periodically using walk-forward window
        if len(self.rf_training_data) % 30 == 0 and len(self.rf_training_data) >= self.rf_min_samples:
            self.train_rf_model(walk_forward_window=self.rf_walk_forward_window)

    # ==========================================================================
    # SWING MOMENTUM STRATEGY METHODS (for low VIX regimes)
    # ==========================================================================

    def calculate_swing_momentum(self, symbol: str, trade_date) -> float:
        """
        Calculate 12-day momentum for swing strategy.
        Returns momentum as percentage return.
        """
        if symbol not in self.data:
            return None

        daily_df = self.data[symbol].get('daily')
        if daily_df is None or daily_df.empty:
            return None

        # Filter to dates before trade_date (no look-ahead)
        trade_date_ts = pd.Timestamp(trade_date)
        daily_df = daily_df[daily_df['date'] < trade_date_ts].copy()

        if len(daily_df) < self.swing_momentum_lookback + 1:
            return None

        current_price = daily_df.iloc[-1]['close']
        price_n_days_ago = daily_df.iloc[-(self.swing_momentum_lookback + 1)]['close']

        momentum = (current_price / price_n_days_ago) - 1
        return momentum

    def calculate_swing_sma(self, symbol: str, trade_date) -> float:
        """Calculate simple moving average for swing strategy."""
        if symbol not in self.data:
            return None

        daily_df = self.data[symbol].get('daily')
        if daily_df is None or daily_df.empty:
            return None

        # Filter to dates before trade_date
        trade_date_ts = pd.Timestamp(trade_date)
        daily_df = daily_df[daily_df['date'] < trade_date_ts].copy()

        if len(daily_df) < self.swing_sma_period:
            return None

        return daily_df['close'].tail(self.swing_sma_period).mean()

    def calculate_swing_atr(self, symbol: str, trade_date) -> float:
        """Calculate Average True Range for swing strategy stop sizing."""
        if symbol not in self.data:
            return None

        daily_df = self.data[symbol].get('daily')
        if daily_df is None or daily_df.empty:
            return None

        # Filter to dates before trade_date
        trade_date_ts = pd.Timestamp(trade_date)
        daily_df = daily_df[daily_df['date'] < trade_date_ts].copy()

        if len(daily_df) < self.swing_atr_period + 1:
            return None

        df = daily_df.tail(self.swing_atr_period + 1).copy()

        # True Range calculation
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        atr = df['true_range'].tail(self.swing_atr_period).mean()
        return atr

    def get_swing_daily_close(self, symbol: str, trade_date) -> float:
        """Get the closing price for a given date."""
        if symbol not in self.data:
            return None

        daily_df = self.data[symbol].get('daily')
        if daily_df is None or daily_df.empty:
            return None

        # Get price on trade_date (or most recent before it)
        trade_date_ts = pd.Timestamp(trade_date)
        daily_df = daily_df[daily_df['date'] <= trade_date_ts].copy()
        if daily_df.empty:
            return None

        return daily_df.iloc[-1]['close']

    def check_swing_exit(self, symbol: str, trade_date, current_price: float) -> str:
        """
        Check exit conditions for swing position.
        Returns exit reason or None if no exit.
        """
        if symbol not in self.swing_positions:
            return None

        pos = self.swing_positions[symbol]
        if pos.get('quantity', 0) <= 0:
            return None

        entry_date = pos['entry_date']
        days_held = (trade_date - entry_date).days

        # Update trailing stop if new high
        if current_price > pos['highest_price']:
            pos['highest_price'] = current_price
            pos['trailing_stop'] = current_price - (self.swing_atr_multiplier * pos['atr'])

        # Exit 1: Trailing stop hit
        if current_price < pos['trailing_stop']:
            return 'TRAILING_STOP'

        # Exit 2: Momentum reversal
        momentum = self.calculate_swing_momentum(symbol, trade_date)
        if momentum is not None and momentum < -self.swing_momentum_threshold:
            return 'MOMENTUM_REVERSAL'

        # Exit 3: Below SMA (only after min hold)
        if days_held >= self.swing_min_hold_days:
            sma = self.calculate_swing_sma(symbol, trade_date)
            if sma is not None and current_price < sma:
                return 'BELOW_SMA'

        # Exit 4: Max hold period
        if days_held >= self.swing_max_hold_days:
            return 'MAX_HOLD'

        return None

    # =========================================================================
    # HMM REGIME DETECTION
    # =========================================================================
    
    def train_hmm_model(self, returns: np.ndarray, vix_values: np.ndarray = None):
        """
        Train Hidden Markov Model on historical returns and VIX.
        
        The HMM learns latent volatility regimes:
        - State 0: Low volatility (typically calm, trending)
        - State 1: Medium volatility (transition)
        - State 2: High volatility (crisis/opportunity)
        
        Args:
            returns: Array of daily returns
            vix_values: Optional VIX levels (same length as returns)
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            print("Warning: hmmlearn not installed. HMM disabled.")
            return
        
        if len(returns) < 100:
            print("Warning: Insufficient data for HMM training (need 100+ days)")
            return
        
        # Prepare features: returns + VIX (normalized)
        if vix_values is not None and len(vix_values) == len(returns):
            features = np.column_stack([
                returns.reshape(-1, 1),
                (vix_values / 100).reshape(-1, 1)  # Normalize VIX
            ])
        else:
            features = returns.reshape(-1, 1)
        
        # Train Gaussian HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.hmm_n_states,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        try:
            self.hmm_model.fit(features)
            self.hmm_trained = True
            
            # Analyze learned states (sort by volatility)
            state_vols = []
            for i in range(self.hmm_n_states):
                vol = np.sqrt(self.hmm_model.covars_[i][0, 0]) * np.sqrt(252) * 100
                state_vols.append((i, vol))
            
            # Sort states by volatility (low to high)
            state_vols.sort(key=lambda x: x[1])
            self.hmm_state_order = {state_vols[i][0]: i for i in range(len(state_vols))}
            
            # Map: 0=LOW_VOL, 1=MED_VOL, 2=HIGH_VOL
            self.hmm_regime_names = {0: 'LOW_VOL', 1: 'MED_VOL', 2: 'HIGH_VOL'}
            
        except Exception as e:
            print(f"Warning: HMM training failed: {e}")
            self.hmm_trained = False
    
    def predict_hmm_regime(self, returns: np.ndarray, vix_values: np.ndarray = None) -> tuple:
        """
        Predict current regime using trained HMM.
        
        Args:
            returns: Recent returns (at least 20 days)
            vix_values: Optional VIX levels
            
        Returns:
            (regime_name, regime_probabilities) or (None, None) if not trained
        """
        if not self.hmm_trained or self.hmm_model is None:
            return None, None
        
        if len(returns) < 20:
            return None, None
        
        # Prepare features
        if vix_values is not None and len(vix_values) == len(returns):
            features = np.column_stack([
                returns.reshape(-1, 1),
                (vix_values / 100).reshape(-1, 1)
            ])
        else:
            features = returns.reshape(-1, 1)
        
        try:
            # Predict
            regime_sequence = self.hmm_model.predict(features)
            regime_probs = self.hmm_model.predict_proba(features)
            
            # Get current regime (last observation)
            raw_regime = regime_sequence[-1]
            current_probs = regime_probs[-1]
            
            # Map to ordered regime (0=LOW, 1=MED, 2=HIGH)
            ordered_regime = self.hmm_state_order.get(raw_regime, raw_regime)
            regime_name = self.hmm_regime_names.get(ordered_regime, f'STATE_{ordered_regime}')
            
            return regime_name, current_probs
            
        except Exception as e:
            return None, None
    
    def get_hmm_trading_action(self, regime_name: str, vix: float) -> str:
        """
        Determine trading action based on HMM regime.
        
        Args:
            regime_name: Current HMM regime ('LOW_VOL', 'MED_VOL', 'HIGH_VOL')
            vix: Current VIX level
            
        Returns:
            'INTRADAY', 'SWING', or 'CASH'
        """
        if regime_name == 'HIGH_VOL':
            # High volatility regime - intraday momentum works best
            return 'INTRADAY'
        elif regime_name == 'LOW_VOL':
            # Low volatility - only swing if VIX very low and trending
            if vix < self.swing_max_vix:
                return 'SWING'
            else:
                return 'CASH'
        else:  # MED_VOL
            # Transition regime - be cautious
            if vix >= 17:
                return 'INTRADAY'
            else:
                return 'CASH'
    
    # =========================================================================
    # AUTOCORRELATION FOR TREND DETECTION
    # =========================================================================
    
    def calculate_autocorrelation(self, symbol: str, trade_date, lookback: int = 20) -> float:
        """
        Calculate return autocorrelation (lag-1) for trend detection.
        
        Positive autocorr = trending (momentum)
        Negative autocorr = mean-reverting (avoid swing)
        
        Args:
            symbol: Stock symbol
            trade_date: Current date
            lookback: Number of days for calculation
            
        Returns:
            Autocorrelation coefficient or None
        """
        if symbol not in self.data:
            return None
        
        daily_df = self.data[symbol].get('daily')
        if daily_df is None or daily_df.empty:
            return None
        
        # Filter to dates before trade_date
        trade_date_ts = pd.Timestamp(trade_date)
        daily_df = daily_df[daily_df['date'] < trade_date_ts].copy()
        
        if len(daily_df) < lookback + 1:
            return None
        
        # Calculate returns
        returns = daily_df['close'].pct_change().tail(lookback + 1).dropna()
        
        if len(returns) < lookback:
            return None
        
        # Lag-1 autocorrelation
        returns_array = returns.values
        autocorr = np.corrcoef(returns_array[:-1], returns_array[1:])[0, 1]
        
        return autocorr if not np.isnan(autocorr) else None

    def calculate_market_autocorrelation(self, trade_date) -> float:
        """
        Calculate market-wide autocorrelation for regime detection.
        
        Uses the primary symbol (first in list, typically SPY) as market proxy.
        
        Positive autocorr (> 0.05) = Trending market
          → Low VIX days may have real momentum, OK to trade
        Negative autocorr (< 0) = Mean-reverting / choppy market
          → Low VIX days are traps, apply VIX filter
          
        This allows:
        - 2023-2024: Trending bull, autocorr > 0 → trade low VIX days
        - 2025: Choppy (tariffs), autocorr < 0 → filter low VIX days
        
        Args:
            trade_date: Current date
            
        Returns:
            Autocorrelation coefficient or None
        """
        # Use first symbol as market proxy (usually SPY)
        market_symbol = self.symbols[0] if self.symbols else None
        if market_symbol is None:
            return None
        
        autocorr = self.calculate_autocorrelation(
            market_symbol, 
            trade_date, 
            lookback=self.autocorr_lookback
        )
        
        self.current_market_autocorr = autocorr
        return autocorr

    # =========================================================================
    # STOCK-SPECIFIC STRATEGIES (for individual stocks, not ETFs)
    # =========================================================================

    def is_etf(self, symbol: str) -> bool:
        """
        Check if symbol is an ETF (SPY, QQQ) vs individual stock.

        Args:
            symbol: Stock symbol

        Returns:
            True if ETF (uses momentum strategy), False if stock (uses mean reversion/gap)
        """
        return symbol.upper() in ['SPY', 'QQQ']

    def calculate_stock_vol_regime(self, symbol: str, trade_date, lookback: int = None) -> tuple:
        """
        Calculate stock-specific volatility regime using realized volatility.

        Instead of using VIX, this uses the stock's own historical volatility
        to determine if it's in a high or low vol regime relative to itself.

        Args:
            symbol: Stock symbol
            trade_date: Current date
            lookback: Lookback period for vol calculation (default: self.vol_lookback)

        Returns:
            (current_vol, vol_rank) tuple:
                - current_vol: Current realized volatility (annualized)
                - vol_rank: Percentile rank of current vol vs historical (0-1)
        """
        if lookback is None:
            lookback = self.vol_lookback

        if symbol not in self.data:
            return None, None

        daily_df = self.data[symbol].get('daily')
        if daily_df is None or daily_df.empty:
            return None, None

        # Filter to dates before trade_date
        trade_date_ts = pd.Timestamp(trade_date)
        daily_df = daily_df[daily_df['date'] < trade_date_ts].copy()

        if len(daily_df) < lookback * 3:  # Need enough history for vol rank
            return None, None

        # Calculate returns
        daily_df['returns'] = daily_df['close'].pct_change()

        # Current volatility (most recent lookback period)
        current_returns = daily_df['returns'].tail(lookback).dropna()
        if len(current_returns) < lookback:
            return None, None

        current_vol = current_returns.std() * np.sqrt(252)  # Annualized

        # Calculate rolling volatility for vol rank
        rolling_vols = []
        for i in range(lookback, len(daily_df)):
            window_returns = daily_df['returns'].iloc[i-lookback:i]
            if len(window_returns) == lookback:
                vol = window_returns.std() * np.sqrt(252)
                rolling_vols.append(vol)

        if len(rolling_vols) < 20:  # Need at least 20 periods for percentile
            return current_vol, 0.5  # Default to median

        # Calculate vol rank (percentile of current vol)
        vol_rank = sum(1 for v in rolling_vols if v < current_vol) / len(rolling_vols)

        # Cache the result
        if symbol not in self.stock_vol_regimes:
            self.stock_vol_regimes[symbol] = {}
        self.stock_vol_regimes[symbol][trade_date] = (current_vol, vol_rank)

        return current_vol, vol_rank

    def generate_mean_reversion_signal(
        self,
        symbol: str,
        current_price: float,
        open_price: float,
        prev_close: float,
        vwap: float,
        sigma: float,
        vol_rank: float = None
    ) -> str:
        """
        Generate mean reversion signal (inverse of momentum).

        Strategy: Fade breakouts instead of following them.
        - If price breaks UP too far from VWAP → SELL (expect reversion down)
        - If price breaks DOWN too far from VWAP → BUY (expect reversion up)

        This is the opposite of the momentum strategy used for ETFs.

        Args:
            symbol: Stock symbol
            current_price: Current price
            open_price: Opening price
            prev_close: Previous day's close
            vwap: Volume-weighted average price
            sigma: Noise boundary (volatility)
            vol_rank: Optional vol rank (0-1) for regime detection

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if not self.use_stock_strategies:
            return 'HOLD'

        # Check if we're in vol-adaptive mode
        if self.stock_strategy_mode == 'vol_adaptive' and vol_rank is not None:
            # High vol stocks → use mean reversion
            # Low vol stocks → use momentum (like ETFs)
            if vol_rank < 0.3:  # Low vol regime (bottom 30%)
                # Use momentum instead of mean reversion
                if current_price > vwap * (1 + sigma):
                    return 'BUY'  # Momentum up
                elif current_price < vwap * (1 - sigma):
                    return 'SELL'  # Momentum down
                return 'HOLD'

        # Mean reversion logic (fade breakouts)
        # Price too high → expect reversion down → SELL
        if current_price > vwap * (1 + sigma):
            return 'SELL'

        # Price too low → expect reversion up → BUY
        elif current_price < vwap * (1 - sigma):
            return 'BUY'

        return 'HOLD'

    def generate_gap_signal(
        self,
        symbol: str,
        trade_date,
        current_time,
        open_price: float,
        prev_close: float,
        current_price: float
    ) -> str:
        """
        Generate gap capture signal (trade overnight gaps).

        Strategy: Trade in the direction of the gap at market open (9:31 AM),
        expecting the gap to continue in the same direction intraday.
        Exit by gap_exit_minutes (e.g., 60 minutes = 10:30 AM).

        Args:
            symbol: Stock symbol
            trade_date: Current date
            current_time: Current time
            open_price: Opening price
            prev_close: Previous day's close
            current_price: Current price

        Returns:
            'BUY', 'SELL', 'EXIT', or 'HOLD'
        """
        if not self.use_stock_strategies:
            return 'HOLD'

        # Calculate gap percentage
        gap_pct = (open_price - prev_close) / prev_close

        # Check if we have an existing gap position
        if symbol in self.gap_positions:
            gap_pos = self.gap_positions[symbol]
            entry_time = gap_pos['entry_time']

            # Calculate minutes since entry
            time_delta = (current_time - entry_time).total_seconds() / 60

            # Exit if time threshold reached
            if time_delta >= self.gap_exit_minutes:
                return 'EXIT'

            # Otherwise hold the position
            return 'HOLD'

        # Only enter new positions at market open (9:31-9:35 AM)
        if current_time.hour == 9 and 31 <= current_time.minute <= 35:
            # Check if gap is significant enough
            if abs(gap_pct) >= self.gap_threshold:
                # Gap up → BUY (expect continuation)
                if gap_pct > 0:
                    return 'BUY'
                # Gap down → SELL (expect continuation down)
                else:
                    return 'SELL'

        return 'HOLD'

    def check_swing_entry(self, symbol: str, trade_date, current_price: float, vix: float = None) -> bool:
        """
        Check entry conditions for swing position.
        
        Standard mode: momentum > 1% + above SMA
        Strict mode (Option B): VIX < 15 + momentum > 3% + positive autocorr
        
        Returns True if should enter.
        """
        # Already have position?
        if symbol in self.swing_positions and self.swing_positions[symbol].get('quantity', 0) > 0:
            return False
        
        # =====================================================================
        # STRICT MODE: Much tighter entry conditions
        # Only enter swing in VERY calm, STRONGLY trending markets
        # =====================================================================
        if self.swing_strict_mode:
            # Condition 1: VIX must be VERY low (< 15)
            if vix is not None and vix >= self.swing_max_vix:
                return False
            
            # Condition 2: Strong momentum (> 3%, not just 1%)
            momentum = self.calculate_swing_momentum(symbol, trade_date)
            if momentum is None or momentum <= self.swing_min_momentum:
                return False
            
            # Condition 3: Price above SMA (trend confirmation)
            sma = self.calculate_swing_sma(symbol, trade_date)
            if sma is None or current_price <= sma:
                return False
            
            # Condition 4: Positive autocorrelation (truly trending, not choppy)
            autocorr = self.calculate_autocorrelation(symbol, trade_date)
            if autocorr is None or autocorr < self.swing_min_autocorr:
                return False
            
            # All strict conditions met
            return True
        
        # =====================================================================
        # STANDARD MODE: Original entry conditions
        # =====================================================================
        
        # Entry condition 1: Positive momentum above threshold
        momentum = self.calculate_swing_momentum(symbol, trade_date)
        if momentum is None or momentum <= self.swing_momentum_threshold:
            return False

        # Entry condition 2: Price above SMA (trend confirmation)
        sma = self.calculate_swing_sma(symbol, trade_date)
        if sma is None or current_price <= sma:
            return False

        return True

    def calculate_swing_position_size(self, symbol: str, current_price: float, capital: float) -> int:
        """Calculate position size for swing trade using volatility targeting."""
        # Get realized volatility (use daily returns std)
        if symbol not in self.data:
            return 0

        daily_df = self.data[symbol].get('daily')
        if daily_df is None or len(daily_df) < 21:
            return 0

        returns = daily_df['close'].pct_change().tail(20)
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        if annualized_vol <= 0:
            annualized_vol = 0.20  # Default 20%

        # Volatility scalar (target 15% annualized, capped at max_leverage)
        target_vol = 0.15
        vol_scalar = min(target_vol / annualized_vol, self.max_leverage)

        # Base allocation 40%, max 60%
        base_allocation = 0.40
        max_allocation = 0.60

        position_value = capital * base_allocation * vol_scalar
        max_value = capital * max_allocation
        position_value = min(position_value, max_value)

        shares = int(position_value / current_price)
        return shares

    def fetch_data(self):
        """
        Requires IBKR CSV data. Run extract_ibkr_data() first to create CSV files.
        """
        raise RuntimeError(
            "No data source available. Please run extract_ibkr_data() first:\n"
            "  from ibkr_backtest import extract_ibkr_data\n"
            "  extract_ibkr_data(['NVDA', 'SPY', ...], duration='1 Y')\n"
            "This requires TWS/IBKR Gateway to be running."
        )

    def calculate_egarch_vol(self, returns: np.ndarray) -> float:
        """Calculate EGARCH volatility forecast (fallback only)."""
        try:
            if len(returns) < self.lag:
                return returns.std()

            recent = returns[-self.lag:] * 100  # Scale for arch library

            model = arch_model(
                recent,
                vol='GARCH',
                p=self.p_terms,
                o=self.o_terms,
                q=self.q_terms
            )

            res = model.fit(disp='off', show_warning=False, options={'maxiter': 100})
            forecast = res.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.values[-1, 0]) / 100

            return vol

        except Exception:
            return returns.std() if len(returns) > 0 else 0.02

    def calculate_intraday_sigma(
        self,
        symbol: str,
        trade_date,
        minutes_from_open: int
    ) -> float:
        """
        Get pre-computed intraday sigma from cache.
        Falls back to default if not found.

        Sigma = mean of |move from open by this time| over past 14 days.
        """
        date_str = str(trade_date)

        # Fast O(1) lookup from pre-computed cache
        if symbol in self.intraday_sigma_cache:
            if date_str in self.intraday_sigma_cache[symbol]:
                minute_cache = self.intraday_sigma_cache[symbol][date_str]

                # Exact match
                if minutes_from_open in minute_cache:
                    return minute_cache[minutes_from_open]

                # Find closest minute >= requested (in case of gaps)
                for m in sorted(minute_cache.keys()):
                    if m >= minutes_from_open:
                        return minute_cache[m]

                # If no future minute found, use last available
                if minute_cache:
                    return minute_cache[max(minute_cache.keys())]

        # Fallback default
        return 0.02

    def calculate_regression_params(
        self,
        symbol: str,
        trade_date,
        lookback_days: int = 20
    ) -> tuple:
        """
        Calculate regression parameters (intercept, coef, R²) for first-half vs last-half hour returns.
        Uses walk-forward approach: only uses data before trade_date.

        Returns:
            (intercept, coef, r_squared)
        """
        try:
            intraday_df = self.data[symbol]['intraday']

            # Get unique trading days before current date
            all_dates = sorted(intraday_df['date'].unique())
            past_dates = [d for d in all_dates if d < trade_date]

            if len(past_dates) < lookback_days:
                return 0.0, 0.5, 0.0  # Default: no predictive power

            lookback_dates = past_dates[-lookback_days:]

            first_half_returns = []
            last_half_returns = []

            for day in lookback_dates:
                day_data = intraday_df[intraday_df['date'] == day].sort_values('datetime')

                if len(day_data) < 20:  # Need enough bars
                    continue

                try:
                    open_price = day_data.iloc[0]['open']

                    # Price at 10 AM (30 minutes from open)
                    at_10am = day_data[day_data['minutes_from_open'] >= 30]
                    if at_10am.empty:
                        continue
                    price_10am = at_10am.iloc[0]['close']

                    # Price at 3:30 PM (360 minutes from open)
                    at_330pm = day_data[day_data['minutes_from_open'] >= 360]
                    if at_330pm.empty:
                        continue
                    price_330pm = at_330pm.iloc[0]['close']

                    # Closing price
                    close_price = day_data.iloc[-1]['close']

                    # Calculate returns
                    first_half_return = (price_10am / open_price) - 1
                    last_half_return = (close_price / price_330pm) - 1

                    # Filter outliers
                    if abs(first_half_return) < 0.1 and abs(last_half_return) < 0.1:
                        first_half_returns.append(first_half_return)
                        last_half_returns.append(last_half_return)

                except Exception:
                    continue

            if len(first_half_returns) < 5:
                return 0.0, 0.5, 0.0

            # Fit linear regression
            X = np.array(first_half_returns).reshape(-1, 1)
            y = np.array(last_half_returns)

            # Simple linear regression
            X_mean = X.mean()
            y_mean = y.mean()
            coef = np.sum((X.flatten() - X_mean) * (y - y_mean)) / np.sum((X.flatten() - X_mean) ** 2)
            intercept = y_mean - coef * X_mean

            # Calculate R²
            y_pred = intercept + coef * X.flatten()
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return intercept, coef, max(0, r_squared)

        except Exception:
            return 0.0, 0.5, 0.0

    def generate_signal(
        self,
        current_price: float,
        open_price: float,
        prev_close: float,
        vwap: float,
        sigma: float
    ) -> str:
        """Generate trading signal based on noise boundaries and VWAP."""

        # Apply volatility multiplier (paper finding: optimal = 1.5)
        adjusted_sigma = sigma * self.volatility_multiplier
        upper_bound = max(open_price, prev_close) * (1 + adjusted_sigma)
        lower_bound = min(open_price, prev_close) * (1 - adjusted_sigma)

        # Anti-pump/dump filter (2x sigma)
        extreme_upper = max(open_price, prev_close) * (1 + sigma * 2)
        extreme_lower = min(open_price, prev_close) * (1 - sigma * 2)

        is_too_extended = current_price > extreme_upper
        is_too_dumped = current_price < extreme_lower

        # BUY signal
        if (current_price > upper_bound and
            current_price > vwap and
            not is_too_extended):
            return 'BUY'

        # SELL signal (for shorting - can disable if long-only)
        if (current_price < lower_bound and
            current_price < vwap and
            not is_too_dumped):
            return 'SELL'

        return 'HOLD'

    def apply_transaction_costs(self, price: float, shares: int) -> float:
        """Calculate total transaction cost."""
        commission = self.commission_per_share * abs(shares)
        slippage = 0.001 * abs(shares)
        return commission + slippage

    def run_backtest(self, long_only: bool = True) -> pd.DataFrame:
        """
        Run the backtest simulation - PURE INTRADAY (all positions close at EOD).

        Args:
            long_only: If True, ignore SELL signals (no shorting). Default False.
        """
        if not self.data:
            self.fetch_data()

        print(f"\nRunning backtest from {self.start_date} to {self.end_date}...")
        print(f"  Long only: {long_only}")
        print(f"  Initial capital: ${self.initial_capital:,.0f}")
        print(f"  Trade frequency: Every {self.trade_freq} minutes after 10:00 AM")
        print(f"  Target volatility: {self.target_vol:.0%}")
        print(f"  Max leverage: {self.max_leverage}x")
        print(f"  RF signals: {'Enabled' if self.use_rf else 'Disabled'}")
        vix_filter_str = 'Disabled'
        if self.vix_min is not None and self.vix_max is not None:
            vix_filter_str = f'VIX {self.vix_min} to {self.vix_max}'
        elif self.vix_min is not None:
            vix_filter_str = f'VIX >= {self.vix_min}'
        print(f"  VIX filter (ETFs only): {vix_filter_str}")
        if self.use_adaptive_threshold:
            print(f"  Adaptive VIX: Enabled (lookback={self.adaptive_lookback} days)")
            print(f"    Thresholds: {self.adaptive_thresholds}")
        if self.use_autocorr_filter:
            print(f"  Autocorr Filter: Enabled (lookback={self.autocorr_lookback}, threshold={self.autocorr_threshold})")
            print(f"    → Override VIX filter if market is trending (autocorr > {self.autocorr_threshold})")
        if self.skip_danger_zone:
            print(f"  Danger zone: Skip VIX {self.danger_zone_min}-{self.danger_zone_max} (cash only)")
        swing_desc = 'Disabled'
        if self.use_swing and self.vix_min:
            if self.swing_strict_mode:
                swing_desc = f"STRICT MODE (VIX < {self.swing_max_vix}, momentum > {self.swing_min_momentum:.0%}, autocorr > {self.swing_min_autocorr})"
            else:
                swing_desc = f"ETFs: VIX < {self.vix_min} only | Stocks: no VIX filter (exit ETFs when VIX >= {self.vix_min})"
        print(f"  Swing strategy: {swing_desc}")
        if self.use_hmm_regime:
            print(f"  HMM Regime: Enabled ({self.hmm_n_states} states, retrain every {self.hmm_retrain_frequency} days)")
        print(f"  Early stop: {'Exit if loss > ' + f'{self.early_stop_pct:.2%}' + ' in first 30 min' if self.early_stop_pct > 0 else 'Disabled'}")
        print(f"  Last entry time: {self.last_entry_time.strftime('%H:%M')}")
        print(f"  Skip OPEX: {'Yes' if self.skip_opex else 'No'}")
        print(f"  Exit: max(upper_bound, VWAP) trailing stop (paper spec)")

        # Reset swing positions for fresh backtest
        self.swing_positions = {}
        self.swing_trades = []
        
        # Reset HMM state
        self.hmm_model = None
        self.hmm_trained = False
        self.hmm_last_train_idx = 0
        self.current_hmm_regime = None
        
        # Reset rolling returns for adaptive threshold
        self.rolling_returns = []
        self.current_adaptive_threshold = self.vix_min  # Start with default

        # Initialize tracking - ONLY cash, no overnight positions
        capital = self.initial_capital

        trades = []
        equity_curve = []
        daily_pnl_list = []

        # Get all unique trading dates
        first_symbol = self.symbols[0]
        if first_symbol not in self.data:
            print("No data available")
            return pd.DataFrame()

        intraday_df = self.data[first_symbol]['intraday']
        all_dates = sorted(intraday_df['date'].unique())

        # Filter to start_date/end_date range
        start_dt = pd.to_datetime(self.start_date).date() if self.start_date else None
        end_dt = pd.to_datetime(self.end_date).date() if self.end_date else None
        trading_dates = [d for d in all_dates
                        if (start_dt is None or d >= start_dt)
                        and (end_dt is None or d <= end_dt)]
        total_days = len(trading_dates)

        for day_idx, trade_date in enumerate(trading_dates):
            daily_pnl = 0.0
            # Progress reporting
            if day_idx % 100 == 0:
                pct = (day_idx / total_days) * 100
                print(f"  Progress: {day_idx}/{total_days} days ({pct:.1f}%) - {trade_date}")

            # VIX-based regime routing
            vix_value = self.vix_data.get(trade_date, None)
            # VIX filter only applies to ETFs, not stocks
            # Stocks trade regardless of VIX level (mean reversion works in all regimes)
            etf_vix_ok = True  # Default: ETFs can trade

            # Determine effective VIX threshold (adaptive or static)
            if self.use_adaptive_threshold and len(self.rolling_returns) >= self.adaptive_lookback:
                rolling_sharpe = self.calculate_rolling_sharpe(self.rolling_returns)
                effective_vix_min = self.get_adaptive_vix_threshold(rolling_sharpe)
                self.current_adaptive_threshold = effective_vix_min

                # Log threshold changes (first occurrence of each day's threshold)
                if day_idx % 100 == 0:
                    print(f"    [ADAPTIVE] Rolling Sharpe: {rolling_sharpe:.2f} -> VIX threshold: {effective_vix_min}")
            else:
                effective_vix_min = self.vix_min

            if effective_vix_min is not None:
                if vix_value is None:
                    # No VIX data - block ETF trading but stocks can still trade
                    etf_vix_ok = False
                elif vix_value < effective_vix_min:
                    # Low VIX regime - check if we should override with autocorr filter
                    etf_vix_ok = False

                    # AUTOCORRELATION-AWARE FILTER:
                    # If market is trending (positive autocorr), low VIX days may have
                    # real momentum - override VIX filter and trade intraday
                    if self.use_autocorr_filter:
                        market_autocorr = self.calculate_market_autocorrelation(trade_date)
                        if market_autocorr is not None:
                            # Negative threshold: block if BELOW (mean-reverting)
                            # Positive threshold was wrong - 2023-24 had autocorr ~0, not > 0.05
                            if market_autocorr >= -self.autocorr_threshold:
                                # NOT mean-reverting - safe to trade low VIX days
                                etf_vix_ok = True
                                if day_idx % 50 == 0:
                                    print(f"    [AUTOCORR] Override: autocorr={market_autocorr:.3f} >= {-self.autocorr_threshold} (not mean-reverting)")
                            else:
                                # Mean-reverting - keep VIX filter active for ETFs
                                if day_idx % 50 == 0:
                                    print(f"    [AUTOCORR] Block ETFs: autocorr={market_autocorr:.3f} < {-self.autocorr_threshold} (mean-reverting)")
                        # else: Market is choppy - keep VIX filter active for ETFs

            # Also check vix_max (for bucket analysis)
            if self.vix_max is not None and vix_value is not None:
                if vix_value >= self.vix_max:
                    # VIX above max - block ETF trading
                    etf_vix_ok = False
            
            # ============================================================
            # HMM REGIME DETECTION (optional)
            # ============================================================
            hmm_action = None
            if self.use_hmm_regime and day_idx >= self.hmm_lookback:
                # Get historical returns and VIX for HMM
                hist_dates = trading_dates[max(0, day_idx - self.hmm_lookback):day_idx]
                
                # Get returns from daily data
                daily_df = self.data[first_symbol].get('daily')
                if daily_df is not None and len(daily_df) > 0:
                    hist_returns = []
                    hist_vix = []
                    for hd in hist_dates:
                        hd_ts = pd.Timestamp(hd)
                        day_data = daily_df[daily_df['date'].dt.date == hd] if hasattr(daily_df['date'].iloc[0], 'date') else daily_df[daily_df['date'] == hd]
                        if len(day_data) > 0:
                            # Calculate return from close prices
                            pass
                        if hd in self.vix_data:
                            hist_vix.append(self.vix_data[hd])
                    
                    # Simpler: use daily returns from price series
                    mask = daily_df['date'].apply(lambda x: x.date() if hasattr(x, 'date') else x).isin(hist_dates)
                    hist_prices = daily_df.loc[mask, 'close'].values
                    if len(hist_prices) > 20:
                        hist_returns = np.diff(hist_prices) / hist_prices[:-1]
                        hist_vix = [self.vix_data.get(d, 17) for d in hist_dates[1:]]
                        
                        # Train or retrain HMM
                        should_train = (
                            not self.hmm_trained or 
                            (day_idx - self.hmm_last_train_idx) >= self.hmm_retrain_frequency
                        )
                        
                        if should_train and len(hist_returns) >= 100:
                            self.train_hmm_model(np.array(hist_returns), np.array(hist_vix))
                            self.hmm_last_train_idx = day_idx
                            if day_idx % 100 == 0 and self.hmm_trained:
                                print(f"    [HMM] Retrained at day {day_idx}")
                        
                        # Predict current regime
                        if self.hmm_trained:
                            regime_name, regime_probs = self.predict_hmm_regime(
                                np.array(hist_returns[-50:]), 
                                np.array(hist_vix[-50:]) if len(hist_vix) >= 50 else None
                            )
                            self.current_hmm_regime = regime_name
                            
                            if regime_name is not None:
                                hmm_action = self.get_hmm_trading_action(regime_name, vix_value or 17)
                                
                                # HMM can override VIX-based decisions for ETFs
                                if hmm_action == 'INTRADAY' and not etf_vix_ok:
                                    # HMM says trade even though VIX is low
                                    # Only override if we're confident (high vol regime)
                                    if regime_name == 'HIGH_VOL':
                                        etf_vix_ok = True
                                        if day_idx % 100 == 0:
                                            print(f"    [HMM] Override: {regime_name} -> INTRADAY despite VIX={vix_value:.1f}")
                                elif hmm_action == 'CASH' and etf_vix_ok:
                                    # HMM says avoid, but VIX says trade
                                    # Be conservative - trust HMM's caution
                                    pass  # Don't override, VIX signal is strong
            
            # Check for danger zone (VIX 15-17 is historically toxic)
            # Danger zone blocks ETF trading; stocks continue (no VIX filter for stocks)
            in_danger_zone = False
            if self.skip_danger_zone and vix_value is not None:
                if self.danger_zone_min <= vix_value < self.danger_zone_max:
                    in_danger_zone = True
                    etf_vix_ok = False  # No ETF intraday in danger zone
                    # Will also skip swing entries below

            # ============================================================
            # SWING STRATEGY: Check exits and entries for low VIX regime
            # ============================================================
            # Always check swing exits (even in high VIX - trailing stop still applies)
            for symbol in self.symbols:
                if symbol not in self.swing_positions:
                    continue
                pos = self.swing_positions[symbol]
                if pos.get('quantity', 0) <= 0:
                    continue

                # Get current price (close of day)
                current_price = self.get_swing_daily_close(symbol, trade_date)
                if current_price is None:
                    continue

                # Check exit conditions
                # VIX REGIME EXIT: ETFs exit swing when VIX crosses above threshold
                is_etf = self.is_etf(symbol)
                if is_etf and vix_value is not None and vix_value >= self.vix_min:
                    exit_reason = 'VIX_REGIME_CHANGE'
                else:
                    exit_reason = self.check_swing_exit(symbol, trade_date, current_price)

                if exit_reason:
                    entry_price = pos['entry_price']
                    shares = pos['quantity']
                    exit_cost = self.apply_transaction_costs(current_price, shares)
                    proceeds = shares * current_price - exit_cost
                    pnl = (current_price - entry_price) * shares - exit_cost

                    capital += proceeds  # Add full sale proceeds
                    daily_pnl += pnl  # PnL for reporting

                    self.swing_trades.append({
                        'type': 'SWING',
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': trade_date,
                        'exit_price': current_price,
                        'exit_reason': exit_reason,
                        'shares': shares,
                        'pnl': pnl,
                        'return': (current_price / entry_price) - 1
                    })

                    print(f"    [SWING EXIT] {symbol}: {exit_reason}, PnL: ${pnl:,.2f}")

                    # Clear position
                    self.swing_positions[symbol] = {'quantity': 0}

            # Swing entries (NOT in danger zone)
            # ETFs: only swing when VIX < 17 (when intraday is blocked)
            # Stocks: no VIX filter - always eligible for swing
            if self.use_swing and vix_value is not None and not in_danger_zone:
                for symbol in self.symbols:
                    is_etf = self.is_etf(symbol)

                    # ETFs: only swing when VIX is LOW (intraday is blocked)
                    if is_etf and etf_vix_ok:
                        continue  # VIX >= 17, use intraday strategy for ETFs

                    # Stocks: no VIX filter - always eligible for swing

                    current_price = self.get_swing_daily_close(symbol, trade_date)
                    if current_price is None:
                        continue

                    if self.check_swing_entry(symbol, trade_date, current_price, vix=vix_value):
                        # Calculate position size
                        shares = self.calculate_swing_position_size(symbol, current_price, capital)
                        if shares <= 0:
                            continue

                        # Entry cost
                        entry_cost = self.apply_transaction_costs(current_price, shares)
                        atr = self.calculate_swing_atr(symbol, trade_date)
                        
                        # Use wider stops in strict mode or if explicitly enabled
                        atr_mult = self.swing_atr_multiplier
                        if self.swing_wider_stops or self.swing_strict_mode:
                            atr_mult = 3.0  # 3x ATR instead of 2x

                        self.swing_positions[symbol] = {
                            'quantity': shares,
                            'entry_price': current_price,
                            'entry_date': trade_date,
                            'highest_price': current_price,
                            'trailing_stop': current_price - (atr_mult * atr) if atr else current_price * 0.93,
                            'atr': atr if atr else current_price * 0.02
                        }

                        # Deduct full position cost from capital (shares + transaction costs)
                        position_cost = shares * current_price + entry_cost
                        capital -= position_cost
                        daily_pnl -= entry_cost  # Only transaction costs affect daily PnL

                        momentum = self.calculate_swing_momentum(symbol, trade_date)
                        autocorr = self.calculate_autocorrelation(symbol, trade_date) if self.swing_strict_mode else None
                        autocorr_str = f", autocorr: {autocorr:.2f}" if autocorr is not None else ""
                        print(f"    [SWING ENTRY] {symbol}: {shares} shares @ ${current_price:.2f}, "
                              f"momentum: {momentum:.2%}, VIX: {vix_value:.1f}{autocorr_str}")

            # NOTE: No early exit here - stocks trade regardless of VIX
            # ETFs are filtered per-symbol below based on etf_vix_ok

            # OPEX filter: skip monthly options expiration (3rd Friday)
            if self.skip_opex and self._is_opex_day(trade_date):
                continue

            # Update monthly allocations
            self._update_monthly_allocations(trade_date)

            # Track open prices and previous closes
            open_prices = {}
            prev_closes = {}
            trade_date_str = str(trade_date)

            for symbol in self.symbols:
                if symbol not in self.data:
                    continue

                intraday = self.data[symbol]['intraday']
                intraday['date_str'] = intraday['date'].apply(lambda x: str(x))
                prev_day_bars = intraday[intraday['date_str'] < trade_date_str]

                if len(prev_day_bars) > 0:
                    prev_closes[symbol] = prev_day_bars.iloc[-1]['close']
                else:
                    prev_closes[symbol] = None

            # Daily P&L tracker
            daily_pnl = 0.0

            # Intraday position tracking (reset each day - all intraday positions close at EOD)
            # Tracks: {'symbol': {'direction': 1 or -1, 'entry_price': float, 'entry_vwap': float}}
            self.intraday_positions = {}  # Clear at start of each day

            # Process each symbol for this day
            for symbol in self.symbols:
                if symbol not in self.data:
                    continue

                intraday = self.data[symbol]['intraday']
                intraday['date_str'] = intraday['date'].apply(lambda x: str(x))
                day_bars = intraday[intraday['date_str'] == trade_date_str].copy()

                if day_bars.empty:
                    continue

                open_prices[symbol] = day_bars.iloc[0]['open']
                if prev_closes[symbol] is None:
                    prev_closes[symbol] = open_prices[symbol]

                # Get dvol for position sizing (14-day std of daily returns)
                dvol = self.dvol.get(symbol, {}).get(trade_date, 0.02)

                # Paper formula: anchors based on max/min of open & prev_close (adjusted for dividends)
                # TODO: Add dividend data loading if available
                # For now, dividend_amount = 0 (most days have no dividend)
                dividend_amount = 0.0
                prev_close_adjusted = prev_closes[symbol] - dividend_amount
                anchor_high = max(open_prices[symbol], prev_close_adjusted)
                anchor_low = min(open_prices[symbol], prev_close_adjusted)

                # Build exposure array for the day with TIME-VARYING SIGMA
                # Paper: σ at time T = avg of |move from open by time T| over last 14 days
                close_prices = day_bars['close'].values
                minutes = day_bars['minutes_from_open'].values
                vwap_prices = day_bars['vwap'].values

                # Calculate time-varying boundaries for each bar
                signals = np.zeros(len(day_bars))
                upper_bounds = np.zeros(len(day_bars))
                lower_bounds = np.zeros(len(day_bars))

                # ============================================================
                # STRATEGY ROUTING: ETF (momentum) vs Stock (mean reversion/gap)
                # ============================================================
                is_etf_symbol = self.is_etf(symbol)

                # VIX filter only applies to ETFs - skip ETFs when VIX is unfavorable
                # Stocks trade regardless of VIX (no VIX filter for stocks)
                if is_etf_symbol and not etf_vix_ok:
                    continue

                # For stocks, calculate volatility regime once per day
                stock_vol_rank = None
                if not is_etf_symbol and self.use_stock_vol_regime:
                    _, stock_vol_rank = self.calculate_stock_vol_regime(symbol, trade_date)

                for bar_idx in range(len(day_bars)):
                    mins_from_open = minutes[bar_idx]

                    # Get VIX-adaptive parameters if enabled
                    if self.use_vix_adaptive:
                        adaptive_lookback, adaptive_vm = self._get_adaptive_params(trade_date)
                    else:
                        adaptive_lookback = self.lag
                        adaptive_vm = self.volatility_multiplier

                    # Calculate sigma for this bar (HAR-scaled or backward-looking)
                    if self.use_har_forecast:
                        # CORRECTED HAR: Use scale factor to preserve intraday structure
                        # Old approach replaced sigma entirely, destroying time-varying bounds
                        # New approach: sigma = backward_sigma * HAR_scale_factor

                        # Get sigma history up to yesterday (only on first bar of day)
                        if bar_idx == 0:
                            trade_date_pd = pd.Timestamp(trade_date)
                            historical_sigmas = [
                                entry['sigma']
                                for entry in self.har_sigma_history.get(symbol, [])
                                if entry['date'] < trade_date  # Strictly before today
                            ]

                            days_since_start = (trade_date_pd - pd.Timestamp(self.start_date)).days
                            should_refit = (days_since_start % self.har_refit_frequency == 0)

                            # Refit HAR model periodically
                            if should_refit and len(historical_sigmas) >= 30:
                                if symbol not in self.har_forecaster:
                                    self.har_forecaster[symbol] = ZarattiniSigmaForecaster()

                                fit_result = self.har_forecaster[symbol].fit(
                                    np.array(historical_sigmas)
                                )

                                # Store fitted mean for scale factor calculation
                                if fit_result:
                                    if not hasattr(self, 'har_fitted_means'):
                                        self.har_fitted_means = {}
                                    self.har_fitted_means[symbol] = np.mean(historical_sigmas)

                                    if days_since_start % 100 == 0:  # Print occasionally
                                        print(f"\n  HAR Model Refit for {symbol} on {trade_date}:")
                                        print(f"    R²: {fit_result['r2']:.3f}")
                                        print(f"    Fitted mean: {self.har_fitted_means[symbol]:.4f}")
                                        print(f"    β_daily:   {fit_result['coef_d']:.4f}")
                                        print(f"    β_weekly:  {fit_result['coef_w']:.4f}")
                                        print(f"    β_monthly: {fit_result['coef_m']:.4f}")

                            # Calculate HAR scale factor for today
                            if symbol in self.har_forecaster and len(historical_sigmas) >= 22:
                                sigma_forecast = self.har_forecaster[symbol].forecast(
                                    np.array(historical_sigmas)
                                )

                                # Get fitted mean (use recent mean as fallback)
                                fitted_mean = getattr(self, 'har_fitted_means', {}).get(
                                    symbol, np.mean(historical_sigmas[-30:])
                                )

                                # Calculate scale factor: forecast / historical mean
                                # > 1.0 means tomorrow expected MORE volatile
                                # < 1.0 means tomorrow expected LESS volatile
                                if fitted_mean > 0:
                                    har_scale = sigma_forecast / fitted_mean
                                else:
                                    har_scale = 1.0

                                # Clamp to reasonable range (avoid extreme adjustments)
                                har_scale = np.clip(har_scale, 0.5, 2.0)

                                # Store for use across all bars today
                                if not hasattr(self, 'har_daily_scale'):
                                    self.har_daily_scale = {}
                                self.har_daily_scale[symbol] = har_scale

                                # Store forecast for analysis
                                if symbol not in self.har_forecasts:
                                    self.har_forecasts[symbol] = {}
                                self.har_forecasts[symbol][trade_date] = {
                                    'forecast': sigma_forecast,
                                    'scale_factor': har_scale,
                                    'fitted_mean': fitted_mean
                                }
                            else:
                                # Not enough history, use scale = 1.0
                                if not hasattr(self, 'har_daily_scale'):
                                    self.har_daily_scale = {}
                                self.har_daily_scale[symbol] = 1.0

                        # CRITICAL FIX: Get backward-looking sigma (preserves intraday structure)
                        sigma_backward = self.calculate_intraday_sigma(symbol, trade_date, mins_from_open)

                        # Apply HAR scale factor
                        har_scale = getattr(self, 'har_daily_scale', {}).get(symbol, 1.0)
                        sigma = sigma_backward * har_scale

                    else:
                        # Original: backward-looking sigma
                        sigma = self.calculate_intraday_sigma(symbol, trade_date, mins_from_open)

                    # Apply volatility multiplier (VIX-adaptive if enabled, otherwise fixed)
                    adjusted_sigma = sigma * adaptive_vm

                    # NOW CALCULATE BOUNDARIES using adjusted_sigma
                    upper_bounds[bar_idx] = anchor_high * (1 + adjusted_sigma)
                    lower_bounds[bar_idx] = anchor_low * (1 - adjusted_sigma)

                    # ============================================================
                    # SIGNAL GENERATION: Route based on symbol type
                    # ============================================================
                    base_signal = 0  # Start with no signal

                    if is_etf_symbol:
                        # ETF STRATEGY: Momentum (follow breakouts)
                        # ZARATTINI SIGNALS: Noise boundary breakout WITH VWAP confirmation
                        if close_prices[bar_idx] > upper_bounds[bar_idx] and close_prices[bar_idx] > vwap_prices[bar_idx]:
                            base_signal = 1  # LONG
                        elif close_prices[bar_idx] < lower_bounds[bar_idx] and close_prices[bar_idx] < vwap_prices[bar_idx]:
                            base_signal = -1  # SHORT
                    else:
                        # STOCK STRATEGY: Mean reversion or gap capture
                        if self.stock_strategy_mode in ['mean_reversion', 'hybrid', 'vol_adaptive']:
                            # Mean reversion with VWAP exit target
                            current_price = close_prices[bar_idx]
                            current_vwap = vwap_prices[bar_idx]

                            # Check if we have an open position for this symbol
                            intraday_pos = self.intraday_positions.get(symbol)

                            if intraday_pos is not None:
                                # IN POSITION - check for exit at VWAP
                                direction = intraday_pos['direction']

                                if direction == 1:  # Long position
                                    # Exit when price reverts UP to VWAP
                                    if current_price >= current_vwap:
                                        base_signal = 0  # EXIT
                                        del self.intraday_positions[symbol]
                                    else:
                                        base_signal = 1  # Stay long

                                elif direction == -1:  # Short position
                                    # Exit when price reverts DOWN to VWAP
                                    if current_price <= current_vwap:
                                        base_signal = 0  # EXIT
                                        del self.intraday_positions[symbol]
                                    else:
                                        base_signal = -1  # Stay short
                            else:
                                # NO POSITION - check for entry
                                signal_str = self.generate_mean_reversion_signal(
                                    symbol=symbol,
                                    current_price=current_price,
                                    open_price=open_prices[symbol],
                                    prev_close=prev_closes[symbol],
                                    vwap=current_vwap,
                                    sigma=adjusted_sigma,
                                    vol_rank=stock_vol_rank
                                )

                                if signal_str == 'BUY':
                                    base_signal = 1
                                    # Record entry
                                    self.intraday_positions[symbol] = {
                                        'direction': 1,
                                        'entry_price': current_price,
                                        'entry_vwap': current_vwap
                                    }
                                elif signal_str == 'SELL':
                                    base_signal = -1
                                    # Record entry
                                    self.intraday_positions[symbol] = {
                                        'direction': -1,
                                        'entry_price': current_price,
                                        'entry_vwap': current_vwap
                                    }

                        if self.stock_strategy_mode in ['gap_capture', 'hybrid']:
                            # Gap signal (only at market open)
                            bar_time = day_bars.iloc[bar_idx]['datetime']
                            if isinstance(bar_time, str):
                                bar_time = pd.to_datetime(bar_time)

                            gap_signal = self.generate_gap_signal(
                                symbol=symbol,
                                trade_date=trade_date,
                                current_time=bar_time,
                                open_price=open_prices[symbol],
                                prev_close=prev_closes[symbol],
                                current_price=close_prices[bar_idx]
                            )

                            if gap_signal == 'BUY':
                                base_signal = 1
                            elif gap_signal == 'SELL':
                                base_signal = -1
                            elif gap_signal == 'EXIT' and symbol in self.gap_positions:
                                # Exit gap position
                                base_signal = 0
                                del self.gap_positions[symbol]

                    # ============================================================
                    # RF FILTERING: Only take trades if RF model is confident
                    # ============================================================
                    final_signal = base_signal

                    if self.use_rf and base_signal != 0 and self.rf_model is not None:
                        # Extract features for this bar
                        rf_features = self.extract_rf_features(
                            symbol, trade_date, bar_idx, day_bars, prev_closes[symbol]
                        )

                        if rf_features is not None:
                            # Get RF prediction and confidence
                            rf_signal_str, rf_confidence = self.get_rf_signal(rf_features)

                            # Only take LONG signals if RF agrees and is confident
                            if base_signal == 1:  # Want to go LONG
                                if rf_signal_str == 'BUY' and rf_confidence >= self.rf_confidence_threshold:
                                    final_signal = 1  # RF confirms: take the trade
                                else:
                                    final_signal = 0  # RF blocks: skip this trade

                            # For SHORT signals, we also check RF (but RF only predicts BUY/HOLD)
                            # If base wants SHORT but RF says BUY, that's a strong conflict - block it
                            elif base_signal == -1:  # Want to go SHORT
                                if rf_signal_str == 'BUY' and rf_confidence >= self.rf_confidence_threshold:
                                    final_signal = 0  # RF strongly disagrees: block SHORT
                                else:
                                    final_signal = -1  # RF doesn't object: allow SHORT

                    # Commit the final signal (after RF filtering)
                    signals[bar_idx] = final_signal

                # Only trade at trade_freq intervals after first 30 min
                trade_mask = (minutes % self.trade_freq == 0) & (minutes >= 30)
                exposure = np.full(len(day_bars), np.nan)
                exposure[trade_mask] = signals[trade_mask]

                # Forward fill exposure (0 = explicit exit, stay flat until new signal)
                last_valid = 0  # Start flat
                filled_exposure = []
                for val in exposure:
                    if not np.isnan(val):
                        last_valid = val  # Update position (1=long, -1=short, 0=flat)
                    filled_exposure.append(last_valid)
                exposure = pd.Series(filled_exposure).shift(1).fillna(0).values

                # ============================================================
                # PAPER'S METHOD: Minute-by-Minute P&L Calculation
                # Calculate position size ONCE per day (not per bar)
                # ============================================================
                
                if dvol > 0:
                    vol_scalar = min(self.target_vol / dvol, self.max_leverage)
                else:
                    vol_scalar = 1.0

                # Position size based on day's open price (paper spec)
                shares = int((capital * self.allocations[symbol]) / open_prices[symbol] * vol_scalar)
                
                if shares > 0:
                    # Calculate minute-by-minute price changes
                    # price_changes[0] = 0, price_changes[i] = close[i] - close[i-1]
                    price_changes = np.diff(close_prices, prepend=close_prices[0])

                    # Calculate P&L for each minute based on exposure
                    # If exposure[i] = 1 (long): gain/lose (close[i] - close[i-1]) * shares
                    # If exposure[i] = -1 (short): gain/lose -(close[i] - close[i-1]) * shares
                    # If exposure[i] = 0: no P&L
                    minute_pnl = exposure * price_changes * shares

                    # Count trades (number of exposure changes)
                    exposure_with_final_zero = np.append(exposure, 0)
                    exposure_changes = np.diff(exposure_with_final_zero)
                    trades_count = np.sum(np.abs(exposure_changes))

                    # Calculate transaction costs (paper spec)
                    min_commission = 0.35  # Minimum $0.35 per order
                    commission_per_trade = max(self.commission_per_share * shares, min_commission)
                    total_commission = trades_count * commission_per_trade

                    # Slippage: flat $0.001 per share per trade (paper spec)
                    slippage_per_trade = 0.001 * shares
                    total_slippage = trades_count * slippage_per_trade

                    # Net P&L for the day
                    gross_pnl = np.sum(minute_pnl)
                    net_pnl = gross_pnl - total_commission - total_slippage
                    daily_pnl += net_pnl

                    # Record trades for reporting
                    if trades_count > 0:
                        # Find all trade points (where exposure changes)
                        trade_indices = np.where(exposure_changes != 0)[0]
                        
                        for trade_idx in trade_indices:
                            if trade_idx >= len(day_bars):
                                continue
                                
                            bar = day_bars.iloc[trade_idx]
                            
                            # Determine action
                            if trade_idx == 0:
                                prev_exposure = 0
                            else:
                                prev_exposure = exposure[trade_idx - 1]
                            
                            curr_exposure = exposure[trade_idx]
                            
                            if prev_exposure == 0 and curr_exposure > 0:
                                action = 'BUY'
                            elif prev_exposure == 0 and curr_exposure < 0:
                                action = 'SHORT'
                            elif prev_exposure > 0 and curr_exposure == 0:
                                action = 'SELL'
                            elif prev_exposure < 0 and curr_exposure == 0:
                                action = 'COVER'
                            else:
                                action = 'ADJUST'
                            
                            # Calculate individual trade P&L for this transition
                            trade_pnl = net_pnl / trades_count if trades_count > 0 else 0
                            
                            trades.append({
                                'datetime': bar['datetime'],
                                'date': trade_date,
                                'symbol': symbol,
                                'action': action,
                                'price': bar['close'],
                                'shares': shares,
                                'pnl': trade_pnl
                            })
                            
                            # RF Training: Track entry trades for outcome prediction
                            if self.use_rf and action in ['BUY', 'SHORT']:
                                # Extract features at entry point
                                rf_features = self.extract_rf_features(
                                    symbol, trade_date, trade_idx, day_bars, prev_closes[symbol]
                                )
                                
                                if rf_features:
                                    # Find the corresponding exit
                                    exit_idx = trade_idx + 1
                                    while exit_idx < len(exposure) and exposure[exit_idx] == curr_exposure:
                                        exit_idx += 1
                                    
                                    if exit_idx < len(close_prices):
                                        entry_price = close_prices[trade_idx]
                                        exit_price = close_prices[exit_idx]
                                        
                                        # Record outcome
                                        self.record_trade_outcome(
                                            rf_features, entry_price, exit_price, 
                                            action, bar['datetime']
                                        )

                        # Summary trade record for daily reporting
                        trades.append({
                            'datetime': day_bars.iloc[-1]['datetime'],
                            'date': trade_date,
                            'symbol': symbol,
                            'action': 'DAY_SUMMARY',
                            'price': close_prices[-1],
                            'shares': shares,
                            'trades_count': int(trades_count),
                            'gross_pnl': gross_pnl,
                            'net_pnl': net_pnl
                        })

                        # Log intraday trading activity
                        if trades_count > 0:
                            print(f"    [INTRADAY] {symbol}: {int(trades_count)} trades, "
                                  f"gross: ${gross_pnl:,.2f}, net: ${net_pnl:,.2f}")

            # Update capital with realized P&L
            capital += daily_pnl
            daily_pnl_list.append(daily_pnl)

            # Track swing position value (may have swing positions even in high VIX)
            swing_value = 0
            for sym, pos in self.swing_positions.items():
                if pos.get('quantity', 0) > 0:
                    price = self.get_swing_daily_close(sym, trade_date)
                    if price:
                        swing_value += pos['quantity'] * price

            equity_curve.append({
                'date': trade_date,
                'portfolio_value': capital + swing_value,
                'capital': capital,
                'swing_value': swing_value,
                'daily_pnl': daily_pnl,
                'regime': 'INTRADAY'
            })
            
            # Track rolling returns for adaptive threshold (computed in-loop)
            if self.use_adaptive_threshold and len(equity_curve) > 1:
                prev_val = equity_curve[-2]['portfolio_value']
                curr_val = equity_curve[-1]['portfolio_value']
                if prev_val > 0:
                    daily_return = (curr_val - prev_val) / prev_val
                    self.rolling_returns.append(daily_return)

        # Calculate daily returns
        daily_returns_list = []
        for i in range(1, len(equity_curve)):
            prev_val = equity_curve[i-1]['portfolio_value']
            curr_val = equity_curve[i]['portfolio_value']
            if prev_val > 0:
                daily_returns_list.append((curr_val - prev_val) / prev_val)

        # Liquidate all swing positions at end of backtest
        final_swing_proceeds = 0
        if self.swing_positions:
            print(f"\n  Liquidating {len(self.swing_positions)} open positions at backtest end...")
            for sym, pos in self.swing_positions.items():
                if pos.get('quantity', 0) > 0:
                    exit_price = self.get_swing_daily_close(sym, trading_dates[-1])
                    if exit_price:
                        entry_price = pos['entry_price']
                        shares = pos['quantity']
                        exit_cost = self.apply_transaction_costs(exit_price, shares)
                        proceeds = shares * exit_price - exit_cost
                        pnl = (exit_price - entry_price) * shares - exit_cost
                        final_swing_proceeds += proceeds
                        print(f"    - {sym}: {shares} shares @ ${exit_price:.2f} (entry: ${entry_price:.2f}), PnL: ${pnl:,.2f}")

            # Add liquidation proceeds to capital
            capital += final_swing_proceeds
            self.swing_positions.clear()

        # Store results
        self.trades = pd.DataFrame(trades)
        self.equity_curve = pd.DataFrame(equity_curve)
        self.backtest_daily_returns = np.array(daily_returns_list)

        print(f"\n  Total trades: {len(trades)}")
        print(f"  Trading days: {len(trading_dates)}")
        print(f"  Final portfolio value (all positions closed): ${capital:,.2f}")

        return self.equity_curve

    def calculate_rolling_sharpe(self, returns_list: list, lookback: int = None) -> float:
        """
        Calculate rolling Sharpe ratio from recent daily returns.
        
        Args:
            returns_list: List of daily returns (most recent at end)
            lookback: Number of days to use (default: self.adaptive_lookback)
            
        Returns:
            Annualized Sharpe ratio (0 if insufficient data)
        """
        if lookback is None:
            lookback = self.adaptive_lookback
            
        if len(returns_list) < lookback:
            # Not enough data yet, return neutral value
            return 0.0
        
        # Get most recent returns
        recent_returns = np.array(returns_list[-lookback:])
        
        # Calculate Sharpe (assuming 0 risk-free rate for simplicity)
        if len(recent_returns) > 1:
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns, ddof=1)
            
            if std_return > 0:
                sharpe = (mean_return / std_return) * np.sqrt(252)
                return np.clip(sharpe, -10, 10)  # Bound extreme values
        
        return 0.0
    
    def get_adaptive_vix_threshold(self, rolling_sharpe: float) -> float:
        """
        Get VIX threshold based on rolling Sharpe performance.
        
        Args:
            rolling_sharpe: Current rolling Sharpe ratio
            
        Returns:
            VIX threshold to use
        """
        # Sort thresholds by Sharpe cutoff
        sorted_cutoffs = sorted(self.adaptive_thresholds.keys())
        
        for cutoff in sorted_cutoffs:
            if rolling_sharpe < cutoff:
                return self.adaptive_thresholds[cutoff]
        
        # Default to most aggressive if Sharpe is very high
        return min(self.adaptive_thresholds.values())

    def calculate_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if self.backtest_daily_returns is None or len(self.backtest_daily_returns) == 0:
            return {}

        returns = self.backtest_daily_returns
        equity = self.equity_curve['portfolio_value'].values

        # Basic returns
        total_return = (equity[-1] / equity[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Risk metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Sharpe Ratio (assuming 5% risk-free rate)
        rf_daily = 0.05 / 252
        excess_returns = returns - rf_daily
        std_excess = np.std(excess_returns)
        if std_excess > 1e-10:  # Avoid division by near-zero
            sharpe = np.mean(excess_returns) / std_excess * np.sqrt(252)
            # Clamp to reasonable range
            sharpe = np.clip(sharpe, -10, 10)
        else:
            sharpe = 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else daily_vol
        if downside_std > 1e-10:
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
            sortino = np.clip(sortino, -10, 10)
        else:
            sortino = 0

        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        if len(self.trades) > 0:
            winning_trades = self.trades[self.trades['pnl'] > 0]
            losing_trades = self.trades[self.trades['pnl'] < 0]

            win_rate = len(winning_trades) / len(self.trades[self.trades['pnl'] != 0]) if len(self.trades[self.trades['pnl'] != 0]) > 0 else 0

            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 1

            # Profit factor = sum of wins / sum of losses (correct formula)
            total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trades),
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05
        }

        return metrics

    def walk_forward_test(
        self,
        train_days: int = 20,
        test_days: int = 10,
        long_only: bool = False
    ) -> pd.DataFrame:
        """
        Walk-forward optimization/validation for intraday strategy.

        Trains on rolling window, tests on out-of-sample period.
        """
        if not self.data:
            self.fetch_data()

        print(f"\nWalk-Forward Test: {train_days} days train, {test_days} days test")

        results = []

        # Get trading dates from data
        first_symbol = self.symbols[0]
        if first_symbol not in self.data:
            return pd.DataFrame()

        intraday_df = self.data[first_symbol]['intraday']
        all_dates = sorted(intraday_df['date'].unique())

        if len(all_dates) < train_days + test_days:
            print(f"  Not enough data for walk-forward. Have {len(all_dates)} days, need {train_days + test_days}")
            return pd.DataFrame()

        # Walk through periods
        period_start_idx = 0

        while period_start_idx + train_days + test_days <= len(all_dates):
            train_start = all_dates[period_start_idx]
            train_end = all_dates[period_start_idx + train_days - 1]
            test_start = all_dates[period_start_idx + train_days]
            test_end_idx = min(period_start_idx + train_days + test_days - 1, len(all_dates) - 1)
            test_end = all_dates[test_end_idx]

            print(f"\n  Period: Train {train_start} to {train_end}, Test {test_start} to {test_end}")

            # Filter data for test period (include train data for lookback)
            test_data = {}
            for symbol in self.symbols:
                if symbol not in self.data:
                    continue

                intraday = self.data[symbol]['intraday']
                daily = self.data[symbol]['daily']

                # Include train period for sigma lookback, but only trade during test
                test_intraday = intraday[
                    (intraday['date'] >= train_start) &  # Include train for lookback
                    (intraday['date'] <= test_end)
                ].copy()

                test_data[symbol] = {
                    'intraday': test_intraday,
                    'daily': daily  # Keep full daily for EGARCH calculation
                }
                print(f"    DEBUG {symbol}: {len(test_intraday)} intraday bars")

            # Create sub-backtester
            bt = StrategyBacktester(
                symbols=self.symbols,
                start_date=str(test_start),
                end_date=str(test_end),
                initial_capital=self.initial_capital,
                lag=self.lag,
                target_vol=self.target_vol
            )
            bt.data = test_data

            bt.run_backtest(long_only=long_only)
            metrics = bt.calculate_metrics()

            if metrics:
                metrics['train_start'] = train_start
                metrics['train_end'] = train_end
                metrics['test_start'] = test_start
                metrics['test_end'] = test_end
                results.append(metrics)

            # Roll forward
            period_start_idx += test_days

        self.walk_forward_results = pd.DataFrame(results)
        return self.walk_forward_results

    def print_report(self):
        """Print a comprehensive performance report."""
        metrics = self.calculate_metrics()

        if not metrics:
            print("No metrics to report. Run backtest first.")
            return

        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE REPORT")
        print("=" * 60)

        print(f"\nPeriod: {self.start_date} to {self.end_date}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Final Value: ${self.equity_curve['portfolio_value'].iloc[-1]:,.0f}")

        print("\n" + "-" * 40)
        print("RETURNS")
        print("-" * 40)
        print(f"Total Return:      {metrics['total_return']*100:>10.2f}%")
        print(f"Annual Return:     {metrics['annual_return']*100:>10.2f}%")
        print(f"Annual Volatility: {metrics['annual_volatility']*100:>10.2f}%")

        print("\n" + "-" * 40)
        print("RISK-ADJUSTED METRICS")
        print("-" * 40)
        print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
        print(f"Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}")
        print(f"Max Drawdown:      {metrics['max_drawdown']*100:>10.2f}%")

        print("\n" + "-" * 40)
        print("TRADE STATISTICS")
        print("-" * 40)
        print(f"Total Trades:      {metrics['total_trades']:>10}")
        print(f"Win Rate:          {metrics['win_rate']*100:>10.2f}%")
        print(f"Profit Factor:     {metrics['profit_factor']:>10.2f}")
        print(f"Avg Win:           ${metrics['avg_win']:>10,.2f}")
        print(f"Avg Loss:          ${metrics['avg_loss']:>10,.2f}")

        print("\n" + "-" * 40)
        print("STATISTICAL SIGNIFICANCE")
        print("-" * 40)
        print(f"t-statistic:       {metrics['t_statistic']:>10.3f}")
        print(f"p-value:           {metrics['p_value']:>10.4f}")
        sig_str = "YES ✓" if metrics['statistically_significant'] else "NO ✗"
        print(f"Significant (p<0.05): {sig_str:>7}")

        print("\n" + "=" * 60)

        return metrics

    def plot_results(self, save_path: str = None, show: bool = True):
        """
        Plot comprehensive backtest results.

        Args:
            save_path: If provided, saves the figure to this path
            show: If True, displays the plot
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No results to plot. Run backtest first.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Backtest Results: {self.start_date} to {self.end_date}', fontsize=14, fontweight='bold')

        # Convert dates for plotting
        dates = pd.to_datetime(self.equity_curve['date'])

        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(dates, self.equity_curve['portfolio_value'], 'b-', linewidth=1.5, label='Portfolio')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Drawdown
        ax2 = axes[0, 1]
        equity = self.equity_curve['portfolio_value'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(dates, drawdown, 'r-', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. Daily Returns Distribution
        ax3 = axes[1, 0]
        if self.backtest_daily_returns is not None and len(self.backtest_daily_returns) > 0:
            returns_pct = self.backtest_daily_returns * 100
            ax3.hist(returns_pct, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax3.axvline(x=returns_pct.mean(), color='green', linestyle='-', linewidth=1.5, label=f'Mean: {returns_pct.mean():.2f}%')
            ax3.set_title('Daily Returns Distribution')
            ax3.set_xlabel('Daily Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Cumulative Returns by Month
        ax4 = axes[1, 1]
        self.equity_curve['month'] = pd.to_datetime(self.equity_curve['date']).dt.to_period('M')
        monthly_returns = self.equity_curve.groupby('month').apply(
            lambda x: (x['portfolio_value'].iloc[-1] / x['portfolio_value'].iloc[0] - 1) * 100
            if len(x) > 0 else 0
        )

        colors = ['green' if r > 0 else 'red' for r in monthly_returns.values]
        bars = ax4.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('Monthly Returns')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3, axis='y')

        # Set x-axis labels to month names
        if len(monthly_returns) <= 24:
            ax4.set_xticks(range(len(monthly_returns)))
            ax4.set_xticklabels([str(m) for m in monthly_returns.index], rotation=45, ha='right')
        else:
            # Too many months, show every 3rd
            tick_positions = range(0, len(monthly_returns), 3)
            ax4.set_xticks(tick_positions)
            ax4.set_xticklabels([str(monthly_returns.index[i]) for i in tick_positions], rotation=45, ha='right')

        # Add metrics text box
        metrics = self.calculate_metrics()
        if metrics:
            textstr = '\n'.join([
                f"Total Return: {metrics['total_return']*100:.1f}%",
                f"Annual Return: {metrics['annual_return']*100:.1f}%",
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
                f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%",
                f"Win Rate: {metrics['win_rate']*100:.1f}%",
                f"Total Trades: {metrics['total_trades']}"
            ])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            fig.text(0.98, 0.02, textstr, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right', bbox=props, family='monospace')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")

        if show:
            plt.show()

        return fig

    # ============================================================
    # DIAGNOSTIC METHODS FOR REGIME ANALYSIS
    # ============================================================

    def analyze_vix_buckets(self, vix_buckets=None, long_only=False):
        """
        Analyze intraday strategy performance across different VIX levels.
        
        Uses VIX range filtering (vix_min/vix_max) to ensure statistics are computed correctly.
        
        Args:
            vix_buckets: List of (min_vix, max_vix) tuples. Default: [(10,12), (12,15), (15,17), (17,20), (20,25), (25,99)]
            long_only: Whether to test long-only strategy
            
        Returns:
            DataFrame with performance by VIX bucket
        """
        if vix_buckets is None:
            vix_buckets = [(10,12), (12,15), (15,17), (17,20), (20,25), (25,99)]
        
        print("\n" + "=" * 80)
        print("VIX BUCKET ANALYSIS: Intraday Strategy Performance by VIX Level")
        print("=" * 80)
        
        results = []
        
        # Store original dates before loop (in case loop skips all buckets)
        original_start = self.start_date
        original_end = self.end_date
        
        for vix_min_bucket, vix_max_bucket in vix_buckets:
            # Count days in this VIX bucket (within date range)
            start_dt = pd.to_datetime(self.start_date).date() if self.start_date else None
            end_dt = pd.to_datetime(self.end_date).date() if self.end_date else None
            
            valid_dates = [date for date, vix in self.vix_data.items() 
                          if vix is not None 
                          and vix_min_bucket <= vix < vix_max_bucket
                          and (start_dt is None or date >= start_dt)
                          and (end_dt is None or date <= end_dt)]
            
            if len(valid_dates) < 5:  # Skip if too few days
                print(f"\nVIX {vix_min_bucket}-{vix_max_bucket}: Skipped (only {len(valid_dates)} days)")
                continue
            
            print(f"\nTesting VIX {vix_min_bucket}-{vix_max_bucket} ({len(valid_dates)} days)...")
            
            # Create sub-backtester with VIX RANGE filter (keeps all data, filters by VIX)
            # This ensures statistics are computed correctly
            bt = StrategyBacktester(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                lag=self.lag,
                target_vol=self.target_vol,
                max_leverage=self.max_leverage,
                trade_freq=self.trade_freq,
                vix_min=vix_min_bucket,  # Only trade when VIX >= bucket min
                vix_max=vix_max_bucket,  # Only trade when VIX < bucket max (new param)
                use_swing=False,  # Only test intraday
                use_rf=False  # Disable RF for clean test
            )
            
            # Share data and caches with parent
            bt.data = self.data
            bt.vix_data = self.vix_data
            bt.dvol = self.dvol
            bt.noise_area = self.noise_area
            bt.intraday_sigma_cache = self.intraday_sigma_cache
            
            # Run backtest
            bt.run_backtest(long_only=long_only)
            metrics = bt.calculate_metrics()
            
            if metrics:
                results.append({
                    'vix_bucket': f"{vix_min_bucket}-{vix_max_bucket}",
                    'vix_min': vix_min_bucket,
                    'vix_max': vix_max_bucket,
                    'days': len(valid_dates),
                    'return_pct': metrics['total_return'] * 100,
                    'annual_return_pct': metrics['annual_return'] * 100,
                    'sharpe': metrics['sharpe_ratio'],
                    'max_dd_pct': metrics['max_drawdown'] * 100,
                    'win_rate_pct': metrics['win_rate'] * 100,
                    'total_trades': metrics['total_trades'],
                    'trades_per_day': metrics['total_trades'] / len(valid_dates) if len(valid_dates) > 0 else 0
                })
                
                print(f"  Return: {metrics['total_return']*100:>7.1f}% | Sharpe: {metrics['sharpe_ratio']:>5.2f} | Win: {metrics['win_rate']*100:>5.1f}% | Trades: {metrics['total_trades']}")
        
        # Restore original dates
        self.start_date = original_start
        self.end_date = original_end
        
        results_df = pd.DataFrame(results)
        
        # Print summary table
        print("\n" + "=" * 80)
        print("SUMMARY: Performance by VIX Level")
        print("=" * 80)
        print(f"{'VIX Bucket':<12} {'Days':>6} {'Return%':>9} {'Annual%':>9} {'Sharpe':>7} {'Win%':>6} {'Trades':>7}")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"{row['vix_bucket']:<12} {row['days']:>6.0f} {row['return_pct']:>9.1f} {row['annual_return_pct']:>9.1f} {row['sharpe']:>7.2f} {row['win_rate_pct']:>6.1f} {row['total_trades']:>7.0f}")
        print("=" * 80)
        
        return results_df

    def compare_regime_strategies(self, test_year, vix_threshold=17, long_only=False):
        """
        Compare three approaches for a given year:
        1. Intraday only (no swing, no VIX filter)
        2. Swing only (always swing, never intraday)
        3. Regime switching (VIX >= threshold = intraday, else swing)
        
        Args:
            test_year: Year to test (e.g., 2025)
            vix_threshold: VIX threshold for regime switching
            long_only: Whether to use long-only strategy
            
        Returns:
            DataFrame comparing the three approaches
        """
        print("\n" + "=" * 80)
        print(f"REGIME STRATEGY COMPARISON: {test_year}")
        print("=" * 80)
        
        start_date = f"{test_year}-01-01"
        end_date = f"{test_year}-12-31"
        
        results = []
        
        # Test 1: Intraday Only
        print(f"\n1. Testing INTRADAY ONLY (no VIX filter, no swing)...")
        bt1 = StrategyBacktester(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            lag=self.lag,
            target_vol=self.target_vol,
            max_leverage=self.max_leverage,
            trade_freq=self.trade_freq,
            vix_min=None,  # No filter
            use_swing=False,  # No swing
            use_rf=False
        )
        bt1.data = self.data
        bt1.vix_data = self.vix_data
        bt1.dvol = self.dvol
        bt1.noise_area = self.noise_area
        bt1.intraday_sigma_cache = self.intraday_sigma_cache
        
        bt1.run_backtest(long_only=long_only)
        metrics1 = bt1.calculate_metrics()
        
        if metrics1:
            results.append({
                'strategy': 'Intraday Only',
                'return_pct': metrics1['total_return'] * 100,
                'sharpe': metrics1['sharpe_ratio'],
                'max_dd_pct': metrics1['max_drawdown'] * 100,
                'win_rate_pct': metrics1['win_rate'] * 100,
                'total_trades': metrics1['total_trades']
            })
            print(f"   Return: {metrics1['total_return']*100:.1f}% | Sharpe: {metrics1['sharpe_ratio']:.2f}")
        
        # Test 2: Swing Only
        print(f"\n2. Testing SWING ONLY (always swing, never intraday)...")
        bt2 = StrategyBacktester(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            lag=self.lag,
            target_vol=self.target_vol,
            max_leverage=self.max_leverage,
            trade_freq=self.trade_freq,
            vix_min=999,  # Block all intraday
            use_swing=True,  # Force swing
            use_rf=False
        )
        bt2.data = self.data
        bt2.vix_data = self.vix_data
        bt2.dvol = self.dvol
        bt2.noise_area = self.noise_area
        bt2.intraday_sigma_cache = self.intraday_sigma_cache
        
        bt2.run_backtest(long_only=long_only)
        metrics2 = bt2.calculate_metrics()
        
        if metrics2:
            results.append({
                'strategy': 'Swing Only',
                'return_pct': metrics2['total_return'] * 100,
                'sharpe': metrics2['sharpe_ratio'],
                'max_dd_pct': metrics2['max_drawdown'] * 100,
                'win_rate_pct': metrics2['win_rate'] * 100,
                'total_trades': metrics2['total_trades']
            })
            print(f"   Return: {metrics2['total_return']*100:.1f}% | Sharpe: {metrics2['sharpe_ratio']:.2f}")
        
        # Test 3: Regime Switching
        print(f"\n3. Testing REGIME SWITCHING (VIX >= {vix_threshold} = intraday, else swing)...")
        bt3 = StrategyBacktester(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            lag=self.lag,
            target_vol=self.target_vol,
            max_leverage=self.max_leverage,
            trade_freq=self.trade_freq,
            vix_min=vix_threshold,  # Regime threshold
            use_swing=True,  # Enable swing for low VIX
            use_rf=False
        )
        bt3.data = self.data
        bt3.vix_data = self.vix_data
        bt3.dvol = self.dvol
        bt3.noise_area = self.noise_area
        bt3.intraday_sigma_cache = self.intraday_sigma_cache
        
        bt3.run_backtest(long_only=long_only)
        metrics3 = bt3.calculate_metrics()
        
        if metrics3:
            results.append({
                'strategy': f'Regime Switch (VIX≥{vix_threshold})',
                'return_pct': metrics3['total_return'] * 100,
                'sharpe': metrics3['sharpe_ratio'],
                'max_dd_pct': metrics3['max_drawdown'] * 100,
                'win_rate_pct': metrics3['win_rate'] * 100,
                'total_trades': metrics3['total_trades']
            })
            print(f"   Return: {metrics3['total_return']*100:.1f}% | Sharpe: {metrics3['sharpe_ratio']:.2f}")
        
        results_df = pd.DataFrame(results)
        
        # Print comparison table
        print("\n" + "=" * 80)
        print(f"COMPARISON: {test_year}")
        print("=" * 80)
        print(f"{'Strategy':<30} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Win%':>7} {'Trades':>7}")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"{row['strategy']:<30} {row['return_pct']:>10.1f} {row['sharpe']:>8.2f} {row['max_dd_pct']:>8.1f} {row['win_rate_pct']:>7.1f} {row['total_trades']:>7.0f}")
        print("=" * 80)
        
        # Highlight best performer
        best_idx = results_df['return_pct'].idxmax()
        best_strategy = results_df.loc[best_idx, 'strategy']
        best_return = results_df.loc[best_idx, 'return_pct']
        print(f"\nBest Strategy: {best_strategy} ({best_return:.1f}%)")
        
        return results_df

    def diagnose_regime_problem(self, compare_years=None, vix_thresholds=None, long_only=False):
        """
        Comprehensive diagnostic to understand why different years perform differently.
        
        Args:
            compare_years: List of years to compare (e.g., [2021, 2025])
            vix_thresholds: List of VIX thresholds to test (e.g., [12, 15, 17, 20])
            long_only: Whether to use long-only strategy
            
        Returns:
            dict with diagnostic results
        """
        if compare_years is None:
            compare_years = [2021, 2022, 2023, 2024, 2025]
        
        if vix_thresholds is None:
            vix_thresholds = [12, 15, 17, 20]
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE REGIME DIAGNOSTIC")
        print("=" * 80)
        
        diagnostics = {}
        
        # Part 1: VIX distribution by year
        print("\nPart 1: VIX Distribution Analysis")
        print("-" * 80)
        
        vix_dist = {}
        for year in compare_years:
            year_vix = [vix for date, vix in self.vix_data.items() 
                       if date.year == year and vix is not None]
            
            if year_vix:
                vix_dist[year] = {
                    'mean': np.mean(year_vix),
                    'median': np.median(year_vix),
                    'std': np.std(year_vix),
                    'days_low': sum(1 for v in year_vix if v < 12),
                    'days_mid': sum(1 for v in year_vix if 12 <= v < 17),
                    'days_high': sum(1 for v in year_vix if v >= 17),
                    'total_days': len(year_vix)
                }
                
                print(f"\n{year}:")
                print(f"  Mean VIX: {vix_dist[year]['mean']:.1f}")
                print(f"  Days VIX < 12:  {vix_dist[year]['days_low']:>3} ({vix_dist[year]['days_low']/vix_dist[year]['total_days']*100:>5.1f}%)")
                print(f"  Days VIX 12-17: {vix_dist[year]['days_mid']:>3} ({vix_dist[year]['days_mid']/vix_dist[year]['total_days']*100:>5.1f}%)")
                print(f"  Days VIX >= 17: {vix_dist[year]['days_high']:>3} ({vix_dist[year]['days_high']/vix_dist[year]['total_days']*100:>5.1f}%)")
        
        diagnostics['vix_distribution'] = vix_dist
        
        # Part 2: VIX bucket performance by year
        print("\n" + "=" * 80)
        print("Part 2: Intraday Performance by VIX Level (Across Years)")
        print("-" * 80)
        
        bucket_results = []
        for year in compare_years:
            print(f"\n{year}:")
            for vix_min, vix_max in [(10,12), (12,15), (15,17), (17,20), (20,99)]:
                valid_dates = [date for date, vix in self.vix_data.items() 
                              if date.year == year and vix is not None and vix_min <= vix < vix_max]
                
                if len(valid_dates) < 3:
                    continue
                
                # Quick backtest this bucket
                filtered_data = {}
                for symbol in self.symbols:
                    if symbol not in self.data:
                        continue
                    
                    intraday = self.data[symbol]['intraday'].copy()
                    intraday['date_obj'] = pd.to_datetime(intraday['date']).dt.date
                    filtered_intraday = intraday[intraday['date_obj'].isin(valid_dates)]
                    
                    filtered_data[symbol] = {
                        'intraday': filtered_intraday,
                        'daily': self.data[symbol]['daily']
                    }
                
                bt = StrategyBacktester(
                    symbols=self.symbols,
                    start_date=str(min(valid_dates)),
                    end_date=str(max(valid_dates)),
                    initial_capital=self.initial_capital,
                    lag=self.lag,
                    target_vol=self.target_vol,
                    vix_min=None,
                    use_swing=False,
                    use_rf=False
                )
                bt.data = filtered_data
                bt.vix_data = self.vix_data
                bt.dvol = self.dvol
                bt.noise_area = self.noise_area
                bt.intraday_sigma_cache = self.intraday_sigma_cache
                
                bt.run_backtest(long_only=long_only)
                metrics = bt.calculate_metrics()
                
                if metrics:
                    bucket_results.append({
                        'year': year,
                        'vix_bucket': f"{vix_min}-{vix_max}",
                        'days': len(valid_dates),
                        'return_pct': metrics['total_return'] * 100,
                        'sharpe': metrics['sharpe_ratio'],
                        'win_rate_pct': metrics['win_rate'] * 100
                    })
                    
                    print(f"  VIX {vix_min:>2}-{vix_max:<2}: {metrics['total_return']*100:>7.1f}% (Sharpe: {metrics['sharpe_ratio']:>5.2f}, {len(valid_dates)} days)")
        
        diagnostics['bucket_performance'] = pd.DataFrame(bucket_results)
        
        # Part 3: Optimal VIX threshold by year
        print("\n" + "=" * 80)
        print("Part 3: Optimal VIX Threshold Analysis")
        print("-" * 80)
        
        threshold_results = []
        for year in compare_years:
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            
            print(f"\n{year}: Testing thresholds {vix_thresholds}...")
            
            for threshold in vix_thresholds:
                bt = StrategyBacktester(
                    symbols=self.symbols,
                    start_date=year_start,
                    end_date=year_end,
                    initial_capital=self.initial_capital,
                    lag=self.lag,
                    target_vol=self.target_vol,
                    vix_min=threshold,
                    use_swing=True,
                    use_rf=False
                )
                bt.data = self.data
                bt.vix_data = self.vix_data
                bt.dvol = self.dvol
                bt.noise_area = self.noise_area
                bt.intraday_sigma_cache = self.intraday_sigma_cache
                
                bt.run_backtest(long_only=long_only)
                metrics = bt.calculate_metrics()
                
                if metrics:
                    threshold_results.append({
                        'year': year,
                        'vix_threshold': threshold,
                        'return_pct': metrics['total_return'] * 100,
                        'sharpe': metrics['sharpe_ratio']
                    })
                    
                    print(f"  VIX >= {threshold}: {metrics['total_return']*100:>7.1f}% (Sharpe: {metrics['sharpe_ratio']:>5.2f})")
        
        diagnostics['threshold_optimization'] = pd.DataFrame(threshold_results)
        
        # Summary
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        
        print("\nKey Findings:")
        print("1. VIX Distribution: Check if 2025 had more low-VIX days")
        print("2. Bucket Performance: Check if intraday fails at different VIX levels in 2025")
        print("3. Optimal Threshold: Different years may need different thresholds")
        
        return diagnostics


def run_parameter_sensitivity(
    symbols: list,
    start_date: str,
    end_date: str,
    lags: list = [7, 14, 21, 28],
    target_vols: list = [0.15, 0.20, 0.25, 0.30]
) -> pd.DataFrame:
    """Run parameter sensitivity analysis."""

    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    results = []

    for lag in lags:
        for target_vol in target_vols:
            print(f"\nTesting lag={lag}, target_vol={target_vol:.0%}...")

            bt = StrategyBacktester(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                lag=lag,
                target_vol=target_vol
            )
            bt.load_from_csv()
            bt.run_backtest(long_only=False)
            metrics = bt.calculate_metrics()

            results.append({
                'lag': lag,
                'target_vol': target_vol,
                'sharpe': metrics.get('sharpe_ratio', 0),
                'annual_return': metrics.get('annual_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            })

    df = pd.DataFrame(results)

    print("\n" + "-" * 40)
    print("RESULTS (Sharpe by parameter combination)")
    print("-" * 40)

    pivot = df.pivot(index='lag', columns='target_vol', values='sharpe')
    print(pivot.round(2).to_string())

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Index ETFs only - intraday momentum is a market-wide effect
    SYMBOLS = ['NVDA']

    # Check if IBKR CSV data exists (prefer 1m, fall back to 15m)
    CSV_EXISTS = (
        os.path.exists(os.path.join(DATA_DIR, f'{SYMBOLS[0]}_1m.csv')) or
        os.path.exists(os.path.join(DATA_DIR, f'{SYMBOLS[0]}_15m.csv'))
    )

    if not CSV_EXISTS:
        print("=" * 60)
        print("ERROR: No IBKR data found!")
        print("=" * 60)
        print("\nPlease extract IBKR data first:")
        print("  1. Start TWS or IBKR Gateway")
        print("  2. Run: python -c \"from ibkr_backtest import extract_ibkr_data; extract_ibkr_data()\"")
        print("  3. Then run this backtest again")
        print("=" * 60)
        exit(1)

    # Year-by-year comparison to detect decay vs variance
    test_periods = [
        # ('2020-01-01', '2020-12-31', '2020 (COVID)'),
        # ('2021-01-01', '2021-12-31', '2021 (Bull)'),
        # ('2022-01-01', '2022-12-31', '2022 (Bear)'),
        ('2023-01-01', '2023-12-31', '2023 (Recovery)'),
        # ('2024-01-01', '2024-12-31', '2024'),
        # ('2025-01-01', '2025-12-01', '2025 (Tariffs)'),
    ]

    VIX_MIN = 0  # VIX filter threshold

    print("=" * 60)
    print(f"ZARATTINI BACKTEST - DECAY ANALYSIS (VIX >= {VIX_MIN})")
    print("=" * 60)

    results_summary = []

    for start, end, label in test_periods:
        bt = StrategyBacktester(
            symbols=SYMBOLS,
            start_date=start,
            end_date=end,
            lag=14,
            vix_min=VIX_MIN,
            early_stop_pct=0.003,  # Exit if loss > 0.3% in first 30 min
            last_entry_time="16:00",
            skip_opex=False,
            use_rf=False,
            use_swing=False
        )
        bt.load_from_csv()
        bt.run_backtest(long_only=False)
        metrics = bt.calculate_metrics()

        results_summary.append({
            'period': label,
            'return': metrics['total_return'] * 100,
            'sharpe': metrics['sharpe_ratio'],
            'win_rate': metrics['win_rate'] * 100,
            'trades': metrics['total_trades']
        })

        print(f"\n{label}:")
        print(f"  Return: {metrics['total_return']*100:.1f}%  |  Sharpe: {metrics['sharpe_ratio']:.2f}  |  Win: {metrics['win_rate']*100:.1f}%  |  Trades: {metrics['total_trades']}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY - YEAR BY YEAR (VIX >= {})".format(VIX_MIN))
    print("=" * 60)
    print(f"{'Period':<20} {'Return':>10} {'Sharpe':>10} {'Win%':>10} {'Trades':>10}")
    print("-" * 60)
    for r in results_summary:
        print(f"{r['period']:<20} {r['return']:>9.1f}% {r['sharpe']:>10.2f} {r['win_rate']:>9.1f}% {r['trades']:>10}")
    print("-" * 60)

    # Calculate trend
    returns = [r['return'] for r in results_summary]
    if len(returns) >= 3:
        first_half_avg = np.mean(returns[:len(returns)//2])
        second_half_avg = np.mean(returns[len(returns)//2:])
        print(f"\nFirst half avg return:  {first_half_avg:.1f}%")
        print(f"Second half avg return: {second_half_avg:.1f}%")
        if second_half_avg < first_half_avg * 0.7:
            print("⚠️  DECAY DETECTED: Returns declining over time")
        else:
            print("✓  No clear decay pattern - likely variance")