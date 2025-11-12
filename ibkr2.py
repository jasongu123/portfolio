import inspect
import json
import pickle
import random
import threading
import time, datetime
import queue
import pandas as pd
import numpy as np
import os
from threading import Thread
from lightweight_charts import Chart
from wandb import sklearn
from gpt4o_technical_analyst import analyze_chart
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.client import Contract, Order, ScannerSubscription
from ibapi.tag_value import TagValue
import sqlite3
import pytz
from arch import arch_model
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

# create a queue for data coming from Interactive Brokers API
data_queue = queue.Queue()
historical_data_queue = queue.Queue()
scanner_data_queue = queue.Queue()
calc_queue = queue.Queue()  # New queue for calculations
# Add this near the top, after other global variables like data_queue
current_df = pd.DataFrame()

global_trading_system = None
global_trading_strategy = None

# a list for keeping track of any indicator lines
current_lines = []

# initial chart symbol to show
INITIAL_SYMBOL = "NVDA"

# settings for live trading vs. paper trading mode
LIVE_TRADING = True
LIVE_TRADING_PORT = 7496
PAPER_TRADING_PORT = 7497
TRADING_PORT = LIVE_TRADING_PORT
if LIVE_TRADING:
    TRADING_PORT = LIVE_TRADING_PORT

# these defaults are fine
DEFAULT_HOST = '127.0.0.1'
DEFAULT_CLIENT_ID = 1

# def get_recent_bars(symbol, timeframe, duration):
#     """
#     Gets historical bar data from IBKR
#     This reuses the existing historical data functionality but packages it nicely
#     """
#     data_queue.queue.clear()  # Clear any existing data
    
#     contract = Contract()
#     contract.symbol = symbol
#     contract.secType = 'STK'
#     contract.exchange = 'SMART'
#     contract.currency = 'USD'
    
#     client.reqHistoricalData(
#         1, contract, '', duration, timeframe, 'TRADES',
#         1, 1, False, []
#     )
    
#     # Collect the data
#     bars = []
#     timeout = time.time() + 10  # 10 second timeout
#     while time.time() < timeout:
#         try:
#             bar = data_queue.get(timeout=0.1)
#             bars.append(bar)
#         except queue.Empty:
#             break
    
#     if not bars:
#         return pd.DataFrame()
        
class ExchangeCache:
    def __init__(self, client_instance, cache_file='exchange_cache.json'):
        self.client = client_instance
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
        self.cache_lock = threading.Lock()
        
    def load_cache(self):
        """Load cached exchange information from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} exchange entries from cache")
        except Exception as e:
            print(f"Error loading exchange cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save current cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving exchange cache: {e}")
    
    def get_exchange(self, symbol):
        """Get exchange info for symbol (from cache or IBKR)"""
        with self.cache_lock:
            # Check cache first
            if symbol in self.cache:
                return self.cache[symbol]['primary'], self.cache[symbol]['exchange']
            
            # If not in cache, query IBKR
            primary, exchange = get_correct_exchange(symbol, self.client)
            
            # Update cache
            self.cache[symbol] = {
                'primary': primary,
                'exchange': exchange,
                'last_updated': time.time()
            }
            
            # Save cache periodically (could be optimized to save less frequently)
            self.save_cache()
            
            return primary, exchange

class TradingSystem:
    def __init__(self, existing_client):
        """Initialize the trading system with enhanced strategy"""
        self.client = existing_client
        self.exchange_cache = ExchangeCache(existing_client)
        self.db_conn = self.init_database()
        self.db_lock = threading.Lock()
        self.trading_day = None
        self.risk_manager = RiskManager(existing_client, self.exchange_cache)
        self.client.add_callback_handler(self.risk_manager)
        self.trading_strategy = EnhancedTradingStrategy(self.risk_manager, existing_client)
        self.fade_strategy = FadeStrategy(self.risk_manager, existing_client, self.exchange_cache)
        
        
        self.is_market_open = False
        # Define your universe of stocks (same as backtest)
        self.monitored_symbols = ['NVDA', 'JPM', 'GOOG', 'WMT', 'QQQ', 'SPY']  # Or any symbols you want
     
        self.current_positions = {}  # symbol -> quantity
        self.position_update_event = threading.Event()
        
        # Load existing positions on startup
        self.load_existing_positions()

    def init_database(self):
        """Creates or connects to a database to persist trading data"""
        conn = sqlite3.connect('trading_system.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # Create tables for tracking trades and daily PnL
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                realized_pnl REAL,
                unrealized_pnl REAL,
                total_pnl REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER,
                average_cost REAL,
                current_price REAL,
                last_updated TIMESTAMP
            )
        ''')
        
        return conn    
    def check_trading_signals(self):
        """Check for trading signals using the enhanced strategy"""
        # Update allocations monthly
        print("Checking trading signals...")
        self.trading_strategy.update_allocations(self.monitored_symbols)
        
        for symbol in self.monitored_symbols:
            try:
                # The enhanced strategy handles its own data collection
                signal = self.trading_strategy.generate_signal(symbol)
                
                # Execute trade if signal is generated
                if signal != 'HOLD':
                    print(f"Enhanced signal for {symbol}: {signal}")
                    self.trading_strategy.execute_trade(symbol, signal)
                    
            except Exception as e:
                print(f"Error checking signals for {symbol}: {e}")

    def check_fade_signals(self, symbol, current_time):
        """Check and execute fade strategy signals"""
        try:
            # Get current market data
            current_price = get_current_price(symbol)
            if current_price is None:
                return
            
            # Get noise boundaries from momentum strategy
            if symbol in self.trading_strategy.symbol_data:
                data = self.trading_strategy.symbol_data[symbol]
                if data.get('egarch_vol') is not None:
                    # Calculate noise boundaries
                    egarch_vol = data['egarch_vol']
                    
                    # Get today's open and previous close
                    intraday_df = data.get('intraday_data', pd.DataFrame())
                    if not intraday_df.empty:
                        today = current_time.date()
                        today_data = intraday_df[intraday_df['day'] == today]
                        
                        if not today_data.empty:
                            open_price = today_data.iloc[0]['open']
                            
                            # Get previous close
                            yesterday_data = intraday_df[intraday_df['day'] < today]
                            if not yesterday_data.empty:
                                prev_close = yesterday_data.iloc[-1]['close']
                            else:
                                prev_close = open_price
                            
                            # Calculate boundaries (symmetric around open_price)
                            upper_bound = open_price * (1 + egarch_vol)
                            lower_bound = open_price * (1 - egarch_vol)
                            
                            # Check for fade entries
                            fade_signal, trigger_level = self.fade_strategy.should_fade(
                                symbol, current_price, (upper_bound, lower_bound), current_time
                            )
                            
                            if fade_signal:
                                print(f"FADE SIGNAL: {fade_signal} for {symbol} at ${current_price:.2f}")
                                self.fade_strategy.execute_fade_trade(symbol, fade_signal, current_price, current_time)
                            
                            # Check for fade exits
                            exit_reason = self.fade_strategy.check_fade_exits(symbol, current_price, current_time)
                            if exit_reason:
                                print(f"FADE EXIT: {exit_reason} for {symbol}")
                                self.fade_strategy.execute_fade_exit(symbol, exit_reason, current_price)
            
        except Exception as e:
            print(f"Error in fade signal check for {symbol}: {e}")

    def load_existing_positions(self):
        """Load existing positions from IBKR at startup"""
        print("Loading existing positions from IBKR...")
        
        # Clear any existing position data
        self.current_positions.clear()
        self.position_update_event.clear()
        
        # Add temporary callback to handle position updates
        original_position = self.client.position
        original_position_end = self.client.positionEnd
        
        def temp_position(account, contract, pos, avgCost):
            """Temporary callback to capture position data"""
            if contract.secType == 'STK' and pos != 0:
                symbol = contract.symbol
                print(f"Found existing position: {symbol} - {pos} shares @ ${avgCost:.2f}")
                
                # Update our tracking
                self.current_positions[symbol] = pos
                
                # Update the strategy's position tracking if it's one of our monitored symbols
                if symbol in self.monitored_symbols:
                    self.trading_strategy.symbol_data[symbol]['current_position'] = pos
                
                # Update database
                with self.db_lock:
                    cursor = self.db_conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO positions 
                        (symbol, quantity, average_cost, current_price, last_updated)
                        VALUES (?, ?, ?, ?, datetime('now'))
                    ''', (symbol, pos, avgCost, avgCost))  # Use avgCost as current_price initially
                    self.db_conn.commit()
        
        def temp_position_end():
            """Temporary callback when all positions have been received"""
            self.position_update_event.set()
            print("Finished loading positions")
        
        try:
            # Set temporary callbacks
            self.client.position = temp_position
            self.client.positionEnd = temp_position_end
            
            # Request current positions
            self.client.reqPositions()
            
            # Wait for positions to be loaded (with timeout)
            if self.position_update_event.wait(timeout=10):
                print(f"Successfully loaded {len(self.current_positions)} existing positions")
                
                # Display summary of loaded positions
                if self.current_positions:
                    print("\nCurrent positions summary:")
                    for symbol, quantity in self.current_positions.items():
                        print(f"  {symbol}: {quantity} shares")
                else:
                    print("No existing positions found")
            else:
                print("Timeout waiting for position data - proceeding without existing positions")
                
        except Exception as e:
            print(f"Error loading existing positions: {e}")
        finally:
            # Restore original callbacks
            self.client.position = original_position
            self.client.positionEnd = original_position_end
            
            # Cancel the position updates request
            self.client.cancelPositions()
    
    def update_positions(self):
        """Updates position values and PnL from IBKR"""
        # Request account updates from IBKR
        self.client.reqAccountUpdates(True, "")
        
        # Also request position updates to keep our tracking current
        self.client.reqPositions()
        time.sleep(0.5)  # Give it time to update
        self.client.cancelPositions()
        
    def end_trading_day(self):
        """Called when market closes for the day"""
        print(f"Ending trading day: {self.trading_day}")
        
        # Update final positions and P&L
        self.update_positions()
        
        # Update ML model with today's trades
        if self.trading_strategy:
            self.trading_strategy.update_rf_model_daily()
            print("Daily RF model update completed")
    
    def run(self):
        """Modified run loop to match the backtest trading frequency"""

        print("Starting trading system - loading existing positions first...")
        self.load_existing_positions()

        while True:
            current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
            
            if self.is_trading_day(current_time):
                print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Checking trading day status...")
                if self.trading_day != current_time.date():
                    self.start_trading_day(current_time.date())
                
                if self.is_market_hours(current_time):
                    self.is_market_open = True
                    self.update_positions()
                    
                    # Check every minute, but the strategy internally
                    # only trades at specified intervals (every 15 min)
                    minutes_from_open = (current_time.hour - 9) * 60 + (current_time.minute - 30)
                    
                    # Only check signals after first 30 minutes and at trade frequency
                    if minutes_from_open >= 1 or minutes_from_open % 1 == 0:
                        print("Checking trading signals...")
                        self.check_trading_signals()

                    if minutes_from_open >= 30:  # Start fade checking after first 30 minutes
                        for symbol in self.monitored_symbols:
                            self.check_fade_signals(symbol, current_time)
                        
                else:
                    print(f"Market closed. Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    if self.is_market_open:  # Market just closed
                        self.end_trading_day()
                        self.is_market_open = False
            
            time.sleep(30)  # Check every minute
            
    def update_positions(self):
        """Updates position values and PnL from IBKR"""
        # Request account updates from IBKR
        self.client.reqAccountUpdates(True, "")
            
    def start_trading_day(self, date):
        """Initializes a new trading day"""
        self.trading_day = date
        
        # Create new daily PnL record
        with self.db_lock:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO daily_pnl (date, realized_pnl, unrealized_pnl, total_pnl)
                VALUES (?, 0, 0, 0)
            ''', (date.strftime('%Y-%m-%d'),))
            self.db_conn.commit()
        
        # Load previous positions from database
        self.update_positions()

        self.load_existing_positions()

    def is_trading_day(self, dt):
        """Determines if this is a trading day (excluding weekends and holidays)"""
        if dt.weekday() in (5, 6):  # Weekend
            return False
            
        # You would add holiday checking here
        return True
    def is_market_hours(self, dt):
        """Checks if we're in regular market hours"""
        market_open = dt.replace(hour=9, minute=30, second=0)
        market_close = dt.replace(hour=16, minute=0, second=0)
        return market_open <= dt <= market_close

# Additional helper function to add to ibkr1.py
def get_recent_bars_with_vwap(symbol, timeframe, duration, for_calculation=False, 
                             client_instance=None, exchange_cache=None):
    """
    Enhanced version of get_recent_bars that calculates VWAP and other metrics
    needed for the enhanced strategy.
    """
    # Get base data
    df = get_recent_bars(symbol, timeframe, duration, for_calculation, 
                        client_instance, exchange_cache)
    
    if not df.empty and timeframe in ['1 min', '5 mins', '15 mins']:
        # Add day column
        df['day'] = pd.to_datetime(df['date']).dt.date
        
        # Calculate minutes from market open
        df['min_from_open'] = ((pd.to_datetime(df['date']) - 
                               pd.to_datetime(df['date']).dt.normalize()) / 
                               pd.Timedelta(minutes=1)) - (9 * 60 + 30)
        
        # Calculate VWAP for each day
        for day in df['day'].unique():
            day_mask = df['day'] == day
            day_data = df[day_mask]
            
            # Calculate VWAP
            hlc = (day_data['high'] + day_data['low'] + day_data['close']) / 3
            vol_x_hlc = hlc * day_data['volume']
            cum_vol_x_hlc = vol_x_hlc.cumsum()
            cum_volume = day_data['volume'].cumsum()
            
            df.loc[day_mask, 'vwap'] = cum_vol_x_hlc.values / cum_volume.values
    
    return df
    
def get_correct_exchange(symbol, client_instance):
    """
    Determines the correct primary exchange for a stock symbol by querying IBKR.
    Returns a tuple of (primaryExchange, exchange) strings.
    """
    # Create a basic contract with minimal info
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.currency = 'USD'
    contract.exchange = 'SMART'  # Start with SMART routing
    
    # Create a synchronized request mechanism
    exchange_info = {'primary': None, 'valid_exchanges': None}
    request_complete = threading.Event()
    
    # Save original callbacks
    original_contract_details = client_instance.contractDetails
    original_contract_details_end = client_instance.contractDetailsEnd
    
    # Create temporary callbacks
    def temp_contract_details(reqId, details):
        exchange_info['primary'] = details.contract.primaryExchange
        exchange_info['valid_exchanges'] = details.validExchanges
        print(f"Found details for {symbol}: Primary={details.contract.primaryExchange}, Valid={details.validExchanges}")
    
    def temp_contract_details_end(reqId):
        request_complete.set()
    
    try:
        # Set our temporary callbacks
        client_instance.contractDetails = temp_contract_details
        client_instance.contractDetailsEnd = temp_contract_details_end
        
        # Make the request
        req_id = int(time.time() * 1000) % 10000
        client_instance.reqContractDetails(req_id, contract)
        
        # Wait for completion (with timeout)
        if not request_complete.wait(timeout=10):
            print(f"Timeout waiting for contract details for {symbol}")
            return 'SMART', 'SMART'  # Default fallback
        
        # Return the exchange information
        primary = exchange_info['primary'] if exchange_info['primary'] else 'SMART'
        return primary, 'SMART'
        
    finally:
        # Restore original callbacks
        client_instance.contractDetails = original_contract_details
        client_instance.contractDetailsEnd = original_contract_details_end


def recover_state(self):
    """Recovers the trading state after an unexpected shutdown"""
    cursor = self.db_conn.cursor()
    
    # Get the most recent trading day's data
    cursor.execute('''
        SELECT date, realized_pnl, unrealized_pnl 
        FROM daily_pnl 
        ORDER BY date DESC 
        LIMIT 1
    ''')
    last_record = cursor.fetchone()
    
    if last_record:
        # Verify against IBKR's records
        self.client.reqAccountUpdates(True, "")
        # Wait for account updates to come in
        time.sleep(2)
        
        # Compare and reconcile any differences
        self.reconcile_positions()

class PriceWrapper(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.bid = None
        self.ask = None
        self.price_ready = False
        self.lock = threading.Lock()
        self.request_id = 1
            
    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 1:  # Bid
            self.bid = price
        elif tickType == 2:  # Ask
            self.ask = price
        if self.bid and self.ask:
            self.price_ready = True
            
    def get_price(self, symbol):
        with self.lock:
            # Reset state
            self.bid = None
            self.ask = None
            self.price_ready = False
            
            # Prepare request
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.currency = "USD"
            contract.exchange = "ISLAND"  # Direct NASDAQ access for better data
            
            # Request market data with unique ID
            self.request_id += 1
            current_id = self.request_id
            self.reqMktData(current_id, contract, '', False, False, [])
            
            # Wait for data with timeout
            timeout = time.time() + 5
            while not self.price_ready and time.time() < timeout:
                time.sleep(0.1)
                
            # Cancel the request
            self.cancelMktData(current_id)
            
            # Return result
            if self.bid and self.ask:
                return (self.bid + self.ask) / 2
            return None
                
price_wrapper = PriceWrapper()
price_wrapper.connect('127.0.0.1', TRADING_PORT, 999)

# Start the client thread once
thread = Thread(target=price_wrapper.run)
thread.daemon = True
thread.start()

def get_current_price(symbol):
    """
    Gets the current price using IBKR snapshot data
    Returns the midpoint of bid-ask as an estimate of current price
    """
    # Create a Contract object for the symbol
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.currency = "USD"
    contract.exchange = "SMART"
    
    
    
    # Get the price
    wrapper = PriceWrapper()
    wrapper.connect('127.0.0.1', TRADING_PORT, 123)
    thread = Thread(target=wrapper.run, daemon=True)
    thread.start()
    wrapper.reqMarketDataType(1)   # 1 = real-time streaming
    wrapper.reqMktData(1, contract, '', False, False, [])
    
    # Wait for price (with timeout)
    timeout = time.time() + 5  # 5 second timeout
    while not wrapper.price_ready and time.time() < timeout:
        time.sleep(0.1)
    
    wrapper.disconnect()
    
    if wrapper.bid and wrapper.ask:
        return (wrapper.bid + wrapper.ask) / 2
    else:
        if global_trading_system and hasattr(global_trading_system, 'exchange_cache'):
            df = get_recent_bars(symbol, '1 min', '1 D', 
                                client_instance=client, 
                                exchange_cache=global_trading_system.exchange_cache)
        else:
            df = get_recent_bars(symbol, '1 min', '1 D')
        if not df.empty:
            return df['close'].iloc[-1]
        return None
    
# def calculate_volatility(symbol, window='20 D', use_intraday=True):
#     try:
#         if use_intraday:
#             # Use 5-minute bars if we want intraday volatility
#             df = get_recent_bars(symbol, '5 mins', window, for_calculation=True)
#         else:
#             # Use daily bars for standard volatility
#             df = get_recent_bars(symbol, '1 day', window, for_calculation=True)
            
#         if df.empty:
#             print(f"No data available for {symbol} volatility calculation")
#             return 0.02  # Return a default volatility of 2%
            
#         # Calculate log returns
#         df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
#         # Remove any infinite or NaN values
#         df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
#         if len(df) < 2:
#             print(f"Insufficient data for {symbol} volatility calculation")
#             return 0.02  # Return a default volatility
            
#         # Calculate annualized volatility
#         scaling_factor = np.sqrt(252 * 78) if use_intraday else np.sqrt(252)
#         volatility = df['returns'].std() * scaling_factor
        
#         # Add bounds to prevent extreme values
#         volatility = max(min(volatility, 2.0), 0.01)  # Cap between 1% and 100%
        
#         return volatility
        
#     except Exception as e:
#         print(f"Error calculating volatility: {e}")
#         return 0.02  # Return a default volatility as fallback

def get_recent_bars(symbol, timeframe, duration, for_calculation=False, client_instance=None, exchange_cache=None, end_datetime=None):
    print(f"\nDEBUG: Data request details:")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Duration: {duration}")
    print(f"Called by: {inspect.stack()[1].function}")
    
    
    # Use the provided client instance or fallback to global
    client_to_use = client_instance if client_instance else client
    target_queue = calc_queue if for_calculation else data_queue
    
    # Generate a more unique request ID
    req_id = int(time.time() * 1000) % 10000+random.randint(0, 5000)
    
    # Create a local queue just for this request to avoid conflicts
    local_queue = queue.Queue()
    
    # Store original callback
    original_callback = client_to_use.historicalData
    
    def temp_callback(reqId, bar):
        if reqId != req_id:  # Only process data for our specific request
            return
            
        try:
            if ' ' in str(bar.date):
                t = datetime.datetime.strptime(str(bar.date), '%Y%m%d %H:%M:%S')
            else:
                t = datetime.datetime.fromtimestamp(int(float(bar.date)))
                
            data = {
                'date': t,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': int(bar.volume)
            }
            local_queue.put(data)
            target_queue.put(data)  # Also put in the target queue if needed
            data_queue.put(data)  
            if for_calculation:
                calc_queue.put(data)
        except Exception as e:
            print(f"Error in historical data callback: {e}")
    
    try:
        # Use our temporary callback
        client_to_use.historicalData = temp_callback
        
        primary_exchange, exchange = exchange_cache.get_exchange(symbol)
        # Define contract
        contract = Contract()
        contract.symbol = symbol
        contract.currency = 'USD'
        if symbol == "VIX":
            contract.secType = 'IND'
            contract.exchange = 'CBOE'
            print(f"Using INDEX contract for {symbol}")
        else:
            # Regular stock handling
            contract.secType = 'STK'
            contract.exchange = 'SMART'
            
            if exchange_cache:
                primary_exchange, exchange = exchange_cache.get_exchange(symbol)
                contract.primaryExchange = primary_exchange
            else:
                # Default to common US exchange values
                contract.primaryExchange = 'NASDAQ'
                print(f"Warning: No exchange_cache provided for {symbol}, using defaults")
        
        
        # Make the request
        client_to_use.reqHistoricalData(
            req_id, contract, end_datetime if end_datetime else '', duration, timeframe, 'TRADES',
            1, 1, False, []
        )
        
        # Collect the data with improved timing
        bars = []
        first_timeout = time.time() + 30  # Increased timeout 
        last_data_time = time.time()
        
        while time.time() < first_timeout or time.time() - last_data_time < 3:
            try:
                bar = local_queue.get(timeout=0.5)
                bars.append(bar)
                last_data_time = time.time()
            except queue.Empty:
                if len(bars) > 0 and time.time() - last_data_time > 2:
                    break
                time.sleep(0.1)
        
        if not bars:
            print(f"No data collected for {symbol}!")
            return pd.DataFrame()
            
        df = pd.DataFrame(bars)
        print(f"Successfully collected {len(df)} bars of data for {symbol}")
        return df
        
    finally:
        # Always restore the original callback and cancel the request
        client_to_use.historicalData = original_callback
        client_to_use.cancelHistoricalData(req_id)

class RiskManager(EWrapper):
    def __init__(self, client, exchange_cache=None):
        EWrapper.__init__(self)
        self.client = client
        print(f"DEBUG: RiskManager received exchange_cache: {exchange_cache}")  # Add this line
        print(f"DEBUG: exchange_cache type: {type(exchange_cache)}")
        self.exchange_cache = exchange_cache
        print(f"DEBUG: RiskManager stored exchange_cache: {self.exchange_cache}")
        self.account_value = None
        self.max_position_size = 15000 # $1M max position
        self.max_daily_loss = 500      # $500 max daily loss
        self.position_sizing_pct = 0.05   # 1% risk per trade
        self.current_positions = {}
        self.daily_pnl = 0
        
        self.account_update_event = threading.Event()
        self.last_update_time = 0
        self.update_frequency = 5  # Expected update frequency in seconds
        
        # Start continuous account updates immediately
        self.start_continuous_updates()
        
    def start_continuous_updates(self):
        """
        Start receiving continuous account updates.
        This should be called once and left running.
        """
        try:
            # Request continuous updates for all accounts
            self.client.reqAccountUpdates(True, "")
            print("Started continuous account updates")
        except Exception as e:
            print(f"Error starting account updates: {e}")
    
    def stop_continuous_updates(self):
        """
        Stop receiving account updates.
        Only call this when shutting down the system.
        """
        try:
            self.client.reqAccountUpdates(False, "")
            print("Stopped continuous account updates")
        except Exception as e:
            print(f"Error stopping account updates: {e}")
    
    def updateAccountValue(self, key, value, currency, accountName):
        """
        Callback for account value updates.
        This will be called continuously by IBKR.
        """
        if key == "NetLiquidation" and currency == "USD":
            try:
                new_value = float(value)
                self.account_value = new_value
                self.last_update_time = time.time()
                
                # Log updates to monitor frequency
                print(f"Account value updated: ${new_value:,.2f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
                
            except ValueError:
                print(f"Invalid account value received: {value}")
        
        # Also track daily P&L updates
        elif key == "RealizedPnL" and currency == "USD":
            try:
                self.daily_realized_pnl = float(value)
            except ValueError:
                pass
                
        elif key == "UnrealizedPnL" and currency == "USD":
            try:
                self.daily_unrealized_pnl = float(value)
                self.daily_pnl = getattr(self, 'daily_realized_pnl', 0) + self.daily_unrealized_pnl
            except ValueError:
                pass
    
    def get_account_value(self):
        """
        Get the current account value.
        Returns cached value if recent, otherwise waits for fresh update.
        """
        current_time = time.time()
        
        # Check if we have a recent value (within last 10 seconds)
        if self.account_value is not None and (current_time - self.last_update_time) < 10:
            print(f"Using recent account value: ${self.account_value:,.2f} (age: {current_time - self.last_update_time:.1f}s)")
            return self.account_value
        
        # If value is stale or missing, wait for an update
        print("Account value is stale or missing, waiting for update...")
        
        # Store current value to detect changes
        initial_value = self.account_value
        initial_time = self.last_update_time
        
        # Wait up to 5 seconds for a fresh update
        timeout = current_time + 5
        while time.time() < timeout:
            if self.last_update_time > initial_time:
                # We got a fresh update
                print(f"Received fresh account value: ${self.account_value:,.2f}")
                return self.account_value
            time.sleep(0.1)
        
        # If we still don't have a value, return a default
        if self.account_value is not None:
            print(f"Warning: Using stale account value: ${self.account_value:,.2f} (age: {current_time - self.last_update_time:.1f}s)")
            return self.account_value
        else:
            print("ERROR: No account value available, using default")
            return 100000  # Default value
        
    
    # def updateAccountValue(self, key, value, currency, accountName):
    #     if key == "NetLiquidation" and currency == "USD":
    #         try:
    #             self.account_value = float(value)
    #             self.last_update_time = time.time()
    #             self.account_update_event.set()
    #             print(f"Account value updated: ${self.account_value}")
    #         except ValueError:
    #             print(f"Invalid account value received: {value}")
                
    # def accountDownloadEnd(self, accountName):
    #     if self.account_value is None:
    #         print("Warning: No account value received during update")
    #     self.account_update_event.set()    
        
    # def get_account_value(self):
    #     current_time = time.time()
    #     if self.account_value is not None and (current_time - self.last_update_time) < 60:
    #         print(f"RiskManager: Using cached account value: ${self.account_value}")
    #         return self.account_value

    #     self.account_update_event.clear()
    #     self.account_value = None

    #     if not self.client.isConnected():
    #         print("Error: Client not connected to IBKR")
    #         return 100000

    #     try:
    #         self.client.reqAccountUpdates(True, "")
    #         print("RiskManager: Requested account updates")
    #     except Exception as e:
    #         print(f"Error requesting account updates: {e}")
    #         return 100000

    #     if self.account_update_event.wait(timeout=15):
    #         self.client.reqAccountUpdates(False, "")
    #         if self.account_value is not None:
    #             print(f"RiskManager: Returning account value: ${self.account_value}")
    #             return self.account_value
    #         else:
    #             print("Warning: No account value received")
    #             return 100000
    #     else:
    #         print("Warning: Timeout waiting for account value")
    #         return 100000   

    def validate_order(self, symbol, action, quantity, current_price):
        """
        Validates if an order meets risk management criteria
        Returns: (bool, str) - (is_valid, reason_if_invalid)
        """
        # Calculate position value
        position_value = quantity * current_price
        
        # Check if it exceeds max position size
        if position_value > self.max_position_size:
            position_value = self.max_position_size
            quantity = position_value / current_price
            
            
        # Check existing position for symbol
        current_pos = self.current_positions.get(symbol, 0)
        if action == "BUY":
            new_position = current_pos + quantity
        else:
            new_position = current_pos - quantity
            
        # Validate against daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit reached"
            
        return True, ""

    # def calculate_position_size(self, symbol, price, volatility):
    #     """
    # Calculates appropriate position size based on volatility and risk parameters
    # """
    #     try:
    #         account_value = self.get_account_value()
    #         risk_amount = account_value * self.position_sizing_pct
    #         print(risk_amount)
        
    #     # Add safety checks for price and volatility
    #         if price is None or volatility is None or price <= 0 or volatility <= 0:
    #             print(f"Warning: Invalid price ({price}) or volatility ({volatility})")
    #             return 0
            
    #         position_size = int(risk_amount * (1-volatility) / price)
    #         return max(0, position_size)  # Ensure we don't return negative positions
        
    #     except Exception as e:
    #         print(f"Error calculating position size: {e}")
    #         return 0  # Return 0 as a safe fallback
    def calculate_position_size(self,
                                   symbol,
                                   price=None,
                                   confidence_level=0.95,
                                   holding_period_days=1):
        
        # Step 1: Basic setup and validation
        account_value = self.get_account_value()
        risk_budget = account_value * self.position_sizing_pct
        
        if price is None:
            price = get_current_price(symbol)
            if price is None:
                print(f"Could not fetch current price for {symbol}")
                return 0
        
        print(f"=== Position Sizing Analysis for {symbol} ===")
        print(f"Account Value: ${account_value:,.2f}")
        print(f"Risk Budget: ${risk_budget:,.2f} ({self.position_sizing_pct*100}%)")
        print(f"Current Price: ${price:.2f}")
        
        # Step 2: Get data at multiple frequencies for comprehensive analysis
        var_estimates = {}
        
        # Method 1: Daily returns (most appropriate for daily VaR)
        try:
            daily_var = self._calculate_daily_var(symbol, confidence_level, holding_period_days)
            var_estimates['daily_data'] = daily_var
            print(f"Daily VaR (using daily returns): {daily_var*100:.3f}%")
        except Exception as e:
            print(f"Could not calculate daily VaR: {e}")
        
        # Method 2: Intraday returns with proper scaling
        try:
            intraday_var = self._calculate_intraday_var(symbol, confidence_level, holding_period_days)
            var_estimates['intraday_data'] = intraday_var
            print(f"Intraday VaR (using 1-min returns): {intraday_var*100:.3f}%")
        except Exception as e:
            print(f"Could not calculate intraday VaR: {e}")
        
        # Method 3: Hybrid approach using realized volatility
        try:
            realized_vol_var = self._calculate_realized_volatility_var(symbol, confidence_level, holding_period_days)
            var_estimates['realized_vol'] = realized_vol_var
            print(f"Realized Volatility VaR: {realized_vol_var*100:.3f}%")
        except Exception as e:
            print(f"Could not calculate realized volatility VaR: {e}")
        
        # Step 3: Choose the most conservative (largest) VaR estimate
        # This provides a safety margin and acknowledges model uncertainty
        if var_estimates:
            chosen_var = max(var_estimates.values())
            chosen_method = [k for k, v in var_estimates.items() if v == chosen_var][0]
            print(f"Selected VaR: {chosen_var*100:.3f}% (from {chosen_method})")
        else:
            print("No VaR estimates available, using conservative default")
            chosen_var = 0.02  # 2% default conservative estimate
        
        # Step 4: Apply minimum VaR threshold to prevent unrealistic position sizes
        min_var_threshold = 0.008  # 0.8% minimum - more realistic than 0.5%
        effective_var = max(chosen_var, min_var_threshold)
        
        if effective_var > chosen_var:
            print(f"Applied minimum VaR threshold: {effective_var*100:.3f}%")
        
        # Step 5: Calculate position size with multiple constraints
        var_based_shares = int(risk_budget / (price * effective_var))
        max_position_pct = 0.12  # 12% maximum position size
        max_affordable_shares = int((account_value * max_position_pct) / price)
        simple_budget_shares = int(risk_budget / price)
        
        # Take the minimum of all constraints
        final_shares = min(var_based_shares, max_affordable_shares, simple_budget_shares)
        
        print(f"Position size analysis:")
        print(f"  VaR-based max: {var_based_shares} shares")
        print(f"  Affordability max: {max_affordable_shares} shares")
        print(f"  Simple budget max: {simple_budget_shares} shares")
        print(f"  Final position: {final_shares} shares")
        print(f"  Position value: ${final_shares * price:,.2f}")
        print(f"  Portfolio allocation: {(final_shares * price / account_value)*100:.2f}%")
        
        return max(final_shares, 0)

    def _calculate_daily_var(self, symbol, confidence_level, holding_period_days):
        """
        Calculate VaR using daily returns - most appropriate for daily risk estimates.
        This avoids the scaling issues that come with high-frequency data.
        """
        # Get daily data for the past year
        df = get_recent_bars(
            symbol, 
            '1 day',  # Use daily bars for daily VaR
            '1 Y',  # One trading year
            for_calculation=True,
            client_instance=self.client,
            exchange_cache=getattr(self, 'exchange_cache', None)
        )
        
        if len(df) < 30:  # Need at least 30 daily observations
            raise ValueError("Insufficient daily data for VaR calculation")
        
        # Calculate daily returns
        returns = df['close'].pct_change().dropna()
        
        # Calculate VaR directly from daily returns (no scaling needed)
        sorted_returns = np.sort(returns.values)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        daily_var = abs(sorted_returns[var_index])
        
        # Scale for holding period if different from 1 day
        if holding_period_days != 1:
            daily_var = daily_var * np.sqrt(holding_period_days)
        
        return daily_var

    def _calculate_intraday_var(self, symbol, confidence_level, holding_period_days):
        """
        Calculate VaR using intraday returns with more sophisticated scaling.
        This approach acknowledges the limitations of simple square-root scaling.
        """
        # Get several days of 1-minute data
        df = get_recent_bars(
            symbol,
            '1 min',
            '5 D',  # Get 5 days of minute data
            for_calculation=True,
            client_instance=self.client,
            exchange_cache=getattr(self, 'exchange_cache', None)
        )
        
        if len(df) < 100:
            raise ValueError("Insufficient intraday data for VaR calculation")
        
        # Group by trading day and calculate daily returns from intraday data
        df['date'] = df['date'].dt.date
        daily_rets = df.groupby('date')['close'].agg(['first', 'last'])
        daily_returns = (daily_rets['last'] / daily_rets['first'] - 1).dropna()
        
        if len(daily_returns) < 3:
            raise ValueError("Need at least 3 days of intraday data")
        
        # Calculate VaR from these daily returns (derived from intraday data)
        sorted_returns = np.sort(daily_returns.values)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        daily_var = abs(sorted_returns[var_index])
        
        # Apply holding period scaling
        return daily_var * np.sqrt(holding_period_days)

    def _calculate_realized_volatility_var(self, symbol, confidence_level, holding_period_days):
        """
        Calculate VaR using realized volatility approach.
        This method uses intraday data to estimate daily volatility more accurately.
        """
        # Get intraday data
        df = get_recent_bars(
            symbol,
            '5 mins',  # Use 5-minute bars for better balance
            '10 D',
            for_calculation=True,
            client_instance=self.client,
            exchange_cache=getattr(self, 'exchange_cache', None)
        )
        
        if len(df) < 50:
            raise ValueError("Insufficient data for realized volatility calculation")
        
        # Calculate 5-minute returns
        df['return'] = df['close'].pct_change()
        
        # Group by day and calculate realized volatility for each day
        df['date'] = df['date'].dt.date
        daily_realized_vol = df.groupby('date')['return'].apply(
            lambda x: np.sqrt(np.sum(x**2) * 78)  # 78 = 390 minutes / 5 minute bars
        ).dropna()
        
        if len(daily_realized_vol) < 3:
            raise ValueError("Insufficient realized volatility observations")
        
        # Use the 95th percentile of realized volatility as VaR estimate
        var_percentile = np.percentile(daily_realized_vol, confidence_level * 100)
        
        # Scale for holding period
        return var_percentile * np.sqrt(holding_period_days)

class FadeStrategy:
    """
    Implements a fade/mean reversion strategy that trades against extreme moves
    """
    
    def __init__(self, risk_manager, client, exchange_cache):
        self.risk_manager = risk_manager
        self.client = client
        self.exchange_cache = exchange_cache
        
        # Fade strategy parameters
        self.extreme_multiplier = 2  # How many times normal volatility to trigger fade
        self.fade_duration_minutes = 15  # Hold fade positions for max 15 minutes
        self.max_fade_positions = 3  # Limit number of concurrent fade positions
        
        # Track fade positions
        self.fade_positions = {}  # symbol -> {'entry_time', 'entry_price', 'quantity', 'direction'}
        
    def should_fade(self, symbol, current_price, noise_boundaries, current_time):
        """
        Determine if we should fade an extreme move
        """
        upper_bound, lower_bound = noise_boundaries
        
        # Calculate extreme boundaries (2.5x normal volatility)
        volatility_range = upper_bound - lower_bound
        extreme_upper = upper_bound + (volatility_range * (self.extreme_multiplier - 1) / 2)
        extreme_lower = lower_bound - (volatility_range * (self.extreme_multiplier - 1) / 2)
        
        # Check for extreme moves
        extreme_high = current_price > extreme_upper
        extreme_low = current_price < extreme_lower
        
        # Don't fade if we already have max positions
        if len(self.fade_positions) >= self.max_fade_positions:
            return None, None
            
        # Don't fade the same symbol if we already have a position
        if symbol in self.fade_positions:
            return None, None
            
        if extreme_high:
            return 'FADE_SHORT', extreme_upper
        elif extreme_low:
            return 'FADE_LONG', extreme_lower
            
        return None, None
    
    def check_fade_exits(self, symbol, current_price, current_time):
        """
        Check if we should exit existing fade positions
        """
        if symbol not in self.fade_positions:
            return None
            
        position = self.fade_positions[symbol]
        entry_time = position['entry_time']
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Time-based exit (15 minutes max hold)
        time_elapsed = (current_time - entry_time).total_seconds() / 60
        if time_elapsed >= self.fade_duration_minutes:
            return 'TIME_EXIT'
        
        # Profit target exit (price reverts back toward entry level)
        if direction == 'SHORT':
            # Exit short fade if price drops significantly below entry
            if current_price < entry_price * 0.995:  # 0.5% profit target
                return 'PROFIT_EXIT'
        elif direction == 'LONG':
            # Exit long fade if price rises significantly above entry
            if current_price > entry_price * 1.005:  # 0.5% profit target
                return 'PROFIT_EXIT'
        
        # Stop loss exit (move continues against us)
        if direction == 'SHORT':
            # Stop loss if price continues higher
            if current_price > entry_price * 1.01:  # 1% stop loss
                return 'STOP_EXIT'
        elif direction == 'LONG':
            # Stop loss if price continues lower
            if current_price < entry_price * 0.99:  # 1% stop loss
                return 'STOP_EXIT'
        
        return None
    
    def execute_fade_trade(self, symbol, signal, current_price, current_time):
        """
        Execute fade trade with small position size
        """
        try:
            # Calculate small position size for fade (much smaller than momentum trades)
            account_value = self.risk_manager.get_account_value()
            fade_position_value = account_value * 0.02  # Only 2% of account per fade
            quantity = int(fade_position_value / current_price)
            
            if quantity == 0:
                return False
            
            # Create order
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.currency = "USD"
            contract.exchange = "SMART"
            
            if self.exchange_cache:
                primary, _ = self.exchange_cache.get_exchange(symbol)
                contract.primaryExchange = primary
            
            order = Order()
            order.orderType = "MKT"
            order.totalQuantity = quantity
            order.eTradeOnly = False
            order.firmQuoteOnly = False
            
            if signal == 'FADE_SHORT':
                order.action = "SELL"
                direction = 'SHORT'
            elif signal == 'FADE_LONG':
                order.action = "BUY"
                direction = 'LONG'
            else:
                return False
            
            # Place order
            if hasattr(self.client, 'order_id'):
                self.client.order_id += 1
                self.client.placeOrder(self.client.order_id, contract, order)
                
                # Track the fade position
                self.fade_positions[symbol] = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'direction': direction
                }
                
                print(f"FADE STRATEGY: {signal} - {quantity} shares of {symbol} at ${current_price:.2f}")
                return True
                
        except Exception as e:
            print(f"Error executing fade trade: {e}")
            return False
    
    def execute_fade_exit(self, symbol, exit_reason, current_price):
        """
        Execute fade exit trade
        """
        if symbol not in self.fade_positions:
            return False
            
        try:
            position = self.fade_positions[symbol]
            quantity = position['quantity']
            direction = position['direction']
            
            # Create exit order (opposite direction)
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.currency = "USD"
            contract.exchange = "SMART"
            
            if self.exchange_cache:
                primary, _ = self.exchange_cache.get_exchange(symbol)
                contract.primaryExchange = primary
            
            order = Order()
            order.orderType = "MKT"
            order.totalQuantity = quantity
            order.eTradeOnly = False
            order.firmQuoteOnly = False
            
            # Exit order is opposite of entry
            if direction == 'SHORT':
                order.action = "BUY"  # Cover short
            else:
                order.action = "SELL"  # Sell long
            
            # Place exit order
            if hasattr(self.client, 'order_id'):
                self.client.order_id += 1
                self.client.placeOrder(self.client.order_id, contract, order)
                
                # Calculate P&L
                entry_price = position['entry_price']
                if direction == 'SHORT':
                    pnl = (entry_price - current_price) * quantity
                else:
                    pnl = (current_price - entry_price) * quantity
                
                print(f"FADE EXIT ({exit_reason}): {symbol} - P&L: ${pnl:.2f}")
                
                # Remove from tracking
                del self.fade_positions[symbol]
                return True
                
        except Exception as e:
            print(f"Error executing fade exit: {e}")
            return False
        
class EnhancedTradingStrategy:
    """
    Advanced trading strategy that implements EGARCH volatility modeling
    and intraday mean reversion based on open-to-close regression analysis.
    """
    
    def __init__(self, risk_manager, client):
        self.risk_manager = risk_manager
        self.client = client
        
        # Strategy parameters from backtest
        self.lag = 14  # Days for volatility calculation
        self.target_vol = 0.2  # 20% annualized target volatility
        self.p_terms = 1  # GARCH p terms
        self.o_terms = 0  # GARCH o terms  
        self.q_terms = 1  # GARCH q terms
        self.trade_freq = 15  # Check signals every 30 minutes
        self.max_leverage = 3  # Maximum leverage allowed
        self.portfolio_allocation = 1
        self.historical_vix_cache = {}  # ADD THIS LINE
        self.vix_cache_loaded = False
        
         # NEW: First 30 minutes strategy parameters
        self.orb_period = 5  # Minutes to establish opening range
        self.gap_threshold = 0.0125  # 1.25% gap threshold for strong gaps
        self.volume_multiplier = 1.5  # Volume should be 1.5x average
        self.early_stop_pct = 0.0075  # 0.75% stop loss for early trades
        self.orb_extension = 0.005  # 0.5% extension beyond range for entry
        
        # Data storage for each symbol
        self.symbol_data = defaultdict(lambda: {
            'daily_data': pd.DataFrame(),
            'intraday_data': pd.DataFrame(),
            'egarch_vol': None,
            'regression_params': {'intercept': None, 'coef': None},
            'prev_month_vol': None,
            'current_position': 0,
            'last_signal_time': None,
            'trade_times': [],
            'opening_range': {'high': None, 'low': None, 'established': False},
            'gap_size': None,
            'pre_market_vol': None,
            'early_trade_active': False,
            'early_entry_price': None})
        
        # Allocations (will be updated monthly based on inverse volatility)
        self.allocations = {}
        self.last_allocation_update = None
        # Add Random Forest components
        self.rf_model = None
        self.rf_scaler = StandardScaler()
        self.rf_training_data = []
        self.rf_min_training_samples = 30  # Need enough data before using RF
        self.use_rf_signals = False  # Start with traditional strategy
        
        # Try to load existing model if available
        self.load_rf_model()
        
    def load_historical_vix_cache(self, lookback_days=365):
        """
        Load historical VIX data into cache for fast lookups.
        Call this once at startup or before training RF model.
        """
        try:
            # Try loading from file cache first
            cache_file = 'historical_vix_cache.pkl'
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        # Check if cache is recent enough (within 7 days)
                        cache_age = (datetime.datetime.now().date() - max(cached_data.keys())).days
                        if cache_age <= 7 and len(cached_data) >= lookback_days * 0.9:
                            self.historical_vix_cache = cached_data
                            self.vix_cache_loaded = True
                            print(f"Loaded VIX cache from file: {len(cached_data)} days")
                            print(f"Date range: {min(cached_data.keys())} to {max(cached_data.keys())}")
                            return True
                        else:
                            print(f"VIX cache outdated (age: {cache_age} days) or insufficient, reloading...")
                except Exception as e:
                    print(f"Error loading VIX cache file: {e}, will download fresh data")
            
            print(f"Downloading {lookback_days} days of historical VIX data...")
            
            # Create specific VIX contract (not using get_recent_bars wrapper)
            contract = Contract()
            contract.symbol = "VIX"
            contract.secType = "IND"  # Index, not stock
            contract.exchange = "CBOE"
            contract.currency = "USD"
            
            # Use a dedicated request for VIX
            vix_data = []
            req_id = int(time.time() * 1000) % 10000 + random.randint(0, 5000)
            
            # Create temporary queue for this request
            vix_queue = queue.Queue()
            
            # Store original callback
            original_callback = self.client.historicalData
            
            def vix_callback(reqId, bar):
                if reqId != req_id:
                    return
                    
                try:
                    # VIX data often comes as YYYYMMDD format for daily bars
                    date_str = str(bar.date).strip()
                    
                    # Try parsing as simple date first (most common for daily VIX)
                    if len(date_str) == 8 and date_str.isdigit():
                        t = datetime.datetime.strptime(date_str, '%Y%m%d')
                    elif ' ' in date_str:
                        # Has time component
                        if '-' in date_str:
                            t = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        else:
                            t = datetime.datetime.strptime(date_str, '%Y%m%d %H:%M:%S')
                    else:
                        # Fallback: try as timestamp
                        timestamp = float(bar.date)
                        if timestamp > 946684800:  # After year 2000
                            t = datetime.datetime.fromtimestamp(timestamp)
                        else:
                            print(f"Warning: Unusual VIX timestamp {timestamp}")
                            return
                    
                    data = {
                        'date': t,
                        'close': bar.close
                    }
                    vix_queue.put(data)
                    
                except Exception as e:
                    print(f"Error parsing VIX bar: {e}, bar.date={bar.date}")
            
            try:
                # Set temporary callback
                self.client.historicalData = vix_callback
                
                # Request VIX data
                self.client.reqHistoricalData(
                    req_id, 
                    contract, 
                    '',  # End date (empty = now)
                    f'{lookback_days} D',
                    '1 day',
                    'TRADES',
                    0,  # Use RTH=0 for indices
                    1,
                    False,
                    []
                )
                
                # Collect data
                timeout = time.time() + 30
                while time.time() < timeout:
                    try:
                        data = vix_queue.get(timeout=0.5)
                        vix_data.append(data)
                    except queue.Empty:
                        if len(vix_data) > 0 and time.time() - timeout > 2:
                            break
                
            finally:
                # Restore original callback
                self.client.historicalData = original_callback
                self.client.cancelHistoricalData(req_id)
            
            if not vix_data:
                print("Warning: No VIX data collected")
                self.vix_cache_loaded = False
                return False
            
            # Create cache dictionary
            for data in vix_data:
                date = pd.to_datetime(data['date']).date()
                self.historical_vix_cache[date] = data['close']
            
            # Save to file cache
            try:
                with open('historical_vix_cache.pkl', 'wb') as f:
                    pickle.dump(self.historical_vix_cache, f)
                print(f"Saved VIX cache to file")
            except Exception as e:
                print(f"Warning: Could not save VIX cache to file: {e}")
            
            print(f"Loaded VIX data for {len(self.historical_vix_cache)} days")
            if self.historical_vix_cache:
                print(f"VIX date range: {min(self.historical_vix_cache.keys())} to {max(self.historical_vix_cache.keys())}")
            self.vix_cache_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading historical VIX cache: {e}")
            import traceback
            traceback.print_exc()
            self.vix_cache_loaded = False
            return False
        
    def get_historical_vix(self, date, default=15.0):
        """
        Get VIX value for a specific historical date.
        
        Args:
            date: datetime.date, datetime.datetime, pd.Timestamp, or string
            default: Default VIX value if not found (default: 15.0)
            
        Returns:
            VIX close value for that date, or default if not available
        """
        try:
            # Convert various date formats to date object
            if isinstance(date, str):
                date = pd.to_datetime(date).date()
            elif isinstance(date, (datetime.datetime, pd.Timestamp)):
                date = date.date()
            elif isinstance(date, datetime.date):
                pass  # Already correct format
            else:
                print(f"Warning: Unexpected date type {type(date)}, using default VIX")
                return default
            
            # Load cache if not already loaded
            if not self.vix_cache_loaded:
                print("VIX cache not loaded, loading now...")
                self.load_historical_vix_cache(lookback_days=500)  # Load enough history
            
            # Look up VIX value for this date
            if date in self.historical_vix_cache:
                return self.historical_vix_cache[date]
            
            # If exact date not found, try to find nearest trading day
            # This handles weekends and holidays
            nearest_vix = self._find_nearest_vix(date, max_days=7)
            if nearest_vix is not None:
                return nearest_vix
            
            print(f"Warning: No VIX data for {date}, using default {default}")
            return default
            
        except Exception as e:
            print(f"Error getting historical VIX for {date}: {e}")
            return default
    
    def _find_nearest_vix(self, target_date, max_days=7):
        """
        Find VIX value from nearest trading day within max_days.
        Searches backwards first (prefer previous trading day).
        """
        try:
            # Search backwards first (previous trading days)
            for days_back in range(1, max_days + 1):
                check_date = target_date - datetime.timedelta(days=days_back)
                if check_date in self.historical_vix_cache:
                    return self.historical_vix_cache[check_date]
            
            # If not found backwards, search forwards
            for days_forward in range(1, max_days + 1):
                check_date = target_date + datetime.timedelta(days=days_forward)
                if check_date in self.historical_vix_cache:
                    return self.historical_vix_cache[check_date]
            
            return None
            
        except Exception as e:
            print(f"Error finding nearest VIX: {e}")
            return None
        
    def get_current_vix(self):
        """Get current VIX level from IBKR"""
        try:
            # Create VIX contract
            contract = Contract()
            contract.symbol = "VIX"
            contract.conId = 13455763
            contract.secType = "IND"  # Index
            contract.exchange = "CBOE"
            contract.currency = "USD"
            
            # Use a temporary wrapper to get VIX data
            wrapper = PriceWrapper()
            wrapper.connect('127.0.0.1', TRADING_PORT, 124)
            thread = Thread(target=wrapper.run, daemon=True)
            thread.start()
            
            wrapper.reqMarketDataType(1)
            wrapper.reqMktData(2, contract, '', False, False, [])
            
            # Wait for price
            timeout = time.time() + 5
            while not wrapper.price_ready and time.time() < timeout:
                time.sleep(0.1)
            
            wrapper.disconnect()
            
            if wrapper.bid and wrapper.ask:
                vix_value = (wrapper.bid + wrapper.ask) / 2
                return vix_value
            else:
                # Fallback: try to get historical data
                df = get_recent_bars("VIX", '1 day', '1 D', 
                                    client_instance=self.client,
                                    exchange_cache=self.risk_manager.exchange_cache)
                if not df.empty:
                    return df['close'].iloc[-1]
                return 15.0  # Default if can't get VIX
                
        except Exception as e:
            print(f"Error getting VIX: {e}")
            return 15.0  # Default VIX level
    def collect_historical_data_incrementally(self, symbol, total_days_needed=250, chunk_size_days=10):
        """
        Collect historical data in chunks, ensuring we get exactly the trading days we need
        """
        all_data = pd.DataFrame()
        trading_days_collected = 0
        
        # Start from now and work backwards
        current_end_date = datetime.datetime.now()
        
        while trading_days_collected < total_days_needed:
            try:
                # Format the end date for IBKR API
                end_date_str = current_end_date.strftime('%Y%m%d %H:%M:%S')
                
                print(f"Fetching {symbol}: requesting {chunk_size_days}D ending {end_date_str}")
                
                # Use your modified get_recent_bars with end_datetime
                df_chunk = get_recent_bars(
                    symbol,
                    '1 min',
                    f'{chunk_size_days} D',
                    for_calculation=True,
                    client_instance=self.client,
                    exchange_cache=self.risk_manager.exchange_cache,
                    end_datetime=end_date_str  # Pass the specific end date
                )
                
                if not df_chunk.empty:
                    # Convert dates to datetime if needed
                    df_chunk['date'] = pd.to_datetime(df_chunk['date'])
                    
                    # Remove any weekend data
                    df_chunk = df_chunk[df_chunk['date'].dt.weekday < 5]
                    
                    # Count unique trading days in this chunk
                    unique_trading_days = df_chunk['date'].dt.date.nunique()
                    trading_days_collected += unique_trading_days
                    
                    # Append to our collection (prepend since we're going backwards)
                    if all_data.empty:
                        all_data = df_chunk
                    else:
                        # Remove any overlapping data
                        min_date_new = df_chunk['date'].min()
                        all_data = all_data[all_data['date'] > min_date_new]
                        all_data = pd.concat([df_chunk, all_data], ignore_index=True)
                    
                    print(f"  Got {unique_trading_days} trading days, total: {trading_days_collected}/{total_days_needed}")
                    print(f"  Date range: {df_chunk['date'].min()} to {df_chunk['date'].max()}")
                    
                    # Move the end date back for the next chunk
                    # Use the earliest date from this chunk minus 1 day
                    earliest_date = df_chunk['date'].min()
                    current_end_date = earliest_date - datetime.timedelta(days=1)
                    
                    # Skip weekends
                    while current_end_date.weekday() > 4:
                        current_end_date -= datetime.timedelta(days=1)
                    
                else:
                    print(f"  No data received, adjusting date and retrying...")
                    # Move back further and try again
                    current_end_date -= datetime.timedelta(days=chunk_size_days * 2)
                    
                # Rate limiting
                time.sleep(15)  # Wait 15 seconds between requests
                
            except Exception as e:
                print(f"Error fetching chunk: {e}")
                time.sleep(30)
                # Try moving back further
                current_end_date -= datetime.timedelta(days=chunk_size_days)
        
        # Final cleanup
        all_data = all_data.drop_duplicates(subset=['date'], keep='last')
        all_data = all_data.sort_values('date').reset_index(drop=True)
        
        # Final validation
        final_trading_days = all_data['date'].dt.date.nunique()
        print(f"Completed {symbol}: {len(all_data)} bars over {final_trading_days} trading days")
        print(f"Date range: {all_data['date'].min()} to {all_data['date'].max()}")
        
        return all_data
                
        # Remove any duplicate bars
        # all_data = all_data.drop_duplicates(subset=['date'], keep='last')
        # all_data = all_data.sort_values('date').reset_index(drop=True)
    def initialize_rf_with_full_history(self, symbols, total_days=250, force_retrain=False):
        """
        Initialize RF with full historical data collected incrementally
        This will take time but gets you the full dataset
        
        Args:
            symbols: List of symbols to train on
            total_days: Number of days of historical data to use
            force_retrain: If True, ignore cache and retrain from scratch
        """
        # Check for cached training samples
        if os.path.exists('rf_training_samples.pkl') and not force_retrain:
            try:
                with open('rf_training_samples.pkl', 'rb') as f:
                    cached = pickle.load(f)
                    training_samples = cached.get('samples', [])
                    cache_info = cached.get('info', {})
                
                print(f"\n{'='*60}")
                print(f"Found cached RF training data!")
                print(f"{'='*60}")
                print(f"  Training samples: {len(training_samples)}")
                print(f"  Symbols: {cache_info.get('symbols', 'Unknown')}")
                print(f"  Created: {cache_info.get('created', 'Unknown')}")
                print(f"  Days: {cache_info.get('total_days', 'Unknown')}")
                print(f"{'='*60}")
                
                # Automatically use cache (no prompt)
                self.rf_training_data = training_samples
                self.train_rf_model()
                print("\n✓ RF model loaded from cache successfully!")
                return
            except Exception as e:
                print(f"Error loading cached training samples: {e}, will retrain...")
        print(f"Starting full historical data collection for {len(symbols)} symbols")
        print(f"This will take approximately {len(symbols) * 25} minutes...")
        
        print("\nStep 1: Loading historical VIX data for feature extraction...")
        vix_loaded = self.load_historical_vix_cache(lookback_days=total_days + 50)
        if not vix_loaded:
            print("Warning: VIX cache failed to load, will use default VIX values")
        
        print(f"\nStep 2: Processing {len(symbols)} symbols...")
        print(f"This will take approximately {len(symbols) * 25} minutes...")
        
        training_samples = []
        
        for i, symbol in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}")
            
            cache_file = f'historical_data_{symbol}.pkl'
            if os.path.exists(cache_file):
                try:
                    print(f"  ✓ Loading cached historical data for {symbol}...")
                    historical_df = pd.read_pickle(cache_file)
                    
                    # Verify it has enough data
                    trading_days = historical_df['date'].dt.date.nunique()
                    if trading_days >= total_days * 0.9:  # At least 90% of requested days
                        print(f"  ✓ Using cached data ({trading_days} trading days)")
                    else:
                        print(f"  ⚠ Cached data insufficient ({trading_days} days), re-downloading...")
                        historical_df = self.collect_historical_data_incrementally(
                            symbol, total_days_needed=total_days, chunk_size_days=10
                        )
                        historical_df.to_pickle(cache_file)
                except Exception as e:
                    print(f"  ⚠ Error loading cache: {e}, re-downloading...")
                    historical_df = self.collect_historical_data_incrementally(
                        symbol, total_days_needed=total_days, chunk_size_days=10
                    )
                    historical_df.to_pickle(cache_file)
            else:
                # No cache, collect fresh data
                print(f"  ⏳ No cache found, downloading historical data...")
                historical_df = self.collect_historical_data_incrementally(
                    symbol, total_days_needed=total_days, chunk_size_days=10
                )
                historical_df.to_pickle(cache_file)
                print(f"  ✓ Saved {symbol} data to cache")
            
            if not historical_df.empty:
                # Save to disk as backup
                historical_df.to_pickle(f'historical_data_{symbol}.pkl')
                
                # Extract trades
                simulated_trades = self.simulate_historical_trades(symbol, historical_df)
                
                for trade in simulated_trades:
                    features = self.extract_historical_features(trade)
                    if features:
                        # Long-only strategy: BUY=1 if profitable, else 0
                        # SELL signals are treated as "don't trade" (label=0)
                        if trade['signal'] == 'BUY':
                            label = 1 if trade['profit_pct'] > 0.002 else 0
                        else:  # SELL signals become negative examples
                            label = 0  # Don't trade

                        training_samples.append({
                            'features': features,
                            'label': label,
                            'profit_pct': trade['profit_pct'],
                            'signal': trade['signal']  # Keep for diagnostics
                        })
                
                print(f"  Extracted {len(simulated_trades)} trades from {symbol}")
            
            # Longer delay between symbols
            if i < len(symbols) - 1:
                print(f"  Waiting 60 seconds before next symbol...")
                time.sleep(60)
        
        # Save training samples with metadata
        cache_data = {
            'samples': training_samples,
            'info': {
                'symbols': symbols,
                'total_days': total_days,
                'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_samples': len(training_samples)
            }
        }
        
        with open('rf_training_samples.pkl', 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"\n{'='*60}")
        print(f"Training samples saved to cache!")
        print(f"  Total samples: {len(training_samples)}")
        print(f"  Symbols: {symbols}")
        print(f"  Cache file: rf_training_samples.pkl")
        print(f"{'='*60}\n")
        
        if len(training_samples) >= 50:
            self.rf_training_data = training_samples
            self.train_rf_model()
    
    def simulate_historical_trades(self, symbol, historical_df):
        """Extract trades that would have occurred historically using actual strategy logic"""
        trades = []
        debug_sell_count = 0  # Track SELL trades for debugging

        # Add day column if not present
        historical_df['day'] = pd.to_datetime(historical_df['date']).dt.date
        
        # Get all unique days
        unique_days = sorted(historical_df['day'].unique())
        
        # Need at least 14 days history to calculate boundaries
        if len(unique_days) < 15:
            return trades
        
        # Start from day 15 (need 14 days history)
        for day_idx in range(14, len(unique_days)):
            current_day = unique_days[day_idx]
            day_data = historical_df[historical_df['day'] == current_day].reset_index(drop=True)
            
            # Skip if insufficient intraday data
            if len(day_data) < 100:
                continue
            
            # Get previous 14 days data for sigma calculation
            prev_days = unique_days[day_idx-14:day_idx]
            prev_data = historical_df[historical_df['day'].isin(prev_days)]
            
            # Calculate sigma for each time of day (following your actual method)
            sigma_by_time = {}
            for minutes in range(0, 390, 15):  # Every 15 minutes
                moves = []
                for prev_day in prev_days:
                    prev_day_data = prev_data[prev_data['day'] == prev_day]
                    if len(prev_day_data) > minutes:
                        open_price = prev_day_data.iloc[0]['open']
                        price_at_time = prev_day_data.iloc[min(minutes, len(prev_day_data)-1)]['close']
                        move = abs((price_at_time / open_price) - 1)
                        moves.append(move)
                
                if moves:
                    sigma_by_time[minutes] = np.mean(moves)
                else:
                    sigma_by_time[minutes] = 0.01  # Default 1%
            
            # Get today's open and yesterday's close
            open_price = day_data.iloc[0]['open']
            
            # Get previous day's close
            if day_idx > 0:
                prev_day = unique_days[day_idx - 1]
                prev_day_data = historical_df[historical_df['day'] == prev_day]
                if not prev_day_data.empty:
                    prev_close = prev_day_data.iloc[-1]['close']
                else:
                    prev_close = open_price
            else:
                prev_close = open_price
            
            position_opened = False
            
            # Check for signals every 30 minutes after first 30 minutes
            for i in range(30, min(len(day_data), 390), 15):
                if position_opened:
                    break
                
                current_bar = day_data.iloc[i]
                prior_bars = day_data.iloc[:i]
                
                # Calculate boundaries following the paper's formula (matches live trading)
                sigma = sigma_by_time.get(i, 0.01)
                upper_bound = max(open_price, prev_close) * (1 + sigma)
                lower_bound = min(open_price, prev_close) * (1 - sigma)
                
                current_price = current_bar['close']
                vwap = self.calculate_vwap_from_data(prior_bars)
                
                # Check for signal (matching live trading logic - no regression filter)
                signal = 'HOLD'

                # Calculate first-half hour return for regression features
                first_half_return = 0
                if i >= 30:
                    price_10am = day_data.iloc[30]['close'] if len(day_data) > 30 else open_price
                    first_half_return = (price_10am / open_price) - 1

                    # Generate signals based on price vs boundaries and VWAP
                    # This matches the live trading logic (lines 3659-3663)
                    if current_price > upper_bound and current_price > vwap:
                        signal = 'BUY'
                    elif current_price < lower_bound and current_price < vwap:
                        signal = 'SELL'
                        
                if signal == 'SELL' and len([t for t in trades if t['signal'] == 'SELL']) < 5:
                    print(f"\n{'='*70}")
                    print(f"SELL SIGNAL DEBUG - {symbol} on {current_day}")
                    print(f"{'='*70}")
                    print(f"Time: {current_bar['date']}, i={i} minutes from open")
                    print(f"\nPRICE LEVELS:")
                    print(f"  Current price:  ${current_price:.2f}")
                    print(f"  Open price:     ${open_price:.2f}")
                    print(f"  Prev close:     ${prev_close:.2f}")
                    print(f"  Lower bound:    ${lower_bound:.2f}")
                    print(f"  Upper bound:    ${upper_bound:.2f}")
                    print(f"  VWAP:           ${vwap:.2f}")

                    print(f"\nSIGNAL CONDITIONS (NO REGRESSION FILTER):")
                    print(f"  current_price < lower_bound: {current_price < lower_bound} ({current_price:.2f} < {lower_bound:.2f})")
                    print(f"  current_price < vwap:        {current_price < vwap} ({current_price:.2f} < {vwap:.2f})")

                    print(f"\nFEATURES THAT WILL BE RECORDED:")
                    print(f"  price_vs_open: {(current_price/open_price - 1)*100:+.3f}%")
                    print(f"  price_vs_vwap: {(current_price/vwap - 1)*100:+.3f}%")
                    print(f"  gap_size:      {(open_price/prev_close - 1)*100:+.3f}%")

                    print(f"\nEXPECTED VALUES FOR SHORT:")
                    print(f"  price_vs_open should be NEGATIVE (price below open)")
                    print(f"  price_vs_vwap should be NEGATIVE (price below VWAP)")

                    if (current_price/open_price - 1) > 0:
                        print(f"\n⚠️  WARNING: price_vs_open is POSITIVE!")
                        print(f"   This means price is ABOVE open, not below lower boundary!")

                    if (current_price/vwap - 1) > 0:
                        print(f"\n⚠️  WARNING: price_vs_vwap is POSITIVE!")
                        print(f"   This means price is ABOVE VWAP, contradicting signal condition!")

                    print(f"{'='*70}\n")
                        
                if signal != 'HOLD':
                    # Find exit using improved trailing stop logic (matching live trading)
                    remaining_bars = day_data.iloc[i:]
                    exit_idx = len(remaining_bars) - 1
                    exit_price = remaining_bars.iloc[-1]['close']
                    
                    entry_price = current_price
                    position_high = entry_price  # Track high for longs
                    position_low = entry_price   # Track low for shorts
                    
                    for j in range(1, len(remaining_bars)):
                        bar = remaining_bars.iloc[j]
                        current_bar_price = bar['close']
                        
                        # Recalculate boundaries for current time
                        current_minutes = min(i + j, 390)
                        current_sigma = sigma_by_time.get(current_minutes - (current_minutes % 15), sigma)
                        current_upper = max(open_price, prev_close) * (1 + current_sigma)
                        current_lower = min(open_price, prev_close) * (1 - current_sigma)
                        
                        # Calculate current VWAP (not used for exits anymore)
                        current_vwap = self.calculate_vwap_from_data(day_data.iloc[:i+j+1])
                        
                        if signal == 'BUY':
                            # Update high-water mark
                            if current_bar_price > position_high:
                                position_high = current_bar_price
                            
                            # Calculate P&L
                            pnl_pct = (current_bar_price / entry_price) - 1
                            
                            # Exit conditions (matching live trading logic):
                            # 1. Profit protection: Once up 1%, trail by 0.5% from high
                            if pnl_pct > 0.01:
                                trailing_price = position_high * (1 - 0.005)
                                if current_bar_price < trailing_price:
                                    exit_price = current_bar_price
                                    break
                            
                            # 2. Stop loss: -0.75%
                            elif pnl_pct < -0.0075:
                                exit_price = current_bar_price
                                break
                            
                            # 3. Dynamic smart exit: use HIGHEST support level below price
                            else:
                                support_levels = []
                                if current_upper < current_bar_price:
                                    support_levels.append(current_upper)
                                if current_vwap < current_bar_price:
                                    support_levels.append(current_vwap)
                                if open_price < current_bar_price:
                                    support_levels.append(open_price)
                                
                                if support_levels:
                                    exit_level = max(support_levels)  # Most conservative
                                    if current_bar_price < exit_level:
                                        exit_price = current_bar_price
                                        break
                            
                            # 4. End of day: exit at 3:50 PM (380 min from open)
                            if current_minutes >= 380:
                                exit_price = current_bar_price
                                break
                        
                        else:  # SELL (short)
                            # Update low-water mark
                            if current_bar_price < position_low:
                                position_low = current_bar_price
                            
                            # Calculate P&L for short
                            pnl_pct = (entry_price / current_bar_price) - 1
                            
                            # Exit conditions for shorts:
                            # 1. Profit protection
                            if pnl_pct > 0.01:
                                trailing_price = position_low * (1 + 0.005)
                                if current_bar_price > trailing_price:
                                    exit_price = current_bar_price
                                    break
                            
                            # 2. Stop loss
                            elif pnl_pct < -0.0075:
                                exit_price = current_bar_price
                                break
                            
                            # 3. Dynamic smart exit: use LOWEST resistance level above price
                            else:
                                resistance_levels = []
                                if current_lower > current_bar_price:
                                    resistance_levels.append(current_lower)
                                if current_vwap > current_bar_price:
                                    resistance_levels.append(current_vwap)
                                if open_price > current_bar_price:
                                    resistance_levels.append(open_price)
                                
                                if resistance_levels:
                                    exit_level = min(resistance_levels)  # Most conservative
                                    if current_bar_price > exit_level:
                                        exit_price = current_bar_price
                                        break
                            
                            # 4. End of day
                            if current_minutes >= 380:
                                exit_price = current_bar_price
                                break
                    
                    # Calculate profit (entry_price already set above)
                    if signal == 'BUY':
                        profit_pct = (exit_price / entry_price) - 1
                    else:
                        profit_pct = (entry_price / exit_price) - 1
                    
                    trades.append({
                        'symbol': symbol,
                        'date': current_day,
                        'entry_time': current_bar['date'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'signal': signal,
                        'profit_pct': profit_pct,
                        'outcome': 1 if profit_pct > 0.002 else -1,
                        'features_data': prior_bars,
                        'vwap_at_entry': vwap,
                        'open_price': open_price,
                        'prev_close': prev_close,
                        'first_half_return': first_half_return,  # Add regression feature
                        'debug_count': debug_sell_count if signal == 'SELL' else 999
                    })

                    if signal == 'SELL':
                        debug_sell_count += 1

                    position_opened = True
        
        return trades
    
    def extract_historical_features(self, trade):
        """Extract features from historical trade data"""
        features = {}

        try:
            features_data = trade['features_data']
            entry_price = trade['entry_price']
            signal = trade['signal']

            # Use values passed from simulate_historical_trades
            # These were calculated at signal generation time
            open_price = trade['open_price']
            vwap = trade['vwap_at_entry']
            prev_close = trade['prev_close']

            # Calculate price features
            features['price_vs_open'] = (entry_price / open_price) - 1
            features['price_vs_vwap'] = (entry_price / vwap) - 1
            features['gap_size'] = (open_price / prev_close) - 1

            # Volume features
            features['volume_ratio'] = features_data['volume'].sum() / features_data['volume'].mean()

            # Volatility
            returns = features_data['close'].pct_change().dropna()
            features['realized_vol'] = returns.std() if len(returns) > 0 else 0.02

            # Time features
            entry_time = trade['entry_time']
            features['minutes_from_open'] = (entry_time.hour - 9) * 60 + (entry_time.minute - 30)
            features['hour_of_day'] = entry_time.hour
            features['day_of_week'] = entry_time.weekday()

            # VIX
            trade_date = trade['date']
            features['vix_level'] = self.get_historical_vix(trade_date)
            
            # DEBUG: Check if VIX is actually loaded
            if not hasattr(self, '_vix_debug_printed'):
                self._vix_debug_printed = True
                print(f"\n{'='*60}")
                print(f"VIX CACHE DEBUG")
                print(f"{'='*60}")
                print(f"Cache loaded: {self.vix_cache_loaded}")
                print(f"Cache size: {len(self.historical_vix_cache)} days")
                if self.historical_vix_cache:
                    print(f"Date range: {min(self.historical_vix_cache.keys())} to {max(self.historical_vix_cache.keys())}")
                    print(f"Sample VIX values:")
                    for d in list(self.historical_vix_cache.keys())[:5]:
                        print(f"  {d}: {self.historical_vix_cache[d]:.2f}")
                print(f"VIX for {trade_date}: {features['vix_level']:.2f}")
                print(f"{'='*60}\n")

            # Simple momentum
            if len(features_data) >= 20:
                features['sma_20_vs_price'] = (entry_price / features_data['close'].tail(20).mean()) - 1
            else:
                features['sma_20_vs_price'] = 0

            # Regression features (first-half hour return)
            features['first_half_return'] = trade.get('first_half_return', 0)

            # DEBUG OUTPUT FOR FIRST FEW SELL TRADES
            if signal == 'SELL' and trade.get('debug_count', 999) < 3:
                print(f"\n{'='*70}")
                print(f"FEATURE EXTRACTION DEBUG - {trade['symbol']} on {trade['date']}")
                print(f"{'='*70}")
                print(f"Raw values:")
                print(f"  entry_price: ${entry_price:.2f}")
                print(f"  open_price:  ${open_price:.2f}")
                print(f"  vwap:        ${vwap:.2f}")
                print(f"  prev_close:  ${prev_close:.2f}")

                print(f"\nCalculated features:")
                print(f"  price_vs_open: {features['price_vs_open']*100:+.3f}%")
                print(f"  price_vs_vwap: {features['price_vs_vwap']*100:+.3f}%")
                print(f"  gap_size:      {features['gap_size']*100:+.3f}%")

                print(f"\nExpected for SELL:")
                print(f"  price_vs_open should be NEGATIVE (price below open)")
                print(f"  price_vs_vwap should be NEGATIVE (price below VWAP)")

                # Check if values make sense
                if entry_price > open_price:
                    print(f"\n⚠️  BUG: entry_price (${entry_price:.2f}) > open_price (${open_price:.2f})")
                    print(f"   This means price is ABOVE open, not below!")
                if entry_price > vwap:
                    print(f"⚠️  BUG: entry_price (${entry_price:.2f}) > vwap (${vwap:.2f})")
                    print(f"   This means price is ABOVE VWAP, contradicting SELL signal!")

                print(f"{'='*70}\n")

            return features

        except Exception as e:
            print(f"Error extracting historical features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_rf_features(self, symbol):
        """Extract features for Random Forest prediction (LIVE)"""
        data = self.symbol_data[symbol]
        features = {}
        
        try:
            current_time = datetime.datetime.now()
            minutes_from_open = (current_time.hour - 9) * 60 + (current_time.minute - 30)
            
            # Price features
            intraday_df = data.get('intraday_data', pd.DataFrame())
            if not intraday_df.empty:
                today_data = intraday_df[intraday_df['day'] == current_time.date()]
                if not today_data.empty:
                    open_price = today_data.iloc[0]['open']
                    current_price = today_data.iloc[-1]['close']
                    current_vwap = today_data.iloc[-1]['vwap']
                    
                    features['price_vs_open'] = (current_price / open_price) - 1
                    features['price_vs_vwap'] = (current_price / current_vwap) - 1
            
            # Volatility features
            features['realized_vol'] = data.get('egarch_vol', 0.02)
            
            # Gap and time features
            features['gap_size'] = data.get('gap_size', 0)
            features['minutes_from_open'] = minutes_from_open
            features['hour_of_day'] = current_time.hour
            features['day_of_week'] = current_time.weekday()
            
            # Market regime
            features['vix_level'] = self.get_current_vix()
            
            # Volume features
            if not today_data.empty:
                features['volume_ratio'] = today_data['volume'].sum() / self._calculate_average_volume(symbol, len(today_data))
            else:
                features['volume_ratio'] = 1.0
            
            # SMA feature
            features['sma_20_vs_price'] = 0  # Simplified

            # Regression features (first-half hour return)
            if not today_data.empty and minutes_from_open >= 30:
                # Calculate first half-hour return (9:30 to 10:00)
                price_10am = self._get_price_at_time(today_data, 10, 0)
                if price_10am is not None:
                    open_price = today_data.iloc[0]['open']
                    features['first_half_return'] = (price_10am / open_price) - 1
                else:
                    features['first_half_return'] = 0
            else:
                features['first_half_return'] = 0

            return features
            
        except Exception as e:
            print(f"Error extracting features for {symbol}: {e}")
            return None
    
    def calculate_vwap_from_data(self, data):
        """Calculate VWAP from historical data"""
        if data.empty:
            return 0
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).sum() / data['volume'].sum()
        
        return vwap if not pd.isna(vwap) else data['close'].mean()
    
    def train_rf_model(self):
        """Train RF with lower sample requirements"""
        if len(self.rf_training_data) < self.rf_min_training_samples:
            print(f"Need {self.rf_min_training_samples} samples, have {len(self.rf_training_data)}")
            return

        try:
            X = []
            y = []

            for sample in self.rf_training_data:
                X.append(list(sample['features'].values()))
                y.append(sample['label'])

            X = np.array(X)
            y = np.array(y)

            # ANALYZE CLASS DISTRIBUTION
            print(f"\n{'='*70}")
            print("TRAINING DATA CLASS DISTRIBUTION (Long-Only)")
            print(f"{'='*70}")
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            for label, count in zip(unique, counts):
                label_name = {1: 'BUY (Trade)', 0: 'HOLD (Don\'t Trade)'}.get(label, f'Unknown({label})')
                pct = (count / total) * 100
                print(f"  {label_name:20s} (label={label:2d}): {count:5d} samples ({pct:5.1f}%)")
            print(f"{'='*70}\n")

            # Check for severe imbalance
            if len(unique) < 2:
                print("⚠️  WARNING: Only one class in training data! Cannot train classifier.")
                return

            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 10:
                print(f"⚠️  WARNING: Severe class imbalance (ratio {imbalance_ratio:.1f}:1)")
                print(f"   Model may be biased toward majority class!")
                print(f"   Consider using class_weight='balanced' in RandomForest\n")
            
            # Scale features
            X_scaled = self.rf_scaler.fit_transform(X)

            # Use class balancing if severe imbalance detected
            class_weight = 'balanced' if imbalance_ratio > 3 else None
            if class_weight:
                print(f"Using class_weight='balanced' to handle imbalance\n")

            # Use simpler RF for small data
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
            
            # Cross-validation
            cv_scores = cross_val_score(self.rf_model, X_scaled, y, cv=5)
            print(f"Cross-validation scores: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
            
            # Final fit
            self.rf_model.fit(X_scaled, y)
            
            if hasattr(self.rf_model, 'oob_score_'):
                print(f"Out-of-bag score: {self.rf_model.oob_score_:.2f}")
            
            # Feature importance
            feature_names = list(self.rf_training_data[-1]['features'].keys())
            importances = self.rf_model.feature_importances_
            for name, imp in zip(feature_names, importances):
                print(f"  {name}: {imp:.3f}")
            
            self.use_rf_signals = True
            self.save_rf_model()
            
        except Exception as e:
            print(f"Error training RF: {e}")
            
    
    
    def generate_main_signal(self, symbol, current_data=None):
        """Enhanced signal generation with Random Forest"""
        # Get traditional signal
        traditional_signal = self.generate_traditional_signal(symbol, current_data)
        
        # If RF model is trained and ready
        if self.use_rf_signals and self.rf_model is not None:
            features = self.extract_rf_features(symbol)
            
            if features:
                try:
                    # Convert to array for prediction
                    feature_array = np.array([list(features.values())])
                    feature_scaled = self.rf_scaler.transform(feature_array)
                    
                    # Get RF prediction and probability
                    rf_signal = self.rf_model.predict(feature_scaled)[0]
                    rf_proba = self.rf_model.predict_proba(feature_scaled)[0]

                    # Binary classification: 1=BUY, 0=HOLD
                    rf_signal_str = 'BUY' if rf_signal == 1 else 'HOLD'

                    # Get confidence (probability of BUY class)
                    if len(rf_proba) > 1:
                        buy_confidence = rf_proba[1]  # Probability of class 1 (BUY)
                    else:
                        buy_confidence = rf_proba[0] if rf_signal == 1 else 1 - rf_proba[0]
                    
                    print(f"[{symbol}] RF Signal: {rf_signal_str}, Confidence: {buy_confidence:.2f}")
                    print(f"[{symbol}] Traditional Signal: {traditional_signal}")

                    # Combine signals (long-only strategy)
                    if buy_confidence > 0.65:
                        if rf_signal_str == 'BUY' and traditional_signal == 'BUY':
                            print(f"[{symbol}] STRONG BUY - Both models agree")
                            return 'BUY'
                        elif rf_signal_str == 'BUY' and traditional_signal == 'HOLD':
                            print(f"[{symbol}] RF says BUY but traditional says HOLD - using RF")
                            return 'BUY'
                        elif rf_signal_str == 'HOLD':
                            print(f"[{symbol}] RF suggests caution - holding")
                            return 'HOLD'
                    else:
                        # Low confidence - defer to traditional signal but only for BUYs
                        if traditional_signal == 'BUY':
                            print(f"[{symbol}] Low RF confidence, using traditional BUY signal")
                            return 'BUY'
                        else:
                            print(f"[{symbol}] Low RF confidence, traditional not BUY - holding")
                            return 'HOLD'
                    
                except Exception as e:
                    print(f"Error in RF prediction: {e}")
        
        return traditional_signal
    
    def record_trade_outcome(self, symbol, entry_time, entry_price, exit_price, signal):
        """Record trade outcomes for training RF model"""
        features = self.extract_rf_features(symbol)
        if features:
            # Calculate outcome
            if signal == 'BUY':
                profit_pct = (exit_price / entry_price) - 1
            else:
                profit_pct = (entry_price / exit_price) - 1
            
            # Label
            if profit_pct > 0.002:
                label = 1
            elif profit_pct < -0.002:
                label = -1
            else:
                label = 0
            
            self.rf_training_data.append({
                'features': features,
                'label': label,
                'profit_pct': profit_pct,
                'timestamp': entry_time
            })
            
            # Retrain periodically
            if len(self.rf_training_data) % 20 == 0:
                self.train_rf_model()
    
    def update_rf_model_daily(self):
        """Called at end of trading day to update model"""
        print("Updating RF model with today's data...")
        
        for symbol, data in self.symbol_data.items():
            if data.get('last_signal_time', datetime.datetime.min).date() == datetime.datetime.now().date():
                if data.get('last_entry_price') and data.get('last_signal'):
                    self.record_trade_outcome(
                        symbol,
                        data['last_entry_time'],
                        data['last_entry_price'],
                        get_current_price(symbol),
                        data['last_signal']
                    )
        
        # Retrain if we have new samples
        if len(self.rf_training_data) >= self.rf_min_training_samples + 10:
            self.train_rf_model()
    
    def save_rf_model(self):
        """Save RF model to disk"""
        try:
            joblib.dump(self.rf_model, 'rf_trading_model.pkl')
            joblib.dump(self.rf_scaler, 'rf_scaler.pkl')
            with open('rf_features.json', 'w') as f:
                json.dump(list(self.rf_training_data[-1]['features'].keys()), f)
            print("RF model saved")
        except Exception as e:
            print(f"Error saving RF model: {e}")
    
    def load_rf_model(self):
        """Load RF model from disk if available"""
        try:
            if os.path.exists('rf_trading_model.pkl'):
                self.rf_model = joblib.load('rf_trading_model.pkl')
                self.rf_scaler = joblib.load('rf_scaler.pkl')
                self.use_rf_signals = True
                print("RF model loaded successfully")
        except Exception as e:
            print(f"Error loading RF model: {e}")
    def calculate_opening_metrics(self, symbol):
        """Calculate gap and opening range metrics for first 30 min trading"""
        data = self.symbol_data[symbol]
        intraday_df = data['intraday_data']
        daily_df = data['daily_data']
        
        if intraday_df.empty or daily_df.empty:
            return False
        
        # Get today's data
        today = datetime.datetime.now().date()
        today_data = intraday_df[intraday_df['day'] == today]
        
        if today_data.empty:
            return False
        
        # Calculate gap
        open_price = today_data.iloc[0]['open']
        yesterday_data = intraday_df[intraday_df['day'] < today]
        
        if not yesterday_data.empty:
            prev_close = yesterday_data.iloc[-1]['close']
            data['gap_size'] = (open_price / prev_close) - 1
        else:
            data['gap_size'] = 0
        
        # Calculate opening range (first N minutes)
        current_time = datetime.datetime.now()
        minutes_from_open = (current_time.hour - 9) * 60 + (current_time.minute - 30)
        
        if minutes_from_open >= self.orb_period and not data['opening_range']['established']:
            # Get data for first 15 minutes
            orb_data = today_data[today_data['min_from_open'] <= self.orb_period]
            
            if len(orb_data) > 0:
                data['opening_range']['high'] = orb_data['high'].max()
                data['opening_range']['low'] = orb_data['low'].min()
                data['opening_range']['established'] = True
                
                # Calculate pre-market volatility estimate
                orb_range = (data['opening_range']['high'] - data['opening_range']['low']) / open_price
                data['pre_market_vol'] = orb_range
                
                print(f"[{symbol}] Opening range established: ${data['opening_range']['low']:.2f} - ${data['opening_range']['high']:.2f}")
                print(f"[{symbol}] Gap: {data['gap_size']*100:.2f}%, Pre-market vol: {orb_range*100:.2f}%")
                
        return True
    
    def generate_early_signal(self, symbol):
        """Generate trading signal for first 30 minutes using ORB strategy"""
        data = self.symbol_data[symbol]
        
        # Ensure we have required data
        if not self.calculate_opening_metrics(symbol):
            return 'HOLD'
        
        # Wait for opening range to be established
        if not data['opening_range']['established']:
            return 'HOLD'
        
        # Get current data
        intraday_df = data['intraday_data']
        today = datetime.datetime.now().date()
        today_data = intraday_df[intraday_df['day'] == today]
        
        if today_data.empty:
            return 'HOLD'
        
        current_price = today_data.iloc[-1]['close']
        current_volume = today_data['volume'].sum()
        
        # Calculate average volume for this time of day
        avg_volume = self._calculate_average_volume(symbol, len(today_data))
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Get opening range
        orb_high = data['opening_range']['high']
        orb_low = data['opening_range']['low']
        gap_size = data['gap_size']
        
        print(f"\n{'='*50}")
        print(f"[{symbol}] Early Trading Signal Analysis")
        print(f"{'='*50}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Opening Range: ${orb_low:.2f} - ${orb_high:.2f}")
        print(f"Gap: {gap_size*100:.2f}%")
        print(f"Volume Ratio: {volume_ratio:.2f}x")
        
        # Check if we have an active early trade
        if data['early_trade_active'] and data['current_position'] != 0:
            # Manage existing position
            return self._manage_early_position(symbol, current_price)
        
        # Entry signals
        
        # LONG signal: Break above opening range with strong gap and volume
        long_trigger = orb_high * (1 + self.orb_extension)
        if (current_price > long_trigger and 
            gap_size > self.gap_threshold and 
            volume_ratio > self.volume_multiplier):
            
            print(f"*** EARLY LONG SIGNAL - ORB Breakout ***")
            data['early_trade_active'] = True
            data['early_entry_price'] = current_price
            return 'BUY'
        
        # SHORT signal: Break below opening range with negative gap and volume
        short_trigger = orb_low * (1 - self.orb_extension)
        if (current_price < short_trigger and 
            gap_size < -self.gap_threshold and 
            volume_ratio > self.volume_multiplier):
            
            print(f"*** EARLY SHORT SIGNAL - ORB Breakdown ***")
            data['early_trade_active'] = True
            data['early_entry_price'] = current_price
            return 'SELL'
        
        # Fade signals (counter-trend) for extreme gaps without follow-through
        
        # Fade long gap: Large positive gap but price falling back into range
        if (gap_size > self.gap_threshold * 2 and  # Very large gap
            current_price < orb_high and  # Back inside range
            current_price < orb_low + (orb_high - orb_low) * 0.3):  # In lower 30% of range
            
            print(f"*** EARLY FADE SHORT - Gap Fade ***")
            data['early_trade_active'] = True
            data['early_entry_price'] = current_price
            return 'SELL'
        
        # Fade short gap: Large negative gap but price rising back into range
        if (gap_size < -self.gap_threshold * 2 and  # Very large negative gap
            current_price > orb_low and  # Back inside range
            current_price > orb_high - (orb_high - orb_low) * 0.3):  # In upper 30% of range
            
            print(f"*** EARLY FADE LONG - Gap Fade ***")
            data['early_trade_active'] = True
            data['early_entry_price'] = current_price
            return 'BUY'
        
        return 'HOLD'
    
    def generate_signal(self, symbol, current_data=None):
        """
        Main signal generation that routes to appropriate strategy based on time
        """
        print(f"Generating signal for {symbol} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ensure we have fresh data
        # if not self.collect_required_data(symbol):
        #     print('Aborting signal generation due to missing data')
        #     return 'HOLD'
        
        # Check time from market open
        current_time = datetime.datetime.now()
        minutes_from_open = (current_time.hour - 9) * 60 + (current_time.minute - 30)
        
        # Route to appropriate strategy
        if minutes_from_open < 30:
            # Use early trading strategy for first 30 minutes
            print(f"[{symbol}] Using EARLY TRADING strategy (minutes from open: {minutes_from_open})")
            if not self.collect_required_data(symbol):
                print('Aborting signal generation due to missing data')
                return 'HOLD'
            return self.generate_early_signal(symbol)
        else:
            # Check if we need to close any early positions first
            data = self.symbol_data[symbol]
            if data['early_trade_active'] and data['current_position'] != 0:
                print(f"[{symbol}] Closing early position before switching strategies")
                # Force close early positions when transitioning
                if data['current_position'] > 0:
                    return 'SELL'
                else:
                    return 'BUY'
            
            # Use existing EGARCH + regression strategy after 30 minutes
            print(f"[{symbol}] Using MAIN strategy (minutes from open: {minutes_from_open})")
            return self.generate_main_signal(symbol)
    
    def _manage_early_position(self, symbol, current_price):
        """Manage positions opened during first 30 minutes"""
        data = self.symbol_data[symbol]
        entry_price = data['early_entry_price']
        current_pos = data['current_position']
        
        if entry_price is None or current_pos == 0:
            return 'HOLD'
        
        # Calculate P&L
        if current_pos > 0:  # Long position
            pnl = (current_price / entry_price) - 1
            stop_price = entry_price * (1 - self.early_stop_pct)
            
            # Exit conditions
            if current_price <= stop_price:
                print(f"*** EARLY STOP LOSS - Exit Long ***")
                data['early_trade_active'] = False
                return 'SELL'
            
            # Take profit at 2x risk (2% gain)
            if pnl > self.early_stop_pct * 2:
                print(f"*** EARLY TAKE PROFIT - Exit Long ***")
                data['early_trade_active'] = False
                return 'SELL'
                
        else:  # Short position
            pnl = (entry_price / current_price) - 1
            stop_price = entry_price * (1 + self.early_stop_pct)
            
            # Exit conditions
            if current_price >= stop_price:
                print(f"*** EARLY STOP LOSS - Exit Short ***")
                data['early_trade_active'] = False
                return 'BUY'
            
            # Take profit at 2x risk
            if pnl > self.early_stop_pct * 2:
                print(f"*** EARLY TAKE PROFIT - Exit Short ***")
                data['early_trade_active'] = False
                return 'BUY'
        
        return 'HOLD'
    
    def _calculate_average_volume(self, symbol, num_bars):
        """Calculate average volume for the same time period over past days"""
        data = self.symbol_data[symbol]
        intraday_df = data['intraday_data']
        
        if intraday_df.empty:
            return 0
        
        # Get past 10 days of data for the same time period
        volumes = []
        unique_days = sorted(intraday_df['day'].unique())[:-1]  # Exclude today
        
        for day in unique_days[-10:]:
            day_data = intraday_df[intraday_df['day'] == day]
            if len(day_data) >= num_bars:
                volumes.append(day_data.iloc[:num_bars]['volume'].sum())
        
        return np.mean(volumes) if volumes else 0
        
    def collect_required_data(self, symbol):
        """
        Collects all required data for the strategy from TWS API.
        This replaces the CSV data loading from backtest.py
        """
        try:
            # Get intraday data for today
            intraday_df = self._get_intraday_data(symbol, '1 min', '15 D')
            if intraday_df.empty:
                print(f"No intraday data available for {symbol}")
                return False
                
            # Get daily data for volatility and regression
            daily_df = self._get_daily_data(symbol, '1 day', '60 D')
            if daily_df.empty:
                print(f"No daily data available for {symbol}")
                return False
            
            # Process and store the data
            self._process_market_data(symbol, intraday_df, daily_df)
            
            return True
            
        except Exception as e:
            print(f"Error collecting data for {symbol}: {e}")
            return False
    def _get_intraday_data(self, symbol, timeframe, duration):
        """Get intraday data from TWS API"""
        # Use the existing get_recent_bars function from ibkr1.py
        df = get_recent_bars(
            symbol, timeframe, duration, 
            for_calculation=True,
            client_instance=self.client,
            exchange_cache=self.risk_manager.exchange_cache
        )
        
        if not df.empty:
            # Add required calculated fields
            df['day'] = pd.to_datetime(df['date']).dt.date
            df['min_from_open'] = ((pd.to_datetime(df['date']) - 
                                   pd.to_datetime(df['date']).dt.normalize()) / 
                                   pd.Timedelta(minutes=1)) - (9 * 60 + 30)
            
            # Calculate VWAP
            for day in df['day'].unique():
                day_mask = df['day'] == day
                day_data = df[day_mask]
                
                # Calculate typical price and volume-weighted typical price
                hlc = (day_data['high'] + day_data['low'] + day_data['close']) / 3
                vol_x_hlc = hlc * day_data['volume']
                cum_vol_x_hlc = vol_x_hlc.cumsum()
                cum_volume = day_data['volume'].cumsum()
                
                # Set VWAP for this day only
            df.loc[day_mask, 'vwap'] = cum_vol_x_hlc.values / cum_volume.values
            
        return df
    
    def _get_daily_data(self, symbol, timeframe, duration):
        """Get daily data from TWS API"""
        df = get_recent_bars(
            symbol, timeframe, duration,
            for_calculation=True,
            client_instance=self.client,
            exchange_cache=self.risk_manager.exchange_cache
        )
        
        if not df.empty:
            # Calculate daily returns
            mask_1970 = df['date'].dt.year <= 1980  # Catch 1970s dates
        
            if mask_1970.any():
                print(f"⚠️  Found {mask_1970.sum()} rows with suspicious old dates (1970s), fixing...")
                
                # Method 1: Try to reconstruct from the raw bar.date if available
                # Method 2: Use sequential dating based on current date
                end_date = datetime.datetime.now().date()
                
                # Create a sequence of trading days going backwards
                fixed_dates = []
                current_date = end_date
                
                for i in range(len(df)):
                    while not self.is_trading_day(current_date):
                        current_date -= datetime.timedelta(days=1)
                    
                    # Set to market close time (4 PM ET)
                    fixed_datetime = datetime.datetime.combine(current_date, datetime.time(16, 0))
                    fixed_dates.append(fixed_datetime)
                    
                    # Move to previous trading day
                    current_date -= datetime.timedelta(days=1)
                
                # Reverse to get chronological order (oldest first)
                fixed_dates.reverse()
                
                # Apply the fixed dates
                df['date'] = fixed_dates
                
                print(f"✅ Fixed dates. New range: {df['date'].min()} to {df['date'].max()}")
            df['daily_return'] = df['close'].pct_change()
            df['daily_log_return'] = np.log(df['close'] / df['close'].shift(1))
            
        return df
    
    def _calculate_intraday_sigma(self, symbol):
        """
        Calculate intraday sigma for each 15-minute interval.
        This matches the paper's methodology for calculating noise boundaries.
        """
        try:
            data = self.symbol_data[symbol]
            intraday_df = data.get('intraday_data', pd.DataFrame())
            
            if intraday_df.empty:
                print(f"No intraday data available for sigma calculation for {symbol}")
                return
            
            # Get the last 14 trading days
            unique_days = sorted(intraday_df['day'].unique())
            if len(unique_days) < 14:
                print(f"Insufficient days ({len(unique_days)}) for sigma calculation")
                return
            
            last_14_days = unique_days[-14:]
            
            # Calculate sigma for each 15-minute interval
            sigma_by_time = {}
            
            for minutes in range(0, 391, 15):  # Every 15 minutes from 0 to 390
                moves = []
                
                for day in last_14_days:
                    day_data = intraday_df[intraday_df['day'] == day]
                    
                    if len(day_data) < 10:
                        continue
                    
                    # Get opening price
                    open_price = day_data.iloc[0]['open']
                    
                    # Find price at this time
                    time_mask = (day_data['min_from_open'] >= minutes - 2) & (day_data['min_from_open'] <= minutes + 2)
                    
                    if time_mask.any():
                        price_at_time = day_data[time_mask].iloc[0]['close']
                        # Calculate absolute move from open
                        move = abs((price_at_time / open_price) - 1)
                        moves.append(move)
                
                # Calculate average move for this time
                if moves:
                    sigma_by_time[minutes] = np.mean(moves)
                else:
                    # Fallback to EGARCH if no data
                    sigma_by_time[minutes] = data.get('egarch_vol', 0.01)
            
            # Store the calculated sigmas
            data['intraday_sigma'] = sigma_by_time
            
            print(f"Calculated intraday sigma for {symbol}: {len(sigma_by_time)} intervals")
            
        except Exception as e:
            print(f"Error calculating intraday sigma for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def _process_market_data(self, symbol, intraday_df, daily_df):
        """Process and store market data for strategy calculations"""
        data = self.symbol_data[symbol]
        # Store the data
        data['intraday_data'] = intraday_df
        data['daily_data'] = daily_df
        
        if 'prev_month_vol' not in data or data['prev_month_vol'] is None:
            data['prev_month_vol'] = 0.02  # Default 2% volatility
        print("a")
        # Calculate EGARCH volatility forecast
        self._calculate_egarch_forecast(symbol, daily_df)
        
        # Calculate intraday sigma (NEW)
        self._calculate_intraday_sigma(symbol)
        
        # Calculate regression parameters if we have enough data
        self._calculate_regression_params(symbol, daily_df)
        
        # Calculate previous month volatility
        self._calculate_prev_month_vol(symbol, daily_df)
    
    def _calculate_egarch_forecast(self, symbol, daily_df):
        """Calculate EGARCH volatility forecast"""
        try:
            # Need at least lag + 1 days of data
            if len(daily_df) < self.lag + 1:
                print(f"Insufficient data for EGARCH calculation for {symbol}")
                return
            
            # Get recent returns for GARCH model
            recent_returns = daily_df['daily_log_return'].dropna().tail(self.lag).values * 100
            
            # Fit GARCH model
            model = arch_model(
                recent_returns, 
                vol='GARCH', 
                p=self.p_terms, 
                o=self.o_terms, 
                q=self.q_terms
            )
            
            res = model.fit(disp='off', show_warning=False, options={'maxiter': 100})
            
            # Get volatility forecast
            forecast = res.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
            
            self.symbol_data[symbol]['egarch_vol'] = vol_forecast
            print(f"EGARCH volatility forecast for {symbol}: {vol_forecast:.4f}")
            
        except Exception as e:
            print(f"Error calculating EGARCH for {symbol}: {e}")
            # Use simple volatility as fallback
            self.symbol_data[symbol]['egarch_vol'] = daily_df['daily_return'].std()
    def is_market_holiday(self, date):
        """
        Check if a given date is a US market holiday.
        This is a basic implementation - you might want to use a more comprehensive library.
        """
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, datetime.datetime):
            date = date.date()
        
        year = date.year
        
        # Define major US market holidays
        holidays = []
        
        # New Year's Day (observed)
        new_years = datetime.date(year, 1, 1)
        if new_years.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 1, 3))  # Monday
        elif new_years.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 1, 2))  # Monday
        else:
            holidays.append(new_years)
        
        # Martin Luther King Jr. Day (3rd Monday in January)
        jan_1 = datetime.date(year, 1, 1)
        days_to_monday = (7 - jan_1.weekday()) % 7
        first_monday = jan_1 + datetime.timedelta(days=days_to_monday)
        mlk_day = first_monday + datetime.timedelta(days=14)  # 3rd Monday
        holidays.append(mlk_day)
        
        # Presidents Day (3rd Monday in February)
        feb_1 = datetime.date(year, 2, 1)
        days_to_monday = (7 - feb_1.weekday()) % 7
        first_monday = feb_1 + datetime.timedelta(days=days_to_monday)
        presidents_day = first_monday + datetime.timedelta(days=14)  # 3rd Monday
        holidays.append(presidents_day)
        
        # Good Friday (Friday before Easter) - complex calculation
        # Using simplified approximation for major years
        good_friday_dates = {
            2024: datetime.date(2024, 3, 29),
            2025: datetime.date(2025, 4, 18),
            2026: datetime.date(2026, 4, 3),
            2027: datetime.date(2027, 3, 26),
            2028: datetime.date(2028, 4, 14),
            2029: datetime.date(2029, 3, 30),
            2030: datetime.date(2030, 4, 19),
        }
        if year in good_friday_dates:
            holidays.append(good_friday_dates[year])
        
        # Memorial Day (last Monday in May)
        may_31 = datetime.date(year, 5, 31)
        days_back_to_monday = (may_31.weekday() - 0) % 7
        memorial_day = may_31 - datetime.timedelta(days=days_back_to_monday)
        holidays.append(memorial_day)
        
        # Juneteenth (June 19, observed if weekend)
        juneteenth = datetime.date(year, 6, 19)
        if juneteenth.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 6, 18))  # Friday
        elif juneteenth.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 6, 20))  # Monday
        else:
            holidays.append(juneteenth)
        
        # Independence Day (July 4, observed if weekend)
        july_4 = datetime.date(year, 7, 4)
        if july_4.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 7, 3))  # Friday
        elif july_4.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 7, 5))  # Monday
        else:
            holidays.append(july_4)
        
        # Labor Day (1st Monday in September)
        sep_1 = datetime.date(year, 9, 1)
        days_to_monday = (7 - sep_1.weekday()) % 7
        labor_day = sep_1 + datetime.timedelta(days=days_to_monday)
        holidays.append(labor_day)
        
        # Thanksgiving (4th Thursday in November)
        nov_1 = datetime.date(year, 11, 1)
        days_to_thursday = (3 - nov_1.weekday()) % 7  # Thursday = 3
        first_thursday = nov_1 + datetime.timedelta(days=days_to_thursday)
        thanksgiving = first_thursday + datetime.timedelta(days=21)  # 4th Thursday
        holidays.append(thanksgiving)
        
        # Christmas (December 25, observed if weekend)
        christmas = datetime.date(year, 12, 25)
        if christmas.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 12, 24))  # Friday
        elif christmas.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 12, 26))  # Monday
        else:
            holidays.append(christmas)
        
        return date in holidays
    def is_trading_day(self, date):
        """Check if a date is a trading day (not weekend or holiday)"""
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, datetime.datetime):
            date = date.date()
        
        # Check if it's a weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's a holiday
        if self.is_market_holiday(date):
            return False
        
        return True
    def get_previous_trading_day(self, date):
        """Get the previous trading day before the given date"""
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif isinstance(date, datetime.datetime):
            date = date.date()
        
        # Go back one day at a time until we find a trading day
        prev_date = date - datetime.timedelta(days=1)
        while not self.is_trading_day(prev_date):
            prev_date -= datetime.timedelta(days=1)
        
        return prev_date
    # def _calculate_regression_params(self, symbol, daily_df):
    #     """Calculate monthly regression parameters for open-to-close prediction with proper alignment"""
    #     try:
    #         # Get current month data
    #         current_month = datetime.datetime.now().strftime('%Y-%m')
    #         month_mask = pd.to_datetime(daily_df['date']).dt.strftime('%Y-%m') == current_month
            
    #         if month_mask.sum() < 5:  # Need at least 5 days of data
    #             # Use previous month if current month has insufficient data
    #             prev_month = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m')
    #             month_mask = pd.to_datetime(daily_df['date']).dt.strftime('%Y-%m') == prev_month
            
    #         month_data = daily_df[month_mask].copy()
            
    #         if len(month_data) < 5:
    #             print(f"Insufficient data for regression calculation for {symbol}")
    #             return
            
    #         first_half_hour_returns = []
    #         last_half_hour_returns = []
            
    #         for date in month_data['date']:
    #             # Get intraday data for this specific date
    #             date_str = pd.to_datetime(date).strftime('%Y%m%d')
    #             intraday_data = self._get_intraday_data_for_date(symbol, date)
                
    #             if intraday_data is not None and len(intraday_data) > 0:
    #                 # Calculate first half-hour return (9:30 to 10:00)
    #                 open_price = intraday_data.iloc[0]['open']
    #                 price_10am = self._get_price_at_time(intraday_data, 10, 0)
                    
    #                 # Calculate last half-hour return (3:30 to 4:00)
    #                 price_330pm = self._get_price_at_time(intraday_data, 15, 30)
    #                 close_price = intraday_data.iloc[-1]['close']
                    
    #                 if price_10am and price_330pm:
    #                     first_half_return = (price_10am / open_price) - 1
    #                     last_half_return = (close_price / price_330pm) - 1
                        
    #                     first_half_hour_returns.append(first_half_return)
    #                     last_half_hour_returns.append(last_half_return)
            
    #         if len(first_half_hour_returns) < 3:
    #             print(f"Insufficient intraday data for regression: {len(first_half_hour_returns)} observations")
    #             return
            
    #         # Sort by date to ensure proper order
    #         month_data = month_data.sort_values('date').reset_index(drop=True)
            
    #         # Calculate open returns (open vs previous close) with proper alignment
    #         # IMPORTANT: We need to ensure we're using the previous TRADING DAY's close
    #         month_data['prev_close'] = month_data['close'].shift(1)
            
    #         # For the first row, we need the previous trading day's close from outside this month
    #         if len(month_data) > 0:
    #             first_date = pd.to_datetime(month_data.iloc[0]['date']).date()
    #             prev_trading_date = self.get_previous_trading_day(first_date)
                
    #             # Try to find the previous close from the full dataset
    #             prev_close_mask = pd.to_datetime(daily_df['date']).dt.date == prev_trading_date
    #             if prev_close_mask.any():
    #                 prev_close_value = daily_df[prev_close_mask]['close'].iloc[-1]
    #                 month_data.loc[0, 'prev_close'] = prev_close_value
    #                 print(f"Using previous trading day ({prev_trading_date}) close: {prev_close_value}")
            
    #         # Calculate returns
    #         month_data['open_return'] = month_data['open'] / month_data['prev_close'] - 1
            
    #         # Ensure daily_return is calculated if not already present
    #         if 'daily_return' not in month_data.columns:
    #             month_data['daily_return'] = month_data['close'] / month_data['open'] - 1
            
    #         # CRITICAL FIX: Drop NaN values from BOTH series together to maintain alignment
    #         regression_data = month_data[['open_return', 'daily_return']].dropna()
            
    #         print(f"Regression data for {symbol}:")
    #         print(f"  Month data shape: {month_data.shape}")
    #         print(f"  After dropna shape: {regression_data.shape}")
    #         print(f"  Open return range: {regression_data['open_return'].min():.4f} to {regression_data['open_return'].max():.4f}")
    #         print(f"  Daily return range: {regression_data['daily_return'].min():.4f} to {regression_data['daily_return'].max():.4f}")
            
    #         if len(regression_data) < 3:
    #             print(f"Insufficient aligned data for regression: {len(regression_data)} rows")
    #             return
            
    #         # Prepare data for regression - now they're guaranteed to be the same length
    #         X = regression_data['open_return'].values.reshape(-1, 1)
    #         y = regression_data['daily_return'].values
            
    #         # Verify they're the same length
    #         assert len(X) == len(y), f"X and y still have different lengths: {len(X)} vs {len(y)}"
            
    #         # Fit regression model
    #         model = LinearRegression()
    #         model.fit(X, y)
            
    #         # Store results
    #         self.symbol_data[symbol]['regression_params']['intercept'] = model.intercept_
    #         self.symbol_data[symbol]['regression_params']['coef'] = model.coef_[0]
            
    #         # Calculate R-squared for diagnostics
    #         r_squared = model.score(X, y)
            
    #         print(f"Regression for {symbol}:")
    #         print(f"  Intercept: {model.intercept_:.6f}")
    #         print(f"  Coefficient: {model.coef_[0]:.6f}")
    #         print(f"  R-squared: {r_squared:.4f}")
    #         print(f"  Sample size: {len(X)} observations")
            
    #     except Exception as e:
    #         print(f"Error calculating regression for {symbol}: {e}")
    #         import traceback
    #         traceback.print_exc()
        
    # def _calculate_prev_month_vol(self, symbol, daily_df):
    #     """Calculate previous month's volatility"""
    #     try:
    #         # Get previous month data
    #         prev_month = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m')
    #         month_mask = pd.to_datetime(daily_df['date']).dt.strftime('%Y-%m') == prev_month
            
    #         if month_mask.sum() > 0:
    #             prev_month_vol = daily_df[month_mask]['daily_return'].std()
    #             self.symbol_data[symbol]['prev_month_vol'] = prev_month_vol
    #         else:
    #             # Use overall volatility as fallback
    #             self.symbol_data[symbol]['prev_month_vol'] = daily_df['daily_return'].std()
                
    #     except Exception as e:
    #         print(f"Error calculating previous month volatility for {symbol}: {e}")
    def _calculate_prev_month_vol(self, symbol, daily_df):
        """Calculate previous month's volatility with better error handling"""
        try:
            if daily_df.empty or 'date' not in daily_df.columns:
                print(f"No data available for {symbol}")
                self.symbol_data[symbol]['prev_month_vol'] = 0.02
                return
            
            # Check if dates are already datetime objects
            if pd.api.types.is_datetime64_any_dtype(daily_df['date']):
                # Dates are already datetime, use them directly
                dates_series = daily_df['date']
            else:
                # Convert to datetime if they're not already
                dates_series = pd.to_datetime(daily_df['date'], errors='coerce')
            # Get previous month data
            prev_month = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m')
            month_strings = dates_series.dt.strftime('%Y-%m')
            print(f"Available months in data: {month_strings.unique()}")
        
        # Create the mask
            month_mask = month_strings == prev_month
            if month_mask.sum() >= 5:  # Need at least 5 days for meaningful volatility
                prev_month_vol = daily_df[month_mask]['daily_return'].std()
                
                # Validate the calculated volatility
                if pd.isna(prev_month_vol) or prev_month_vol <= 0:
                    print(f"Invalid volatility calculated for {symbol}: {prev_month_vol}")
                    prev_month_vol = daily_df['daily_return'].std()  # Use overall volatility
            else:
                # Use overall volatility as fallback
                print(month_mask)
                print(f"Insufficient data for {symbol} in {prev_month}, using overall volatility")
                prev_month_vol = daily_df['daily_return'].std()
            
            # Final validation
            if pd.isna(prev_month_vol) or prev_month_vol <= 0:
                prev_month_vol = 0.02  # Default 2% volatility
                
            self.symbol_data[symbol]['prev_month_vol'] = prev_month_vol
            print(f"Set volatility for {symbol}: {prev_month_vol:.4f}")
            
        except Exception as e:
            print(f"Error calculating previous month volatility for {symbol}: {e}")
            # Set a default value on error
            self.symbol_data[symbol]['prev_month_vol'] = 0.02
            print(f"Set default volatility for {symbol}: 0.02")
        
    # def update_allocations(self, symbols):
    #     """Update portfolio allocations based on inverse volatility (monthly)"""
    #     current_month = datetime.datetime.now().strftime('%Y-%m')
        
    #     # Only update allocations once per month
    #     if self.last_allocation_update == current_month:
    #         return
        
    #     volatilities = {}
    #     for symbol in symbols:
    #         print(symbol)
    #         print(self.symbol_data[symbol])
    #         vol = self.symbol_data[symbol].get('prev_month_vol', 0.02)  # Default 2% vol
    #         print(vol)
    #         if vol > 0:
    #             volatilities[symbol] = self.target_vol / vol
        
    #     # Normalize to sum to 1
    #     total_vol_adjusted = sum(volatilities.values())
    #     if total_vol_adjusted > 0:
    #         self.allocations = {
    #             symbol: round(vol_adj / total_vol_adjusted, 2) 
    #             for symbol, vol_adj in volatilities.items()
    #         }
    #     else:
    #         # Equal weight fallback
    #         self.allocations = {symbol: round(1/len(symbols), 2) for symbol in symbols}
        
    #     self.last_allocation_update = current_month
    #     print(f"Updated allocations: {self.allocations}")
    def update_allocations(self, symbols):
        """Update portfolio allocations based on inverse volatility (monthly)"""
        current_month = datetime.datetime.now().strftime('%Y-%m')
        print(current_month)
        # Only update allocations once per month
        if self.last_allocation_update == current_month:
            return
        
        volatilities = {}
        for symbol in symbols:
            # Get volatility with multiple fallback layers
            vol = self.symbol_data[symbol].get('prev_month_vol', None)
            
            # First fallback: check if vol is None or invalid
            if vol is None or vol <= 0:
                print(f"Warning: Invalid volatility for {symbol}, calculating fresh...")
                # Try to calculate it now
                try:
                    daily_df = self._get_daily_data(symbol, '1 day', '60 D')
                    if not daily_df.empty:
                        self._calculate_prev_month_vol(symbol, daily_df)
                        vol = self.symbol_data[symbol].get('prev_month_vol', None)
                except Exception as e:
                    print(f"Error calculating volatility for {symbol}: {e}")
            
            # Second fallback: use default if still invalid
            if vol is None or vol <= 0:
                vol = 0.02  # Default 2% volatility
                print(f"Using default volatility of {vol} for {symbol}")
            
            # Store the inverse volatility weighted value
            volatilities[symbol] = self.target_vol / vol
        
        # Normalize to sum to 1
        total_vol_adjusted = sum(volatilities.values())
        if total_vol_adjusted > 0:
            self.allocations = {
                symbol: round(vol_adj / total_vol_adjusted, 2) 
                for symbol, vol_adj in volatilities.items()
            }
        else:
            # Equal weight fallback
            self.allocations = {symbol: round(1/len(symbols), 2) for symbol in symbols}
        
        self.last_allocation_update = current_month
        print(f"Updated allocations: {self.allocations}")
    
    # def generate_main_signal(self, symbol, current_data=None):
    #     """
    #     Generate trading signal based on EGARCH bands and regression confirmation.
    #     This is the core logic from backtest.py adapted for live trading.
    #     """
    #     try:
    #         # Ensure we have fresh data
    #         if not self.collect_required_data(symbol):
    #             print('Aborting signal generation due to missing data')
    #             return 'HOLD'
            
    #         data = self.symbol_data[symbol]
            
    #         # Check if we have all required parameters
    #         if (data['egarch_vol'] is None or 
    #             data['regression_params']['intercept'] is None):
    #             print(f"Missing required parameters for {symbol}")
    #             return 'HOLD'
            
    #         # Get current market data
    #         intraday_df = data['intraday_data']
    #         if intraday_df.empty:
    #             return 'HOLD'
            
    #         # Get today's data
    #         today = datetime.datetime.now().date()
    #         today_data = intraday_df[intraday_df['day'] == today]
            
    #         if today_data.empty:
    #             print(f"No data for today for {symbol}")
    #             return 'HOLD'
            
    #         # Get market open price and previous close
    #         try:
    #             open_price = today_data.iloc[0]['open']
    #             # Get previous day's close
    #             yesterday_data = intraday_df[intraday_df['day'] < today]
    #             if not yesterday_data.empty:
    #                 prev_close = yesterday_data.iloc[-1]['close']
    #             else:
    #                 prev_close = open_price  # Fallback
    #         except Exception as e:
    #             print(f"Error getting prices: {e}")
    #             return 'HOLD'
            
    #         # Calculate bands
    #         egarch_vol = data['egarch_vol']
    #         upper_bound = max(open_price, prev_close) * (1 + egarch_vol)
    #         lower_bound = min(open_price, prev_close) * (1 - egarch_vol)
            
    #         # Calculate open return
    #         open_return = (open_price / prev_close) - 1
            
    #         # Predict close return using regression
    #         intercept = data['regression_params']['intercept']
    #         coef = data['regression_params']['coef']
    #         predicted_close_return = intercept + coef * open_return
            
    #         # Check regression signal (both open and predicted close in same direction)
    #         regression_signal = (open_return * predicted_close_return) > 0
            
    #         if not regression_signal:
    #             return 'HOLD'
            
    #         # Get current price and VWAP
    #         current_price = today_data.iloc[-1]['close']
    #         current_vwap = today_data.iloc[-1]['vwap']
            
    #         # Check if it's time to trade (based on trade frequency)
    #         current_time = datetime.datetime.now()
    #         minutes_from_open = (current_time.hour - 9) * 60 + (current_time.minute - 30)
            
    #         print(f"\n{'='*50}")
    #         print(f"[{symbol}] Signal Generation Debug at {current_time.strftime('%H:%M:%S')}")
    #         print(f"{'='*50}")
    #         print(f"Time Check:")
    #         print(f"  Minutes from open: {minutes_from_open}")
    #         print(f"  Valid trade time? {minutes_from_open >= 30 and minutes_from_open % self.trade_freq == 0}")
            
    #         print(f"\nPrice Analysis:")
    #         print(f"  Previous close: ${prev_close:.2f}")
    #         print(f"  Open price: ${open_price:.2f}")
    #         print(f"  Current price: ${current_price:.2f}")
    #         print(f"  VWAP: ${current_vwap:.2f}")
            
    #         print(f"\nVolatility Bands:")
    #         print(f"  EGARCH volatility: {egarch_vol:.4f} ({egarch_vol*100:.2f}%)")
    #         print(f"  Upper bound: ${upper_bound:.2f}")
    #         print(f"  Lower bound: ${lower_bound:.2f}")
    #         print(f"  Price vs Upper: ${current_price:.2f} {'>' if current_price > upper_bound else '<='} ${upper_bound:.2f}")
    #         print(f"  Price vs Lower: ${current_price:.2f} {'<' if current_price < lower_bound else '>='} ${lower_bound:.2f}")
            
    #         print(f"\nRegression Analysis:")
    #         print(f"  Open return: {open_return:.4f} ({open_return*100:.2f}%)")
    #         print(f"  Predicted close return: {predicted_close_return:.4f} ({predicted_close_return*100:.2f}%)")
    #         print(f"  Regression signal (same direction): {regression_signal}")
    #         print(f"  Regression params: intercept={intercept:.4f}, coef={coef:.4f}")
            
    #         print(f"\nSignal Conditions:")
    #         print(f"  BUY conditions:")
    #         print(f"    - Price > Upper: {current_price > upper_bound}")
    #         print(f"    - Price > VWAP: {current_price > current_vwap}")
    #         print(f"    - Open return > 0: {open_return > 0}")
    #         print(f"    - Predicted return > 0: {predicted_close_return > 0}")
    #         print(f"    - ALL MET: {all([current_price > upper_bound, current_price > current_vwap, open_return > 0, predicted_close_return > 0])}")
            
    #         print(f"  SELL conditions:")
    #         print(f"    - Price < Lower: {current_price < lower_bound}")
    #         print(f"    - Price < VWAP: {current_price < current_vwap}")
    #         print(f"    - Open return < 0: {open_return < 0}")
    #         print(f"    - Predicted return < 0: {predicted_close_return < 0}")
    #         print(f"    - ALL MET: {all([current_price < lower_bound, current_price < current_vwap, open_return < 0, predicted_close_return < 0])}")
            
    #         print(f"\nCurrent Position: {data['current_position']}")
    #         print(f"{'='*50}\n")
            
    #         if minutes_from_open < 30:  # Don't trade in first 30 minutes
    #             print(f"[{symbol}] Skipping - within first 30 minutes")
    #             return 'HOLD'
            
    #         # Check if we're at a valid trade time
    #         if minutes_from_open % self.trade_freq != 0:
    #             return 'HOLD'
            
    #         if not regression_signal:
    #             print(f"[{symbol}] No regression signal")
    #             return 'HOLD'
            
    #         # Generate signals based on bands and VWAP
    #         if (current_price > upper_bound and 
    #             current_price > current_vwap and 
    #             open_return > 0 and 
    #             predicted_close_return > 0):
    #             print(f"BUY signal for {symbol}: price={current_price:.2f}, upper={upper_bound:.2f}")
    #             return 'BUY'
            
    #         elif (current_price < lower_bound and 
    #               current_price < current_vwap and 
    #               open_return < 0 and 
    #               predicted_close_return < 0):
    #             print(f"SELL signal for {symbol}: price={current_price:.2f}, lower={lower_bound:.2f}")
    #             return 'SELL'
            
    #         # Check for exit signal (price reverting to mean)
    #         current_pos = data['current_position']
    #         if current_pos != 0:
    #             # Exit long position
    #             if current_pos > 0 and (current_price < current_vwap or current_price < open_price):
    #                 print(f"EXIT LONG signal for {symbol}")
    #                 return 'SELL'
    #             # Exit short position  
    #             elif current_pos < 0 and (current_price > current_vwap or current_price > open_price):
    #                 print(f"EXIT SHORT signal for {symbol}")
    #                 return 'BUY'
            
    #         return 'HOLD'
            
    #     except Exception as e:
    #         print(f"Error generating signal for {symbol}: {e}")
    #         print(f"[{symbol}] Error generating signal: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return 'HOLD'
    
    # Add this helper method to properly calculate regression for the strategy
    def _calculate_half_hour_regression_from_intraday(self, symbol):
        """
        Calculate regression parameters using actual intraday data.
        This is a simplified version that uses the existing intraday data.
        """
        try:
            intraday_df = self.symbol_data[symbol].get('intraday_data', pd.DataFrame())
            if intraday_df.empty:
                return
            
            # Get data for the last 20 trading days
            unique_days = sorted(intraday_df['day'].unique())[-20:]
            
            first_half_returns = []
            last_half_returns = []
            
            for day in unique_days:
                day_data = intraday_df[intraday_df['day'] == day]
                if len(day_data) < 10:  # Need enough data points
                    continue
                    
                # Get opening price
                open_price = day_data.iloc[0]['open']
                
                # Find price around 10 AM (30 minutes from open)
                mask_10am = (day_data['min_from_open'] >= 25) & (day_data['min_from_open'] <= 35)
                if mask_10am.any():
                    price_10am = day_data[mask_10am].iloc[0]['close']
                else:
                    continue
                
                # Find price around 3:30 PM (360 minutes from open)
                mask_330pm = (day_data['min_from_open'] >= 355) & (day_data['min_from_open'] <= 365)
                if mask_330pm.any():
                    price_330pm = day_data[mask_330pm].iloc[0]['close']
                else:
                    continue
                    
                # Get closing price
                close_price = day_data.iloc[-1]['close']
                
                # Calculate returns
                first_half_return = (price_10am / open_price) - 1
                last_half_return = (close_price / price_330pm) - 1
                
                first_half_returns.append(first_half_return)
                last_half_returns.append(last_half_return)
            
            if len(first_half_returns) >= 5:  # Need at least 5 observations
                # Fit regression
                X = np.array(first_half_returns).reshape(-1, 1)
                y = np.array(last_half_returns)
                
                model = LinearRegression()
                model.fit(X, y)
                
                self.symbol_data[symbol]['regression_params']['intercept'] = model.intercept_
                self.symbol_data[symbol]['regression_params']['coef'] = model.coef_[0]
                
                print(f"Updated half-hour regression for {symbol}: α={model.intercept_:.4f}, β={model.coef_[0]:.4f}")
            
        except Exception as e:
            print(f"Error in half-hour regression calculation: {e}")

    def _calculate_regression_params(self, symbol, daily_df):
        """
        Calculate regression parameters for first vs last half-hour returns.
        This version works with the intraday data already collected.
        """
        try:
            # Get the intraday data we already have
            intraday_df = self.symbol_data[symbol].get('intraday_data', pd.DataFrame())
            
            if intraday_df.empty:
                print(f"No intraday data available for {symbol} regression calculation")
                # Set default values instead of leaving as None
                self.symbol_data[symbol]['regression_params']['intercept'] = 0.0
                self.symbol_data[symbol]['regression_params']['coef'] = 0.5
                return
            
            # Get unique trading days from the intraday data
            unique_days = intraday_df['day'].unique()
            
            # We need at least 5 days for meaningful regression
            if len(unique_days) < 5:
                print(f"Insufficient days ({len(unique_days)}) for regression calculation")
                # Set default values
                self.symbol_data[symbol]['regression_params']['intercept'] = 0.0
                self.symbol_data[symbol]['regression_params']['coef'] = 0.5
                return
            
            # Calculate first and last half-hour returns for each day
            first_half_returns = []
            last_half_returns = []
            
            for day in unique_days:
                day_data = intraday_df[intraday_df['day'] == day].copy()
                
                # Skip days with insufficient data
                if len(day_data) < 50:  # Need enough bars for the full day
                    continue
                
                # Sort by time to ensure chronological order
                day_data = day_data.sort_values('date')
                
                try:
                    # Get opening price (first bar of the day)
                    open_price = day_data.iloc[0]['open']
                    
                    # Find price around 10:00 AM (30 minutes from market open)
                    # min_from_open should be around 30
                    mask_10am = (day_data['min_from_open'] >= 28) & (day_data['min_from_open'] <= 32)
                    if mask_10am.any():
                        price_10am = day_data[mask_10am].iloc[0]['close']
                    else:
                        # If no exact match, find closest
                        closest_10am_idx = (day_data['min_from_open'] - 30).abs().idxmin()
                        price_10am = day_data.loc[closest_10am_idx, 'close']
                    
                    # Find price around 3:30 PM (360 minutes from market open)
                    # min_from_open should be around 360
                    mask_330pm = (day_data['min_from_open'] >= 358) & (day_data['min_from_open'] <= 362)
                    if mask_330pm.any():
                        price_330pm = day_data[mask_330pm].iloc[0]['close']
                    else:
                        # If no exact match, find closest
                        closest_330pm_idx = (day_data['min_from_open'] - 360).abs().idxmin()
                        price_330pm = day_data.loc[closest_330pm_idx, 'close']
                    
                    # Get closing price (last bar of the day)
                    close_price = day_data.iloc[-1]['close']
                    
                    # Calculate the returns
                    first_half_return = (price_10am / open_price) - 1
                    last_half_return = (close_price / price_330pm) - 1
                    
                    # Only add if returns are reasonable (not extreme outliers)
                    if abs(first_half_return) < 0.1 and abs(last_half_return) < 0.1:
                        first_half_returns.append(first_half_return)
                        last_half_returns.append(last_half_return)
                        
                except Exception as e:
                    print(f"Error processing day {day}: {e}")
                    continue
            
            # Now fit the regression if we have enough data points
            if len(first_half_returns) >= 5:
                X = np.array(first_half_returns).reshape(-1, 1)
                y = np.array(last_half_returns)
                
                # Fit the linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Store the parameters
                self.symbol_data[symbol]['regression_params']['intercept'] = model.intercept_
                self.symbol_data[symbol]['regression_params']['coef'] = model.coef_[0]
                
                # Calculate R-squared for diagnostics
                r_squared = model.score(X, y)
                
                print(f"Successfully calculated regression for {symbol}:")
                print(f"  Intercept (α): {model.intercept_:.6f}")
                print(f"  Coefficient (β): {model.coef_[0]:.6f}")
                print(f"  R-squared: {r_squared:.4f}")
                print(f"  Sample size: {len(X)} days")
            else:
                print(f"Insufficient data points ({len(first_half_returns)}) for regression")
                # Set reasonable default values
                self.symbol_data[symbol]['regression_params']['intercept'] = 0.0
                self.symbol_data[symbol]['regression_params']['coef'] = 0.5
                
        except Exception as e:
            print(f"Error in regression calculation for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            # Set default values on error
            self.symbol_data[symbol]['regression_params']['intercept'] = 0.0
            self.symbol_data[symbol]['regression_params']['coef'] = 0.5

    def _get_intraday_data_for_date(self, symbol, date):
        """Get intraday data for a specific date"""
        try:
            # This is a helper function - you might need to adjust based on your data source
            # For now, we'll use the existing intraday data and filter by date
            all_intraday = self.symbol_data[symbol].get('intraday_data', pd.DataFrame())
            if not all_intraday.empty:
                date_only = pd.to_datetime(date).date()
                mask = all_intraday['day'] == date_only
                return all_intraday[mask]
            return None
        except:
            return None

    def _get_price_at_time(self, intraday_df, hour, minute):
        """Get price at a specific time from intraday data"""
        try:
            target_minutes = (hour - 9) * 60 + (minute - 30)
            # Find the closest bar to target time
            time_diff = abs(intraday_df['min_from_open'] - target_minutes)
            closest_idx = time_diff.idxmin()
            return intraday_df.loc[closest_idx, 'close']
        except:
            return None

    def generate_traditional_signal(self, symbol, current_data=None):
        """
        Generate trading signal based on noise boundaries, VWAP, and regression confirmation.
        Corrected to match the PDF strategy exactly.
        
        
        """
        
        print(f"Generating signal for {symbol} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            current_time = datetime.datetime.now()
            minutes_from_open = (current_time.hour - 9) * 60 + (current_time.minute - 30)
            if minutes_from_open < 30:
                print(f"[{symbol}] Skipping - within first 30 minutes")
                return 'HOLD'
            if minutes_from_open % self.trade_freq > 2:
                    print(f"[{symbol}] Skipping - not at valid trade time")
                    return 'HOLD'
            # Ensure we have fresh data
            if not self.collect_required_data(symbol):
                print('Aborting signal generation due to missing data')
                return 'HOLD'
            
            data = self.symbol_data[symbol]
            
            # Check if we have all required parameters
            if (data['egarch_vol'] is None or 
                data['regression_params']['intercept'] is None):
                print(f"Missing required parameters for {symbol}")
                return 'HOLD'
            
            # Get current market data
            intraday_df = data['intraday_data']
            if intraday_df.empty:
                return 'HOLD'
            
            # Get today's data
            today = datetime.datetime.now().date()
            today_data = intraday_df[intraday_df['day'] == today]
            
            if today_data.empty:
                print(f"No data for today for {symbol}")
                return 'HOLD'
            

            
            # Get required prices
            try:
                open_price = today_data.iloc[0]['open']
                
                # Get price at 10 AM for first half-hour return
                price_10am = self._get_price_at_time(today_data, 10, 0)
                if price_10am is None:
                    print(f"Could not get 10 AM price for {symbol}")
                    return 'HOLD'
                
                # Get previous day's close
                yesterday_data = intraday_df[intraday_df['day'] < today]
                if not yesterday_data.empty:
                    prev_close = yesterday_data.iloc[-1]['close']
                else:
                    prev_close = open_price  # Fallback
                    
                current_price = today_data.iloc[-1]['close']
                current_vwap = today_data.iloc[-1]['vwap']
                
            except Exception as e:
                print(f"Error getting prices: {e}")
                return 'HOLD'
            
            # Calculate noise boundaries using paper's intraday sigma (matches training)
            intraday_sigma = data.get('intraday_sigma', {})

            # Get sigma for current time (round to nearest 15 minutes)
            minutes_rounded = (minutes_from_open // 15) * 15
            sigma = intraday_sigma.get(minutes_rounded, None)

            # Fallback to EGARCH if intraday sigma not available
            if sigma is None:
                print(f"Warning: No intraday sigma for {minutes_rounded} min, using EGARCH")
                sigma = data.get('egarch_vol', 0.01)

            # Use paper's formula: max/min to account for gap direction
            upper_bound = max(open_price, prev_close) * (1 + sigma)
            lower_bound = min(open_price, prev_close) * (1 - sigma)
            
            # Calculate first half-hour return (used as RF feature, not as filter)
            first_half_hour_return = (price_10am / open_price) - 1

            current_pos = data['current_position']
            
            print(f"\n{'='*50}")
            print(f"[{symbol}] Signal Generation at {current_time.strftime('%H:%M:%S')}")
            print(f"{'='*50}")

            print(f"\nPosition Status:")
            print(f"  Current position: {current_pos} shares")
            print(f"  Position type: {'LONG' if current_pos > 0 else 'SHORT' if current_pos < 0 else 'ZERO'}")
        
            
            print(f"\nPrice Analysis:")
            print(f"  Previous close: ${prev_close:.2f}")
            print(f"  Open price: ${open_price:.2f}")
            print(f"  Price at 10 AM: ${price_10am:.2f}")
            print(f"  Current price: ${current_price:.2f}")
            print(f"  VWAP: ${current_vwap:.2f}")
            
            print(f"\nNoise Boundaries:")
            print(f"  Intraday sigma (paper): {sigma:.4f} ({sigma*100:.2f}%)")
            print(f"  Upper bound: ${upper_bound:.2f}")
            print(f"  Lower bound: ${lower_bound:.2f}")
            
            print(f"\nFirst Half-Hour Return (RF Feature):")
            print(f"  {first_half_hour_return:.4f} ({first_half_hour_return*100:.2f}%)")

            # Anti-pump filter: Block buying if price too extended (mirrors FadeStrategy logic)
            # This only affects ENTRIES - exits still let winners run with trailing stop
            extreme_multiplier = 2  # Same as FadeStrategy
            reference_price = max(open_price, prev_close)
            extreme_upper = reference_price * (1 + sigma * extreme_multiplier)
            is_too_extended = current_price > extreme_upper

            print(f"\nAnti-Pump Filter (Entry Only):")
            print(f"  Reference price: ${reference_price:.2f}")
            print(f"  Extreme upper ({extreme_multiplier}x sigma): ${extreme_upper:.2f}")
            print(f"  Current vs extreme: {((current_price / extreme_upper) - 1)*100:+.2f}%")
            print(f"  Too extended: {is_too_extended}")
            if is_too_extended:
                print(f"  ⚠️  BLOCKING NEW ENTRY - Would chase pump at ${current_price:.2f} > ${extreme_upper:.2f}")
                print(f"  Note: Existing positions still trail upward with 0.5% stop")

            # Generate signals without regression filter (RF will learn when to use it)
            buy_signal_active = (current_price > upper_bound and
                           current_price > current_vwap and
                           not is_too_extended)  # Block new entries when too extended

            # Note: SELL signals won't be used (long-only strategy)
            sell_signal_active = (current_price < lower_bound and
                            current_price < current_vwap)
        
            print(f"\nSignal Status:")
            print(f"  BUY signal active: {buy_signal_active}")
            print(f"  SELL signal active: {sell_signal_active}")
        
        # PRIORITY 1: Handle position exits when signals are no longer valid
            if current_pos != 0:
                print(f"\nEvaluating exit conditions for existing position...")
            

            print(f"\nSignal Conditions (No Regression Filter):")
            print(f"  BUY Signal:")
            print(f"    - Price > Upper bound: {current_price > upper_bound}")
            print(f"    - Price > VWAP: {current_price > current_vwap}")
            print(f"    - SIGNAL: {'BUY' if buy_signal_active else 'NO'}")

            print(f"  SELL Signal (Not used - Long-Only):")
            print(f"    - Price < Lower bound: {current_price < lower_bound}")
            print(f"    - Price < VWAP: {current_price < current_vwap}")
            print(f"    - SIGNAL: {'SELL' if sell_signal_active else 'NO'} (ignored)")
            
            print(f"\nCurrent Position: {data['current_position']}")
            print(f"{'='*50}\n")
            
            # HANDLE EXITS FOR EXISTING POSITIONS
            if current_pos != 0:
                print(f"\nEvaluating exit conditions for existing position...")
                
                # Track entry price and high-water mark for trailing stop
                if 'entry_price' not in data or data['entry_price'] is None:
                    data['entry_price'] = current_price  # Initialize if missing
                    data['position_high'] = current_price  # Track highest price seen
                    data['position_low'] = current_price   # Track lowest price seen
                
                # Update high-water mark for long positions
                if current_pos > 0:
                    if current_price > data.get('position_high', current_price):
                        data['position_high'] = current_price
                        print(f"  📈 New position high: ${current_price:.2f}")
                
                # Update low-water mark for short positions
                elif current_pos < 0:
                    if current_price < data.get('position_low', current_price):
                        data['position_low'] = current_price
                        print(f"  📉 New position low: ${current_price:.2f}")
                
                # Calculate unrealized P&L
                if current_pos > 0:
                    pnl_pct = (current_price / data['entry_price']) - 1
                else:  # Short position
                    pnl_pct = (data['entry_price'] / current_price) - 1
                
                print(f"  💰 Position P&L: {pnl_pct*100:+.2f}%")
                
                # LONG POSITION EXITS
                if current_pos > 0:
                    # 1. PROFIT PROTECTION: Once you're up 1%, use tighter trailing stop
                    if pnl_pct > 0.01:  # 1% profit threshold
                        # Trail by 0.5% from the HIGH (not from entry, not from VWAP)
                        trailing_pct = 0.005  # 0.5% trailing stop when profitable
                        trailing_price = data['position_high'] * (1 - trailing_pct)
                        
                        print(f"  🎯 Profit mode: Trailing stop at ${trailing_price:.2f} (0.5% below high of ${data['position_high']:.2f})")
                        
                        if current_price < trailing_price:
                            print(f"*** PROFIT-TAKING EXIT LONG for {symbol} ***")
                            print(f"*** Price ${current_price:.2f} < Trailing stop ${trailing_price:.2f} ***")
                            print(f"*** Captured {pnl_pct*100:+.2f}% profit from high of ${data['position_high']:.2f} ***")
                            data['entry_price'] = None
                            data['position_high'] = None
                            return 'SELL'

                    # 2. STOP LOSS: Hard stop at -0.75% loss
                    elif pnl_pct < -0.0075:
                        print(f"*** STOP LOSS EXIT LONG for {symbol}: Down {pnl_pct*100:.2f}% ***")
                        data['entry_price'] = None
                        data['position_high'] = None
                        return 'SELL'

                    # 3. DYNAMIC SMART EXIT: Exit at the HIGHEST support level below current price
                    #    This is the key - use the most conservative exit that protects gains
                    else:
                        # Find the highest level that's below current price
                        support_levels = []

                        if upper_bound < current_price:
                            support_levels.append(('upper_bound', upper_bound))
                        if current_vwap < current_price:
                            support_levels.append(('VWAP', current_vwap))
                        if open_price < current_price:
                            support_levels.append(('open', open_price))

                        if support_levels:
                            # Use the HIGHEST support level (most conservative exit)
                            exit_level_name, exit_level = max(support_levels, key=lambda x: x[1])

                            print(f"  📊 Support levels below price:")
                            for name, level in support_levels:
                                print(f"    - {name}: ${level:.2f}")
                            print(f"  🎯 Using most conservative: {exit_level_name} at ${exit_level:.2f}")

                            if current_price < exit_level:
                                print(f"*** SMART EXIT LONG for {symbol} ***")
                                print(f"*** Price ${current_price:.2f} < {exit_level_name} ${exit_level:.2f} ***")
                                data['entry_price'] = None
                                data['position_high'] = None
                                return 'SELL'

                # 4. END-OF-DAY EXIT: Close all positions 10 minutes before close
                if current_pos > 0 and minutes_from_open >= 380:  # 6 hours 20 min = 3:50 PM
                    print(f"*** END-OF-DAY EXIT LONG for {symbol} ***")
                    data['entry_price'] = None
                    data['position_high'] = None
                    return 'SELL'
                
                # SHORT POSITION EXITS (mirror logic)
                elif current_pos < 0:
                    # Profit protection for shorts
                    if pnl_pct > 0.01:
                        trailing_pct = 0.005
                        trailing_price = data['position_low'] * (1 + trailing_pct)
                        
                        print(f"  🎯 Profit mode: Trailing stop at ${trailing_price:.2f} (0.5% above low of ${data['position_low']:.2f})")
                        
                        if current_price > trailing_price:
                            print(f"*** PROFIT-TAKING EXIT SHORT for {symbol} ***")
                            print(f"*** Price ${current_price:.2f} > Trailing stop ${trailing_price:.2f} ***")
                            print(f"*** Captured {pnl_pct*100:+.2f}% profit from low of ${data['position_low']:.2f} ***")
                            data['entry_price'] = None
                            data['position_low'] = None
                            return 'BUY'
                    
                    # Stop loss
                    elif pnl_pct < -0.0075:
                        print(f"*** STOP LOSS EXIT SHORT for {symbol}: Down {pnl_pct*100:.2f}% ***")
                        data['entry_price'] = None
                        data['position_low'] = None
                        return 'BUY'
                    
                    # Dynamic smart exit - find LOWEST resistance level above current price
                    else:
                        resistance_levels = []
                        
                        if lower_bound > current_price:
                            resistance_levels.append(('lower_bound', lower_bound))
                        if current_vwap > current_price:
                            resistance_levels.append(('VWAP', current_vwap))
                        if open_price > current_price:
                            resistance_levels.append(('open', open_price))
                        
                        if resistance_levels:
                            # Use the LOWEST resistance level (most conservative exit)
                            exit_level_name, exit_level = min(resistance_levels, key=lambda x: x[1])
                            
                            print(f"  📊 Resistance levels above price:")
                            for name, level in resistance_levels:
                                print(f"    - {name}: ${level:.2f}")
                            print(f"  🎯 Using most conservative: {exit_level_name} at ${exit_level:.2f}")
                            
                            if current_price > exit_level:
                                print(f"*** SMART EXIT SHORT for {symbol} ***")
                                print(f"*** Price ${current_price:.2f} > {exit_level_name} ${exit_level:.2f} ***")
                                data['entry_price'] = None
                                data['position_low'] = None
                                return 'BUY'
                    
                    # End of day
                    if minutes_from_open >= 380:
                        print(f"*** END-OF-DAY EXIT SHORT for {symbol} ***")
                        data['entry_price'] = None
                        data['position_low'] = None
                        return 'BUY'
        
            # HANDLE ENTRIES (only if flat)
            if current_pos == 0:  # Only enter new positions when flat
                print(f"\nEvaluating entry conditions (currently flat)...")
            
                # Generate BUY signal
                if buy_signal_active:
                    print(f"*** NEW LONG ENTRY for {symbol} ***")
                    data['entry_price'] = current_price
                    data['position_high'] = current_price
                    return 'BUY'
            
                # Generate SELL signal
                elif sell_signal_active:
                    print(f"*** NEW SHORT ENTRY for {symbol} ***")
                    data['entry_price'] = current_price
                    data['position_low'] = current_price
                    return 'SELL'
        
            else:
                print(f"\nSkipping new entries - already have position of {current_pos} shares")
            
            return 'HOLD'
            
        except Exception as e:
            print(f"Error generating signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return 'HOLD'
    
    def calculate_position_size(self, symbol, current_price):
        """
        Calculate position size based on volatility targeting approach from backtest.
        """
        try:
            # Get account value
            account_value = self.risk_manager.get_account_value()
            
            available_capital = account_value * self.portfolio_allocation
            
            # Get allocation for this symbol
            allocation = self.allocations.get(symbol, 0.33)  # Default 33% if not set
            
            # Get current volatility
            dvol = self.symbol_data[symbol].get('egarch_vol', 0.02)  # Default 2%
            
            # Calculate leverage based on volatility target
            vol_scalar = min(self.target_vol / dvol, self.max_leverage)
            
            # Calculate position size
            position_value = available_capital * allocation * vol_scalar
            shares = int(position_value / current_price)
            
            print(f"Position sizing for {symbol}:")
            print(f"  Account value: ${account_value:,.2f}")
            print(f"  Allocation: {allocation:.2%}")
            print(f"  Volatility: {dvol:.4f}")
            print(f"  Vol scalar: {vol_scalar:.2f}")
            print(f"  Position: {shares} shares (${position_value:,.2f})")
            
            return shares
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0
    
    # def execute_trade(self, symbol, signal):
    #     """Execute trade with enhanced position sizing"""
    #     if signal == 'HOLD':
    #         return
        
    #     # Get current price
    #     current_price = get_current_price(symbol)
    #     if current_price is None:
    #         print(f"Could not get current price for {symbol}")
    #         return
        
    #     # Calculate position size using volatility targeting
    #     quantity = self.calculate_position_size(symbol, current_price)
        
    #     if quantity == 0:
    #         print(f"Position size is 0 for {symbol}, skipping trade")
    #         return
        
    #     # Track current position
    #     data = self.symbol_data[symbol]
    #     if signal == 'BUY':
    #         data['current_position'] += quantity
    #     else:
    #         data['current_position'] -= quantity
        
    #     # Create contract and order
    #     contract = Contract()
    #     contract.symbol = symbol
    #     contract.secType = "STK"
    #     contract.currency = "USD"
    #     contract.exchange = "SMART"
        
    #     # Get primary exchange
    #     if hasattr(self.risk_manager, 'exchange_cache'):
    #         primary, _ = self.risk_manager.exchange_cache.get_exchange(symbol)
    #         contract.primaryExchange = primary
        
    #     order = Order()
    #     order.orderType = "MKT"
    #     order.totalQuantity = quantity
    #     order.action = signal
    #     order.eTradeOnly = False
    #     order.firmQuoteOnly = False
        
    #     # Place order
    #     if self.client.order_id:
    #         self.client._id += 1
    #         self.client.place(self.client.order_id, contract, order)
    #         print(f"ENHANCED STRATEGY: Placed {signal} order for {quantity} shares of {symbol}")
            
    #         # Record trade time
    #         data['trade_times'].append(datetime.datetime.now())
    #         data['last_signal_time'] = datetime.datetime.now()
    def execute_trade(self, symbol, signal):
        """Execute trade with enhanced position sizing matching the PDF strategy"""
        if signal == 'HOLD':
            return
        
        # Get current price
        current_price = get_current_price(symbol)
        if current_price is None:
            print(f"Could not get current price for {symbol}")
            return
        
        # Calculate position size using the STRATEGY's volatility-targeting method
        # This matches the PDF formula: AUM * allocation * min(leverage, target_vol/actual_vol)
        data = self.symbol_data[symbol]
        current_pos = data['current_position']
    
        print(f"\n{'='*40}")
        print(f"EXECUTING TRADE for {symbol}")
        print(f"Signal: {signal}")
        print(f"Current Position: {current_pos}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"{'='*40}")
        if current_pos == 0:
            try:
                # Get account value
                account_value = self.risk_manager.get_account_value()
                available_capital = account_value * self.portfolio_allocation
                
                # Get allocation for this symbol
                allocation = self.allocations.get(symbol, 0.2)  # Default 33% if not set
                
                # Get current volatility
                dvol = self.symbol_data[symbol].get('egarch_vol', 0.02)  # Default 2%
                
                # Calculate leverage based on volatility target (from PDF formula)
                vol_scalar = min(self.target_vol / dvol, self.max_leverage)
                
                # Calculate position size
                position_value = available_capital * allocation * vol_scalar
                quantity = int(position_value / current_price)
                
                print(f"Position sizing for {symbol}:")
                print(f"  Account value: ${account_value:,.2f}")
                print(f"  Allocation: {allocation:.2%}")
                print(f"  Volatility: {dvol:.4f}")
                print(f"  Vol scalar: {vol_scalar:.2f}")
                print(f"  Position: {quantity} shares (${position_value:,.2f})")
                
            except Exception as e:
                print(f"Error calculating position size: {e}")
                quantity = 0
        
        
        # Track current position
        if signal == 'BUY':
            if current_pos <= 0:  # Buying to open long or close short
                if current_pos < 0:
                    # Closing short position
                    data['current_position'] = 0
                    print(f"Closed short position of {abs(current_pos)} shares")
                else:
                    # Opening new long position
                    data['current_position'] = quantity
                    print(f"Opening long position of {quantity} shares")
            else:
                print(f"Warning: BUY signal but already long {current_pos} shares")
                return
        
        elif signal == 'SELL':
            print("a")
            if current_pos >= 0:  # Selling to open short or close long
                if current_pos > 0:
                    # Closing long position
                    quantity=data['current_position']
                    print(quantity)
                    data['current_position'] = 0
                    print(f"Closed long position of {current_pos} shares")
                else:
                    # Opening new short position
                    data['current_position'] = -quantity
                    print(f"Opening short position of {quantity} shares")
            else:
                print(f"Warning: SELL signal but already short {abs(current_pos)} shares")
                return
            
        if quantity == 0:
            print(f"Position size is 0 for {symbol}, skipping trade")
            return
        
        # Create contract and order
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        
        # Get primary exchange
        if hasattr(self.risk_manager, 'exchange_cache'):
            primary, _ = self.risk_manager.exchange_cache.get_exchange(symbol)
            contract.primaryExchange = primary
        
        order = Order()
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.action = signal
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        # Place order
        if self.client.order_id:
            self.client.order_id += 1
            self.client.placeOrder(self.client.order_id, contract, order)
            print(f"ENHANCED STRATEGY: Placed {signal} order for {quantity} shares of {symbol}")
            
            # Record trade time
            data['trade_times'].append(datetime.datetime.now())
            data['last_signal_time'] = datetime.datetime.now()

# Client for connecting to Interactive Brokers
class PTLClient(EWrapper, EClient):
     
    def __init__(self, host, port, client_id):
        EClient.__init__(self, self)
        self.positions = {}
        self.callback_handlers = []
        self.connect(host, port, client_id)
        thread = Thread(target=self.run)
        thread.start()
        self._is_connected = False
        self._connection_timeout = time.time() + 10

    def add_callback_handler(self, handler):
        if handler not in self.callback_handlers:
            self.callback_handlers.append(handler)

    def updateAccountValue(self, key, value, currency, accountName):
        for handler in self.callback_handlers:
            if hasattr(handler, 'updateAccountValue'):
                handler.updateAccountValue(key, value, currency, accountName)
        print(f"PTLClient received account update: {key} = {value}")

    def accountDownloadEnd(self, accountName):
        for handler in self.callback_handlers:
            if hasattr(handler, 'accountDownloadEnd'):
                handler.accountDownloadEnd(accountName)
        print(f"PTLClient account download ended for {accountName}")


    def error(self, req_id, code, msg, misc=None):
        if code in [2104, 2106, 2158]:
            print(msg)
        else:
            print('Error {}: {}'.format(code, msg))


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.order_id = orderId
        print(f"next valid id is {self.order_id}")
        self._is_connected = True
        try:
            self.reqMarketDataType(1)
            print("Successfully requested real-time data")
        except Exception as e:
            print(f"ERROR requesting real-time data: {e}")
            print("Falling back to delayed data...")
            try:
                self.reqMarketDataType(3)
            except Exception as e2:
                print(f"ERROR setting delayed data: {e2}")
        return self.order_id

    # callback when historical data is received from Interactive Brokers
    # def historicalData(self, req_id, bar):
    #     """Callback for when historical data is received from Interactive Brokers"""
    #     print(f"Raw bar.date: {bar.date}, type: {type(bar.date)}, str: '{str(bar.date)}'")
    
    #     try:
    #         date_str = str(bar.date)
            
    #         # Check if it's a datetime string with time component
    #         if ' ' in date_str and ':' in date_str and len(date_str) >= 15:
    #             # Format: "20240530 09:30:00"
    #             t = datetime.datetime.strptime(date_str, '%Y%m%d %H:%M:%S')
                
    #         # Check if it's just a date string (common for daily bars)
    #         elif len(date_str) == 8 and date_str.isdigit():
    #             # Format: "20240530"
    #             t = datetime.datetime.strptime(date_str, '%Y%m%d')
                
    #         else:
    #             # It might be a Unix timestamp
    #             timestamp = float(bar.date)
                
    #             # Check if it's a reasonable Unix timestamp (after year 2000)
    #             # Unix timestamp for Jan 1, 2000 is 946684800
    #             if timestamp > 946684800:
    #                 t = datetime.datetime.fromtimestamp(timestamp)
    #             else:
    #                 # It might be days since 1970-01-01 (common for daily data)
    #                 # or some other format
    #                 print(f"Unusual timestamp value: {timestamp}")
                    
    #                 # Try interpreting as days since epoch
    #                 days_since_epoch = int(timestamp)
    #                 t = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=days_since_epoch)
                    
    #                 # If that gives us a date in 1970s, something is wrong
    #                 if t.year < 2000:
    #                     print(f"Warning: Parsed date {t} seems incorrect for bar.date={bar.date}")
    #                     # Default to current date as fallback
    #                     t = datetime.datetime.now()
            
    #         data = {
    #             'date': t,
    #             'open': bar.open,
    #             'high': bar.high,
    #             'low': bar.low,
    #             'close': bar.close,
    #             'volume': int(bar.volume)
    #         }
            
    #         # Add debug output to verify dates
    #         if req_id % 10 == 0:  # Only print every 10th bar to avoid spam
    #             print(f"Parsed date: {t} from bar.date: {bar.date}")
            
    #         # Put the data into the queue
    #         data_queue.put(data)
            
    #     except Exception as e:
    #         print(f"Error processing bar data: {e}")
    #         print(f"bar.date value: {bar.date}, type: {type(bar.date)}")
    #         import traceback
    #         traceback.print_exc()
    
   
    
    def historicalData(self, req_id, bar):
        """Callback for when historical data is received from Interactive Brokers"""
        # print(f"Raw bar.date: {bar.date}, type: {type(bar.date)}")
        
        try:
            date_str = str(bar.date).strip()
            
            # Method 1: Check if it's a formatted datetime string with time component
            if ' ' in date_str and ':' in date_str and len(date_str) >= 15:
                try:
                    # Format: "20240530 09:30:00" or "2024-05-30 09:30:00"
                    if '-' in date_str:
                        t = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        t = datetime.datetime.strptime(date_str, '%Y%m%d %H:%M:%S')
                    #print(f"Parsed as formatted datetime: {t}")
                    
                except ValueError:
                    # Try other common formats
                    try:
                        t = datetime.datetime.strptime(date_str, '%Y%m%d  %H:%M:%S')  # Double space
                    except ValueError:
                        raise ValueError(f"Could not parse datetime string: {date_str}")
            
            # Method 2: Check if it's just a date string (YYYYMMDD format for daily bars)
            elif len(date_str) == 8 and date_str.isdigit():
                t = datetime.datetime.strptime(date_str, '%Y%m%d')
                #print(f"Parsed as date string: {t}")
            
            # Method 3: Handle numeric timestamps
            else:
                try:
                    timestamp = float(bar.date)
                    print(f"Processing numeric timestamp: {timestamp}")
                    
                    # Check if it's a reasonable Unix timestamp (seconds since 1970-01-01)
                    # Unix timestamp for Jan 1, 2020 is 1577836800
                    # Unix timestamp for Jan 1, 2030 is 1893456000
                    if 1577836800 <= timestamp <= 1893456000:
                        # This looks like a valid recent Unix timestamp
                        t = datetime.datetime.fromtimestamp(timestamp)
                        print(f"Parsed as Unix timestamp (seconds): {t}")
                    
                    # Check if it might be milliseconds (divide by 1000)
                    elif 1577836800000 <= timestamp <= 1893456000000:
                        t = datetime.datetime.fromtimestamp(timestamp / 1000)
                        print(f"Parsed as Unix timestamp (milliseconds): {t}")
                    
                    # For very small numbers, this might be days since a different epoch
                    # Interactive Brokers sometimes uses different epoch references
                    elif 0 <= timestamp <= 50000:  # Reasonable range for days since some epoch
                        print(f"Warning: Small timestamp value {timestamp} - this might be days since an IB-specific epoch")
                        
                        # Try different epoch references that IB might use
                        # Option 1: Days since 1900-01-01 (Excel-style)
                        try:
                            base_date = datetime.datetime(1900, 1, 1)
                            t = base_date + datetime.timedelta(days=int(timestamp))
                            if 2020 <= t.year <= 2030:  # Reasonable modern date
                                print(f"Parsed as days since 1900-01-01: {t}")
                            else:
                                raise ValueError("Date out of reasonable range")
                        except:
                            # Option 2: Days since 1970-01-01 (but this usually gives 1970s dates)
                            # Option 3: Try using the current date as fallback
                            print(f"Could not parse timestamp {timestamp}, using current time")
                            t = datetime.datetime.now()
                    else:
                        print(f"Timestamp {timestamp} is outside expected ranges, using current time")
                        t = datetime.datetime.now()
                        
                except (ValueError, TypeError) as e:
                    print(f"Error converting to float: {e}, using current time")
                    t = datetime.datetime.now()
            
            # Create the data dictionary
            data = {
                'date': t,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': int(bar.volume)
            }
            
            # Verify the date makes sense before adding to queue
            if t.year < 2020 or t.year > 2030:
                print(f"Warning: Parsed date {t} seems unreasonable for recent market data")
                # You might want to skip this data point or use a fallback
                return
            
            # Add debug output for verification
            if req_id % 10 == 0:  # Print every 10th bar to avoid spam
                print(f"Successfully parsed: {t} from raw: {bar.date}")
            
            # Put the data into the queue
            #self.debug_timestamp_formats(req_id, bar)
            data_queue.put(data)
            
        except Exception as e:
            print(f"Critical error processing bar data: {e}")
            print(f"bar.date value: '{bar.date}', type: {type(bar.date)}")
            import traceback
            traceback.print_exc()
            # Don't add invalid data to the queue

    # callback when all historical data has been received
    def historicalDataEnd(self, reqId, start, end):
        print(f"end of data {start} {end}")
            
        # we can update the chart once all data has been received
        update_chart()


    # callback to log order status, we can put more behavior here if needed
    def orderStatus(self, order_id, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print(f"order status {order_id} {status} {filled} {remaining} {avgFillPrice}")    


    # callback for when a scan finishes
    def scannerData(self, req_id, rank, details, distance, benchmark, projection, legsStr):
        super().scannerData(req_id, rank, details, distance, benchmark, projection, legsStr)
        print("got scanner data")
        print(details.contract)

        data = {
            'secType': details.contract.secType,
            'secId': details.contract.secId,
            'exchange': details.contract.primaryExchange,
            'symbol': details.contract.symbol
        }

        print(data)
        
        # Put the data into the queue
        scanner_data_queue.put(data)


# called by charting library when the
def get_bar_data(symbol, timeframe):
    print(f"\nStarting historical data request:")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    
    # 1. Clear any existing data first
    try:
        data_queue.queue.clear()
        print("Cleared existing data queue")
    except Exception as e:
        print(f"Warning: Could not clear data queue: {e}")
    
    # 2. Set up contract
    try:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        print(f"Contract created: {contract.symbol} {contract.secType}")
    except Exception as e:
        print(f"Error creating contract: {e}")
        return
        
    # 3. Set up request parameters
    what_to_show = 'TRADES'
    duration = '7 D'  # Reduced from 30D to 7D to reduce potential data load issues
    req_id = int(time.time()) % 10000  # Unique request ID
    
    print(f"Request parameters:")
    print(f"- Request ID: {req_id}")
    print(f"- Duration: {duration}")
    print(f"- What to show: {what_to_show}")
    
    # 4. Show loading indicator
    try:
        chart.spinner(True)
        print("Started loading spinner")
    except Exception as e:
        print(f"Warning: Could not show spinner: {e}")
    
    # 5. Make the historical data request
    try:
        client.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime='',  # Empty string means current time
            durationStr=duration,
            barSizeSetting=timeframe,
            whatToShow=what_to_show,
            useRTH=1,  # Regular trading hours only
            formatDate=1,  # Use UNIX timestamp format for intraday data
            keepUpToDate=True,
            chartOptions=[]
        )
        print("Historical data request sent successfully")
        client.reqPositions()
        # Wait for data to start coming in
        time.sleep(1)
        
    except Exception as e:
        print(f"Error requesting historical data: {e}")
        chart.spinner(False)
        return
        
    # 6. Set the chart watermark
    try:
        chart.watermark(symbol)
        print("Chart watermark set")
    except Exception as e:
        print(f"Warning: Could not set watermark: {e}")
    


# handler for the screenshot button
def take_screenshot(key):
    os.makedirs('screenshots', exist_ok=True)
    img = chart.screenshot()
    t = time.time()
    chart_filename = f"screenshots/screenshot-{t}.png"
    analysis_filename = f"screenshots/screenshot-{t}.md"

    with open(chart_filename, 'wb') as f:
        f.write(img)

    analysis = analyze_chart(chart_filename)

    print(analysis)

    with open(analysis_filename, "w") as text_file:
        text_file.write(analysis)


# handles when the user uses an order hotkey combination
def place_order(key):
    symbol = chart.topbar['symbol'].value
    
    # Get current market data
    current_price = get_current_price(symbol)  # You'll need to implement this
    #volatility = calculate_volatility(symbol)   # You'll need to implement this
   #print(f"Current price: {current_price}, Volatility: {volatility}")
    
    # Calculate position size based on risk
    # risk_manager = RiskManager(client)  # Create an instance
    # client.add_callback_handler(risk_manager)  # Add the risk manager as a callback handler
    # quantity = risk_manager.calculate_position_size(symbol, current_price)
    quantity = global_trading_system.risk_manager.calculate_position_size(symbol, current_price)
    
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.currency = "USD"
    contract.exchange = "SMART"
    contract.primaryExchange='NYSE'
    
    order = Order()
    order.orderType = "MKT"
    order.totalQuantity = quantity
    order.eTradeOnly=False
    order.firmQuoteOnly=False
    
    # Set action based on key pressed
    if key == 'O':
        order.action = "BUY"
    elif key == 'P':
        order.action = "SELL"
    
    # Validate order before placing
    is_valid, reason = global_trading_system.risk_manager.validate_order(
        symbol, order.action, quantity, current_price
    )
    
    if not is_valid:
        print(f"Order rejected: {reason}")
        return
        
    # Place the order if valid
    if client.order_id:
        client.order_id=client.nextValidId(client.order_id)
        client.placeOrder(client.order_id, contract, order)
        print(f"Placed {order.action} order for {quantity} shares of {symbol}")


# implement an Interactive Brokers market scanner
def do_scan(scan_code):
    scannerSubscription = ScannerSubscription()
    scannerSubscription.instrument = "STK"
    scannerSubscription.locationCode = "STK.US.MAJOR"
    scannerSubscription.scanCode = scan_code

    tagValues = []
    tagValues.append(TagValue("optVolumeAbove", "1000"))
    tagValues.append(TagValue("avgVolumeAbove", "10000"))

    client.reqScannerSubscription(7002, scannerSubscription, [], tagValues)
    time.sleep(1)

    display_scan()

    client.cancelScannerSubscription(7002)


#  get new bar data when the user enters a different symbol
def on_search(chart, searched_string):
    get_bar_data(searched_string, chart.topbar['timeframe'].value)
    chart.topbar['symbol'].set(searched_string)


# get new bar data when the user changes timeframes
def on_timeframe_selection(chart):
    print("selected timeframe")
    print(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    

# callback for when the user changes the position of the horizontal line
def on_horizontal_line_move(chart, line):
    print(f'Horizontal line moved to: {line.price}')


# called when we want to render scan results
def display_scan():
    # function to call when one of the scan results is clicked
    def on_row_click(row):
        chart.topbar['symbol'].set(row['symbol'])
        get_bar_data(row['symbol'], '5 mins')

    # create a table on the UI, pass callback function for when a row is clicked
    table = chart.create_table(
                    width=0.4, 
                    height=0.5,
                    headings=('symbol', 'value'),
                    widths=(0.7, 0.3),
                    alignments=('left', 'center'),
                    position='left', func=on_row_click
                )

    # poll queue for any new scan results
    try:
        while True:
            data = scanner_data_queue.get_nowait()
            # create a new row in the table for each scan result
            table.new_row(data['symbol'], '')
    except queue.Empty:
        print("empty queue")
    finally:
        print("done")

def set_marker_data(self, marker_id, data):
    """Helper method to set data for a marker by ID"""
    # Implementation will depend on how your chart library works
    if hasattr(self, 'markers') and marker_id in self.markers:
        self.markers[marker_id].set_data(data)
    else:
        # Try direct approach
        self.execute_js(f"setMarkerData('{marker_id}', {data.to_json(orient='records')})")
# called when we want to update what is rendered on the chart 
def update_chart():
    """
    Updates the chart with new data while properly handling element removal and updates.
    Uses the correct methods for removing and updating chart elements.
    """
    global current_df, current_lines, chart
    
    try:
        # Show loading indicator while we update
        chart.spinner(True)
        time.sleep(0.2)  # Give spinner time to appear
        
        # Store current lines in a local variable before clearing them
        lines_to_remove = current_lines.copy() if current_lines else []
        current_lines = []  # Reset the global list
        
        # Clear the chart by setting empty data
        try:
            chart.set(pd.DataFrame())
            time.sleep(0.3)  # Wait for clear to complete
            
            # Remove existing lines one by one
            for line in lines_to_remove:
                try:
                    # Check if it's a HorizontalLine
                    if hasattr(line, 'remove'):
                        line.remove()  # Use remove() instead
                    elif hasattr(line, 'set'):
                        line.set(pd.DataFrame())
                    else:
                        # Fallback - try to access the underlying object
                        chart.remove_component(line)
                except Exception as e:
                    print(f"Warning while removing line: {e}")
            
        except Exception as e:
            print(f"Warning during cleanup: {e}")

        # Collect all available data from the queue
        bars = []
        try:
            while True:
                data = data_queue.get_nowait()
                bars.append(data)
        except queue.Empty:
            pass

        # Check if we have data to display
        if not bars:
            print("No data available for chart update")
            chart.spinner(False)
            return

        # Create dataframe from collected bars
        df = pd.DataFrame(bars)
        if df.empty:
            print("Empty dataframe created, skipping update")
            chart.spinner(False)
            return
        
        current_df = df.copy()

        # Update main chart data
        time.sleep(0.2)  # Small delay before setting new data
        try:
            chart.set(df)
            time.sleep(0.3)  # Wait for chart to process new data
        except Exception as e:
            print(f"Error updating main chart data: {e}")
            chart.spinner(False)
            return
        
        

        # Only add indicators if we have enough data points
        if len(df) > 20:  # Ensure we have enough data for meaningful SMAs
            # Add moving averages
            sma_windows = [5, 10, 20, 50, 100]
            
            for window in sma_windows:
                try:
                    # Create a new line for each SMA
                    line = chart.create_line(name=f'SMA {window}')
                    
                    # Calculate and format SMA data
                    sma_data = pd.DataFrame({
                        'time': df['date'],
                        f'SMA {window}': df['close'].rolling(window=window).mean()
                    }).dropna()
                    
                    # Add small delay between adding indicators
                    time.sleep(0.2)
                    
                    # Set the line data
                    line.set(sma_data)
                    
                    # Keep track of created line
                    current_lines.append(line)
                    
                except Exception as e:
                    print(f"Error adding SMA-{window} line: {e}")
                    continue

            # Add horizontal line at the highest price
            try:
                horizontal = chart.horizontal_line(
                    df['high'].max(), 
                    func=on_horizontal_line_move
                )
                current_lines.append(horizontal)
            except Exception as e:
                print(f"Error adding horizontal line: {e}")
                
            if global_trading_strategy:
                try:
                    #add_trading_signals_to_chart(chart, df, global_trading_strategy)
                    print("Trading signals added to chart")
                except Exception as e:
                    print(f"Error adding trading signals: {e}")

    except Exception as e:
        print(f"Critical error in chart update: {e}")
        
    finally:
        # Always ensure the spinner is removed
        try:
            chart.spinner(False)
        except Exception as e:
            print(f"Error removing spinner: {e}")
            
def add_trading_signals_to_chart(chart, df, strategy):
    """Add buy/sell signals to the chart"""
    if len(df) < strategy.long_window:
        return
    
    # Calculate SMAs
    short_sma = strategy.calculate_ema(df, strategy.short_window)
    long_sma = strategy.calculate_ema(df, strategy.long_window)
    
    # Find crossover points
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(df)):
        if i >= strategy.long_window:
            # Check for golden cross (buy signal)
            if short_sma.iloc[i-1] <= long_sma.iloc[i-1] and short_sma.iloc[i] > long_sma.iloc[i]:
                buy_signals.append({'time': df['date'].iloc[i], 'price': df['low'].iloc[i] * 0.999})
            
            # Check for death cross (sell signal)
            elif short_sma.iloc[i-1] >= long_sma.iloc[i-1] and short_sma.iloc[i] < long_sma.iloc[i]:
                sell_signals.append({'time': df['date'].iloc[i], 'price': df['high'].iloc[i] * 1.001})
    
    # Add markers to chart
    if buy_signals:
        try:
            buy_df = pd.DataFrame(buy_signals)
            marker_id = chart.marker(text='▲', position='below', shape='up', color='#26a69a')
        
            # Check if it's a string (ID) or object
            if isinstance(marker_id, str):
                # Use the chart to set the data for this marker ID
                chart.set_marker_data(marker_id, buy_df)
            else:
                # It's an object with set method
                marker_id.set(buy_df)
                print(f"Added {len(buy_signals)} buy signals to chart")
        except Exception as e:
            print(f"Error adding buy markers: {e}")
    
    if sell_signals:
        try:
            sell_df = pd.DataFrame(sell_signals)
            sell_marker = chart.marker(text='▼', position='above', shape='down', color='#ef5350')
            sell_marker.set(sell_df)
            print(f"Added {len(sell_signals)} sell signals to chart")
        except Exception as e:
            print(f"Error adding sell markers: {e}")
            
def start_automated_trading(client, symbols=None):
    """Start the automated trading system"""
    global global_trading_system, global_trading_strategy
    
    trading_system = TradingSystem(client)
    global_trading_system = trading_system
    global_trading_strategy = trading_system.trading_strategy
    
    
    # Set monitored symbols if provided
    if symbols:
        trading_system.monitored_symbols = symbols
    
    # Start the trading system in a separate thread
    trading_thread = Thread(target=trading_system.run)
    trading_thread.daemon = True  # Allow program to exit even if thread is running
    trading_thread.start()
    
    print(f"Automated trading started for symbols: {trading_system.monitored_symbols}")
    return trading_system

if __name__ == '__main__':
    # create a client object
    client = PTLClient(DEFAULT_HOST, TRADING_PORT, DEFAULT_CLIENT_ID)
    
    
    symbols = ['NVDA', 'AAPL', 'JPM', 'GOOG', 'WMT', 'QQQ', 'SPY']
    trading_system = start_automated_trading(client, symbols=symbols)
    print("Initializing Random Forest model...")
    trading_system.trading_strategy.initialize_rf_with_full_history(
        symbols=symbols,
        total_days=250  # About 250 days of trading data
    )
    
    
    # create chart object, specify display settings
    chart = Chart(toolbox=True, width=1000, inner_width=0.6, inner_height=1)

    # hotkey to place a buy order
    chart.hotkey('shift', 'O', place_order)

    # hotkey to place a sell order
    chart.hotkey('shift', 'P', place_order)

    chart.legend(True)
    
    # set up a function to call when searching for symbol
    chart.events.search += on_search

    # set up top bar
    chart.topbar.textbox('symbol', INITIAL_SYMBOL)

    # give ability to switch between timeframes
    chart.topbar.switcher('timeframe', ('1 min', '5 mins', '15 mins', '1 hour'), default='5 mins', func=on_timeframe_selection)

    # populate initial chart
    get_bar_data(INITIAL_SYMBOL, '5 mins')

    # run a market scanner
    do_scan("HOT_BY_VOLUME")

    # create a button for taking a screenshot of the chart
    chart.topbar.button('screenshot', 'Screenshot', func=take_screenshot)

    # show the chart
    chart.show(block=True)