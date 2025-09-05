import inspect
import json
import random
import threading
import time, datetime
import queue
import pandas as pd
import numpy as np
import os
from threading import Thread
from lightweight_charts import Chart
from gpt4o_technical_analyst import analyze_chart
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.client import Contract, Order, ScannerSubscription
from ibapi.tag_value import TagValue
import sqlite3
import pytz
from arch import arch_model
from sklearn.linear_model import LinearRegression

# create a queue for data coming from Interactive Brokers API
data_queue = queue.Queue()
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
    
#     
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
        with self.cache_lock:
            # Check cache first
            if symbol in self.cache:
                return self.cache[symbol]['primary'], self.cache[symbol]['exchange']
            

            primary, exchange = get_correct_exchange(symbol, self.client)
            

            self.cache[symbol] = {
                'primary': primary,
                'exchange': exchange,
                'last_updated': time.time()
            }
            
            self.save_cache()
            
            return primary, exchange

class TradingSystem:
    def __init__(self, existing_client):  
        """
        Initialize the trading system using an existing client connection
        rather than creating a new one
        """
        self.client = existing_client  
        self.exchange_cache = ExchangeCache(existing_client)
        print(f"DEBUG: TradingSystem created exchange_cache: {self.exchange_cache}")  
        print(f"DEBUG: exchange_cache type: {type(self.exchange_cache)}")
        self.db_conn = self.init_database()
        self.db_lock = threading.Lock()
        self.trading_day = None
        self.risk_manager = RiskManager(existing_client, self.exchange_cache)
        print(f"DEBUG: Passing exchange_cache to RiskManager: {self.exchange_cache}")
        self.client.add_callback_handler(self.risk_manager) 
        self.trading_strategy = TradingStrategy(self.risk_manager, existing_client)
        self.is_market_open = False
        self.monitored_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
    def init_database(self):
        conn = sqlite3.connect('trading_system.db', check_same_thread=False)
        cursor = conn.cursor()
        
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
        for symbol in self.monitored_symbols:
            try:
                df = get_recent_bars(symbol, '1 min', '2 D', for_calculation=True, client_instance=self.client, exchange_cache=self.exchange_cache)
            
                if df.empty:
                    print(f"No data available for {symbol}")
                    continue
                
                signal = self.trading_strategy.generate_signal(symbol, df)
            
            # Execute trade if signal is generated
                if signal != 'HOLD':
                    print(f"Signal for {symbol}: {signal}")
                    self.trading_strategy.execute_trade(symbol, signal)
                
            except Exception as e:
                print(f"Error checking signals for {symbol}: {e}")

    def run(self):
        while True:
            current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
            
            if self.is_trading_day(current_time):
                if self.trading_day != current_time.date():
                    self.start_trading_day(current_time.date())
                
                if self.is_market_hours(current_time):
                    self.is_market_open = True
                    self.update_positions()  
                    
                    if current_time.minute % 1 == 0:  
                        self.check_trading_signals()
                        
                else:
                    if self.is_market_open:  # Market just closed
                        self.end_trading_day()
                        self.is_market_open = False
            
            time.sleep(60)  # Update every minute

    def update_positions(self):
        self.client.reqAccountUpdates(True, "")
        
    def updateAccountValue(self, key, value, currency, accountName):
        if key == "RealizedPnL":
            self.update_realized_pnl(float(value))
        elif key == "UnrealizedPnL":
            self.update_unrealized_pnl(float(value))

    def update_realized_pnl(self, pnl):
        with self.db_lock:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                UPDATE daily_pnl 
                SET realized_pnl = ?, total_pnl = realized_pnl + unrealized_pnl
                WHERE date = ?
            ''', (pnl, self.trading_day.strftime('%Y-%m-%d')))
            self.db_conn.commit()

    def start_trading_day(self, date):
        self.trading_day = date
        
        with self.db_lock:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO daily_pnl (date, realized_pnl, unrealized_pnl, total_pnl)
                VALUES (?, 0, 0, 0)
            ''', (date.strftime('%Y-%m-%d'),))
            self.db_conn.commit()
        
        self.update_positions()

    def is_trading_day(self, dt):
        if dt.weekday() in (5, 6): 
            return False
            
        return True

    def is_market_hours(self, dt):
        market_open = dt.replace(hour=9, minute=30, second=0)
        market_close = dt.replace(hour=16, minute=0, second=0)
        return market_open <= dt <= market_close
    
def get_correct_exchange(symbol, client_instance):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.currency = 'USD'
    contract.exchange = 'SMART' 
    
    exchange_info = {'primary': None, 'valid_exchanges': None}
    request_complete = threading.Event()
    
    original_contract_details = client_instance.contractDetails
    original_contract_details_end = client_instance.contractDetailsEnd
    
    def temp_contract_details(reqId, details):
        exchange_info['primary'] = details.contract.primaryExchange
        exchange_info['valid_exchanges'] = details.validExchanges
        print(f"Found details for {symbol}: Primary={details.contract.primaryExchange}, Valid={details.validExchanges}")
    
    def temp_contract_details_end(reqId):
        request_complete.set()
    
    try:
        client_instance.contractDetails = temp_contract_details
        client_instance.contractDetailsEnd = temp_contract_details_end
        
        req_id = int(time.time() * 1000) % 10000
        client_instance.reqContractDetails(req_id, contract)
        
        if not request_complete.wait(timeout=10):
            print(f"Timeout waiting for contract details for {symbol}")
            return 'SMART', 'SMART'  
        
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
    

def get_recent_bars(symbol, timeframe, duration, for_calculation=False, client_instance=None, exchange_cache=None):
    print(f"\nDEBUG: Data request details:")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Duration: {duration}")
    print(f"Called by: {inspect.stack()[1].function}")
    
    if exchange_cache:
        primary_exchange, exchange = exchange_cache.get_exchange(symbol)
    else:
        # Default to common US exchange values
        primary_exchange = 'NYSE'
        exchange = 'SMART'
        print(f"Warning: No exchange_cache provided for {symbol}, using defaults")
    
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
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        contract.primaryExchange = primary_exchange
        
        # Make the request
        client_to_use.reqHistoricalData(
            req_id, contract, '', duration, timeframe, 'TRADES',
            1, 1, False, []
        )
        
        # Collect the data with improved timing
        bars = []
        first_timeout = time.time() + 10  # Increased timeout 
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
        self.max_position_size = 15000 
        self.max_daily_loss = 500      
        self.position_sizing_pct = 0.05   
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
            return 30000  # Default value
        
    
    

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
        
class TradingStrategy:
    def __init__(self, risk_manager, client):
        self.risk_manager = risk_manager
        self.client = client
        self.short_window = 10  # Short-term SMA
        self.long_window = 50   # Long-term SMA
        self.current_position = {}  # Track positions per symbol
        self.last_signal = {}       # Track last signal per symbol
        
    def calculate_ema(self, df, window):
        """Calculate Simple Moving Average"""
        return df['close'].ewm(span=window, adjust=False).mean()
    
    def generate_signal(self, symbol, df):
        """
        Generate trading signal based on EMA crossover
        Returns: 'BUY', 'SELL', or 'HOLD'
        """
        if len(df) < self.long_window:
            return 'HOLD'
            
        # Calculate SMAs
        short_sma = self.calculate_ema(df, self.short_window)
        long_sma = self.calculate_ema(df, self.long_window)
        
        # Get the last two values for crossover detection
        short_values = short_sma.tail(3).values
        long_values = long_sma.tail(3).values
        
        # Check for patterns rather than just two points
        if (short_values[0] <= long_values[0] and 
            short_values[1] <= long_values[1] and 
            short_values[2] > long_values[2]):
            print(short_values, long_values)
            print(f"BUY signal for {symbol}")
            return 'BUY'
        elif (short_values[0] >= long_values[0] and 
            short_values[1] >= long_values[1] and 
            short_values[2] < long_values[2]):
            print(short_values, long_values)
            print(f"SELL signal for {symbol}")
            return 'SELL'
        else:
            print(short_values, long_values)
            print(f"HOLD signal for {symbol}")
            return 'HOLD'
    
    def execute_trade(self, symbol, signal):
        """Execute trade based on signal"""
        if signal == 'HOLD':
            return
            
        # Get current price and volatility
        current_price = get_current_price(symbol)
 #       volatility = calculate_volatility(symbol)
        
        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(symbol, current_price)
        
        # Validate order
        is_valid, reason = self.risk_manager.validate_order(
            symbol, signal, quantity, current_price
        )
        
        if not is_valid:
            print(f"Order rejected: {reason}")
            return
            
        # Create contract and order
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        contract.primaryExchange = 'NYSE'
        
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
            print(f"AUTO-TRADE: Placed {signal} order for {quantity} shares of {symbol}")
            
            # Update tracking
            self.current_position[symbol] = quantity if signal == 'BUY' else -quantity
            self.last_signal[symbol] = signal

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
    def historicalData(self, req_id, bar):
        """Callback for when historical data is received from Interactive Brokers"""
    
        try:
        # IBKR might send timestamps in different formats depending on the timeframe
        # For intraday data, it's usually a Unix timestamp
        # For daily data, it's usually a string like "20241227 09:30:00"
        
            if ' ' in str(bar.date):  # Check if it's a date string format
            # Parse datetime string format
                t = datetime.datetime.strptime(str(bar.date), '%Y%m%d %H:%M:%S')
            else:
            # Handle Unix timestamp format
                t = datetime.datetime.fromtimestamp(int(float(bar.date)))
        
        
            data = {
            'date': t,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': int(bar.volume)
        }
        
        # Put the data into the queue
            data_queue.put(data)

        
        except Exception as e:
            print(f"Error processing bar data: {e}")
            print(f"Error occurred with bar.date: {bar.date}, type: {type(bar.date)}")


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
        data_queue.put(data)


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
            data = data_queue.get_nowait()
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
                    add_trading_signals_to_chart(chart, df, global_trading_strategy)
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
    
    trading_system = start_automated_trading(client, symbols=['NVDA'])

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
