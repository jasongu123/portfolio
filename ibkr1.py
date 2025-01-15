import inspect
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

# create a queue for data coming from Interactive Brokers API
data_queue = queue.Queue()
calc_queue = queue.Queue()  # New queue for calculations

# a list for keeping track of any indicator lines
current_lines = []

# initial chart symbol to show
INITIAL_SYMBOL = "DJT"

# settings for live trading vs. paper trading mode
LIVE_TRADING = False
LIVE_TRADING_PORT = 7496
PAPER_TRADING_PORT = 7497
TRADING_PORT = PAPER_TRADING_PORT
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
        

class TradingSystem:
    def __init__(self, existing_client):  # Accept the existing client as a parameter
        """
        Initialize the trading system using an existing client connection
        rather than creating a new one
        """
        self.client = existing_client  # Use the existing client
        self.db_conn = self.init_database()
        self.trading_day = None
        self.risk_manager = RiskManager()
        self.is_market_open = False
        
    def init_database(self):
        """Creates or connects to a database to persist trading data"""
        conn = sqlite3.connect('trading_system.db')
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

    def run(self):
        """Main loop for the trading system"""
        while True:
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            
            # Check if it's a trading day
            if self.is_trading_day(current_time):
                if self.trading_day != current_time.date():
                    self.start_trading_day(current_time.date())
                
                if self.is_market_hours(current_time):
                    self.is_market_open = True
                    self.update_positions()  # Update position values
                else:
                    if self.is_market_open:  # Market just closed
                        self.end_trading_day()
                        self.is_market_open = False
            
            time.sleep(60)  # Update every minute

    def update_positions(self):
        """Updates position values and PnL from IBKR"""
        # Request account updates from IBKR
        self.client.reqAccountUpdates(True, "")
        
    def updateAccountValue(self, key, value, currency, accountName):
        """Callback from IBKR with account information"""
        if key == "RealizedPnL":
            self.update_realized_pnl(float(value))
        elif key == "UnrealizedPnL":
            self.update_unrealized_pnl(float(value))

    def update_realized_pnl(self, pnl):
        """Updates realized PnL in database"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            UPDATE daily_pnl 
            SET realized_pnl = ?, total_pnl = realized_pnl + unrealized_pnl
            WHERE date = ?
        ''', (pnl, self.trading_day.strftime('%Y-%m-%d')))
        self.db_conn.commit()

    def start_trading_day(self, date):
        """Initializes a new trading day"""
        self.trading_day = date
        
        # Create new daily PnL record
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO daily_pnl (date, realized_pnl, unrealized_pnl, total_pnl)
            VALUES (?, 0, 0, 0)
        ''', (date.strftime('%Y-%m-%d'),))
        self.db_conn.commit()
        
        # Load previous positions from database
        self.load_positions()

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
    
    # Create a simple wrapper to get the price
    class PriceWrapper(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.bid = None
            self.ask = None
            self.price_ready = False
            
        def tickPrice(self, reqId, tickType, price, attrib):
            if tickType == 1:  # Bid
                self.bid = price
            elif tickType == 2:  # Ask
                self.ask = price
                
            if self.bid and self.ask:
                self.price_ready = True
    
    # Get the price
    wrapper = PriceWrapper()
    wrapper.connect('127.0.0.1', TRADING_PORT, 123)
    wrapper.reqMktData(1, contract, '', False, False, [])
    
    # Wait for price (with timeout)
    timeout = time.time() + 5  # 5 second timeout
    while not wrapper.price_ready and time.time() < timeout:
        time.sleep(0.1)
    
    wrapper.disconnect()
    
    if wrapper.bid and wrapper.ask:
        return (wrapper.bid + wrapper.ask) / 2
    else:
        # Fallback to last close price from our historical data
        df = get_recent_bars(symbol, '1 min', '1 D')
        if not df.empty:
            return df['close'].iloc[-1]
        return None
    
def calculate_volatility(symbol, window='20 D', use_intraday=True):
    try:
        if use_intraday:
            # Use 5-minute bars if we want intraday volatility
            df = get_recent_bars(symbol, '5 mins', window, for_calculation=True)
        else:
            # Use daily bars for standard volatility
            df = get_recent_bars(symbol, '1 day', window, for_calculation=True)
            
        if df.empty:
            print(f"No data available for {symbol} volatility calculation")
            return 0.02  # Return a default volatility of 2%
            
        # Calculate log returns
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Remove any infinite or NaN values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < 2:
            print(f"Insufficient data for {symbol} volatility calculation")
            return 0.02  # Return a default volatility
            
        # Calculate annualized volatility
        scaling_factor = np.sqrt(252 * 78) if use_intraday else np.sqrt(252)
        volatility = df['returns'].std() * scaling_factor
        
        # Add bounds to prevent extreme values
        volatility = max(min(volatility, 1.0), 0.01)  # Cap between 1% and 100%
        
        return volatility
        
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return 0.02  # Return a default volatility as fallback

def get_recent_bars(symbol, timeframe, duration, for_calculation=False):
    print(f"\nDEBUG: Data request details:")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Duration: {duration}")
    print(f"Called by: {inspect.stack()[1].function}")  # This shows which function called us
    target_queue = calc_queue if for_calculation else data_queue
    print(f"\nRequesting data for {symbol}: {timeframe} bars over {duration}")
    
    # Clear the appropriate queue
    target_queue.queue.clear()
    
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    contract.primaryExchange='NYSE'
    

    req_id = int(time.time()) % 10000
    
    # Store original callback
    original_callback = client.historicalData
    
    # Create a temporary callback that puts data in our target queue
    def temp_callback(reqId, bar):
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
            target_queue.put(data)
        except Exception as e:
            print(f"Error in historical data callback: {e}")
    
    try:
        # Use our temporary callback
        client.historicalData = temp_callback
        
        client.reqHistoricalData(
            req_id, contract, '', duration, timeframe, 'TRADES',
            1, 1, False, []
        )
        
        # Collect the data
        bars = []
        first_timeout = time.time() + 5  # Initial waiting period
        last_data_time = time.time()
        while time.time() < first_timeout or time.time() - last_data_time < 2:
            try:
                bar = target_queue.get(timeout=0.5)
                bars.append(bar)
                last_data_time = time.time()  # Update the last time we got data
            except queue.Empty:
                time.sleep(0.1)
                if len(bars) > 0:  # If we have some data, do one final check
                    try:
                        while True:
                            bar = target_queue.get_nowait()
                            bars.append(bar)
                    except queue.Empty:
                        break
                break
        
        if not bars:
            print("No data was collected!")
            return pd.DataFrame()
            
        df = pd.DataFrame(bars)
        print(f"Collected {len(df)} bars of data")
        return df
        
    finally:
        # Always restore the original callback
        client.historicalData = original_callback
        client.cancelHistoricalData(req_id)

class RiskManager:
    def __init__(self):
        self.max_position_size = 3000000  # $1M max position
        self.max_daily_loss = 50000       # $50K max daily loss
        self.position_sizing_pct = 0.02   # 2% risk per trade
        self.current_positions = {}
        self.daily_pnl = 0
        self.account_update_event = threading.Event()
        
    def get_account_value(self):

        # Request account updates
        client.reqAccountUpdates(True, "")
        
        # Wait for the account update callback (maximum 5 seconds)
        if self.account_update_event.wait(timeout=5):
            return self.account_value
        else:
            print("Warning: Timeout waiting for account value")
            return 200000000  # Fallback default value    

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

    def calculate_position_size(self, symbol, price, volatility):
        """
    Calculates appropriate position size based on volatility and risk parameters
    """
        try:
            account_value = self.get_account_value()
            risk_amount = account_value * self.position_sizing_pct
            print(risk_amount)
        
        # Add safety checks for price and volatility
            if price is None or volatility is None or price <= 0 or volatility <= 0:
                print(f"Warning: Invalid price ({price}) or volatility ({volatility})")
                return 0
            
            position_size = int(risk_amount / (price * volatility))
            return max(0, position_size)  # Ensure we don't return negative positions
        
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0  # Return 0 as a safe fallback

# Client for connecting to Interactive Brokers
class PTLClient(EWrapper, EClient):
     
    def __init__(self, host, port, client_id):
        EClient.__init__(self, self) 
        
        self.connect(host, port, client_id)

        # create a new Thread
        thread = Thread(target=self.run)
        thread.start()


    def error(self, req_id, code, msg, misc=None):
        if code in [2104, 2106, 2158]:
            print(msg)
        else:
            print('Error {}: {}'.format(code, msg))


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.order_id = orderId
        print(f"next valid id is {self.order_id}")

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
            keepUpToDate=False,
            chartOptions=[]
        )
        print("Historical data request sent successfully")
        
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
    volatility = calculate_volatility(symbol)   # You'll need to implement this
    print(f"Current price: {current_price}, Volatility: {volatility}")
    
    # Calculate position size based on risk
    risk_manager = RiskManager()  # Create an instance
    quantity = risk_manager.calculate_position_size(symbol, current_price, volatility)
    
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
    is_valid, reason = risk_manager.validate_order(
        symbol, order.action, quantity, current_price
    )
    
    if not is_valid:
        print(f"Order rejected: {reason}")
        return
        
    # Place the order if valid
    if client.order_id:
        #client.order_id=client.nextValidId(client.order_id)
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


# called when we want to update what is rendered on the chart 
def update_chart():
    """
    Updates the chart with new data while properly handling element removal and updates.
    Uses the correct methods for removing and updating chart elements.
    """
    global current_lines, chart

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
                    # The proper way to remove a line is to set it to empty data
                    line.set(pd.DataFrame())
                    time.sleep(0.1)  # Small delay between operations
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

    except Exception as e:
        print(f"Critical error in chart update: {e}")
        
    finally:
        # Always ensure the spinner is removed
        try:
            chart.spinner(False)
        except Exception as e:
            print(f"Error removing spinner: {e}")

if __name__ == '__main__':
    # create a client object
    client = PTLClient(DEFAULT_HOST, TRADING_PORT, DEFAULT_CLIENT_ID)

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
    chart.topbar.switcher('timeframe', ('5 mins', '15 mins', '1 hour'), default='5 mins', func=on_timeframe_selection)

    # populate initial chart
    get_bar_data(INITIAL_SYMBOL, '5 mins')

    # run a market scanner
    do_scan("HOT_BY_VOLUME")

    # create a button for taking a screenshot of the chart
    chart.topbar.button('screenshot', 'Screenshot', func=take_screenshot)

    # show the chart
    chart.show(block=True)