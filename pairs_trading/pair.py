from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

ticker_symbol = ['AAPL','MSFT','GOOG','AMZN','NVDA','META','AMD']

data = yf.download(ticker_symbol, start='2019-01-01', end='2025-01-01')
data = data['Close']
returns = data.pct_change()
returns=returns.dropna()
print(returns)
nvda=returns['NVDA']
comp=pd.DataFrame(index=returns.index)

comp['AAPL']=(returns['AAPL']-nvda)*100
comp['MSFT']=(returns['MSFT']-nvda)*100
comp['GOOG']=(returns['GOOG']-nvda)*100
comp['AMZN']=(returns['AMZN']-nvda)*100
comp['META']=(returns['META']-nvda)*100
comp['AMD']=(returns['AMD']-nvda)*100

print(comp)

avg=pd.DataFrame(index=returns.index)

avg['AAPL']=comp['AAPL'].rolling(20).mean()
avg['MSFT']=comp['AAPL'].rolling(20).mean()
avg['GOOG']=comp['GOOG'].rolling(20).mean()
avg['AMZN']=comp['AMZN'].rolling(20).mean()
avg['META']=comp['META'].rolling(20).mean()
avg['AMD']=comp['AMD'].rolling(20).mean()

sd=pd.DataFrame(index=returns.index)

sd['AAPL']=comp['AAPL'].rolling(20).std()
sd['MSFT']=comp['MSFT'].rolling(20).std()
sd['GOOG']=comp['GOOG'].rolling(20).std()
sd['AMZN']=comp['AMZN'].rolling(20).std()
sd['META']=comp['META'].rolling(20).std()
sd['AMD']=comp['AMD'].rolling(20).std()
avgaggr=pd.DataFrame()
avgaggr['Average']=avg.mean(axis=1).groupby(avg.index.date).mean()
avg=avg.dropna()
sd=sd.dropna()
avgaggr=avgaggr.dropna()
print(avg)
print(avgaggr)
b=pd.DataFrame()

b['Upper Bound']=avgaggr['Average']+2*avgaggr['Average'].rolling(20).std()
b['Lower Bound']=avgaggr['Average']-2*avgaggr['Average'].rolling(20).std()

figure, axis = plt.subplots(3,3)
plt.figure(1)
axis[0,0].plot(avg['AAPL'],label="AAPL-NVDA Avg Spread")
axis[0,0].plot(avg['AAPL']+2*sd['AAPL'],label="AAPL Upper Bound")
axis[0,0].plot(avg['AAPL']-2*sd['AAPL'],label="AAPL Lower Bound")
axis[0,1].plot(avg['MSFT'],label="MSFT-NVDA Avg Spread")
axis[0,1].plot(avg['MSFT']+2*sd['MSFT'],label="MSFT Upper Bound")
axis[0,1].plot(avg['MSFT']-2*sd['MSFT'],label="MSFT Lower Bound")
axis[0,2].plot(avg['GOOG'],label="GOOG-NVDA Avg Spread")
axis[0,2].plot(avg['GOOG']+2*sd['GOOG'],label="GOOG Upper Bound")
axis[0,2].plot(avg['GOOG']-2*sd['GOOG'],label="GOOG Lower Bound")
axis[1,0].plot(avg['AMZN'],label="AMZN-NVDA Avg Spread")
axis[1,0].plot(avg['AMZN']+2*sd['AMZN'],label="AMZN Upper Bound")
axis[1,0].plot(avg['AMZN']-2*sd['AMZN'],label="AMZN Lower Bound")
axis[1,1].plot(avg['META'],label="META-NVDA Avg Spread")
axis[1,1].plot(avg['META']+2*sd['META'],label="META Upper Bound")
axis[1,1].plot(avg['META']-2*sd['META'],label="META Lower Bound")
axis[1,2].plot(avg['AMD'],label="AMD-NVDA Avg Spread")
axis[1,2].plot(avg['AMD']+2*sd['AMD'],label="AMD Upper Bound")
axis[1,2].plot(avg['AMD']-2*sd['AMD'],label="AMD Lower Bound")

plt.figure(2)
plt.plot(avg['AAPL'],label="AAPL-NVDA Avg Spread")
plt.plot(avg['MSFT'],label="MSFT-NVDA Avg Spread")
plt.plot(avg['GOOG'],label="GOOG-NVDA Avg Spread")
plt.plot(avg['AMZN'],label="AMZN-NVDA Avg Spread")
plt.plot(avg['META'],label="META-NVDA Avg Spread")
plt.plot(avg['AMD'],label="AMD-NVDA Avg Spread")
plt.plot(avgaggr['Average'],label="Average of Aggregate Spread")
plt.plot(b['Upper Bound'],label="Upper Bound")
plt.plot(b['Lower Bound'],label="Lower Bound")
plt.legend()
plt.ylabel('Spread')
plt.xlabel('Day')
plt.show()

data['AAPL Spread']=avg['AAPL']
data['MSFT Spread']=avg['MSFT']
data['GOOG Spread']=avg['GOOG']
data['AMZN Spread']=avg['AMZN']
data['META Spread']=avg['META']
data['AMD Spread']=avg['AMD']
data['Total Average']=avgaggr['Average']
data['Upper Bound']=b['Upper Bound']
data['Lower Bound']=b['Lower Bound']
data=data.dropna()
print(data)

returns = pd.Series()
portfolio_size=1000000
position = 0
ticker_symbol.remove('NVDA')
positions = {ticker: {'position': 0, 'entry_price': 0, 'shares': 0} for ticker in ticker_symbol}
returns=pd.DataFrame(index=data.index)
returns['Current Value']=0

for index, row in data.iterrows():
    current_portfolio_value = portfolio_size
    
    # First, check if we need to exit any existing positions
    for ticker in ticker_symbol:
        spread = f'{ticker} Spread'
        if positions[ticker]['position'] != 0:  # If we have an active position
            # Check exit conditions
            if ((positions[ticker]['position'] == -1 and row[spread] < row['Total Average']) or 
                (positions[ticker]['position'] == 1 and row[spread] > row['Total Average'])):
                # Calculate exit price based on position type
                exit_price = row[ticker] if positions[ticker]['position'] == 1 else row['NVDA']
                
                # Calculate PnL
                pnl = (exit_price - positions[ticker]['entry_price']) * positions[ticker]['shares']
                portfolio_size += pnl
                print(f"Exit {ticker}/NVDA pair trade at {exit_price} on {index}, PnL: {pnl}")
                
                # Reset position
                positions[ticker]['position'] = 0
                positions[ticker]['entry_price'] = 0
                positions[ticker]['shares'] = 0
    
    # Then, look for new entry opportunities
    for ticker in ticker_symbol:
        spread = f'{ticker} Spread'
        # Only consider new positions if we don't have an existing position in this pair
        if positions[ticker]['position'] == 0:
            trade_size = portfolio_size * 0.1  # 10% position size
            
            # Check for long stock signal
            if row[spread] < row['Lower Bound']:
                positions[ticker]['position'] = 1
                positions[ticker]['entry_price'] = row[ticker]
                positions[ticker]['shares'] = trade_size / row[ticker]
                print(f"Buy {ticker} at {row[ticker]} on {index}")
                
            # Check for long NVDA signal
            elif row[spread] > row['Upper Bound']:
                positions[ticker]['position'] = -1
                positions[ticker]['entry_price'] = row['NVDA']
                positions[ticker]['shares'] = trade_size / row['NVDA']
                print(f"Buy NVDA against {ticker} at {row['NVDA']} on {index}")
    
    # Calculate current portfolio value including all open positions
    current_value = portfolio_size
    for ticker in ticker_symbol:
        if positions[ticker]['position'] == 1:
            # Long stock position
            current_value += (row[ticker] - positions[ticker]['entry_price']) * positions[ticker]['shares']
        elif positions[ticker]['position'] == -1:
            # Long NVDA position
            current_value += (row['NVDA'] - positions[ticker]['entry_price']) * positions[ticker]['shares']
    
    # Record the current portfolio value
    returns.loc[index, 'Current Value'] = current_value
                
    
print(f"Final Portfolio Value: {portfolio_size}")
print(f"Return: {(portfolio_size/1000000 - 1)*100}%")
annualreturn = ((portfolio_size/1000000) ** (252/len(returns)) - 1)
# Calculate annualized volatility
annualvol = returns['Current Value'].pct_change().std() * np.sqrt(252)

# Calculate Sharpe ratio
print(f"Annual Return: {annualreturn*100}%")

# Calculate Sharpe ratio
sharpe = (annualreturn - 0.04) / annualvol
# print(f"Sharpe Ratio: {(returns.pct_change().mean()/returns.pct_change().std())*np.sqrt(252)}")
print(f"Sharpe Ratio: {sharpe}")

print(((portfolio_size/1000000 - 1)-0.04)/(len(avg)/252))
print(returns.pct_change().std())

plt.figure(figsize=(12, 6))
plt.plot(returns['Current Value'], label='Portfolio Value', color='blue')
plt.axhline(y=1000000, color='r', linestyle='--', label='Initial Investment ($1M)')

# Enhance the visualization
plt.title('Pairs Trading Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Format y-axis to show values in millions
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])

# Rotate x-axis dates for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Calculate and display key statistics
total_return = ((returns['Current Value'].iloc[-1] / 1000000) - 1) * 100
max_value = returns['Current Value'].max()
max_drawdown = ((returns['Current Value'] - returns['Current Value'].cummax()) / returns['Current Value'].cummax()).min() * 100

# Add text box with performance metrics
plt.text(0.02, 0.98, f'Total Return: {total_return:.2f}%\nMax Portfolio Value: ${max_value:,.2f}\nMax Drawdown: {max_drawdown:.2f}%', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.show()

returns['Daily_Change'] = returns['Current Value'].pct_change()

# Function to analyze significant portfolio value changes
def analyze_portfolio_changes(returns_df, threshold=0.05):  # 5% threshold
    """
    Analyzes and prints significant changes in portfolio value.
    
    Args:
        returns_df: DataFrame containing portfolio values
        threshold: Minimum percentage change to report
    """
    significant_changes = returns_df[abs(returns_df['Daily_Change']) > threshold].copy()
    
    if len(significant_changes) > 0:
        print("\nSignificant Portfolio Value Changes:")
        print("=" * 80)
        print(f"{'Date':<12} {'Previous Value':>15} {'Current Value':>15} {'Change %':>10} {'Position':>10}")
        print("-" * 80)
        
        for idx in significant_changes.index:
            prev_value = returns_df['Current Value'][returns_df.index.get_loc(idx) - 1]
            curr_value = returns_df['Current Value'][idx]
            change_pct = significant_changes['Daily_Change'][idx] * 100
            
            print(f"{idx.strftime('%Y-%m-%d'):<12} "
                  f"${prev_value:>14,.2f} "
                  f"${curr_value:>14,.2f} "
                  f"{change_pct:>9.2f}% "
                  f"{returns_df['Position'][idx] if 'Position' in returns_df else 'N/A':>10}")
    else:
        print("\nNo significant portfolio value changes found.")

# Let's also calculate and print some summary statistics
def print_portfolio_statistics(returns_df):
    """
    Prints comprehensive portfolio performance statistics.
    """
    print("\nPortfolio Statistics:")
    print("=" * 50)
    
    # Calculate key metrics
    total_days = len(returns_df)
    starting_value = returns_df['Current Value'].iloc[0]
    ending_value = returns_df['Current Value'].iloc[-1]
    total_return = ((ending_value / starting_value) - 1) * 100
    
    # Calculate drawdowns
    rolling_max = returns_df['Current Value'].cummax()
    drawdowns = (returns_df['Current Value'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Print statistics
    print(f"Total Trading Days: {total_days}")
    print(f"Starting Portfolio Value: ${starting_value:,.2f}")
    print(f"Ending Portfolio Value: ${ending_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Print volatility statistics
    daily_returns = returns_df['Daily_Change'].dropna()
    print(f"Daily Return Volatility: {daily_returns.std() * 100:.2f}%")
    print(f"Number of Profitable Days: {len(daily_returns[daily_returns > 0])}")
    print(f"Number of Unprofitable Days: {len(daily_returns[daily_returns < 0])}")

# Run the analysis
analyze_portfolio_changes(returns)
print_portfolio_statistics(returns)

spy_data = yf.download('SPY', start='2019-01-01', end='2025-01-01')['Close']

# Calculate SPY returns and create a comparable series starting at our initial portfolio value
spy_returns = spy_data / spy_data.iloc[0]  # Normalize to starting point
spy_portfolio = spy_returns * 1000000  # Scale to our initial portfolio size

# Create a comparison visualization
plt.figure(figsize=(12, 6))

# Plot strategy performance
plt.plot(returns['Current Value'], 
         label='Pairs Trading Strategy', 
         color='blue', 
         linewidth=2)

# Plot SPY performance
plt.plot(spy_portfolio.reindex(returns.index), 
         label='S&P 500', 
         color='red', 
         linestyle='--', 
         linewidth=2)

# Add initial investment reference line
plt.axhline(y=1000000, 
            color='gray', 
            linestyle=':', 
            label='Initial Investment ($1M)')

# Enhance the visualization
plt.title('Strategy Performance vs S&P 500')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Format y-axis to show values in millions
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])

# Rotate x-axis dates for better readability
plt.xticks(rotation=45)

# Calculate and display performance metrics
strategy_return = ((returns['Current Value'].iloc[-1] / 1000000) - 1) * 100
spy_return = ((spy_portfolio.iloc[-1] / 1000000) - 1) * 100
excess_return = strategy_return - spy_return

# Calculate Sharpe Ratios (assuming risk-free rate of 4%)
rf_rate = 0.04
strategy_sharpe = (annualreturn - 0.04) / annualvol
spyreturn = ((spy_portfolio.iloc[-1]/1000000) ** (252/len(returns)) - 1)
# Calculate annualized volatility
spyvol = spy_portfolio.pct_change().std() * np.sqrt(252) 
spy_sharpe = (spyreturn - rf_rate) / spyvol

# Add performance statistics text box
stats_text = (
    f'Strategy Return: {strategy_return:.2f}%\n'
    f'S&P 500 Return: {spy_return:.2f}%\n'
    f'Excess Return: {excess_return:.2f}%\n'
    f'Strategy Sharpe: {strategy_sharpe:.2f}\n'
    f'S&P 500 Sharpe: {spy_sharpe:.2f}'
)

plt.text(0.02, 0.98, stats_text,
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()

# Calculate and print additional comparative statistics
print("\nDetailed Performance Comparison:")
print("=" * 50)

# Calculate maximum drawdowns
def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min() * 100

strategy_drawdown = calculate_max_drawdown(returns['Current Value'])
spy_drawdown = calculate_max_drawdown(spy_portfolio)

# Calculate rolling correlations
correlation = returns['Current Value'].pct_change().corr(spy_portfolio.pct_change())

print(f"Maximum Drawdown:")
print(f"  Strategy: {strategy_drawdown:.2f}%")
print(f"  S&P 500: {spy_drawdown:.2f}%")
print(f"\nCorrelation to S&P 500: {correlation:.2f}")
