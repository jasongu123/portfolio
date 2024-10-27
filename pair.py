# This is the backtest of a mean reversion pairs trading strategy that buys a stock whenever it grows too slowly compared to the rest of the 
# Magnificent 7 compared to Nvidia. And buys Nvidia whenever it grows too slowly relative to the rest of the Magnificent 7.

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

ticker_symbol = ['AAPL','MSFT','GOOG','AMZN','NVDA','META','AMD']

data = yf.download(ticker_symbol, start='2023-04-01', end='2024-10-24')
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
for index, row in data.iterrows():
    if row['AMD Spread'] < row['Lower Bound'] and position != 1:  # Buy stock signal
        position = 1
        entry_price = row['AMD']  # using AMD price
        trade_size = portfolio_size * 0.1  # Using 10% position size
        shares = trade_size / entry_price
        print(f"Buy AMD at {entry_price} on {index}")
        
    elif row['AMD Spread'] > row['Upper Bound'] and position != -1:  # Buy NVDA signal
        position = -1
        entry_price = row['NVDA']
        trade_size = portfolio_size * 0.1
        shares = trade_size / entry_price
        print(f"Buy NVDA at {entry_price} on {index}")
    
    # Exit when spread returns to mean
    elif row['AMD Spread'] < row['Total Average'] and position != 0:  # Exit signal
        exit_price = row['AMD'] if position == 1 else row['NVDA']
        position = 0
        pnl = (exit_price - entry_price) * shares
        portfolio_size += pnl
        print(f"Exit trade at {exit_price} on {index}, PnL: {pnl}")
    
    #returns.concat(portfolio_size)
for index, row in data.iterrows():
    if row['AAPL Spread'] < row['Lower Bound'] and position != 1:  # Buy stock signal
        position = 1
        entry_price = row['AAPL']  # using AMD price
        trade_size = portfolio_size * 0.1  # Using 10% position size
        shares = trade_size / entry_price
        print(f"Buy AAPL at {entry_price} on {index}")
        
    elif row['AAPL Spread'] > row['Upper Bound'] and position != -1:  # Buy NVDA signal
        position = -1
        entry_price = row['NVDA']
        trade_size = portfolio_size * 0.1
        shares = trade_size / entry_price
        print(f"Buy NVDA at {entry_price} on {index}")
    
    # Exit when spread returns to mean
    elif row['AAPL Spread'] < row['Total Average'] and position != 0:  # Exit signal
        exit_price = row['AAPL'] if position == 1 else row['NVDA']
        position = 0
        pnl = (exit_price - entry_price) * shares
        portfolio_size += pnl
        print(f"Exit trade at {exit_price} on {index}, PnL: {pnl}")
for index, row in data.iterrows():
    if row['MSFT Spread'] < row['Lower Bound'] and position != 1:  # Buy stock signal
        position = 1
        entry_price = row['MSFT']  # using AMD price
        trade_size = portfolio_size * 0.1  # Using 10% position size
        shares = trade_size / entry_price
        print(f"Buy MSFT at {entry_price} on {index}")
        
    elif row['MSFT Spread'] > row['Upper Bound'] and position != -1:  # Buy NVDA signal
        position = -1
        entry_price = row['NVDA']
        trade_size = portfolio_size * 0.1
        shares = trade_size / entry_price
        print(f"Buy NVDA at {entry_price} on {index}")
    
    # Exit when spread returns to mean
    elif row['MSFT Spread'] < row['Total Average'] and position != 0:  # Exit signal
        exit_price = row['MSFT'] if position == 1 else row['NVDA']
        position = 0
        pnl = (exit_price - entry_price) * shares
        portfolio_size += pnl
        print(f"Exit trade at {exit_price} on {index}, PnL: {pnl}")
for index, row in data.iterrows():
    if row['GOOG Spread'] < row['Lower Bound'] and position != 1:  # Buy stock signal
        position = 1
        entry_price = row['GOOG']  # using AMD price
        trade_size = portfolio_size * 0.1  # Using 10% position size
        shares = trade_size / entry_price
        print(f"Buy GOOG at {entry_price} on {index}")
        
    elif row['GOOG Spread'] > row['Upper Bound'] and position != -1:  # Buy NVDA signal
        position = -1
        entry_price = row['NVDA']
        trade_size = portfolio_size * 0.1
        shares = trade_size / entry_price
        print(f"Buy NVDA at {entry_price} on {index}")
    
    # Exit when spread returns to mean
    elif row['GOOG Spread'] < row['Total Average'] and position != 0:  # Exit signal
        exit_price = row['GOOG'] if position == 1 else row['NVDA']
        position = 0
        pnl = (exit_price - entry_price) * shares
        portfolio_size += pnl
        print(f"Exit trade at {exit_price} on {index}, PnL: {pnl}")
for index, row in data.iterrows():
    if row['AMZN Spread'] < row['Lower Bound'] and position != 1:  # Buy stock signal
        position = 1
        entry_price = row['AMZN']  # using AMD price
        trade_size = portfolio_size * 0.1  # Using 10% position size
        shares = trade_size / entry_price
        print(f"Buy AMZN at {entry_price} on {index}")
        
    elif row['AMZN Spread'] > row['Upper Bound'] and position != -1:  # Buy NVDA signal
        position = -1
        entry_price = row['NVDA']
        trade_size = portfolio_size * 0.1
        shares = trade_size / entry_price
        print(f"Buy NVDA at {entry_price} on {index}")
    
    # Exit when spread returns to mean
    elif row['AMZN Spread'] < row['Total Average'] and position != 0:  # Exit signal
        exit_price = row['AMZN'] if position == 1 else row['NVDA']
        position = 0
        pnl = (exit_price - entry_price) * shares
        portfolio_size += pnl
        print(f"Exit trade at {exit_price} on {index}, PnL: {pnl}")
for index, row in data.iterrows():
    if row['META Spread'] < row['Lower Bound'] and position != 1:  # Buy stock signal
        position = 1
        entry_price = row['META']  # using AMD price
        trade_size = portfolio_size * 0.1  # Using 10% position size
        shares = trade_size / entry_price
        print(f"Buy META at {entry_price} on {index}")
        
    elif row['META Spread'] > row['Upper Bound'] and position != -1:  # Buy NVDA signal
        position = -1
        entry_price = row['NVDA']
        trade_size = portfolio_size * 0.1
        shares = trade_size / entry_price
        print(f"Buy NVDA at {entry_price} on {index}")
    
    # Exit when spread returns to mean
    elif row['META Spread'] < row['Total Average'] and position != 0:  # Exit signal
        exit_price = row['META'] if position == 1 else row['NVDA']
        position = 0
        pnl = (exit_price - entry_price) * shares
        portfolio_size += pnl
        print(f"Exit trade at {exit_price} on {index}, PnL: {pnl}")
print(f"Final Portfolio Value: {portfolio_size}")
print(f"Return: {(portfolio_size/1000000 - 1)*100}%")
print(f"Sharpe Ratio: {(returns.pct_change().mean()/returns.pct_change().std())*np.sqrt(252)}")







