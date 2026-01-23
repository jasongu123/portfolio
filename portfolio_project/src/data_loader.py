"""
Data Loader Module for Portfolio Optimization Project
======================================================
Downloads NASDAQ-100 stock data from Yahoo Finance and prepares it for analysis.

Author: Jason
Project: Portfolio Optimization with EVT Methods
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')
