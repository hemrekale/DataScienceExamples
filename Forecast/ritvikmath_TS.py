import yfinance as yf
import pandas as pd
import numpy as np

tickerSymbol = 'MSFT'
tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period = '1d', start = '2010-01-01',end = '2021-02-23')

tickerDf.head()


# source : https://www.youtube.com/watch?v=y8opUEd05Dg&list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3&index=21
