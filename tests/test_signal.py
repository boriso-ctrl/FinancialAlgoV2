from financial_algorithms.data import load_daily_prices
from financial_algorithms.signals import sma_signal

tickers = ["AAPL"]
prices = load_daily_prices(tickers)

signal = sma_signal(prices["AAPL"])
print(signal.tail())
