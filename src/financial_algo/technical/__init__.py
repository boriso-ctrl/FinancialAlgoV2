"""Technical analysis package — price/volume-based indicators, signals, and strategies.

Re-exports from the existing modules so both import paths work:
    from financial_algo.technical import indicators, signals, regimes
    from financial_algo.indicators import rsi  # still works
"""

from financial_algo import indicators, signals, regimes, backtest, portfolio
from financial_algo.strategies.base import Strategy

__all__ = [
    "indicators",
    "signals",
    "regimes",
    "backtest",
    "portfolio",
    "Strategy",
]
