"""Compatibility shim: prefer ``financial_algorithms.data``."""

from financial_algorithms.data import load_daily_prices, load_daily_returns

__all__ = ["load_daily_prices", "load_daily_returns"]
