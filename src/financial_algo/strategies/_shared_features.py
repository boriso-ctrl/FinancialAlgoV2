"""Shared vectorized feature helpers used by ML and DL strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.indicators import (
    bollinger_bandwidth as _bb_bw,
    bollinger_pctb as _bb_pctb,
    ema,
    price_volume_corr as _pv_corr,
)


def momentum_df(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """N-day return, NaN-filled to 0."""
    return prices.pct_change(lookback).fillna(0.0)


def realized_vol_df(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Annualised realized volatility for every column."""
    ret = prices.pct_change().fillna(0.0)
    return ret.rolling(window).std().fillna(0.0) * np.sqrt(252)


def rsi_df(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Vectorized RSI for all columns at once."""
    delta = prices.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    # Guard division by zero
    avg_loss_safe = avg_loss.replace(0.0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)  # neutral when undefined


def zscore_df(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling z-score for all columns."""
    roll_mean = prices.rolling(window).mean()
    roll_std = prices.rolling(window).std().replace(0.0, np.nan)
    z = (prices - roll_mean) / roll_std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def above_sma(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Binary: 1 if price >= SMA, else 0."""
    sma = prices.rolling(window, min_periods=window // 2).mean()
    return (prices >= sma).astype(float).fillna(0.0)


# ---------------------------------------------------------------------------
# DataFrame wrappers for Alpha158 indicators (Sprint 4)
# ---------------------------------------------------------------------------


def macd_df(
    prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.DataFrame]:
    """Column-wise MACD for all tickers.

    Returns dict with keys 'macd', 'signal', 'histogram', each a DataFrame.
    """
    macd_out: dict[str, pd.DataFrame] = {
        "macd": pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float),
        "signal": pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float),
        "histogram": pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float),
    }
    for col in prices.columns:
        macd_line = ema(prices[col], fast) - ema(prices[col], slow)
        signal_line = ema(macd_line, signal)
        macd_out["macd"][col] = macd_line
        macd_out["signal"][col] = signal_line
        macd_out["histogram"][col] = macd_line - signal_line
    return macd_out


def bollinger_pctb_df(
    prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Column-wise Bollinger %B for all tickers."""
    return prices.apply(lambda s: _bb_pctb(s, window, num_std))


def bollinger_bandwidth_df(
    prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Column-wise Bollinger Bandwidth for all tickers."""
    return prices.apply(lambda s: _bb_bw(s, window, num_std))


def price_volume_corr_df(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Column-wise price-volume correlation for all tickers."""
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        if col in volume.columns:
            result[col] = _pv_corr(close[col], volume[col], window)
        else:
            result[col] = 0.0
    return result
