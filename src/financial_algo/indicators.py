"""Technical indicators for financial time-series data."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def moving_average(prices: Sequence[float], window: int) -> list[float]:
    """Return the simple moving average of *prices* over *window* periods.

    Parameters
    ----------
    prices:
        Sequence of closing prices (oldest first).
    window:
        Look-back period (number of bars).

    Returns
    -------
    list[float]
        SMA values.  The first ``window - 1`` elements are ``float('nan')``.

    Raises
    ------
    ValueError
        If *window* is less than 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    prices = list(prices)
    result: list[float] = []
    for i, _ in enumerate(prices):
        if i < window - 1:
            result.append(float("nan"))
        else:
            result.append(sum(prices[i - window + 1 : i + 1]) / window)
    return result


def rsi(prices: Sequence[float], period: int = 14) -> list[float]:
    """Return the Relative Strength Index (RSI) for *prices*.

    Parameters
    ----------
    prices:
        Sequence of closing prices (oldest first).
    period:
        Look-back period. Defaults to 14.

    Returns
    -------
    list[float]
        RSI values in the range [0, 100].  The first *period* elements are
        ``float('nan')``.

    Raises
    ------
    ValueError
        If *period* is less than 1 or *prices* has fewer than 2 elements.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    prices = list(prices)
    if len(prices) < 2:
        raise ValueError("prices must contain at least 2 elements")

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    result: list[float] = [float("nan")] * len(prices)

    for i in range(period, len(prices)):
        window = changes[i - period : i]
        gains = [c for c in window if c > 0]
        losses = [-c for c in window if c < 0]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def bollinger_bands(
    prices: Sequence[float],
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[list[float], list[float], list[float]]:
    """Return the Bollinger Bands for *prices*.

    Parameters
    ----------
    prices:
        Sequence of closing prices (oldest first).
    window:
        Look-back period. Defaults to 20.
    num_std:
        Number of standard deviations for the bands. Defaults to 2.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        A three-element tuple ``(upper_band, middle_band, lower_band)``.
        The first ``window - 1`` elements in each band are ``float('nan')``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    prices = list(prices)
    middle = moving_average(prices, window)
    upper: list[float] = []
    lower: list[float] = []

    for i in range(len(prices)):
        if i < window - 1:
            upper.append(float("nan"))
            lower.append(float("nan"))
        else:
            subset = prices[i - window + 1 : i + 1]
            mean = middle[i]
            variance = sum((x - mean) ** 2 for x in subset) / window
            std = math.sqrt(variance)
            upper.append(mean + num_std * std)
            lower.append(mean - num_std * std)

    return upper, middle, lower


# ---------------------------------------------------------------------------
# Pandas-based indicators (used by strategy & regime layers)
# ---------------------------------------------------------------------------


def realized_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualised realised volatility from log returns.

    Parameters
    ----------
    prices:
        Price series (DatetimeIndex).
    window:
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Annualised volatility (assumes 252 trading days/year).
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential moving average.

    Parameters
    ----------
    prices:
        Price series.
    span:
        EMA span (number of periods).
    """
    return prices.ewm(span=span, adjust=False).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: ``(x - mean) / std``.

    Parameters
    ----------
    series:
        Any numeric Series.
    window:
        Rolling look-back window.
    """
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range.

    Parameters
    ----------
    high, low, close:
        OHLC price series (aligned index).
    period:
        Smoothing period.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def breadth_count(
    prices: pd.DataFrame,
    lookback: int = 50,
) -> pd.Series:
    """Fraction of columns whose price is above its own SMA.

    Parameters
    ----------
    prices:
        DataFrame of prices, one column per ticker.
    lookback:
        SMA window.

    Returns
    -------
    pd.Series
        Value in ``[0, 1]`` — fraction of tickers above their SMA.
    """
    sma = prices.rolling(lookback).mean()
    above = (prices > sma).astype(float)
    return above.mean(axis=1)


def drawdown(equity: pd.Series) -> pd.Series:
    """Running drawdown from peak.

    Parameters
    ----------
    equity:
        Equity / NAV series.

    Returns
    -------
    pd.Series
        Drawdown as a negative fraction (e.g. -0.10 = −10 %).
    """
    peak = equity.cummax()
    return (equity - peak) / peak
