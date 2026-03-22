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
    std = roll.std()
    std = std.replace(0, np.nan)  # avoid division by zero -> inf
    return (series - roll.mean()) / std


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
        Drawdown as a negative fraction (e.g. -0.10 = -10 %).
    """
    peak = equity.cummax()
    return (equity - peak) / peak


# ---------------------------------------------------------------------------
# New indicators — Sprint 1 (open-source research integration)
# ---------------------------------------------------------------------------


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R.

    Measures the close relative to the highest-high/lowest-low range.

    Parameters
    ----------
    high, low, close:
        OHLC price series (aligned index).
    period:
        Lookback period. Default 14.

    Returns
    -------
    pd.Series
        Values in [-100, 0]. Near 0 = overbought; near -100 = oversold.
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return (highest_high - close) / denom * -100.0


def dv2(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ma_period: int = 2,
    rank_period: int = 252,
) -> pd.Series:
    """DV2 short-term mean-reversion oscillator (Connors/Alvarez).

    Computes the percentile rank of the 2-day smoothed close/midprice
    ratio over the past year.

    Parameters
    ----------
    high, low, close:
        OHLC price series (aligned index).
    ma_period:
        Short smoothing window. Default 2.
    rank_period:
        Rolling window for percentile rank. Default 252 (1 trading year).

    Returns
    -------
    pd.Series
        DV2 in [0, 100]. Below 25 = oversold; above 75 = overbought.
    """
    mid = (high + low) / 2.0
    denom = mid.replace(0, np.nan)
    ratio = (close / denom).fillna(1.0)
    smoothed = ratio.rolling(ma_period).mean()
    ranked = smoothed.rolling(rank_period).rank(pct=True)
    return (ranked * 100.0).fillna(np.nan)


def kst(
    close: pd.Series,
    r1: int = 10,
    r2: int = 15,
    r3: int = 20,
    r4: int = 30,
    s1: int = 10,
    s2: int = 10,
    s3: int = 10,
    s4: int = 10,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Know Sure Thing (KST) multi-horizon momentum composite.

    Weights four ROC calculations at increasing lookbacks.

    Parameters
    ----------
    close:
        Close price series.
    r1..r4:
        ROC lookback periods (bars). Defaults: 10, 15, 20, 30.
    s1..s4:
        SMA smoothing periods for each ROC. Default 10 each.
    signal_period:
        SMA period for the signal line. Default 9.

    Returns
    -------
    pd.DataFrame
        Columns ``'kst'`` and ``'kst_signal'``.
    """
    def _roc(c: pd.Series, n: int) -> pd.Series:
        return c.pct_change(n).fillna(0.0) * 100.0

    k = (
        _roc(close, r1).rolling(s1).mean() * 1
        + _roc(close, r2).rolling(s2).mean() * 2
        + _roc(close, r3).rolling(s3).mean() * 3
        + _roc(close, r4).rolling(s4).mean() * 4
    )
    return pd.DataFrame({"kst": k, "kst_signal": k.rolling(signal_period).mean()})


def tsi(
    close: pd.Series,
    long_period: int = 25,
    short_period: int = 13,
) -> pd.Series:
    """True Strength Index (TSI).

    Double-smooths price change and its absolute value via EMA, then
    takes their ratio scaled to [-100, 100].

    Parameters
    ----------
    close:
        Close price series.
    long_period:
        Outer (longer) EMA span. Default 25.
    short_period:
        Inner (shorter) EMA span. Default 13.

    Returns
    -------
    pd.Series
        TSI values. Positive = bullish trend; negative = bearish.
        Zero-crossings signal trend changes (less whipsaw than SMA-200).
    """
    pc = close.diff(1)
    smooth_pc = (
        pc.ewm(span=long_period, adjust=False)
        .mean()
        .ewm(span=short_period, adjust=False)
        .mean()
    )
    smooth_abs = (
        pc.abs()
        .ewm(span=long_period, adjust=False)
        .mean()
        .ewm(span=short_period, adjust=False)
        .mean()
    )
    denom = smooth_abs.replace(0, np.nan)
    return (smooth_pc / denom * 100.0).fillna(0.0)


def pgo(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Pretty Good Oscillator (PGO).

    Normalises the close-to-SMA distance by ATR, producing a
    dimensionless breakout strength score.

    Parameters
    ----------
    high, low, close:
        OHLC price series (aligned index).
    period:
        Lookback for SMA and ATR. Default 14.

    Returns
    -------
    pd.Series
        PGO values. Above +3 = strong breakout; below -3 = breakdown.
    """
    sma = close.rolling(period).mean()
    atr_vals = atr(high, low, close, period)
    denom = atr_vals.replace(0, np.nan)
    return (close - sma) / denom


def rmi(
    close: pd.Series,
    period: int = 20,
    lookback: int = 5,
) -> pd.Series:
    """Relative Momentum Index (RMI).

    Like RSI but momentum comparison uses *lookback* bars instead of 1,
    producing a smoother oscillator less prone to single-bar noise.

    Parameters
    ----------
    close:
        Close price series.
    period:
        EMA smoothing period. Default 20.
    lookback:
        Price-change lookback (replaces RSI's fixed 1-bar). Default 5.

    Returns
    -------
    pd.Series
        RMI values in [0, 100]. 50 = neutral.
    """
    delta = close.diff(lookback)
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    ema_up = up.ewm(span=period, adjust=False).mean()
    ema_dn = dn.ewm(span=period, adjust=False).mean()
    total = (ema_up + ema_dn).replace(0, np.nan)
    return (ema_up / total * 100.0).fillna(50.0)


def rolling_skew(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness of returns.

    Parameters
    ----------
    returns:
        Return series (e.g. from ``prices.pct_change().fillna(0)``).
    window:
        Rolling window in bars. Default 20.

    Returns
    -------
    pd.Series
        Rolling skewness. Values < -0.5 indicate left-tail / crash risk.
    """
    return returns.rolling(window).skew()


def rolling_kurt(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling excess kurtosis of returns.

    Parameters
    ----------
    returns:
        Return series (e.g. from ``prices.pct_change().fillna(0)``).
    window:
        Rolling window in bars. Default 20.

    Returns
    -------
    pd.Series
        Rolling excess kurtosis (Fisher definition, normal = 0).
        Values > 1 indicate fat tails on both sides.
    """
    return returns.rolling(window).kurt()


def hurst_exponent(
    close: pd.Series,
    window: int = 252,
    min_lag: int = 2,
    max_lag: int = 20,
) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range analysis.

    Estimates whether the series is trending (H > 0.5), mean-reverting
    (H < 0.5), or random (H ~= 0.5).

    Parameters
    ----------
    close:
        Close price series.
    window:
        Rolling estimation window. Default 252 (1 year).
    min_lag, max_lag:
        Lag range for the log-log regression. Fewer lags = faster.

    Returns
    -------
    pd.Series
        Hurst exponent clipped to [0.05, 0.95].

    Notes
    -----
    Uses ``rolling().apply()`` internally; compute on one reference series
    (e.g. SPY) rather than all assets to avoid excessive runtime.
    """
    lags = np.arange(min_lag, max_lag + 1, dtype=np.float64)
    log_lags = np.log(lags)

    def _hurst(arr: np.ndarray) -> float:
        tau = np.array(
            [np.std(arr[int(lag):] - arr[: len(arr) - int(lag)]) for lag in lags]
        )
        valid = tau > 0
        if valid.sum() < 3:
            return 0.5
        poly = np.polyfit(log_lags[valid], np.log(tau[valid]), 1)
        return float(np.clip(poly[0], 0.05, 0.95))

    return close.rolling(window).apply(_hurst, raw=True)


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Relative volume: ratio of current volume to its rolling mean.

    Parameters
    ----------
    volume:
        Volume series.
    window:
        Rolling mean window. Default 20.

    Returns
    -------
    pd.Series
        Relative volume. 1.0 = average; > 1.5 = above-average confirmation.
    """
    mean_vol = volume.rolling(window).mean().replace(0, np.nan)
    return (volume / mean_vol).fillna(1.0)


def rsi_change_rate(close: pd.Series, period: int = 14) -> pd.Series:
    """Rate of change of RSI — momentum of momentum.

    Positive and growing = accelerating upside. Crosses zero signals
    an RSI acceleration reversal before the RSI itself turns.

    Parameters
    ----------
    close:
        Close price series.
    period:
        RSI period. Default 14.

    Returns
    -------
    pd.Series
        RSI change rate as a percentage of the prior RSI value.
    """
    delta = close.diff(1)
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_up = up.ewm(alpha=alpha, adjust=False).mean()
    avg_dn = dn.ewm(alpha=alpha, adjust=False).mean()
    denom = (avg_up + avg_dn).replace(0, np.nan)
    rsi_s = (avg_up / denom * 100.0).fillna(50.0)
    prev = rsi_s.shift(1).replace(0, np.nan)
    return ((rsi_s - rsi_s.shift(1)) / prev * 100.0).fillna(0.0)


# ---------------------------------------------------------------------------
# Alpha158 gap-fill indicators (Sprint 4)
# ---------------------------------------------------------------------------


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    close:
        Close price series.
    fast:
        Fast EMA span. Default 12.
    slow:
        Slow EMA span. Default 26.
    signal:
        Signal line EMA span. Default 9.

    Returns
    -------
    pd.DataFrame
        Columns ``'macd'``, ``'signal'``, ``'histogram'``.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def bollinger_pctb(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Bollinger %B -- position of close within the bands.

    Parameters
    ----------
    close:
        Close price series.
    window:
        Rolling window for SMA and std. Default 20.
    num_std:
        Number of standard deviations. Default 2.0.

    Returns
    -------
    pd.Series
        %B values. 0 = at lower band, 1 = at upper band.
    """
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    denom = (upper - lower).replace(0, np.nan)
    pctb = (close - lower) / denom
    return pctb.replace([np.inf, -np.inf], np.nan).fillna(0.5)


def bollinger_bandwidth(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Bollinger Bandwidth -- band width relative to middle band.

    Parameters
    ----------
    close:
        Close price series.
    window:
        Rolling window. Default 20.
    num_std:
        Number of standard deviations. Default 2.0.

    Returns
    -------
    pd.Series
        Bandwidth values (dimensionless).
    """
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    denom = middle.replace(0, np.nan)
    bw = (upper - lower) / denom
    return bw.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def daily_price_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Daily price range as a fraction of close.

    Parameters
    ----------
    high, low, close:
        OHLC price series (aligned index).

    Returns
    -------
    pd.Series
        ``(high - low) / close``, inf-safe.
    """
    denom = close.replace(0, np.nan)
    result = (high - low) / denom
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def volume_std(volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of volume.

    Parameters
    ----------
    volume:
        Volume series.
    window:
        Rolling window. Default 20.

    Returns
    -------
    pd.Series
        Rolling volume std, NaN-safe.
    """
    return volume.rolling(window).std().fillna(0.0)


def price_volume_corr(
    close: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Rolling correlation between close returns and volume changes.

    Parameters
    ----------
    close:
        Close price series.
    volume:
        Volume series.
    window:
        Rolling correlation window. Default 10.

    Returns
    -------
    pd.Series
        Correlation in [-1, 1], NaN-safe.
    """
    ret = close.pct_change().fillna(0.0)
    vol_chg = volume.pct_change().fillna(0.0)
    corr = ret.rolling(window).corr(vol_chg)
    return corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def close_open_ratio(close: pd.Series, open_prices: pd.Series) -> pd.Series:
    """Ratio of close to open price.

    Parameters
    ----------
    close:
        Close price series.
    open_prices:
        Open price series.

    Returns
    -------
    pd.Series
        ``close / open``, inf-safe, fillna(1.0).
    """
    denom = open_prices.replace(0, np.nan)
    ratio = close / denom
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def upper_shadow_pct(
    high: pd.Series,
    open_prices: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Upper shadow as a fraction of close price.

    Parameters
    ----------
    high:
        High price series.
    open_prices:
        Open price series.
    close:
        Close price series.

    Returns
    -------
    pd.Series
        ``(high - max(open, close)) / close``, inf-safe.
    """
    body_top = np.maximum(open_prices, close)
    denom = close.replace(0, np.nan)
    result = (high - body_top) / denom
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def lower_shadow_pct(
    low: pd.Series,
    open_prices: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Lower shadow as a fraction of close price.

    Parameters
    ----------
    low:
        Low price series.
    open_prices:
        Open price series.
    close:
        Close price series.

    Returns
    -------
    pd.Series
        ``(min(open, close) - low) / close``, inf-safe.
    """
    body_bottom = np.minimum(open_prices, close)
    denom = close.replace(0, np.nan)
    result = (body_bottom - low) / denom
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_quantile_ratio(
    close: pd.Series,
    window: int = 20,
    q_low: float = 0.2,
    q_high: float = 0.8,
) -> pd.Series:
    """Position of close within rolling quantile range.

    Parameters
    ----------
    close:
        Close price series.
    window:
        Rolling window. Default 20.
    q_low:
        Lower quantile. Default 0.2.
    q_high:
        Upper quantile. Default 0.8.

    Returns
    -------
    pd.Series
        ``(close - q_lo) / (q_hi - q_lo)``, inf-safe, fillna(0.5).
    """
    q_lo = close.rolling(window).quantile(q_low)
    q_hi = close.rolling(window).quantile(q_high)
    denom = (q_hi - q_lo).replace(0, np.nan)
    ratio = (close - q_lo) / denom
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
