"""Intraday strategy base class and NaN-safe vectorized feature pack.

All features are computed as pandas Series operations — no Python loops.
Input: OHLCV DataFrame with columns [open, high, low, close, volume].
Output: pd.Series of float signals in [-1.0, 1.0].
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class IntradayStrategy(ABC):
    """Base class for all intraday/HFT strategies.

    Subclasses implement :meth:`generate_signal`, which takes a single-ticker
    OHLCV DataFrame (indexed by UTC timestamp) and returns a float Series of
    target positions in [-1.0, 1.0].

    Execution controls (applied by the evaluator harness):
        rebalance_bars   -- minimum bars before position can change (default 5)
        change_threshold -- ignore weight changes below this (default 0.25)
        quant_step       -- quantize positions to multiples of this (default 0.5)
    """

    name: str = "IntradayBase"
    timeframe: str = "1min"

    @abstractmethod
    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Generate intraday position signal.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame with columns [open, high, low, close, volume],
            indexed by UTC timestamps at the resolution of ``self.timeframe``.

        Returns
        -------
        pd.Series
            Float series in [-1.0, 1.0], same index as ``ohlcv``.
            +1 = max long, 0 = flat, -1 = max short.
            Must contain no NaN or inf values.
        """


# ---------------------------------------------------------------------------
# NaN-safe vectorized feature library
# ---------------------------------------------------------------------------


def feat_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, NaN-safe."""
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def feat_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram.

    Returns
    -------
    macd_line, signal_line, histogram
    """
    ema_fast = feat_ema(close, fast)
    ema_slow = feat_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = feat_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def feat_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder smoothing (EMA with alpha=1/period)."""
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    rs = (avg_gain / avg_loss.replace(0, np.nan)).fillna(0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def feat_bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, mid, lower, bandwidth, %B.

    Returns
    -------
    upper, mid, lower, bandwidth, pct_b
    """
    mid = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std().fillna(0.0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = (upper - lower) / mid.replace(0, np.nan)
    band_width = band_width.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = (upper - lower).replace(0, np.nan)
    pct_b = ((close - lower) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    return upper, mid, lower, band_width, pct_b


def feat_vwap(
    close: pd.Series,
    volume: pd.Series,
    session_reset: bool = True,
) -> pd.Series:
    """Cumulative session VWAP.

    If ``session_reset`` is True (default), resets at each calendar day
    boundary so the VWAP reflects the intraday session, not all-time.
    """
    if session_reset:
        date_key = close.index.normalize() if hasattr(close.index, "normalize") else close.index.date
        pv = close * volume
        cum_pv = pv.groupby(date_key).cumsum()
        cum_vol = volume.groupby(date_key).cumsum()
    else:
        cum_pv = (close * volume).cumsum()
        cum_vol = volume.cumsum()

    vwap = (cum_pv / cum_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return vwap.ffill().fillna(close)


def feat_rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    """Relative volume: current bar / trailing average volume. NaN-safe."""
    avg_vol = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
    rvol = (volume / avg_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return rvol


def feat_realized_vol(
    close: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Rolling realized volatility (std of log returns). NaN-safe."""
    price_ratio = (close / close.shift(1).replace(0, np.nan)).clip(lower=1e-10)
    log_ret = np.log(price_ratio).fillna(0.0)
    rvol = log_ret.rolling(window, min_periods=2).std().fillna(0.0)
    return rvol


def feat_vol_ratio(
    close: pd.Series,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.Series:
    """Short-term / long-term realized vol ratio (impulse detection). NaN-safe."""
    short_vol = feat_realized_vol(close, short_window)
    long_vol = feat_realized_vol(close, long_window).replace(0, np.nan)
    ratio = (short_vol / long_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return ratio


def feat_return_vol_corr(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling correlation between returns and volume (order flow proxy)."""
    ret = close.pct_change().fillna(0.0)
    corr = ret.rolling(window, min_periods=5).corr(volume).fillna(0.0)
    return corr.clip(-1.0, 1.0)


def feat_kst(close: pd.Series) -> pd.Series:
    """Know Sure Thing (KST) momentum oscillator, NaN-safe."""
    def rcma(s: pd.Series, r: int, m: int) -> pd.Series:
        roc = (s / s.shift(r).replace(0, np.nan) - 1.0).fillna(0.0)
        return roc.rolling(m, min_periods=1).mean()

    kst = (
        rcma(close, 10, 10) * 1
        + rcma(close, 15, 10) * 2
        + rcma(close, 20, 10) * 3
        + rcma(close, 30, 15) * 4
    )
    return kst.fillna(0.0)


def feat_tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True Strength Index, NaN-safe."""
    delta = close.diff().fillna(0.0)
    smooth1 = feat_ema(feat_ema(delta, long), short)
    smooth2 = feat_ema(feat_ema(delta.abs(), long), short).replace(0, np.nan)
    tsi = (100.0 * smooth1 / smooth2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return tsi


def feat_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range, NaN-safe."""
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()
    return atr.fillna(0.0)


def _sanitize(signal: pd.Series) -> pd.Series:
    """Final NaN/inf guard — every generate_signal should call this."""
    return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
